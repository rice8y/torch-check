//! Secure, all-or-nothing acquisition and caching of official `PyTorch` wheel indexes.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use directories::BaseDirs;
use fs2::FileExt;
use futures_util::{StreamExt, stream};
use html5ever::tendril::StrTendril;
use html5ever::tokenizer::states::{Rawtext, Rcdata, ScriptData};
use html5ever::tokenizer::{
    BufferQueue, StartTag, TagToken, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
};
use pep440_rs::{Version, VersionSpecifiers};
use percent_encoding::percent_decode_str;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use url::Url;

use crate::core::{CudaVariant, IndexSnapshot, MetadataInfo, MetadataOrigin, TorchWheel};

const INDEX_SCHEMA_VERSION: u32 = 1;
const OFFICIAL_ROOT: &str = "https://download.pytorch.org/whl/";
const CACHE_FILE_PREFIX: &str = "index-v1";
const LOCK_FILE: &str = ".index-v1.lock";
const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_TOTAL_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_RESPONSE_LIMIT: usize = 16 * 1024 * 1024;
const MAX_CACHE_BYTES: u64 = 128 * 1024 * 1024;
const DEFAULT_CONCURRENCY: usize = 6;
const MAX_REDIRECTS: usize = 5;
const CLOCK_SKEW_SECONDS: u64 = 5 * 60;
const CACHE_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_LOCK_POLL: Duration = Duration::from_millis(25);
const MAX_VARIANTS: usize = 64;
const MAX_FETCH_PAGES: usize = MAX_VARIANTS * ALLOWED_PACKAGES.len();
const MAX_ANCHORS_PER_PAGE: usize = 100_000;
const MAX_WHEELS_PER_PAGE: usize = 50_000;
const MAX_TOTAL_WHEELS: usize = 250_000;
const MAX_COMPRESSED_TAGS: usize = 64;
const MAX_FILENAME_BYTES: usize = 1024;
const MAX_URL_BYTES: usize = 8192;
const MAX_REQUIRES_PYTHON_BYTES: usize = 1024;
const ALLOWED_PACKAGES: [&str; 3] = ["torch", "torchvision", "torchaudio"];
const ALLOWED_HOSTS: [&str; 2] = ["download.pytorch.org", "download-r2.pytorch.org"];

/// Configuration for official wheel-index acquisition.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Cache directory. The XDG cache directory is used when omitted.
    pub cache_dir: Option<PathBuf>,
    /// Forbid all network access and require a complete cache snapshot.
    pub offline: bool,
    /// Bypass a fresh cache and attempt a network refresh.
    pub refresh: bool,
    /// Cache freshness duration.
    pub ttl: Duration,
    /// Total timeout for each HTTP request.
    pub request_timeout: Duration,
    /// Deadline for the complete root and package-index refresh.
    pub total_timeout: Duration,
    /// Maximum decompressed bytes accepted for one index response.
    pub max_response_bytes: usize,
    /// Maximum number of package indexes fetched concurrently.
    pub max_concurrency: usize,
    /// Packages to fetch. Only torch, torchvision, and torchaudio are accepted.
    pub packages: Vec<String>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            offline: false,
            refresh: false,
            ttl: DEFAULT_TTL,
            request_timeout: DEFAULT_TIMEOUT,
            total_timeout: DEFAULT_TOTAL_TIMEOUT,
            max_response_bytes: DEFAULT_RESPONSE_LIMIT,
            max_concurrency: DEFAULT_CONCURRENCY,
            packages: vec!["torch".to_owned()],
        }
    }
}

/// A complete snapshot and its per-invocation provenance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LoadedIndex {
    /// Complete wheel metadata snapshot.
    pub snapshot: IndexSnapshot,
    /// Whether the snapshot came from network or cache.
    pub metadata: MetadataInfo,
}

/// Failure to obtain a complete, trustworthy wheel-index snapshot.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// An option combination or value was invalid.
    #[error("invalid index options: {0}")]
    InvalidOptions(String),
    /// No platform cache directory was available.
    #[error("could not determine an XDG cache directory")]
    CacheDirectoryUnavailable,
    /// A cache filesystem operation failed.
    #[error("cache operation failed at {path}: {source}")]
    CacheIo {
        /// Affected filesystem path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The cache JSON or its semantic schema was invalid.
    #[error("invalid cache snapshot at {path}: {reason}")]
    InvalidCache {
        /// Cache path.
        path: PathBuf,
        /// Validation failure.
        reason: String,
    },
    /// Strict offline mode had no complete cache for the request.
    #[error("offline mode requires a valid cache containing: {packages}")]
    OfflineCacheUnavailable {
        /// Comma-separated requested packages.
        packages: String,
    },
    /// The HTTP client could not be built.
    #[error("could not build HTTP client: {0}")]
    Client(#[source] reqwest::Error),
    /// An HTTP request failed.
    #[error("request failed for {url}: {source}")]
    Request {
        /// Requested URL.
        url: String,
        /// Underlying HTTP error.
        #[source]
        source: reqwest::Error,
    },
    /// An index returned a non-success HTTP response.
    #[error("index returned HTTP {status} for {url}")]
    HttpStatus {
        /// Final response URL.
        url: String,
        /// Numeric HTTP status.
        status: u16,
    },
    /// A URL violated the official-source policy.
    #[error("rejected index URL {url}: {reason}")]
    UrlPolicy {
        /// Rejected URL.
        url: String,
        /// Policy reason.
        reason: String,
    },
    /// A decompressed response exceeded the configured bound.
    #[error("decompressed response from {url} exceeded {limit} bytes")]
    ResponseTooLarge {
        /// Response URL.
        url: String,
        /// Configured byte limit.
        limit: usize,
    },
    /// The complete metadata refresh exceeded its deadline.
    #[error("complete index refresh exceeded {timeout_seconds} seconds")]
    FetchTimedOut {
        /// Configured whole-refresh timeout.
        timeout_seconds: u64,
    },
    /// An index response was not usable HTML.
    #[error("invalid index response from {url}: {reason}")]
    InvalidIndex {
        /// Index URL.
        url: String,
        /// Parse or validation error.
        reason: String,
    },
    /// At least one discovered package index failed; no partial snapshot exists.
    #[error("incomplete index fetch: {0}")]
    IncompleteFetch(String),
    /// A blocking cache task failed unexpectedly.
    #[error("cache task failed: {0}")]
    CacheTask(#[from] tokio::task::JoinError),
    /// Cache serialization failed.
    #[error("could not serialize cache snapshot: {0}")]
    CacheSerialization(#[from] serde_json::Error),
    /// Another process held the cache lock beyond the bounded wait.
    #[error("timed out waiting for cache lock at {path}")]
    CacheLockTimedOut {
        /// Lock file path.
        path: PathBuf,
    },
}

#[derive(Debug, Clone)]
struct FetchPolicy {
    root: Url,
    allowed_hosts: Arc<BTreeSet<String>>,
    require_https: bool,
}

impl FetchPolicy {
    fn official() -> Result<Self, IndexError> {
        let root = Url::parse(OFFICIAL_ROOT).map_err(|error| {
            IndexError::InvalidOptions(format!("invalid built-in index URL: {error}"))
        })?;
        let allowed_hosts = ALLOWED_HOSTS
            .iter()
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        let policy = Self {
            root,
            allowed_hosts: Arc::new(allowed_hosts),
            require_https: true,
        };
        policy.validate_url(&policy.root)?;
        Ok(policy)
    }

    fn validate_url(&self, url: &Url) -> Result<(), IndexError> {
        if self.require_https && url.scheme() != "https" {
            return Err(url_policy_error(url, "HTTPS is required"));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(url_policy_error(url, "credentials are forbidden"));
        }
        let host = url
            .host_str()
            .ok_or_else(|| url_policy_error(url, "host is missing"))?;
        if !self.allowed_hosts.contains(host) {
            return Err(url_policy_error(url, "host is not on the exact allowlist"));
        }
        if self.require_https && url.port().is_some_and(|port| port != 443) {
            return Err(url_policy_error(url, "non-default HTTPS port is forbidden"));
        }
        Ok(())
    }
}

fn url_policy_error(url: &Url, reason: &str) -> IndexError {
    IndexError::UrlPolicy {
        url: url.to_string(),
        reason: reason.to_owned(),
    }
}

/// Loads a complete official `PyTorch` wheel index according to `options`.
///
/// Offline mode never constructs an HTTP client. Online refreshes are
/// all-or-nothing: failure of any discovered variant/package page prevents a
/// new snapshot, after which a complete older cache may be used as
/// `stale-if-error`.
///
/// # Errors
///
/// Returns [`IndexError`] when options are invalid, a complete cache cannot be
/// obtained, an official index violates the URL or metadata policy, or secure
/// cache I/O fails.
pub async fn load_index(options: &IndexOptions) -> Result<LoadedIndex, IndexError> {
    let packages = validate_options(options)?;
    let policy = FetchPolicy::official()?;
    let cache_dir = cache_directory(options)?;
    let cache_path = cache_path_for(&cache_dir, &packages);
    let now = unix_now()?;

    let lock_dir = cache_dir.clone();
    let lock = tokio::task::spawn_blocking(move || open_cache_lock(&lock_dir)).await??;

    let read_dir = cache_dir.clone();
    let read_packages = packages.clone();
    let read_policy = policy.clone();
    let cached_result = tokio::task::spawn_blocking(move || {
        read_best_cache(&read_dir, &read_packages, &read_policy, now)
    })
    .await?;
    drop(lock);
    let (cached, cache_error) = match cached_result {
        Ok(snapshot) => (snapshot, None),
        Err(error) => (None, Some(error)),
    };
    let usable_cache = cached.filter(|snapshot| snapshot_contains(snapshot, &packages));

    if options.offline {
        if let Some(error) = cache_error {
            return Err(error);
        }
        let snapshot = usable_cache.ok_or_else(|| IndexError::OfflineCacheUnavailable {
            packages: packages.join(","),
        })?;
        return Ok(loaded(
            snapshot,
            MetadataOrigin::OfflineCache,
            now,
            options.ttl,
        ));
    }

    if !options.refresh {
        if let Some(snapshot) = usable_cache.as_ref() {
            if !is_stale(snapshot, now, options.ttl) {
                let loaded = loaded(
                    snapshot.clone(),
                    MetadataOrigin::FreshCache,
                    now,
                    options.ttl,
                );
                return Ok(loaded);
            }
        }
    }

    let fetch_result = match tokio::time::timeout(
        options.total_timeout,
        fetch_snapshot(&policy, options, packages),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(IndexError::FetchTimedOut {
            timeout_seconds: options.total_timeout.as_secs(),
        }),
    };
    match fetch_result {
        Ok(snapshot) => {
            let write_path = cache_path.clone();
            let write_dir = cache_dir.clone();
            let write_snapshot = snapshot.clone();
            tokio::task::spawn_blocking(move || {
                let _lock = open_cache_lock(&write_dir)?;
                write_cache(&write_dir, &write_path, &write_snapshot)
            })
            .await??;
            Ok(loaded(snapshot, MetadataOrigin::Network, now, options.ttl))
        }
        Err(network_error) => {
            if let Some(snapshot) = usable_cache {
                Ok(loaded(
                    snapshot,
                    MetadataOrigin::StaleIfError,
                    now,
                    options.ttl,
                ))
            } else if let Some(cache_error) = cache_error {
                Err(IndexError::IncompleteFetch(format!(
                    "{network_error}; cached snapshot was unusable: {cache_error}"
                )))
            } else {
                Err(network_error)
            }
        }
    }
}

fn validate_options(options: &IndexOptions) -> Result<Vec<String>, IndexError> {
    if options.offline && options.refresh {
        return Err(IndexError::InvalidOptions(
            "--offline and --refresh are mutually exclusive".to_owned(),
        ));
    }
    if options.request_timeout.is_zero() {
        return Err(IndexError::InvalidOptions(
            "request timeout must be non-zero".to_owned(),
        ));
    }
    if options.total_timeout.is_zero() {
        return Err(IndexError::InvalidOptions(
            "total refresh timeout must be non-zero".to_owned(),
        ));
    }
    if options.max_response_bytes == 0 {
        return Err(IndexError::InvalidOptions(
            "response-size limit must be non-zero".to_owned(),
        ));
    }
    if options.max_concurrency == 0 {
        return Err(IndexError::InvalidOptions(
            "concurrency must be non-zero".to_owned(),
        ));
    }
    let packages = options
        .packages
        .iter()
        .map(|package| package.trim().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    if packages.is_empty() || !packages.contains("torch") {
        return Err(IndexError::InvalidOptions(
            "packages must contain torch".to_owned(),
        ));
    }
    if let Some(package) = packages
        .iter()
        .find(|package| !ALLOWED_PACKAGES.contains(&package.as_str()))
    {
        return Err(IndexError::InvalidOptions(format!(
            "unsupported package index: {package}"
        )));
    }
    Ok(packages.into_iter().collect())
}

fn cache_directory(options: &IndexOptions) -> Result<PathBuf, IndexError> {
    if let Some(path) = &options.cache_dir {
        return Ok(path.clone());
    }
    BaseDirs::new()
        .map(|directories| directories.cache_dir().join("torch-check"))
        .ok_or(IndexError::CacheDirectoryUnavailable)
}

fn unix_now() -> Result<u64, IndexError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| {
            IndexError::InvalidOptions(format!("system clock predates UNIX epoch: {error}"))
        })
}

fn loaded(snapshot: IndexSnapshot, origin: MetadataOrigin, now: u64, ttl: Duration) -> LoadedIndex {
    let age_seconds = now.saturating_sub(snapshot.fetched_at);
    let stale = age_seconds > ttl.as_secs();
    let metadata = MetadataInfo {
        origin,
        fetched_at: snapshot.fetched_at,
        age_seconds,
        stale,
        source: snapshot.source.clone(),
    };
    LoadedIndex { snapshot, metadata }
}

fn is_stale(snapshot: &IndexSnapshot, now: u64, ttl: Duration) -> bool {
    now.saturating_sub(snapshot.fetched_at) > ttl.as_secs()
}

fn snapshot_contains(snapshot: &IndexSnapshot, packages: &[String]) -> bool {
    packages
        .iter()
        .all(|package| snapshot.packages.contains(package))
}

fn cache_path_for(cache_dir: &Path, packages: &[String]) -> PathBuf {
    cache_dir.join(format!("{CACHE_FILE_PREFIX}-{}.json", packages.join("-")))
}

fn cache_package_sets(requested: &[String]) -> Vec<Vec<String>> {
    let mut candidates = Vec::new();
    for include_audio in [false, true] {
        for include_vision in [false, true] {
            let mut packages = vec!["torch".to_owned()];
            if include_audio {
                packages.push("torchaudio".to_owned());
            }
            if include_vision {
                packages.push("torchvision".to_owned());
            }
            packages.sort();
            if requested.iter().all(|package| packages.contains(package)) {
                candidates.push(packages);
            }
        }
    }
    candidates.sort_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    candidates
}

fn read_best_cache(
    cache_dir: &Path,
    requested: &[String],
    policy: &FetchPolicy,
    now: u64,
) -> Result<Option<IndexSnapshot>, IndexError> {
    let mut best: Option<IndexSnapshot> = None;
    let mut first_error = None;
    for packages in cache_package_sets(requested) {
        let path = cache_path_for(cache_dir, &packages);
        match read_cache(&path, policy, now) {
            Ok(Some(snapshot)) if snapshot_contains(&snapshot, requested) => {
                if best
                    .as_ref()
                    .is_none_or(|current| snapshot.fetched_at > current.fetched_at)
                {
                    best = Some(snapshot);
                }
            }
            Ok(_) => {}
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    if best.is_some() {
        Ok(best)
    } else if let Some(error) = first_error {
        Err(error)
    } else {
        Ok(None)
    }
}

async fn fetch_snapshot(
    policy: &FetchPolicy,
    options: &IndexOptions,
    packages: Vec<String>,
) -> Result<IndexSnapshot, IndexError> {
    let fetched_at = unix_now()?;
    let client = build_client(policy, options.request_timeout)?;
    let root_html = fetch_html(
        &client,
        policy,
        policy.root.clone(),
        options.max_response_bytes,
    )
    .await?;
    let variants = discover_variants(&root_html, policy)?;
    if variants.is_empty() {
        return Err(IndexError::InvalidIndex {
            url: policy.root.to_string(),
            reason: "no cpu or CUDA variants were discovered".to_owned(),
        });
    }

    let jobs = variants
        .iter()
        .flat_map(|variant| {
            packages
                .iter()
                .map(move |package| (variant.clone(), package.clone()))
        })
        .collect::<Vec<_>>();
    if jobs.len() > MAX_FETCH_PAGES {
        return Err(IndexError::InvalidIndex {
            url: policy.root.to_string(),
            reason: format!("index expanded to more than {MAX_FETCH_PAGES} package pages"),
        });
    }
    let mut results = stream::iter(jobs.into_iter().map(|(variant, package)| {
        let client = client.clone();
        let policy = policy.clone();
        let limit = options.max_response_bytes;
        async move {
            let relative = format!("{variant}/{package}/");
            let page_url =
                policy
                    .root
                    .join(&relative)
                    .map_err(|error| IndexError::InvalidIndex {
                        url: policy.root.to_string(),
                        reason: format!("could not construct package URL: {error}"),
                    })?;
            let html = fetch_html(&client, &policy, page_url.clone(), limit).await?;
            parse_wheel_page(&html, &page_url, &policy, &variant, &package)
        }
    }))
    .buffer_unordered(options.max_concurrency);
    let mut wheels = Vec::new();
    let mut failures = Vec::new();
    while let Some(result) = results.next().await {
        match result {
            Ok(mut page_wheels) => {
                if wheels.len().saturating_add(page_wheels.len()) > MAX_TOTAL_WHEELS {
                    return Err(IndexError::InvalidIndex {
                        url: policy.root.to_string(),
                        reason: format!("snapshot exceeds the {MAX_TOTAL_WHEELS}-wheel limit"),
                    });
                }
                wheels.append(&mut page_wheels);
            }
            Err(error) => failures.push(error.to_string()),
        }
    }
    if !failures.is_empty() {
        failures.sort();
        return Err(IndexError::IncompleteFetch(failures.join(" | ")));
    }
    wheels.sort_by(|left, right| {
        (&left.package, &left.variant, &left.filename, &left.url).cmp(&(
            &right.package,
            &right.variant,
            &right.filename,
            &right.url,
        ))
    });
    wheels.dedup();
    let snapshot = IndexSnapshot {
        schema_version: INDEX_SCHEMA_VERSION,
        fetched_at,
        source: policy.root.to_string(),
        packages,
        variants,
        wheels,
    };
    validate_snapshot(&snapshot, policy, fetched_at)?;
    Ok(snapshot)
}

#[cfg(test)]
fn collect_complete_pages(
    results: Vec<Result<Vec<TorchWheel>, IndexError>>,
) -> Result<Vec<TorchWheel>, IndexError> {
    let mut failures = Vec::new();
    let mut wheels = Vec::new();
    for result in results {
        match result {
            Ok(mut page_wheels) => wheels.append(&mut page_wheels),
            Err(error) => failures.push(error.to_string()),
        }
    }
    if failures.is_empty() {
        Ok(wheels)
    } else {
        failures.sort();
        Err(IndexError::IncompleteFetch(failures.join(" | ")))
    }
}

fn build_client(policy: &FetchPolicy, timeout: Duration) -> Result<reqwest::Client, IndexError> {
    let redirect_policy = policy.clone();
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout.min(Duration::from_secs(10)))
        .https_only(policy.require_https)
        .redirect(reqwest::redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                return attempt.error("redirect limit exceeded");
            }
            match redirect_policy.validate_url(attempt.url()) {
                Ok(()) => attempt.follow(),
                Err(error) => attempt.error(error.to_string()),
            }
        }))
        .user_agent(concat!("torch-check/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(IndexError::Client)
}

async fn fetch_html(
    client: &reqwest::Client,
    policy: &FetchPolicy,
    url: Url,
    limit: usize,
) -> Result<String, IndexError> {
    policy.validate_url(&url)?;
    let requested = url.to_string();
    let response = client
        .get(url)
        .header(ACCEPT, "text/html, application/vnd.pypi.simple.v1+html")
        .send()
        .await
        .map_err(|source| IndexError::Request {
            url: requested,
            source,
        })?;
    policy.validate_url(response.url())?;
    let final_url = response.url().to_string();
    if !response.status().is_success() {
        return Err(IndexError::HttpStatus {
            url: final_url,
            status: response.status().as_u16(),
        });
    }
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(IndexError::ResponseTooLarge {
            url: final_url,
            limit,
        });
    }
    if let Some(content_type) = response.headers().get(CONTENT_TYPE) {
        let content_type = content_type
            .to_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !content_type.starts_with("text/html")
            && !content_type.starts_with("application/vnd.pypi.simple.v1+html")
        {
            return Err(IndexError::InvalidIndex {
                url: final_url,
                reason: format!("unexpected content type {content_type}"),
            });
        }
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| IndexError::Request {
            url: final_url.clone(),
            source,
        })?;
        append_bounded(&mut bytes, &chunk, limit, &final_url)?;
    }
    String::from_utf8(bytes).map_err(|error| IndexError::InvalidIndex {
        url: final_url,
        reason: format!("response is not UTF-8: {error}"),
    })
}

fn append_bounded(
    bytes: &mut Vec<u8>,
    chunk: &[u8],
    limit: usize,
    url: &str,
) -> Result<(), IndexError> {
    if bytes.len().saturating_add(chunk.len()) > limit {
        return Err(IndexError::ResponseTooLarge {
            url: url.to_owned(),
            limit,
        });
    }
    bytes.extend_from_slice(chunk);
    Ok(())
}

#[derive(Debug, Default)]
struct IndexAnchor {
    href: String,
    requires_python: Option<String>,
    yanked: bool,
}

#[derive(Debug, Default)]
struct AnchorSink {
    anchors: RefCell<Vec<IndexAnchor>>,
    overflowed: Cell<bool>,
}

impl TokenSink for AnchorSink {
    type Handle = ();

    fn process_token(&self, token: Token, _line_number: u64) -> TokenSinkResult<Self::Handle> {
        let TagToken(tag) = token else {
            return TokenSinkResult::Continue;
        };
        if tag.kind != StartTag {
            return TokenSinkResult::Continue;
        }

        let tag_name = tag.name.as_ref();
        let raw_state = match tag_name {
            "title" | "textarea" => Some(Rcdata),
            "style" | "xmp" | "iframe" | "noembed" | "noframes" | "noscript" => Some(Rawtext),
            "script" => Some(ScriptData),
            _ => None,
        };

        if tag_name == "a" {
            let mut href = None;
            let mut requires_python = None;
            let mut yanked = false;
            for attribute in tag.attrs {
                match attribute.name.local.as_ref() {
                    "href" => href = Some(attribute.value.to_string()),
                    "data-requires-python" => {
                        requires_python = Some(attribute.value.to_string());
                    }
                    "data-yanked" => yanked = true,
                    _ => {}
                }
            }
            if let Some(href) = href {
                let mut anchors = self.anchors.borrow_mut();
                if anchors.len() >= MAX_ANCHORS_PER_PAGE {
                    self.overflowed.set(true);
                } else {
                    anchors.push(IndexAnchor {
                        href,
                        requires_python,
                        yanked,
                    });
                }
            }
        }

        raw_state.map_or(TokenSinkResult::Continue, TokenSinkResult::RawData)
    }
}

fn parse_anchors(html: &str, url: &Url) -> Result<Vec<IndexAnchor>, IndexError> {
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(html));
    let tokenizer = Tokenizer::new(AnchorSink::default(), TokenizerOpts::default());
    let _ = tokenizer.feed(&input);
    tokenizer.end();
    if !input.is_empty() {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: "HTML tokenizer did not consume the complete document".to_owned(),
        });
    }
    if tokenizer.sink.overflowed.get() {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: format!("index exceeds the {MAX_ANCHORS_PER_PAGE}-anchor limit"),
        });
    }
    Ok(tokenizer.sink.anchors.into_inner())
}

fn discover_variants(html: &str, policy: &FetchPolicy) -> Result<Vec<CudaVariant>, IndexError> {
    let mut variants = BTreeSet::new();
    for anchor in parse_anchors(html, &policy.root)? {
        let href = anchor.href;
        if href.len() > MAX_URL_BYTES {
            return Err(IndexError::InvalidIndex {
                url: policy.root.to_string(),
                reason: format!("index link exceeds the {MAX_URL_BYTES}-byte limit"),
            });
        }
        let Ok(url) = policy.root.join(&href) else {
            continue;
        };
        let Some(name) = single_child_name(&policy.root, &url) else {
            continue;
        };
        let Ok(variant) = name.parse::<CudaVariant>() else {
            continue;
        };
        policy.validate_url(&url)?;
        variants.insert(variant);
        if variants.len() > MAX_VARIANTS {
            return Err(IndexError::InvalidIndex {
                url: policy.root.to_string(),
                reason: format!("root index exceeds the {MAX_VARIANTS}-variant limit"),
            });
        }
    }
    Ok(variants.into_iter().collect())
}

fn single_child_name(root: &Url, url: &Url) -> Option<String> {
    if url.query().is_some() || url.fragment().is_some() {
        return None;
    }
    let root_segments = root.path_segments()?.collect::<Vec<_>>();
    let url_segments = url
        .path_segments()?
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    let root_segments = root_segments
        .into_iter()
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if url_segments.len() != root_segments.len() + 1 || !url_segments.starts_with(&root_segments) {
        return None;
    }
    percent_decode_str(url_segments.last()?)
        .decode_utf8()
        .ok()
        .map(std::borrow::Cow::into_owned)
}

fn parse_wheel_page(
    html: &str,
    page_url: &Url,
    policy: &FetchPolicy,
    variant: &CudaVariant,
    package: &str,
) -> Result<Vec<TorchWheel>, IndexError> {
    let mut wheels = BTreeMap::<(String, String), TorchWheel>::new();
    for anchor in parse_anchors(html, page_url)? {
        let href = anchor.href;
        if href.len() > MAX_URL_BYTES {
            return Err(IndexError::InvalidIndex {
                url: page_url.to_string(),
                reason: format!("wheel link exceeds the {MAX_URL_BYTES}-byte limit"),
            });
        }
        let url = page_url
            .join(&href)
            .map_err(|error| IndexError::InvalidIndex {
                url: page_url.to_string(),
                reason: format!("invalid wheel link {href:?}: {error}"),
            })?;
        let Some(raw_filename) = url.path_segments().and_then(Iterator::last) else {
            continue;
        };
        let filename = percent_decode_str(raw_filename)
            .decode_utf8()
            .map_err(|error| IndexError::InvalidIndex {
                url: page_url.to_string(),
                reason: format!("wheel filename is not UTF-8: {error}"),
            })?
            .into_owned();
        if !filename.to_ascii_lowercase().ends_with(".whl") {
            continue;
        }
        policy.validate_url(&url)?;
        let requires_python = parse_requires_python(anchor.requires_python.as_deref(), page_url)?;
        let wheel = parse_wheel_link(
            package,
            variant,
            filename,
            &url,
            anchor.yanked,
            requires_python,
        )?;
        let key = (wheel.filename.clone(), wheel.url.clone());
        if let Some(existing) = wheels.insert(key, wheel.clone()) {
            if existing != wheel {
                return Err(IndexError::InvalidIndex {
                    url: page_url.to_string(),
                    reason: format!("conflicting duplicate link for {}", wheel.filename),
                });
            }
        }
        if wheels.len() > MAX_WHEELS_PER_PAGE {
            return Err(IndexError::InvalidIndex {
                url: page_url.to_string(),
                reason: format!("package index exceeds the {MAX_WHEELS_PER_PAGE}-wheel limit"),
            });
        }
    }
    Ok(wheels.into_values().collect())
}

fn parse_requires_python(
    value: Option<&str>,
    page_url: &Url,
) -> Result<Option<String>, IndexError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if value.len() > MAX_REQUIRES_PYTHON_BYTES {
        return Err(IndexError::InvalidIndex {
            url: page_url.to_string(),
            reason: format!(
                "data-requires-python exceeds the {MAX_REQUIRES_PYTHON_BYTES}-byte limit"
            ),
        });
    }
    VersionSpecifiers::from_str(value).map_err(|error| IndexError::InvalidIndex {
        url: page_url.to_string(),
        reason: format!("invalid data-requires-python {value:?}: {error}"),
    })?;
    Ok(Some(value.to_owned()))
}

fn parse_wheel_link(
    package: &str,
    variant: &CudaVariant,
    filename: String,
    url: &Url,
    yanked: bool,
    requires_python: Option<String>,
) -> Result<TorchWheel, IndexError> {
    if filename.len() > MAX_FILENAME_BYTES || url.as_str().len() > MAX_URL_BYTES {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: "wheel filename or URL exceeds the configured length limit".to_owned(),
        });
    }
    if filename
        .chars()
        .any(|character| matches!(character, '/' | '\\' | '\0'))
    {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: "decoded wheel filename contains a path separator".to_owned(),
        });
    }
    let stem = filename
        .strip_suffix(".whl")
        .ok_or_else(|| IndexError::InvalidIndex {
            url: url.to_string(),
            reason: format!("invalid wheel filename {filename}"),
        })?;
    let package_prefix = format!("{package}-");
    let body = stem
        .strip_prefix(&package_prefix)
        .ok_or_else(|| IndexError::InvalidIndex {
            url: url.to_string(),
            reason: format!("wheel filename does not match package {package}: {filename}"),
        })?;
    let parts = body.rsplitn(4, '-').collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: format!("wheel filename has invalid tag fields: {filename}"),
        });
    }
    let version_field = if let Some((version, build)) = parts[3].split_once('-') {
        if build.is_empty()
            || !build.as_bytes()[0].is_ascii_digit()
            || !build
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(IndexError::InvalidIndex {
                url: url.to_string(),
                reason: format!("invalid wheel build tag in {filename}"),
            });
        }
        version
    } else {
        parts[3]
    };
    let version = Version::from_str(version_field).map_err(|error| IndexError::InvalidIndex {
        url: url.to_string(),
        reason: format!("invalid PEP 440 wheel version in {filename}: {error}"),
    })?;
    let version = version.to_string();
    let public_version = version
        .split_once('+')
        .map_or(version.as_str(), |(public, _)| public)
        .to_owned();
    let python_tags = expand_tags(parts[2], "python", url)?;
    let abi_tags = expand_tags(parts[1], "ABI", url)?;
    let platform_tags = expand_tags(parts[0], "platform", url)?;
    let sha256 = sha256_fragment(url)?;
    Ok(TorchWheel {
        package: package.to_owned(),
        filename,
        version,
        public_version,
        variant: variant.clone(),
        python_tags,
        abi_tags,
        platform_tags,
        url: url.to_string(),
        sha256,
        yanked,
        requires_python,
    })
}

fn expand_tags(value: &str, kind: &str, url: &Url) -> Result<Vec<String>, IndexError> {
    let tags = value
        .split('.')
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if tags.is_empty()
        || tags.len() > MAX_COMPRESSED_TAGS
        || tags.iter().any(|tag| {
            tag.is_empty()
                || !tag
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
    {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: format!("invalid compressed {kind} tag {value:?}"),
        });
    }
    Ok(tags)
}

fn sha256_fragment(url: &Url) -> Result<Option<String>, IndexError> {
    let Some(fragment) = url.fragment() else {
        return Ok(None);
    };
    let mut hashes = fragment
        .split('&')
        .filter_map(|field| field.split_once('='))
        .filter(|(name, _)| *name == "sha256")
        .map(|(_, value)| value);
    let Some(hash) = hashes.next() else {
        return Ok(None);
    };
    if hashes.next().is_some()
        || hash.len() != 64
        || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(IndexError::InvalidIndex {
            url: url.to_string(),
            reason: "invalid SHA-256 URL fragment".to_owned(),
        });
    }
    Ok(Some(hash.to_ascii_lowercase()))
}

fn validate_snapshot(
    snapshot: &IndexSnapshot,
    policy: &FetchPolicy,
    now: u64,
) -> Result<(), IndexError> {
    if snapshot.schema_version != INDEX_SCHEMA_VERSION {
        return Err(schema_error(format!(
            "unsupported schema version {}",
            snapshot.schema_version
        )));
    }
    if snapshot.variants.len() > MAX_VARIANTS || snapshot.wheels.len() > MAX_TOTAL_WHEELS {
        return Err(schema_error("snapshot exceeds a structural count limit"));
    }
    if snapshot.source != policy.root.as_str() {
        return Err(schema_error("source URL does not match the official root"));
    }
    if snapshot.fetched_at > now.saturating_add(CLOCK_SKEW_SECONDS) {
        return Err(schema_error(
            "fetch timestamp is implausibly far in the future",
        ));
    }
    let packages = snapshot.packages.iter().cloned().collect::<BTreeSet<_>>();
    if packages.len() != snapshot.packages.len()
        || packages.is_empty()
        || !packages.contains("torch")
        || packages
            .iter()
            .any(|package| !ALLOWED_PACKAGES.contains(&package.as_str()))
    {
        return Err(schema_error(
            "package list is empty, duplicated, or unsupported",
        ));
    }
    let variants = snapshot.variants.iter().cloned().collect::<BTreeSet<_>>();
    if variants.is_empty() || variants.len() != snapshot.variants.len() {
        return Err(schema_error("variant list is empty or duplicated"));
    }
    if snapshot.wheels.is_empty() {
        return Err(schema_error("snapshot contains no wheel links"));
    }
    let mut wheel_keys = BTreeSet::new();
    for wheel in &snapshot.wheels {
        if !packages.contains(&wheel.package) || !variants.contains(&wheel.variant) {
            return Err(schema_error(
                "wheel references a package or variant outside the snapshot",
            ));
        }
        let url = Url::parse(&wheel.url)
            .map_err(|error| schema_error(format!("invalid wheel URL: {error}")))?;
        policy
            .validate_url(&url)
            .map_err(|error| schema_error(error.to_string()))?;
        let parsed = parse_wheel_link(
            &wheel.package,
            &wheel.variant,
            wheel.filename.clone(),
            &url,
            wheel.yanked,
            wheel.requires_python.clone(),
        )
        .map_err(|error| schema_error(error.to_string()))?;
        let linked_filename = Url::parse(&wheel.url)
            .ok()
            .and_then(|url| {
                url.path_segments()
                    .and_then(Iterator::last)
                    .map(str::to_owned)
            })
            .and_then(|name| {
                percent_decode_str(&name)
                    .decode_utf8()
                    .ok()
                    .map(std::borrow::Cow::into_owned)
            });
        if linked_filename.as_deref() != Some(wheel.filename.as_str()) {
            return Err(schema_error(format!(
                "wheel URL filename disagrees with {}",
                wheel.filename
            )));
        }
        if &parsed != wheel {
            return Err(schema_error(format!(
                "wheel fields disagree with {}",
                wheel.filename
            )));
        }
        if let Some(requires_python) = &wheel.requires_python {
            VersionSpecifiers::from_str(requires_python)
                .map_err(|error| schema_error(format!("invalid requires-python: {error}")))?;
        }
        if !wheel_keys.insert((
            wheel.package.clone(),
            wheel.variant.clone(),
            wheel.filename.clone(),
            wheel.url.clone(),
        )) {
            return Err(schema_error("duplicate wheel entry"));
        }
    }
    Ok(())
}

fn schema_error(reason: impl Into<String>) -> IndexError {
    IndexError::InvalidIndex {
        url: OFFICIAL_ROOT.to_owned(),
        reason: reason.into(),
    }
}

fn open_cache_lock(cache_dir: &Path) -> Result<File, IndexError> {
    ensure_cache_directory(cache_dir)?;
    let lock_path = cache_dir.join(LOCK_FILE);
    let file = secure_open(&lock_path, true, true)?;
    set_mode(&lock_path, &file, 0o600)?;
    let started = Instant::now();
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if started.elapsed() >= CACHE_LOCK_TIMEOUT {
                    return Err(IndexError::CacheLockTimedOut { path: lock_path });
                }
                thread::sleep(CACHE_LOCK_POLL);
            }
            Err(source) => {
                return Err(IndexError::CacheIo {
                    path: lock_path,
                    source,
                });
            }
        }
    }
}

fn ensure_cache_directory(path: &Path) -> Result<(), IndexError> {
    let created = match fs::symlink_metadata(path) {
        Ok(_) => false,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(path).map_err(|source| IndexError::CacheIo {
                path: path.to_path_buf(),
                source,
            })?;
            true
        }
        Err(source) => {
            return Err(IndexError::CacheIo {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let metadata = fs::symlink_metadata(path).map_err(|source| IndexError::CacheIo {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(IndexError::InvalidCache {
            path: path.to_path_buf(),
            reason: "cache directory must be a real directory, not a symlink".to_owned(),
        });
    }
    if created {
        set_path_mode(path, 0o700)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o022 != 0 {
            return Err(IndexError::InvalidCache {
                path: path.to_path_buf(),
                reason: "cache directory must not be group- or world-writable".to_owned(),
            });
        }
    }
    Ok(())
}

fn read_cache(
    path: &Path,
    policy: &FetchPolicy,
    now: u64,
) -> Result<Option<IndexSnapshot>, IndexError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(IndexError::CacheIo {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(IndexError::InvalidCache {
            path: path.to_path_buf(),
            reason: "cache entry must be a regular file".to_owned(),
        });
    }
    if metadata.len() > MAX_CACHE_BYTES {
        return Err(IndexError::InvalidCache {
            path: path.to_path_buf(),
            reason: "cache exceeds the size limit".to_owned(),
        });
    }
    let file = secure_open(path, false, false)?;
    set_mode(path, &file, 0o600)?;
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    file.take(MAX_CACHE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| IndexError::CacheIo {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > MAX_CACHE_BYTES {
        return Err(IndexError::InvalidCache {
            path: path.to_path_buf(),
            reason: "cache exceeds the size limit".to_owned(),
        });
    }
    let snapshot = serde_json::from_slice::<IndexSnapshot>(&bytes).map_err(|error| {
        IndexError::InvalidCache {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    validate_snapshot(&snapshot, policy, now).map_err(|error| IndexError::InvalidCache {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })?;
    Ok(Some(snapshot))
}

fn write_cache(cache_dir: &Path, path: &Path, snapshot: &IndexSnapshot) -> Result<(), IndexError> {
    ensure_cache_directory(cache_dir)?;
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(IndexError::InvalidCache {
                path: path.to_path_buf(),
                reason: "refusing to replace a non-regular cache entry".to_owned(),
            });
        }
    }
    let mut temporary = tempfile::Builder::new()
        .prefix(".index-v1-")
        .tempfile_in(cache_dir)
        .map_err(|source| IndexError::CacheIo {
            path: cache_dir.to_path_buf(),
            source,
        })?;
    set_mode(temporary.path(), temporary.as_file(), 0o600)?;
    serde_json::to_writer(temporary.as_file_mut(), snapshot)?;
    temporary
        .as_file_mut()
        .write_all(b"\n")
        .map_err(|source| IndexError::CacheIo {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|source| IndexError::CacheIo {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    if temporary
        .as_file()
        .metadata()
        .map_err(|source| IndexError::CacheIo {
            path: temporary.path().to_path_buf(),
            source,
        })?
        .len()
        > MAX_CACHE_BYTES
    {
        return Err(IndexError::InvalidCache {
            path: temporary.path().to_path_buf(),
            reason: "serialized cache exceeds the size limit".to_owned(),
        });
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| IndexError::CacheIo {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .persist(path)
        .map_err(|error| IndexError::CacheIo {
            path: path.to_path_buf(),
            source: error.error,
        })?;
    #[cfg(unix)]
    sync_cache_directory(cache_dir)?;
    Ok(())
}

#[cfg(unix)]
fn sync_cache_directory(path: &Path) -> Result<(), IndexError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| IndexError::CacheIo {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn secure_open(path: &Path, write: bool, create: bool) -> Result<File, IndexError> {
    let mut options = OpenOptions::new();
    options.read(true).write(write).create(create);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    options.open(path).map_err(|source| IndexError::CacheIo {
        path: path.to_path_buf(),
        source,
    })
}

fn set_mode(path: &Path, file: &File, mode: u32) -> Result<(), IndexError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(mode))
            .map_err(|source| IndexError::CacheIo {
                path: path.to_path_buf(),
                source,
            })?;
    }
    let _ = (path, file, mode);
    Ok(())
}

fn set_path_mode(path: &Path, mode: u32) -> Result<(), IndexError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|source| {
            IndexError::CacheIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
    }
    let _ = (path, mode);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy() -> FetchPolicy {
        FetchPolicy::official().expect("valid built-in policy")
    }

    #[test]
    fn defaults_request_only_torch() {
        let options = IndexOptions::default();
        assert_eq!(
            validate_options(&options).expect("valid defaults"),
            vec!["torch".to_owned()]
        );
        assert_eq!(options.ttl, Duration::from_secs(86_400));
    }

    #[test]
    fn invalid_package_and_conflicting_modes_are_rejected() {
        let mut options = IndexOptions {
            offline: true,
            refresh: true,
            ..IndexOptions::default()
        };
        assert!(validate_options(&options).is_err());
        options.packages = vec!["torch".to_owned()];
        options.total_timeout = Duration::ZERO;
        assert!(validate_options(&options).is_err());
        options.offline = false;
        options.refresh = false;
        options.packages.push("evil".to_owned());
        assert!(validate_options(&options).is_err());
    }

    #[test]
    fn variant_discovery_accepts_only_direct_cpu_and_cuda_children() {
        let html = r#"
          <a href="cpu/">cpu</a><a href="cu124/">cu124</a>
          <a href="rocm6.3/">rocm</a><a href="cu124/torch/">nested</a>
          <a href="https://example.invalid/cu130/">foreign</a>
        "#;
        let variants = discover_variants(html, &test_policy()).expect("parse variants");
        assert_eq!(
            variants,
            [CudaVariant::Cpu, "cu124".parse().expect("variant")]
        );
    }

    #[test]
    fn anchor_tokenizer_ignores_markup_inside_raw_text_elements() {
        let html = r#"
          <script><a href="cu130/">not a link</a></script>
          <style><a href="cu131/">not a link</a></style>
          <a href="cu124/">cu124</a>
        "#;
        assert_eq!(
            discover_variants(html, &test_policy()).expect("parse variants"),
            ["cu124".parse().expect("variant")]
        );
    }

    #[test]
    fn index_structure_limits_bound_variant_and_tag_expansion() {
        use std::fmt::Write as _;

        let html = (100..=164).fold(String::new(), |mut html, variant| {
            write!(&mut html, r#"<a href="cu{variant}/">cu{variant}</a>"#)
                .expect("write HTML fixture");
            html
        });
        assert!(matches!(
            discover_variants(&html, &test_policy()),
            Err(IndexError::InvalidIndex { .. })
        ));

        let tags = std::iter::repeat_n("cp313", MAX_COMPRESSED_TAGS + 1)
            .collect::<Vec<_>>()
            .join(".");
        assert!(matches!(
            expand_tags(
                &tags,
                "python",
                &Url::parse("https://download.pytorch.org/whl/cpu/torch/").expect("URL")
            ),
            Err(IndexError::InvalidIndex { .. })
        ));
    }

    #[test]
    fn parses_encoded_local_version_compressed_tags_hash_and_attributes() {
        let policy = test_policy();
        let page = Url::parse("https://download.pytorch.org/whl/cu124/torch/").expect("URL");
        let hash = "a".repeat(64);
        let html = format!(
            r#"<a href="../../torch-2.6.0%2Bcu124-cp312.cp313-cp312.cp313-manylinux_2_17_x86_64.whl#sha256={hash}" data-requires-python="&gt;=3.9" data-yanked="bad build">wheel</a>"#
        );
        let wheels = parse_wheel_page(
            &html,
            &page,
            &policy,
            &"cu124".parse().expect("variant"),
            "torch",
        )
        .expect("parse wheel");
        assert_eq!(wheels.len(), 1);
        let wheel = &wheels[0];
        assert_eq!(wheel.version, "2.6.0+cu124");
        assert_eq!(wheel.public_version, "2.6.0");
        assert_eq!(wheel.python_tags, ["cp312", "cp313"]);
        assert_eq!(wheel.sha256.as_deref(), Some(hash.as_str()));
        assert!(wheel.yanked);
        assert_eq!(wheel.requires_python.as_deref(), Some(">=3.9"));
    }

    #[test]
    fn malformed_pep440_and_hash_fail_the_whole_page() {
        let policy = test_policy();
        let page = Url::parse("https://download.pytorch.org/whl/cu124/torch/").expect("URL");
        for href in [
            "torch-not!pep440-cp313-cp313-linux_x86_64.whl",
            "torch-2.6.0-cp313-cp313-linux_x86_64.whl#sha256=abcd",
        ] {
            let html = format!(r#"<a href="{href}">wheel</a>"#);
            assert!(
                parse_wheel_page(
                    &html,
                    &page,
                    &policy,
                    &"cu124".parse().expect("variant"),
                    "torch"
                )
                .is_err(),
                "accepted {href}"
            );
        }
    }

    #[test]
    fn accepts_the_optional_pep427_build_tag_used_by_official_wheels() {
        let url = Url::parse(
            "https://download-r2.pytorch.org/whl/torch-2.0.0-1-cp311-cp311-manylinux2014_aarch64.whl",
        )
        .expect("URL");
        let wheel = parse_wheel_link(
            "torch",
            &"cu124".parse().expect("variant"),
            "torch-2.0.0-1-cp311-cp311-manylinux2014_aarch64.whl".to_owned(),
            &url,
            false,
            None,
        )
        .expect("build-tagged wheel");
        assert_eq!(wheel.version, "2.0.0");
    }

    #[test]
    fn url_policy_is_exact_and_https_only() {
        let policy = test_policy();
        for accepted in [
            "https://download.pytorch.org/whl/",
            "https://download-r2.pytorch.org/whl/a.whl",
        ] {
            policy
                .validate_url(&Url::parse(accepted).expect("URL"))
                .expect("allowed");
        }
        for rejected in [
            "http://download.pytorch.org/whl/",
            "https://download.pytorch.org.evil.test/whl/",
            "https://example.invalid/whl/",
            "https://download.pytorch.org:444/whl/",
        ] {
            assert!(
                policy
                    .validate_url(&Url::parse(rejected).expect("URL"))
                    .is_err()
            );
        }
    }

    #[test]
    fn cache_requires_all_requested_packages() {
        let snapshot = IndexSnapshot {
            schema_version: 1,
            fetched_at: 1,
            source: OFFICIAL_ROOT.to_owned(),
            packages: vec!["torch".to_owned(), "torchvision".to_owned()],
            variants: vec![CudaVariant::Cpu],
            wheels: Vec::new(),
        };
        assert!(snapshot_contains(&snapshot, &["torch".to_owned()]));
        assert!(!snapshot_contains(
            &snapshot,
            &["torch".to_owned(), "torchaudio".to_owned()]
        ));
    }

    #[test]
    fn cache_files_are_isolated_by_package_set_and_supersets_are_discoverable() {
        let directory = Path::new("/cache");
        let torch = vec!["torch".to_owned()];
        let full = vec![
            "torch".to_owned(),
            "torchaudio".to_owned(),
            "torchvision".to_owned(),
        ];
        assert_ne!(
            cache_path_for(directory, &torch),
            cache_path_for(directory, &full)
        );
        assert!(cache_package_sets(&torch).contains(&full));
    }

    #[cfg(unix)]
    #[test]
    fn existing_cache_directory_permissions_are_not_mutated() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary directory");
        let cache = directory.path().join("existing-cache");
        fs::create_dir(&cache).expect("create cache");
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o755))
            .expect("set fixture permissions");

        ensure_cache_directory(&cache).expect("accept safe existing directory");

        assert_eq!(
            fs::metadata(&cache).expect("metadata").permissions().mode() & 0o777,
            0o755
        );
    }

    #[test]
    fn decompressed_response_limit_is_enforced_without_partial_append() {
        let mut bytes = b"1234".to_vec();
        let error = append_bounded(&mut bytes, b"56789", 8, OFFICIAL_ROOT)
            .expect_err("must reject oversized response");
        assert!(matches!(error, IndexError::ResponseTooLarge { .. }));
        assert_eq!(bytes, b"1234");
    }

    #[test]
    fn one_failed_page_prevents_a_partial_snapshot() {
        let results = vec![
            Ok(Vec::new()),
            Err(IndexError::HttpStatus {
                url: "https://download.pytorch.org/whl/cu124/torch/".to_owned(),
                status: 503,
            }),
        ];
        assert!(matches!(
            collect_complete_pages(results),
            Err(IndexError::IncompleteFetch(_))
        ));
    }

    #[test]
    fn cache_write_is_atomic_and_permissions_are_private() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let cache_dir = directory.path().join("cache");
        ensure_cache_directory(&cache_dir).expect("prepare cache");
        let path = cache_path_for(&cache_dir, &["torch".to_owned()]);
        let wheel_url = format!(
            "https://download.pytorch.org/whl/cpu/torch-2.6.0-cp313-cp313-linux_x86_64.whl#sha256={}",
            "a".repeat(64)
        );
        let snapshot = IndexSnapshot {
            schema_version: INDEX_SCHEMA_VERSION,
            fetched_at: unix_now().expect("clock"),
            source: OFFICIAL_ROOT.to_owned(),
            packages: vec!["torch".to_owned()],
            variants: vec![CudaVariant::Cpu],
            wheels: vec![
                parse_wheel_link(
                    "torch",
                    &CudaVariant::Cpu,
                    "torch-2.6.0-cp313-cp313-linux_x86_64.whl".to_owned(),
                    &Url::parse(&wheel_url).expect("URL"),
                    false,
                    None,
                )
                .expect("wheel"),
            ],
        };
        write_cache(&cache_dir, &path, &snapshot).expect("write cache");
        let loaded = read_cache(&path, &test_policy(), unix_now().expect("clock"))
            .expect("read cache")
            .expect("snapshot");
        assert_eq!(loaded, snapshot);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&cache_dir)
                    .expect("metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
                0o600
            );
        }
    }
}
