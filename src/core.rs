//! Stable data types shared by detection, index, resolution, and verification.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Version of the machine-readable output contract.
pub const SCHEMA_VERSION: u32 = 1;

/// Returns whether a character can alter terminal layout or displayed text,
/// or is visually confusable with an ordinary ASCII space.
///
/// This covers C0/C1 controls, Unicode line/paragraph separators, and the
/// currently assigned Unicode format-control ranges. Callers either escape
/// these characters for diagnostics or reject them where exact copy/paste
/// semantics are required.
pub(crate) fn is_unsafe_terminal_character(character: char) -> bool {
    character.is_control()
        || (character.is_whitespace() && character != ' ')
        || matches!(
            character,
            '\u{00ad}'
                | '\u{0600}'..='\u{0605}'
                | '\u{061c}'
                | '\u{06dd}'
                | '\u{070f}'
                | '\u{0890}'..='\u{0891}'
                | '\u{08e2}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{2028}'..='\u{202e}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{110bd}'
                | '\u{110cd}'
                | '\u{13430}'..='\u{1343f}'
                | '\u{1bca0}'..='\u{1bca3}'
                | '\u{1d173}'..='\u{1d17a}'
                | '\u{e0001}'
                | '\u{e0020}'..='\u{e007f}'
        )
}

/// A dotted numeric version such as an NVIDIA driver or glibc version.
///
/// Components are compared numerically, while the original representation is
/// retained for diagnostics and machine-readable inspection output. This is
/// significant for NVIDIA driver versions such as `530.30.02`, where the
/// vendor-reported zero padding should not be discarded when the value is
/// displayed.
#[derive(Debug, Clone)]
pub struct NumericVersion {
    components: Vec<u32>,
    representation: String,
}

impl NumericVersion {
    /// Builds a version from one or more numeric components.
    pub fn new(components: Vec<u32>) -> Result<Self, VersionParseError> {
        if components.is_empty() {
            return Err(VersionParseError::Empty);
        }
        let representation = components
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(".");
        Ok(Self {
            components,
            representation,
        })
    }

    /// Returns the numeric components.
    pub fn components(&self) -> &[u32] {
        &self.components
    }

    /// Returns a component, treating a missing trailing component as zero.
    pub fn component(&self, index: usize) -> u32 {
        self.components.get(index).copied().unwrap_or(0)
    }
}

impl PartialEq for NumericVersion {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for NumericVersion {}

impl Hash for NumericVersion {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let significant_len = self
            .components
            .iter()
            .rposition(|component| *component != 0)
            .map_or(1, |index| index + 1);
        self.components[..significant_len].hash(state);
    }
}

impl FromStr for NumericVersion {
    type Err = VersionParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() {
            return Err(VersionParseError::Empty);
        }
        let mut components = Vec::new();
        for part in value.split('.') {
            if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(VersionParseError::InvalidComponent(part.to_owned()));
            }
            components.push(
                part.parse::<u32>()
                    .map_err(|_| VersionParseError::Overflow(part.to_owned()))?,
            );
        }
        Ok(Self {
            components,
            representation: value.to_owned(),
        })
    }
}

impl Display for NumericVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.representation)
    }
}

impl Ord for NumericVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        let len = self.components.len().max(other.components.len());
        (0..len)
            .map(|index| self.component(index).cmp(&other.component(index)))
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for NumericVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Serialize for NumericVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for NumericVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Error returned for a malformed dotted numeric version.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum VersionParseError {
    /// No version was provided.
    #[error("version is empty")]
    Empty,
    /// A component was not numeric.
    #[error("invalid numeric version component: {0}")]
    InvalidComponent(String),
    /// A component exceeded the supported numeric range.
    #[error("numeric version component is too large: {0}")]
    Overflow(String),
}

/// A PyTorch wheel accelerator index.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CudaVariant {
    /// CPU-only wheels.
    Cpu,
    /// A CUDA runtime wheel, for example CUDA 12.4 (`cu124`).
    Cuda {
        /// CUDA major version.
        major: u16,
        /// CUDA minor version.
        minor: u16,
    },
}

impl CudaVariant {
    /// Returns the CUDA major/minor pair, or `None` for a CPU wheel.
    pub fn cuda_version(&self) -> Option<(u16, u16)> {
        match self {
            Self::Cpu => None,
            Self::Cuda { major, minor } => Some((*major, *minor)),
        }
    }

    /// Returns whether this is a CUDA wheel.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda { .. })
    }
}

impl Display for CudaVariant {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda { major, minor } => write!(formatter, "cu{major}{minor}"),
        }
    }
}

impl FromStr for CudaVariant {
    type Err = CudaVariantParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized == "cpu" {
            return Ok(Self::Cpu);
        }
        let digits = normalized
            .strip_prefix("cu")
            .ok_or_else(|| CudaVariantParseError(value.to_owned()))?;
        if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(CudaVariantParseError(value.to_owned()));
        }
        let split = if digits.len() >= 3 { 2 } else { 1 };
        let (major, minor) = digits.split_at(split);
        let major = major
            .parse::<u16>()
            .map_err(|_| CudaVariantParseError(value.to_owned()))?;
        let minor = minor
            .parse::<u16>()
            .map_err(|_| CudaVariantParseError(value.to_owned()))?;
        Ok(Self::Cuda { major, minor })
    }
}

impl Serialize for CudaVariant {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for CudaVariant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Error returned for an invalid CUDA wheel index name.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("invalid CUDA variant: {0}")]
pub struct CudaVariantParseError(pub String);

/// Operating-system family detected for the target interpreter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum OperatingSystem {
    /// Linux.
    Linux,
    /// Windows.
    Windows,
    /// macOS.
    Macos,
    /// An OS outside the currently supported matrix.
    Other(String),
}

impl Display for OperatingSystem {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linux => formatter.write_str("linux"),
            Self::Windows => formatter.write_str("windows"),
            Self::Macos => formatter.write_str("macos"),
            Self::Other(name) => formatter.write_str(name),
        }
    }
}

impl Serialize for OperatingSystem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for OperatingSystem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            "macos" => Self::Macos,
            _ => Self::Other(value),
        })
    }
}

/// CPU architecture detected on the host.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Architecture {
    /// AMD/Intel 64-bit architecture.
    X86_64,
    /// 64-bit Arm architecture.
    Aarch64,
    /// An architecture outside the currently supported matrix.
    Other(String),
}

impl Display for Architecture {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X86_64 => formatter.write_str("x86_64"),
            Self::Aarch64 => formatter.write_str("aarch64"),
            Self::Other(name) => formatter.write_str(name),
        }
    }
}

impl Serialize for Architecture {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Architecture {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "x86_64" | "amd64" => Self::X86_64,
            "aarch64" | "arm64" => Self::Aarch64,
            _ => Self::Other(value),
        })
    }
}

/// Host platform details.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// OS family.
    pub os: OperatingSystem,
    /// CPU architecture.
    pub architecture: Architecture,
    /// Kernel release, when available.
    pub kernel_version: Option<String>,
    /// Human-readable Linux distribution name, when available.
    pub distribution: Option<String>,
}

/// How compatible tags were obtained for a Python interpreter.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagSource {
    /// Supplied by a library caller from `packaging.tags.sys_tags()`.
    ///
    /// The built-in detector deliberately does not import site packages in its isolated probe.
    Packaging,
    /// Produced by the dependency-free built-in tag implementation.
    Builtin,
}

/// Properties of the selected Python interpreter.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PythonInfo {
    /// Exact executable used for probes and generated commands.
    pub executable: PathBuf,
    /// Python implementation name, normalized to lowercase.
    pub implementation: String,
    /// Interpreter version.
    pub version: NumericVersion,
    /// `SOABI`, if supplied by the interpreter.
    pub soabi: Option<String>,
    /// `sys.implementation.cache_tag`.
    pub cache_tag: Option<String>,
    /// `sysconfig.get_platform()`.
    pub platform: String,
    /// Pointer width in bits.
    pub pointer_width: u16,
    /// Whether this is a free-threaded CPython build.
    pub free_threaded: bool,
    /// Active virtual-environment prefix, if any.
    pub virtual_environment: Option<PathBuf>,
    /// Compatible PEP 425 tag triples, in interpreter preference order.
    pub compatible_tags: Vec<String>,
    /// Source used for `compatible_tags`.
    pub tag_source: TagSource,
}

impl PythonInfo {
    /// Returns the native CPython ABI tag, such as `cp313` or `cp313t`.
    pub fn cpython_abi_tag(&self) -> Option<String> {
        if self.implementation != "cpython" {
            return None;
        }
        let suffix = if self.free_threaded { "t" } else { "" };
        Some(format!(
            "cp{}{}{}",
            self.version.component(0),
            self.version.component(1),
            suffix
        ))
    }
}

/// GPU compute capability.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Compute-capability major number.
    pub major: u16,
    /// Compute-capability minor number.
    pub minor: u16,
}

impl Display for ComputeCapability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

impl FromStr for ComputeCapability {
    type Err = VersionParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let version: NumericVersion = value.parse()?;
        Ok(Self {
            major: u16::try_from(version.component(0))
                .map_err(|_| VersionParseError::Overflow(value.to_owned()))?,
            minor: u16::try_from(version.component(1))
                .map_err(|_| VersionParseError::Overflow(value.to_owned()))?,
        })
    }
}

/// A GPU returned by `nvidia-smi`.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct NvidiaGpu {
    /// Physical index reported by NVIDIA.
    pub index: u32,
    /// NVIDIA UUID, when available.
    pub uuid: Option<String>,
    /// Product name.
    pub name: String,
    /// Compute capability, if the installed query interface exposes it.
    pub compute_capability: Option<ComputeCapability>,
    /// Driver version reported for the device.
    pub driver_version: NumericVersion,
}

/// Result of trying to inspect the NVIDIA stack.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NvidiaDetectionStatus {
    /// At least one device was queried successfully.
    Detected,
    /// `nvidia-smi` was not found.
    CommandUnavailable,
    /// NVIDIA explicitly reported that there were no devices.
    NoDevices,
    /// The command was present but inspection failed.
    Failed,
    /// Inspection exceeded its timeout.
    TimedOut,
}

/// NVIDIA driver and device information.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct NvidiaInfo {
    /// Detection outcome.
    pub status: NvidiaDetectionStatus,
    /// Lowest driver version across selected devices.
    pub driver_version: Option<NumericVersion>,
    /// The informational `CUDA Version` field shown by `nvidia-smi`.
    pub reported_cuda_version: Option<NumericVersion>,
    /// Selected devices.
    pub gpus: Vec<NvidiaGpu>,
}

/// Where a local CUDA Toolkit version was obtained.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolkitSource {
    /// `CUDA_HOME`.
    CudaHome,
    /// `CUDA_PATH`.
    CudaPath,
    /// An `nvcc` found through `PATH`.
    Nvcc,
    /// A CUDA `version.json` file.
    VersionJson,
}

/// Optional local CUDA Toolkit installation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaToolkitInfo {
    /// Toolkit version.
    pub version: NumericVersion,
    /// Toolkit root, when known.
    pub root: Option<PathBuf>,
    /// Detection source.
    pub source: ToolkitSource,
}

/// Stable diagnostic code emitted during environment detection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCode {
    /// Host OS is outside the supported resolution target.
    UnsupportedOperatingSystem,
    /// Host architecture is outside the supported resolution target.
    UnsupportedArchitecture,
    /// Host C library is known to be unsupported by the available wheel policy.
    UnsupportedLibc,
    /// The C library could not be identified.
    GlibcUnknown,
    /// No Python interpreter could be selected.
    PythonUnavailable,
    /// The selected Python is not CPython.
    UnsupportedPythonImplementation,
    /// Legacy code retained for schema compatibility with reports that treated built-in tags as a
    /// fallback.
    BuiltinPythonTags,
    /// `nvidia-smi` was unavailable.
    NvidiaSmiUnavailable,
    /// NVIDIA reported no devices.
    NvidiaNoDevices,
    /// NVIDIA inspection failed.
    NvidiaInspectionFailed,
    /// NVIDIA inspection timed out.
    NvidiaInspectionTimedOut,
    /// Compute capability was unavailable from an older query interface.
    ComputeCapabilityUnknown,
    /// Selected GPU indices were invalid.
    InvalidGpuSelection,
    /// CUDA visibility may differ from physical NVIDIA indices.
    CudaVisibleDevicesSet,
    /// Multiple driver versions were returned unexpectedly.
    InconsistentDriverVersions,
    /// Local CUDA Toolkit was not found; this is informational for wheels.
    CudaToolkitNotFound,
    /// The optional cuDNN information query failed after core verification passed.
    CudnnInspectionFailed,
}

/// Severity of an environment diagnostic.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// Additional context only.
    Info,
    /// A condition that reduces confidence without proving incompatibility.
    Warning,
    /// An explicit unsupported or invalid condition.
    Error,
}

/// Structured diagnostic with stable parameters.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Stable code.
    pub code: DiagnosticCode,
    /// Severity.
    pub severity: DiagnosticSeverity,
    /// Machine-readable parameters.
    pub details: BTreeMap<String, String>,
}

/// Complete environment snapshot.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    /// Platform details.
    #[serde(flatten)]
    pub platform: PlatformInfo,
    /// glibc version on GNU/Linux, when known.
    pub glibc: Option<NumericVersion>,
    /// Selected Python interpreter, when detected.
    pub python: Option<PythonInfo>,
    /// NVIDIA stack result.
    pub nvidia: NvidiaInfo,
    /// Local toolkit; not required to run official PyTorch CUDA wheels.
    pub cuda_toolkit: Option<CudaToolkitInfo>,
    /// Structured detection diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// One official wheel link normalized from the PyTorch package index.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TorchWheel {
    /// Normalized package name (`torch`, `torchvision`, or `torchaudio`).
    pub package: String,
    /// Wheel filename after percent-decoding.
    pub filename: String,
    /// Full PEP 440 version, including any local CUDA label.
    pub version: String,
    /// Public PEP 440 version used in install commands.
    pub public_version: String,
    /// Index variant from which the wheel was obtained.
    pub variant: CudaVariant,
    /// Compressed Python tags expanded to individual values.
    pub python_tags: Vec<String>,
    /// Compressed ABI tags expanded to individual values.
    pub abi_tags: Vec<String>,
    /// Compressed platform tags expanded to individual values.
    pub platform_tags: Vec<String>,
    /// Official HTTPS wheel URL.
    pub url: String,
    /// SHA-256 from the URL fragment, when supplied.
    pub sha256: Option<String>,
    /// Whether the simple index marks this file as yanked.
    pub yanked: bool,
    /// Optional `data-requires-python` value.
    pub requires_python: Option<String>,
}

/// Complete, all-or-nothing metadata snapshot.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Cache schema version.
    pub schema_version: u32,
    /// UNIX timestamp at the start of the successful fetch.
    pub fetched_at: u64,
    /// Root source URL.
    pub source: String,
    /// Package indexes included in this complete snapshot.
    pub packages: Vec<String>,
    /// Variants fetched successfully.
    pub variants: Vec<CudaVariant>,
    /// Parsed official wheels.
    pub wheels: Vec<TorchWheel>,
}

/// How metadata was obtained for a command invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataOrigin {
    /// Downloaded during this invocation.
    Network,
    /// Loaded from a fresh cache entry.
    FreshCache,
    /// Loaded from stale cache in explicit offline mode.
    OfflineCache,
    /// Loaded from stale cache after a network error.
    StaleIfError,
}

/// Metadata provenance attached to resolver output.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataInfo {
    /// Origin used for this invocation.
    pub origin: MetadataOrigin,
    /// Fetch timestamp from the snapshot.
    pub fetched_at: u64,
    /// Age at resolution time.
    pub age_seconds: u64,
    /// Whether the age exceeds the configured TTL.
    pub stale: bool,
    /// Official root URL.
    pub source: String,
}

/// A single compatibility dimension.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// The dimension is satisfied.
    Pass,
    /// The dimension proves the candidate cannot be used.
    Fail,
    /// There is not enough trustworthy information.
    Unknown,
    /// The dimension does not apply, for example GPU coverage on CPU wheels.
    NotApplicable,
}

/// Stable reason code used by JSON output and human rendering.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// The exact official wheel exists.
    WheelExists,
    /// No wheel exists for the requested version/variant.
    WheelMissing,
    /// Matching files exist, but none supports the selected Python tag.
    PythonWheelMissing,
    /// Python tag matched.
    PythonTagMatches,
    /// ABI tag matched.
    AbiTagMatches,
    /// The wheel is yanked and is excluded from recommendations.
    WheelYanked,
    /// Platform tag matched.
    PlatformTagMatches,
    /// Platform tag is incompatible.
    PlatformMismatch,
    /// glibc is below a manylinux floor.
    GlibcTooOld,
    /// glibc could not be checked.
    GlibcUnknown,
    /// CUDA family minimum driver is met.
    DriverSupportsCudaFamily,
    /// Normal driver minimum for the exact CUDA release is met.
    DriverSupportsCudaVariant,
    /// Candidate relies on CUDA minor-version compatibility.
    UsesCudaMinorCompatibility,
    /// Driver is below the CUDA-family minimum.
    DriverTooOld,
    /// NVIDIA driver information is unavailable.
    DriverUnknown,
    /// CPU wheel does not need an NVIDIA driver.
    DriverNotRequired,
    /// GPU architecture coverage is explicitly known.
    GpuArchitectureSupported,
    /// At least one selected GPU is explicitly unsupported.
    GpuArchitectureUnsupported,
    /// Static wheel GPU coverage is not trustworthy for this pair.
    GpuArchitectureUnknown,
    /// No NVIDIA device was found for a CUDA candidate.
    NvidiaGpuUnavailable,
    /// User-supplied version constraint matched.
    VersionConstraintMatches,
    /// User-supplied version constraint did not match.
    VersionConstraintMismatch,
    /// Stable release preferred.
    StableRelease,
    /// Pre-release is not selected by default.
    Prerelease,
    /// This configuration is listed by the official PyTorch release matrix.
    OfficialReleaseConfiguration,
    /// The wheel exists in the index but is absent from the reviewed release matrix.
    NotOfficialReleaseConfiguration,
    /// No release-matrix preference was available.
    OfficialPreferenceUnknown,
    /// Dynamic verification passed in this invocation.
    RuntimeVerified,
    /// Dynamic verification was not run.
    RuntimeNotRun,
    /// Dynamic verification failed.
    RuntimeVerificationFailed,
}

/// Stable warning code for a candidate.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCode {
    /// PTX JIT or newer driver-dependent features may still fail.
    PtxOrNewDriverFeatureMayFail,
    /// Static GPU architecture coverage is unknown.
    GpuArchitectureUnknown,
    /// Metadata is older than the configured TTL.
    StaleMetadata,
    /// The NVIDIA query interface was unavailable or failed.
    NvidiaDetectionIncomplete,
    /// Legacy code retained for schema compatibility with candidates that treated built-in tags as
    /// a fallback.
    BuiltinPythonTags,
    /// Official release preference is not known for this pair.
    OfficialPreferenceUnknown,
    /// The indexed wheel is not part of the reviewed official release configuration.
    NotOfficialReleaseConfiguration,
}

/// A structured reason or warning parameter map.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecisionReason {
    /// Stable code.
    pub code: ReasonCode,
    /// Machine-readable parameters.
    pub details: BTreeMap<String, String>,
}

impl DecisionReason {
    /// Creates a reason without parameters.
    pub fn new(code: ReasonCode) -> Self {
        Self {
            code,
            details: BTreeMap::new(),
        }
    }

    /// Adds one parameter.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

/// Candidate warning with stable parameters.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CandidateWarning {
    /// Stable code.
    pub code: WarningCode,
    /// Machine-readable parameters.
    pub details: BTreeMap<String, String>,
}

impl CandidateWarning {
    /// Creates a warning without parameters.
    pub fn new(code: WarningCode) -> Self {
        Self {
            code,
            details: BTreeMap::new(),
        }
    }
}

/// Result for one orthogonal compatibility dimension.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityCheck {
    /// Pass/fail/unknown/not-applicable status.
    pub status: CheckStatus,
    /// Ordered reasons for this dimension.
    pub reasons: Vec<DecisionReason>,
}

impl CompatibilityCheck {
    /// Creates a check result.
    pub fn new(status: CheckStatus, reasons: Vec<DecisionReason>) -> Self {
        Self { status, reasons }
    }
}

/// Orthogonal checks retained for every candidate.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityChecks {
    /// Official wheel existence.
    pub wheel: CompatibilityCheck,
    /// Python and ABI tags.
    pub python: CompatibilityCheck,
    /// OS, architecture, and libc tags.
    pub platform: CompatibilityCheck,
    /// Static GPU architecture coverage.
    pub gpu_architecture: CompatibilityCheck,
    /// NVIDIA driver compatibility.
    pub driver: CompatibilityCheck,
    /// Dynamic verification; normally not run during static resolution.
    pub runtime: CompatibilityCheck,
}

/// User-facing aggregate candidate state derived from orthogonal checks.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompatibilityStatus {
    /// Dynamic validation succeeded for this exact environment.
    Verified,
    /// All static checks pass with the normal exact-variant driver minimum.
    DirectCompatible,
    /// All static checks pass using CUDA minor-version compatibility.
    MinorCompatible,
    /// Nothing disproves compatibility, but at least one dimension is unknown.
    Unverified,
    /// A known environment constraint fails.
    Incompatible,
    /// No official wheel exists for the requested version/variant/Python ABI.
    Unavailable,
}

/// A resolved PyTorch configuration.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Public PyTorch version.
    pub torch_version: String,
    /// CUDA or CPU wheel index.
    pub variant: CudaVariant,
    /// Aggregate state.
    pub compatibility: CompatibilityStatus,
    /// Orthogonal evidence.
    pub checks: CompatibilityChecks,
    /// The best exact wheel for the selected interpreter, when present.
    pub wheel: Option<TorchWheel>,
    /// Whether this is a stable PEP 440 release.
    pub stable: bool,
    /// Rank in the maintained official release configuration, if known.
    pub official_preference: Option<u16>,
    /// Ordered warnings.
    pub warnings: Vec<CandidateWarning>,
}

/// An executable command represented without shell evaluation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// Executable name/path.
    pub program: String,
    /// Exact argument vector.
    pub args: Vec<String>,
    /// POSIX-shell display rendering for humans.
    pub display: String,
}

/// Installer mode selected by the user.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Installer {
    /// `<python> -m pip install`, or `pip install` only for a caller-proven active target.
    Pip,
    /// `uv pip install`, with `--python <python>` unless the active target is proven.
    Uv,
    /// `uv add --index ...` for a project dependency, normally with `--python <python>`.
    UvAdd,
}

/// Recommendation and alternatives produced from one complete snapshot.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecommendationReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Environment used for resolution.
    pub environment: Environment,
    /// Metadata provenance.
    pub metadata: MetadataInfo,
    /// Highest-ranked usable candidate, if any.
    pub recommendation: Option<Candidate>,
    /// Remaining usable candidates.
    pub alternatives: Vec<Candidate>,
    /// Incompatible or unavailable candidates requested for explanation.
    pub excluded: Vec<Candidate>,
    /// Install command for the recommendation.
    pub install: Option<CommandSpec>,
}

/// `candidates` JSON envelope.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CandidatesReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Environment used for resolution.
    pub environment: Environment,
    /// Metadata provenance.
    pub metadata: MetadataInfo,
    /// Ordered candidates selected for display.
    pub candidates: Vec<Candidate>,
}

/// `explain` JSON envelope.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExplainReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Environment used for resolution.
    pub environment: Environment,
    /// Metadata provenance.
    pub metadata: MetadataInfo,
    /// Exact requested candidate and all orthogonal checks.
    pub candidate: Candidate,
}

/// `inspect` JSON envelope.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct InspectReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Environment snapshot.
    pub environment: Environment,
}

/// Stable top-level error category for machine-readable failures.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Invalid command input that was not rejected directly by the argument parser.
    Usage,
    /// Environment detection could not complete.
    Detection,
    /// Network or cached metadata was unavailable.
    Metadata,
    /// Compatibility resolution failed.
    Resolution,
    /// Runtime verification failed before it could produce a normal report.
    Verification,
    /// Unexpected internal failure.
    Internal,
}

/// Machine-readable error body.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Stable category.
    pub kind: ErrorKind,
    /// Stable, lower-snake-case error code.
    pub code: String,
    /// Human-readable context. Consumers should branch on `kind` and `code`.
    pub message: String,
}

/// JSON error envelope used by every subcommand.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Structured error.
    pub error: ErrorBody,
}

/// One runtime validation step.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerificationCheck {
    /// Stable step name.
    pub name: String,
    /// Whether the step passed.
    pub passed: bool,
    /// Optional non-sensitive detail.
    pub detail: Option<String>,
}

/// One device exercised by the verification subprocess.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerifiedDevice {
    /// Logical PyTorch device index.
    pub index: u32,
    /// Device name.
    pub name: String,
    /// Device capability.
    pub capability: ComputeCapability,
    /// Whether allocation, elementwise, matmul, and synchronization passed.
    pub operations_ok: bool,
}

/// Mapping applied when a physical `nvidia-smi` selection is pinned for verification.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerificationGpuMapping {
    /// Physical index reported by `nvidia-smi`.
    pub physical_index: u32,
    /// Logical CUDA index after UUID-based visibility narrowing.
    pub logical_index: u32,
    /// Full NVIDIA GPU UUID, omitted by `--redact`.
    pub uuid: Option<String>,
}

/// Dynamic validation result tied to an exact interpreter and device set.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Output schema version.
    pub schema_version: u32,
    /// Selected interpreter.
    pub python_executable: PathBuf,
    /// Aggregate runtime result.
    pub status: CompatibilityStatus,
    /// Imported PyTorch version.
    pub torch_version: Option<String>,
    /// CUDA version compiled into PyTorch.
    pub compiled_cuda: Option<String>,
    /// `torch.cuda.is_available()`.
    pub cuda_available: Option<bool>,
    /// Logical CUDA device count reported by PyTorch.
    pub device_count: Option<u32>,
    /// Device architecture list compiled into PyTorch.
    pub arch_list: Vec<String>,
    /// Devices actually exercised.
    pub devices: Vec<VerifiedDevice>,
    /// Physical-to-logical mapping used for an explicit `--gpu` selection.
    pub gpu_selection: Vec<VerificationGpuMapping>,
    /// cuDNN availability, treated as informational.
    pub cudnn_available: Option<bool>,
    /// Ordered validation steps.
    pub checks: Vec<VerificationCheck>,
    /// Non-fatal environment and runtime diagnostics relevant to this verification.
    pub diagnostics: Vec<Diagnostic>,
    /// Failure detail suitable for diagnostics, if validation failed.
    pub error: Option<String>,
}

/// Documented process exit codes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(i32)]
pub enum ExitCode {
    /// Successful result with no warnings.
    Success = 0,
    /// Successful result with diagnostic warnings.
    Warning = 1,
    /// Incompatible environment or failed runtime verification.
    Incompatible = 2,
    /// Required metadata could not be obtained.
    MetadataFailure = 3,
    /// Unexpected internal error.
    InternalError = 4,
}

impl ExitCode {
    /// Returns the integer process status.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn numeric_versions_ignore_insignificant_trailing_zeroes_for_ordering() {
        let short: NumericVersion = "525.60".parse().expect("valid version");
        let long: NumericVersion = "525.60.0".parse().expect("valid version");
        assert_eq!(short.cmp(&long), Ordering::Equal);
        assert_eq!(short, long);
        assert_eq!(long.to_string(), "525.60.0");
    }

    #[test]
    fn numeric_versions_compare_each_component_numerically() {
        let older: NumericVersion = "530.9.99".parse().expect("valid version");
        let newer: NumericVersion = "530.30.2".parse().expect("valid version");
        assert!(older < newer);
    }

    #[test]
    fn numeric_versions_preserve_reported_zero_padding() {
        for reported in ["530.30.02", "575.57.08"] {
            let version: NumericVersion = reported.parse().expect("valid version");
            assert_eq!(version.to_string(), reported);
            assert_eq!(
                serde_json::to_string(&version).expect("serialize"),
                format!("\"{reported}\"")
            );
        }

        let padded: NumericVersion = "530.30.02".parse().expect("valid version");
        let canonical: NumericVersion = "530.30.2".parse().expect("valid version");
        assert_eq!(padded, canonical, "zero padding must not affect comparison");
    }

    #[test]
    fn terminal_unsafe_characters_cover_controls_formats_and_non_ascii_separators() {
        for character in [
            '\n',
            '\u{001b}',
            '\u{00a0}',
            '\u{061c}',
            '\u{1680}',
            '\u{2007}',
            '\u{2028}',
            '\u{2029}',
            '\u{202e}',
            '\u{202f}',
            '\u{205f}',
            '\u{3000}',
            '\u{e0001}',
        ] {
            assert!(
                is_unsafe_terminal_character(character),
                "U+{:04X} should be unsafe",
                u32::from(character)
            );
        }
        for character in [' ', 'a', '日', '🙂'] {
            assert!(
                !is_unsafe_terminal_character(character),
                "U+{:04X} should be preserved",
                u32::from(character)
            );
        }
    }

    #[test]
    fn cuda_variants_round_trip() {
        for value in ["cpu", "cu92", "cu118", "cu124", "cu132"] {
            let parsed: CudaVariant = value.parse().expect("valid CUDA variant");
            assert_eq!(parsed.to_string(), value);
        }
    }

    #[test]
    fn invalid_cuda_variants_are_rejected() {
        for value in ["", "cuda124", "cu", "cu12x", "rocm6.3"] {
            assert!(value.parse::<CudaVariant>().is_err(), "accepted {value}");
        }
    }

    proptest! {
        #[test]
        fn numeric_version_order_matches_zero_padded_tuple_order(
            left in prop::collection::vec(any::<u16>(), 1..8),
            right in prop::collection::vec(any::<u16>(), 1..8),
        ) {
            let left = left.into_iter().map(u32::from).collect::<Vec<_>>();
            let right = right.into_iter().map(u32::from).collect::<Vec<_>>();
            let width = left.len().max(right.len());
            let expected = (0..width)
                .map(|index| {
                    left.get(index)
                        .copied()
                        .unwrap_or(0)
                        .cmp(&right.get(index).copied().unwrap_or(0))
                })
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal);
            let left = NumericVersion::new(left).expect("non-empty generated version");
            let right = NumericVersion::new(right).expect("non-empty generated version");

            prop_assert_eq!(left.cmp(&right), expected);
            prop_assert_eq!(left == right, expected == Ordering::Equal);
        }
    }
}
