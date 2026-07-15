//! Compatibility resolution, recommendation ranking, and install command generation.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::str::FromStr;

use pep440_rs::{Version, VersionSpecifiers};
use serde::Deserialize;

use crate::core::{
    Architecture, Candidate, CandidateWarning, CheckStatus, CommandSpec, CompatibilityCheck,
    CompatibilityChecks, CompatibilityStatus, ComputeCapability, CudaVariant, DecisionReason,
    DiagnosticSeverity, Environment, Installer, MetadataInfo, NumericVersion,
    NvidiaDetectionStatus, OperatingSystem, PythonInfo, ReasonCode, RecommendationReport,
    SCHEMA_VERSION, TorchWheel, WarningCode, is_unsafe_terminal_character,
};

const DRIVER_RULES_JSON: &str = include_str!("../data/cuda-driver-rules.json");
const RELEASE_RULES_JSON: &str = include_str!("../data/pytorch-release-rules.json");

/// Optional companion packages supported by generated install commands.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum CompanionPackage {
    /// `torchvision`.
    Torchvision,
    /// `torchaudio`.
    Torchaudio,
}

impl CompanionPackage {
    /// Normalized distribution name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Torchvision => "torchvision",
            Self::Torchaudio => "torchaudio",
        }
    }
}

impl FromStr for CompanionPackage {
    type Err = ResolverError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "torchvision" => Ok(Self::Torchvision),
            "torchaudio" => Ok(Self::Torchaudio),
            _ => Err(ResolverError::UnsupportedCompanion(value.to_owned())),
        }
    }
}

/// Settings that affect candidate filtering and command generation.
#[derive(Debug, Clone)]
pub struct ResolverOptions {
    /// Optional PEP 440 version specifier. A bare version means exact equality.
    pub torch_version: Option<String>,
    /// Include development/pre-release wheels in recommendation ranking.
    pub include_prerelease: bool,
    /// Installer command style.
    pub installer: Installer,
    /// Companion distributions to pin to the matching release.
    pub companions: BTreeSet<CompanionPackage>,
}

impl Default for ResolverOptions {
    fn default() -> Self {
        Self {
            torch_version: None,
            include_prerelease: false,
            installer: Installer::Pip,
            companions: BTreeSet::new(),
        }
    }
}

/// Resolver failure caused by malformed static data or user input.
#[derive(Debug, thiserror::Error)]
pub enum ResolverError {
    /// Detection produced one or more explicit blocking diagnostics.
    #[error("environment contains blocking diagnostics: {0}")]
    InvalidEnvironment(String),
    /// Embedded data did not satisfy its schema.
    #[error("invalid embedded compatibility data: {0}")]
    InvalidData(String),
    /// A version constraint was not valid PEP 440.
    #[error("invalid PyTorch version specifier `{value}`: {message}")]
    InvalidVersionSpecifier {
        /// Original user input.
        value: String,
        /// Parser diagnostic.
        message: String,
    },
    /// A wheel contained an invalid PEP 440 version after index normalization.
    #[error("invalid normalized wheel version `{0}")]
    InvalidWheelVersion(String),
    /// `--with` named a package outside the supported compatibility table.
    #[error("unsupported companion package: {0}")]
    UnsupportedCompanion(String),
    /// An install command could not be generated without a selected Python.
    #[error("a Python interpreter is required to generate an install command")]
    PythonRequired,
    /// A selected interpreter path cannot be represented in the JSON/command contract.
    #[error("the selected Python path is not valid UTF-8")]
    NonUtf8PythonPath,
    /// A selected interpreter path would make the displayed shell command unsafe or ambiguous.
    #[error(
        "the selected Python path contains a terminal-unsafe control, format, or separator character"
    )]
    UnsafePythonPath,
    /// Companion version mapping is unavailable for this PyTorch release.
    #[error("no verified {package} mapping for PyTorch {torch_version}")]
    CompanionMappingMissing {
        /// Requested companion distribution.
        package: &'static str,
        /// Selected PyTorch public version.
        torch_version: String,
    },
}

#[derive(Debug, Deserialize)]
struct DriverRules {
    schema_version: u32,
    families: Vec<DriverFamilyRule>,
    variants: Vec<DriverVariantRule>,
}

#[derive(Debug, Deserialize)]
struct DriverFamilyRule {
    cuda_major: u16,
    linux_min_driver: NumericVersion,
}

#[derive(Debug, Deserialize)]
struct DriverVariantRule {
    variant: CudaVariant,
    linux_min_driver: NumericVersion,
}

#[derive(Debug, Deserialize)]
struct ReleaseRules {
    schema_version: u32,
    releases: Vec<ReleaseRule>,
    gpu_architectures: Vec<GpuArchitectureRule>,
}

#[derive(Debug, Deserialize)]
struct ReleaseRule {
    series: String,
    preferred_variant: CudaVariant,
    variants: Vec<CudaVariant>,
    torchvision_minor: Option<u16>,
    torchaudio_series: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GpuArchitectureRule {
    series: String,
    versions: Vec<String>,
    variants: Vec<CudaVariant>,
    platform: String,
    cubin_capabilities: Vec<String>,
    ptx_capability: Option<String>,
}

struct RuleSet {
    drivers: DriverRules,
    releases: ReleaseRules,
}

impl RuleSet {
    fn load() -> Result<Self, ResolverError> {
        let drivers: DriverRules = serde_json::from_str(DRIVER_RULES_JSON)
            .map_err(|error| ResolverError::InvalidData(error.to_string()))?;
        let releases: ReleaseRules = serde_json::from_str(RELEASE_RULES_JSON)
            .map_err(|error| ResolverError::InvalidData(error.to_string()))?;
        if drivers.schema_version != 1 || releases.schema_version != 3 {
            return Err(ResolverError::InvalidData(
                "unsupported embedded schema version".to_owned(),
            ));
        }
        let driver_families = drivers
            .families
            .iter()
            .map(|rule| rule.cuda_major)
            .collect::<BTreeSet<_>>();
        let driver_variants = drivers
            .variants
            .iter()
            .map(|rule| &rule.variant)
            .collect::<BTreeSet<_>>();
        if driver_families.len() != drivers.families.len()
            || driver_variants.len() != drivers.variants.len()
        {
            return Err(ResolverError::InvalidData(
                "driver rules contain duplicate families or variants".to_owned(),
            ));
        }
        let release_series_keys = releases
            .releases
            .iter()
            .map(|release| release.series.as_str())
            .collect::<BTreeSet<_>>();
        if release_series_keys.len() != releases.releases.len() {
            return Err(ResolverError::InvalidData(
                "release rules contain duplicate series".to_owned(),
            ));
        }
        for release in &releases.releases {
            let variants = release.variants.iter().collect::<BTreeSet<_>>();
            if variants.len() != release.variants.len()
                || !variants.contains(&release.preferred_variant)
            {
                return Err(ResolverError::InvalidData(format!(
                    "release {} has duplicate variants or an invalid preferred variant",
                    release.series
                )));
            }
        }
        let mut architecture_keys = BTreeSet::new();
        let mut architecture_variant_keys = BTreeSet::new();
        for architecture in &releases.gpu_architectures {
            if architecture.platform != "linux_x86_64"
                || architecture.versions.is_empty()
                || architecture.variants.is_empty()
                || !architecture_keys.insert((
                    architecture.series.as_str(),
                    architecture.platform.as_str(),
                    architecture.versions.clone(),
                    architecture.variants.clone(),
                ))
            {
                return Err(ResolverError::InvalidData(format!(
                    "architecture rule {} has an unsupported platform, empty versions/variants, or duplicate key",
                    architecture.series
                )));
            }
            let parsed_versions = architecture
                .versions
                .iter()
                .map(|version| {
                    Version::from_str(version).map_err(|_| {
                        ResolverError::InvalidData(format!(
                            "architecture rule {} has invalid version {version}",
                            architecture.series
                        ))
                    })
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            if parsed_versions.len() != architecture.versions.len()
                || architecture.versions.iter().any(|version| {
                    release_series(version).as_deref() != Some(architecture.series.as_str())
                })
            {
                return Err(ResolverError::InvalidData(format!(
                    "architecture rule {} has duplicate versions or a version outside the series",
                    architecture.series
                )));
            }
            if architecture
                .variants
                .iter()
                .any(|variant| !variant.is_cuda())
            {
                return Err(ResolverError::InvalidData(format!(
                    "architecture rule {} references a non-CUDA variant",
                    architecture.series
                )));
            }
            for version in &parsed_versions {
                for variant in &architecture.variants {
                    if !architecture_variant_keys.insert((
                        version.clone(),
                        architecture.platform.as_str(),
                        variant,
                    )) {
                        return Err(ResolverError::InvalidData(format!(
                            "architecture rules overlap for PyTorch {version} {variant}"
                        )));
                    }
                }
            }
            let parsed = architecture
                .cubin_capabilities
                .iter()
                .map(|value| value.parse::<ComputeCapability>())
                .collect::<Result<BTreeSet<_>, _>>()
                .map_err(|error| ResolverError::InvalidData(error.to_string()))?;
            if parsed.is_empty() || parsed.len() != architecture.cubin_capabilities.len() {
                return Err(ResolverError::InvalidData(format!(
                    "architecture rule {} has empty or duplicate cubin capabilities",
                    architecture.series
                )));
            }
            if let Some(ptx) = &architecture.ptx_capability {
                let parsed_ptx = ptx
                    .parse::<ComputeCapability>()
                    .map_err(|error| ResolverError::InvalidData(error.to_string()))?;
                if !parsed.contains(&parsed_ptx) {
                    return Err(ResolverError::InvalidData(format!(
                        "architecture rule {} has PTX without a matching cubin capability",
                        architecture.series
                    )));
                }
            }
        }
        Ok(Self { drivers, releases })
    }

    fn family_driver_minimum(&self, major: u16) -> Option<&NumericVersion> {
        self.drivers
            .families
            .iter()
            .find(|rule| rule.cuda_major == major)
            .map(|rule| &rule.linux_min_driver)
    }

    fn variant_driver_minimum(&self, variant: &CudaVariant) -> Option<&NumericVersion> {
        self.drivers
            .variants
            .iter()
            .find(|rule| &rule.variant == variant)
            .map(|rule| &rule.linux_min_driver)
    }

    fn release_rule(&self, version: &str) -> Option<&ReleaseRule> {
        let series = release_series(version)?;
        self.releases
            .releases
            .iter()
            .find(|rule| rule.series == series)
    }

    fn official_preference(&self, version: &str, variant: &CudaVariant) -> Option<u16> {
        let rule = self.release_rule(version)?;
        if &rule.preferred_variant == variant {
            return Some(0);
        }
        rule.variants
            .iter()
            .filter(|known| *known != &rule.preferred_variant)
            .position(|known| known == variant)
            .and_then(|position| u16::try_from(position + 1).ok())
    }

    fn architecture_rule(
        &self,
        version: &str,
        variant: &CudaVariant,
    ) -> Option<&GpuArchitectureRule> {
        let requested = Version::from_str(version).ok()?;
        self.releases.gpu_architectures.iter().find(|rule| {
            rule.platform == "linux_x86_64"
                && rule
                    .versions
                    .iter()
                    .any(|known| Version::from_str(known).is_ok_and(|known| known == requested))
                && rule.variants.iter().any(|known| known == variant)
        })
    }
}

/// Resolves all candidates and constructs the recommendation report.
pub fn resolve(
    environment: &Environment,
    snapshot: &crate::core::IndexSnapshot,
    metadata: MetadataInfo,
    options: &ResolverOptions,
) -> Result<RecommendationReport, ResolverError> {
    validate_environment(environment)?;
    let rules = RuleSet::load()?;
    let version_specifier = parse_version_specifier(options.torch_version.as_deref())?;
    let mut groups: BTreeMap<(String, CudaVariant), Vec<&TorchWheel>> = BTreeMap::new();
    for wheel in snapshot
        .wheels
        .iter()
        .filter(|wheel| wheel.package == "torch")
    {
        groups
            .entry((wheel.public_version.clone(), wheel.variant.clone()))
            .or_default()
            .push(wheel);
    }

    let mut candidates = Vec::with_capacity(groups.len());
    for ((version, variant), mut wheels) in groups {
        let parsed_version = Version::from_str(&version)
            .map_err(|_| ResolverError::InvalidWheelVersion(version.clone()))?;
        if version_specifier
            .as_ref()
            .is_some_and(|specifier| !specifier.contains(&parsed_version))
        {
            continue;
        }
        wheels.sort_by(|left, right| left.filename.cmp(&right.filename));
        candidates.push(evaluate_group(
            environment,
            snapshot,
            &metadata,
            options,
            &rules,
            version_specifier.as_ref(),
            &version,
            &variant,
            &wheels,
        )?);
    }

    sort_candidates(&mut candidates, environment);
    let mut usable = Vec::new();
    let mut excluded = Vec::new();
    for candidate in candidates {
        match candidate.compatibility {
            CompatibilityStatus::Verified
            | CompatibilityStatus::DirectCompatible
            | CompatibilityStatus::MinorCompatible
            | CompatibilityStatus::Unverified => usable.push(candidate),
            CompatibilityStatus::Incompatible | CompatibilityStatus::Unavailable => {
                excluded.push(candidate);
            }
        }
    }

    let recommendation_position = usable
        .iter()
        .position(is_eligible_cuda_recommendation)
        .or_else(|| usable.iter().position(is_reviewed_direct_cpu));
    let recommendation = recommendation_position.map(|position| usable.remove(position));
    let alternatives = usable;
    let install = recommendation
        .as_ref()
        .map(|candidate| build_install_command(environment, snapshot, candidate, options, &rules))
        .transpose()?;

    Ok(RecommendationReport {
        schema_version: SCHEMA_VERSION,
        environment: environment.clone(),
        metadata,
        recommendation,
        alternatives,
        excluded,
        install,
    })
}

/// Explains one exact version/variant pair, even when no official wheel exists.
pub fn explain(
    environment: &Environment,
    snapshot: &crate::core::IndexSnapshot,
    metadata: MetadataInfo,
    torch_version: &str,
    variant: &CudaVariant,
    options: &ResolverOptions,
) -> Result<Candidate, ResolverError> {
    validate_environment(environment)?;
    let rules = RuleSet::load()?;
    let requested_version = Version::from_str(torch_version)
        .map_err(|_| ResolverError::InvalidWheelVersion(torch_version.to_owned()))?;
    let mut wheels: Vec<&TorchWheel> = snapshot
        .wheels
        .iter()
        .filter(|wheel| {
            wheel.package == "torch"
                && Version::from_str(&wheel.public_version)
                    .is_ok_and(|version| version == requested_version)
                && &wheel.variant == variant
        })
        .collect();
    wheels.sort_by(|left, right| left.filename.cmp(&right.filename));
    let canonical_version = requested_version.to_string();
    if wheels.is_empty() {
        return Ok(missing_candidate(&canonical_version, variant, &metadata));
    }
    let canonical_version = wheels[0].public_version.clone();
    evaluate_group(
        environment,
        snapshot,
        &metadata,
        options,
        &rules,
        None,
        &canonical_version,
        variant,
        &wheels,
    )
}

fn validate_environment(environment: &Environment) -> Result<(), ResolverError> {
    let mut codes = environment
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| format!("{:?}", diagnostic.code))
        .collect::<Vec<_>>();
    if codes.is_empty() {
        return Ok(());
    }
    codes.sort();
    codes.dedup();
    Err(ResolverError::InvalidEnvironment(codes.join(", ")))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_group(
    environment: &Environment,
    snapshot: &crate::core::IndexSnapshot,
    metadata: &MetadataInfo,
    options: &ResolverOptions,
    rules: &RuleSet,
    version_specifier: Option<&VersionSpecifiers>,
    version: &str,
    variant: &CudaVariant,
    wheels: &[&TorchWheel],
) -> Result<Candidate, ResolverError> {
    let parsed_version = Version::from_str(version)
        .map_err(|_| ResolverError::InvalidWheelVersion(version.to_owned()))?;
    let stable = !parsed_version.any_prerelease();

    let mut wheel_reasons = vec![
        DecisionReason::new(ReasonCode::WheelExists).with_detail("count", wheels.len().to_string()),
    ];
    let mut wheel_status = CheckStatus::Pass;
    if let Some(specifier) = version_specifier {
        if specifier.contains(&parsed_version) {
            wheel_reasons.push(DecisionReason::new(ReasonCode::VersionConstraintMatches));
        } else {
            wheel_status = CheckStatus::Fail;
            wheel_reasons.push(DecisionReason::new(ReasonCode::VersionConstraintMismatch));
        }
    }
    if !stable && !options.include_prerelease {
        wheel_status = CheckStatus::Fail;
        wheel_reasons.push(DecisionReason::new(ReasonCode::Prerelease));
    } else if stable {
        wheel_reasons.push(DecisionReason::new(ReasonCode::StableRelease));
    }

    let official_preference = rules.official_preference(version, variant);
    if let Some(preference) = official_preference {
        wheel_reasons.push(
            DecisionReason::new(ReasonCode::OfficialReleaseConfiguration)
                .with_detail("preference", preference.to_string()),
        );
    } else if rules.release_rule(version).is_some() {
        wheel_reasons.push(DecisionReason::new(
            ReasonCode::NotOfficialReleaseConfiguration,
        ));
    } else {
        wheel_reasons.push(DecisionReason::new(ReasonCode::OfficialPreferenceUnknown));
    }

    let (python_check, python_wheels) = evaluate_python(environment.python.as_ref(), wheels);
    let (platform_check, selected_wheel) =
        evaluate_platform(environment, environment.python.as_ref(), &python_wheels);

    if selected_wheel.is_some() {
        for companion in &options.companions {
            let Ok(companion_version) = companion_version(rules, *companion, version) else {
                wheel_status = CheckStatus::Fail;
                wheel_reasons.push(
                    DecisionReason::new(ReasonCode::WheelMissing)
                        .with_detail("package", companion.as_str())
                        .with_detail("version", "mapping_unavailable"),
                );
                continue;
            };
            let exists = matching_companion_wheel(
                snapshot,
                environment,
                companion.as_str(),
                &companion_version,
                variant,
            );
            if !exists {
                wheel_status = CheckStatus::Fail;
                wheel_reasons.push(
                    DecisionReason::new(ReasonCode::WheelMissing)
                        .with_detail("package", companion.as_str())
                        .with_detail("version", companion_version),
                );
            }
        }
    }

    let gpu_architecture = evaluate_gpu_architecture(environment, version, variant, rules);
    let driver = evaluate_driver(environment, variant, rules);
    let runtime = CompatibilityCheck::new(
        CheckStatus::Unknown,
        vec![DecisionReason::new(ReasonCode::RuntimeNotRun)],
    );
    let checks = CompatibilityChecks {
        wheel: CompatibilityCheck::new(wheel_status, wheel_reasons),
        python: python_check,
        platform: platform_check,
        gpu_architecture,
        driver,
        runtime,
    };
    let compatibility = aggregate_status(&checks);

    let mut warnings = Vec::new();
    if checks
        .driver
        .reasons
        .iter()
        .any(|reason| reason.code == ReasonCode::UsesCudaMinorCompatibility)
    {
        warnings.push(CandidateWarning::new(
            WarningCode::PtxOrNewDriverFeatureMayFail,
        ));
    }
    if checks.gpu_architecture.status == CheckStatus::Unknown {
        warnings.push(CandidateWarning::new(WarningCode::GpuArchitectureUnknown));
    }
    if metadata.stale || metadata.origin == crate::core::MetadataOrigin::StaleIfError {
        warnings.push(CandidateWarning::new(WarningCode::StaleMetadata));
    }
    if official_preference.is_none() && rules.release_rule(version).is_some() {
        warnings.push(CandidateWarning::new(
            WarningCode::NotOfficialReleaseConfiguration,
        ));
    } else if official_preference.is_none() {
        warnings.push(CandidateWarning::new(
            WarningCode::OfficialPreferenceUnknown,
        ));
    }
    if matches!(
        environment.nvidia.status,
        NvidiaDetectionStatus::CommandUnavailable
            | NvidiaDetectionStatus::Failed
            | NvidiaDetectionStatus::TimedOut
    ) && variant.is_cuda()
    {
        warnings.push(CandidateWarning::new(
            WarningCode::NvidiaDetectionIncomplete,
        ));
    }
    warnings.sort_by_key(|warning| warning.code);
    warnings.dedup_by_key(|warning| warning.code);

    Ok(Candidate {
        torch_version: version.to_owned(),
        variant: variant.clone(),
        compatibility,
        checks,
        wheel: selected_wheel.cloned(),
        stable,
        official_preference,
        warnings,
    })
}

fn evaluate_python<'a>(
    python: Option<&PythonInfo>,
    wheels: &[&'a TorchWheel],
) -> (CompatibilityCheck, Vec<&'a TorchWheel>) {
    let Some(python) = python else {
        return (
            CompatibilityCheck::new(
                CheckStatus::Unknown,
                vec![
                    DecisionReason::new(ReasonCode::PythonWheelMissing)
                        .with_detail("reason", "python_unavailable"),
                ],
            ),
            Vec::new(),
        );
    };

    let mut matches = Vec::new();
    let mut saw_yanked_match = false;
    for wheel in wheels {
        if !requires_python_matches(wheel.requires_python.as_deref(), python) {
            continue;
        }
        if python_abi_matches(wheel, python) {
            if wheel.yanked {
                saw_yanked_match = true;
            } else {
                matches.push(*wheel);
            }
        }
    }
    if matches.is_empty() {
        let mut reasons = vec![
            DecisionReason::new(ReasonCode::PythonWheelMissing)
                .with_detail("python", python.version.to_string())
                .with_detail(
                    "abi",
                    python
                        .cpython_abi_tag()
                        .unwrap_or_else(|| "unknown".to_owned()),
                ),
        ];
        if saw_yanked_match {
            reasons.push(DecisionReason::new(ReasonCode::WheelYanked));
        }
        return (
            CompatibilityCheck::new(CheckStatus::Fail, reasons),
            Vec::new(),
        );
    }
    (
        CompatibilityCheck::new(
            CheckStatus::Pass,
            vec![
                DecisionReason::new(ReasonCode::PythonTagMatches),
                DecisionReason::new(ReasonCode::AbiTagMatches),
            ],
        ),
        matches,
    )
}

fn python_abi_matches(wheel: &TorchWheel, python: &PythonInfo) -> bool {
    let compatible_pairs: HashSet<(&str, &str)> = python
        .compatible_tags
        .iter()
        .filter_map(|tag| {
            let mut parts = tag.splitn(3, '-');
            Some((parts.next()?, parts.next()?))
        })
        .collect();
    if !compatible_pairs.is_empty()
        && wheel.python_tags.iter().any(|python_tag| {
            wheel
                .abi_tags
                .iter()
                .any(|abi_tag| compatible_pairs.contains(&(python_tag.as_str(), abi_tag.as_str())))
        })
    {
        return true;
    }

    let Some(native_abi) = python.cpython_abi_tag() else {
        return false;
    };
    let native_python = native_abi.trim_end_matches('t');
    wheel.python_tags.iter().any(|python_tag| {
        wheel.abi_tags.iter().any(|abi_tag| {
            (python_tag == native_python && abi_tag == &native_abi)
                || (python_tag == "py3" && abi_tag == "none")
                || abi3_matches(python_tag, abi_tag, python)
        })
    })
}

fn abi3_matches(python_tag: &str, abi_tag: &str, python: &PythonInfo) -> bool {
    if abi_tag != "abi3" || python.free_threaded || !python_tag.starts_with("cp3") {
        return false;
    }
    python_tag
        .strip_prefix("cp3")
        .and_then(|minor| minor.parse::<u32>().ok())
        .is_some_and(|minor| minor <= python.version.component(1))
}

fn requires_python_matches(specifier: Option<&str>, python: &PythonInfo) -> bool {
    let Some(specifier) = specifier else {
        return true;
    };
    let Ok(specifiers) = VersionSpecifiers::from_str(specifier) else {
        return false;
    };
    let Ok(version) = Version::from_str(&python.version.to_string()) else {
        return false;
    };
    specifiers.contains(&version)
}

fn evaluate_platform<'a>(
    environment: &Environment,
    _python: Option<&PythonInfo>,
    wheels: &[&'a TorchWheel],
) -> (CompatibilityCheck, Option<&'a TorchWheel>) {
    if environment.platform.os != OperatingSystem::Linux
        || environment.platform.architecture != Architecture::X86_64
    {
        return (
            CompatibilityCheck::new(
                CheckStatus::Fail,
                vec![
                    DecisionReason::new(ReasonCode::PlatformMismatch)
                        .with_detail("os", environment.platform.os.to_string())
                        .with_detail(
                            "architecture",
                            environment.platform.architecture.to_string(),
                        ),
                ],
            ),
            None,
        );
    }
    if wheels.is_empty() {
        return (
            CompatibilityCheck::new(
                CheckStatus::Unknown,
                vec![
                    DecisionReason::new(ReasonCode::PlatformMismatch)
                        .with_detail("reason", "no_python_wheel"),
                ],
            ),
            None,
        );
    }
    let mut unknown = None;
    let mut oldest_required_glibc: Option<NumericVersion> = None;
    for wheel in wheels {
        match platform_wheel_status(wheel, environment) {
            PlatformMatch::Pass => {
                return (
                    CompatibilityCheck::new(
                        CheckStatus::Pass,
                        vec![DecisionReason::new(ReasonCode::PlatformTagMatches)],
                    ),
                    Some(*wheel),
                );
            }
            PlatformMatch::Unknown => unknown = unknown.or(Some(*wheel)),
            PlatformMatch::GlibcTooOld(required) => {
                if oldest_required_glibc
                    .as_ref()
                    .is_none_or(|current| &required < current)
                {
                    oldest_required_glibc = Some(required);
                }
            }
            PlatformMatch::Fail => {}
        }
    }
    if let Some(wheel) = unknown {
        return (
            CompatibilityCheck::new(
                CheckStatus::Unknown,
                vec![DecisionReason::new(ReasonCode::GlibcUnknown)],
            ),
            Some(wheel),
        );
    }
    let reason = if let Some(required) = oldest_required_glibc {
        DecisionReason::new(ReasonCode::GlibcTooOld)
            .with_detail("required", required.to_string())
            .with_detail(
                "detected",
                environment
                    .glibc
                    .as_ref()
                    .map_or_else(|| "unknown".to_owned(), ToString::to_string),
            )
    } else {
        DecisionReason::new(ReasonCode::PlatformMismatch)
    };
    (
        CompatibilityCheck::new(CheckStatus::Fail, vec![reason]),
        None,
    )
}

enum PlatformMatch {
    Pass,
    Unknown,
    GlibcTooOld(NumericVersion),
    Fail,
}

fn platform_wheel_status(wheel: &TorchWheel, environment: &Environment) -> PlatformMatch {
    if environment.platform.os != OperatingSystem::Linux
        || environment.platform.architecture != Architecture::X86_64
    {
        return PlatformMatch::Fail;
    }
    let mut saw_unknown = false;
    let mut too_old = None;
    for tag in &wheel.platform_tags {
        if tag == "any" {
            return PlatformMatch::Pass;
        }
        if tag == "linux_x86_64" {
            saw_unknown = true;
            continue;
        }
        let required = if tag == "manylinux1_x86_64" {
            "2.5".parse().ok()
        } else if tag == "manylinux2010_x86_64" {
            "2.12".parse().ok()
        } else if tag == "manylinux2014_x86_64" {
            "2.17".parse().ok()
        } else {
            parse_manylinux_floor(tag)
        };
        if let Some(required) = required {
            match &environment.glibc {
                Some(glibc) if glibc >= &required => return PlatformMatch::Pass,
                Some(_) => too_old = Some(required),
                None => saw_unknown = true,
            }
        }
    }
    if saw_unknown {
        PlatformMatch::Unknown
    } else if let Some(required) = too_old {
        PlatformMatch::GlibcTooOld(required)
    } else {
        PlatformMatch::Fail
    }
}

fn parse_manylinux_floor(tag: &str) -> Option<NumericVersion> {
    let rest = tag.strip_prefix("manylinux_")?;
    let rest = rest.strip_suffix("_x86_64")?;
    let (major, minor) = rest.split_once('_')?;
    format!("{major}.{minor}").parse().ok()
}

fn evaluate_gpu_architecture(
    environment: &Environment,
    version: &str,
    variant: &CudaVariant,
    rules: &RuleSet,
) -> CompatibilityCheck {
    if !variant.is_cuda() {
        return CompatibilityCheck::new(CheckStatus::NotApplicable, Vec::new());
    }
    if environment.nvidia.status == NvidiaDetectionStatus::NoDevices {
        return CompatibilityCheck::new(
            CheckStatus::Fail,
            vec![DecisionReason::new(ReasonCode::NvidiaGpuUnavailable)],
        );
    }
    if environment.nvidia.gpus.is_empty()
        || environment
            .nvidia
            .gpus
            .iter()
            .any(|gpu| gpu.compute_capability.is_none())
    {
        return CompatibilityCheck::new(
            CheckStatus::Unknown,
            vec![DecisionReason::new(ReasonCode::GpuArchitectureUnknown)],
        );
    }
    let Some(rule) = rules.architecture_rule(version, variant) else {
        return CompatibilityCheck::new(
            CheckStatus::Unknown,
            vec![DecisionReason::new(ReasonCode::GpuArchitectureUnknown)],
        );
    };
    let cubins = rule
        .cubin_capabilities
        .iter()
        .filter_map(|capability| capability.parse().ok())
        .collect::<Vec<ComputeCapability>>();
    let ptx = rule
        .ptx_capability
        .as_deref()
        .and_then(|capability| capability.parse::<ComputeCapability>().ok());
    let unsupported: Vec<String> = environment
        .nvidia
        .gpus
        .iter()
        .filter_map(|gpu| gpu.compute_capability)
        .filter(|capability| {
            let cubin_compatible = cubins
                .iter()
                .any(|cubin| cubin.major == capability.major && cubin.minor <= capability.minor);
            let ptx_compatible = ptx.is_some_and(|minimum| *capability >= minimum);
            !cubin_compatible && !ptx_compatible
        })
        .map(|capability| capability.to_string())
        .collect();
    if unsupported.is_empty() {
        CompatibilityCheck::new(
            CheckStatus::Pass,
            vec![DecisionReason::new(ReasonCode::GpuArchitectureSupported)],
        )
    } else {
        CompatibilityCheck::new(
            CheckStatus::Fail,
            vec![
                DecisionReason::new(ReasonCode::GpuArchitectureUnsupported)
                    .with_detail("capabilities", unsupported.join(",")),
            ],
        )
    }
}

fn evaluate_driver(
    environment: &Environment,
    variant: &CudaVariant,
    rules: &RuleSet,
) -> CompatibilityCheck {
    let Some((major, _minor)) = variant.cuda_version() else {
        return CompatibilityCheck::new(
            CheckStatus::NotApplicable,
            vec![DecisionReason::new(ReasonCode::DriverNotRequired)],
        );
    };
    if environment.nvidia.status == NvidiaDetectionStatus::NoDevices {
        return CompatibilityCheck::new(
            CheckStatus::Fail,
            vec![DecisionReason::new(ReasonCode::NvidiaGpuUnavailable)],
        );
    }
    let Some(driver) = environment.nvidia.driver_version.as_ref() else {
        return CompatibilityCheck::new(
            CheckStatus::Unknown,
            vec![DecisionReason::new(ReasonCode::DriverUnknown)],
        );
    };
    let family_minimum = rules.family_driver_minimum(major);
    let variant_minimum = rules.variant_driver_minimum(variant);
    if let Some(variant_minimum) = variant_minimum {
        if driver >= variant_minimum {
            let mut reasons = Vec::new();
            if let Some(family_minimum) = family_minimum {
                if driver >= family_minimum {
                    reasons.push(
                        DecisionReason::new(ReasonCode::DriverSupportsCudaFamily)
                            .with_detail("minimum", family_minimum.to_string()),
                    );
                }
            }
            reasons.push(
                DecisionReason::new(ReasonCode::DriverSupportsCudaVariant)
                    .with_detail("minimum", variant_minimum.to_string()),
            );
            return CompatibilityCheck::new(CheckStatus::Pass, reasons);
        }
    }
    let Some(family_minimum) = family_minimum else {
        return CompatibilityCheck::new(
            CheckStatus::Unknown,
            vec![
                DecisionReason::new(ReasonCode::DriverUnknown)
                    .with_detail("cuda_major", major.to_string()),
            ],
        );
    };
    if driver < family_minimum {
        return CompatibilityCheck::new(
            CheckStatus::Fail,
            vec![
                DecisionReason::new(ReasonCode::DriverTooOld)
                    .with_detail("detected", driver.to_string())
                    .with_detail("minimum", family_minimum.to_string())
                    .with_detail("cuda_major", major.to_string()),
            ],
        );
    }
    let family_reason = DecisionReason::new(ReasonCode::DriverSupportsCudaFamily)
        .with_detail("minimum", family_minimum.to_string());
    if let Some(variant_minimum) = variant_minimum {
        CompatibilityCheck::new(
            CheckStatus::Pass,
            vec![
                family_reason,
                DecisionReason::new(ReasonCode::UsesCudaMinorCompatibility)
                    .with_detail("normal_minimum", variant_minimum.to_string()),
            ],
        )
    } else {
        CompatibilityCheck::new(CheckStatus::Unknown, vec![family_reason])
    }
}

fn aggregate_status(checks: &CompatibilityChecks) -> CompatibilityStatus {
    if checks.runtime.status == CheckStatus::Pass {
        return CompatibilityStatus::Verified;
    }
    let wheel_unavailable = checks.wheel.reasons.iter().any(|reason| {
        matches!(
            reason.code,
            ReasonCode::WheelMissing | ReasonCode::WheelYanked
        )
    });
    if checks.python.status == CheckStatus::Fail || wheel_unavailable {
        return CompatibilityStatus::Unavailable;
    }
    let dimensions = [
        &checks.wheel,
        &checks.platform,
        &checks.gpu_architecture,
        &checks.driver,
    ];
    if dimensions
        .iter()
        .any(|check| check.status == CheckStatus::Fail)
    {
        return CompatibilityStatus::Incompatible;
    }
    if dimensions
        .iter()
        .any(|check| check.status == CheckStatus::Unknown)
    {
        return CompatibilityStatus::Unverified;
    }
    if checks
        .driver
        .reasons
        .iter()
        .any(|reason| reason.code == ReasonCode::UsesCudaMinorCompatibility)
    {
        CompatibilityStatus::MinorCompatible
    } else {
        CompatibilityStatus::DirectCompatible
    }
}

fn missing_candidate(version: &str, variant: &CudaVariant, metadata: &MetadataInfo) -> Candidate {
    let warnings = if metadata.stale || metadata.origin == crate::core::MetadataOrigin::StaleIfError
    {
        vec![CandidateWarning::new(WarningCode::StaleMetadata)]
    } else {
        Vec::new()
    };
    Candidate {
        torch_version: version.to_owned(),
        variant: variant.clone(),
        compatibility: CompatibilityStatus::Unavailable,
        checks: CompatibilityChecks {
            wheel: CompatibilityCheck::new(
                CheckStatus::Fail,
                vec![DecisionReason::new(ReasonCode::WheelMissing)],
            ),
            python: CompatibilityCheck::new(CheckStatus::Unknown, Vec::new()),
            platform: CompatibilityCheck::new(CheckStatus::Unknown, Vec::new()),
            gpu_architecture: CompatibilityCheck::new(CheckStatus::Unknown, Vec::new()),
            driver: CompatibilityCheck::new(CheckStatus::Unknown, Vec::new()),
            runtime: CompatibilityCheck::new(
                CheckStatus::Unknown,
                vec![DecisionReason::new(ReasonCode::RuntimeNotRun)],
            ),
        },
        wheel: None,
        stable: is_stable_version(version),
        official_preference: None,
        warnings,
    }
}

fn sort_candidates(candidates: &mut [Candidate], environment: &Environment) {
    let nvidia_detected = environment.nvidia.status == NvidiaDetectionStatus::Detected
        && !environment.nvidia.gpus.is_empty()
        && environment.nvidia.driver_version.is_some();
    candidates.sort_by(|left, right| {
        right
            .stable
            .cmp(&left.stable)
            .then_with(|| {
                if nvidia_detected {
                    Ordering::Equal
                } else {
                    accelerator_rank(right, false).cmp(&accelerator_rank(left, false))
                }
            })
            .then_with(|| compare_versions_desc(&left.torch_version, &right.torch_version))
            .then_with(|| {
                if nvidia_detected {
                    accelerator_rank(right, true).cmp(&accelerator_rank(left, true))
                } else {
                    Ordering::Equal
                }
            })
            .then_with(|| compatibility_rank(right).cmp(&compatibility_rank(left)))
            .then_with(|| {
                left.official_preference
                    .unwrap_or(u16::MAX)
                    .cmp(&right.official_preference.unwrap_or(u16::MAX))
            })
            .then_with(|| left.variant.cmp(&right.variant))
    });
}

fn is_eligible_cuda_recommendation(candidate: &Candidate) -> bool {
    candidate.variant.is_cuda()
        && candidate.official_preference.is_some()
        && matches!(
            candidate.compatibility,
            CompatibilityStatus::Verified
                | CompatibilityStatus::DirectCompatible
                | CompatibilityStatus::MinorCompatible
        )
}

fn is_reviewed_direct_cpu(candidate: &Candidate) -> bool {
    !candidate.variant.is_cuda()
        && candidate.official_preference.is_some()
        && matches!(
            candidate.compatibility,
            CompatibilityStatus::Verified | CompatibilityStatus::DirectCompatible
        )
}

fn compare_versions_desc(left: &str, right: &str) -> Ordering {
    match (Version::from_str(left), Version::from_str(right)) {
        (Ok(left), Ok(right)) => right.cmp(&left),
        _ => right.cmp(left),
    }
}

fn accelerator_rank(candidate: &Candidate, nvidia_detected: bool) -> u8 {
    u8::from(candidate.variant.is_cuda() == nvidia_detected)
}

fn compatibility_rank(candidate: &Candidate) -> u8 {
    match candidate.compatibility {
        CompatibilityStatus::Verified => 5,
        CompatibilityStatus::DirectCompatible => 4,
        CompatibilityStatus::MinorCompatible => 3,
        CompatibilityStatus::Unverified => 2,
        CompatibilityStatus::Incompatible => 1,
        CompatibilityStatus::Unavailable => 0,
    }
}

fn parse_version_specifier(
    value: Option<&str>,
) -> Result<Option<VersionSpecifiers>, ResolverError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim().strip_prefix("torch").unwrap_or(value.trim());
    let normalized = if value.starts_with(['<', '>', '=', '!', '~']) {
        value.to_owned()
    } else {
        format!("=={value}")
    };
    VersionSpecifiers::from_str(&normalized)
        .map(Some)
        .map_err(|error| ResolverError::InvalidVersionSpecifier {
            value: value.to_owned(),
            message: error.to_string(),
        })
}

fn is_stable_version(version: &str) -> bool {
    Version::from_str(version).is_ok_and(|version| !version.any_prerelease())
}

fn release_series(version: &str) -> Option<String> {
    let public = version.split('+').next().unwrap_or(version);
    let mut release = public.split('.');
    Some(format!("{}.{}", release.next()?, release.next()?))
}

fn companion_version(
    rules: &RuleSet,
    package: CompanionPackage,
    torch_version: &str,
) -> Result<String, ResolverError> {
    let parsed = release_components(torch_version).ok_or_else(|| {
        ResolverError::CompanionMappingMissing {
            package: package.as_str(),
            torch_version: torch_version.to_owned(),
        }
    })?;
    if parsed.0 == 2 && parsed.1 == 0 {
        let special = match (package, parsed.2) {
            (CompanionPackage::Torchvision, 0) => Some("0.15.1"),
            (CompanionPackage::Torchvision, 1) => Some("0.15.2"),
            (CompanionPackage::Torchaudio, 0) => Some("2.0.1"),
            (CompanionPackage::Torchaudio, 1) => Some("2.0.2"),
            _ => None,
        };
        return special
            .map(str::to_owned)
            .ok_or_else(|| ResolverError::CompanionMappingMissing {
                package: package.as_str(),
                torch_version: torch_version.to_owned(),
            });
    }
    let Some(rule) = rules.release_rule(torch_version) else {
        return Err(ResolverError::CompanionMappingMissing {
            package: package.as_str(),
            torch_version: torch_version.to_owned(),
        });
    };
    match package {
        CompanionPackage::Torchvision => rule
            .torchvision_minor
            .map(|minor| format!("0.{minor}.{}", parsed.2)),
        CompanionPackage::Torchaudio => rule
            .torchaudio_series
            .as_ref()
            .map(|series| format!("{series}.{}", parsed.2)),
    }
    .ok_or_else(|| ResolverError::CompanionMappingMissing {
        package: package.as_str(),
        torch_version: torch_version.to_owned(),
    })
}

fn release_components(version: &str) -> Option<(u16, u16, u16)> {
    let public = version.split('+').next()?;
    let mut components = public.split('.');
    Some((
        components.next()?.parse().ok()?,
        components.next()?.parse().ok()?,
        components.next()?.parse().ok()?,
    ))
}

fn matching_companion_wheel(
    snapshot: &crate::core::IndexSnapshot,
    environment: &Environment,
    package: &str,
    version: &str,
    variant: &CudaVariant,
) -> bool {
    let group: Vec<&TorchWheel> = snapshot
        .wheels
        .iter()
        .filter(|wheel| {
            wheel.package == package
                && wheel.public_version == version
                && &wheel.variant == variant
                && !wheel.yanked
        })
        .collect();
    let (_, python_matches) = evaluate_python(environment.python.as_ref(), &group);
    let (platform, wheel) =
        evaluate_platform(environment, environment.python.as_ref(), &python_matches);
    wheel.is_some() && platform.status == CheckStatus::Pass
}

fn build_install_command(
    environment: &Environment,
    snapshot: &crate::core::IndexSnapshot,
    candidate: &Candidate,
    options: &ResolverOptions,
    rules: &RuleSet,
) -> Result<CommandSpec, ResolverError> {
    let python = environment
        .python
        .as_ref()
        .ok_or(ResolverError::PythonRequired)?;
    let python_executable = path_to_string(&python.executable)?;
    let index_url = format!("https://download.pytorch.org/whl/{}", candidate.variant);
    let mut requirements = vec![format!("torch=={}", candidate.torch_version)];
    for companion in &options.companions {
        requirements.push(format!(
            "{}=={}",
            companion.as_str(),
            companion_version(rules, *companion, &candidate.torch_version)?
        ));
    }
    // Re-check here so command construction never emits an unchecked companion pin.
    for requirement in requirements.iter().skip(1) {
        let (package, version) = requirement.split_once("==").ok_or_else(|| {
            ResolverError::InvalidData("invalid generated companion requirement".to_owned())
        })?;
        if !matching_companion_wheel(snapshot, environment, package, version, &candidate.variant) {
            return Err(ResolverError::CompanionMappingMissing {
                package: if package == "torchvision" {
                    "torchvision"
                } else {
                    "torchaudio"
                },
                torch_version: candidate.torch_version.clone(),
            });
        }
    }

    let (program, mut args) = match options.installer {
        Installer::Pip => (
            python_executable.clone(),
            vec![
                "-m".to_owned(),
                "pip".to_owned(),
                "install".to_owned(),
                "--isolated".to_owned(),
            ],
        ),
        Installer::Uv => (
            "uv".to_owned(),
            vec![
                "pip".to_owned(),
                "install".to_owned(),
                "--python".to_owned(),
                python_executable.clone(),
                "--default-index".to_owned(),
                index_url.clone(),
            ],
        ),
        Installer::UvAdd => (
            "uv".to_owned(),
            vec![
                "add".to_owned(),
                "--python".to_owned(),
                python_executable,
                "--index".to_owned(),
                format!("pytorch={index_url}"),
            ],
        ),
    };
    if options.installer == Installer::Pip {
        args.push("--index-url".to_owned());
        args.push(index_url);
    }
    args.extend(requirements);
    let display = shell_display(&program, &args);
    Ok(CommandSpec {
        program,
        args,
        display,
    })
}

fn path_to_string(path: &Path) -> Result<String, ResolverError> {
    let value = path.to_str().ok_or(ResolverError::NonUtf8PythonPath)?;
    if value.chars().any(is_unsafe_terminal_character) {
        return Err(ResolverError::UnsafePythonPath);
    }
    Ok(value.to_owned())
}

fn shell_display(program: &str, args: &[String]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(|part| shell_escape::escape(part.into()).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::core::{
        CudaToolkitInfo, Diagnostic, DiagnosticCode, MetadataOrigin, NvidiaGpu, NvidiaInfo,
        PlatformInfo, PythonInfo, TagSource, ToolkitSource,
    };

    fn environment(driver: &str, reported: &str) -> Environment {
        Environment {
            platform: PlatformInfo {
                os: OperatingSystem::Linux,
                architecture: Architecture::X86_64,
                kernel_version: Some("6.8.0".to_owned()),
                distribution: Some("Ubuntu 22.04".to_owned()),
            },
            glibc: Some("2.35".parse().expect("valid glibc")),
            python: Some(PythonInfo {
                executable: PathBuf::from("/usr/bin/python3"),
                implementation: "cpython".to_owned(),
                version: "3.13.0".parse().expect("valid Python"),
                soabi: Some("cpython-313-x86_64-linux-gnu".to_owned()),
                cache_tag: Some("cpython-313".to_owned()),
                platform: "linux-x86_64".to_owned(),
                pointer_width: 64,
                free_threaded: false,
                virtual_environment: None,
                compatible_tags: vec![
                    "cp313-cp313-manylinux_2_28_x86_64".to_owned(),
                    "cp313-cp313-manylinux_2_17_x86_64".to_owned(),
                ],
                tag_source: TagSource::Packaging,
            }),
            nvidia: NvidiaInfo {
                status: NvidiaDetectionStatus::Detected,
                driver_version: Some(driver.parse().expect("valid driver")),
                reported_cuda_version: Some(reported.parse().expect("valid reported CUDA")),
                gpus: vec![NvidiaGpu {
                    index: 0,
                    uuid: Some("GPU-test".to_owned()),
                    name: "NVIDIA A100".to_owned(),
                    compute_capability: Some(ComputeCapability { major: 8, minor: 0 }),
                    driver_version: driver.parse().expect("valid driver"),
                }],
            },
            cuda_toolkit: None,
            diagnostics: Vec::new(),
        }
    }

    fn wheel(version: &str, variant: &str) -> TorchWheel {
        TorchWheel {
            package: "torch".to_owned(),
            filename: format!("torch-{version}+{variant}-cp313-cp313-manylinux_2_28_x86_64.whl"),
            version: format!("{version}+{variant}"),
            public_version: version.to_owned(),
            variant: variant.parse().expect("valid variant"),
            python_tags: vec!["cp313".to_owned()],
            abi_tags: vec!["cp313".to_owned()],
            platform_tags: vec!["manylinux_2_28_x86_64".to_owned()],
            url: format!("https://download-r2.pytorch.org/whl/{variant}/test.whl"),
            sha256: Some("00".repeat(32)),
            yanked: false,
            requires_python: Some(">=3.9".to_owned()),
        }
    }

    fn metadata() -> MetadataInfo {
        MetadataInfo {
            origin: MetadataOrigin::Network,
            fetched_at: 1,
            age_seconds: 0,
            stale: false,
            source: "https://download.pytorch.org/whl/".to_owned(),
        }
    }

    fn snapshot(wheels: Vec<TorchWheel>) -> crate::core::IndexSnapshot {
        crate::core::IndexSnapshot {
            schema_version: 1,
            fetched_at: 1,
            source: "https://download.pytorch.org/whl/".to_owned(),
            packages: vec!["torch".to_owned()],
            variants: vec!["cu121".parse().expect("valid variant")],
            wheels,
        }
    }

    #[test]
    fn case_a_normal_minimum_is_direct() {
        let env = environment("530.30.02", "12.1");
        let snapshot = snapshot(vec![wheel("2.6.0", "cu121")]);
        let report = resolve(&env, &snapshot, metadata(), &ResolverOptions::default())
            .expect("resolution succeeds");
        assert!(report.recommendation.is_none());
        assert_eq!(
            report.alternatives[0].compatibility,
            CompatibilityStatus::DirectCompatible,
            "tag-pinned 2.6 architecture evidence should complete the static checks"
        );
        let driver = evaluate_driver(
            &env,
            &"cu121".parse().expect("variant"),
            &RuleSet::load().expect("rules"),
        );
        assert!(
            driver
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::DriverSupportsCudaVariant })
        );
    }

    #[test]
    fn case_b_reported_cuda_is_not_a_ceiling() {
        let env = environment("530.30.02", "12.1");
        let driver = evaluate_driver(
            &env,
            &"cu124".parse().expect("variant"),
            &RuleSet::load().expect("rules"),
        );
        assert_eq!(driver.status, CheckStatus::Pass);
        assert!(
            driver
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::UsesCudaMinorCompatibility })
        );
    }

    #[test]
    fn case_c_cuda_13_requires_newer_driver_family() {
        let env = environment("530.30.02", "12.1");
        let driver = evaluate_driver(
            &env,
            &"cu130".parse().expect("variant"),
            &RuleSet::load().expect("rules"),
        );
        assert_eq!(driver.status, CheckStatus::Fail);
        assert_eq!(driver.reasons[0].code, ReasonCode::DriverTooOld);
    }

    #[test]
    fn local_toolkit_does_not_change_driver_compatibility() {
        let without = environment("530.30.02", "12.1");
        let mut with = without.clone();
        with.cuda_toolkit = Some(CudaToolkitInfo {
            version: "12.8".parse().expect("toolkit"),
            root: Some(PathBuf::from("/usr/local/cuda")),
            source: ToolkitSource::VersionJson,
        });
        let rules = RuleSet::load().expect("rules");
        let variant = "cu124".parse().expect("variant");
        assert_eq!(
            evaluate_driver(&without, &variant, &rules),
            evaluate_driver(&with, &variant, &rules)
        );
    }

    #[test]
    fn cp313_environment_marks_cp312_only_wheel_unavailable() {
        let env = environment("530.30.02", "12.1");
        let mut cp312 = wheel("2.6.0", "cu121");
        cp312.filename = "torch-2.6.0+cu121-cp312-cp312-manylinux_2_28_x86_64.whl".to_owned();
        cp312.python_tags = vec!["cp312".to_owned()];
        cp312.abi_tags = vec!["cp312".to_owned()];
        let report = resolve(
            &env,
            &snapshot(vec![cp312]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution succeeds");

        assert!(report.recommendation.is_none());
        assert_eq!(
            report.excluded[0].compatibility,
            CompatibilityStatus::Unavailable
        );
        assert_eq!(report.excluded[0].checks.python.status, CheckStatus::Fail);
    }

    #[test]
    fn resolver_rejects_blocking_detection_diagnostics() {
        let mut env = environment("530.30.02", "12.1");
        env.diagnostics.push(Diagnostic {
            code: DiagnosticCode::UnsupportedLibc,
            severity: DiagnosticSeverity::Error,
            details: BTreeMap::from([("libc".to_owned(), "musl".to_owned())]),
        });

        assert!(matches!(
            resolve(
                &env,
                &snapshot(vec![wheel("2.6.0", "cpu")]),
                metadata(),
                &ResolverOptions::default(),
            ),
            Err(ResolverError::InvalidEnvironment(_))
        ));
    }

    #[test]
    fn reported_cuda_does_not_change_driver_compatibility() {
        let first = environment("530.30.02", "12.1");
        let second = environment("530.30.02", "12.9");
        let rules = RuleSet::load().expect("rules");
        let variant = "cu124".parse().expect("variant");
        assert_eq!(
            evaluate_driver(&first, &variant, &rules),
            evaluate_driver(&second, &variant, &rules)
        );
    }

    #[test]
    fn manylinux_floor_is_numeric() {
        let mut env = environment("580.65.06", "13.0");
        env.glibc = Some("2.27".parse().expect("glibc"));
        assert!(matches!(
            platform_wheel_status(&wheel("2.13.0", "cu130"), &env),
            PlatformMatch::GlibcTooOld(_)
        ));
        env.glibc = Some("2.28".parse().expect("glibc"));
        assert!(matches!(
            platform_wheel_status(&wheel("2.13.0", "cu130"), &env),
            PlatformMatch::Pass
        ));
    }

    #[test]
    fn generated_pip_command_targets_selected_python_and_is_shell_quoted() {
        let mut env = environment("580.65.06", "13.0");
        env.python.as_mut().expect("python").executable =
            PathBuf::from("/tmp/Python Env/bin/python");
        let candidate = Candidate {
            torch_version: "2.13.0".to_owned(),
            variant: "cu130".parse().expect("variant"),
            compatibility: CompatibilityStatus::DirectCompatible,
            checks: CompatibilityChecks {
                wheel: CompatibilityCheck::new(CheckStatus::Pass, Vec::new()),
                python: CompatibilityCheck::new(CheckStatus::Pass, Vec::new()),
                platform: CompatibilityCheck::new(CheckStatus::Pass, Vec::new()),
                gpu_architecture: CompatibilityCheck::new(CheckStatus::Pass, Vec::new()),
                driver: CompatibilityCheck::new(CheckStatus::Pass, Vec::new()),
                runtime: CompatibilityCheck::new(CheckStatus::Unknown, Vec::new()),
            },
            wheel: None,
            stable: true,
            official_preference: Some(1),
            warnings: Vec::new(),
        };
        let command = build_install_command(
            &env,
            &snapshot(vec![]),
            &candidate,
            &ResolverOptions::default(),
            &RuleSet::load().expect("rules"),
        )
        .expect("command");
        assert_eq!(command.program, "/tmp/Python Env/bin/python");
        assert!(command.display.starts_with("'/tmp/Python Env/bin/python'"));
        assert!(command.args.contains(&"--index-url".to_owned()));
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_python_paths_never_become_lossy_install_commands() {
        use std::os::unix::ffi::OsStringExt;

        let path = PathBuf::from(std::ffi::OsString::from_vec(vec![b'/', b'x', 0xff]));
        assert!(matches!(
            path_to_string(&path),
            Err(ResolverError::NonUtf8PythonPath)
        ));
    }

    #[test]
    fn terminal_control_characters_in_python_paths_are_rejected() {
        for path in [
            "/tmp/python\nshim",
            "/tmp/python\u{00a0}shim",
            "/tmp/python\u{061c}shim",
            "/tmp/python\u{2028}shim",
            "/tmp/python\u{3000}shim",
        ] {
            assert!(matches!(
                path_to_string(Path::new(path)),
                Err(ResolverError::UnsafePythonPath)
            ));
        }
        for path in [
            "/tmp/Python Env/bin/python",
            "/tmp/python'$;shim",
            "/tmp/日本語/python",
        ] {
            assert_eq!(path_to_string(Path::new(path)).expect("safe path"), path);
        }
    }

    #[test]
    fn pep440_alpha_beta_rc_and_dev_releases_are_not_stable() {
        for version in ["2.14.0a1", "2.14.0b1", "2.14.0rc1", "2.14.0.dev1"] {
            assert!(
                !is_stable_version(version),
                "{version} must be a prerelease"
            );
        }
        assert!(is_stable_version("2.14.0"));
    }

    #[test]
    fn cuda_11_0_exact_driver_exception_precedes_family_minimum() {
        let env = environment("450.36.06", "11.0");
        let driver = evaluate_driver(
            &env,
            &"cu110".parse().expect("variant"),
            &RuleSet::load().expect("rules"),
        );
        assert_eq!(driver.status, CheckStatus::Pass);
        assert!(
            driver
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::DriverSupportsCudaVariant })
        );
        assert!(
            !driver
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::UsesCudaMinorCompatibility })
        );
    }

    #[test]
    fn gpu_architecture_accepts_same_major_cubin_and_forward_ptx() {
        let rules = RuleSet::load().expect("rules");
        let variant = "cu130".parse().expect("variant");
        let mut env = environment("580.65.06", "13.0");
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability { major: 8, minor: 9 });
        assert_eq!(
            evaluate_gpu_architecture(&env, "2.13.0", &variant, &rules).status,
            CheckStatus::Pass,
            "sm_86 cubin is compatible with a compute capability 8.9 device"
        );

        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability {
            major: 12,
            minor: 1,
        });
        assert_eq!(
            evaluate_gpu_architecture(&env, "2.13.0", &variant, &rules).status,
            CheckStatus::Pass,
            "compute_120 PTX is forward-compatible with 12.1"
        );
    }

    #[test]
    fn architecture_evidence_does_not_leak_to_an_unreviewed_patch_release() {
        let rules = RuleSet::load().expect("rules");
        let variant = "cu128".parse().expect("variant");
        let mut env = environment("575.57.08", "12.9");
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability {
            major: 12,
            minor: 0,
        });

        assert_eq!(
            evaluate_gpu_architecture(&env, "2.10.0", &variant, &rules).status,
            CheckStatus::Pass
        );
        assert_eq!(
            evaluate_gpu_architecture(&env, "2.10.1", &variant, &rules).status,
            CheckStatus::Unknown,
            "a future patch must not inherit v2.10.0 build evidence"
        );
    }

    #[test]
    fn official_default_cuda_variant_has_first_preference() {
        let rules = RuleSet::load().expect("rules");
        assert_eq!(
            rules.official_preference("2.13.0", &"cu130".parse().expect("variant")),
            Some(0)
        );
        assert!(rules.official_preference("2.13.0", &"cu126".parse().expect("variant")) > Some(0));
    }

    #[test]
    fn bacchus_recommends_reviewed_minor_compatible_cuda() {
        let mut env = environment("530.30.02", "12.1");
        env.nvidia.gpus[0].name = "NVIDIA RTX A6000".to_owned();
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability { major: 8, minor: 6 });
        let report = resolve(
            &env,
            &snapshot(vec![
                wheel("2.13.0", "cu126"),
                wheel("2.13.0", "cu129"),
                wheel("2.13.0", "cpu"),
            ]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution succeeds");

        let recommendation = report.recommendation.expect("reviewed CUDA candidate");
        assert_eq!(recommendation.variant, "cu126".parse().expect("variant"));
        assert_eq!(
            recommendation.compatibility,
            CompatibilityStatus::MinorCompatible
        );
        assert!(
            recommendation
                .warnings
                .iter()
                .any(|warning| { warning.code == WarningCode::PtxOrNewDriverFeatureMayFail })
        );

        let index_only = report
            .alternatives
            .iter()
            .find(|candidate| candidate.variant == "cu129".parse().expect("variant"))
            .expect("index-only CUDA candidate remains visible");
        assert!(
            index_only
                .warnings
                .iter()
                .any(|warning| { warning.code == WarningCode::NotOfficialReleaseConfiguration })
        );
        assert!(
            index_only
                .checks
                .wheel
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::NotOfficialReleaseConfiguration })
        );
    }

    #[test]
    fn hestia_recommends_newest_reviewed_cuda_with_sm120_support() {
        let mut env = environment("575.57.08", "12.9");
        env.nvidia.gpus[0].name =
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".to_owned();
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability {
            major: 12,
            minor: 0,
        });
        let report = resolve(
            &env,
            &snapshot(vec![
                wheel("2.13.0", "cu126"),
                wheel("2.13.0", "cu129"),
                wheel("2.13.0", "cu130"),
                wheel("2.13.0", "cpu"),
                wheel("2.11.0", "cu126"),
                wheel("2.11.0", "cu128"),
                wheel("2.11.0", "cpu"),
                wheel("2.10.0", "cu126"),
                wheel("2.10.0", "cu128"),
                wheel("2.10.0", "cpu"),
            ]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution succeeds");

        let recommendation = report.recommendation.expect("reviewed CUDA candidate");
        assert_eq!(recommendation.torch_version, "2.11.0");
        assert_eq!(recommendation.variant, "cu128".parse().expect("variant"));
        assert_eq!(
            recommendation.compatibility,
            CompatibilityStatus::DirectCompatible
        );
        assert!(
            report
                .install
                .as_ref()
                .expect("CUDA install command")
                .args
                .iter()
                .any(|argument| argument == "https://download.pytorch.org/whl/cu128")
        );

        let older_cuda = report
            .alternatives
            .iter()
            .find(|candidate| {
                candidate.torch_version == "2.10.0"
                    && candidate.variant == "cu128".parse().expect("variant")
            })
            .expect("older statically compatible CUDA remains an alternative");
        assert_eq!(
            older_cuda.compatibility,
            CompatibilityStatus::DirectCompatible
        );

        let cpu_fallback = report
            .alternatives
            .iter()
            .find(|candidate| {
                candidate.torch_version == "2.13.0" && candidate.variant == CudaVariant::Cpu
            })
            .expect("newer CPU build remains available as a fallback");
        assert_eq!(
            cpu_fallback.compatibility,
            CompatibilityStatus::DirectCompatible
        );

        let index_only = report
            .alternatives
            .iter()
            .find(|candidate| {
                candidate.torch_version == "2.13.0"
                    && candidate.variant == "cu129".parse().expect("variant")
            })
            .expect("cu129 remains an alternative");
        assert_eq!(index_only.compatibility, CompatibilityStatus::Unverified);
        assert!(
            index_only
                .warnings
                .iter()
                .any(|warning| { warning.code == WarningCode::NotOfficialReleaseConfiguration })
        );
        assert!(
            index_only
                .checks
                .wheel
                .reasons
                .iter()
                .any(|reason| { reason.code == ReasonCode::NotOfficialReleaseConfiguration })
        );

        for version in ["2.11.0", "2.10.0"] {
            let unsupported = report
                .excluded
                .iter()
                .find(|candidate| {
                    candidate.torch_version == version
                        && candidate.variant == "cu126".parse().expect("variant")
                })
                .expect("CUDA 12.6 does not contain sm_120");
            assert_eq!(unsupported.compatibility, CompatibilityStatus::Incompatible);
            assert!(
                unsupported
                    .checks
                    .gpu_architecture
                    .reasons
                    .iter()
                    .any(|reason| reason.code == ReasonCode::GpuArchitectureUnsupported)
            );
        }
    }

    #[test]
    fn tagged_2_9_patch_evidence_covers_reviewed_and_index_only_sm120_wheels() {
        let mut env = environment("575.57.08", "12.9");
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability {
            major: 12,
            minor: 0,
        });
        let report = resolve(
            &env,
            &snapshot(vec![
                wheel("2.9.1", "cu126"),
                wheel("2.9.1", "cu128"),
                wheel("2.9.1", "cu129"),
            ]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution succeeds");

        let recommendation = report.recommendation.expect("reviewed CUDA candidate");
        assert_eq!(recommendation.torch_version, "2.9.1");
        assert_eq!(recommendation.variant, "cu128".parse().expect("variant"));
        assert_eq!(
            recommendation.compatibility,
            CompatibilityStatus::DirectCompatible
        );

        let index_only = report
            .alternatives
            .iter()
            .find(|candidate| candidate.variant == "cu129".parse().expect("variant"))
            .expect("index-only build remains inspectable");
        assert_eq!(
            index_only.compatibility,
            CompatibilityStatus::DirectCompatible
        );
        assert!(index_only.official_preference.is_none());
        assert!(
            index_only
                .warnings
                .iter()
                .any(|warning| warning.code == WarningCode::NotOfficialReleaseConfiguration)
        );

        let unsupported = report
            .excluded
            .iter()
            .find(|candidate| candidate.variant == "cu126".parse().expect("variant"))
            .expect("cu126 lacks sm_120");
        assert_eq!(
            unsupported.checks.gpu_architecture.status,
            CheckStatus::Fail
        );
    }

    #[test]
    fn hestia_uses_cpu_fallback_when_no_reviewed_sm120_cuda_wheel_is_available() {
        let mut env = environment("575.57.08", "12.9");
        env.nvidia.gpus[0].compute_capability = Some(ComputeCapability {
            major: 12,
            minor: 0,
        });
        let report = resolve(
            &env,
            &snapshot(vec![
                wheel("2.13.0", "cu126"),
                wheel("2.13.0", "cu129"),
                wheel("2.13.0", "cu130"),
                wheel("2.13.0", "cpu"),
                wheel("2.11.0", "cu126"),
            ]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution succeeds");

        let recommendation = report.recommendation.expect("reviewed CPU fallback");
        assert_eq!(recommendation.torch_version, "2.13.0");
        assert_eq!(recommendation.variant, CudaVariant::Cpu);
        assert_eq!(
            recommendation.compatibility,
            CompatibilityStatus::DirectCompatible
        );
        assert!(
            report
                .install
                .as_ref()
                .expect("CPU install command")
                .args
                .iter()
                .any(|argument| argument == "https://download.pytorch.org/whl/cpu")
        );
    }

    #[test]
    fn version_constraint_is_a_hard_candidate_filter() {
        let env = environment("580.65.06", "13.0");
        let options = ResolverOptions {
            torch_version: Some("2.10".to_owned()),
            ..ResolverOptions::default()
        };
        let report = resolve(
            &env,
            &snapshot(vec![
                wheel("2.13.0", "cu130"),
                wheel("2.13.0", "cpu"),
                wheel("2.10.0", "cu128"),
                wheel("2.10.0", "cpu"),
            ]),
            metadata(),
            &options,
        )
        .expect("resolution succeeds");

        let candidates = report
            .recommendation
            .iter()
            .chain(&report.alternatives)
            .chain(&report.excluded)
            .collect::<Vec<_>>();
        assert!(!candidates.is_empty());
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.torch_version == "2.10.0")
        );
        assert!(candidates.iter().all(|candidate| {
            candidate
                .checks
                .wheel
                .reasons
                .iter()
                .all(|reason| reason.code != ReasonCode::VersionConstraintMismatch)
        }));
    }

    #[test]
    fn generic_linux_wheel_does_not_claim_a_verified_glibc_floor() {
        let env = environment("580.65.06", "13.0");
        let mut generic = wheel("2.13.0", "cu130");
        generic.platform_tags = vec!["linux_x86_64".to_owned()];
        assert!(matches!(
            platform_wheel_status(&generic, &env),
            PlatformMatch::Unknown
        ));
    }

    #[test]
    fn unsupported_platform_cannot_pass_via_interpreter_tags() {
        let mut env = environment("580.65.06", "13.0");
        env.platform.os = OperatingSystem::Macos;
        let candidate = wheel("2.13.0", "cu130");
        let (check, selected) = evaluate_platform(&env, env.python.as_ref(), &[&candidate]);
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(selected.is_none());
    }

    #[test]
    fn cpu_fallback_beats_newer_unverified_cuda_when_nvidia_is_unavailable() {
        let mut env = environment("580.65.06", "13.0");
        env.nvidia = NvidiaInfo {
            status: NvidiaDetectionStatus::CommandUnavailable,
            driver_version: None,
            reported_cuda_version: None,
            gpus: Vec::new(),
        };
        let report = resolve(
            &env,
            &snapshot(vec![wheel("2.13.0", "cu130"), wheel("2.12.0", "cpu")]),
            metadata(),
            &ResolverOptions::default(),
        )
        .expect("resolution");
        assert_eq!(
            report.recommendation.expect("CPU fallback").variant,
            CudaVariant::Cpu
        );
    }

    #[test]
    fn missing_legacy_companion_mapping_excludes_only_that_candidate() {
        let env = environment("580.65.06", "13.0");
        let mut vision = wheel("0.28.0", "cu130");
        vision.package = "torchvision".to_owned();
        let mut index = snapshot(vec![
            wheel("1.13.0", "cu130"),
            wheel("2.13.0", "cu130"),
            vision,
        ]);
        index.packages.push("torchvision".to_owned());
        let options = ResolverOptions {
            companions: [CompanionPackage::Torchvision].into_iter().collect(),
            ..ResolverOptions::default()
        };

        let report = resolve(&env, &index, metadata(), &options).expect("resolution continues");

        assert_eq!(
            report
                .recommendation
                .expect("current mapped release")
                .torch_version,
            "2.13.0"
        );
        assert!(report.excluded.iter().any(|candidate| {
            candidate.torch_version == "1.13.0"
                && candidate
                    .checks
                    .wheel
                    .reasons
                    .iter()
                    .any(|reason| reason.code == ReasonCode::WheelMissing)
        }));
    }

    #[test]
    fn explain_uses_pep440_normalized_version_equality() {
        let env = environment("530.30.02", "12.1");
        let index = snapshot(vec![wheel("2.6.0", "cu121")]);
        let candidate = explain(
            &env,
            &index,
            metadata(),
            "02.06",
            &"cu121".parse().expect("variant"),
            &ResolverOptions::default(),
        )
        .expect("explanation");
        assert_eq!(candidate.torch_version, "2.6.0");
        assert!(candidate.wheel.is_some());

        let index = snapshot(vec![wheel("2.12.0", "cpu")]);
        for requested in ["2.12", "v2.12.0"] {
            let candidate = explain(
                &env,
                &index,
                metadata(),
                requested,
                &CudaVariant::Cpu,
                &ResolverOptions::default(),
            )
            .expect("normalized explanation");
            assert!(candidate.wheel.is_some(), "{requested} should equal 2.12.0");
        }
    }
}
