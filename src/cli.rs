//! Command-line parsing, rendering, redaction, and exit-code handling.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::io::{self, IsTerminal, Write as IoWrite};
use std::path::PathBuf;
use std::process::ExitCode as ProcessExitCode;
use std::str::FromStr;

use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use serde::Serialize;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::core::{
    Candidate, CandidateWarning, CandidatesReport, CheckStatus, CommandSpec, CompatibilityCheck,
    CompatibilityStatus, DecisionReason, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    Environment, ErrorBody, ErrorKind, ErrorReport, ExitCode, ExplainReport, InspectReport,
    Installer, MetadataInfo, MetadataOrigin, NvidiaDetectionStatus, ReasonCode,
    RecommendationReport, SCHEMA_VERSION, VerificationGpuMapping, VerificationReport, WarningCode,
    is_unsafe_terminal_character,
};
use crate::detect::{DetectOptions, detect_environment};
use crate::index::{IndexOptions, LoadedIndex, load_index};
use crate::resolver::{CompanionPackage, ResolverOptions, explain, resolve};
use crate::verify::{VerifyOptions, verify_installed};

/// Runs the CLI and returns its documented process status.
pub async fn run() -> ProcessExitCode {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    let prepared = prepare_arguments(arguments.clone());
    let cli = match Cli::try_parse_from(prepared) {
        Ok(cli) => cli,
        Err(error) => return emit_parse_error(error, &arguments),
    };
    let code = execute(cli).await;
    process_exit_code(code)
}

fn emit_parse_error(error: clap::Error, arguments: &[OsString]) -> ProcessExitCode {
    if matches!(
        error.kind(),
        clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
    ) {
        let _ = error.print();
        return process_exit_code(ExitCode::Success);
    }
    let redact = redact_requested(arguments);
    if json_format_requested(arguments) {
        let message = if redact {
            "invalid command-line arguments; run torch-check --help".to_owned()
        } else {
            error.to_string()
        };
        let code = emit_error(
            OutputFormat::Json,
            ErrorKind::Usage,
            "invalid_arguments",
            &message,
            ExitCode::Incompatible,
        );
        process_exit_code(code)
    } else {
        if redact {
            eprintln!("torch-check: invalid command-line arguments; run torch-check --help");
        } else {
            let error = terminal_safe_parse_error(arguments, error.kind());
            let _ = error.print();
        }
        process_exit_code(ExitCode::Incompatible)
    }
}

fn terminal_safe_parse_error(
    arguments: &[OsString],
    original_kind: clap::error::ErrorKind,
) -> clap::Error {
    let arguments = arguments
        .iter()
        .map(|argument| OsString::from(terminal_text(&argument.to_string_lossy())))
        .collect::<Vec<_>>();
    match Cli::try_parse_from(prepare_arguments(arguments)) {
        Err(error) if error.kind() == original_kind => error,
        Err(_) | Ok(_) => Cli::command().error(
            original_kind,
            "invalid command-line arguments; run torch-check --help",
        ),
    }
}

fn json_format_requested(arguments: &[OsString]) -> bool {
    let arguments = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    arguments.iter().any(|argument| argument == "--format=json")
        || arguments
            .windows(2)
            .any(|pair| pair[0] == "--format" && pair[1] == "json")
}

fn redact_requested(arguments: &[OsString]) -> bool {
    arguments.iter().any(|argument| {
        let argument = argument.to_string_lossy();
        argument == "--redact" || argument.starts_with("--redact=")
    })
}

fn prepare_arguments(mut arguments: Vec<OsString>) -> Vec<OsString> {
    let mut index = 1;
    while index < arguments.len() {
        let argument = arguments[index].to_string_lossy();
        if matches!(
            argument.as_ref(),
            "inspect" | "recommend" | "candidates" | "explain" | "verify" | "completions" | "man"
        ) {
            return arguments;
        }
        if recommend_option(&argument) {
            arguments.insert(index, OsString::from("recommend"));
            return arguments;
        }
        if global_value_option(&argument) && !argument.contains('=') {
            index = index.saturating_add(2);
            continue;
        }
        if argument == "--"
            || (!global_value_option(&argument)
                && !matches!(
                    argument.as_ref(),
                    "--refresh" | "--offline" | "--redact" | "--help" | "-h" | "--version" | "-V"
                ))
        {
            return arguments;
        }
        index += 1;
    }
    arguments
}

fn recommend_option(argument: &str) -> bool {
    ["--torch-version", "--prerelease", "--installer", "--with"]
        .iter()
        .any(|option| argument == *option || argument.starts_with(&format!("{option}=")))
}

fn global_value_option(argument: &str) -> bool {
    ["--python", "--gpu", "--format", "--cache-dir"]
        .iter()
        .any(|option| argument == *option || argument.starts_with(&format!("{option}=")))
}

fn process_exit_code(code: ExitCode) -> ProcessExitCode {
    ProcessExitCode::from(u8::try_from(code.as_i32()).unwrap_or(ExitCode::InternalError as u8))
}

#[derive(Debug, Parser)]
#[command(
    name = "torch-check",
    version,
    about = "Find a safe PyTorch wheel for a Linux/NVIDIA/Python environment",
    long_about = "Inspect a Python/NVIDIA environment, enumerate wheels that actually exist on the official PyTorch indexes, and recommend only reviewed configurations whose driver, ABI, platform, and GPU compatibility can be established statically."
)]
struct Cli {
    /// Python interpreter to inspect, verify, and use in generated commands.
    #[arg(long, global = true, value_name = "PATH")]
    python: Option<PathBuf>,

    /// Physical nvidia-smi GPU indices to consider (comma-separated). Empty means every device.
    #[arg(long, global = true, value_delimiter = ',', value_name = "INDEX")]
    gpu: Vec<u32>,

    /// Human-readable or stable machine-readable output.
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Human)]
    format: OutputFormat,

    /// Bypass a fresh wheel-index cache.
    #[arg(long, global = true, conflicts_with = "offline")]
    refresh: bool,

    /// Never access the network; require a complete cached index.
    #[arg(long, global = true, conflicts_with = "refresh")]
    offline: bool,

    /// Override the platform cache directory.
    #[arg(long, global = true, value_name = "PATH")]
    cache_dir: Option<PathBuf>,

    /// Remove GPU UUIDs and local interpreter paths from output.
    #[arg(long, global = true)]
    redact: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Display detected OS, Python, NVIDIA, GPU, glibc, and Toolkit details.
    Inspect,
    /// Select the highest-ranked reviewed, statically compatible configuration.
    Recommend(RecommendArgs),
    /// List candidates by review and compatibility stage.
    Candidates(CandidatesArgs),
    /// Explain every check for one exact PyTorch/CUDA pair.
    Explain(ExplainArgs),
    /// Import installed PyTorch and execute bounded runtime checks.
    Verify,
    /// Generate a shell completion script.
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: Shell,
    },
    /// Generate the roff man page on stdout.
    Man,
}

#[derive(Debug, Clone, Args, Default)]
struct RecommendArgs {
    /// PEP 440 constraint, for example '>=2.10,<3'; a bare version is exact.
    #[arg(long, value_name = "SPECIFIER")]
    torch_version: Option<String>,

    /// Permit pre-release/development wheels in recommendation ranking.
    #[arg(long)]
    prerelease: bool,

    /// Installation command style.
    #[arg(long, value_enum, default_value_t = InstallerArg::Pip)]
    installer: InstallerArg,

    /// Add version-compatible companion packages after verifying their wheels exist.
    #[arg(long = "with", value_delimiter = ',', value_name = "PACKAGES")]
    companions: Vec<CompanionPackage>,
}

#[derive(Debug, Clone, Args, Default)]
struct CandidatesArgs {
    /// Also include candidates needing release review or static/runtime verification.
    #[arg(long, conflicts_with = "all")]
    unverified: bool,

    /// Include every compatibility status, including incompatible and unavailable candidates.
    #[arg(long, conflicts_with = "unverified")]
    all: bool,

    /// PEP 440 constraint, for example '>=2.10,<3'; a bare version is exact.
    #[arg(long, value_name = "SPECIFIER")]
    torch_version: Option<String>,

    /// Include pre-release/development wheels.
    #[arg(long)]
    prerelease: bool,

    /// Require companion wheels to exist for each listed candidate.
    #[arg(long = "with", value_delimiter = ',', value_name = "PACKAGES")]
    companions: Vec<CompanionPackage>,
}

#[derive(Debug, Clone, Args)]
struct ExplainArgs {
    /// Exact requirement, normally `torch==VERSION`.
    #[arg(value_name = "TORCH_REQUIREMENT")]
    requirement: String,

    /// Exact official CUDA/CPU index variant.
    #[arg(long, value_name = "VARIANT")]
    cuda: crate::core::CudaVariant,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum, Default)]
enum InstallerArg {
    #[default]
    Pip,
    Uv,
    UvAdd,
}

impl From<InstallerArg> for Installer {
    fn from(value: InstallerArg) -> Self {
        match value {
            InstallerArg::Pip => Self::Pip,
            InstallerArg::Uv => Self::Uv,
            InstallerArg::UvAdd => Self::UvAdd,
        }
    }
}

async fn execute(mut cli: Cli) -> ExitCode {
    let command = cli
        .command
        .take()
        .unwrap_or_else(|| Commands::Recommend(RecommendArgs::default()));
    match command {
        Commands::Completions { shell } => generate_completions(shell),
        Commands::Man => generate_man(),
        Commands::Inspect => command_inspect(&cli),
        Commands::Verify => command_verify(&cli),
        Commands::Recommend(args) => command_recommend(&cli, &args).await,
        Commands::Candidates(args) => command_candidates(&cli, &args).await,
        Commands::Explain(args) => command_explain(&cli, &args).await,
    }
}

fn command_inspect(cli: &Cli) -> ExitCode {
    let environment = match detect(cli) {
        Ok(environment) => environment,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Detection,
                "environment_detection_failed",
                &error.to_string(),
                ExitCode::InternalError,
            );
        }
    };
    let mut report = InspectReport {
        schema_version: SCHEMA_VERSION,
        environment,
    };
    if cli.redact {
        redact_environment(&mut report.environment);
    }
    let status = environment_exit_code(&report.environment);
    match cli.format {
        OutputFormat::Human => print_environment(&report.environment),
        OutputFormat::Json => {
            if let Err(error) = print_json(&report) {
                return emit_serialization_error(error);
            }
        }
    }
    status
}

fn command_verify(cli: &Cli) -> ExitCode {
    let environment = match detect(cli) {
        Ok(environment) => environment,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Detection,
                "environment_detection_failed",
                &error.to_string(),
                ExitCode::InternalError,
            );
        }
    };
    if let Some(error) = reject_environment_errors(cli, &environment) {
        return error;
    }
    let Some(python) = environment.python.as_ref() else {
        return emit_error(
            cli.format,
            ErrorKind::Verification,
            "python_unavailable",
            "no Python interpreter was available for verification",
            ExitCode::Incompatible,
        );
    };
    let mut options = VerifyOptions::new(python.executable.clone());
    let selection = match verification_gpu_selection(cli, &environment) {
        Ok(selection) => selection,
        Err(message) => {
            return emit_error(
                cli.format,
                ErrorKind::Verification,
                "gpu_visibility_mapping_failed",
                &message,
                ExitCode::Incompatible,
            );
        }
    };
    options.device_indices = selection.device_indices;
    options.cuda_visible_devices = selection.cuda_visible_devices;
    let mut report = verify_installed(&options);
    report.gpu_selection = selection.mappings;
    let mut diagnostics = environment.diagnostics.clone();
    diagnostics.append(&mut report.diagnostics);
    report.diagnostics = diagnostics;
    if cli.redact {
        redact_verification(&mut report);
    }
    match cli.format {
        OutputFormat::Human => print_verification(&report),
        OutputFormat::Json => {
            if let Err(error) = print_json(&report) {
                return emit_serialization_error(error);
            }
        }
    }
    if report.status == CompatibilityStatus::Verified {
        if report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Warning)
        {
            ExitCode::Warning
        } else {
            ExitCode::Success
        }
    } else {
        ExitCode::Incompatible
    }
}

#[derive(Debug, Default)]
struct VerificationGpuSelection {
    device_indices: Option<Vec<u32>>,
    cuda_visible_devices: Option<String>,
    mappings: Vec<VerificationGpuMapping>,
}

fn verification_gpu_selection(
    cli: &Cli,
    environment: &Environment,
) -> Result<VerificationGpuSelection, String> {
    if cli.gpu.is_empty() {
        return Ok(VerificationGpuSelection::default());
    }
    if environment.nvidia.gpus.is_empty() {
        return Err("no selected physical NVIDIA GPU was detected".to_owned());
    }

    let mut uuids = Vec::with_capacity(environment.nvidia.gpus.len());
    for gpu in &environment.nvidia.gpus {
        let uuid = gpu
            .uuid
            .as_deref()
            .filter(|uuid| valid_gpu_uuid(uuid))
            .ok_or_else(|| {
                format!(
                    "physical GPU {} has no safe, queryable UUID for verification",
                    gpu.index
                )
            })?;
        uuids.push(uuid.to_owned());
    }
    let count = u32::try_from(uuids.len())
        .map_err(|_| "selected CUDA device count exceeds supported range".to_owned())?;
    let mappings = environment
        .nvidia
        .gpus
        .iter()
        .zip(0..count)
        .map(|(gpu, logical_index)| VerificationGpuMapping {
            physical_index: gpu.index,
            logical_index,
            uuid: gpu.uuid.clone(),
        })
        .collect();
    Ok(VerificationGpuSelection {
        device_indices: Some((0..count).collect()),
        cuda_visible_devices: Some(uuids.join(",")),
        mappings,
    })
}

fn valid_gpu_uuid(value: &str) -> bool {
    value.starts_with("GPU-")
        && value.len() > 4
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

async fn command_recommend(cli: &Cli, args: &RecommendArgs) -> ExitCode {
    let environment = match detect(cli) {
        Ok(environment) => environment,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Detection,
                "environment_detection_failed",
                &error.to_string(),
                ExitCode::InternalError,
            );
        }
    };
    if let Some(error) = reject_environment_errors(cli, &environment) {
        return error;
    }
    let companions = companion_set(&args.companions);
    let loaded = match index(cli, &companions).await {
        Ok(loaded) => loaded,
        Err(error) => return error,
    };
    let options = ResolverOptions {
        torch_version: args.torch_version.clone(),
        include_prerelease: args.prerelease,
        installer: args.installer.into(),
        pin_python_in_install_command: cli.python.is_some(),
        companions,
    };
    let mut report = match resolve(&environment, &loaded.snapshot, loaded.metadata, &options) {
        Ok(report) => report,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Resolution,
                "resolution_failed",
                &error.to_string(),
                ExitCode::Incompatible,
            );
        }
    };
    let status = recommendation_exit_code(&report);
    if cli.redact {
        redact_recommendation(&mut report);
    }
    match cli.format {
        OutputFormat::Human => print_recommendation(&report),
        OutputFormat::Json => {
            if let Err(error) = print_json(&report) {
                return emit_serialization_error(error);
            }
        }
    }
    status
}

async fn command_candidates(cli: &Cli, args: &CandidatesArgs) -> ExitCode {
    let environment = match detect(cli) {
        Ok(environment) => environment,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Detection,
                "environment_detection_failed",
                &error.to_string(),
                ExitCode::InternalError,
            );
        }
    };
    if let Some(error) = reject_environment_errors(cli, &environment) {
        return error;
    }
    let companions = companion_set(&args.companions);
    let loaded = match index(cli, &companions).await {
        Ok(loaded) => loaded,
        Err(error) => return error,
    };
    let options = ResolverOptions {
        torch_version: args.torch_version.clone(),
        include_prerelease: args.prerelease,
        installer: Installer::Pip,
        pin_python_in_install_command: cli.python.is_some(),
        companions,
    };
    let report = match resolve(&environment, &loaded.snapshot, loaded.metadata, &options) {
        Ok(report) => report,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Resolution,
                "resolution_failed",
                &error.to_string(),
                ExitCode::Incompatible,
            );
        }
    };
    let mut candidates = Vec::new();
    if let Some(candidate) = report.recommendation {
        candidates.push(candidate);
    }
    candidates.extend(report.alternatives);
    if args.all {
        candidates.extend(report.excluded);
    }
    candidates.retain(|candidate| candidate_visible_in_candidates(candidate, args));
    let mut output = CandidatesReport {
        schema_version: SCHEMA_VERSION,
        environment: report.environment,
        metadata: report.metadata,
        candidates,
    };
    let status = candidates_exit_code(&output);
    if cli.redact {
        redact_environment(&mut output.environment);
        for candidate in &mut output.candidates {
            redact_candidate(candidate);
        }
    }
    match cli.format {
        OutputFormat::Human => print_candidates(&output),
        OutputFormat::Json => {
            if let Err(error) = print_json(&output) {
                return emit_serialization_error(error);
            }
        }
    }
    status
}

fn candidate_visible_in_candidates(candidate: &Candidate, args: &CandidatesArgs) -> bool {
    match candidate.compatibility {
        CompatibilityStatus::Verified
        | CompatibilityStatus::DirectCompatible
        | CompatibilityStatus::MinorCompatible => {
            !candidate_needs_review(candidate) || args.unverified || args.all
        }
        CompatibilityStatus::Unverified => args.unverified || args.all,
        CompatibilityStatus::Incompatible | CompatibilityStatus::Unavailable => args.all,
    }
}

async fn command_explain(cli: &Cli, args: &ExplainArgs) -> ExitCode {
    let torch_version = match exact_torch_version(&args.requirement) {
        Ok(version) => version,
        Err(message) => {
            return emit_error(
                cli.format,
                ErrorKind::Usage,
                "invalid_torch_requirement",
                &message,
                ExitCode::Incompatible,
            );
        }
    };
    let environment = match detect(cli) {
        Ok(environment) => environment,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Detection,
                "environment_detection_failed",
                &error.to_string(),
                ExitCode::InternalError,
            );
        }
    };
    if let Some(error) = reject_environment_errors(cli, &environment) {
        return error;
    }
    let loaded = match index(cli, &BTreeSet::new()).await {
        Ok(loaded) => loaded,
        Err(error) => return error,
    };
    let candidate = match explain(
        &environment,
        &loaded.snapshot,
        loaded.metadata.clone(),
        &torch_version,
        &args.cuda,
        &ResolverOptions::default(),
    ) {
        Ok(candidate) => candidate,
        Err(error) => {
            return emit_error(
                cli.format,
                ErrorKind::Resolution,
                "explanation_failed",
                &error.to_string(),
                ExitCode::Incompatible,
            );
        }
    };
    let status = candidate_exit_code(&candidate, &environment, &loaded.metadata);
    let mut report = ExplainReport {
        schema_version: SCHEMA_VERSION,
        environment,
        metadata: loaded.metadata,
        candidate,
    };
    if cli.redact {
        redact_environment(&mut report.environment);
        redact_candidate(&mut report.candidate);
    }
    match cli.format {
        OutputFormat::Human => print_explanation(&report),
        OutputFormat::Json => {
            if let Err(error) = print_json(&report) {
                return emit_serialization_error(error);
            }
        }
    }
    status
}

fn detect(cli: &Cli) -> Result<Environment, crate::detect::DetectError> {
    detect_environment(&DetectOptions {
        python: cli.python.clone(),
        gpu_indices: cli.gpu.clone(),
        ..DetectOptions::default()
    })
}

async fn index(
    cli: &Cli,
    companions: &BTreeSet<CompanionPackage>,
) -> Result<LoadedIndex, ExitCode> {
    let mut packages = vec!["torch".to_owned()];
    packages.extend(
        companions
            .iter()
            .map(|companion| companion.as_str().to_owned()),
    );
    let options = IndexOptions {
        cache_dir: cli.cache_dir.clone(),
        offline: cli.offline,
        refresh: cli.refresh,
        packages,
        ..IndexOptions::default()
    };
    load_index(&options).await.map_err(|error| {
        emit_error(
            cli.format,
            ErrorKind::Metadata,
            "wheel_metadata_unavailable",
            &error.to_string(),
            ExitCode::MetadataFailure,
        )
    })
}

fn companion_set(values: &[CompanionPackage]) -> BTreeSet<CompanionPackage> {
    values.iter().copied().collect()
}

fn exact_torch_version(requirement: &str) -> Result<String, String> {
    let value = requirement.trim();
    let version = value
        .strip_prefix("torch==")
        .or_else(|| value.strip_prefix("=="))
        .unwrap_or(value);
    if version.is_empty() || version.contains(',') || version.starts_with(['<', '>', '!', '~', '='])
    {
        return Err("explain requires one exact version such as `torch==2.12.1`".to_owned());
    }
    pep440_rs::Version::from_str(version)
        .map(|version| version.to_string())
        .map_err(|error| format!("invalid exact PyTorch version `{version}`: {error}"))
}

fn recommendation_exit_code(report: &RecommendationReport) -> ExitCode {
    let Some(candidate) = report.recommendation.as_ref() else {
        return ExitCode::Incompatible;
    };
    let status = candidate_exit_code(candidate, &report.environment, &report.metadata);
    if status == ExitCode::Success && is_nvidia_cpu_fallback(report) {
        ExitCode::Warning
    } else {
        status
    }
}

fn is_nvidia_cpu_fallback(report: &RecommendationReport) -> bool {
    report.recommendation.as_ref().is_some_and(|candidate| {
        !candidate.variant.is_cuda()
            && report.environment.nvidia.status == NvidiaDetectionStatus::Detected
            && !report.environment.nvidia.gpus.is_empty()
            && report
                .alternatives
                .iter()
                .chain(&report.excluded)
                .any(|candidate| candidate.variant.is_cuda())
    })
}

fn candidate_exit_code(
    candidate: &Candidate,
    environment: &Environment,
    metadata: &MetadataInfo,
) -> ExitCode {
    if has_environment_errors(environment) {
        return ExitCode::Incompatible;
    }
    match candidate.compatibility {
        CompatibilityStatus::Incompatible | CompatibilityStatus::Unavailable => {
            ExitCode::Incompatible
        }
        _ if candidate_has_warnings(candidate)
            || has_environment_warnings(environment)
            || metadata_has_warning(metadata) =>
        {
            ExitCode::Warning
        }
        _ => ExitCode::Success,
    }
}

fn candidate_has_warnings(candidate: &Candidate) -> bool {
    !candidate.warnings.is_empty()
        || matches!(
            candidate.compatibility,
            CompatibilityStatus::MinorCompatible | CompatibilityStatus::Unverified
        )
}

fn candidate_needs_review(candidate: &Candidate) -> bool {
    candidate.compatibility == CompatibilityStatus::Unverified
        || candidate.official_preference.is_none()
}

fn candidates_exit_code(report: &CandidatesReport) -> ExitCode {
    let has_usable_candidate = report.candidates.iter().any(|candidate| {
        matches!(
            candidate.compatibility,
            CompatibilityStatus::Verified
                | CompatibilityStatus::DirectCompatible
                | CompatibilityStatus::MinorCompatible
                | CompatibilityStatus::Unverified
        )
    });
    if !has_usable_candidate {
        ExitCode::Incompatible
    } else if report.candidates.iter().any(candidate_has_warnings)
        || has_environment_warnings(&report.environment)
        || metadata_has_warning(&report.metadata)
    {
        ExitCode::Warning
    } else {
        ExitCode::Success
    }
}

fn has_environment_warnings(environment: &Environment) -> bool {
    environment
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Warning)
}

fn has_environment_errors(environment: &Environment) -> bool {
    environment
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
}

fn environment_exit_code(environment: &Environment) -> ExitCode {
    if has_environment_errors(environment) {
        ExitCode::Incompatible
    } else if has_environment_warnings(environment) {
        ExitCode::Warning
    } else {
        ExitCode::Success
    }
}

fn reject_environment_errors(cli: &Cli, environment: &Environment) -> Option<ExitCode> {
    let codes = environment
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| json_name(&diagnostic.code))
        .collect::<Vec<_>>();
    (!codes.is_empty()).then(|| {
        emit_error(
            cli.format,
            ErrorKind::Detection,
            "environment_unsupported_or_invalid",
            &format!(
                "environment contains blocking diagnostics: {}",
                codes.join(", ")
            ),
            ExitCode::Incompatible,
        )
    })
}

fn metadata_has_warning(metadata: &MetadataInfo) -> bool {
    metadata.stale || metadata.origin == crate::core::MetadataOrigin::StaleIfError
}

fn generate_completions(shell: Shell) -> ExitCode {
    let mut command = Cli::command();
    let name = command.get_name().to_owned();
    clap_complete::generate(shell, &mut command, name, &mut io::stdout());
    ExitCode::Success
}

fn generate_man() -> ExitCode {
    let man = clap_mangen::Man::new(Cli::command());
    if let Err(error) = man.render(&mut io::stdout()) {
        eprintln!("torch-check: could not render man page: {error}");
        ExitCode::InternalError
    } else {
        ExitCode::Success
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<(), serde_json::Error> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    // A trailing newline keeps the output convenient while remaining one JSON document.
    writeln!(lock).map_err(serde_json::Error::io)
}

fn emit_error(
    format: OutputFormat,
    kind: ErrorKind,
    code: &str,
    message: &str,
    exit_code: ExitCode,
) -> ExitCode {
    let report = ErrorReport {
        schema_version: SCHEMA_VERSION,
        error: ErrorBody {
            kind,
            code: code.to_owned(),
            message: message.to_owned(),
        },
    };
    match format {
        OutputFormat::Human => eprintln!("torch-check: {}", terminal_text(message)),
        OutputFormat::Json => {
            if let Err(error) = print_json(&report) {
                eprintln!("torch-check: could not serialize an error report: {error}");
                return ExitCode::InternalError;
            }
        }
    }
    exit_code
}

fn emit_serialization_error(error: serde_json::Error) -> ExitCode {
    eprintln!("torch-check: could not serialize output: {error}");
    ExitCode::InternalError
}

const DEFAULT_HUMAN_WIDTH: usize = 80;
const MIN_HUMAN_WIDTH: usize = 44;
const MAX_HUMAN_WIDTH: usize = 120;
const OTHER_OPTIONS_LIMIT: usize = 3;
const GPU_ALTERNATIVES_LIMIT: usize = 2;
const CPU_FALLBACK_LIMIT: usize = 1;
const NEEDS_VERIFICATION_LIMIT: usize = 2;

#[derive(Debug, Clone, Eq, PartialEq)]
struct HumanGpuGroup {
    name: String,
    capability: Option<crate::core::ComputeCapability>,
    driver: crate::core::NumericVersion,
    indices: Vec<u32>,
}

fn print_environment(environment: &Environment) {
    emit_human(&render_environment(environment, human_output_width()));
}

fn render_environment(environment: &Environment, width: usize) -> String {
    let mut output = String::new();
    write_environment(&mut output, environment, width);
    output
}

fn write_environment(output: &mut String, environment: &Environment, width: usize) {
    let distribution = terminal_text(
        environment
            .platform
            .distribution
            .as_deref()
            .unwrap_or_else(|| operating_system_name(&environment.platform.os)),
    );
    let architecture = terminal_text(&environment.platform.architecture.to_string());
    let kernel = terminal_text(optional(environment.platform.kernel_version.as_deref()));
    let glibc = environment
        .glibc
        .as_ref()
        .map_or_else(|| "unknown".to_owned(), ToString::to_string);

    let _ = writeln!(output, "System");
    write_wrapped(
        output,
        "  ",
        "  ",
        &format!("{distribution} · {architecture} · kernel {kernel}"),
        width,
    );
    if let Some(python) = &environment.python {
        let abi = terminal_text(
            &python
                .cpython_abi_tag()
                .unwrap_or_else(|| optional(python.soabi.as_deref()).to_owned()),
        );
        write_wrapped(
            output,
            "  ",
            "  ",
            &format!(
                "{} {} · {abi} · glibc {glibc}",
                python_implementation_name(&python.implementation),
                python.version
            ),
            width,
        );
        write_wrapped(
            output,
            "  Python ",
            "    ",
            &terminal_text(&python.executable.to_string_lossy()),
            width,
        );
    } else {
        write_wrapped(
            output,
            "  ",
            "  ",
            &format!("Python not found · glibc {glibc}"),
            width,
        );
    }

    let _ = writeln!(output);
    let _ = writeln!(output, "NVIDIA");
    let groups = group_gpus(environment);
    if groups.is_empty() {
        write_wrapped(
            output,
            "  ",
            "  ",
            nvidia_status_text(environment.nvidia.status),
            width,
        );
        let driver = environment
            .nvidia
            .driver_version
            .as_ref()
            .map_or_else(|| "unknown".to_owned(), ToString::to_string);
        write_wrapped(output, "  ", "  ", &format!("Driver {driver}"), width);
    } else {
        for group in groups {
            let count = if group.indices.len() > 1 {
                format!("{}× ", group.indices.len())
            } else {
                String::new()
            };
            let capability = group
                .capability
                .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
            let index_label = if group.indices.len() > 1 {
                "GPUs"
            } else {
                "GPU"
            };
            write_wrapped(
                output,
                "  ",
                "    ",
                &format!("{count}{}", terminal_text(&group.name)),
                width,
            );
            write_wrapped(
                output,
                "    ",
                "    ",
                &format!(
                    "CC {capability} · driver {} · {index_label} {}",
                    group.driver,
                    format_gpu_indices(&group.indices)
                ),
                width,
            );
        }
    }
    let reported = environment
        .nvidia
        .reported_cuda_version
        .as_ref()
        .map_or_else(|| "unknown".to_owned(), ToString::to_string);
    let toolkit = environment.cuda_toolkit.as_ref().map_or_else(
        || "not found".to_owned(),
        |toolkit| toolkit.version.to_string(),
    );
    write_wrapped(
        output,
        "  ",
        "  ",
        &format!("Reported CUDA {reported} · toolkit {toolkit} · informational for wheels"),
        width,
    );

    let diagnostics = environment
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code != DiagnosticCode::CudaToolkitNotFound)
        .collect::<Vec<_>>();
    if !diagnostics.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Environment notes");
        for diagnostic in diagnostics {
            write_wrapped(
                output,
                &format!("  {}: ", diagnostic_severity_name(diagnostic.severity)),
                "    ",
                &diagnostic_text(diagnostic),
                width,
            );
        }
    }
}

fn print_recommendation(report: &RecommendationReport) {
    emit_human(&render_recommendation(report, human_output_width()));
}

fn render_recommendation(report: &RecommendationReport, width: usize) -> String {
    let mut output = render_environment(&report.environment, width);
    let recommendation_is_unverified = report
        .recommendation
        .as_ref()
        .is_some_and(|candidate| candidate.compatibility == CompatibilityStatus::Unverified);
    let nvidia_detected = report.environment.nvidia.status == NvidiaDetectionStatus::Detected
        && !report.environment.nvidia.gpus.is_empty();
    let recommendation_is_cpu_fallback = is_nvidia_cpu_fallback(report);
    let mut displayed_candidate_count = 0;

    let _ = writeln!(output);
    if recommendation_is_unverified {
        let _ = writeln!(output, "Recommendation");
        write_wrapped(
            &mut output,
            "  ",
            "  ",
            "No configuration passed every static safety check.",
            width,
        );
    } else {
        let _ = writeln!(
            output,
            "{}",
            if recommendation_is_cpu_fallback {
                "CPU fallback"
            } else {
                "Recommendation"
            }
        );
        if let Some(candidate) = &report.recommendation {
            write_candidate_summary(&mut output, candidate, "  ", width, true);
            displayed_candidate_count += 1;
            if recommendation_is_cpu_fallback {
                write_wrapped(
                    &mut output,
                    "  Note: ",
                    "    ",
                    "No reviewed CUDA candidate passed all static checks.",
                    width,
                );
            }
            if let Some(install) = &report.install {
                let _ = writeln!(output);
                let _ = writeln!(output, "Install");
                write_command(&mut output, install, width);
            }
        } else {
            write_wrapped(
                &mut output,
                "  ",
                "  ",
                "No recommendation-eligible reviewed configuration matched the selected constraints.",
                width,
            );
        }
    }

    let safe_alternatives = report
        .alternatives
        .iter()
        .filter(|candidate| !candidate_needs_review(candidate))
        .collect::<Vec<_>>();
    if nvidia_detected {
        let safe_gpu_alternatives = safe_alternatives
            .iter()
            .copied()
            .filter(|candidate| candidate.variant.is_cuda())
            .take(GPU_ALTERNATIVES_LIMIT)
            .collect::<Vec<_>>();
        if !safe_gpu_alternatives.is_empty() {
            let _ = writeln!(output);
            let _ = writeln!(output, "GPU alternatives");
            for candidate in &safe_gpu_alternatives {
                write_candidate_summary(&mut output, candidate, "  ", width, false);
            }
            displayed_candidate_count += safe_gpu_alternatives.len();
        }
    }

    let mut needs_verification = report
        .alternatives
        .iter()
        .filter(|candidate| candidate_needs_review(candidate))
        .collect::<Vec<_>>();
    if recommendation_is_unverified {
        needs_verification.extend(report.recommendation.iter());
    }
    needs_verification.sort_by_key(|candidate| review_display_priority(candidate));
    if !needs_verification.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Needs review or verification");
        let shown = needs_verification
            .iter()
            .take(NEEDS_VERIFICATION_LIMIT)
            .copied()
            .collect::<Vec<_>>();
        for candidate in &shown {
            write_candidate_summary(&mut output, candidate, "  ", width, true);
        }
        displayed_candidate_count += shown.len();
        if needs_verification.len() > NEEDS_VERIFICATION_LIMIT {
            write_wrapped(
                &mut output,
                "  ",
                "  ",
                &format!(
                    "… {} more candidate(s) requiring review or verification.",
                    needs_verification.len() - NEEDS_VERIFICATION_LIMIT
                ),
                width,
            );
        }
    }

    if nvidia_detected
        && report
            .recommendation
            .as_ref()
            .is_some_and(|candidate| candidate.variant.is_cuda())
    {
        let cpu_fallbacks = safe_alternatives
            .iter()
            .copied()
            .filter(|candidate| !candidate.variant.is_cuda())
            .take(CPU_FALLBACK_LIMIT)
            .collect::<Vec<_>>();
        if !cpu_fallbacks.is_empty() {
            let _ = writeln!(output);
            let _ = writeln!(output, "CPU fallback");
            for candidate in &cpu_fallbacks {
                write_candidate_summary(&mut output, candidate, "  ", width, false);
            }
            displayed_candidate_count += cpu_fallbacks.len();
        }
    }

    let upgrade_hint = driver_upgrade_hint(report);
    let show_generic_options = !nvidia_detected && !safe_alternatives.is_empty();
    if show_generic_options || upgrade_hint.is_some() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Other options");
    }
    if show_generic_options {
        for candidate in safe_alternatives.iter().take(OTHER_OPTIONS_LIMIT) {
            write_candidate_summary(&mut output, candidate, "  ", width, false);
        }
        displayed_candidate_count += safe_alternatives.len().min(OTHER_OPTIONS_LIMIT);
    }
    if let Some((candidate, reason)) = upgrade_hint {
        write_wrapped(
            &mut output,
            "  Driver upgrade: ",
            "    ",
            &driver_upgrade_text(candidate, reason),
            width,
        );
    }

    let total_candidates = usize::from(report.recommendation.is_some())
        + report.alternatives.len()
        + report.excluded.len();
    if total_candidates > displayed_candidate_count {
        let _ = writeln!(output);
        write_wrapped(
            &mut output,
            "  ",
            "  ",
            &format!(
                "Showing a summary of {total_candidates} candidates. Run `torch-check candidates --all` for every option and exclusion reason."
            ),
            width,
        );
    }

    let _ = writeln!(output);
    write_metadata_footer(&mut output, &report.metadata, width);
    output
}

fn review_display_priority(candidate: &Candidate) -> u8 {
    match (candidate.variant.is_cuda(), candidate.official_preference) {
        (true, Some(_)) => 0,
        (true, None) => 1,
        (false, _) => 2,
    }
}

fn print_candidates(report: &CandidatesReport) {
    let width = human_output_width();
    let mut output = render_environment(&report.environment, width);
    let _ = writeln!(output);
    let _ = writeln!(output, "Candidates ({})", report.candidates.len());
    if report.candidates.is_empty() {
        let _ = writeln!(output, "  No candidates matched.");
    }
    for candidate in &report.candidates {
        write_candidate_summary(&mut output, candidate, "  ", width, true);
    }
    let _ = writeln!(output);
    write_metadata_footer(&mut output, &report.metadata, width);
    emit_human(&output);
}

fn print_explanation(report: &ExplainReport) {
    let width = human_output_width();
    let mut output = render_environment(&report.environment, width);
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "PyTorch {} + {} ({})",
        report.candidate.torch_version,
        report.candidate.variant,
        compatibility_name(report.candidate.compatibility)
    );
    let checks = &report.candidate.checks;
    write_check(&mut output, "Wheel", &checks.wheel, width);
    write_check(&mut output, "Python", &checks.python, width);
    write_check(&mut output, "Platform", &checks.platform, width);
    write_check(
        &mut output,
        "GPU architecture",
        &checks.gpu_architecture,
        width,
    );
    write_check(&mut output, "Driver", &checks.driver, width);
    write_check(&mut output, "Runtime", &checks.runtime, width);
    for warning in &report.candidate.warnings {
        write_wrapped(
            &mut output,
            "  Warning: ",
            "    ",
            &warning_text(warning),
            width,
        );
    }
    let _ = writeln!(output);
    write_metadata_footer(&mut output, &report.metadata, width);
    emit_human(&output);
}

fn write_check(output: &mut String, name: &str, check: &CompatibilityCheck, width: usize) {
    let _ = writeln!(output, "  {name}: {}", check_status_name(check.status));
    for reason in &check.reasons {
        write_wrapped(output, "    - ", "      ", &reason_text(reason), width);
    }
}

fn print_verification(report: &VerificationReport) {
    let width = human_output_width();
    let mut output = String::new();
    let _ = writeln!(output, "Verification");
    write_wrapped(
        &mut output,
        "  ",
        "  ",
        &format!(
            "{} · PyTorch {} · compiled CUDA {}",
            compatibility_name(report.status),
            terminal_text(optional(report.torch_version.as_deref())),
            terminal_text(optional(report.compiled_cuda.as_deref()))
        ),
        width,
    );
    write_wrapped(
        &mut output,
        "  Python ",
        "    ",
        &terminal_text(&report.python_executable.to_string_lossy()),
        width,
    );
    write_wrapped(
        &mut output,
        "  ",
        "  ",
        &format!(
            "CUDA available {} · devices {} · cuDNN {}",
            optional_bool(report.cuda_available),
            report
                .device_count
                .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
            optional_bool(report.cudnn_available)
        ),
        width,
    );
    for device in &report.devices {
        write_wrapped(
            &mut output,
            "  ",
            "    ",
            &format!(
                "GPU {}: {} · SM {} · operations {}",
                device.index,
                terminal_text(&device.name),
                device.capability,
                if device.operations_ok {
                    "passed"
                } else {
                    "failed"
                }
            ),
            width,
        );
    }
    for mapping in &report.gpu_selection {
        let uuid = mapping
            .uuid
            .as_deref()
            .map_or_else(String::new, |uuid| format!(" · {}", terminal_text(uuid)));
        write_wrapped(
            &mut output,
            "  ",
            "    ",
            &format!(
                "GPU mapping: physical {} → logical {}{uuid}",
                mapping.physical_index, mapping.logical_index
            ),
            width,
        );
    }
    if !report.checks.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Checks");
    }
    for check in &report.checks {
        let detail = check.detail.as_deref().map_or_else(String::new, |detail| {
            format!(" · {}", terminal_text(detail))
        });
        write_wrapped(
            &mut output,
            if check.passed { "  pass  " } else { "  fail  " },
            "        ",
            &format!("{}{detail}", terminal_text(&check.name)),
            width,
        );
    }
    for diagnostic in &report.diagnostics {
        write_wrapped(
            &mut output,
            &format!("  {}: ", diagnostic_severity_name(diagnostic.severity)),
            "    ",
            &diagnostic_text(diagnostic),
            width,
        );
    }
    if let Some(error) = &report.error {
        write_wrapped(
            &mut output,
            "  Error: ",
            "    ",
            &terminal_text(error),
            width,
        );
    }
    emit_human(&output);
}

fn write_candidate_summary(
    output: &mut String,
    candidate: &Candidate,
    indent: &str,
    width: usize,
    include_details: bool,
) {
    write_wrapped(
        output,
        indent,
        &format!("{indent}  "),
        &format!(
            "PyTorch {} + {} ({})",
            candidate.torch_version,
            candidate.variant,
            compatibility_name(candidate.compatibility)
        ),
        width,
    );
    if include_details {
        for warning in &candidate.warnings {
            write_wrapped(
                output,
                &format!("{indent}  - "),
                &format!("{indent}    "),
                &warning_text(warning),
                width,
            );
        }
    }
    if matches!(
        candidate.compatibility,
        CompatibilityStatus::Incompatible | CompatibilityStatus::Unavailable
    ) {
        for reason in failing_reasons(candidate) {
            write_wrapped(
                output,
                &format!("{indent}  - "),
                &format!("{indent}    "),
                &reason_text(reason),
                width,
            );
        }
    }
}

fn failing_reasons(candidate: &Candidate) -> impl Iterator<Item = &DecisionReason> {
    [
        &candidate.checks.wheel,
        &candidate.checks.python,
        &candidate.checks.platform,
        &candidate.checks.gpu_architecture,
        &candidate.checks.driver,
    ]
    .into_iter()
    .filter(|check| check.status == CheckStatus::Fail)
    .flat_map(|check| check.reasons.iter())
}

fn human_output_width() -> usize {
    let configured = std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let detected = io::stdout().is_terminal().then(|| {
        terminal_size::terminal_size().map(|(terminal_size::Width(width), _)| usize::from(width))
    });
    configured
        .or_else(|| detected.flatten())
        .unwrap_or(DEFAULT_HUMAN_WIDTH)
        .clamp(MIN_HUMAN_WIDTH, MAX_HUMAN_WIDTH)
}

fn emit_human(output: &str) {
    if !human_color_enabled() {
        print!("{output}");
        return;
    }
    let mut styled = String::with_capacity(output.len() + 128);
    for line in output.split_inclusive('\n') {
        let (content, newline) = line
            .strip_suffix('\n')
            .map_or((line, ""), |content| (content, "\n"));
        if let Some(code) = human_style_code(content) {
            let _ = write!(styled, "\u{1b}[{code}m{content}\u{1b}[0m{newline}");
        } else {
            styled.push_str(content);
            styled.push_str(newline);
        }
    }
    print!("{styled}");
}

fn human_color_enabled() -> bool {
    io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").map_or(true, |term| term != "dumb")
        && std::env::var("CLICOLOR").map_or(true, |value| value != "0")
}

fn human_style_code(line: &str) -> Option<&'static str> {
    if matches!(
        line,
        "System"
            | "NVIDIA"
            | "Environment notes"
            | "Recommendation"
            | "Install"
            | "GPU alternatives"
            | "CPU fallback"
            | "Needs review or verification"
            | "Other options"
            | "Verification"
            | "Checks"
    ) || line.starts_with("Candidates (")
    {
        Some("1;36")
    } else if line.contains("(verified)") || line.contains("(direct-compatible)") {
        Some("32")
    } else if line.contains("(minor-compatible)")
        || line.contains("(unverified)")
        || line.trim_start().starts_with("Note:")
        || line.trim_start().starts_with("Warning:")
        || line.trim_start().starts_with("Driver upgrade:")
    {
        Some("33")
    } else if line.contains("(incompatible)")
        || line.contains("(unavailable)")
        || line.trim_start().starts_with("Error:")
        || line.trim_start().starts_with("fail  ")
    {
        Some("31")
    } else if line.trim_start().starts_with("pass  ") {
        Some("32")
    } else if line.starts_with("Metadata:") {
        Some("2")
    } else {
        None
    }
}

fn group_gpus(environment: &Environment) -> Vec<HumanGpuGroup> {
    let mut grouped = BTreeMap::<
        (
            String,
            Option<crate::core::ComputeCapability>,
            crate::core::NumericVersion,
        ),
        HumanGpuGroup,
    >::new();
    for gpu in &environment.nvidia.gpus {
        let key = (
            gpu.name.clone(),
            gpu.compute_capability,
            gpu.driver_version.clone(),
        );
        grouped
            .entry(key)
            .and_modify(|group| group.indices.push(gpu.index))
            .or_insert_with(|| HumanGpuGroup {
                name: gpu.name.clone(),
                capability: gpu.compute_capability,
                driver: gpu.driver_version.clone(),
                indices: vec![gpu.index],
            });
    }
    let mut groups = grouped.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.indices.sort_unstable();
        group.indices.dedup();
    }
    groups.sort_by_key(|group| group.indices.first().copied().unwrap_or(u32::MAX));
    groups
}

fn format_gpu_indices(indices: &[u32]) -> String {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut ranges = Vec::new();
    let mut position = 0;
    while position < sorted.len() {
        let start = sorted[position];
        let mut end = start;
        while position + 1 < sorted.len() && sorted[position + 1] == end.saturating_add(1) {
            position += 1;
            end = sorted[position];
        }
        if start == end {
            ranges.push(start.to_string());
        } else {
            ranges.push(format!("{start}–{end}"));
        }
        position += 1;
    }
    if ranges.is_empty() {
        "none".to_owned()
    } else {
        ranges.join(", ")
    }
}

fn write_wrapped(
    output: &mut String,
    first_prefix: &str,
    continuation_prefix: &str,
    text: &str,
    width: usize,
) {
    let width = width.max(MIN_HUMAN_WIDTH);
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        let _ = writeln!(output, "{first_prefix}");
        return;
    }

    let mut first_line = true;
    let mut content = String::new();
    let mut word_index = 0;
    while word_index < words.len() {
        let prefix = if first_line {
            first_prefix
        } else {
            continuation_prefix
        };
        let separator_width = usize::from(!content.is_empty());
        let available = width
            .saturating_sub(display_width(prefix))
            .saturating_sub(display_width(&content))
            .saturating_sub(separator_width);
        let word = words[word_index];
        let word_width = display_width(word);

        if word_width <= available {
            if !content.is_empty() {
                content.push(' ');
            }
            content.push_str(word);
            word_index += 1;
            continue;
        }

        if !content.is_empty() {
            let _ = writeln!(output, "{prefix}{content}");
            content.clear();
            first_line = false;
            continue;
        }

        let capacity = width.saturating_sub(display_width(prefix)).max(1);
        let split = byte_index_after_width(word, capacity);
        let (head, tail) = word.split_at(split);
        let _ = writeln!(output, "{prefix}{head}");
        first_line = false;
        if tail.is_empty() {
            word_index += 1;
        } else {
            // Replace the current borrowed word with owned chunks by finishing the
            // remaining token directly. Long paths and URLs are rare, but must not
            // force every following line past the terminal width.
            let mut remaining = tail;
            while display_width(remaining)
                > width
                    .saturating_sub(display_width(continuation_prefix))
                    .max(1)
            {
                let capacity = width
                    .saturating_sub(display_width(continuation_prefix))
                    .max(1);
                let split = byte_index_after_width(remaining, capacity);
                let (chunk, rest) = remaining.split_at(split);
                let _ = writeln!(output, "{continuation_prefix}{chunk}");
                remaining = rest;
            }
            content.push_str(remaining);
            word_index += 1;
        }
    }

    let prefix = if first_line {
        first_prefix
    } else {
        continuation_prefix
    };
    if !content.is_empty() {
        let _ = writeln!(output, "{prefix}{content}");
    }
}

fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
}

fn byte_index_after_width(value: &str, maximum_width: usize) -> usize {
    let mut width: usize = 0;
    for (index, character) in value.char_indices() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width.saturating_add(character_width) > maximum_width {
            return if index == 0 {
                character.len_utf8()
            } else {
                index
            };
        }
        width = width.saturating_add(character_width);
    }
    value.len()
}

fn write_command(output: &mut String, command: &CommandSpec, width: usize) {
    let tokens = std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        // Resolver-generated commands contain no terminal controls. Keep this
        // defensive fallback for reports constructed through the public data
        // model: showing a visibly escaped token is safer than allowing a
        // forged command to alter terminal layout.
        .map(|token| {
            if token.chars().any(is_unsafe_terminal_character) {
                terminal_text(token)
            } else {
                token.to_owned()
            }
        })
        .collect::<Vec<_>>();
    let first_prefix = "  ";
    let continuation_prefix = "    ";
    let width = width.max(MIN_HUMAN_WIDTH);
    let mut line = first_prefix.to_owned();
    let mut line_has_token = false;
    for (index, token) in tokens.iter().enumerate() {
        let escaped = shell_escape_token(token);
        let separator = if line_has_token { " " } else { "" };
        let needs_continuation = index + 1 < tokens.len();
        let reserve = usize::from(needs_continuation) * 2;
        let candidate_width =
            display_width(&line) + display_width(separator) + display_width(&escaped) + reserve;
        if candidate_width <= width {
            line.push_str(separator);
            line.push_str(&escaped);
            line_has_token = true;
            continue;
        }

        if line_has_token {
            let _ = writeln!(output, "{line} \\");
            line = continuation_prefix.to_owned();
            line_has_token = false;
        }

        let standalone_width = display_width(&line) + display_width(&escaped) + reserve;
        if standalone_width <= width {
            line.push_str(&escaped);
            line_has_token = true;
            continue;
        }

        write_split_shell_token(output, token, &line, width, needs_continuation);
        if needs_continuation {
            line = continuation_prefix.to_owned();
        } else {
            return;
        }
    }
    let _ = writeln!(output, "{line}");
}

fn shell_escape_token(value: &str) -> String {
    shell_escape::escape(value.into()).into_owned()
}

fn write_split_shell_token(
    output: &mut String,
    token: &str,
    first_prefix: &str,
    width: usize,
    has_following_token: bool,
) {
    let mut remaining = token;
    let mut prefix = first_prefix;
    loop {
        let escaped_remaining = shell_escape_token(remaining);
        let final_suffix = if has_following_token { " \\" } else { "" };
        let final_capacity = width
            .saturating_sub(display_width(prefix))
            .saturating_sub(display_width(final_suffix));
        if display_width(&escaped_remaining) <= final_capacity {
            let _ = writeln!(output, "{prefix}{escaped_remaining}{final_suffix}");
            return;
        }

        let chunk_capacity = width
            .saturating_sub(display_width(prefix))
            .saturating_sub(1)
            .max(1);
        let split = shell_chunk_end(remaining, chunk_capacity);
        let (chunk, tail) = remaining.split_at(split);
        let escaped_chunk = shell_escape_token(chunk);
        let _ = writeln!(output, "{prefix}{escaped_chunk}\\");
        remaining = tail;
        // No indentation is allowed inside a shell token: the backslash-newline
        // is removed before tokenization, so adjacent escaped chunks concatenate.
        prefix = "";
    }
}

fn shell_chunk_end(value: &str, maximum_width: usize) -> usize {
    let mut last_fitting_end = 0;
    for (index, character) in value.char_indices() {
        let end = index + character.len_utf8();
        if display_width(&shell_escape_token(&value[..end])) <= maximum_width {
            last_fitting_end = end;
        }
    }
    if last_fitting_end == 0 {
        value.chars().next().map_or(0, char::len_utf8)
    } else {
        last_fitting_end
    }
}

fn warning_text(warning: &CandidateWarning) -> String {
    let text = match warning.code {
        WarningCode::PtxOrNewDriverFeatureMayFail => {
            "PTX JIT or features that require a newer driver may fail."
        }
        WarningCode::GpuArchitectureUnknown => {
            "No reviewed architecture evidence is linked to this exact wheel version; verify after installation."
        }
        WarningCode::StaleMetadata => "Wheel metadata is stale.",
        WarningCode::NvidiaDetectionIncomplete => {
            "NVIDIA detection was incomplete, so CUDA compatibility is uncertain."
        }
        WarningCode::BuiltinPythonTags => {
            "Python wheel tags were generated by the conservative built-in fallback."
        }
        WarningCode::OfficialPreferenceUnknown => {
            "No reviewed PyTorch release preference is available for this configuration."
        }
        WarningCode::NotOfficialReleaseConfiguration => {
            "The wheel exists in the index but is not in the reviewed PyTorch release configuration."
        }
    };
    format!("{text}{}", details_suffix(&warning.details))
}

fn reason_text(reason: &DecisionReason) -> String {
    let detail = |key: &str| reason.details.get(key).map(|value| terminal_text(value));
    match reason.code {
        ReasonCode::WheelExists => format!(
            "Official wheel exists{}.",
            detail("count").map_or_else(String::new, |count| format!(" ({count} file(s))"))
        ),
        ReasonCode::WheelMissing => "No matching official wheel exists.".to_owned(),
        ReasonCode::PythonWheelMissing => {
            "No wheel matches the selected Python version and ABI.".to_owned()
        }
        ReasonCode::PythonTagMatches => "Python tag matches.".to_owned(),
        ReasonCode::AbiTagMatches => "Python ABI tag matches.".to_owned(),
        ReasonCode::WheelYanked => "The wheel was yanked from the official index.".to_owned(),
        ReasonCode::PlatformTagMatches => "Platform tag matches.".to_owned(),
        ReasonCode::PlatformMismatch => "The wheel does not match this platform.".to_owned(),
        ReasonCode::GlibcTooOld => format!(
            "glibc {} is too old; requires {} or newer.",
            detail("detected").unwrap_or_else(|| "unknown".to_owned()),
            detail("required").unwrap_or_else(|| "a newer version".to_owned())
        ),
        ReasonCode::GlibcUnknown => "glibc compatibility could not be confirmed.".to_owned(),
        ReasonCode::DriverSupportsCudaFamily => format!(
            "The NVIDIA driver supports this CUDA family (minimum {}).",
            detail("minimum").unwrap_or_else(|| "unknown".to_owned())
        ),
        ReasonCode::DriverSupportsCudaVariant => format!(
            "The NVIDIA driver meets the normal CUDA variant minimum ({}).",
            detail("minimum").unwrap_or_else(|| "unknown".to_owned())
        ),
        ReasonCode::UsesCudaMinorCompatibility => format!(
            "Uses CUDA minor-version compatibility; the normal driver minimum is {}.",
            detail("normal_minimum").unwrap_or_else(|| "unknown".to_owned())
        ),
        ReasonCode::DriverTooOld => format!(
            "NVIDIA driver {} is too old; requires {} or newer.",
            detail("detected").unwrap_or_else(|| "unknown".to_owned()),
            detail("minimum").unwrap_or_else(|| "a newer driver".to_owned())
        ),
        ReasonCode::DriverUnknown => "NVIDIA driver compatibility is unknown.".to_owned(),
        ReasonCode::DriverNotRequired => {
            "A CPU wheel does not require an NVIDIA driver.".to_owned()
        }
        ReasonCode::GpuArchitectureSupported => {
            "The wheel supports every selected GPU architecture.".to_owned()
        }
        ReasonCode::GpuArchitectureUnsupported => format!(
            "The wheel does not support selected GPU architecture {}.",
            detail("capabilities").unwrap_or_else(|| "unknown".to_owned())
        ),
        ReasonCode::GpuArchitectureUnknown => {
            "Static GPU architecture coverage is not established for this exact wheel version."
                .to_owned()
        }
        ReasonCode::NvidiaGpuUnavailable => "No NVIDIA GPU is available for this wheel.".to_owned(),
        ReasonCode::VersionConstraintMatches => {
            "The requested version constraint matches.".to_owned()
        }
        ReasonCode::VersionConstraintMismatch => {
            "The requested version constraint does not match.".to_owned()
        }
        ReasonCode::StableRelease => "This is a stable PyTorch release.".to_owned(),
        ReasonCode::Prerelease => "Pre-release wheels are disabled by default.".to_owned(),
        ReasonCode::OfficialReleaseConfiguration => {
            "This is a reviewed PyTorch release configuration.".to_owned()
        }
        ReasonCode::NotOfficialReleaseConfiguration => {
            "The wheel is not part of the reviewed PyTorch release configuration.".to_owned()
        }
        ReasonCode::OfficialPreferenceUnknown => {
            "No reviewed release preference is available for this configuration.".to_owned()
        }
        ReasonCode::RuntimeVerified => "Runtime verification passed.".to_owned(),
        ReasonCode::RuntimeNotRun => "Runtime verification has not been run.".to_owned(),
        ReasonCode::RuntimeVerificationFailed => "Runtime verification failed.".to_owned(),
    }
}

fn driver_upgrade_hint(report: &RecommendationReport) -> Option<(&Candidate, &DecisionReason)> {
    let recommendation_version = report
        .recommendation
        .as_ref()
        .and_then(|candidate| pep440_rs::Version::from_str(&candidate.torch_version).ok());
    report.excluded.iter().find_map(|candidate| {
        if let Some(recommendation_version) = &recommendation_version {
            let Ok(candidate_version) = pep440_rs::Version::from_str(&candidate.torch_version)
            else {
                return None;
            };
            if candidate_version < *recommendation_version {
                return None;
            }
        }
        if !candidate.variant.is_cuda()
            || candidate.official_preference.is_none()
            || candidate.checks.wheel.status != CheckStatus::Pass
            || candidate.checks.python.status != CheckStatus::Pass
            || candidate.checks.platform.status != CheckStatus::Pass
            || candidate.checks.gpu_architecture.status != CheckStatus::Pass
        {
            return None;
        }
        failing_reasons(candidate)
            .find(|reason| reason.code == ReasonCode::DriverTooOld)
            .map(|reason| (candidate, reason))
    })
}

fn driver_upgrade_text(candidate: &Candidate, reason: &DecisionReason) -> String {
    let detected = terminal_text(
        reason
            .details
            .get("detected")
            .map_or("unknown", String::as_str),
    );
    let minimum = terminal_text(
        reason
            .details
            .get("minimum")
            .map_or("unknown", String::as_str),
    );
    format!(
        "PyTorch {} + {} requires NVIDIA driver {minimum} or newer (detected {detected}).",
        candidate.torch_version, candidate.variant
    )
}

fn diagnostic_text(diagnostic: &Diagnostic) -> String {
    let text = match diagnostic.code {
        DiagnosticCode::UnsupportedOperatingSystem => "The operating system is unsupported.",
        DiagnosticCode::UnsupportedArchitecture => "The CPU architecture is unsupported.",
        DiagnosticCode::UnsupportedLibc => "The host C library is unsupported.",
        DiagnosticCode::GlibcUnknown => "glibc could not be identified.",
        DiagnosticCode::PythonUnavailable => "No usable Python interpreter was found.",
        DiagnosticCode::UnsupportedPythonImplementation => {
            "The selected Python implementation is unsupported."
        }
        DiagnosticCode::BuiltinPythonTags => "Python tags use the conservative built-in fallback.",
        DiagnosticCode::NvidiaSmiUnavailable => "nvidia-smi was not found.",
        DiagnosticCode::NvidiaNoDevices => "NVIDIA reported no visible devices.",
        DiagnosticCode::NvidiaInspectionFailed => "NVIDIA inspection failed.",
        DiagnosticCode::NvidiaInspectionTimedOut => "NVIDIA inspection timed out.",
        DiagnosticCode::ComputeCapabilityUnknown => {
            "One or more GPU compute capabilities are unknown."
        }
        DiagnosticCode::InvalidGpuSelection => "The requested GPU selection is invalid.",
        DiagnosticCode::CudaVisibleDevicesSet => {
            "CUDA_VISIBLE_DEVICES changes the visible GPU mapping."
        }
        DiagnosticCode::InconsistentDriverVersions => {
            "Selected GPUs reported inconsistent driver versions."
        }
        DiagnosticCode::CudaToolkitNotFound => {
            "A local CUDA Toolkit was not found; official wheels do not require one."
        }
        DiagnosticCode::CudnnInspectionFailed => "Optional cuDNN inspection failed.",
    };
    format!("{text}{}", details_suffix(&diagnostic.details))
}

fn diagnostic_severity_name(severity: DiagnosticSeverity) -> &'static str {
    match severity {
        DiagnosticSeverity::Info => "Info",
        DiagnosticSeverity::Warning => "Warning",
        DiagnosticSeverity::Error => "Error",
    }
}

fn metadata_origin_name(origin: MetadataOrigin) -> &'static str {
    match origin {
        MetadataOrigin::Network => "downloaded",
        MetadataOrigin::FreshCache => "fresh cache",
        MetadataOrigin::OfflineCache => "offline cache",
        MetadataOrigin::StaleIfError => "stale fallback cache",
    }
}

fn human_duration(seconds: u64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    if days > 0 {
        if hours > 0 {
            format!("{days}d {hours}h")
        } else {
            format!("{days}d")
        }
    } else if hours > 0 {
        if minutes > 0 {
            format!("{hours}h {minutes}m")
        } else {
            format!("{hours}h")
        }
    } else if minutes > 0 {
        format!("{minutes}m")
    } else {
        format!("{seconds}s")
    }
}

fn python_implementation_name(implementation: &str) -> String {
    if implementation.eq_ignore_ascii_case("cpython") {
        "CPython".to_owned()
    } else {
        terminal_text(implementation)
    }
}

fn nvidia_status_text(status: NvidiaDetectionStatus) -> &'static str {
    match status {
        NvidiaDetectionStatus::Detected => "NVIDIA detected, but no selected GPU was returned.",
        NvidiaDetectionStatus::CommandUnavailable => "nvidia-smi is unavailable.",
        NvidiaDetectionStatus::NoDevices => "No NVIDIA GPU was detected.",
        NvidiaDetectionStatus::Failed => "NVIDIA inspection failed.",
        NvidiaDetectionStatus::TimedOut => "NVIDIA inspection timed out.",
    }
}

fn optional_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}

fn write_metadata_footer(output: &mut String, metadata: &MetadataInfo, width: usize) {
    let source = url::Url::parse(&metadata.source)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| terminal_text(&metadata.source));
    write_wrapped(
        output,
        "Metadata: ",
        "  ",
        &format!(
            "{} · {} old · {}",
            metadata_origin_name(metadata.origin),
            human_duration(metadata.age_seconds),
            terminal_text(&source)
        ),
        width,
    );
}

fn operating_system_name(os: &crate::core::OperatingSystem) -> &str {
    match os {
        crate::core::OperatingSystem::Linux => "linux",
        crate::core::OperatingSystem::Windows => "windows",
        crate::core::OperatingSystem::Macos => "macos",
        crate::core::OperatingSystem::Other(value) => value,
    }
}

fn optional(value: Option<&str>) -> &str {
    value.unwrap_or("unknown")
}

fn compatibility_name(status: CompatibilityStatus) -> &'static str {
    match status {
        CompatibilityStatus::Verified => "verified",
        CompatibilityStatus::DirectCompatible => "direct-compatible",
        CompatibilityStatus::MinorCompatible => "minor-compatible",
        CompatibilityStatus::Unverified => "unverified",
        CompatibilityStatus::Incompatible => "incompatible",
        CompatibilityStatus::Unavailable => "unavailable",
    }
}

fn check_status_name(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "pass",
        CheckStatus::Fail => "fail",
        CheckStatus::Unknown => "unknown",
        CheckStatus::NotApplicable => "not-applicable",
    }
}

fn json_name<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn details_suffix(details: &std::collections::BTreeMap<String, String>) -> String {
    if details.is_empty() {
        String::new()
    } else {
        format!(
            " ({})",
            details
                .iter()
                .map(|(key, value)| { format!("{}={}", terminal_text(key), terminal_text(value)) })
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn redact_environment(environment: &mut Environment) {
    let mut sensitive_values = Vec::new();
    if let Some(python) = &mut environment.python {
        sensitive_values.push(python.executable.to_string_lossy().into_owned());
        if let Some(virtual_environment) = &python.virtual_environment {
            sensitive_values.push(virtual_environment.to_string_lossy().into_owned());
        }
        python.executable = PathBuf::from("<redacted>");
        python.virtual_environment = python
            .virtual_environment
            .as_ref()
            .map(|_| PathBuf::from("<redacted>"));
    }
    for gpu in &mut environment.nvidia.gpus {
        gpu.uuid = None;
    }
    if let Some(toolkit) = &mut environment.cuda_toolkit {
        if let Some(root) = &toolkit.root {
            sensitive_values.push(root.to_string_lossy().into_owned());
        }
        toolkit.root = toolkit.root.as_ref().map(|_| PathBuf::from("<redacted>"));
    }
    for diagnostic in &mut environment.diagnostics {
        if matches!(
            diagnostic.code,
            DiagnosticCode::CudaVisibleDevicesSet | DiagnosticCode::PythonUnavailable
        ) {
            for value in diagnostic.details.values_mut() {
                *value = "<redacted>".to_owned();
            }
            continue;
        }
        for value in diagnostic.details.values_mut() {
            redact_text(value, &sensitive_values);
        }
    }
}

fn redact_recommendation(report: &mut RecommendationReport) {
    let original_python = report
        .environment
        .python
        .as_ref()
        .map(|python| python.executable.to_string_lossy().into_owned());
    redact_environment(&mut report.environment);
    if let Some(command) = &mut report.install {
        if let Some(original) = original_python {
            if command.program == original {
                command.program = "<redacted>".to_owned();
            }
            for argument in &mut command.args {
                if argument == &original {
                    *argument = "<redacted>".to_owned();
                }
            }
        }
        command.display = shell_display(&command.program, &command.args);
    }
    if let Some(candidate) = &mut report.recommendation {
        redact_candidate(candidate);
    }
    for candidate in &mut report.alternatives {
        redact_candidate(candidate);
    }
    for candidate in &mut report.excluded {
        redact_candidate(candidate);
    }
}

fn redact_candidate(_candidate: &mut Candidate) {
    // Candidate fields currently contain only public package metadata. Keep the hook so newly
    // added local fields must pass through an explicit redaction review.
}

fn redact_verification(report: &mut VerificationReport) {
    let mut sensitive_values = vec![report.python_executable.to_string_lossy().into_owned()];
    if let Some(root) = report
        .python_executable
        .parent()
        .and_then(std::path::Path::parent)
    {
        sensitive_values.push(root.to_string_lossy().into_owned());
    }
    report.python_executable = PathBuf::from("<redacted>");
    for mapping in &mut report.gpu_selection {
        mapping.uuid = None;
    }
    for check in &mut report.checks {
        if let Some(detail) = &mut check.detail {
            redact_text(detail, &sensitive_values);
        }
    }
    for diagnostic in &mut report.diagnostics {
        if matches!(
            diagnostic.code,
            DiagnosticCode::CudaVisibleDevicesSet | DiagnosticCode::PythonUnavailable
        ) {
            for value in diagnostic.details.values_mut() {
                *value = "<redacted>".to_owned();
            }
            continue;
        }
        for value in diagnostic.details.values_mut() {
            redact_text(value, &sensitive_values);
        }
    }
    if let Some(error) = &mut report.error {
        redact_text(error, &sensitive_values);
    }
}

fn redact_text(value: &mut String, sensitive_values: &[String]) {
    for sensitive in sensitive_values {
        if !sensitive.is_empty() {
            *value = value.replace(sensitive, "<redacted>");
        }
    }
}

fn shell_display(program: &str, args: &[String]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(|part| shell_escape::escape(part.into()).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

fn terminal_text(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if is_unsafe_terminal_character(character) {
            escaped.extend(character.escape_default());
        } else {
            escaped.push(character);
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn human_environment() -> Environment {
        let driver: crate::core::NumericVersion = "575.57.08".parse().expect("fixture driver");
        Environment {
            platform: crate::core::PlatformInfo {
                os: crate::core::OperatingSystem::Linux,
                architecture: crate::core::Architecture::X86_64,
                kernel_version: Some("5.19.0-32-generic".to_owned()),
                distribution: Some("Ubuntu 22.04.2 LTS".to_owned()),
            },
            glibc: Some("2.35".parse().expect("fixture glibc")),
            python: Some(crate::core::PythonInfo {
                executable: PathBuf::from("/usr/bin/python3"),
                implementation: "cpython".to_owned(),
                version: "3.10.12".parse().expect("fixture Python"),
                soabi: Some("cpython-310-x86_64-linux-gnu".to_owned()),
                cache_tag: Some("cpython-310".to_owned()),
                platform: "linux-x86_64".to_owned(),
                pointer_width: 64,
                free_threaded: false,
                virtual_environment: None,
                compatible_tags: vec!["cp310-cp310-manylinux_2_28_x86_64".to_owned()],
                tag_source: crate::core::TagSource::Packaging,
            }),
            nvidia: crate::core::NvidiaInfo {
                status: NvidiaDetectionStatus::Detected,
                driver_version: Some(driver.clone()),
                reported_cuda_version: Some("12.9".parse().expect("reported CUDA")),
                gpus: (0..4)
                    .map(|index| crate::core::NvidiaGpu {
                        index,
                        uuid: Some(format!("GPU-{index}")),
                        name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".to_owned(),
                        compute_capability: Some(crate::core::ComputeCapability {
                            major: 12,
                            minor: 0,
                        }),
                        driver_version: driver.clone(),
                    })
                    .collect(),
            },
            cuda_toolkit: Some(crate::core::CudaToolkitInfo {
                version: "12.1.1".parse().expect("fixture toolkit"),
                root: Some(PathBuf::from("/usr/local/cuda")),
                source: crate::core::ToolkitSource::Nvcc,
            }),
            diagnostics: Vec::new(),
        }
    }

    fn compatibility_check(
        status: CheckStatus,
        reasons: impl IntoIterator<Item = DecisionReason>,
    ) -> CompatibilityCheck {
        CompatibilityCheck::new(status, reasons.into_iter().collect())
    }

    fn human_candidate(
        version: &str,
        variant: &str,
        compatibility: CompatibilityStatus,
        official_preference: Option<u16>,
    ) -> Candidate {
        let variant = variant.parse().expect("fixture variant");
        let cuda = crate::core::CudaVariant::is_cuda(&variant);
        Candidate {
            torch_version: version.to_owned(),
            variant,
            compatibility,
            checks: crate::core::CompatibilityChecks {
                wheel: compatibility_check(
                    CheckStatus::Pass,
                    [DecisionReason::new(ReasonCode::WheelExists)],
                ),
                python: compatibility_check(
                    CheckStatus::Pass,
                    [DecisionReason::new(ReasonCode::PythonTagMatches)],
                ),
                platform: compatibility_check(
                    CheckStatus::Pass,
                    [DecisionReason::new(ReasonCode::PlatformTagMatches)],
                ),
                gpu_architecture: if cuda {
                    compatibility_check(
                        CheckStatus::Pass,
                        [DecisionReason::new(ReasonCode::GpuArchitectureSupported)],
                    )
                } else {
                    compatibility_check(CheckStatus::NotApplicable, [])
                },
                driver: if cuda {
                    compatibility_check(
                        CheckStatus::Pass,
                        [DecisionReason::new(ReasonCode::DriverSupportsCudaVariant)],
                    )
                } else {
                    compatibility_check(
                        CheckStatus::NotApplicable,
                        [DecisionReason::new(ReasonCode::DriverNotRequired)],
                    )
                },
                runtime: compatibility_check(
                    CheckStatus::Unknown,
                    [DecisionReason::new(ReasonCode::RuntimeNotRun)],
                ),
            },
            wheel: None,
            stable: true,
            official_preference,
            warnings: Vec::new(),
        }
    }

    fn hestia_report() -> RecommendationReport {
        let recommendation = human_candidate(
            "2.13.0",
            "cpu",
            CompatibilityStatus::DirectCompatible,
            Some(3),
        );
        let mut unverified =
            human_candidate("2.13.0", "cu129", CompatibilityStatus::Unverified, None);
        unverified.checks.wheel.reasons.push(DecisionReason::new(
            ReasonCode::NotOfficialReleaseConfiguration,
        ));
        unverified.checks.gpu_architecture = compatibility_check(
            CheckStatus::Unknown,
            [DecisionReason::new(ReasonCode::GpuArchitectureUnknown)],
        );
        unverified.warnings = vec![
            CandidateWarning::new(WarningCode::GpuArchitectureUnknown),
            CandidateWarning::new(WarningCode::NotOfficialReleaseConfiguration),
        ];

        let mut requires_driver = human_candidate(
            "2.13.0",
            "cu130",
            CompatibilityStatus::Incompatible,
            Some(0),
        );
        requires_driver.checks.driver = compatibility_check(
            CheckStatus::Fail,
            [DecisionReason::new(ReasonCode::DriverTooOld)
                .with_detail("detected", "575.57.08")
                .with_detail("minimum", "580.65.06")],
        );

        RecommendationReport {
            schema_version: SCHEMA_VERSION,
            environment: human_environment(),
            metadata: MetadataInfo {
                origin: MetadataOrigin::FreshCache,
                fetched_at: 1,
                age_seconds: 27_330,
                stale: false,
                source: "https://download.pytorch.org/whl/".to_owned(),
            },
            recommendation: Some(recommendation),
            alternatives: vec![unverified],
            excluded: vec![requires_driver],
            install: Some(CommandSpec {
                program: "/usr/bin/python3".to_owned(),
                args: vec![
                    "-m".to_owned(),
                    "pip".to_owned(),
                    "install".to_owned(),
                    "--isolated".to_owned(),
                    "--index-url".to_owned(),
                    "https://download.pytorch.org/whl/cpu".to_owned(),
                    "torch==2.13.0".to_owned(),
                ],
                display: String::new(),
            }),
        }
    }

    #[test]
    fn exact_requirement_parser_accepts_documented_form() {
        assert_eq!(
            exact_torch_version("torch==2.12.1").expect("exact version"),
            "2.12.1"
        );
        assert_eq!(exact_torch_version("2.12").expect("exact version"), "2.12");
        assert_eq!(
            exact_torch_version("v2.12.0").expect("normalized exact version"),
            "2.12.0"
        );
    }

    #[test]
    fn exact_requirement_parser_rejects_ranges() {
        assert!(exact_torch_version("torch>=2.10").is_err());
        assert!(exact_torch_version("torch==2.10,!=2.10.1").is_err());
    }

    #[test]
    fn companion_set_is_deduplicated() {
        assert_eq!(
            companion_set(&[
                CompanionPackage::Torchvision,
                CompanionPackage::Torchvision,
                CompanionPackage::Torchaudio,
            ])
            .len(),
            2
        );
    }

    #[test]
    fn cli_contract_is_constructible() {
        Cli::command().debug_assert();
    }

    #[test]
    fn candidate_visibility_modes_are_explicit_and_mutually_exclusive() {
        let default =
            Cli::try_parse_from(["torch-check", "candidates"]).expect("default candidates mode");
        let Some(Commands::Candidates(default)) = default.command else {
            panic!("candidates subcommand");
        };
        assert!(!default.unverified);
        assert!(!default.all);

        let including_unverified =
            Cli::try_parse_from(["torch-check", "candidates", "--unverified"])
                .expect("unverified candidates mode");
        let Some(Commands::Candidates(including_unverified)) = including_unverified.command else {
            panic!("candidates subcommand");
        };
        assert!(including_unverified.unverified);
        assert!(!including_unverified.all);

        let all = Cli::try_parse_from(["torch-check", "candidates", "--all"])
            .expect("all candidates mode");
        let Some(Commands::Candidates(all)) = all.command else {
            panic!("candidates subcommand");
        };
        assert!(!all.unverified);
        assert!(all.all);

        let conflict = Cli::try_parse_from(["torch-check", "candidates", "--unverified", "--all"])
            .expect_err("candidate visibility modes must not be combined");
        assert_eq!(conflict.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn candidate_visibility_filters_statuses_for_human_and_json_output() {
        let default = CandidatesArgs::default();
        let including_unverified = CandidatesArgs {
            unverified: true,
            ..CandidatesArgs::default()
        };
        let all = CandidatesArgs {
            all: true,
            ..CandidatesArgs::default()
        };

        for status in [
            CompatibilityStatus::Verified,
            CompatibilityStatus::DirectCompatible,
            CompatibilityStatus::MinorCompatible,
        ] {
            let candidate = human_candidate("2.10.0", "cu128", status, Some(0));
            assert!(candidate_visible_in_candidates(&candidate, &default));
            assert!(candidate_visible_in_candidates(
                &candidate,
                &including_unverified
            ));
            assert!(candidate_visible_in_candidates(&candidate, &all));
        }

        let unverified =
            human_candidate("2.10.0", "cu128", CompatibilityStatus::Unverified, Some(0));
        assert!(!candidate_visible_in_candidates(&unverified, &default));
        assert!(candidate_visible_in_candidates(
            &unverified,
            &including_unverified
        ));
        assert!(candidate_visible_in_candidates(&unverified, &all));

        let index_only_direct = human_candidate(
            "2.10.0",
            "cu129",
            CompatibilityStatus::DirectCompatible,
            None,
        );
        assert!(!candidate_visible_in_candidates(
            &index_only_direct,
            &default
        ));
        assert!(candidate_visible_in_candidates(
            &index_only_direct,
            &including_unverified
        ));
        assert!(candidate_visible_in_candidates(&index_only_direct, &all));

        for status in [
            CompatibilityStatus::Incompatible,
            CompatibilityStatus::Unavailable,
        ] {
            let candidate = human_candidate("2.10.0", "cu126", status, Some(1));
            assert!(!candidate_visible_in_candidates(&candidate, &default));
            assert!(!candidate_visible_in_candidates(
                &candidate,
                &including_unverified
            ));
            assert!(candidate_visible_in_candidates(&candidate, &all));
        }
    }

    #[test]
    fn bare_recommendation_options_insert_the_recommend_alias() {
        let arguments = prepare_arguments(
            [
                "torch-check",
                "--format=json",
                "--installer",
                "uv",
                "--with",
                "torchvision",
            ]
            .into_iter()
            .map(OsString::from)
            .collect(),
        );
        let cli = Cli::try_parse_from(arguments).expect("bare recommend options should parse");
        let Some(Commands::Recommend(args)) = cli.command else {
            panic!("recommend subcommand should be inserted");
        };
        assert_eq!(args.installer, InstallerArg::Uv);
        assert_eq!(args.companions, vec![CompanionPackage::Torchvision]);
    }

    #[test]
    fn explicit_subcommands_are_not_rewritten() {
        let arguments = ["torch-check", "--redact", "inspect"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert_eq!(prepare_arguments(arguments.clone()), arguments);
    }

    #[test]
    fn explicit_gpu_verification_uses_full_uuids_and_zero_based_logical_indices() {
        let cli =
            Cli::try_parse_from(["torch-check", "--gpu", "2,0", "verify"]).expect("valid CLI");
        let mut environment = crate::core::Environment {
            platform: crate::core::PlatformInfo {
                os: crate::core::OperatingSystem::Linux,
                architecture: crate::core::Architecture::X86_64,
                kernel_version: None,
                distribution: None,
            },
            glibc: None,
            python: None,
            nvidia: crate::core::NvidiaInfo {
                status: crate::core::NvidiaDetectionStatus::Detected,
                driver_version: None,
                reported_cuda_version: None,
                gpus: Vec::new(),
            },
            cuda_toolkit: None,
            diagnostics: Vec::new(),
        };
        for (index, uuid) in [(0, "GPU-aaaaaaaa"), (2, "GPU-bbbbbbbb")] {
            environment.nvidia.gpus.push(crate::core::NvidiaGpu {
                index,
                uuid: Some(uuid.to_owned()),
                name: format!("GPU {index}"),
                compute_capability: None,
                driver_version: crate::core::NumericVersion::from_str("580.65.06")
                    .expect("version"),
            });
        }

        let selection = verification_gpu_selection(&cli, &environment).expect("safe selection");
        assert_eq!(selection.device_indices, Some(vec![0, 1]));
        assert_eq!(
            selection.cuda_visible_devices.as_deref(),
            Some("GPU-aaaaaaaa,GPU-bbbbbbbb")
        );
        assert_eq!(selection.mappings[0].physical_index, 0);
        assert_eq!(selection.mappings[0].logical_index, 0);
        assert_eq!(selection.mappings[1].physical_index, 2);
        assert_eq!(selection.mappings[1].logical_index, 1);
    }

    #[test]
    fn identical_gpus_are_grouped_without_hiding_indices_or_driver_formatting() {
        let mut environment = human_environment();
        environment.nvidia.gpus[1].driver_version =
            "575.57.08.0".parse().expect("equivalent driver");
        let groups = group_gpus(&environment);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].indices, vec![0, 1, 2, 3]);
        assert_eq!(groups[0].driver.to_string(), "575.57.08");
        assert_eq!(format_gpu_indices(&groups[0].indices), "0–3");
        assert_eq!(format_gpu_indices(&[0, 1, 3, 5, 6]), "0–1, 3, 5–6");
    }

    #[test]
    fn hestia_human_output_is_compact_actionable_and_free_of_raw_codes() {
        let report = hestia_report();
        let rendered = render_recommendation(&report, 80);

        for expected in [
            "4× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "GPUs 0–3",
            "informational for wheels",
            "PyTorch 2.13.0 + cpu (direct-compatible)",
            "CPU fallback",
            "No reviewed CUDA candidate passed all static checks.",
            "Needs review or verification",
            "not in the reviewed PyTorch release",
            "Driver upgrade:",
            "580.65.06",
            "newer (detected 575.57.08)",
            "7h 35m old",
            "torch-check candidates --all",
        ] {
            assert!(
                rendered.contains(expected),
                "missing `{expected}`:\n{rendered}"
            );
        }
        for raw_code in [
            "gpu_architecture_unknown",
            "not_official_release_configuration",
            "warning:",
        ] {
            assert!(
                !rendered.contains(raw_code),
                "raw diagnostic code leaked: {raw_code}\n{rendered}"
            );
        }
        assert!(
            rendered.contains(" \\\n"),
            "install command was not wrapped"
        );
        assert!(
            rendered.lines().all(|line| display_width(line) <= 80),
            "rendered line exceeded width:\n{rendered}"
        );
    }

    #[test]
    fn nvidia_summary_reserves_gpu_alternatives_and_one_cpu_fallback() {
        let mut report = hestia_report();
        report.recommendation = Some(human_candidate(
            "2.11.0",
            "cu128",
            CompatibilityStatus::DirectCompatible,
            Some(0),
        ));
        report.alternatives = vec![
            human_candidate(
                "2.13.0",
                "cpu",
                CompatibilityStatus::DirectCompatible,
                Some(3),
            ),
            human_candidate(
                "2.12.1",
                "cpu",
                CompatibilityStatus::DirectCompatible,
                Some(3),
            ),
            human_candidate(
                "2.10.0",
                "cu128",
                CompatibilityStatus::DirectCompatible,
                Some(0),
            ),
            human_candidate(
                "2.9.1",
                "cu128",
                CompatibilityStatus::DirectCompatible,
                Some(0),
            ),
            human_candidate(
                "2.11.0",
                "cpu",
                CompatibilityStatus::DirectCompatible,
                Some(3),
            ),
        ];
        let install = report.install.as_mut().expect("install command");
        install.args[5] = "https://download.pytorch.org/whl/cu128".to_owned();
        install.args[6] = "torch==2.11.0".to_owned();

        let rendered = render_recommendation(&report, 80);
        for expected in [
            "GPU alternatives",
            "PyTorch 2.10.0 + cu128 (direct-compatible)",
            "PyTorch 2.9.1 + cu128 (direct-compatible)",
            "CPU fallback",
            "PyTorch 2.13.0 + cpu (direct-compatible)",
            "Driver upgrade:",
            "PyTorch 2.13.0 + cu130",
        ] {
            assert!(
                rendered.contains(expected),
                "missing `{expected}`:\n{rendered}"
            );
        }
        for hidden_cpu in ["PyTorch 2.12.1 + cpu", "PyTorch 2.11.0 + cpu"] {
            assert!(
                !rendered.contains(hidden_cpu),
                "extra CPU fallback consumed a semantic slot: {hidden_cpu}\n{rendered}"
            );
        }
        assert!(
            rendered.find("GPU alternatives") < rendered.find("CPU fallback"),
            "GPU alternatives must precede the CPU fallback:\n{rendered}"
        );
    }

    #[test]
    fn needs_review_prioritizes_reviewed_cuda_over_newer_index_only_wheels() {
        let mut report = hestia_report();
        let newest_index_only = report.alternatives.remove(0);
        let mut second_index_only = newest_index_only.clone();
        second_index_only.torch_version = "2.12.1".to_owned();
        let mut reviewed_cuda =
            human_candidate("2.10.0", "cu128", CompatibilityStatus::Unverified, Some(0));
        reviewed_cuda.checks.gpu_architecture = compatibility_check(
            CheckStatus::Unknown,
            [DecisionReason::new(ReasonCode::GpuArchitectureUnknown)],
        );
        reviewed_cuda.warnings = vec![CandidateWarning::new(WarningCode::GpuArchitectureUnknown)];
        report.alternatives = vec![newest_index_only, second_index_only, reviewed_cuda];

        let rendered = render_recommendation(&report, 80);
        let reviewed_position = rendered
            .find("PyTorch 2.10.0 + cu128")
            .expect("reviewed CUDA candidate");
        let index_only_position = rendered
            .find("PyTorch 2.13.0 + cu129")
            .expect("index-only CUDA candidate");
        assert!(
            reviewed_position < index_only_position,
            "reviewed CUDA must be shown first even when it is older:\n{rendered}"
        );
        assert!(
            !rendered.contains("PyTorch 2.12.1 + cu129"),
            "the semantic review limit should remain concise:\n{rendered}"
        );
        assert!(
            rendered.contains("… 1 more candidate(s) requiring review or verification."),
            "hidden review candidates must be counted:\n{rendered}"
        );
    }

    #[test]
    fn wrapped_human_text_uses_terminal_cell_width() {
        let mut rendered = String::new();
        write_wrapped(
            &mut rendered,
            "  ",
            "    ",
            &format!("GPU {}", "画像処理装置".repeat(12)),
            44,
        );
        assert!(
            rendered.lines().all(|line| display_width(line) <= 44),
            "wide text exceeded terminal width:\n{rendered}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn long_shell_tokens_wrap_without_changing_argument_boundaries() {
        let long_argument = format!(
            "/tmp/{}",
            "日本語 path with 'quotes' and $dollars\\slashes/".repeat(5)
        );
        let command = CommandSpec {
            program: "printf".to_owned(),
            args: vec!["%s".to_owned(), long_argument.clone()],
            display: String::new(),
        };
        let mut rendered = String::new();
        write_command(&mut rendered, &command, 44);

        assert!(
            rendered.lines().all(|line| display_width(line) <= 44),
            "shell command exceeded terminal width:\n{rendered}"
        );
        assert!(
            rendered.lines().any(|line| !line.starts_with(' ')),
            "an intra-token continuation must begin in column zero:\n{rendered}"
        );
        let result = std::process::Command::new("/bin/sh")
            .args(["-c", rendered.as_str()])
            .output()
            .expect("run rendered POSIX shell command");
        assert!(
            result.status.success(),
            "rendered command failed: {}\n{rendered}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(result.stdout, long_argument.as_bytes());

        let long_program = CommandSpec {
            program: format!("python-{}", "x".repeat(100)),
            args: vec!["--version".to_owned()],
            display: String::new(),
        };
        let mut rendered_program = String::new();
        write_command(&mut rendered_program, &long_program, 44);
        assert!(
            rendered_program
                .lines()
                .all(|line| display_width(line) <= 44),
            "long program exceeded terminal width:\n{rendered_program}"
        );
        let syntax = std::process::Command::new("/bin/sh")
            .args(["-n", "-c", rendered_program.as_str()])
            .status()
            .expect("parse rendered POSIX shell command");
        assert!(syntax.success());
    }

    #[test]
    fn terminal_text_escapes_bidi_and_unicode_line_controls() {
        let rendered =
            terminal_text("left\u{00a0}\u{061c}middle\u{1680}\u{2028}next\u{2029}\u{3000}right");
        for unsafe_character in [
            '\u{00a0}', '\u{061c}', '\u{1680}', '\u{2028}', '\u{2029}', '\u{3000}',
        ] {
            assert!(!rendered.contains(unsafe_character));
        }
        for escaped in [
            "\\u{a0}",
            "\\u{61c}",
            "\\u{1680}",
            "\\u{2028}",
            "\\u{2029}",
            "\\u{3000}",
        ] {
            assert!(rendered.contains(escaped), "missing {escaped}: {rendered}");
        }
        assert_eq!(terminal_text(&rendered), rendered);
        assert_eq!(
            terminal_text("ASCII space · 日本語 🙂"),
            "ASCII space · 日本語 🙂"
        );
    }

    #[test]
    fn nvidia_cpu_fallback_returns_warning_exit_status() {
        let report = hestia_report();
        assert!(is_nvidia_cpu_fallback(&report));
        assert_eq!(recommendation_exit_code(&report), ExitCode::Warning);
    }

    #[test]
    fn candidates_with_only_unverified_results_return_warning_not_incompatible() {
        let report = hestia_report();
        let candidates = CandidatesReport {
            schema_version: SCHEMA_VERSION,
            environment: report.environment.clone(),
            metadata: report.metadata.clone(),
            candidates: report.alternatives.clone(),
        };
        assert_eq!(candidates_exit_code(&candidates), ExitCode::Warning);

        let excluded_only = CandidatesReport {
            candidates: report.excluded,
            ..candidates
        };
        assert_eq!(candidates_exit_code(&excluded_only), ExitCode::Incompatible);
    }

    #[test]
    fn unreviewed_direct_alternative_keeps_its_warning_visible() {
        let mut report = hestia_report();
        let mut future_cpu =
            human_candidate("2.14.0", "cpu", CompatibilityStatus::DirectCompatible, None);
        future_cpu.warnings = vec![CandidateWarning::new(
            WarningCode::OfficialPreferenceUnknown,
        )];
        report.recommendation = None;
        report.install = None;
        report.alternatives = vec![future_cpu];
        report.excluded.clear();

        let rendered = render_recommendation(&report, 80);
        assert!(rendered.contains("No recommendation-eligible reviewed configuration"));
        assert!(rendered.contains("constraints."));
        assert!(rendered.contains("Needs review or verification"));
        assert!(rendered.contains("No reviewed PyTorch release preference is available"));
        assert!(!rendered.contains("Other options"));
    }

    #[test]
    fn bacchus_human_output_keeps_the_reviewed_minor_compatible_gpu_choice() {
        let mut report = hestia_report();
        report.environment.nvidia.driver_version = Some("530.30.02".parse().expect("driver"));
        report.environment.nvidia.reported_cuda_version = Some("12.1".parse().expect("CUDA"));
        report.environment.nvidia.gpus = (0..8)
            .map(|index| crate::core::NvidiaGpu {
                index,
                uuid: Some(format!("GPU-{index}")),
                name: "NVIDIA RTX A6000".to_owned(),
                compute_capability: Some(crate::core::ComputeCapability { major: 8, minor: 6 }),
                driver_version: "530.30.02".parse().expect("driver"),
            })
            .collect();
        let mut recommendation = human_candidate(
            "2.13.0",
            "cu126",
            CompatibilityStatus::MinorCompatible,
            Some(1),
        );
        recommendation.warnings = vec![CandidateWarning::new(
            WarningCode::PtxOrNewDriverFeatureMayFail,
        )];
        report.recommendation = Some(recommendation);
        report.alternatives = vec![human_candidate(
            "2.13.0",
            "cpu",
            CompatibilityStatus::DirectCompatible,
            Some(3),
        )];
        report.excluded.clear();
        report.install.as_mut().expect("install").args = vec![
            "-m".to_owned(),
            "pip".to_owned(),
            "install".to_owned(),
            "--isolated".to_owned(),
            "--index-url".to_owned(),
            "https://download.pytorch.org/whl/cu126".to_owned(),
            "torch==2.13.0".to_owned(),
        ];

        let rendered = render_recommendation(&report, 80);
        assert!(rendered.contains("8× NVIDIA RTX A6000"));
        assert!(rendered.contains("PyTorch 2.13.0 + cu126 (minor-compatible)"));
        assert!(rendered.contains("PTX JIT or features that require a newer driver may fail."));
        assert!(rendered.contains("CPU fallback"));
        assert!(rendered.contains("PyTorch 2.13.0 + cpu (direct-compatible)"));
        assert!(!rendered.contains("No reviewed CUDA candidate passed all static checks."));
        assert_eq!(recommendation_exit_code(&report), ExitCode::Warning);
    }

    #[test]
    fn human_duration_uses_two_highest_useful_units() {
        assert_eq!(human_duration(27_330), "7h 35m");
        assert_eq!(human_duration(90_000), "1d 1h");
        assert_eq!(human_duration(30), "30s");
    }

    #[test]
    fn tty_styles_distinguish_sections_and_candidate_states() {
        assert_eq!(human_style_code("Recommendation"), Some("1;36"));
        assert_eq!(human_style_code("GPU alternatives"), Some("1;36"));
        assert_eq!(human_style_code("CPU fallback"), Some("1;36"));
        assert_eq!(
            human_style_code("  PyTorch 2.13.0 + cpu (direct-compatible)"),
            Some("32")
        );
        assert_eq!(
            human_style_code("  PyTorch 2.13.0 + cu129 (unverified)"),
            Some("33")
        );
        assert_eq!(human_style_code("  ordinary detail"), None);
    }
}
