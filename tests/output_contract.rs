//! Integration tests for the versioned machine-readable output contract.

use std::path::PathBuf;

use serde::Serialize;
use torch_check::core::{
    Architecture, Candidate, CandidatesReport, CheckStatus, CommandSpec, CompatibilityCheck,
    CompatibilityChecks, CompatibilityStatus, CudaVariant, DecisionReason, Environment, ErrorBody,
    ErrorKind, ErrorReport, ExplainReport, InspectReport, Installer, MetadataInfo, MetadataOrigin,
    NumericVersion, NvidiaDetectionStatus, NvidiaInfo, OperatingSystem, PlatformInfo, PythonInfo,
    ReasonCode, RecommendationReport, SCHEMA_VERSION, TagSource, TorchWheel, VerificationCheck,
    VerificationReport,
};

fn version(value: &str) -> NumericVersion {
    value.parse().expect("fixture version must be valid")
}

fn environment() -> Environment {
    Environment {
        platform: PlatformInfo {
            os: OperatingSystem::Linux,
            architecture: Architecture::X86_64,
            kernel_version: Some("6.8.0".to_owned()),
            distribution: Some("Fixture Linux".to_owned()),
        },
        glibc: Some(version("2.28")),
        python: Some(PythonInfo {
            executable: PathBuf::from("/opt/fixture/bin/python"),
            implementation: "cpython".to_owned(),
            version: version("3.12.8"),
            soabi: Some("cpython-312-x86_64-linux-gnu".to_owned()),
            cache_tag: Some("cpython-312".to_owned()),
            platform: "linux-x86_64".to_owned(),
            pointer_width: 64,
            free_threaded: false,
            virtual_environment: Some(PathBuf::from("/opt/fixture")),
            compatible_tags: vec![
                "cp312-cp312-manylinux_2_28_x86_64".to_owned(),
                "py3-none-any".to_owned(),
            ],
            tag_source: TagSource::Packaging,
        }),
        nvidia: NvidiaInfo {
            status: NvidiaDetectionStatus::NoDevices,
            driver_version: None,
            reported_cuda_version: None,
            gpus: Vec::new(),
        },
        cuda_toolkit: None,
        diagnostics: Vec::new(),
    }
}

fn check(status: CheckStatus, reason: ReasonCode) -> CompatibilityCheck {
    CompatibilityCheck::new(status, vec![DecisionReason::new(reason)])
}

fn candidate() -> Candidate {
    Candidate {
        torch_version: "2.6.0".to_owned(),
        variant: CudaVariant::Cpu,
        compatibility: CompatibilityStatus::DirectCompatible,
        checks: CompatibilityChecks {
            wheel: check(CheckStatus::Pass, ReasonCode::WheelExists),
            python: check(CheckStatus::Pass, ReasonCode::PythonTagMatches),
            platform: check(CheckStatus::Pass, ReasonCode::PlatformTagMatches),
            gpu_architecture: check(
                CheckStatus::NotApplicable,
                ReasonCode::GpuArchitectureSupported,
            ),
            driver: check(CheckStatus::NotApplicable, ReasonCode::DriverNotRequired),
            runtime: check(CheckStatus::Unknown, ReasonCode::RuntimeNotRun),
        },
        wheel: Some(TorchWheel {
            package: "torch".to_owned(),
            filename: "torch-2.6.0-cp312-cp312-manylinux_2_28_x86_64.whl".to_owned(),
            version: "2.6.0".to_owned(),
            public_version: "2.6.0".to_owned(),
            variant: CudaVariant::Cpu,
            python_tags: vec!["cp312".to_owned()],
            abi_tags: vec!["cp312".to_owned()],
            platform_tags: vec!["manylinux_2_28_x86_64".to_owned()],
            url: "https://download.pytorch.org/whl/cpu/torch-2.6.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=0000000000000000000000000000000000000000000000000000000000000000".to_owned(),
            sha256: Some("0".repeat(64)),
            yanked: false,
            requires_python: Some(">=3.9".to_owned()),
        }),
        stable: true,
        official_preference: Some(0),
        warnings: Vec::new(),
    }
}

fn metadata() -> MetadataInfo {
    MetadataInfo {
        origin: MetadataOrigin::FreshCache,
        fetched_at: 1_750_000_000,
        age_seconds: 30,
        stale: false,
        source: "https://download.pytorch.org/whl/".to_owned(),
    }
}

fn assert_matches_schema<T: Serialize>(name: &str, report: &T) {
    let schema = serde_json::from_str(include_str!("../data/schemas/output-v1.schema.json"))
        .expect("output schema must be valid JSON");
    let validator = jsonschema::validator_for(&schema).expect("output schema must compile");
    let instance = serde_json::to_value(report).expect("report must serialize");
    let errors = validator
        .iter_errors(&instance)
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
    assert!(
        errors.is_empty(),
        "{name} violated output-v1.schema.json:\n{}\ninstance: {}",
        errors.join("\n"),
        serde_json::to_string_pretty(&instance).expect("pretty JSON")
    );
}

#[test]
fn every_json_envelope_conforms_to_the_published_schema() {
    let selected = candidate();
    let inspect = InspectReport {
        schema_version: SCHEMA_VERSION,
        environment: environment(),
    };
    let recommendation = RecommendationReport {
        schema_version: SCHEMA_VERSION,
        environment: environment(),
        metadata: metadata(),
        recommendation: Some(selected.clone()),
        alternatives: Vec::new(),
        excluded: Vec::new(),
        install: Some(CommandSpec {
            program: "/opt/fixture/bin/python".to_owned(),
            args: vec![
                "-m".to_owned(),
                "pip".to_owned(),
                "install".to_owned(),
                "torch==2.6.0".to_owned(),
                "--index-url".to_owned(),
                "https://download.pytorch.org/whl/cpu".to_owned(),
            ],
            display: "/opt/fixture/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu".to_owned(),
        }),
    };
    let candidates = CandidatesReport {
        schema_version: SCHEMA_VERSION,
        environment: environment(),
        metadata: metadata(),
        candidates: vec![selected.clone()],
    };
    let explain = ExplainReport {
        schema_version: SCHEMA_VERSION,
        environment: environment(),
        metadata: metadata(),
        candidate: selected,
    };
    let verification = VerificationReport {
        schema_version: SCHEMA_VERSION,
        python_executable: PathBuf::from("/opt/fixture/bin/python"),
        status: CompatibilityStatus::Verified,
        torch_version: Some("2.6.0".to_owned()),
        compiled_cuda: None,
        cuda_available: Some(false),
        device_count: Some(0),
        arch_list: Vec::new(),
        devices: Vec::new(),
        gpu_selection: Vec::new(),
        cudnn_available: None,
        checks: vec![VerificationCheck {
            name: "torch_import".to_owned(),
            passed: true,
            detail: Some("imported torch 2.6.0".to_owned()),
        }],
        diagnostics: Vec::new(),
        error: None,
    };
    let error = ErrorReport {
        schema_version: SCHEMA_VERSION,
        error: ErrorBody {
            kind: ErrorKind::Metadata,
            code: "offline_cache_unavailable".to_owned(),
            message: "offline cache is unavailable".to_owned(),
        },
    };

    assert_matches_schema("inspect", &inspect);
    assert_matches_schema("recommendation", &recommendation);
    assert_matches_schema("candidates", &candidates);
    assert_matches_schema("explain", &explain);
    assert_matches_schema("verification", &verification);
    assert_matches_schema("error", &error);
}

#[test]
fn recommendation_json_contract_is_snapshotted() {
    let report = RecommendationReport {
        schema_version: SCHEMA_VERSION,
        environment: environment(),
        metadata: metadata(),
        recommendation: Some(candidate()),
        alternatives: Vec::new(),
        excluded: Vec::new(),
        install: None,
    };

    insta::assert_json_snapshot!(report);
}

#[test]
fn user_facing_copy_does_not_contain_known_misleading_claims() {
    let user_facing_sources =
        [include_str!("../src/cli.rs"), include_str!("../README.md")].join("\n");
    for forbidden in [
        "CUDA 12.1までしか使えません",
        "cu124 requires local CUDA Toolkit 12.4",
        "CUDA toolkit not found, so CUDA PyTorch cannot be installed",
    ] {
        assert!(
            !user_facing_sources.contains(forbidden),
            "misleading output phrase is forbidden: {forbidden}"
        );
    }
}

#[test]
fn installer_enum_remains_serializable_for_downstream_consumers() {
    assert_eq!(
        serde_json::to_value(Installer::UvAdd).expect("serialize installer"),
        "uv-add"
    );
}
