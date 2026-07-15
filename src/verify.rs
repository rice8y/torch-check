//! Dynamic validation of the PyTorch installation in a selected Python environment.
//!
//! Verification imports and executes the installed `torch` package in a child process. Only run
//! it against Python environments whose installed code you trust.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use wait_timeout::ChildExt;

use crate::core::{
    CompatibilityStatus, ComputeCapability, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    SCHEMA_VERSION, VerificationCheck, VerificationReport, VerifiedDevice,
};
use crate::process::{
    ReaderThread, ReaderWaitError, isolate_process_tree, output_drain_timeout, spawn_reader_thread,
    terminate_process_group, terminate_process_tree,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PROBE_PROTOCOL_VERSION: u32 = 1;
const PROBE_PREFIX: &[u8] = b"TORCH_CHECK_VERIFY_JSON:";

const PROBE_SCRIPT: &str = r#"
import json
import sys

PREFIX = "TORCH_CHECK_VERIFY_JSON:"


def emit(payload):
    print(PREFIX + json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)


def error_text(exc):
    try:
        value = str(exc)
    except BaseException:
        value = "exception could not be rendered"
    return value[:2048]


payload = {
    "protocol_version": 1,
    "ok": False,
    "stage": "arguments",
    "torch_version": None,
    "compiled_cuda": None,
    "cuda_available": None,
    "device_count": None,
    "arch_list": [],
    "devices": [],
    "cudnn_available": None,
    "cudnn_enabled": None,
    "cudnn_version": None,
    "cudnn_error": None,
    "failed_device_index": None,
    "error_type": None,
    "error": None,
}

try:
    selected = json.loads(sys.argv[1])
    payload["stage"] = "import_torch"
    import torch

    payload["torch_version"] = str(torch.__version__)
    raw_cuda = torch.version.cuda
    payload["compiled_cuda"] = None if raw_cuda is None else str(raw_cuda)

    payload["stage"] = "cuda_availability"
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["device_count"] = int(torch.cuda.device_count())

    if payload["compiled_cuda"] is None:
        if selected is not None:
            raise RuntimeError(
                "explicit CUDA devices were requested, but this is a CPU-only PyTorch build"
            )
        if payload["cuda_available"]:
            raise RuntimeError(
                "an accelerator is available, but this PyTorch build does not report a CUDA runtime"
            )
        payload["stage"] = "cudnn_info"
    elif not payload["cuda_available"]:
        raise RuntimeError(
            "this PyTorch build includes CUDA, but torch.cuda.is_available() is false"
        )
    else:
        payload["stage"] = "architecture_list"
        payload["arch_list"] = [str(item) for item in torch.cuda.get_arch_list()]

        count = payload["device_count"]
        if count <= 0:
            raise RuntimeError("CUDA is available, but PyTorch reported no logical CUDA devices")

        if selected is None:
            selected = list(range(count))
        if not isinstance(selected, list) or not selected:
            raise ValueError("selected logical CUDA device list must be non-empty")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
            raise ValueError("logical CUDA device indices must be integers")
        if len(set(selected)) != len(selected):
            raise ValueError("logical CUDA device indices must be unique")
        if any(index < 0 or index >= count for index in selected):
            raise IndexError("selected logical CUDA device index is out of range")

        for index in selected:
            payload["stage"] = "device_metadata"
            payload["failed_device_index"] = index
            name = str(torch.cuda.get_device_name(index))
            capability = torch.cuda.get_device_capability(index)

            payload["stage"] = "device_operations"
            with torch.cuda.device(index):
                device = torch.device("cuda:" + str(index))

                allocation = torch.empty((8, 8), dtype=torch.float32, device=device)
                allocation.fill_(1.0)

                elementwise = allocation.mul(2.0).add(1.0)
                expected_elementwise = torch.full_like(elementwise, 3.0)
                elementwise_ok = bool(torch.equal(elementwise, expected_elementwise))

                left = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
                identity = torch.eye(8, dtype=torch.float32, device=device)
                product = left @ identity
                matmul_ok = bool(torch.equal(product, left))

                torch.cuda.synchronize(index)

            if not elementwise_ok:
                raise RuntimeError("deterministic elementwise CUDA validation failed")
            if not matmul_ok:
                raise RuntimeError("deterministic CUDA matrix multiplication validation failed")

            payload["devices"].append(
                {
                    "index": index,
                    "name": name,
                    "capability": [int(capability[0]), int(capability[1])],
                    "operations_ok": True,
                }
            )

        payload["failed_device_index"] = None
        payload["stage"] = "cudnn_info"

    try:
        payload["cudnn_available"] = bool(torch.backends.cudnn.is_available())
        payload["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        raw_cudnn_version = torch.backends.cudnn.version()
        payload["cudnn_version"] = (
            None if raw_cudnn_version is None else int(raw_cudnn_version)
        )
    except BaseException as exc:
        # cuDNN is informative for this command and does not gate basic CUDA verification.
        payload["cudnn_error"] = error_text(exc)

    payload["stage"] = "complete"
    payload["ok"] = True
    emit(payload)
except BaseException as exc:
    payload["error_type"] = type(exc).__name__
    payload["error"] = error_text(exc)
    emit(payload)
    sys.exit(20)
"#;

/// Options for validating the PyTorch installation in one Python environment.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Exact Python executable whose installed `torch` package will be imported.
    pub python_executable: PathBuf,
    /// Logical PyTorch CUDA indices to exercise. `None` exercises every logical CUDA device.
    pub device_indices: Option<Vec<u32>>,
    /// Optional CUDA visibility override, normally a comma-separated list of full GPU UUIDs.
    pub cuda_visible_devices: Option<String>,
    /// Maximum wall-clock duration for the complete child process.
    pub timeout: Duration,
    /// Maximum combined stdout and stderr captured from the child process.
    pub max_output_bytes: usize,
}

impl VerifyOptions {
    /// Creates options with a 60-second timeout, a 1 MiB output limit, and all logical GPUs.
    pub fn new(python_executable: impl Into<PathBuf>) -> Self {
        Self {
            python_executable: python_executable.into(),
            device_indices: None,
            cuda_visible_devices: None,
            timeout: DEFAULT_TIMEOUT,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

/// Imports and dynamically validates the installed PyTorch package.
///
/// The selected interpreter is invoked as `python -I -c ...`, without a shell. The isolated flag
/// prevents the current directory, user site-packages, and `PYTHON*` variables from changing the
/// import target. CUDA visibility is preserved by default, or can be narrowed to an explicit set
/// of UUIDs whose logical indices are then supplied through `device_indices`.
///
/// # Security
///
/// Importing `torch` executes code installed in the selected Python environment. This function
/// isolates that execution in a bounded child process, but it is not a sandbox. Only verify trusted
/// environments.
pub fn verify_installed(options: &VerifyOptions) -> VerificationReport {
    let selected = match normalized_device_indices(options.device_indices.as_deref()) {
        Ok(selected) => selected,
        Err(error) => return failure_report(options, "verify_options", error),
    };

    if options.timeout.is_zero() {
        return failure_report(
            options,
            "verify_options",
            "verification timeout must be greater than zero",
        );
    }
    if options.max_output_bytes == 0 {
        return failure_report(
            options,
            "verify_options",
            "verification output limit must be greater than zero",
        );
    }

    let selected_json = match serde_json::to_string(&selected) {
        Ok(value) => value,
        Err(error) => {
            return failure_report(
                options,
                "verify_options",
                format!("could not encode selected CUDA devices: {error}"),
            );
        }
    };

    let command_result = run_probe(options, &selected_json);
    let output = match command_result {
        Ok(output) => output,
        Err(error) => return runner_failure_report(options, error),
    };

    let parsed = extract_probe_payload(&output.stdout);
    match parsed {
        Ok(payload) => report_from_payload(options, selected.as_deref(), output.status, payload),
        Err(parse_error) if !output.status.success() => failure_report(
            options,
            "subprocess_exit",
            format!(
                "verification subprocess {}; {}; stderr: {}",
                describe_exit_status(output.status),
                parse_error,
                bounded_text(&output.stderr)
            ),
        ),
        Err(parse_error) => failure_report(options, "probe_output", parse_error),
    }
}

fn normalized_device_indices(indices: Option<&[u32]>) -> Result<Option<Vec<u32>>, String> {
    let Some(indices) = indices else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Err("selected logical CUDA device list must not be empty".to_owned());
    }

    let unique = indices.iter().copied().collect::<BTreeSet<_>>();
    Ok(Some(unique.into_iter().collect()))
}

#[derive(Debug)]
struct ProbeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

#[derive(Debug)]
enum RunnerFailure {
    Spawn(String),
    Timeout,
    OutputLimit { limit: usize },
    Wait(String),
    Capture(String),
}

fn run_probe(options: &VerifyOptions, selected_json: &str) -> Result<ProbeOutput, RunnerFailure> {
    let mut command = probe_command(options, selected_json);

    let mut child = command
        .spawn()
        .map_err(|error| RunnerFailure::Spawn(error.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RunnerFailure::Capture("stdout pipe was not created".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| RunnerFailure::Capture("stderr pipe was not created".to_owned()))?;

    let budget = Arc::new(OutputBudget::new(options.max_output_bytes));
    let stdout_handle =
        spawn_reader(stdout, Arc::clone(&budget), "stdout").map_err(RunnerFailure::Capture)?;
    let stderr_handle = match spawn_reader(stderr, Arc::clone(&budget), "stderr") {
        Ok(reader) => reader,
        Err(error) => {
            terminate_process_tree(&mut child);
            return Err(RunnerFailure::Capture(error));
        }
    };
    let started = Instant::now();

    let status = match wait_for_child(&mut child, options.timeout, &budget) {
        Ok(status) => status,
        Err(failure) => {
            terminate_process_tree(&mut child);
            return Err(failure);
        }
    };

    terminate_process_group(child.id());

    let stdout = stdout_handle
        .wait(output_drain_timeout(remaining_timeout(
            started,
            options.timeout,
        )))
        .map_err(|error| reader_failure(error, "stdout"))?
        .map_err(RunnerFailure::Capture)?;
    let stderr = stderr_handle
        .wait(output_drain_timeout(remaining_timeout(
            started,
            options.timeout,
        )))
        .map_err(|error| reader_failure(error, "stderr"))?
        .map_err(RunnerFailure::Capture)?;

    if budget.exceeded.load(Ordering::Acquire) {
        return Err(RunnerFailure::OutputLimit {
            limit: options.max_output_bytes,
        });
    }

    Ok(ProbeOutput {
        status,
        stdout,
        stderr,
    })
}

fn probe_command(options: &VerifyOptions, selected_json: &str) -> Command {
    let mut command = Command::new(&options.python_executable);
    command
        .arg("-I")
        .arg("-c")
        .arg(PROBE_SCRIPT)
        .arg(selected_json)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(visible_devices) = &options.cuda_visible_devices {
        command.env("CUDA_VISIBLE_DEVICES", visible_devices);
    }
    isolate_process_tree(&mut command);
    command
}

fn wait_for_child(
    child: &mut Child,
    timeout: Duration,
    budget: &OutputBudget,
) -> Result<ExitStatus, RunnerFailure> {
    let started = Instant::now();
    loop {
        if budget.exceeded.load(Ordering::Acquire) {
            return Err(RunnerFailure::OutputLimit { limit: budget.max });
        }

        let elapsed = started.elapsed();
        if elapsed >= timeout {
            return Err(RunnerFailure::Timeout);
        }
        let remaining = timeout.saturating_sub(elapsed);
        let wait_for = remaining.min(WAIT_POLL_INTERVAL);
        match child.wait_timeout(wait_for) {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(error) => return Err(RunnerFailure::Wait(error.to_string())),
        }
    }
}

#[derive(Debug)]
struct OutputBudget {
    max: usize,
    used: AtomicUsize,
    exceeded: AtomicBool,
}

impl OutputBudget {
    fn new(max: usize) -> Self {
        Self {
            max,
            used: AtomicUsize::new(0),
            exceeded: AtomicBool::new(false),
        }
    }

    fn retain_count(&self, read_count: usize) -> usize {
        let previous = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                Some(used.saturating_add(read_count))
            })
            .unwrap_or(usize::MAX);
        let retain = read_count.min(self.max.saturating_sub(previous));
        if retain < read_count {
            self.exceeded.store(true, Ordering::Release);
        }
        retain
    }
}

fn spawn_reader<R>(
    mut reader: R,
    budget: Arc<OutputBudget>,
    stream: &'static str,
) -> Result<ReaderThread<Result<Vec<u8>, String>>, String>
where
    R: Read + Send + 'static,
{
    spawn_reader_thread(format!("torch-check-verify-{stream}"), move || {
        let mut captured = Vec::new();
        let mut chunk = [0_u8; 8192];
        loop {
            let count = reader.read(&mut chunk).map_err(|error| error.to_string())?;
            if count == 0 {
                return Ok(captured);
            }
            let retain = budget.retain_count(count);
            if retain > 0 {
                captured
                    .write_all(&chunk[..retain])
                    .map_err(|error| error.to_string())?;
            }
        }
    })
    .map_err(|error| format!("failed to create {stream} reader: {error}"))
}

fn reader_failure(error: ReaderWaitError, stream: &str) -> RunnerFailure {
    match error {
        ReaderWaitError::TimedOut => RunnerFailure::Capture(format!(
            "timed out draining {stream} after the verification child exited"
        )),
        ReaderWaitError::Panicked => RunnerFailure::Capture(format!("{stream} reader panicked")),
    }
}

fn remaining_timeout(started: Instant, timeout: Duration) -> Duration {
    timeout.saturating_sub(started.elapsed())
}

#[derive(Debug, Deserialize)]
struct ProbePayload {
    protocol_version: u32,
    ok: bool,
    stage: String,
    #[serde(default)]
    torch_version: Option<String>,
    #[serde(default)]
    compiled_cuda: Option<String>,
    #[serde(default)]
    cuda_available: Option<bool>,
    #[serde(default)]
    device_count: Option<u32>,
    #[serde(default)]
    arch_list: Vec<String>,
    #[serde(default)]
    devices: Vec<ProbeDevice>,
    #[serde(default)]
    cudnn_available: Option<bool>,
    #[serde(default)]
    cudnn_enabled: Option<bool>,
    #[serde(default)]
    cudnn_version: Option<u64>,
    #[serde(default)]
    cudnn_error: Option<String>,
    #[serde(default)]
    failed_device_index: Option<u32>,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeDevice {
    index: u32,
    name: String,
    capability: [u16; 2],
    operations_ok: bool,
}

fn extract_probe_payload(stdout: &[u8]) -> Result<ProbePayload, String> {
    let mut record = None;
    for raw_line in stdout.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let Some(json) = line.strip_prefix(PROBE_PREFIX) else {
            continue;
        };
        if record.replace(json).is_some() {
            return Err("verification subprocess emitted multiple protocol records".to_owned());
        }
    }

    let json = record
        .ok_or_else(|| "verification subprocess did not emit a protocol record".to_owned())?;
    serde_json::from_slice(json)
        .map_err(|error| format!("verification subprocess emitted invalid JSON: {error}"))
}

fn report_from_payload(
    options: &VerifyOptions,
    selected: Option<&[u32]>,
    status: ExitStatus,
    payload: ProbePayload,
) -> VerificationReport {
    if payload.protocol_version != PROBE_PROTOCOL_VERSION {
        return failure_report(
            options,
            "probe_protocol",
            format!(
                "unsupported verification protocol version {}; expected {}",
                payload.protocol_version, PROBE_PROTOCOL_VERSION
            ),
        );
    }

    if !payload.ok {
        return payload_failure_report(options, status, payload);
    }
    if !status.success() {
        return partial_failure_report(
            options,
            "subprocess_exit",
            format!(
                "verification subprocess {} despite reporting success",
                describe_exit_status(status)
            ),
            payload,
        );
    }
    if let Err(error) = validate_success_payload(&payload, selected) {
        return partial_failure_report(options, "probe_output", error, payload);
    }

    success_report(options, payload)
}

fn validate_success_payload(
    payload: &ProbePayload,
    selected: Option<&[u32]>,
) -> Result<(), String> {
    if payload.stage != "complete" {
        return Err(format!(
            "successful probe stopped at unexpected stage: {}",
            bounded_string(&payload.stage)
        ));
    }
    if payload
        .torch_version
        .as_deref()
        .is_none_or(|version| version.trim().is_empty())
    {
        return Err("successful probe omitted the PyTorch version".to_owned());
    }

    let cuda_available = payload
        .cuda_available
        .ok_or_else(|| "successful probe omitted CUDA availability".to_owned())?;
    let device_count = payload
        .device_count
        .ok_or_else(|| "successful probe omitted the logical device count".to_owned())?;

    match payload.compiled_cuda.as_deref() {
        None => {
            if selected.is_some() {
                return Err(
                    "explicit CUDA devices were requested, but the probe imported a CPU-only PyTorch build"
                        .to_owned(),
                );
            }
            if cuda_available {
                return Err(
                    "CPU-only probe unexpectedly reported an available CUDA device".to_owned(),
                );
            }
            if device_count != 0 || !payload.devices.is_empty() {
                return Err("CPU-only probe reported CUDA devices".to_owned());
            }
        }
        Some(version) if version.trim().is_empty() => {
            return Err("probe reported an empty compiled CUDA version".to_owned());
        }
        Some(_) => {
            if !cuda_available {
                return Err(
                    "CUDA build reported successful verification while unavailable".to_owned(),
                );
            }
            if device_count == 0 {
                return Err("CUDA build reported no logical devices".to_owned());
            }

            let expected = match selected {
                Some(indices) => indices.to_vec(),
                None => (0..device_count).collect(),
            };
            if expected.iter().any(|index| *index >= device_count) {
                return Err(format!(
                    "selected logical CUDA device is outside device count {device_count}"
                ));
            }
            let actual = payload
                .devices
                .iter()
                .map(|device| device.index)
                .collect::<Vec<_>>();
            if actual != expected {
                return Err(format!(
                    "probe exercised logical CUDA devices {actual:?}; expected {expected:?}"
                ));
            }
            if payload.devices.iter().any(|device| !device.operations_ok) {
                return Err("probe reported an unsuccessful CUDA device operation".to_owned());
            }
        }
    }

    Ok(())
}

fn success_report(options: &VerifyOptions, payload: ProbePayload) -> VerificationReport {
    let mut checks = vec![
        check(
            "subprocess_exit",
            true,
            "verification subprocess exited successfully",
        ),
        check(
            "torch_import",
            true,
            format!(
                "imported torch {}",
                payload.torch_version.as_deref().unwrap_or("unknown")
            ),
        ),
    ];

    let is_cpu = payload.compiled_cuda.is_none();
    checks.push(check(
        "torch_metadata",
        true,
        match payload.compiled_cuda.as_deref() {
            Some(cuda) => format!("compiled CUDA {cuda}"),
            None => "CPU-only PyTorch build".to_owned(),
        },
    ));
    checks.push(check(
        "cuda_availability",
        true,
        if is_cpu {
            "CUDA unavailable as expected for a CPU-only build".to_owned()
        } else {
            "CUDA runtime and driver initialized".to_owned()
        },
    ));
    checks.push(check(
        "device_count",
        true,
        format!(
            "{} logical CUDA device(s)",
            payload.device_count.unwrap_or(0)
        ),
    ));
    checks.push(check(
        "architecture_list",
        true,
        if is_cpu {
            "not applicable to a CPU-only build".to_owned()
        } else if payload.arch_list.is_empty() {
            "PyTorch returned an empty architecture list; runtime operations passed".to_owned()
        } else {
            payload.arch_list.join(", ")
        },
    ));

    for device in &payload.devices {
        checks.push(check(
            &format!("device_{}_operations", device.index),
            device.operations_ok,
            format!(
                "allocation, deterministic elementwise, matmul, and synchronization on {}",
                device.name
            ),
        ));
    }
    checks.push(cudnn_check(&payload));
    let diagnostics = cudnn_diagnostic(&payload).into_iter().collect();

    VerificationReport {
        schema_version: SCHEMA_VERSION,
        python_executable: options.python_executable.clone(),
        status: CompatibilityStatus::Verified,
        torch_version: payload.torch_version,
        compiled_cuda: payload.compiled_cuda,
        cuda_available: payload.cuda_available,
        device_count: payload.device_count,
        arch_list: payload.arch_list,
        devices: convert_devices(payload.devices),
        gpu_selection: Vec::new(),
        cudnn_available: payload.cudnn_available,
        checks,
        diagnostics,
        error: None,
    }
}

fn payload_failure_report(
    options: &VerifyOptions,
    status: ExitStatus,
    payload: ProbePayload,
) -> VerificationReport {
    let stage = bounded_string(&payload.stage);
    let kind = payload.error_type.as_deref().unwrap_or("runtime error");
    let message = payload
        .error
        .as_deref()
        .unwrap_or("probe failed without a message");
    let device = payload
        .failed_device_index
        .map_or_else(String::new, |index| {
            format!(" on logical CUDA device {index}")
        });
    let error = format!(
        "verification failed at {stage}{device}: {}: {}",
        bounded_string(kind),
        bounded_string(message)
    );

    partial_failure_report_with_status(
        options,
        "runtime_probe",
        format!("{}; subprocess {}", error, describe_exit_status(status)),
        error,
        payload,
    )
}

fn partial_failure_report(
    options: &VerifyOptions,
    check_name: &str,
    detail: impl Into<String>,
    payload: ProbePayload,
) -> VerificationReport {
    let detail = detail.into();
    partial_failure_report_with_status(options, check_name, detail.clone(), detail, payload)
}

fn partial_failure_report_with_status(
    options: &VerifyOptions,
    check_name: &str,
    detail: String,
    error: String,
    payload: ProbePayload,
) -> VerificationReport {
    let mut checks = Vec::new();
    if let Some(version) = payload.torch_version.as_deref() {
        checks.push(check(
            "torch_import",
            true,
            format!("imported torch {version}"),
        ));
    } else if payload.stage == "import_torch" {
        checks.push(check("torch_import", false, detail.clone()));
    }
    for device in &payload.devices {
        checks.push(check(
            &format!("device_{}_operations", device.index),
            device.operations_ok,
            format!("completed runtime operations on {}", device.name),
        ));
    }
    checks.push(check(check_name, false, detail));

    VerificationReport {
        schema_version: SCHEMA_VERSION,
        python_executable: options.python_executable.clone(),
        status: CompatibilityStatus::Incompatible,
        torch_version: payload.torch_version,
        compiled_cuda: payload.compiled_cuda,
        cuda_available: payload.cuda_available,
        device_count: payload.device_count,
        arch_list: payload.arch_list,
        devices: convert_devices(payload.devices),
        gpu_selection: Vec::new(),
        cudnn_available: payload.cudnn_available,
        checks,
        diagnostics: Vec::new(),
        error: Some(bounded_string(&error)),
    }
}

fn convert_devices(devices: Vec<ProbeDevice>) -> Vec<VerifiedDevice> {
    devices
        .into_iter()
        .map(|device| VerifiedDevice {
            index: device.index,
            name: device.name,
            capability: ComputeCapability {
                major: device.capability[0],
                minor: device.capability[1],
            },
            operations_ok: device.operations_ok,
        })
        .collect()
}

fn cudnn_check(payload: &ProbePayload) -> VerificationCheck {
    if let Some(error) = payload.cudnn_error.as_deref() {
        return check(
            "cudnn_info",
            true,
            format!(
                "informational cuDNN query unavailable: {}",
                bounded_string(error)
            ),
        );
    }

    let available = payload
        .cudnn_available
        .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
    let enabled = payload
        .cudnn_enabled
        .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
    let version = payload
        .cudnn_version
        .map_or_else(|| "none".to_owned(), |value| value.to_string());
    check(
        "cudnn_info",
        true,
        format!("available={available}, enabled={enabled}, version={version}"),
    )
}

fn cudnn_diagnostic(payload: &ProbePayload) -> Option<Diagnostic> {
    let error = payload.cudnn_error.as_deref()?;
    Some(Diagnostic {
        code: DiagnosticCode::CudnnInspectionFailed,
        severity: DiagnosticSeverity::Warning,
        details: BTreeMap::from([("error".to_owned(), bounded_string(error))]),
    })
}

fn runner_failure_report(options: &VerifyOptions, failure: RunnerFailure) -> VerificationReport {
    match failure {
        RunnerFailure::Spawn(error) => failure_report(
            options,
            "subprocess_spawn",
            format!("could not start selected Python: {error}"),
        ),
        RunnerFailure::Timeout => failure_report(
            options,
            "subprocess_timeout",
            format!(
                "verification subprocess exceeded {} ms",
                options.timeout.as_millis()
            ),
        ),
        RunnerFailure::OutputLimit { limit } => failure_report(
            options,
            "subprocess_output_limit",
            format!("verification subprocess exceeded the {limit}-byte output limit"),
        ),
        RunnerFailure::Wait(error) => failure_report(
            options,
            "subprocess_wait",
            format!("could not wait for verification subprocess: {error}"),
        ),
        RunnerFailure::Capture(error) => failure_report(
            options,
            "subprocess_output",
            format!("could not capture verification output: {error}"),
        ),
    }
}

fn failure_report(
    options: &VerifyOptions,
    check_name: &str,
    detail: impl Into<String>,
) -> VerificationReport {
    let detail = bounded_string(&detail.into());
    VerificationReport {
        schema_version: SCHEMA_VERSION,
        python_executable: options.python_executable.clone(),
        status: CompatibilityStatus::Incompatible,
        torch_version: None,
        compiled_cuda: None,
        cuda_available: None,
        device_count: None,
        arch_list: Vec::new(),
        devices: Vec::new(),
        gpu_selection: Vec::new(),
        cudnn_available: None,
        checks: vec![check(check_name, false, detail.clone())],
        diagnostics: Vec::new(),
        error: Some(detail),
    }
}

fn check(name: &str, passed: bool, detail: impl Into<String>) -> VerificationCheck {
    VerificationCheck {
        name: name.to_owned(),
        passed,
        detail: Some(bounded_string(&detail.into())),
    }
}

fn bounded_text(bytes: &[u8]) -> String {
    bounded_string(&String::from_utf8_lossy(bytes))
}

fn bounded_string(value: &str) -> String {
    const MAX_CHARS: usize = 1024;
    let mut result = value
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(MAX_CHARS)
        .collect::<String>();
    if value.chars().count() > MAX_CHARS {
        result.push_str("...");
    }
    result.trim().to_owned()
}

fn describe_exit_status(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exited with status {code}");
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("terminated by signal {signal}");
        }
    }

    "terminated without an exit status".to_owned()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    fn options() -> VerifyOptions {
        VerifyOptions {
            python_executable: PathBuf::from("fake-python"),
            device_indices: None,
            cuda_visible_devices: None,
            timeout: Duration::from_secs(5),
            max_output_bytes: 16 * 1024,
        }
    }

    fn exit_status(code: i32) -> ExitStatus {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            ExitStatus::from_raw(code << 8)
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            ExitStatus::from_raw(code as u32)
        }
    }

    fn payload(value: serde_json::Value) -> ProbePayload {
        serde_json::from_value(value).expect("valid test probe payload")
    }

    fn cpu_success_payload() -> ProbePayload {
        payload(json!({
            "protocol_version": 1,
            "ok": true,
            "stage": "complete",
            "torch_version": "2.9.1+cpu",
            "compiled_cuda": null,
            "cuda_available": false,
            "device_count": 0,
            "arch_list": [],
            "devices": [],
            "cudnn_available": false,
            "cudnn_enabled": true,
            "cudnn_version": null
        }))
    }

    fn cuda_success_payload() -> ProbePayload {
        payload(json!({
            "protocol_version": 1,
            "ok": true,
            "stage": "complete",
            "torch_version": "2.9.1+cu128",
            "compiled_cuda": "12.8",
            "cuda_available": true,
            "device_count": 2,
            "arch_list": ["sm_80", "sm_90", "compute_90"],
            "devices": [
                {"index": 0, "name": "GPU zero", "capability": [8, 0], "operations_ok": true},
                {"index": 1, "name": "GPU one", "capability": [9, 0], "operations_ok": true}
            ],
            "cudnn_available": true,
            "cudnn_enabled": true,
            "cudnn_version": 91002
        }))
    }

    #[test]
    fn cpu_only_build_is_verified_when_cuda_is_unavailable() {
        let report = report_from_payload(&options(), None, exit_status(0), cpu_success_payload());

        assert_eq!(report.status, CompatibilityStatus::Verified);
        assert_eq!(report.cuda_available, Some(false));
        assert!(report.devices.is_empty());
        assert!(report.error.is_none());
        assert!(
            report
                .checks
                .iter()
                .find(|check| check.name == "cuda_availability")
                .is_some_and(|check| check.passed)
        );
    }

    #[test]
    fn explicit_cuda_selection_rejects_a_cpu_only_build() {
        let report = report_from_payload(
            &options(),
            Some(&[0]),
            exit_status(0),
            cpu_success_payload(),
        );

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert!(
            report
                .error
                .as_deref()
                .is_some_and(|error| error.contains("CPU-only"))
        );
    }

    #[test]
    fn optional_cudnn_query_failure_is_a_warning_not_a_failed_check() {
        let mut payload = cuda_success_payload();
        payload.cudnn_error = Some("could not load /opt/private/libcudnn.so".to_owned());
        let report = report_from_payload(&options(), None, exit_status(0), payload);

        assert_eq!(report.status, CompatibilityStatus::Verified);
        assert!(
            report
                .checks
                .iter()
                .find(|check| check.name == "cudnn_info")
                .is_some_and(|check| check.passed)
        );
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::CudnnInspectionFailed
                && diagnostic.severity == DiagnosticSeverity::Warning
        }));
    }

    #[test]
    fn cuda_build_exercises_every_logical_device_by_default() {
        let report = report_from_payload(&options(), None, exit_status(0), cuda_success_payload());

        assert_eq!(report.status, CompatibilityStatus::Verified);
        assert_eq!(report.devices.len(), 2);
        assert_eq!(
            report.devices[0].capability,
            ComputeCapability { major: 8, minor: 0 }
        );
        assert_eq!(
            report.devices[1].capability,
            ComputeCapability { major: 9, minor: 0 }
        );
        assert!(report.devices.iter().all(|device| device.operations_ok));
    }

    #[test]
    fn successful_payload_must_cover_exact_selected_devices() {
        let report = report_from_payload(
            &options(),
            Some(&[1]),
            exit_status(0),
            cuda_success_payload(),
        );

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert!(
            report
                .error
                .as_deref()
                .is_some_and(|error| error.contains("expected [1]"))
        );
    }

    #[test]
    fn cuda_build_that_cannot_initialize_cuda_fails_structurally() {
        let failed = payload(json!({
            "protocol_version": 1,
            "ok": false,
            "stage": "cuda_availability",
            "torch_version": "2.9.1+cu128",
            "compiled_cuda": "12.8",
            "cuda_available": false,
            "device_count": 0,
            "error_type": "RuntimeError",
            "error": "this PyTorch build includes CUDA, but torch.cuda.is_available() is false"
        }));

        let report = report_from_payload(&options(), None, exit_status(20), failed);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert_eq!(report.cuda_available, Some(false));
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.name == "runtime_probe" && !check.passed)
        );
    }

    #[test]
    fn parser_ignores_unrelated_stdout_but_rejects_duplicate_records() {
        let json = serde_json::to_string(&json!({
            "protocol_version": 1,
            "ok": false,
            "stage": "import_torch"
        }))
        .expect("serialize fixture");
        let one = format!("library noise\nTORCH_CHECK_VERIFY_JSON:{json}\n");
        assert!(extract_probe_payload(one.as_bytes()).is_ok());

        let duplicate = format!("{one}TORCH_CHECK_VERIFY_JSON:{json}\n");
        assert!(
            extract_probe_payload(duplicate.as_bytes())
                .expect_err("duplicate records must fail")
                .contains("multiple protocol records")
        );
    }

    #[test]
    fn duplicate_selected_indices_are_normalized_deterministically() {
        let normalized = normalized_device_indices(Some(&[2, 0, 2, 1]))
            .expect("indices should normalize")
            .expect("selection should remain explicit");
        assert_eq!(normalized, vec![0, 1, 2]);
    }

    #[cfg(unix)]
    fn fake_python(body: &str) -> (tempfile::TempDir, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("create temporary directory");
        let executable = directory.path().join("fake-python");
        fs::write(&executable, format!("#!/bin/sh\n{body}\n"))
            .expect("write fake Python executable");
        let mut permissions = fs::metadata(&executable)
            .expect("read fake Python metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).expect("make fake Python executable");
        (directory, executable)
    }

    #[cfg(unix)]
    fn shell_single_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }

    #[cfg(unix)]
    fn protocol_line(value: serde_json::Value) -> String {
        format!(
            "TORCH_CHECK_VERIFY_JSON:{}",
            serde_json::to_string(&value).expect("serialize protocol fixture")
        )
    }

    #[cfg(unix)]
    #[test]
    fn probe_command_uses_isolated_mode_and_protocol_arguments() {
        use std::ffi::OsStr;

        let command = probe_command(&options(), "[0,1]");
        let arguments = command.get_args().collect::<Vec<_>>();

        assert_eq!(
            arguments,
            [
                OsStr::new("-I"),
                OsStr::new("-c"),
                OsStr::new(PROBE_SCRIPT),
                OsStr::new("[0,1]"),
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_parses_successful_probe() {
        let _guard = crate::process::lock_subprocess_tests();
        let line = protocol_line(json!({
            "protocol_version": 1,
            "ok": true,
            "stage": "complete",
            "torch_version": "2.9.1+cpu",
            "compiled_cuda": null,
            "cuda_available": false,
            "device_count": 0,
            "arch_list": [],
            "devices": [],
            "cudnn_available": false,
            "cudnn_enabled": true,
            "cudnn_version": null
        }));
        let body = format!("printf '%s\\n' {}", shell_single_quote(&line));
        let (_directory, executable) = fake_python(&body);
        let mut options = options();
        options.python_executable = executable;

        let report = verify_installed(&options);

        assert_eq!(
            report.status,
            CompatibilityStatus::Verified,
            "verification failed: {:?}",
            report.error
        );
        assert_eq!(report.torch_version.as_deref(), Some("2.9.1+cpu"));
    }

    #[cfg(unix)]
    #[test]
    fn explicit_visibility_override_is_passed_by_uuid() {
        let _guard = crate::process::lock_subprocess_tests();
        let line = protocol_line(json!({
            "protocol_version": 1,
            "ok": true,
            "stage": "complete",
            "torch_version": "2.9.1+cu128",
            "compiled_cuda": "12.8",
            "cuda_available": true,
            "device_count": 1,
            "arch_list": ["sm_80"],
            "devices": [
                {"index": 0, "name": "selected GPU", "capability": [8, 0], "operations_ok": true}
            ],
            "cudnn_available": true,
            "cudnn_enabled": true,
            "cudnn_version": 91002
        }));
        let body = format!(
            "[ \"$CUDA_VISIBLE_DEVICES\" = \"GPU-aaaaaaaa\" ] || exit 93\nprintf '%s\\n' {}",
            shell_single_quote(&line)
        );
        let (_directory, executable) = fake_python(&body);
        let mut options = options();
        options.python_executable = executable;
        options.device_indices = Some(vec![0]);
        options.cuda_visible_devices = Some("GPU-aaaaaaaa".to_owned());

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Verified);
        assert_eq!(report.devices.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn structured_import_failure_is_preserved() {
        let _guard = crate::process::lock_subprocess_tests();
        let line = protocol_line(json!({
            "protocol_version": 1,
            "ok": false,
            "stage": "import_torch",
            "error_type": "ModuleNotFoundError",
            "error": "No module named 'torch'"
        }));
        let body = format!("printf '%s\\n' {}\nexit 20", shell_single_quote(&line));
        let (_directory, executable) = fake_python(&body);
        let mut options = options();
        options.python_executable = executable;

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert!(
            report
                .error
                .as_deref()
                .is_some_and(|error| error.contains("ModuleNotFoundError"))
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.name == "torch_import" && !check.passed)
        );
    }

    #[cfg(unix)]
    #[test]
    fn invalid_json_is_a_structured_failure() {
        let _guard = crate::process::lock_subprocess_tests();
        let (_directory, executable) =
            fake_python("printf '%s\\n' 'TORCH_CHECK_VERIFY_JSON:{not-json}'");
        let mut options = options();
        options.python_executable = executable;

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert_eq!(report.checks[0].name, "probe_output");
        assert!(
            report
                .error
                .as_deref()
                .is_some_and(|error| error.contains("invalid JSON"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn timeout_terminates_the_subprocess() {
        let _guard = crate::process::lock_subprocess_tests();
        let (_directory, executable) = fake_python("exec sleep 2");
        let mut options = options();
        options.python_executable = executable;
        options.timeout = Duration::from_millis(50);

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert_eq!(
            report.checks[0].name, "subprocess_timeout",
            "verification failed: {:?}",
            report.error
        );
    }

    #[cfg(unix)]
    #[test]
    fn combined_output_limit_terminates_the_subprocess() {
        let _guard = crate::process::lock_subprocess_tests();
        let (_directory, executable) = fake_python(
            "i=0\nwhile [ \"$i\" -lt 1000 ]; do\n  printf 'xxxxxxxxxxxxxxxx'\n  i=$((i + 1))\ndone",
        );
        let mut options = options();
        options.python_executable = executable;
        options.max_output_bytes = 128;

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert_eq!(report.checks[0].name, "subprocess_output_limit");
    }

    #[cfg(unix)]
    #[test]
    fn completed_probe_does_not_wait_for_descendants_holding_pipes() {
        let _guard = crate::process::lock_subprocess_tests();
        let (_directory, executable) = fake_python("sleep 5 &\nexit 0");
        let mut options = options();
        options.python_executable = executable;
        options.timeout = Duration::from_secs(2);
        let started = Instant::now();

        let report = verify_installed(&options);

        assert_eq!(report.status, CompatibilityStatus::Incompatible);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "inherited pipes must not keep verification blocked"
        );
    }
}
