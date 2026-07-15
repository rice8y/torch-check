//! Defensive host, Python, NVIDIA, and CUDA Toolkit detection.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::ffi::CStr;
use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::str::FromStr;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use csv::ReaderBuilder;
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use wait_timeout::ChildExt;

use crate::core::{
    Architecture, ComputeCapability, CudaToolkitInfo, Diagnostic, DiagnosticCode,
    DiagnosticSeverity, Environment, NumericVersion, NvidiaDetectionStatus, NvidiaGpu, NvidiaInfo,
    OperatingSystem, PlatformInfo, PythonInfo, TagSource, ToolkitSource,
    is_unsafe_terminal_character,
};
use crate::process::{isolate_process_tree, terminate_process_group, terminate_process_tree};

const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(8);
const DEFAULT_MAX_COMMAND_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_TEXT_FILE_BYTES: u64 = 1024 * 1024;
const MAX_ELF_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PYTHON_TAGS: usize = 8192;
const MAX_TAG_LENGTH: usize = 256;

const PYTHON_PROBE: &str = r#"
import json
import platform
import struct
import sys
import sysconfig

tags = []
packaging_available = False

cache_tag = getattr(sys.implementation, "cache_tag", None)
gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
free_threaded = bool(gil_disabled) or bool(cache_tag and cache_tag.endswith("t"))
virtual_environment = sys.prefix if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else None

print(json.dumps({
    "implementation": platform.python_implementation(),
    "version": list(sys.version_info[:3]),
    "soabi": sysconfig.get_config_var("SOABI"),
    "cache_tag": cache_tag,
    "platform": sysconfig.get_platform(),
    "pointer_width": struct.calcsize("P") * 8,
    "free_threaded": free_threaded,
    "virtual_environment": virtual_environment,
    "packaging_available": packaging_available,
    "compatible_tags": tags,
}, separators=(",", ":")))
"#;

/// Options controlling environment detection.
#[derive(Debug, Clone)]
pub struct DetectOptions {
    /// Explicit interpreter path corresponding to `--python`.
    pub python: Option<PathBuf>,
    /// Physical NVIDIA indices to inspect. An empty list selects every GPU.
    pub gpu_indices: Vec<u32>,
    /// Maximum duration of each external command.
    pub command_timeout: Duration,
    /// Maximum bytes retained independently for stdout and stderr.
    pub max_command_output_bytes: usize,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            python: None,
            gpu_indices: Vec::new(),
            command_timeout: DEFAULT_COMMAND_TIMEOUT,
            max_command_output_bytes: DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
        }
    }
}

/// Invalid environment-detection configuration.
#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    /// A command cannot be bounded with the supplied options.
    #[error("invalid detection option: {0}")]
    InvalidOption(&'static str),
}

/// Limits applied to a subprocess invocation.
#[derive(Debug, Clone, Copy)]
pub struct RunCommandOptions {
    /// Hard wall-clock timeout.
    pub timeout: Duration,
    /// Maximum retained stdout bytes.
    pub max_stdout_bytes: usize,
    /// Maximum retained stderr bytes.
    pub max_stderr_bytes: usize,
}

impl Default for RunCommandOptions {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_COMMAND_TIMEOUT,
            max_stdout_bytes: DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
            max_stderr_bytes: DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
        }
    }
}

/// Output captured from a subprocess without invoking a shell.
#[derive(Debug)]
pub struct CapturedOutput {
    /// Process exit status.
    pub status: ExitStatus,
    /// Exact stdout bytes.
    pub stdout: Vec<u8>,
    /// Exact stderr bytes.
    pub stderr: Vec<u8>,
}

/// Captured subprocess stream.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OutputStream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

impl std::fmt::Display for OutputStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stdout => formatter.write_str("stdout"),
            Self::Stderr => formatter.write_str("stderr"),
        }
    }
}

/// Failure from a bounded subprocess invocation.
#[derive(Debug, thiserror::Error)]
pub enum CommandError {
    /// The process could not be created.
    #[error("failed to spawn {program}: {source}")]
    Spawn {
        /// Display form of the executable.
        program: String,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
    /// A reader thread could not be created.
    #[error("failed to create the {stream} reader for {program}: {source}")]
    ReaderSpawn {
        /// Display form of the executable.
        program: String,
        /// Stream being captured.
        stream: OutputStream,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
    /// Waiting for the process failed.
    #[error("failed while waiting for {program}: {source}")]
    Wait {
        /// Display form of the executable.
        program: String,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
    /// Reading one of the output streams failed.
    #[error("failed to read {stream} from {program}: {source}")]
    Read {
        /// Display form of the executable.
        program: String,
        /// Stream that failed.
        stream: OutputStream,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
    /// A reader thread panicked unexpectedly.
    #[error("the {stream} reader for {program} panicked")]
    ReaderPanicked {
        /// Display form of the executable.
        program: String,
        /// Stream that failed.
        stream: OutputStream,
    },
    /// The process did not exit before its deadline.
    #[error("{program} timed out after {timeout:?}")]
    TimedOut {
        /// Display form of the executable.
        program: String,
        /// Configured timeout.
        timeout: Duration,
    },
    /// An output stream exceeded its configured bound.
    #[error("{program} exceeded the {stream} limit of {limit} bytes")]
    OutputLimitExceeded {
        /// Display form of the executable.
        program: String,
        /// Stream that exceeded the limit.
        stream: OutputStream,
        /// Configured limit.
        limit: usize,
    },
}

impl CommandError {
    fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::Spawn { source, .. } if source.kind() == io::ErrorKind::NotFound
        )
    }

    fn is_timeout(&self) -> bool {
        matches!(self, Self::TimedOut { .. })
    }
}

#[derive(Debug)]
struct BoundedRead {
    bytes: Vec<u8>,
    exceeded: bool,
}

/// Runs a subprocess directly with an argument vector, a timeout, and bounded output.
///
/// stdout and stderr are drained concurrently so a verbose child cannot deadlock on
/// a full pipe. Bytes beyond each limit are discarded and reported as an error after
/// the child exits. No shell expansion or command-string interpretation occurs.
pub fn run_command<P, A>(
    program: P,
    args: &[A],
    options: &RunCommandOptions,
) -> Result<CapturedOutput, CommandError>
where
    P: AsRef<OsStr>,
    A: AsRef<OsStr>,
{
    let program = program.as_ref();
    let display_program = program.to_string_lossy().into_owned();
    let mut command = Command::new(program);
    command
        .args(args.iter().map(AsRef::as_ref))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    isolate_process_tree(&mut command);

    let mut child = command.spawn().map_err(|source| CommandError::Spawn {
        program: display_program.clone(),
        source,
    })?;
    let stdout = child.stdout.take().expect("piped stdout must be present");
    let stderr = child.stderr.take().expect("piped stderr must be present");

    let stdout_reader = spawn_reader(
        stdout,
        options.max_stdout_bytes,
        &display_program,
        OutputStream::Stdout,
    )?;
    let stderr_reader = match spawn_reader(
        stderr,
        options.max_stderr_bytes,
        &display_program,
        OutputStream::Stderr,
    ) {
        Ok(reader) => reader,
        Err(error) => {
            terminate_process_tree(&mut child);
            let _ = stdout_reader.join();
            return Err(error);
        }
    };

    let status = match child.wait_timeout(options.timeout) {
        Ok(Some(status)) => status,
        Ok(None) => {
            terminate_process_tree(&mut child);
            let _ = join_reader(stdout_reader, &display_program, OutputStream::Stdout);
            let _ = join_reader(stderr_reader, &display_program, OutputStream::Stderr);
            return Err(CommandError::TimedOut {
                program: display_program,
                timeout: options.timeout,
            });
        }
        Err(source) => {
            terminate_process_tree(&mut child);
            let _ = join_reader(stdout_reader, &display_program, OutputStream::Stdout);
            let _ = join_reader(stderr_reader, &display_program, OutputStream::Stderr);
            return Err(CommandError::Wait {
                program: display_program,
                source,
            });
        }
    };

    terminate_process_group(child.id());

    let stdout = join_reader(stdout_reader, &display_program, OutputStream::Stdout)?;
    let stderr = join_reader(stderr_reader, &display_program, OutputStream::Stderr)?;
    if stdout.exceeded {
        return Err(CommandError::OutputLimitExceeded {
            program: display_program,
            stream: OutputStream::Stdout,
            limit: options.max_stdout_bytes,
        });
    }
    if stderr.exceeded {
        return Err(CommandError::OutputLimitExceeded {
            program: display_program,
            stream: OutputStream::Stderr,
            limit: options.max_stderr_bytes,
        });
    }

    Ok(CapturedOutput {
        status,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
    })
}

fn spawn_reader<R>(
    reader: R,
    limit: usize,
    program: &str,
    stream: OutputStream,
) -> Result<JoinHandle<io::Result<BoundedRead>>, CommandError>
where
    R: Read + Send + 'static,
{
    thread::Builder::new()
        .name(format!("torch-check-{stream}"))
        .spawn(move || read_bounded(reader, limit))
        .map_err(|source| CommandError::ReaderSpawn {
            program: program.to_owned(),
            stream,
            source,
        })
}

fn read_bounded(mut reader: impl Read, limit: usize) -> io::Result<BoundedRead> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    let mut exceeded = false;
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let retained = remaining.min(count);
        bytes.extend_from_slice(&buffer[..retained]);
        exceeded |= retained < count;
    }
    Ok(BoundedRead { bytes, exceeded })
}

fn join_reader(
    reader: JoinHandle<io::Result<BoundedRead>>,
    program: &str,
    stream: OutputStream,
) -> Result<BoundedRead, CommandError> {
    reader
        .join()
        .map_err(|_| CommandError::ReaderPanicked {
            program: program.to_owned(),
            stream,
        })?
        .map_err(|source| CommandError::Read {
            program: program.to_owned(),
            stream,
            source,
        })
}

/// Detects the current host and selected Python/NVIDIA environment.
pub fn detect_environment(options: &DetectOptions) -> Result<Environment, DetectError> {
    if options.command_timeout.is_zero() {
        return Err(DetectError::InvalidOption(
            "command_timeout must be greater than zero",
        ));
    }
    if options.max_command_output_bytes == 0 {
        return Err(DetectError::InvalidOption(
            "max_command_output_bytes must be greater than zero",
        ));
    }

    let command_options = RunCommandOptions {
        timeout: options.command_timeout,
        max_stdout_bytes: options.max_command_output_bytes,
        max_stderr_bytes: options.max_command_output_bytes,
    };
    let mut diagnostics = Vec::new();
    let platform = detect_platform(&command_options);

    if platform.os != OperatingSystem::Linux {
        diagnostics.push(diagnostic(
            DiagnosticCode::UnsupportedOperatingSystem,
            DiagnosticSeverity::Error,
            [("os", operating_system_name(&platform.os))],
        ));
    }
    if platform.architecture != Architecture::X86_64 {
        diagnostics.push(diagnostic(
            DiagnosticCode::UnsupportedArchitecture,
            DiagnosticSeverity::Error,
            [("architecture", architecture_name(&platform.architecture))],
        ));
    }

    let glibc = if platform.os == OperatingSystem::Linux {
        match detect_libc(&command_options) {
            LibcDetection::Glibc(version) => Some(version),
            LibcDetection::Musl(version) => {
                let mut details = vec![("libc", "musl".to_owned())];
                if let Some(version) = version {
                    details.push(("version", version.to_string()));
                }
                diagnostics.push(diagnostic(
                    DiagnosticCode::UnsupportedLibc,
                    DiagnosticSeverity::Error,
                    details,
                ));
                None
            }
            LibcDetection::Unknown => {
                diagnostics.push(diagnostic(
                    DiagnosticCode::GlibcUnknown,
                    DiagnosticSeverity::Warning,
                    std::iter::empty::<(&str, String)>(),
                ));
                None
            }
        }
    } else {
        None
    };

    let python = detect_python(options, &command_options, glibc.as_ref(), &mut diagnostics);
    let nvidia = detect_nvidia(options, &command_options, &mut diagnostics);
    let cuda_toolkit = detect_cuda_toolkit(&command_options);
    if cuda_toolkit.is_none() {
        diagnostics.push(diagnostic(
            DiagnosticCode::CudaToolkitNotFound,
            DiagnosticSeverity::Info,
            [("wheel_compatibility", "not_required")],
        ));
    }

    Ok(Environment {
        platform,
        glibc,
        python,
        nvidia,
        cuda_toolkit,
        diagnostics,
    })
}

fn diagnostic<K, V, I>(code: DiagnosticCode, severity: DiagnosticSeverity, details: I) -> Diagnostic
where
    K: Into<String>,
    V: Into<String>,
    I: IntoIterator<Item = (K, V)>,
{
    Diagnostic {
        code,
        severity,
        details: details
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect(),
    }
}

fn detect_platform(command_options: &RunCommandOptions) -> PlatformInfo {
    let os = match env::consts::OS {
        "linux" => OperatingSystem::Linux,
        "windows" => OperatingSystem::Windows,
        "macos" => OperatingSystem::Macos,
        other => OperatingSystem::Other(other.to_owned()),
    };
    let architecture = normalize_architecture(env::consts::ARCH);
    let kernel_version = read_limited_text(Path::new("/proc/sys/kernel/osrelease"))
        .ok()
        .and_then(|value| nonempty_trimmed(&value))
        .or_else(|| {
            run_command("uname", &["-r"], command_options)
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .and_then(|value| nonempty_trimmed(&value))
        });
    let distribution = if os == OperatingSystem::Linux {
        ["/etc/os-release", "/usr/lib/os-release"]
            .into_iter()
            .find_map(|path| {
                read_limited_text(Path::new(path))
                    .ok()
                    .and_then(|contents| parse_os_release(&contents))
            })
    } else {
        None
    };

    PlatformInfo {
        os,
        architecture,
        kernel_version,
        distribution,
    }
}

fn normalize_architecture(value: &str) -> Architecture {
    match value.to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => Architecture::X86_64,
        "aarch64" | "arm64" => Architecture::Aarch64,
        other => Architecture::Other(other.to_owned()),
    }
}

fn operating_system_name(os: &OperatingSystem) -> String {
    match os {
        OperatingSystem::Linux => "linux".to_owned(),
        OperatingSystem::Windows => "windows".to_owned(),
        OperatingSystem::Macos => "macos".to_owned(),
        OperatingSystem::Other(name) => name.clone(),
    }
}

fn architecture_name(architecture: &Architecture) -> String {
    match architecture {
        Architecture::X86_64 => "x86_64".to_owned(),
        Architecture::Aarch64 => "aarch64".to_owned(),
        Architecture::Other(name) => name.clone(),
    }
}

fn parse_os_release(contents: &str) -> Option<String> {
    let mut values = BTreeMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        {
            continue;
        }
        if let Some(value) = decode_os_release_value(raw_value.trim()) {
            values.insert(key.to_owned(), value);
        }
    }
    values
        .get("PRETTY_NAME")
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| {
            let name = values.get("NAME")?.trim();
            if name.is_empty() {
                return None;
            }
            let version = values
                .get("VERSION_ID")
                .map(|value| value.trim())
                .filter(|value| !value.is_empty());
            Some(match version {
                Some(version) => format!("{name} {version}"),
                None => name.to_owned(),
            })
        })
}

fn decode_os_release_value(value: &str) -> Option<String> {
    if let Some(inner) = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        let mut decoded = String::with_capacity(inner.len());
        let mut escaped = false;
        for character in inner.chars() {
            if escaped {
                if matches!(character, '"' | '\\' | '$' | '`') {
                    decoded.push(character);
                } else {
                    decoded.push('\\');
                    decoded.push(character);
                }
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else {
                decoded.push(character);
            }
        }
        if escaped {
            decoded.push('\\');
        }
        return Some(decoded);
    }
    if let Some(inner) = value
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
    {
        return Some(inner.to_owned());
    }
    if value.starts_with(['"', '\'']) || value.ends_with(['"', '\'']) {
        return None;
    }
    Some(value.to_owned())
}

fn read_limited_text(path: &Path) -> io::Result<String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "text input is not a regular file",
        ));
    }
    if metadata.len() > MAX_TEXT_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "text file exceeds size limit",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_TEXT_FILE_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_TEXT_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "text file exceeds size limit",
        ));
    }
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn read_limited_binary(path: &Path, limit: u64) -> io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "binary input is not a bounded regular file",
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "binary input exceeds size limit",
        ));
    }
    Ok(bytes)
}

fn nonempty_trimmed(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum LibcDetection {
    Glibc(NumericVersion),
    Musl(Option<NumericVersion>),
    Unknown,
}

fn detect_libc(command_options: &RunCommandOptions) -> LibcDetection {
    if let Some(version) = glibc_via_gnu_get_libc_version() {
        return LibcDetection::Glibc(version);
    }
    if let Ok(output) = run_command("ldd", &["--version"], command_options) {
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push('\n');
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        let detected = classify_libc_output(&text);
        if detected != LibcDetection::Unknown {
            return detected;
        }
    }
    detect_libc_from_elf(command_options).unwrap_or(LibcDetection::Unknown)
}

fn detect_libc_from_elf(command_options: &RunCommandOptions) -> Option<LibcDetection> {
    for executable in ["/bin/sh", "/usr/bin/env", "/usr/bin/python3"] {
        let Ok(bytes) = read_limited_binary(Path::new(executable), MAX_ELF_FILE_BYTES) else {
            continue;
        };
        let Some(interpreter) = elf_interpreter(&bytes) else {
            continue;
        };
        let lowercase = interpreter.to_ascii_lowercase();
        if lowercase.contains("musl") {
            return Some(LibcDetection::Musl(None));
        }
        if lowercase.contains("ld-linux") || lowercase.contains("ld64.so") {
            if let Ok(output) = run_command(&interpreter, &["--version"], command_options) {
                let text = combined_output(&output);
                let detected = classify_libc_output(&text);
                if detected != LibcDetection::Unknown {
                    return Some(detected);
                }
            }
            return Some(LibcDetection::Unknown);
        }
    }
    None
}

fn elf_interpreter(bytes: &[u8]) -> Option<String> {
    if bytes.get(..4)? != b"\x7fELF" {
        return None;
    }
    let class = *bytes.get(4)?;
    let little_endian = match *bytes.get(5)? {
        1 => true,
        2 => false,
        _ => return None,
    };
    let u16_at = |offset| read_elf_u16(bytes, offset, little_endian);
    let u32_at = |offset| read_elf_u32(bytes, offset, little_endian);
    let u64_at = |offset| read_elf_u64(bytes, offset, little_endian);
    let (program_offset, entry_size, entry_count, offset_field, size_field) = match class {
        1 => (
            u64::from(u32_at(28)?),
            usize::from(u16_at(42)?),
            usize::from(u16_at(44)?),
            4,
            16,
        ),
        2 => (
            u64_at(32)?,
            usize::from(u16_at(54)?),
            usize::from(u16_at(56)?),
            8,
            32,
        ),
        _ => return None,
    };
    let program_offset = usize::try_from(program_offset).ok()?;
    let minimum_entry_size = if class == 1 { 20 } else { 40 };
    if entry_size < minimum_entry_size || entry_count > 4096 {
        return None;
    }
    let table_size = entry_count.checked_mul(entry_size)?;
    bytes.get(program_offset..program_offset.checked_add(table_size)?)?;
    for index in 0..entry_count {
        let entry = program_offset.checked_add(index.checked_mul(entry_size)?)?;
        if u32_at(entry)? != 3 {
            continue;
        }
        let offset = if class == 1 {
            u64::from(u32_at(entry.checked_add(offset_field)?)?)
        } else {
            u64_at(entry.checked_add(offset_field)?)?
        };
        let size = if class == 1 {
            u64::from(u32_at(entry.checked_add(size_field)?)?)
        } else {
            u64_at(entry.checked_add(size_field)?)?
        };
        let start = usize::try_from(offset).ok()?;
        let length = usize::try_from(size).ok()?;
        if length == 0 || length > 4096 {
            return None;
        }
        let end = start.checked_add(length)?;
        let value = bytes.get(start..end)?;
        let value = value.strip_suffix(&[0]).unwrap_or(value);
        return std::str::from_utf8(value)
            .ok()
            .and_then(nonempty_trimmed)
            .filter(|path| path.starts_with('/'));
    }
    None
}

fn read_elf_u16(bytes: &[u8], offset: usize, little_endian: bool) -> Option<u16> {
    let value: [u8; 2] = bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
    Some(if little_endian {
        u16::from_le_bytes(value)
    } else {
        u16::from_be_bytes(value)
    })
}

fn read_elf_u32(bytes: &[u8], offset: usize, little_endian: bool) -> Option<u32> {
    let value: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(if little_endian {
        u32::from_le_bytes(value)
    } else {
        u32::from_be_bytes(value)
    })
}

fn read_elf_u64(bytes: &[u8], offset: usize, little_endian: bool) -> Option<u64> {
    let value: [u8; 8] = bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?;
    Some(if little_endian {
        u64::from_le_bytes(value)
    } else {
        u64::from_be_bytes(value)
    })
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[allow(unsafe_code)]
fn glibc_via_gnu_get_libc_version() -> Option<NumericVersion> {
    // SAFETY: glibc documents this function as returning a process-lifetime,
    // non-null, NUL-terminated static string. The null check is retained as a
    // defensive boundary in case an unusual interposer violates that contract.
    let pointer = unsafe { libc::gnu_get_libc_version() };
    if pointer.is_null() {
        return None;
    }
    // SAFETY: the pointer contract is described above and the bytes are copied
    // before returning.
    let value = unsafe { CStr::from_ptr(pointer) }.to_str().ok()?;
    parse_numeric_version_prefix(value)
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn glibc_via_gnu_get_libc_version() -> Option<NumericVersion> {
    None
}

#[cfg(test)]
fn parse_glibc_output(output: &str) -> Option<NumericVersion> {
    match classify_libc_output(output) {
        LibcDetection::Glibc(version) => Some(version),
        LibcDetection::Musl(_) | LibcDetection::Unknown => None,
    }
}

fn classify_libc_output(output: &str) -> LibcDetection {
    let lowercase = output.to_ascii_lowercase();
    if lowercase.contains("musl") {
        return LibcDetection::Musl(first_numeric_version(output));
    }
    if lowercase.contains("glibc")
        || lowercase.contains("gnu libc")
        || lowercase.contains("gnu c library")
    {
        return first_numeric_version(output)
            .map(LibcDetection::Glibc)
            .unwrap_or(LibcDetection::Unknown);
    }
    LibcDetection::Unknown
}

fn first_numeric_version(value: &str) -> Option<NumericVersion> {
    let pattern = Regex::new(r"(?m)(?:^|[^0-9])([0-9]+(?:\.[0-9]+)+)").ok()?;
    pattern
        .captures_iter(value)
        .filter_map(|captures| captures.get(1))
        .find_map(|candidate| NumericVersion::from_str(candidate.as_str()).ok())
}

fn parse_numeric_version_prefix(value: &str) -> Option<NumericVersion> {
    let pattern = Regex::new(r"^\s*([0-9]+(?:\.[0-9]+)+)").ok()?;
    let candidate = pattern.captures(value)?.get(1)?.as_str();
    NumericVersion::from_str(candidate).ok()
}

#[derive(Debug, Deserialize)]
struct PythonProbeResult {
    implementation: String,
    version: Vec<u32>,
    soabi: Option<String>,
    cache_tag: Option<String>,
    platform: String,
    pointer_width: u16,
    free_threaded: bool,
    virtual_environment: Option<String>,
    packaging_available: bool,
    compatible_tags: Vec<String>,
}

fn detect_python(
    options: &DetectOptions,
    command_options: &RunCommandOptions,
    glibc: Option<&NumericVersion>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<PythonInfo> {
    let explicit = options.python.is_some();
    let candidates = python_candidates(options.python.as_ref());
    let mut failures = Vec::new();

    for candidate in &candidates {
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-c"),
            OsString::from(PYTHON_PROBE),
        ];
        let output = match run_command(candidate, &args, command_options) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!("{}: {error}", candidate.to_string_lossy()));
                if explicit {
                    break;
                }
                continue;
            }
        };
        if !output.status.success() {
            failures.push(format!(
                "{}: exited {} ({})",
                candidate.to_string_lossy(),
                output.status,
                bounded_lossy(&output.stderr, 256)
            ));
            if explicit {
                break;
            }
            continue;
        }
        let probe: PythonProbeResult = match serde_json::from_slice(&output.stdout) {
            Ok(probe) => probe,
            Err(error) => {
                failures.push(format!(
                    "{}: invalid probe JSON: {error}",
                    candidate.to_string_lossy()
                ));
                if explicit {
                    break;
                }
                continue;
            }
        };
        let python = match python_info_from_probe(probe, candidate, glibc) {
            Ok(python) => python,
            Err(error) => {
                failures.push(format!("{}: {error}", candidate.to_string_lossy()));
                if explicit {
                    break;
                }
                continue;
            }
        };
        match python.executable.to_str() {
            None => failures.push(format!(
                "{}: selected Python path is not valid UTF-8",
                candidate.to_string_lossy()
            )),
            Some(path) if path.chars().any(is_unsafe_terminal_character) => failures
                .push("selected Python path contains a terminal-unsafe character".to_owned()),
            Some(_) => {
                if python.implementation != "cpython" {
                    diagnostics.push(diagnostic(
                        DiagnosticCode::UnsupportedPythonImplementation,
                        DiagnosticSeverity::Error,
                        [("implementation", python.implementation.clone())],
                    ));
                }
                return Some(python);
            }
        }
        if explicit {
            break;
        }
    }

    let mut details = BTreeMap::new();
    if !candidates.is_empty() {
        details.insert(
            "candidates".to_owned(),
            candidates
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if let Some(last_error) = failures.last() {
        details.insert("last_error".to_owned(), truncate_string(last_error, 512));
    }
    diagnostics.push(Diagnostic {
        code: DiagnosticCode::PythonUnavailable,
        severity: DiagnosticSeverity::Error,
        details,
    });
    None
}

fn python_candidates(explicit: Option<&PathBuf>) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return vec![path.clone()];
    }

    let mut candidates = Vec::new();
    if let Some(virtual_environment) = env::var_os("VIRTUAL_ENV").filter(|value| !value.is_empty())
    {
        let root = PathBuf::from(virtual_environment);
        if cfg!(windows) {
            candidates.push(root.join("Scripts").join("python.exe"));
        } else {
            candidates.push(root.join("bin").join("python"));
        }
    }
    candidates.push(PathBuf::from("python3"));
    candidates.push(PathBuf::from("python"));

    let mut seen = HashSet::new();
    candidates.retain(|candidate| seen.insert(candidate.as_os_str().to_os_string()));
    candidates
}

fn python_info_from_probe(
    probe: PythonProbeResult,
    selected_executable: &Path,
    glibc: Option<&NumericVersion>,
) -> Result<PythonInfo, String> {
    if probe.version.len() < 2 || probe.version.len() > 4 {
        return Err("Python returned an invalid version vector".to_owned());
    }
    if probe.pointer_width != 32 && probe.pointer_width != 64 {
        return Err(format!("unsupported pointer width {}", probe.pointer_width));
    }
    if probe.platform.trim().is_empty() {
        return Err("Python returned an empty platform".to_owned());
    }
    let version = NumericVersion::new(probe.version)
        .map_err(|error| format!("invalid Python version: {error}"))?;
    let implementation = probe.implementation.trim().to_ascii_lowercase();
    if implementation.is_empty() {
        return Err("Python returned an empty implementation".to_owned());
    }

    let mut seen = HashSet::new();
    let packaging_tags = probe
        .compatible_tags
        .into_iter()
        .take(MAX_PYTHON_TAGS)
        .filter(|tag| valid_compatibility_tag(tag))
        .filter(|tag| seen.insert(tag.clone()))
        .collect::<Vec<_>>();
    let use_packaging = probe.packaging_available && !packaging_tags.is_empty();
    let compatible_tags = if use_packaging {
        packaging_tags
    } else {
        builtin_compatible_tags(
            &implementation,
            &version,
            probe.free_threaded,
            &probe.platform,
            glibc,
        )
    };

    // Keep the launcher selected by the caller. Resolving `sys.executable` or canonicalizing this
    // path would follow a venv's `bin/python` symlink to the base interpreter and would make both
    // generated install commands and verification target the wrong environment.
    let executable = resolve_invoked_executable(selected_executable);
    let virtual_environment = infer_virtual_environment(&executable).or_else(|| {
        probe
            .virtual_environment
            .and_then(|value| nonempty_trimmed(&value))
            .map(PathBuf::from)
    });
    Ok(PythonInfo {
        executable,
        implementation,
        version,
        soabi: probe.soabi.and_then(|value| nonempty_trimmed(&value)),
        cache_tag: probe.cache_tag.and_then(|value| nonempty_trimmed(&value)),
        platform: probe.platform,
        pointer_width: probe.pointer_width,
        free_threaded: probe.free_threaded,
        virtual_environment,
        compatible_tags,
        tag_source: if use_packaging {
            TagSource::Packaging
        } else {
            TagSource::Builtin
        },
    })
}

fn resolve_invoked_executable(selected: &Path) -> PathBuf {
    if !selected.is_absolute() && selected.components().count() == 1 {
        if let Some(path) = env::var_os("PATH") {
            for directory in env::split_paths(&path) {
                let candidate = directory.join(selected);
                if executable_file(&candidate) {
                    return if candidate.is_absolute() {
                        candidate
                    } else {
                        env::current_dir()
                            .map(|current| current.join(candidate.as_path()))
                            .unwrap_or(candidate)
                    };
                }
            }
        }
        return selected.to_path_buf();
    }

    if selected.is_absolute() {
        selected.to_path_buf()
    } else {
        env::current_dir()
            .map(|directory| directory.join(selected))
            .unwrap_or_else(|_| selected.to_path_buf())
    }
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn infer_virtual_environment(executable: &Path) -> Option<PathBuf> {
    let root = executable.parent()?.parent()?;
    let configuration = root.join("pyvenv.cfg");
    read_limited_text(&configuration).ok()?;
    Some(root.to_path_buf())
}

fn valid_compatibility_tag(tag: &str) -> bool {
    if tag.is_empty() || tag.len() > MAX_TAG_LENGTH || !tag.is_ascii() {
        return false;
    }
    let parts = tag.split('-').collect::<Vec<_>>();
    parts.len() == 3
        && parts.iter().all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
        })
}

fn builtin_compatible_tags(
    implementation: &str,
    version: &NumericVersion,
    free_threaded: bool,
    python_platform: &str,
    glibc: Option<&NumericVersion>,
) -> Vec<String> {
    let major = version.component(0);
    let minor = version.component(1);
    let normalized_platform = normalize_python_platform(python_platform);
    let mut platforms = Vec::new();

    if let (Some(glibc), Some(architecture)) = (glibc, normalized_platform.strip_prefix("linux_")) {
        if glibc.component(0) == 2 && glibc.component(1) >= 5 {
            let current_minor = glibc.component(1);
            let minimum_minor = current_minor.saturating_sub(63).max(5);
            for libc_minor in (minimum_minor..=current_minor).rev() {
                platforms.push(format!("manylinux_2_{libc_minor}_{architecture}"));
            }
            if current_minor >= 17 {
                platforms.push(format!("manylinux2014_{architecture}"));
            }
            if current_minor >= 12 && architecture == "x86_64" {
                platforms.push("manylinux2010_x86_64".to_owned());
            }
            if current_minor >= 5 && architecture == "x86_64" {
                platforms.push("manylinux1_x86_64".to_owned());
            }
        }
    }
    platforms.push(normalized_platform);

    let mut tags = Vec::new();
    if implementation == "cpython" {
        let interpreter = format!("cp{major}{minor}");
        let abi = format!("{interpreter}{}", if free_threaded { "t" } else { "" });
        for platform in &platforms {
            tags.push(format!("{interpreter}-{abi}-{platform}"));
            if !free_threaded {
                tags.push(format!("{interpreter}-abi3-{platform}"));
            }
            tags.push(format!("{interpreter}-none-{platform}"));
        }
    }
    tags.push(format!("py{major}-none-any"));
    if major == 3 {
        tags.push("py3-none-any".to_owned());
    }
    let mut seen = HashSet::new();
    tags.retain(|tag| seen.insert(tag.clone()));
    tags
}

fn normalize_python_platform(platform: &str) -> String {
    platform
        .trim()
        .chars()
        .map(|character| match character {
            '-' | '.' => '_',
            other => other.to_ascii_lowercase(),
        })
        .collect()
}

fn detect_nvidia(
    options: &DetectOptions,
    command_options: &RunCommandOptions,
    diagnostics: &mut Vec<Diagnostic>,
) -> NvidiaInfo {
    let visible_devices = env::var_os("CUDA_VISIBLE_DEVICES");
    if let Some(visible_devices) = &visible_devices {
        diagnostics.push(diagnostic(
            DiagnosticCode::CudaVisibleDevicesSet,
            DiagnosticSeverity::Warning,
            [(
                "value",
                truncate_string(&visible_devices.to_string_lossy(), 256),
            )],
        ));
    }

    let primary_args = [
        "--query-gpu=index,uuid,name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ];
    let primary = match run_command("nvidia-smi", &primary_args, command_options) {
        Ok(output) => output,
        Err(error) => {
            let info = nvidia_command_error(error, diagnostics);
            return require_requested_gpus(info, &options.gpu_indices, diagnostics);
        }
    };
    let primary_text = combined_output(&primary);
    if no_nvidia_devices(&primary_text) {
        let info = nvidia_no_devices(diagnostics);
        return require_requested_gpus(info, &options.gpu_indices, diagnostics);
    }

    let (all_gpus, legacy_query) = if primary.status.success() {
        match parse_nvidia_csv(&primary.stdout, true) {
            Ok(gpus) if !gpus.is_empty() => (gpus, false),
            Ok(_) => {
                let info = nvidia_no_devices(diagnostics);
                return require_requested_gpus(info, &options.gpu_indices, diagnostics);
            }
            Err(_) => match query_nvidia_legacy(command_options) {
                Ok(gpus) => (gpus, true),
                Err(error) => {
                    let info = nvidia_query_failure(error, diagnostics);
                    return require_requested_gpus(info, &options.gpu_indices, diagnostics);
                }
            },
        }
    } else {
        match query_nvidia_legacy(command_options) {
            Ok(gpus) => (gpus, true),
            Err(error) => {
                let info = nvidia_query_failure(error, diagnostics);
                return require_requested_gpus(info, &options.gpu_indices, diagnostics);
            }
        }
    };

    let all_gpus = if let Some(visible_devices) = visible_devices.as_deref() {
        match filter_cuda_visible_devices(all_gpus, visible_devices) {
            Ok(gpus) if gpus.is_empty() => {
                let info = nvidia_no_devices(diagnostics);
                return require_requested_gpus(info, &options.gpu_indices, diagnostics);
            }
            Ok(gpus) => gpus,
            Err(error) => {
                diagnostics.push(diagnostic(
                    DiagnosticCode::NvidiaInspectionFailed,
                    DiagnosticSeverity::Error,
                    [("error", truncate_string(&error, 512))],
                ));
                return empty_nvidia(NvidiaDetectionStatus::Failed);
            }
        }
    } else {
        all_gpus
    };

    let (gpus, invalid_indices) = select_gpus(all_gpus, &options.gpu_indices);
    if !invalid_indices.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidGpuSelection,
            DiagnosticSeverity::Error,
            [(
                "indices",
                invalid_indices
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )],
        ));
        return empty_nvidia(NvidiaDetectionStatus::Failed);
    }
    let unknown_capabilities = gpus
        .iter()
        .filter(|gpu| gpu.compute_capability.is_none())
        .map(|gpu| gpu.index.to_string())
        .collect::<Vec<_>>();
    if legacy_query || !unknown_capabilities.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticCode::ComputeCapabilityUnknown,
            DiagnosticSeverity::Warning,
            [(
                "indices",
                if unknown_capabilities.is_empty() {
                    "all".to_owned()
                } else {
                    unknown_capabilities.join(",")
                },
            )],
        ));
    }

    let versions = gpus
        .iter()
        .map(|gpu| gpu.driver_version.clone())
        .collect::<BTreeSet<_>>();
    if versions.len() > 1 {
        diagnostics.push(diagnostic(
            DiagnosticCode::InconsistentDriverVersions,
            DiagnosticSeverity::Warning,
            [(
                "versions",
                versions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )],
        ));
    }
    let driver_version = versions.into_iter().next();
    let reported_cuda_version = run_command("nvidia-smi", &[] as &[&str], command_options)
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| parse_reported_cuda(&String::from_utf8_lossy(&output.stdout)));

    NvidiaInfo {
        status: NvidiaDetectionStatus::Detected,
        driver_version,
        reported_cuda_version,
        gpus,
    }
}

fn query_nvidia_legacy(
    command_options: &RunCommandOptions,
) -> Result<Vec<NvidiaGpu>, NvidiaQueryError> {
    let args = [
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ];
    let output =
        run_command("nvidia-smi", &args, command_options).map_err(NvidiaQueryError::Command)?;
    let text = combined_output(&output);
    if no_nvidia_devices(&text) || (output.status.success() && output.stdout.is_empty()) {
        return Err(NvidiaQueryError::NoDevices);
    }
    if !output.status.success() {
        return Err(NvidiaQueryError::Failed(bounded_lossy(&output.stderr, 512)));
    }
    parse_nvidia_csv(&output.stdout, false).map_err(NvidiaQueryError::Parse)
}

#[derive(Debug)]
enum NvidiaQueryError {
    Command(CommandError),
    NoDevices,
    Failed(String),
    Parse(String),
}

fn nvidia_query_failure(error: NvidiaQueryError, diagnostics: &mut Vec<Diagnostic>) -> NvidiaInfo {
    match error {
        NvidiaQueryError::NoDevices => nvidia_no_devices(diagnostics),
        NvidiaQueryError::Command(error) => nvidia_command_error(error, diagnostics),
        NvidiaQueryError::Failed(message) | NvidiaQueryError::Parse(message) => {
            diagnostics.push(diagnostic(
                DiagnosticCode::NvidiaInspectionFailed,
                DiagnosticSeverity::Warning,
                [("error", truncate_string(&message, 512))],
            ));
            empty_nvidia(NvidiaDetectionStatus::Failed)
        }
    }
}

fn nvidia_command_error(error: CommandError, diagnostics: &mut Vec<Diagnostic>) -> NvidiaInfo {
    if error.is_not_found() {
        diagnostics.push(diagnostic(
            DiagnosticCode::NvidiaSmiUnavailable,
            DiagnosticSeverity::Warning,
            std::iter::empty::<(&str, &str)>(),
        ));
        empty_nvidia(NvidiaDetectionStatus::CommandUnavailable)
    } else if error.is_timeout() {
        diagnostics.push(diagnostic(
            DiagnosticCode::NvidiaInspectionTimedOut,
            DiagnosticSeverity::Warning,
            [("error", error.to_string())],
        ));
        empty_nvidia(NvidiaDetectionStatus::TimedOut)
    } else {
        diagnostics.push(diagnostic(
            DiagnosticCode::NvidiaInspectionFailed,
            DiagnosticSeverity::Warning,
            [("error", truncate_string(&error.to_string(), 512))],
        ));
        empty_nvidia(NvidiaDetectionStatus::Failed)
    }
}

fn nvidia_no_devices(diagnostics: &mut Vec<Diagnostic>) -> NvidiaInfo {
    diagnostics.push(diagnostic(
        DiagnosticCode::NvidiaNoDevices,
        DiagnosticSeverity::Info,
        std::iter::empty::<(&str, &str)>(),
    ));
    empty_nvidia(NvidiaDetectionStatus::NoDevices)
}

fn empty_nvidia(status: NvidiaDetectionStatus) -> NvidiaInfo {
    NvidiaInfo {
        status,
        driver_version: None,
        reported_cuda_version: None,
        gpus: Vec::new(),
    }
}

fn require_requested_gpus(
    mut info: NvidiaInfo,
    selection: &[u32],
    diagnostics: &mut Vec<Diagnostic>,
) -> NvidiaInfo {
    if selection.is_empty() || !info.gpus.is_empty() {
        return info;
    }
    diagnostics.push(diagnostic(
        DiagnosticCode::InvalidGpuSelection,
        DiagnosticSeverity::Error,
        [(
            "indices",
            selection
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )],
    ));
    info.status = NvidiaDetectionStatus::Failed;
    info
}

fn parse_nvidia_csv(bytes: &[u8], with_compute_capability: bool) -> Result<Vec<NvidiaGpu>, String> {
    let expected_fields = if with_compute_capability { 5 } else { 4 };
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .flexible(false)
        .from_reader(bytes);
    let mut gpus = Vec::new();
    let mut indices = HashSet::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("invalid nvidia-smi CSV: {error}"))?;
        if record.len() != expected_fields {
            return Err(format!(
                "nvidia-smi returned {} fields; expected {expected_fields}",
                record.len()
            ));
        }
        let index = record[0]
            .parse::<u32>()
            .map_err(|_| format!("invalid GPU index: {}", &record[0]))?;
        if !indices.insert(index) {
            return Err(format!("duplicate GPU index: {index}"));
        }
        let uuid = optional_nvidia_value(&record[1]);
        let name = record[2].trim().to_owned();
        if name.is_empty() {
            return Err(format!("GPU {index} has an empty name"));
        }
        let driver_version = NumericVersion::from_str(record[3].trim())
            .map_err(|error| format!("invalid driver version for GPU {index}: {error}"))?;
        let compute_capability = if with_compute_capability {
            optional_nvidia_value(&record[4])
                .map(|value| ComputeCapability::from_str(&value))
                .transpose()
                .map_err(|error| format!("invalid compute capability for GPU {index}: {error}"))?
        } else {
            None
        };
        gpus.push(NvidiaGpu {
            index,
            uuid,
            name,
            compute_capability,
            driver_version,
        });
    }
    gpus.sort_by_key(|gpu| gpu.index);
    Ok(gpus)
}

fn optional_nvidia_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("n/a")
        || value.eq_ignore_ascii_case("not supported")
        || value == "[N/A]"
    {
        None
    } else {
        Some(value.to_owned())
    }
}

fn select_gpus(all_gpus: Vec<NvidiaGpu>, selection: &[u32]) -> (Vec<NvidiaGpu>, Vec<u32>) {
    if selection.is_empty() {
        return (all_gpus, Vec::new());
    }
    let mut by_index = all_gpus
        .into_iter()
        .map(|gpu| (gpu.index, gpu))
        .collect::<BTreeMap<_, _>>();
    let mut requested = BTreeSet::new();
    let mut selected = Vec::new();
    let mut invalid = Vec::new();
    for index in selection {
        if !requested.insert(*index) {
            continue;
        }
        if let Some(gpu) = by_index.remove(index) {
            selected.push(gpu);
        } else {
            invalid.push(*index);
        }
    }
    selected.sort_by_key(|gpu| gpu.index);
    (selected, invalid)
}

fn filter_cuda_visible_devices(
    all_gpus: Vec<NvidiaGpu>,
    value: &OsStr,
) -> Result<Vec<NvidiaGpu>, String> {
    let value = value.to_string_lossy();
    let tokens = value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() || tokens == ["-1"] {
        return Ok(Vec::new());
    }

    let numeric = tokens
        .iter()
        .map(|token| token.parse::<u32>())
        .collect::<Result<Vec<_>, _>>();
    let selected = if let Ok(indices) = numeric {
        let unique = indices.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != indices.len() {
            return Err("CUDA_VISIBLE_DEVICES contains a duplicate GPU ordinal".to_owned());
        }
        let complete_ordinal_set = unique.len() == all_gpus.len()
            && unique
                .iter()
                .copied()
                .eq(0..u32::try_from(all_gpus.len()).unwrap_or(u32::MAX));
        if all_gpus.len() == 1 && indices == [0] || complete_ordinal_set {
            // CUDA ordinals are not guaranteed to equal NVML/nvidia-smi indices. A singleton or
            // the complete set is safe for static compatibility because the GPU union is known;
            // order is irrelevant and explicit verification later pins UUIDs.
            all_gpus
        } else {
            return Err(
                "numeric CUDA_VISIBLE_DEVICES is a subset whose CUDA ordinals cannot be safely mapped to nvidia-smi physical indices; use GPU UUIDs"
                    .to_owned(),
            );
        }
    } else {
        let mut selected = Vec::new();
        for token in tokens {
            if token.starts_with("MIG-") {
                return Err(
                    "MIG CUDA visibility cannot be mapped to physical nvidia-smi GPUs".to_owned(),
                );
            }
            let matches = all_gpus
                .iter()
                .filter(|gpu| {
                    gpu.uuid
                        .as_deref()
                        .is_some_and(|uuid| uuid.starts_with(token))
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(format!(
                    "CUDA_VISIBLE_DEVICES identifier {token:?} did not match exactly one GPU"
                ));
            }
            let gpu = (*matches[0]).clone();
            if selected
                .iter()
                .any(|existing: &NvidiaGpu| existing.index == gpu.index)
            {
                return Err("CUDA_VISIBLE_DEVICES contains a duplicate GPU".to_owned());
            }
            selected.push(gpu);
        }
        selected
    };
    Ok(selected)
}

fn parse_reported_cuda(output: &str) -> Option<NumericVersion> {
    let pattern = Regex::new(r"(?i)CUDA\s+Version\s*:\s*([0-9]+(?:\.[0-9]+){1,2})").ok()?;
    let version = pattern.captures(output)?.get(1)?.as_str();
    NumericVersion::from_str(version).ok()
}

fn no_nvidia_devices(output: &str) -> bool {
    let output = output.to_ascii_lowercase();
    output.contains("no devices were found")
        || output.contains("no nvidia devices")
        || output.contains("no device found")
}

fn combined_output(output: &CapturedOutput) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

fn detect_cuda_toolkit(command_options: &RunCommandOptions) -> Option<CudaToolkitInfo> {
    for (variable, source) in [
        ("CUDA_HOME", ToolkitSource::CudaHome),
        ("CUDA_PATH", ToolkitSource::CudaPath),
    ] {
        if let Some(root) = env::var_os(variable)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            if let Some(version) = toolkit_version_at_root(&root, command_options) {
                return Some(CudaToolkitInfo {
                    version,
                    root: utf8_path(root),
                    source,
                });
            }
        }
    }

    if let Some(executable) = find_executable_in_path("nvcc") {
        if let Some(version) = nvcc_version(&executable, command_options) {
            let root = executable
                .parent()
                .and_then(Path::parent)
                .map(Path::to_path_buf);
            return Some(CudaToolkitInfo {
                version,
                root: root.and_then(utf8_path),
                source: ToolkitSource::Nvcc,
            });
        }
    }

    let default_root = Path::new("/usr/local/cuda");
    if let Some(version) = toolkit_version_at_root(default_root, command_options) {
        return Some(CudaToolkitInfo {
            version,
            root: Some(default_root.to_path_buf()),
            source: ToolkitSource::VersionJson,
        });
    }

    let mut versioned_roots = std::fs::read_dir("/usr/local")
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with("cuda-"))
        })
        .filter_map(|root| {
            toolkit_version_at_root(&root, command_options).map(|version| (version, root))
        })
        .collect::<Vec<_>>();
    versioned_roots.sort_by(|left, right| right.0.cmp(&left.0));
    versioned_roots
        .into_iter()
        .next()
        .map(|(version, root)| CudaToolkitInfo {
            version,
            root: utf8_path(root),
            source: ToolkitSource::VersionJson,
        })
}

fn utf8_path(path: PathBuf) -> Option<PathBuf> {
    if path.to_str().is_some() {
        Some(path)
    } else {
        None
    }
}

fn toolkit_version_at_root(
    root: &Path,
    command_options: &RunCommandOptions,
) -> Option<NumericVersion> {
    let version_json = root.join("version.json");
    if let Ok(contents) = read_limited_text(&version_json) {
        if let Some(version) = parse_cuda_version_json(&contents) {
            return Some(version);
        }
    }
    let version_text = root.join("version.txt");
    if let Ok(contents) = read_limited_text(&version_text) {
        if let Some(version) = first_numeric_version(&contents) {
            return Some(version);
        }
    }
    let nvcc = if cfg!(windows) {
        root.join("bin").join("nvcc.exe")
    } else {
        root.join("bin").join("nvcc")
    };
    nvcc_version(&nvcc, command_options)
}

fn parse_cuda_version_json(contents: &str) -> Option<NumericVersion> {
    let value: Value = serde_json::from_str(contents).ok()?;
    let version = value
        .pointer("/cuda/version")
        .or_else(|| value.get("version"))?;
    match version {
        Value::String(version) => parse_numeric_version_prefix(version),
        Value::Number(version) => parse_numeric_version_prefix(&version.to_string()),
        _ => None,
    }
}

fn nvcc_version(executable: &Path, command_options: &RunCommandOptions) -> Option<NumericVersion> {
    let output = run_command(executable, &["--version"], command_options).ok()?;
    if !output.status.success() {
        return None;
    }
    let text = combined_output(&output);
    parse_nvcc_output(&text)
}

fn parse_nvcc_output(text: &str) -> Option<NumericVersion> {
    let release = Regex::new(r"(?i)\brelease\s+([0-9]+(?:\.[0-9]+){1,2})")
        .ok()?
        .captures(text)
        .and_then(|captures| captures.get(1))
        .and_then(|version| NumericVersion::from_str(version.as_str()).ok());
    release.or_else(|| {
        Regex::new(r"\bV([0-9]+(?:\.[0-9]+){1,2})")
            .ok()?
            .captures(text)
            .and_then(|captures| captures.get(1))
            .and_then(|version| NumericVersion::from_str(version.as_str()).ok())
    })
}

fn find_executable_in_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if cfg!(windows) {
            let candidate = directory.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn bounded_lossy(bytes: &[u8], limit: usize) -> String {
    let retained = bytes.get(..bytes.len().min(limit)).unwrap_or(bytes);
    truncate_string(&String::from_utf8_lossy(retained), limit)
}

fn truncate_string(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut boundary = max_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}...", &value[..boundary])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(value: &str) -> NumericVersion {
        value.parse().expect("valid test version")
    }

    #[test]
    fn parses_os_release_without_shell_evaluation() {
        let input = r#"
NAME="Ubuntu"
VERSION_ID="22.04"
PRETTY_NAME="Ubuntu 22.04.4 LTS"
MALFORMED='unterminated
"#;
        assert_eq!(
            parse_os_release(input).as_deref(),
            Some("Ubuntu 22.04.4 LTS")
        );
    }

    #[test]
    fn falls_back_to_name_and_version_id() {
        assert_eq!(
            parse_os_release("NAME=Debian\nVERSION_ID='12'\n").as_deref(),
            Some("Debian 12")
        );
    }

    #[test]
    fn recognizes_glibc_but_not_musl() {
        assert_eq!(
            parse_glibc_output("ldd (Ubuntu GLIBC 2.35-0ubuntu3.9) 2.35"),
            Some(version("2.35"))
        );
        assert_eq!(
            parse_glibc_output("musl libc (x86_64)\nVersion 1.2.5"),
            None
        );
        assert_eq!(
            classify_libc_output("musl libc (x86_64)\nVersion 1.2.5"),
            LibcDetection::Musl(Some(version("1.2.5")))
        );
        assert_eq!(classify_libc_output("unknown ldd"), LibcDetection::Unknown);
    }

    #[test]
    fn parses_a_bounded_elf_program_interpreter() {
        let interpreter = b"/lib64/ld-linux-x86-64.so.2\0";
        let mut elf = vec![0_u8; 128 + interpreter.len()];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[32..40].copy_from_slice(&64_u64.to_le_bytes());
        elf[54..56].copy_from_slice(&56_u16.to_le_bytes());
        elf[56..58].copy_from_slice(&1_u16.to_le_bytes());
        elf[64..68].copy_from_slice(&3_u32.to_le_bytes());
        elf[72..80].copy_from_slice(&128_u64.to_le_bytes());
        elf[96..104].copy_from_slice(&(interpreter.len() as u64).to_le_bytes());
        elf[128..].copy_from_slice(interpreter);

        assert_eq!(
            elf_interpreter(&elf).as_deref(),
            Some("/lib64/ld-linux-x86-64.so.2")
        );
        elf[54..56].copy_from_slice(&8_u16.to_le_bytes());
        assert!(elf_interpreter(&elf).is_none());
    }

    #[test]
    fn selected_venv_launcher_is_preserved_and_inferred_with_isolated_site_disabled() {
        let directory = tempfile::tempdir().expect("temporary venv");
        let root = directory.path().join("venv");
        let executable = root.join("bin").join("python");
        std::fs::create_dir_all(executable.parent().expect("bin directory"))
            .expect("create venv bin");
        std::fs::write(root.join("pyvenv.cfg"), "home = /usr/bin\n").expect("write pyvenv.cfg");
        let probe: PythonProbeResult = serde_json::from_value(serde_json::json!({
            "implementation": "CPython",
            "version": [3, 11, 9],
            "soabi": "cpython-311-x86_64-linux-gnu",
            "cache_tag": "cpython-311",
            "platform": "linux-x86_64",
            "pointer_width": 64,
            "free_threaded": false,
            "virtual_environment": null,
            "packaging_available": false,
            "compatible_tags": []
        }))
        .expect("probe fixture");

        let python = python_info_from_probe(probe, &executable, Some(&version("2.35")))
            .expect("valid Python probe");

        assert_eq!(python.executable, executable);
        assert_eq!(python.virtual_environment.as_deref(), Some(root.as_path()));
    }

    #[test]
    fn validates_packaging_tags_defensively() {
        assert!(valid_compatibility_tag("cp313-cp313-manylinux_2_17_x86_64"));
        assert!(!valid_compatibility_tag("cp313-cp313"));
        assert!(!valid_compatibility_tag("cp313-cp313-manylinux-x86-64"));
        assert!(!valid_compatibility_tag("cp313-cp313-💥"));
    }

    #[test]
    fn builtin_tags_are_conservative_and_manylinux_aware() {
        let tags = builtin_compatible_tags(
            "cpython",
            &version("3.13.2"),
            false,
            "linux-x86_64",
            Some(&version("2.35")),
        );
        assert_eq!(tags[0], "cp313-cp313-manylinux_2_35_x86_64");
        assert!(tags.contains(&"cp313-cp313-manylinux_2_17_x86_64".to_owned()));
        assert!(tags.contains(&"cp313-cp313-manylinux2014_x86_64".to_owned()));
        assert!(tags.contains(&"py3-none-any".to_owned()));
    }

    #[test]
    fn parses_current_nvidia_csv() {
        let csv = include_bytes!("../tests/fixtures/nvidia/a100.csv");
        let gpus = parse_nvidia_csv(csv, true).expect("valid CSV");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(
            gpus[0].compute_capability,
            Some(ComputeCapability { major: 8, minor: 0 })
        );
        assert_eq!(gpus[0].name, "NVIDIA A100-SXM4-40GB");
        assert_eq!(gpus[0].driver_version.to_string(), "530.30.02");
        assert_eq!(
            serde_json::to_value(&gpus[0]).expect("serialize GPU")["driver_version"],
            "530.30.02"
        );
    }

    #[test]
    fn legacy_nvidia_csv_leaves_compute_capability_unknown() {
        let gpus = parse_nvidia_csv(
            include_bytes!("../tests/fixtures/nvidia/v100-legacy.csv"),
            false,
        )
        .expect("valid legacy CSV");
        assert_eq!(gpus[0].compute_capability, None);
    }

    #[test]
    fn no_device_fixture_is_recognized() {
        assert!(no_nvidia_devices(include_str!(
            "../tests/fixtures/nvidia/no-devices.txt"
        )));
    }

    #[test]
    fn nvcc_fixture_reports_the_release_version() {
        assert_eq!(
            parse_nvcc_output(include_str!("../tests/fixtures/cuda/nvcc-12.4.txt")),
            Some(version("12.4"))
        );
    }

    #[test]
    fn cpython_fixtures_generate_native_builtin_tags() {
        for (fixture, expected) in [
            (
                include_str!("../tests/fixtures/python/cp313.json"),
                "cp313-cp313-manylinux_2_35_x86_64",
            ),
            (
                include_str!("../tests/fixtures/python/cp311.json"),
                "cp311-cp311-manylinux_2_35_x86_64",
            ),
        ] {
            let probe = serde_json::from_str(fixture).expect("valid Python fixture");
            let python = python_info_from_probe(
                probe,
                Path::new("/usr/bin/python3"),
                Some(&version("2.35")),
            )
            .expect("valid Python info");
            assert_eq!(
                python.compatible_tags.first().map(String::as_str),
                Some(expected)
            );
        }
    }

    #[test]
    fn nvidia_csv_rejects_duplicate_indices() {
        let csv = b"0, GPU-a, A100, 530.30.02, 8.0\n0, GPU-b, A100, 530.30.02, 8.0\n";
        assert!(parse_nvidia_csv(csv, true).is_err());
    }

    #[test]
    fn parses_driver_reported_cuda_for_information_only() {
        let output = "| NVIDIA-SMI 530.30.02 Driver Version: 530.30.02 CUDA Version: 12.1 |";
        assert_eq!(parse_reported_cuda(output), Some(version("12.1")));
    }

    #[test]
    fn gpu_selection_uses_physical_indices_and_reports_missing_values() {
        let gpus = vec![
            NvidiaGpu {
                index: 0,
                uuid: None,
                name: "A".to_owned(),
                compute_capability: None,
                driver_version: version("530.30.02"),
            },
            NvidiaGpu {
                index: 2,
                uuid: None,
                name: "B".to_owned(),
                compute_capability: None,
                driver_version: version("530.30.02"),
            },
        ];
        let (selected, invalid) = select_gpus(gpus, &[2, 9, 2]);
        assert_eq!(
            selected.iter().map(|gpu| gpu.index).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(invalid, vec![9]);
    }

    #[test]
    fn parses_cuda_version_json_variants() {
        assert_eq!(
            parse_cuda_version_json(r#"{"cuda":{"name":"CUDA SDK","version":"12.4.1"}}"#),
            Some(version("12.4.1"))
        );
        assert_eq!(
            parse_cuda_version_json(r#"{"version":"11.8"}"#),
            Some(version("11.8"))
        );
    }

    #[test]
    fn bounded_reader_retains_only_the_configured_prefix() {
        let result = read_bounded(&b"abcdefgh"[..], 4).expect("read succeeds");
        assert_eq!(result.bytes, b"abcd");
        assert!(result.exceeded);
    }

    #[cfg(unix)]
    #[test]
    fn command_runner_enforces_timeout() {
        let _guard = crate::process::lock_subprocess_tests();
        let options = RunCommandOptions {
            timeout: Duration::from_millis(20),
            max_stdout_bytes: 1024,
            max_stderr_bytes: 1024,
        };
        let error = run_command("sleep", &["1"], &options).expect_err("must time out");
        assert!(matches!(error, CommandError::TimedOut { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn command_runner_reports_output_limit() {
        let _guard = crate::process::lock_subprocess_tests();
        let options = RunCommandOptions {
            timeout: Duration::from_secs(2),
            max_stdout_bytes: 4,
            max_stderr_bytes: 1024,
        };
        let error = run_command("printf", &["abcdefgh"], &options).expect_err("must exceed limit");
        assert!(matches!(
            error,
            CommandError::OutputLimitExceeded {
                stream: OutputStream::Stdout,
                limit: 4,
                ..
            }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn command_runner_does_not_wait_for_descendants_holding_pipes() {
        let _guard = crate::process::lock_subprocess_tests();
        let options = RunCommandOptions {
            timeout: Duration::from_secs(2),
            max_stdout_bytes: 1024,
            max_stderr_bytes: 1024,
        };
        let started = std::time::Instant::now();
        let output = run_command("sh", &["-c", "sleep 5 &"], &options).expect("parent exits");
        assert!(output.status.success());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "inherited pipes must not keep the runner blocked"
        );
    }

    fn gpu(index: u32, uuid: &str) -> NvidiaGpu {
        NvidiaGpu {
            index,
            uuid: Some(uuid.to_owned()),
            name: format!("GPU {index}"),
            compute_capability: Some(ComputeCapability { major: 8, minor: 0 }),
            driver_version: version("580.65.06"),
        }
    }

    #[test]
    fn cuda_visibility_filters_physical_gpus_in_logical_order() {
        let error = filter_cuda_visible_devices(
            vec![gpu(0, "GPU-zero"), gpu(1, "GPU-one"), gpu(2, "GPU-two")],
            OsStr::new("2,0"),
        )
        .expect_err("numeric CUDA subset cannot be mapped safely");
        assert!(error.contains("cannot be safely mapped"));

        let complete = filter_cuda_visible_devices(
            vec![gpu(0, "GPU-zero"), gpu(1, "GPU-one")],
            OsStr::new("1,0"),
        )
        .expect("the complete GPU union is safe");
        assert_eq!(complete.len(), 2);

        let visible_by_uuid = filter_cuda_visible_devices(
            vec![gpu(0, "GPU-zero"), gpu(1, "GPU-one")],
            OsStr::new("GPU-one"),
        )
        .expect("UUID visibility mapping");
        assert_eq!(visible_by_uuid[0].index, 1);
    }

    #[test]
    fn requested_gpu_is_blocking_when_no_gpu_can_be_detected() {
        let mut diagnostics = Vec::new();
        let info = require_requested_gpus(
            empty_nvidia(NvidiaDetectionStatus::NoDevices),
            &[2],
            &mut diagnostics,
        );
        assert_eq!(info.status, NvidiaDetectionStatus::Failed);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::InvalidGpuSelection
                && diagnostic.severity == DiagnosticSeverity::Error
        }));
    }

    #[test]
    fn non_regular_text_inputs_are_rejected() {
        #[cfg(unix)]
        assert!(read_limited_text(Path::new("/dev/null")).is_err());
    }
}
