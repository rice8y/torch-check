//! End-to-end tests for CLI parsing, stream discipline, and documented exit codes.

use std::process::Command;

fn torch_check() -> Command {
    Command::new(env!("CARGO_BIN_EXE_torch-check"))
}

#[test]
fn redacted_json_parse_errors_are_structured_and_do_not_echo_values() {
    let secret = "/Users/alice/private-gpu-selection";
    let output = torch_check()
        .args(["--format", "json", "--redact", "--gpu", secret, "inspect"])
        .output()
        .expect("run torch-check");

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty(), "JSON mode reserves stderr");
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("one JSON error document");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["error"]["kind"], "usage");
    assert_eq!(value["error"]["code"], "invalid_arguments");
    let serialized = serde_json::to_string(&value).expect("serialize parsed output");
    assert!(!serialized.contains(secret));
    assert_eq!(
        value["error"]["message"],
        "invalid command-line arguments; run torch-check --help"
    );
}

#[test]
fn human_parse_errors_keep_layout_and_escape_argument_controls() {
    let output = torch_check()
        .arg("--candidates")
        .output()
        .expect("run torch-check with an invalid argument");

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 parse error");
    assert!(stderr.contains("found\n\nUsage: torch-check"), "{stderr}");
    assert!(!stderr.contains("found\\n\\nUsage:"), "{stderr}");

    let hostile_argument = "--bad\nINJECT\u{1b}[31m";
    let output = torch_check()
        .arg(hostile_argument)
        .output()
        .expect("run torch-check with a control character");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 parse error");
    assert!(stderr.contains("--bad\\nINJECT\\u{1b}[31m"), "{stderr}");
    assert!(!stderr.contains(hostile_argument), "{stderr}");
    assert!(stderr.contains("\n\nUsage: torch-check"), "{stderr}");

    let hostile_value = "cu12\nSOURCE_INJECT";
    let output = torch_check()
        .args(["explain", "torch==2.10.0", "--cuda"])
        .arg(hostile_value)
        .output()
        .expect("run torch-check with an invalid control-bearing value");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 parse error");
    assert!(stderr.contains("cu12\\nSOURCE_INJECT"), "{stderr}");
    assert!(!stderr.contains(hostile_value), "{stderr}");
    assert!(stderr.contains("\n\nFor more information"), "{stderr}");
}

#[test]
fn long_help_is_snapshotted() {
    let output = torch_check()
        .arg("--help")
        .output()
        .expect("run torch-check help");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help output");
    insta::assert_snapshot!(stdout);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn offline_mode_without_a_cache_returns_metadata_exit_code() {
    let cache = tempfile::tempdir().expect("temporary cache");
    let output = torch_check()
        .args(["--format", "json", "--offline", "--cache-dir"])
        .arg(cache.path())
        .arg("recommend")
        .output()
        .expect("run offline recommendation");

    assert_eq!(output.status.code(), Some(3));
    assert!(output.stderr.is_empty());
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("one JSON error document");
    assert_eq!(value["error"]["kind"], "metadata");
}
