# Security policy

## Supported versions

Security fixes are provided for the latest released minor version of `torch-check`.

## Reporting a vulnerability

Please use [GitHub's private security-advisory feature](https://github.com/rice8y/torch-check/security/advisories/new) for this repository. Do not open a public issue containing an exploit, sensitive environment report, GPU UUID, or filesystem path. Include the affected version, platform, reproduction steps, and expected impact. Maintainers will acknowledge a complete report within seven days and coordinate disclosure after a fix is available.

## Trust boundaries

- Inspection and recommendation execute the selected Python interpreter with `-I -S` and no third-party imports. They also execute bounded system probes found through the current `PATH`, including `uname`, `ldd`, `nvidia-smi`, and `nvcc`, or `nvcc` under a configured `CUDA_HOME`. Use a trusted interpreter, filesystem, `PATH`, and `CUDA_HOME`; this is a diagnostic tool, not a sandbox.
- `verify` additionally imports and executes the `torch` package already installed in the selected Python environment. Run it only for environments you trust.
- Generated install commands are displayed but never executed.
- Package metadata is accepted only from the official HTTPS PyTorch host allowlist and is subject to redirect, timeout, and response-size checks.
- Cache directories and files are opened with restrictive permissions, symlinks are rejected, and complete snapshots are replaced atomically under a lock.

## Reports and redaction

Environment reports can expose interpreter and virtual-environment paths, CUDA Toolkit paths, GPU UUIDs, device names, and selected-device configuration. `--redact` removes known paths and UUIDs from structured fields and diagnostics, and should be used before sharing a report. Device model names, software versions, operating-system details, and other values needed to diagnose compatibility remain visible. Review redacted output before publishing it because free-form errors from operating-system tools may contain information that a future parser does not yet recognize.

`torch-check` has no telemetry. Its only intended network traffic is bounded public metadata retrieval from the official PyTorch wheel host.
