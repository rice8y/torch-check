# torch-check

`torch-check` inspects your Python and NVIDIA environment and recommends a compatible wheel from the official PyTorch index. Recommendations support CPython on Linux x86_64 with glibc, with a CPU fallback when no reviewed CUDA wheel qualifies.

## Install

Install to `~/.local/bin` using a POSIX shell. Pass `--version` or `--install-dir` after `sh -s --` to customize the installation.

```sh
curl -fsSL https://raw.githubusercontent.com/rice8y/torch-check/main/scripts/install.sh | sh
```

Or install from crates.io:

```sh
cargo install torch-check --locked
```

[GitHub Releases](https://github.com/rice8y/torch-check/releases) provides checksummed binaries for Linux x86_64 (tested on Ubuntu 22.04), Linux Arm64, macOS, and Windows x64. Use the Windows archive from PowerShell or Command Prompt. Linux Arm64, macOS, and Windows builds support inspection.

## Usage

```sh
torch-check                                    # Recommend a wheel
torch-check recommend --installer uv           # Generate a uv pip command
torch-check inspect                            # Inspect the environment
torch-check candidates --all                   # List candidates and exclusions
torch-check explain torch==2.12.1 --cuda cu130  # Explain a specific build
torch-check verify                             # Run checks on installed PyTorch
```

Use `--python /path/to/python` to target an interpreter, `--gpu 0,2` to select physical GPU indices from `nvidia-smi`, and `--format json` for [structured output](data/schemas/output-v1.schema.json). Run `torch-check --help` for all options.

Choose `pip`, `uv`, or `uv-add` with `--installer`. Generated commands target the inspected Python environment. `verify` imports the installed `torch` and runs checks on the selected GPUs. Use `--redact` before sharing reports. See [SECURITY.md](SECURITY.md) for security details.

## Compatibility

Wheel tags, glibc, GPU architecture, and the NVIDIA driver determine compatibility. The CUDA version displayed by `nvidia-smi` is informational. Official wheels can run without a local CUDA Toolkit.

`direct-compatible` meets the normal driver minimum. `minor-compatible` relies on CUDA minor-version compatibility, so PTX JIT or newer driver features can still fail. `unverified` indicates missing evidence, and `verified` requires a successful runtime check. Use `candidates --unverified` to include unverified builds.

Wheel metadata is cached for 24 hours. Use `--refresh` to fetch it again or `--offline` to use a complete local snapshot.

<details>
<summary>Compatibility sources</summary>

<!-- BEGIN GENERATED PRIMARY SOURCES -->
Primary sources (generated from the metadata files; do not edit this block manually):

Reviewed compatibility sources — latest registered PyTorch series: 2.15. A registered configuration is not proof of wheel availability or GPU architecture coverage; those are checked separately.

- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA CUDA 13.2.0 release driver table](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html#cuda-driver)
- [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [PyTorch CUDA architecture matrix (ref v2.15.0)](https://github.com/pytorch/pytorch/blob/v2.15.0/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch CUDA architecture matrix (ref v2.14.0)](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch CUDA architecture matrix (ref efe07ed21d89)](https://github.com/pytorch/pytorch/blob/efe07ed21d8969e4ddb161dab0976ec28d3f4d57/RELEASE.md#pytorch-cuda-support-matrix)
- [Previous PyTorch versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 2.15 release build matrix](https://github.com/pytorch/pytorch/blob/release/2.15/.github/scripts/generate_binary_build_matrix.py)
- [PyTorch 2.14 release build matrix](https://github.com/pytorch/pytorch/blob/release/2.14/.github/scripts/generate_binary_build_matrix.py)
- [PyTorch 2.11.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.10.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.9.1 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.9.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.8.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.7.1 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.7.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.7.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.6.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.6.0/.ci/manywheel/build_cuda.sh)
- [TorchVision 0.29 version metadata](https://github.com/pytorch/vision/blob/release/0.29/version.txt)
- [PyTorch issue #190355](https://github.com/pytorch/pytorch/issues/190355)

Observed upstream sources — latest observed PyTorch series: 2.15. Observations are not automatically approved compatibility rules, including when a document also appears in the reviewed list.

- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA CUDA 13.2.0 release driver table](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html#cuda-driver)
- [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
- [PyTorch CUDA architecture matrix (ref v2.15.0)](https://github.com/pytorch/pytorch/blob/v2.15.0/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch CUDA architecture matrix (ref v2.14.0)](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch CUDA architecture matrix (ref efe07ed21d89)](https://github.com/pytorch/pytorch/blob/efe07ed21d8969e4ddb161dab0976ec28d3f4d57/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch CUDA architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)
- [PyTorch 2.11.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.10.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.9.1 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.9.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.8.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.7.1 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.7.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.7.0/.ci/manywheel/build_cuda.sh)
- [PyTorch 2.6.0 Linux wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.6.0/.ci/manywheel/build_cuda.sh)

Wheel availability is queried separately from the [PyTorch official wheel index](https://download.pytorch.org/whl/).
<!-- END GENERATED PRIMARY SOURCES -->

</details>

## License

This project is distributed under either the MIT License or the Apache License, Version 2.0, at your option. See [LICENSE](https://github.com/rice8y/torch-check/blob/main/LICENSE) for details.
