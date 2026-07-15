from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "observe_upstream_metadata.py"
SPEC = importlib.util.spec_from_file_location("observe_upstream_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OBSERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OBSERVER
SPEC.loader.exec_module(OBSERVER)


FIXTURE = """
# Releasing PyTorch

## Release Compatibility Matrix

| PyTorch version | Python | C++ | Stable CUDA | Experimental CUDA | Stable ROCm |
| --- | --- | --- | --- | --- | --- |
| 2.13 | >=3.10, <=3.15 | C++20 | CUDA 12.6, CUDA 13.0 | CUDA 13.2 | ROCm 7.2 |
| 2.12 | >=3.10, <=3.14 | C++17 | CUDA 12.6, CUDA 13.0 | -- | ROCm 7.1 |

### PyTorch CUDA Support Matrix

| CUDA | architectures supported for Linux x86 and Windows builds | notes |
| --- | --- | --- |
| 12.6.3 | Maxwell(5.0), Pascal(6.0), Hopper(9.0) | |
| 13.0.2 | Turing(7.5), Blackwell(10.0, 12.0+PTX) | +PTX on Linux only |
"""

NVIDIA_MINOR_FIXTURE = """
<table>
  <caption>CUDA Toolkit 11.x, 12.x, and 13.x Driver Version Ranges</caption>
  <tr><th>CUDA Toolkit</th><th>Minimum Driver Version</th><th>Upper Range</th></tr>
  <tr><td>CUDA 13.x</td><td>&gt;= 580</td><td>N/A</td></tr>
  <tr><td>CUDA 12.x</td><td>&gt;= 525</td><td>&lt; 580</td></tr>
  <tr><td>CUDA 11.x</td><td>&gt;= 450</td><td>&lt; 525</td></tr>
</table>
<p>450.80.02 (Linux) / 452.39 (Windows)</p>
"""

NVIDIA_RELEASE_ROWS = "\n".join(
    f"<tr><td>CUDA {version} GA</td><td>&gt;=500.1</td><td>N/A</td></tr>"
    for version in [
        "13.2", "13.1", "13.0", "12.9", "12.8", "12.7", "12.6",
        "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8",
        "11.7", "11.6", "11.5", "11.4", "11.3", "11.2", "11.1",
        "11.0",
    ]
)
NVIDIA_RELEASE_FIXTURE = f"""
<table>
  <caption>CUDA Toolkit and Corresponding Driver Versions</caption>
  <tr><th>CUDA Toolkit</th><th>Linux</th><th>Windows</th></tr>
  {NVIDIA_RELEASE_ROWS}
</table>
"""

CUDA_BUILD_210_FIXTURE = r'''#!/usr/bin/env bash
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
case ${CUDA_VERSION} in
    12.6) TORCH_CUDA_ARCH_LIST="5.0;6.0;${TORCH_CUDA_ARCH_LIST}" ;;
    12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};10.0;12.0" ;;
    12.9) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};10.0;12.0+PTX"
        if [[ "$PACKAGE_TYPE" == "libtorch" ]]; then
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//7.0;/}"
        fi
        ;;
    13.0)
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;$([[ "$ARCH" == "aarch64" ]] && echo "11.0;" || echo "")12.0+PTX"
        ;;
    *) exit 1 ;;
esac
'''

CUDA_BUILD_26_FIXTURE = r'''#!/usr/bin/env bash
TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6"
case ${CUDA_VERSION} in
    12.6)
        if [[ "$GPU_ARCH_TYPE" = "cuda-aarch64" ]]; then
            TORCH_CUDA_ARCH_LIST="9.0"
        else
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0"
        fi
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    11.8)
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};3.7;9.0"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    *) exit 1 ;;
esac
'''

CUDA_BUILD_27_FIXTURE = r'''#!/usr/bin/env bash
TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6"
case ${CUDA_VERSION} in
    12.8)
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    12.6)
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    *) exit 1 ;;
esac
'''

CUDA_BUILD_28_FIXTURE = r'''#!/usr/bin/env bash
case ${CUDA_VERSION} in
    12.8)
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"
        ;;
    12.9)
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        if [[ "$PACKAGE_TYPE" == "libtorch" ]]; then
            TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;12.0+PTX"
        fi
        ;;
    *) exit 1 ;;
esac
'''

CUDA_BUILD_29_FIXTURE = r'''#!/usr/bin/env bash
case ${CUDA_VERSION} in
    12.8)
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"
        ;;
    12.9)
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        ;;
    13.0)
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        ;;
    *) exit 1 ;;
esac
'''

CUDA_BUILD_211_FIXTURE = r'''#!/usr/bin/env bash
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0"
case ${CUDA_VERSION} in
    12.6) TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;${TORCH_CUDA_ARCH_LIST//10.0/}" ;;
    12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0" ;;
    12.9) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0+PTX" ;;
    13.0) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};$([[ "$ARCH" == "aarch64" ]] && echo "11.0;" || echo "")12.0" ;;
    *) exit 1 ;;
esac
'''

TAGGED_CUDA_BUILD_FIXTURES = [
    ("2.6", "v2.6.0", CUDA_BUILD_26_FIXTURE),
    ("2.7", "v2.7.0", CUDA_BUILD_27_FIXTURE),
    ("2.7", "v2.7.1", CUDA_BUILD_27_FIXTURE),
    ("2.8", "v2.8.0", CUDA_BUILD_28_FIXTURE),
    ("2.9", "v2.9.0", CUDA_BUILD_29_FIXTURE),
    ("2.9", "v2.9.1", CUDA_BUILD_29_FIXTURE),
    ("2.10", "v2.10.0", CUDA_BUILD_210_FIXTURE),
    ("2.11", "v2.11.0", CUDA_BUILD_211_FIXTURE),
]


class ObserverTests(unittest.TestCase):
    def test_extracts_review_fields_deterministically(self) -> None:
        snapshot = OBSERVER.build_snapshot(
            FIXTURE,
            NVIDIA_MINOR_FIXTURE,
            NVIDIA_RELEASE_FIXTURE,
            TAGGED_CUDA_BUILD_FIXTURES,
        )

        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual(snapshot["authority"], "observation_only")
        self.assertEqual(
            snapshot["release_compatibility"][0],
            {
                "series": "2.13",
                "python": ">=3.10, <=3.15",
                "cxx_standard": "C++20",
                "stable_cuda": ["12.6", "13.0"],
                "experimental_cuda": ["13.2"],
                "stable_rocm": "7.2",
            },
        )
        architectures = snapshot[
            "linux_x86_64_and_windows_cuda_architectures"
        ][1]
        self.assertEqual(architectures["compute_capabilities"], ["7.5", "10.0", "12.0"])
        self.assertEqual(architectures["ptx_capabilities"], ["12.0"])
        self.assertEqual(
            snapshot["nvidia_minor_version_compatibility"]["families"][0],
            {
                "cuda_major": 13,
                "minimum_driver_branch": "580",
                "upper_exclusive": None,
            },
        )
        self.assertEqual(
            snapshot["nvidia_toolkit_driver_versions"][0]["toolkit_version"],
            "13.2",
        )
        tagged = {
            (build["series"], architecture["variant"]): architecture
            for build in snapshot["pytorch_tagged_cuda_builds"]
            for architecture in build["architectures"]
        }
        self.assertEqual(
            tagged[("2.10", "cu128")]["compute_capabilities"],
            ["7.0", "7.5", "8.0", "8.6", "9.0", "10.0", "12.0"],
        )
        self.assertEqual(tagged[("2.10", "cu128")]["ptx_capabilities"], [])
        self.assertNotIn(
            "12.0", tagged[("2.10", "cu126")]["compute_capabilities"]
        )
        self.assertEqual(
            tagged[("2.11", "cu128")]["compute_capabilities"],
            ["7.5", "8.0", "8.6", "9.0", "10.0", "12.0"],
        )
        builds = snapshot["pytorch_tagged_cuda_builds"]
        self.assertEqual(
            [build["tag"] for build in builds],
            [
                "v2.11.0",
                "v2.10.0",
                "v2.9.0",
                "v2.9.1",
                "v2.8.0",
                "v2.7.0",
                "v2.7.1",
                "v2.6.0",
            ],
        )
        self.assertTrue(all(len(build["source_sha256"]) == 64 for build in builds))

    def test_extracts_legacy_literal_and_x86_conditional_architectures(self) -> None:
        cuda_26 = {
            entry["variant"]: entry
            for entry in OBSERVER.parse_cuda_build_architectures(CUDA_BUILD_26_FIXTURE)
        }
        self.assertEqual(
            cuda_26["cu126"]["compute_capabilities"],
            ["5.0", "6.0", "7.0", "7.5", "8.0", "8.6", "9.0"],
        )
        cuda_28 = {
            entry["variant"]: entry
            for entry in OBSERVER.parse_cuda_build_architectures(CUDA_BUILD_28_FIXTURE)
        }
        self.assertEqual(cuda_28["cu129"]["ptx_capabilities"], ["12.0"])

    def test_rejects_base_reference_when_no_base_is_defined(self) -> None:
        invalid = CUDA_BUILD_28_FIXTURE.replace(
            'TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"',
            'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0"',
            1,
        )
        with self.assertRaisesRegex(OBSERVER.ObservationError, "missing base"):
            OBSERVER.parse_cuda_build_architectures(invalid)

    def test_rejects_missing_architecture_table(self) -> None:
        with self.assertRaises(OBSERVER.ObservationError):
            OBSERVER.build_snapshot(
                FIXTURE.split("### PyTorch CUDA Support Matrix")[0],
                NVIDIA_MINOR_FIXTURE,
                NVIDIA_RELEASE_FIXTURE,
                TAGGED_CUDA_BUILD_FIXTURES,
            )

    def test_rejects_shell_execution_in_architecture_assignment(self) -> None:
        malicious = CUDA_BUILD_210_FIXTURE.replace(
            "${TORCH_CUDA_ARCH_LIST};10.0;12.0",
            "${TORCH_CUDA_ARCH_LIST};10.0;$(touch /tmp/observer-pwned)",
            1,
        )
        with self.assertRaisesRegex(
            OBSERVER.ObservationError, "unsupported shell expression"
        ):
            OBSERVER.parse_cuda_build_architectures(malicious)

    def test_rejects_unknown_control_flow_in_cuda_branch(self) -> None:
        conditional = CUDA_BUILD_211_FIXTURE.replace(
            '12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0" ;;',
            '12.8) if [[ "$SOMETHING" ]]; then\n'
            '        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0"\n'
            "        fi\n"
            "        ;;&",
        )
        with self.assertRaisesRegex(OBSERVER.ObservationError, "unsupported condition"):
            OBSERVER.parse_cuda_build_architectures(conditional)

    def test_comment_cannot_terminate_a_cuda_case_branch(self) -> None:
        unterminated = CUDA_BUILD_211_FIXTURE.replace(
            '12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0" ;;',
            '12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0"\n'
            "        # ;;",
        )
        with self.assertRaisesRegex(OBSERVER.ObservationError, "not terminated"):
            OBSERVER.parse_cuda_build_architectures(unterminated)

    def test_requires_every_pinned_release_document(self) -> None:
        with self.assertRaisesRegex(
            OBSERVER.ObservationError, "required release tags"
        ):
            OBSERVER.build_snapshot(
                FIXTURE,
                NVIDIA_MINOR_FIXTURE,
                NVIDIA_RELEASE_FIXTURE,
                TAGGED_CUDA_BUILD_FIXTURES[:-1],
            )

    def test_rejects_non_allowlisted_remote_source(self) -> None:
        with self.assertRaises(OBSERVER.ObservationError):
            OBSERVER.read_source("https://example.com/RELEASE.md")


if __name__ == "__main__":
    unittest.main()
