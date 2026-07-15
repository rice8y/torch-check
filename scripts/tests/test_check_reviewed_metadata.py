from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "check_reviewed_metadata.py"
SPEC = importlib.util.spec_from_file_location("check_reviewed_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def tagged_rule(observed: dict, series: str, variant: str) -> dict:
    build = next(
        entry
        for entry in observed["pytorch_tagged_cuda_builds"]
        if entry["series"] == series
    )
    architecture = next(
        entry for entry in build["architectures"] if entry["variant"] == variant
    )
    ptx = architecture["ptx_capabilities"]
    return {
        "series": series,
        "variants": [variant],
        "platform": "linux_x86_64",
        "cubin_capabilities": architecture["compute_capabilities"],
        "ptx_capability": ptx[0] if ptx else None,
    }


def reviewed_rule(releases: dict, series: str, variant: str) -> dict:
    return next(
        rule
        for rule in releases["gpu_architectures"]
        if rule["series"] == series and variant in rule["variants"]
    )


class ReviewedMetadataTests(unittest.TestCase):
    def test_repository_rules_match_the_observed_sources(self) -> None:
        CHECKER.validate(
            load("cuda-driver-rules.json"),
            load("pytorch-release-rules.json"),
            load("upstream-observed.json"),
            today=dt.date(2026, 7, 15),
        )

    def test_detects_a_driver_rule_not_present_upstream(self) -> None:
        drivers = copy.deepcopy(load("cuda-driver-rules.json"))
        drivers["variants"][0]["linux_min_driver"] = "999.0"
        with self.assertRaises(CHECKER.ReviewError):
            CHECKER.validate(
                drivers,
                load("pytorch-release-rules.json"),
                load("upstream-observed.json"),
                today=dt.date(2026, 7, 15),
            )

    def test_cuda_11_patch_version_is_compared_as_major_minor(self) -> None:
        observed = load("upstream-observed.json")
        matching = [
            entry
            for entry in observed["nvidia_toolkit_driver_versions"]
            if entry["toolkit_version"] == "11.0.1"
        ]
        self.assertEqual(matching[0]["linux_min_driver"], "450.36.06")
        CHECKER.validate(
            load("cuda-driver-rules.json"),
            load("pytorch-release-rules.json"),
            observed,
            today=dt.date(2026, 7, 15),
        )

    def test_detects_a_new_unreviewed_pytorch_release(self) -> None:
        observed = copy.deepcopy(load("upstream-observed.json"))
        observed["release_compatibility"].append(
            {
                "series": "99.0",
                "stable_cuda": ["13.2"],
                "experimental_cuda": [],
            }
        )
        with self.assertRaisesRegex(
            CHECKER.ReviewError, "missing observed PyTorch releases"
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                load("pytorch-release-rules.json"),
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_accepts_series_specific_tagged_architecture_evidence(self) -> None:
        observed = load("upstream-observed.json")
        releases = load("pytorch-release-rules.json")
        for series in ["2.10", "2.11"]:
            expected = tagged_rule(observed, series, "cu128")
            actual = reviewed_rule(releases, series, "cu128")
            self.assertEqual(actual["cubin_capabilities"], expected["cubin_capabilities"])
            self.assertEqual(actual["ptx_capability"], expected["ptx_capability"])
        CHECKER.validate(
            load("cuda-driver-rules.json"),
            releases,
            observed,
            today=dt.date(2026, 7, 15),
        )

    def test_requires_every_official_tagged_architecture_rule(self) -> None:
        observed = load("upstream-observed.json")
        releases = copy.deepcopy(load("pytorch-release-rules.json"))
        releases["gpu_architectures"] = [
            rule
            for rule in releases["gpu_architectures"]
            if not (rule["series"] == "2.11" and "cu128" in rule["variants"])
        ]
        with self.assertRaisesRegex(
            CHECKER.ReviewError,
            "reviewed tagged architecture rules disagree",
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_does_not_conflate_architectures_between_release_series(self) -> None:
        observed = load("upstream-observed.json")
        releases = copy.deepcopy(load("pytorch-release-rules.json"))
        rule = reviewed_rule(releases, "2.10", "cu128")
        rule["cubin_capabilities"] = tagged_rule(observed, "2.11", "cu128")[
            "cubin_capabilities"
        ]
        with self.assertRaisesRegex(
            CHECKER.ReviewError,
            "PyTorch 2.10.0 cu128 cubin capabilities disagree",
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_cu126_cannot_claim_sm120_from_cu128_evidence(self) -> None:
        observed = load("upstream-observed.json")
        releases = copy.deepcopy(load("pytorch-release-rules.json"))
        rule = reviewed_rule(releases, "2.10", "cu126")
        rule["cubin_capabilities"] = [*rule["cubin_capabilities"], "12.0"]
        with self.assertRaisesRegex(
            CHECKER.ReviewError,
            "PyTorch 2.10.0 cu126 cubin capabilities disagree",
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_tagged_series_does_not_fall_back_to_mutable_main_table(self) -> None:
        observed = copy.deepcopy(load("upstream-observed.json"))
        releases = load("pytorch-release-rules.json")
        rule = reviewed_rule(releases, "2.10", "cu128")
        build = next(
            entry
            for entry in observed["pytorch_tagged_cuda_builds"]
            if entry["series"] == "2.10"
        )
        build["architectures"] = [
            entry for entry in build["architectures"] if entry["variant"] != "cu128"
        ]
        observed["linux_x86_64_and_windows_cuda_architectures"].append(
            {
                "cuda": "12.8",
                "compute_capabilities": rule["cubin_capabilities"],
                "ptx_capabilities": [],
            }
        )
        with self.assertRaisesRegex(
            CHECKER.ReviewError,
            "reviewed tagged architecture rules disagree with tagged build evidence",
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_rejects_malformed_tagged_source_provenance(self) -> None:
        observed = copy.deepcopy(load("upstream-observed.json"))
        observed["pytorch_tagged_cuda_builds"][0]["source_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(CHECKER.ReviewError, "invalid SHA-256"):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                load("pytorch-release-rules.json"),
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_requires_the_exact_reviewed_tagged_build_set(self) -> None:
        observed = copy.deepcopy(load("upstream-observed.json"))
        observed["pytorch_tagged_cuda_builds"] = [
            build
            for build in observed["pytorch_tagged_cuda_builds"]
            if build["series"] != "2.10"
        ]
        with self.assertRaisesRegex(CHECKER.ReviewError, "wrong tagged CUDA build set"):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                load("pytorch-release-rules.json"),
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_requires_each_stable_patch_tag_to_match_the_reviewed_rule(self) -> None:
        observed = copy.deepcopy(load("upstream-observed.json"))
        build = next(
            entry
            for entry in observed["pytorch_tagged_cuda_builds"]
            if entry["tag"] == "v2.9.1"
        )
        architecture = next(
            entry for entry in build["architectures"] if entry["variant"] == "cu128"
        )
        architecture["compute_capabilities"].remove("12.0")
        with self.assertRaisesRegex(
            CHECKER.ReviewError, "PyTorch 2.9.1 cu128 cubin capabilities disagree"
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                load("pytorch-release-rules.json"),
                observed,
                today=dt.date(2026, 7, 15),
            )

    def test_future_patch_does_not_inherit_tagged_architecture_evidence(self) -> None:
        releases = copy.deepcopy(load("pytorch-release-rules.json"))
        rule = reviewed_rule(releases, "2.10", "cu126")
        rule["versions"].append("2.10.1")
        with self.assertRaisesRegex(
            CHECKER.ReviewError, "unexpected PyTorch 2.10.1 cu126"
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                load("upstream-observed.json"),
                today=dt.date(2026, 7, 15),
            )

    def test_requires_index_only_tagged_architecture_evidence_to_be_registered(self) -> None:
        observed = load("upstream-observed.json")
        releases = copy.deepcopy(load("pytorch-release-rules.json"))
        releases["gpu_architectures"] = [
            rule
            for rule in releases["gpu_architectures"]
            if not (rule["series"] == "2.10" and "cu129" in rule["variants"])
        ]
        with self.assertRaisesRegex(
            CHECKER.ReviewError, "missing PyTorch 2.10.0 cu129"
        ):
            CHECKER.validate(
                load("cuda-driver-rules.json"),
                releases,
                observed,
                today=dt.date(2026, 7, 15),
            )


if __name__ == "__main__":
    unittest.main()
