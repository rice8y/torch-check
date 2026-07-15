#!/usr/bin/env python3
"""Validate reviewed rules against the last official-source observation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


MAX_REVIEW_AGE_DAYS = 90
PYTORCH_CUDA_BUILD_PATH = ".ci/manywheel/build_cuda.sh"
REQUIRED_TAGGED_CUDA_TAGS = {
    "2.6": ("v2.6.0",),
    "2.7": ("v2.7.0", "v2.7.1"),
    "2.8": ("v2.8.0",),
    "2.9": ("v2.9.0", "v2.9.1"),
    "2.10": ("v2.10.0",),
    "2.11": ("v2.11.0",),
}
MAIN_MATRIX_ARCHITECTURE_SERIES = {"2.12", "2.13"}


class ReviewError(RuntimeError):
    """Raised when reviewed rules and observed official metadata disagree."""


def _variant(cuda: str) -> str:
    components = cuda.split(".")
    if len(components) < 2 or not all(part.isdigit() for part in components[:2]):
        raise ReviewError(f"invalid observed CUDA version: {cuda!r}")
    return f"cu{int(components[0])}{int(components[1])}"


def _cuda_from_variant(variant: str) -> str:
    if not variant.startswith("cu") or not variant[2:].isdigit() or len(variant) < 4:
        raise ReviewError(f"invalid reviewed CUDA variant: {variant!r}")
    digits = variant[2:]
    return f"{int(digits[:-1])}.{int(digits[-1])}"


def _major_minor(cuda: str) -> str:
    components = cuda.split(".")
    if len(components) < 2 or not all(part.isdigit() for part in components[:2]):
        raise ReviewError(f"invalid observed CUDA version: {cuda!r}")
    return f"{int(components[0])}.{int(components[1])}"


def _series_key(series: str) -> tuple[int, int]:
    components = series.split(".")
    if len(components) != 2 or not all(part.isdigit() for part in components):
        raise ReviewError(f"invalid PyTorch release series: {series!r}")
    return int(components[0]), int(components[1])


def _review_date(data: dict[str, Any], path: Path, today: dt.date) -> None:
    try:
        reviewed = dt.date.fromisoformat(data["reviewed_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReviewError(f"{path} has no valid reviewed_at date") from error
    age = (today - reviewed).days
    if age < 0 or age > MAX_REVIEW_AGE_DAYS:
        raise ReviewError(
            f"{path} review age is {age} days; expected 0-{MAX_REVIEW_AGE_DAYS}"
        )
    if not data.get("sources"):
        raise ReviewError(f"{path} has no official source URLs")


def _capability_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ReviewError(f"tagged CUDA build {field} must be a non-empty list")
    if any(
        not isinstance(capability, str)
        or not capability
        or capability != capability.strip()
        or len(capability.split(".")) != 2
        or not all(component.isdigit() for component in capability.split("."))
        or any(str(int(component)) != component for component in capability.split("."))
        for capability in value
    ):
        raise ReviewError(f"tagged CUDA build {field} is malformed")
    if len(set(value)) != len(value):
        raise ReviewError(f"tagged CUDA build {field} contains duplicates")
    return value


def _tagged_cuda_architectures(
    observed: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str]]:
    builds = observed.get("pytorch_tagged_cuda_builds")
    if not isinstance(builds, list) or not builds:
        raise ReviewError("upstream observation has no tagged PyTorch CUDA builds")

    sources = observed.get("sources")
    if not isinstance(sources, list) or any(
        not isinstance(entry, dict) for entry in sources
    ):
        raise ReviewError("upstream observation sources must be a list of objects")
    source_entries = {
        (entry.get("series"), entry.get("tag"), entry.get("url"))
        for entry in sources
        if entry.get("kind") == "linux_x86_64_wheel_cuda_architectures"
    }
    architectures: dict[tuple[str, str], dict[str, Any]] = {}
    tagged_versions: set[str] = set()
    actual_tags: set[tuple[str, str]] = set()
    for build in builds:
        if not isinstance(build, dict):
            raise ReviewError("tagged PyTorch CUDA build entry must be an object")
        series = build.get("series")
        tag = build.get("tag")
        if not isinstance(series, str) or not isinstance(tag, str):
            raise ReviewError("tagged PyTorch CUDA build lacks series or tag")
        _series_key(series)
        if tag not in REQUIRED_TAGGED_CUDA_TAGS.get(series, ()):
            raise ReviewError(
                f"tagged PyTorch CUDA build {series} uses unexpected tag {tag}"
            )
        source_url = build.get("source_url")
        expected_url = (
            f"https://github.com/pytorch/pytorch/blob/{tag}/"
            f"{PYTORCH_CUDA_BUILD_PATH}"
        )
        if source_url != expected_url:
            raise ReviewError(f"tagged PyTorch CUDA build {series} has an invalid source URL")
        if (series, tag, source_url) not in source_entries:
            raise ReviewError(f"tagged PyTorch CUDA build {series} is absent from sources")
        digest = build.get("source_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReviewError(f"tagged PyTorch CUDA build {series} has an invalid SHA-256")
        if build.get("platform") != "linux_x86_64" or build.get(
            "package_type"
        ) != "wheel":
            raise ReviewError(f"tagged PyTorch CUDA build {series} has the wrong target")
        if (series, tag) in actual_tags:
            raise ReviewError(f"duplicate tagged PyTorch CUDA build for {tag}")
        actual_tags.add((series, tag))
        version = tag.removeprefix("v")
        tagged_versions.add(version)

        entries = build.get("architectures")
        if not isinstance(entries, list) or not entries:
            raise ReviewError(f"tagged PyTorch CUDA build {series} has no architectures")
        build_architectures: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ReviewError("tagged CUDA architecture entry must be an object")
            cuda = entry.get("cuda")
            variant = entry.get("variant")
            if not isinstance(cuda, str) or not isinstance(variant, str):
                raise ReviewError("tagged CUDA architecture lacks cuda or variant")
            if _variant(cuda) != variant:
                raise ReviewError(
                    f"tagged PyTorch {series} CUDA {cuda} has inconsistent variant {variant}"
                )
            capabilities = _capability_list(
                entry.get("compute_capabilities"), "compute_capabilities"
            )
            ptx_capabilities = entry.get("ptx_capabilities")
            if not isinstance(ptx_capabilities, list):
                raise ReviewError(
                    "tagged CUDA build ptx_capabilities must be a list"
                )
            if ptx_capabilities:
                _capability_list(ptx_capabilities, "ptx_capabilities")
            if not set(ptx_capabilities).issubset(capabilities):
                raise ReviewError(
                    f"tagged PyTorch {series} {variant} has PTX without a matching capability"
                )
            if variant in build_architectures:
                raise ReviewError(f"duplicate tagged architecture for {tag} {variant}")
            build_architectures[variant] = entry
            architectures[(version, variant)] = entry

    required_tags = {
        (series, tag)
        for series, tags in REQUIRED_TAGGED_CUDA_TAGS.items()
        for tag in tags
    }
    if actual_tags != required_tags:
        missing = sorted(required_tags - actual_tags)
        unexpected = sorted(actual_tags - required_tags)
        details = []
        if missing:
            details.append("missing " + ", ".join(tag for _series, tag in missing))
        if unexpected:
            details.append("unexpected " + ", ".join(tag for _series, tag in unexpected))
        raise ReviewError(
            "upstream observation has the wrong tagged CUDA build set: "
            + "; ".join(details)
        )
    return architectures, tagged_versions


def validate(
    drivers: dict[str, Any],
    releases: dict[str, Any],
    observed: dict[str, Any],
    *,
    today: dt.date,
    driver_path: Path = Path("data/cuda-driver-rules.json"),
    release_path: Path = Path("data/pytorch-release-rules.json"),
) -> None:
    """Raise ``ReviewError`` when reviewed data lacks observed support."""

    if observed.get("schema_version") != 2:
        raise ReviewError("unsupported upstream observation schema version")
    if releases.get("schema_version") != 3:
        raise ReviewError("unsupported reviewed PyTorch rule schema version")
    if observed.get("authority") != "observation_only":
        raise ReviewError("upstream observation must be marked observation_only")
    tagged_architectures, tagged_versions = _tagged_cuda_architectures(observed)
    _review_date(drivers, driver_path, today)
    _review_date(releases, release_path, today)

    exact_driver_pairs = {
        (_major_minor(entry["toolkit_version"]), entry["linux_min_driver"])
        for entry in observed.get("nvidia_toolkit_driver_versions", [])
    }
    minor = observed.get("nvidia_minor_version_compatibility", {})
    branches = {
        entry["cuda_major"]: entry["minimum_driver_branch"]
        for entry in minor.get("families", [])
    }
    exception = minor.get("cuda_11_family_exception", {})
    for family in drivers.get("families", []):
        major = family["cuda_major"]
        minimum = family["linux_min_driver"]
        if str(minimum).split(".", maxsplit=1)[0] != branches.get(major):
            raise ReviewError(f"CUDA {major} family minimum disagrees with NVIDIA branch")
        if major == 11:
            if minimum != exception.get("linux_min_driver"):
                raise ReviewError("CUDA 11 family exception disagrees with NVIDIA")
        elif (f"{major}.0", minimum) not in exact_driver_pairs:
            raise ReviewError(f"CUDA {major} family minimum lacks an exact NVIDIA row")

    for entry in drivers.get("variants", []):
        pair = (_cuda_from_variant(entry["variant"]), entry["linux_min_driver"])
        if pair not in exact_driver_pairs:
            raise ReviewError(
                f"{entry['variant']} minimum {entry['linux_min_driver']} "
                "does not appear in NVIDIA's toolkit table"
            )

    observed_release_entries = observed.get("release_compatibility", [])
    observed_releases = {
        entry["series"]: {
            _variant(cuda)
            for cuda in entry["stable_cuda"] + entry["experimental_cuda"]
        }
        | {"cpu"}
        for entry in observed_release_entries
    }
    if len(observed_releases) != len(observed_release_entries):
        raise ReviewError("observed PyTorch release series are not unique")
    reviewed_release_entries = releases.get("releases", [])
    reviewed_series = {entry["series"] for entry in reviewed_release_entries}
    if len(reviewed_series) != len(reviewed_release_entries):
        raise ReviewError("reviewed PyTorch release series are not unique")
    if not reviewed_series:
        raise ReviewError("reviewed PyTorch release rules are empty")
    oldest_reviewed = min(map(_series_key, reviewed_series))
    missing_releases = {
        series
        for series in observed_releases
        if _series_key(series) >= oldest_reviewed and series not in reviewed_series
    }
    if missing_releases:
        raise ReviewError(
            "reviewed rules are missing observed PyTorch releases: "
            + ", ".join(sorted(missing_releases))
        )

    for release in reviewed_release_entries:
        expected = observed_releases.get(release["series"])
        actual = set(release["variants"])
        if expected is not None and actual != expected:
            raise ReviewError(
                f"PyTorch {release['series']} variants {sorted(actual)} "
                f"disagree with observed {sorted(expected or [])}"
            )
        if release["preferred_variant"] not in actual:
            raise ReviewError(
                f"PyTorch {release['series']} preferred variant is unavailable"
            )

    reviewed_architecture_entries = releases.get("gpu_architectures", [])
    architecture_keys: list[tuple[str, str]] = []
    for rule in reviewed_architecture_entries:
        series = rule.get("series")
        versions = rule.get("versions")
        variants = rule.get("variants")
        if (
            not isinstance(series, str)
            or not isinstance(versions, list)
            or not versions
            or not isinstance(variants, list)
            or not variants
        ):
            raise ReviewError("reviewed GPU architecture rule is malformed")
        if len(set(versions)) != len(versions) or any(
            not isinstance(version, str)
            or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None
            or any(
                str(int(component)) != component for component in version.split(".")
            )
            or ".".join(version.split(".")[:2]) != series
            for version in versions
        ):
            raise ReviewError(
                f"reviewed GPU architecture rule {series} has invalid exact versions"
            )
        architecture_keys.extend(
            (version, variant) for version in versions for variant in variants
        )
    if len(set(architecture_keys)) != len(architecture_keys):
        raise ReviewError("reviewed GPU architecture rules overlap by version and variant")

    required_tagged_architectures = set(tagged_architectures)
    tagged_series = set(REQUIRED_TAGGED_CUDA_TAGS)
    reviewed_tagged_architecture_entries = [
        (version, variant)
        for rule in reviewed_architecture_entries
        for version in rule["versions"]
        if rule["series"] in tagged_series
        for variant in rule["variants"]
    ]
    reviewed_tagged_architectures = set(reviewed_tagged_architecture_entries)
    if len(reviewed_tagged_architectures) != len(
        reviewed_tagged_architecture_entries
    ):
        raise ReviewError("reviewed tagged architecture rules contain overlaps")
    if reviewed_tagged_architectures != required_tagged_architectures:
        missing = sorted(required_tagged_architectures - reviewed_tagged_architectures)
        unexpected = sorted(reviewed_tagged_architectures - required_tagged_architectures)
        details = []
        if missing:
            details.append(
                "missing "
                + ", ".join(f"PyTorch {version} {variant}" for version, variant in missing)
            )
        if unexpected:
            details.append(
                "unexpected "
                + ", ".join(
                    f"PyTorch {version} {variant}" for version, variant in unexpected
                )
            )
        raise ReviewError(
            "reviewed tagged architecture rules disagree with tagged build evidence: "
            + "; ".join(details)
        )

    observed_architectures = {
        _variant(entry["cuda"]): entry
        for entry in observed.get(
            "linux_x86_64_and_windows_cuda_architectures", []
        )
    }
    for rule in reviewed_architecture_entries:
        for version in rule["versions"]:
            for variant in rule["variants"]:
                if version in tagged_versions:
                    architecture = tagged_architectures.get((version, variant))
                elif rule["series"] in MAIN_MATRIX_ARCHITECTURE_SERIES:
                    architecture = observed_architectures.get(variant)
                else:
                    architecture = None
                if architecture is None:
                    raise ReviewError(
                        f"PyTorch {version} {variant} has no observed architecture table"
                    )
                if rule["cubin_capabilities"] != architecture["compute_capabilities"]:
                    raise ReviewError(
                        f"PyTorch {version} {variant} cubin capabilities disagree with PyTorch"
                    )
                expected_ptx = architecture["ptx_capabilities"]
                actual_ptx = (
                    [] if rule["ptx_capability"] is None else [rule["ptx_capability"]]
                )
                if actual_ptx != expected_ptx:
                    raise ReviewError(
                        f"PyTorch {version} {variant} PTX capability disagrees with PyTorch"
                    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"{path} must contain a JSON object")
    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivers", type=Path, default=Path("data/cuda-driver-rules.json")
    )
    parser.add_argument(
        "--releases", type=Path, default=Path("data/pytorch-release-rules.json")
    )
    parser.add_argument(
        "--observed", type=Path, default=Path("data/upstream-observed.json")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        validate(
            _load(args.drivers),
            _load(args.releases),
            _load(args.observed),
            today=dt.date.today(),
            driver_path=args.drivers,
            release_path=args.releases,
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"reviewed metadata check failed: malformed input: {error}", file=sys.stderr)
        return 1
    except ReviewError as error:
        print(f"reviewed metadata check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
