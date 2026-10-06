#!/usr/bin/env python3
"""Synchronize the README source catalog with reviewed and observed metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN GENERATED PRIMARY SOURCES -->"
END = "<!-- END GENERATED PRIMARY SOURCES -->"
SOURCE_HOSTS = {"github.com", "docs.nvidia.com", "pytorch.org"}


class SourceError(Exception):
    """Invalid source metadata or ambiguous README generation boundary."""


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceError(f"{path} must contain a JSON object")
    return value


def source_url(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 or character in "[]()<>\\\"`"
        for character in value
    ):
        raise SourceError("source URL contains unsafe Markdown or whitespace")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in SOURCE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise SourceError(f"source URL is not an allowed official HTTPS URL: {value}")
    return value


def source_info(url: str) -> tuple[int, str, tuple[int, ...]]:
    """Classify by document role, independently of URL spelling.

    Reading order: NVIDIA compatibility basics and driver tables; PyTorch
    overview documents; release-specific build evidence; companion metadata;
    supplementary issues; other official sources. Versions sort newest first
    within each role, not across unrelated document types.
    """
    parsed = urlsplit(url)
    path = parsed.path
    if parsed.netloc == "docs.nvidia.com":
        if path.endswith("/minor-version-compatibility.html"):
            return 0, "NVIDIA CUDA minor-version compatibility", ()
        match = re.search(r"/cuda/archive/([0-9.]+)/", path)
        if match:
            return (
                1,
                f"NVIDIA CUDA {match[1]} release driver table",
                tuple(map(int, match[1].split("."))),
            )
    if parsed.netloc == "pytorch.org" and path == "/get-started/previous-versions/":
        return 4, "Previous PyTorch versions", ()
    if path == "/pytorch/pytorch/blob/main/RELEASE.md":
        if parsed.fragment == "pytorch-cuda-support-matrix":
            return 3, "PyTorch CUDA architecture matrix", ()
        return 2, "PyTorch release matrix", ()
    match = re.fullmatch(
        r"/pytorch/pytorch/blob/v([0-9.]+)/\.ci/manywheel/build_cuda\.sh", path
    )
    if match:
        return (
            6,
            f"PyTorch {match[1]} Linux wheel build configuration",
            tuple(map(int, match[1].split("."))),
        )
    match = re.fullmatch(
        r"/pytorch/pytorch/blob/release/([0-9.]+)/\.github/scripts/"
        r"generate_binary_build_matrix\.py", path
    )
    if match:
        return (
            5,
            f"PyTorch {match[1]} release build matrix",
            tuple(map(int, match[1].split("."))),
        )
    match = re.fullmatch(r"/pytorch/vision/blob/release/([0-9.]+)/version\.txt", path)
    if match:
        return (
            7,
            f"TorchVision {match[1]} version metadata",
            tuple(map(int, match[1].split("."))),
        )
    match = re.fullmatch(r"/pytorch/pytorch/issues/([0-9]+)", path)
    if match:
        return 8, f"PyTorch issue #{match[1]}", (int(match[1]),)
    return 9, f"{parsed.netloc} source", ()


def source_label(url: str) -> str:
    return source_info(url)[1]


def source_sort_key(url: str) -> tuple[int, tuple[int, ...], str]:
    category, _label, version = source_info(url)
    padded = version + (0,) * max(0, 3 - len(version))
    return category, tuple(-component for component in padded), url


def urls(data: dict[str, Any], *, observed: bool = False) -> set[str]:
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourceError("metadata must have a non-empty sources list")
    values = []
    for source in sources:
        if observed:
            if not isinstance(source, dict):
                raise SourceError("observed sources must be objects")
            source = source.get("url")
        values.append(source_url(source))
    return set(values)


def latest_series(entries: Any) -> str:
    if not isinstance(entries, list) or not entries:
        raise SourceError("release configurations must be a non-empty list")
    series = []
    for entry in entries:
        value = entry.get("series") if isinstance(entry, dict) else None
        if not isinstance(value, str) or re.fullmatch(r"[0-9]+\.[0-9]+", value) is None:
            raise SourceError("release series must be a numeric major.minor")
        series.append(value)
    return max(series, key=lambda value: tuple(map(int, value.split("."))))


def render(drivers: dict[str, Any], releases: dict[str, Any], observed: dict[str, Any]) -> str:
    if observed.get("authority") != "observation_only":
        raise SourceError("upstream metadata must be observation_only")
    reviewed_urls = urls(drivers) | urls(releases)
    observed_urls = urls(observed, observed=True)
    reviewed_series = latest_series(releases.get("releases"))
    observed_series = latest_series(observed.get("release_compatibility"))
    lines = [
        BEGIN,
        "Primary sources (generated from the metadata files; do not edit this block manually):",
        "",
        f"Reviewed compatibility sources — latest registered PyTorch series: {reviewed_series}. A registered configuration is not proof of wheel availability or GPU architecture coverage; those are checked separately.",
        "",
        *[f"- [{source_label(url)}]({url})" for url in sorted(reviewed_urls, key=source_sort_key)],
        "",
        f"Observed upstream sources — latest observed PyTorch series: {observed_series}. Observations are not automatically approved compatibility rules, including when a document also appears in the reviewed list.",
        "",
        *[f"- [{source_label(url)}]({url})" for url in sorted(observed_urls, key=source_sort_key)],
        "",
        "Wheel availability is queried separately from the [PyTorch official wheel index](https://download.pytorch.org/whl/).",
        END,
    ]
    return "\n".join(lines)


def replace_block(document: str, block: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise SourceError("README must contain exactly one generated-source marker pair")
    start = document.index(BEGIN)
    end = document.index(END)
    if end < start:
        raise SourceError("generated-source markers must be ordered")
    for index, marker in ((start, BEGIN), (end, END)):
        after = index + len(marker)
        if (index > 0 and document[index - 1] != "\n") or (
            after < len(document) and document[after] != "\n"
        ):
            raise SourceError("generated-source markers must occupy their own lines")
    end += len(END)
    return document[:start] + block + document[end:]


def write_atomic(path: Path, document: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--drivers", type=Path, default=ROOT / "data/cuda-driver-rules.json")
    parser.add_argument("--releases", type=Path, default=ROOT / "data/pytorch-release-rules.json")
    parser.add_argument("--observed", type=Path, default=ROOT / "data/upstream-observed.json")
    parser.add_argument("--check", action="store_true", help="fail on drift without writing")
    args = parser.parse_args(argv)
    try:
        block = render(load(args.drivers), load(args.releases), load(args.observed))
        original = args.readme.read_text(encoding="utf-8")
        updated = replace_block(original, block)
        if updated == original:
            return 0
        if args.check:
            print(
                "README source catalog is out of sync; run: python3 scripts/sync_readme_sources.py",
                file=sys.stderr,
            )
            return 1
        write_atomic(args.readme, updated)
    except (OSError, ValueError, SourceError) as error:
        print(f"README source synchronization failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
