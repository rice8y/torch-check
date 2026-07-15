#!/usr/bin/env python3
"""Record a deterministic, review-only compatibility metadata snapshot.

The script writes only its requested observation output. It never edits the
reviewed rule files. The snapshot contains selected fields from official PyTorch
and NVIDIA documents, suitable for a pull-request diff that a maintainer can
compare with those rules.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence


RAW_SOURCE_URL = "https://raw.githubusercontent.com/pytorch/pytorch/main/RELEASE.md"
HUMAN_SOURCE_URL = "https://github.com/pytorch/pytorch/blob/main/RELEASE.md"
NVIDIA_MINOR_SOURCE_URL = (
    "https://docs.nvidia.com/deploy/cuda-compatibility/"
    "minor-version-compatibility.html"
)
NVIDIA_RELEASE_SOURCE_URL = (
    "https://docs.nvidia.com/cuda/archive/13.2.0/"
    "cuda-toolkit-release-notes/index.html"
)
PYTORCH_SOURCE_HOST = "raw.githubusercontent.com"
NVIDIA_SOURCE_HOST = "docs.nvidia.com"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_HTML_TABLES = 256
MAX_HTML_ROWS = 8192
MAX_HTML_CELLS = 32
FETCH_TIMEOUT_SECONDS = 30
USER_AGENT = "torch-check-upstream-observer/1"

PYTORCH_CUDA_BUILD_TAGS = (
    ("2.6", "v2.6.0"),
    ("2.7", "v2.7.0"),
    ("2.7", "v2.7.1"),
    ("2.8", "v2.8.0"),
    ("2.9", "v2.9.0"),
    ("2.9", "v2.9.1"),
    ("2.10", "v2.10.0"),
    ("2.11", "v2.11.0"),
)
PYTORCH_CUDA_BUILD_PATH = ".ci/manywheel/build_cuda.sh"

_ARCHITECTURE = r"[0-9]+\.[0-9]+(?:\+PTX)?"
_BASE_ARCH_ASSIGNMENT = re.compile(
    r'^\s*TORCH_CUDA_ARCH_LIST="(?P<expression>.*)"\s*(?:#.*)?$'
)
_BRANCH_ARCH_ASSIGNMENT = re.compile(
    r'^\s*TORCH_CUDA_ARCH_LIST="(?P<expression>.*)"\s*(?:;;)?\s*(?:#.*)?$'
)
_CUDA_CASE = re.compile(r"^\s*case\s+\$\{CUDA_VERSION\}\s+in\s*$")
_CUDA_BRANCH = re.compile(
    r"^\s*(?P<versions>[0-9]+\.[0-9]+(?:\|[0-9]+\.[0-9]+)*)\)"
    r"\s*(?P<body>.*)$"
)
_ARCH_LIST_REPLACEMENT = re.compile(
    r"\$\{TORCH_CUDA_ARCH_LIST//(?P<old>[0-9]+\.[0-9]+;?)/\}"
)
_X86_64_EMPTY_ARCH_EXPRESSION = (
    '$([[ "$ARCH" == "aarch64" ]] && echo "11.0;" || echo "")'
)


class ObservationError(RuntimeError):
    """Raised when the official document cannot be fetched or parsed safely."""


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _read_limited(stream: Any) -> bytes:
    payload = stream.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise ObservationError(
            f"upstream document exceeds {MAX_SOURCE_BYTES} bytes"
        )
    return payload


def read_source(
    source: str, allowed_host: str = PYTORCH_SOURCE_HOST
) -> str:
    """Read an HTTPS official source or a bounded local fixture."""

    parsed = urllib.parse.urlparse(source)
    if parsed.scheme:
        if parsed.scheme != "https" or parsed.hostname != allowed_host:
            raise ObservationError(
                f"remote source must use HTTPS on {allowed_host}"
            )
        request = urllib.request.Request(
            source,
            headers={"Accept": "text/plain", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - exact host is validated above
                request, timeout=FETCH_TIMEOUT_SECONDS
            ) as response:
                final = urllib.parse.urlparse(response.geturl())
                if final.scheme != "https" or final.hostname != allowed_host:
                    raise ObservationError("upstream redirected outside the host allowlist")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > MAX_SOURCE_BYTES:
                    raise ObservationError(
                        f"upstream document exceeds {MAX_SOURCE_BYTES} bytes"
                    )
                payload = _read_limited(response)
        except (OSError, ValueError) as error:
            raise ObservationError(f"failed to fetch upstream document: {error}") from error
    else:
        path = Path(source)
        try:
            with path.open("rb") as fixture:
                payload = _read_limited(fixture)
        except OSError as error:
            raise ObservationError(f"failed to read {path}: {error}") from error

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ObservationError("upstream document is not valid UTF-8") from error


def _split_markdown_row(line: str) -> list[str]:
    return [_normalize(cell) for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _find_table(
    document: str, header_matches: Callable[[Sequence[str]], bool]
) -> tuple[list[str], list[dict[str, str]]]:
    lines = document.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        headers = _split_markdown_row(line)
        if not header_matches(headers):
            continue
        if index + 1 >= len(lines) or not _is_separator(
            _split_markdown_row(lines[index + 1])
        ):
            raise ObservationError("matched table has no Markdown separator row")

        rows: list[dict[str, str]] = []
        for row_line in lines[index + 2 :]:
            if not row_line.lstrip().startswith("|"):
                break
            cells = _split_markdown_row(row_line)
            if len(cells) != len(headers):
                raise ObservationError("matched table contains a malformed row")
            rows.append(dict(zip(headers, cells, strict=True)))
        if not rows:
            raise ObservationError("matched table contains no data rows")
        return headers, rows
    raise ObservationError("required upstream Markdown table was not found")


def _cuda_versions(value: str) -> list[str]:
    return _unique(re.findall(r"\bCUDA\s+([0-9]+(?:\.[0-9]+){1,2})\b", value))


def _rocm_version(value: str) -> str | None:
    match = re.search(r"\bROCm\s+([0-9]+(?:\.[0-9]+){1,2})\b", value)
    return match.group(1) if match else None


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(component) for component in value.split("."))


def parse_release_compatibility(document: str) -> list[dict[str, Any]]:
    """Extract the compatibility matrix fields relevant to rule review."""

    expected = [
        "PyTorch version",
        "Python",
        "C++",
        "Stable CUDA",
        "Experimental CUDA",
        "Stable ROCm",
    ]
    _, rows = _find_table(document, lambda headers: list(headers) == expected)
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        series = row["PyTorch version"]
        if not re.fullmatch(r"[0-9]+\.[0-9]+", series):
            raise ObservationError(f"invalid PyTorch series in release matrix: {series!r}")
        if series in seen:
            raise ObservationError(f"duplicate PyTorch series in release matrix: {series}")
        seen.add(series)
        observed.append(
            {
                "series": series,
                "python": row["Python"],
                "cxx_standard": row["C++"],
                "stable_cuda": _cuda_versions(row["Stable CUDA"]),
                "experimental_cuda": _cuda_versions(row["Experimental CUDA"]),
                "stable_rocm": _rocm_version(row["Stable ROCm"]),
            }
        )
    if len(observed) < 2:
        raise ObservationError("release matrix unexpectedly contains fewer than two series")
    return sorted(observed, key=lambda row: _version_key(row["series"]), reverse=True)


def parse_cuda_architectures(document: str) -> list[dict[str, Any]]:
    """Extract the Linux x86/Windows CUDA architecture observation table."""

    headers, rows = _find_table(
        document,
        lambda values: len(values) == 3
        and values[0] == "CUDA"
        and "architectures supported for Linux x86 and Windows builds" in values[1]
        and values[2] == "notes",
    )
    architecture_header = headers[1]
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cuda = row["CUDA"]
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", cuda):
            raise ObservationError(f"invalid CUDA version in architecture table: {cuda!r}")
        if cuda in seen:
            raise ObservationError(f"duplicate CUDA version in architecture table: {cuda}")
        seen.add(cuda)
        architecture_text = row[architecture_header]
        capabilities = _unique(
            re.findall(r"(?<![0-9.])([0-9]+\.[0-9]+)(?:\+PTX)?", architecture_text)
        )
        ptx_capabilities = _unique(
            re.findall(r"(?<![0-9.])([0-9]+\.[0-9]+)\+PTX", architecture_text)
        )
        if not capabilities:
            raise ObservationError(f"no compute capabilities found for CUDA {cuda}")
        observed.append(
            {
                "cuda": cuda,
                "compute_capabilities": capabilities,
                "ptx_capabilities": ptx_capabilities,
                "notes": row["notes"],
            }
        )
    return sorted(observed, key=lambda row: _version_key(row["cuda"]))


def _cuda_build_raw_url(tag: str) -> str:
    return (
        f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/"
        f"{PYTORCH_CUDA_BUILD_PATH}"
    )


def _cuda_build_human_url(tag: str) -> str:
    return (
        f"https://github.com/pytorch/pytorch/blob/{tag}/"
        f"{PYTORCH_CUDA_BUILD_PATH}"
    )


def _cuda_variant(cuda: str) -> str:
    major, minor = cuda.split(".", maxsplit=1)
    if len(minor) != 1 or int(major) < 10:
        raise ObservationError(f"unsupported CUDA branch version: {cuda}")
    return f"cu{int(major)}{int(minor)}"


def _architecture_sequence(value: str) -> tuple[list[str], list[str]]:
    parts = value.split(";")
    if parts and parts[-1] == "":
        parts.pop()
    if not parts or any(not part for part in parts):
        raise ObservationError("CUDA architecture list is empty or malformed")
    if any(re.fullmatch(_ARCHITECTURE, part) is None for part in parts):
        raise ObservationError("CUDA architecture list contains unsupported syntax")

    capabilities = [part.removesuffix("+PTX") for part in parts]
    if len(set(capabilities)) != len(capabilities):
        raise ObservationError("CUDA architecture list contains duplicate capabilities")
    ptx_capabilities = [
        part.removesuffix("+PTX") for part in parts if part.endswith("+PTX")
    ]
    return capabilities, ptx_capabilities


def _evaluate_arch_expression(expression: str, base: str) -> str:
    """Evaluate the tiny, allowlisted assignment grammar used by PyTorch.

    This is deliberately not a shell interpreter. Unknown substitutions,
    command execution, quoting, and control operators fail closed.
    """

    if not base and (
        "${TORCH_CUDA_ARCH_LIST}" in expression
        or _ARCH_LIST_REPLACEMENT.search(expression) is not None
    ):
        raise ObservationError(
            "TORCH_CUDA_ARCH_LIST references a missing base assignment"
        )
    value = expression.replace(_X86_64_EMPTY_ARCH_EXPRESSION, "")
    value = _ARCH_LIST_REPLACEMENT.sub(
        lambda match: base.replace(match.group("old"), ""), value
    )
    value = value.replace("${TORCH_CUDA_ARCH_LIST}", base)
    if any(character in value for character in ("$", "`", "\\", '"', "'")):
        raise ObservationError(
            "TORCH_CUDA_ARCH_LIST uses an unsupported shell expression"
        )
    _architecture_sequence(value)
    return value


def _branch_arch_expression(lines: Sequence[str], versions: Sequence[str]) -> str:
    top_level: list[str] = []
    condition: str | None = None
    in_selected_else = False
    terminated = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'if [[ "$PACKAGE_TYPE" == "libtorch" ]]; then':
            if condition is not None:
                raise ObservationError("nested condition in CUDA case branch")
            condition = "libtorch"
            continue
        if stripped == 'if [[ "$GPU_ARCH_TYPE" = "cuda-aarch64" ]]; then':
            if condition is not None:
                raise ObservationError("nested condition in CUDA case branch")
            condition = "cuda_aarch64"
            in_selected_else = False
            continue
        if stripped == "else":
            if condition != "cuda_aarch64" or in_selected_else:
                raise ObservationError("unmatched else in CUDA case branch")
            in_selected_else = True
            continue
        if stripped == "fi":
            if condition is None:
                raise ObservationError("unmatched fi in CUDA case branch")
            condition = None
            in_selected_else = False
            continue
        if stripped.startswith("if "):
            raise ObservationError("unsupported condition in CUDA case branch")
        code_without_comment = line.split("#", maxsplit=1)[0].rstrip()
        if re.search(r";;\s*$", code_without_comment):
            terminated = True

        assignment = _BRANCH_ARCH_ASSIGNMENT.fullmatch(line)
        if assignment is not None:
            if condition is None or (
                condition == "cuda_aarch64" and in_selected_else
            ):
                top_level.append(assignment.group("expression"))
        elif "TORCH_CUDA_ARCH_LIST" in line:
            raise ObservationError("unsupported TORCH_CUDA_ARCH_LIST assignment")
        elif (
            stripped
            and not stripped.startswith("#")
            and stripped != ";;"
            and stripped != 'EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")'
            and re.fullmatch(
                r'export (?:TORCH_NVCC_FLAGS="[A-Za-z0-9 =_-]+"|BUILD_BUNDLE_PTXAS=1)',
                stripped,
            )
            is None
        ):
            raise ObservationError("unsupported command in CUDA case branch")

    if condition is not None:
        raise ObservationError("unterminated condition in CUDA case branch")
    if not terminated:
        raise ObservationError(
            f"CUDA case branch {','.join(versions)} is not terminated"
        )
    if len(top_level) != 1:
        raise ObservationError(
            f"CUDA case branch {','.join(versions)} must assign "
            "TORCH_CUDA_ARCH_LIST exactly once for wheels"
        )
    return top_level[0]


def parse_cuda_build_architectures(document: str) -> list[dict[str, Any]]:
    """Statically extract Linux x86_64 wheel architecture lists from Bash."""

    if "\x00" in document:
        raise ObservationError("PyTorch CUDA build script contains a NUL byte")
    lines = document.splitlines()
    case_indexes = [index for index, line in enumerate(lines) if _CUDA_CASE.fullmatch(line)]
    if len(case_indexes) != 1:
        raise ObservationError("expected one CUDA_VERSION case statement")
    case_index = case_indexes[0]

    base_assignments = [
        match.group("expression")
        for line in lines[:case_index]
        if (match := _BASE_ARCH_ASSIGNMENT.fullmatch(line)) is not None
    ]
    if len(base_assignments) > 1:
        raise ObservationError(
            "expected at most one base TORCH_CUDA_ARCH_LIST assignment before CUDA case"
        )
    base = base_assignments[0] if base_assignments else ""
    if any(
        "TORCH_CUDA_ARCH_LIST" in line
        and _BASE_ARCH_ASSIGNMENT.fullmatch(line) is None
        for line in lines[:case_index]
    ):
        raise ObservationError("unsupported pre-case TORCH_CUDA_ARCH_LIST use")
    if base_assignments:
        base_index = next(
            index
            for index, line in enumerate(lines[:case_index])
            if _BASE_ARCH_ASSIGNMENT.fullmatch(line) is not None
        )
        if any(line.strip() for line in lines[base_index + 1 : case_index]):
            raise ObservationError(
                "base TORCH_CUDA_ARCH_LIST must immediately precede CUDA case"
            )
        if any(character in base for character in ("$", "`", "\\", '"', "'")):
            raise ObservationError("base TORCH_CUDA_ARCH_LIST must be literal")
        _architecture_sequence(base)

    try:
        esac_index = next(
            index
            for index in range(case_index + 1, len(lines))
            if lines[index].strip() == "esac"
        )
    except StopIteration as error:
        raise ObservationError("CUDA_VERSION case statement is not terminated") from error

    blocks: list[tuple[list[str], list[str]]] = []
    current_versions: list[str] | None = None
    current_lines: list[str] = []
    for line in lines[case_index + 1 : esac_index]:
        branch = _CUDA_BRANCH.fullmatch(line)
        if branch is not None:
            if current_versions is not None:
                blocks.append((current_versions, current_lines))
            current_versions = branch.group("versions").split("|")
            current_lines = [branch.group("body")]
            continue
        if re.match(r"^\s*\*\)", line):
            if current_versions is not None:
                blocks.append((current_versions, current_lines))
            current_versions = None
            current_lines = []
            continue
        if current_versions is not None:
            current_lines.append(line)
    if current_versions is not None:
        blocks.append((current_versions, current_lines))
    if not blocks:
        raise ObservationError("CUDA_VERSION case statement has no numeric branches")

    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for versions, branch_lines in blocks:
        expression = _branch_arch_expression(branch_lines, versions)
        evaluated = _evaluate_arch_expression(expression, base)
        capabilities, ptx_capabilities = _architecture_sequence(evaluated)
        for cuda in versions:
            variant = _cuda_variant(cuda)
            if variant in seen:
                raise ObservationError(f"duplicate CUDA build branch for {variant}")
            seen.add(variant)
            observed.append(
                {
                    "cuda": cuda,
                    "variant": variant,
                    "compute_capabilities": capabilities,
                    "ptx_capabilities": ptx_capabilities,
                }
            )
    return sorted(observed, key=lambda row: _version_key(row["cuda"]))


def tagged_cuda_build_observation(
    series: str, tag: str, document: str
) -> dict[str, Any]:
    """Build a provenance-pinned architecture observation for one release tag."""

    if (
        (series, tag) not in PYTORCH_CUDA_BUILD_TAGS
        or re.fullmatch(r"[0-9]+\.[0-9]+", series) is None
    ):
        raise ObservationError(f"invalid PyTorch release tag mapping: {series} -> {tag}")
    return {
        "series": series,
        "tag": tag,
        "source_url": _cuda_build_human_url(tag),
        "source_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "platform": "linux_x86_64",
        "package_type": "wheel",
        "architectures": parse_cuda_build_architectures(document),
    }


class _BoundedTableParser(html.parser.HTMLParser):
    """Collect simple HTML table captions and rows with structural bounds."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._table_depth = 0
        self._table: dict[str, Any] | None = None
        self._in_caption = False
        self._caption_parts: list[str] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            if self._table_depth == 0:
                if len(self.tables) >= MAX_HTML_TABLES:
                    raise ObservationError("NVIDIA document contains too many tables")
                self._table = {"caption": "", "rows": []}
            self._table_depth += 1
            return
        if self._table_depth != 1 or self._table is None:
            return
        if tag == "caption":
            self._in_caption = True
            self._caption_parts = []
        elif tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            if len(self._row) >= MAX_HTML_CELLS:
                raise ObservationError("NVIDIA table row contains too many cells")
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._table_depth != 1:
            return
        if self._in_caption:
            self._caption_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._table_depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1 or self._table is None:
            return
        if tag == "caption" and self._in_caption:
            self._table["caption"] = _normalize("".join(self._caption_parts))
            self._in_caption = False
        elif tag in {"th", "td"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append(_normalize("".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            rows = self._table["rows"]
            if len(rows) >= MAX_HTML_ROWS:
                raise ObservationError("NVIDIA table contains too many rows")
            if self._row:
                rows.append(self._row)
            self._row = None
            self._cell_parts = None


def _html_tables(document: str) -> list[dict[str, Any]]:
    parser = _BoundedTableParser()
    try:
        parser.feed(document)
        parser.close()
    except (AssertionError, RecursionError) as error:
        raise ObservationError(f"failed to parse NVIDIA HTML: {error}") from error
    return parser.tables


def _table_with_caption(document: str, caption_fragment: str) -> list[list[str]]:
    matches = [
        table["rows"]
        for table in _html_tables(document)
        if caption_fragment in table["caption"]
    ]
    if len(matches) != 1:
        raise ObservationError(
            f"expected one NVIDIA table caption containing {caption_fragment!r}"
        )
    return matches[0]


def parse_nvidia_minor_compatibility(document: str) -> dict[str, Any]:
    """Extract NVIDIA's driver-branch ranges and the CUDA 11 exception."""

    rows = _table_with_caption(document, "Driver Version Ranges")
    families: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 3:
            continue
        toolkit = re.fullmatch(r"CUDA ([0-9]+)\.x", row[0])
        minimum = re.fullmatch(r">=\s*([0-9]+)", row[1])
        if toolkit is None or minimum is None:
            continue
        upper = re.search(r"<\s*([0-9]+)", row[2])
        families.append(
            {
                "cuda_major": int(toolkit.group(1)),
                "minimum_driver_branch": minimum.group(1),
                "upper_exclusive": upper.group(1) if upper else None,
            }
        )
    if {family["cuda_major"] for family in families} != {11, 12, 13}:
        raise ObservationError("NVIDIA compatibility table did not contain CUDA 11-13")
    exception = re.search(
        r"450\.80\.02\s*\(Linux\)\s*/\s*452\.39\s*\(Windows\)",
        document,
    )
    if exception is None:
        raise ObservationError("CUDA 11 family-driver exception was not found")
    return {
        "families": sorted(
            families, key=lambda family: family["cuda_major"], reverse=True
        ),
        "cuda_11_family_exception": {
            "linux_min_driver": "450.80.02",
            "windows_min_driver": "452.39",
        },
    }


def _minimum_driver(value: str) -> str | None:
    if value == "N/A":
        return None
    match = re.fullmatch(r">=\s*([0-9]+(?:\.[0-9]+)+)", value)
    if match is None:
        raise ObservationError(f"invalid NVIDIA minimum driver value: {value!r}")
    return match.group(1)


def parse_nvidia_toolkit_drivers(document: str) -> list[dict[str, Any]]:
    """Extract NVIDIA's exact Linux/Windows toolkit driver table."""

    rows = _table_with_caption(document, "CUDA Toolkit and Corresponding Driver Versions")
    releases: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 3 or not row[0].startswith("CUDA "):
            continue
        match = re.match(r"CUDA\s+([0-9]+(?:\.[0-9]+){1,2})", row[0])
        if match is None or int(match.group(1).split(".", maxsplit=1)[0]) < 11:
            continue
        releases.append(
            {
                "toolkit_release": row[0],
                "toolkit_version": match.group(1),
                "linux_min_driver": _minimum_driver(row[1]),
                "windows_min_driver": _minimum_driver(row[2]),
            }
        )
    if len(releases) < 20:
        raise ObservationError("NVIDIA toolkit driver table was unexpectedly short")
    if len({release["toolkit_release"] for release in releases}) != len(releases):
        raise ObservationError("NVIDIA toolkit driver table contains duplicate releases")
    return releases


def build_snapshot(
    document: str,
    nvidia_minor_document: str,
    nvidia_release_document: str,
    tagged_cuda_build_documents: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Build the deterministic JSON object committed for human review."""

    expected_tags = set(PYTORCH_CUDA_BUILD_TAGS)
    actual_tags = {
        (series, tag) for series, tag, _document in tagged_cuda_build_documents
    }
    if actual_tags != expected_tags or len(actual_tags) != len(
        tagged_cuda_build_documents
    ):
        raise ObservationError(
            "tagged CUDA build documents do not match the required release tags"
        )
    tagged_builds = [
        tagged_cuda_build_observation(series, tag, build_document)
        for series, tag, build_document in tagged_cuda_build_documents
    ]
    tagged_builds.sort(key=lambda build: _version_key(build["series"]), reverse=True)

    return {
        "schema_version": 2,
        "authority": "observation_only",
        "sources": [
            {
                "kind": "release_compatibility_matrix",
                "url": f"{HUMAN_SOURCE_URL}#release-compatibility-matrix",
            },
            {
                "kind": "cuda_architecture_matrix",
                "url": f"{HUMAN_SOURCE_URL}#pytorch-cuda-support-matrix",
            },
            *[
                {
                    "kind": "linux_x86_64_wheel_cuda_architectures",
                    "series": series,
                    "tag": tag,
                    "url": _cuda_build_human_url(tag),
                }
                for series, tag in PYTORCH_CUDA_BUILD_TAGS
            ],
            {
                "kind": "nvidia_minor_version_compatibility",
                "url": NVIDIA_MINOR_SOURCE_URL,
            },
            {
                "kind": "nvidia_toolkit_driver_versions",
                "url": f"{NVIDIA_RELEASE_SOURCE_URL}#cuda-driver",
            },
        ],
        "release_compatibility": parse_release_compatibility(document),
        "linux_x86_64_and_windows_cuda_architectures": parse_cuda_architectures(
            document
        ),
        "pytorch_tagged_cuda_builds": tagged_builds,
        "nvidia_minor_version_compatibility": parse_nvidia_minor_compatibility(
            nvidia_minor_document
        ),
        "nvidia_toolkit_driver_versions": parse_nvidia_toolkit_drivers(
            nvidia_release_document
        ),
    }


def serialized_snapshot(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"


def write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=RAW_SOURCE_URL,
        help="official HTTPS source, or a local Markdown fixture for testing",
    )
    parser.add_argument(
        "--nvidia-minor-source",
        default=NVIDIA_MINOR_SOURCE_URL,
        help="official NVIDIA minor-compatibility HTML, or a local test fixture",
    )
    parser.add_argument(
        "--nvidia-release-source",
        default=NVIDIA_RELEASE_SOURCE_URL,
        help="official NVIDIA release-note HTML, or a local test fixture",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        snapshot = build_snapshot(
            read_source(args.source),
            read_source(args.nvidia_minor_source, NVIDIA_SOURCE_HOST),
            read_source(args.nvidia_release_source, NVIDIA_SOURCE_HOST),
            [
                (series, tag, read_source(_cuda_build_raw_url(tag)))
                for series, tag in PYTORCH_CUDA_BUILD_TAGS
            ],
        )
        write_atomic(args.output, serialized_snapshot(snapshot))
    except ObservationError as error:
        print(f"upstream metadata observation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
