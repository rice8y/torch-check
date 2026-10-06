from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sync_readme_sources", ROOT / "scripts/sync_readme_sources.py"
)
assert SPEC is not None and SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def metadata() -> tuple[dict, dict, dict]:
    return (
        load("cuda-driver-rules.json"),
        load("pytorch-release-rules.json"),
        load("upstream-observed.json"),
    )


class ReadmeSourceTests(unittest.TestCase):
    def test_repository_catalog_is_synchronized(self) -> None:
        original = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(SYNC.replace_block(original, SYNC.render(*metadata())), original)

    def test_reviewed_release_build_sources_are_not_omitted(self) -> None:
        block = SYNC.render(*metadata())
        self.assertIn("PyTorch 2.14 release build matrix", block)
        self.assertIn("PyTorch 2.15 release build matrix", block)
        self.assertIn("TorchVision 0.29 version metadata", block)

    def test_observed_changes_do_not_approve_reviewed_rules(self) -> None:
        drivers, releases, observed = metadata()
        original = copy.deepcopy((drivers, releases, observed))
        observed["release_compatibility"].append({"series": "99.0"})
        url = "https://github.com/pytorch/pytorch/issues/999999"
        observed["sources"].append({"kind": "future_observation", "url": url})
        block = SYNC.render(drivers, releases, observed)
        reviewed, observation = block.split("Observed upstream sources", 1)
        self.assertNotIn("99.0", reviewed)
        self.assertNotIn(url, reviewed)
        self.assertIn("99.0", observation)
        self.assertIn(url, observation)
        self.assertEqual((drivers, releases), original[:2])
        self.assertIn("not automatically approved", observation)

    def test_reviewed_source_edits_are_reflected_without_touching_observations(self) -> None:
        drivers, releases, observed = metadata()
        releases["sources"].append(
            "https://github.com/pytorch/pytorch/blob/release/99.0/.github/scripts/generate_binary_build_matrix.py"
        )
        reviewed = SYNC.render(drivers, releases, observed).split(
            "Observed upstream sources", 1
        )[0]
        self.assertIn("PyTorch 99.0 release build matrix", reviewed)

    def test_generation_is_order_independent_and_deduplicated(self) -> None:
        drivers, releases, observed = metadata()
        expected = SYNC.render(drivers, releases, observed)
        for item in (drivers, releases, observed):
            item["sources"].reverse()
            item["sources"].append(copy.deepcopy(item["sources"][0]))
        self.assertEqual(SYNC.render(drivers, releases, observed), expected)

    def test_labels_do_not_require_a_manual_version_list(self) -> None:
        self.assertEqual(
            SYNC.source_label(
                "https://github.com/pytorch/pytorch/blob/v2.16.1/.ci/manywheel/build_cuda.sh"
            ),
            "PyTorch 2.16.1 Linux wheel build configuration",
        )
        self.assertEqual(
            SYNC.source_label(
                "https://docs.nvidia.com/cuda/archive/13.4.0/cuda-toolkit-release-notes/index.html#cuda-driver"
            ),
            "NVIDIA CUDA 13.4.0 release driver table",
        )

    def test_source_urls_reject_markdown_injection_and_untrusted_origins(self) -> None:
        for url in (
            "http://github.com/pytorch/pytorch",
            "https://example.org/source",
            "https://github.com.evil.example/source",
            "https://user@github.com/source",
            "https://github.com:443/source",
            "https://github.com/source\ntext",
            "https://github.com/source)[]",
            "https://github.com/source\x7f",
            "https://github.com/source\u202e",
            None,
        ):
            with self.subTest(url=url), self.assertRaises(SYNC.SourceError):
                SYNC.source_url(url)

    def test_malformed_metadata_is_rejected(self) -> None:
        drivers, releases, observed = metadata()
        for replacement in ([], [None], [{"series": "2.15\nnot reviewed"}]):
            releases["releases"] = replacement
            with self.assertRaises(SYNC.SourceError):
                SYNC.render(drivers, releases, observed)
        drivers, releases, observed = metadata()
        observed["authority"] = "reviewed"
        with self.assertRaises(SYNC.SourceError):
            SYNC.render(drivers, releases, observed)
        observed["authority"] = "observation_only"
        observed["sources"] = ["https://github.com/pytorch/pytorch"]
        with self.assertRaises(SYNC.SourceError):
            SYNC.render(drivers, releases, observed)

    def test_latest_series_is_numeric(self) -> None:
        self.assertEqual(
            SYNC.latest_series([{"series": "2.9"}, {"series": "2.15"}]), "2.15"
        )
        urls = [
            "https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/manywheel/build_cuda.sh",
            "https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh",
        ]
        self.assertEqual(sorted(urls, key=SYNC.source_sort_key), list(reversed(urls)))

    def test_sources_follow_document_roles_then_descending_numeric_versions(self) -> None:
        expected = [
            "https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html",
            "https://docs.nvidia.com/cuda/archive/13.4.0/cuda-toolkit-release-notes/index.html#cuda-driver",
            "https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html#cuda-driver",
            "https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix",
            "https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix",
            "https://pytorch.org/get-started/previous-versions/",
            "https://github.com/pytorch/pytorch/blob/release/2.15/.github/scripts/generate_binary_build_matrix.py",
            "https://github.com/pytorch/pytorch/blob/release/2.9/.github/scripts/generate_binary_build_matrix.py",
            "https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh",
            "https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh",
            "https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/manywheel/build_cuda.sh",
            "https://github.com/pytorch/vision/blob/release/0.30/version.txt",
            "https://github.com/pytorch/vision/blob/release/0.29/version.txt",
            "https://github.com/pytorch/pytorch/issues/190355",
            "https://github.com/pytorch/pytorch/issues/99999",
            "https://github.com/pytorch/pytorch/other-source",
        ]
        self.assertEqual(sorted(reversed(expected), key=SYNC.source_sort_key), expected)

    def test_reviewed_and_observed_lists_use_the_same_sort_policy(self) -> None:
        block = SYNC.render(*metadata())
        reviewed, observed = block.split("Observed upstream sources", 1)
        for section in (reviewed, observed):
            self.assertLess(
                section.index("NVIDIA CUDA minor-version compatibility"),
                section.index("NVIDIA CUDA 13.2.0 release driver table"),
            )
            self.assertLess(
                section.index("PyTorch release matrix"),
                section.index("PyTorch 2.11.0 Linux wheel build configuration"),
            )
            self.assertLess(
                section.index("PyTorch 2.11.0 Linux wheel build configuration"),
                section.index("PyTorch 2.6.0 Linux wheel build configuration"),
            )
        self.assertLess(
            reviewed.index("PyTorch 2.15 release build matrix"),
            reviewed.index("PyTorch 2.14 release build matrix"),
        )

    def test_generation_preserves_everything_outside_the_markers(self) -> None:
        original = f"# Title\n\n{SYNC.BEGIN}\nold\n{SYNC.END}\n\nTail\n"
        block = f"{SYNC.BEGIN}\nnew\n{SYNC.END}"
        expected = f"# Title\n\n{block}\n\nTail\n"
        self.assertEqual(SYNC.replace_block(original, block), expected)

    def test_missing_duplicated_reversed_or_inline_markers_fail_closed(self) -> None:
        for original in (
            "# No markers\n",
            f"{SYNC.BEGIN}\n{SYNC.BEGIN}\n{SYNC.END}\n",
            f"{SYNC.END}\n{SYNC.BEGIN}\n",
            f"prefix {SYNC.BEGIN}\n{SYNC.END}\n",
            f"{SYNC.BEGIN} suffix\n{SYNC.END}\n",
            f"{SYNC.BEGIN}\n{SYNC.END} suffix\n",
        ):
            with self.subTest(original=original), self.assertRaises(SYNC.SourceError):
                SYNC.replace_block(original, SYNC.render(*metadata()))

    def test_check_is_non_mutating_and_write_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            original = f"Header\n{SYNC.BEGIN}\nold\n{SYNC.END}\nTail\n"
            readme.write_text(original, encoding="utf-8")
            readme.chmod(0o644)
            args = ["--readme", str(readme)]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(SYNC.main([*args, "--check"]), 1)
            self.assertIn("python3 scripts/sync_readme_sources.py", stderr.getvalue())
            self.assertEqual(readme.read_text(encoding="utf-8"), original)
            self.assertEqual(SYNC.main(args), 0)
            if os.name != "nt":
                self.assertEqual(readme.stat().st_mode & 0o777, 0o644)
            modified = readme.stat().st_mtime_ns
            self.assertEqual(SYNC.main(args), 0)
            self.assertEqual(readme.stat().st_mtime_ns, modified)
            self.assertEqual(SYNC.main([*args, "--check"]), 0)

    def test_bot_and_ci_cover_the_generated_readme(self) -> None:
        workflow = (ROOT / ".github/workflows/metadata-check.yml").read_text()
        self.assertLess(
            workflow.index("python3 scripts/sync_readme_sources.py"),
            workflow.index("Detect observation drift"),
        )
        self.assertIn("git status --porcelain -- data/upstream-observed.json README.md", workflow)
        self.assertIn("sha256sum data/upstream-observed.json README.md", workflow)
        self.assertIn("git add data/upstream-observed.json README.md", workflow)
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("python3 scripts/sync_readme_sources.py --check", ci)
        release = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("python3 scripts/sync_readme_sources.py --check", release)

    def test_failed_atomic_write_preserves_readme_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            original = f"Header\n{SYNC.BEGIN}\nold\n{SYNC.END}\nTail\n"
            readme.write_text(original, encoding="utf-8")
            with patch.object(SYNC.os, "replace", side_effect=OSError("mock I/O failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(SYNC.main(["--readme", str(readme)]), 1)
            self.assertEqual(readme.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).iterdir()), [readme])


if __name__ == "__main__":
    unittest.main()
