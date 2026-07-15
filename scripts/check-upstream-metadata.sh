#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
binary="$root/target/release/torch-check"
cache_dir=$(mktemp -d)
trap 'rm -rf "$cache_dir"' EXIT

set +e
"$binary" --cache-dir "$cache_dir" --refresh candidates --format json > "$cache_dir/candidates.json"
status=$?
set -e
if [ "$status" -gt 1 ]; then
  exit "$status"
fi
jq -e '.schema_version == 1 and (.metadata.source == "https://download.pytorch.org/whl/")' "$cache_dir/candidates.json" > /dev/null
python3 "$root/scripts/check_reviewed_metadata.py" \
  --drivers "$root/data/cuda-driver-rules.json" \
  --releases "$root/data/pytorch-release-rules.json" \
  --observed "$root/data/upstream-observed.json"
