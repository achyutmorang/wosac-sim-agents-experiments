#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/_external_refs"
mkdir -p "$TARGET_DIR"

clone_or_pull() {
  local url="$1"
  local name="$2"
  if [[ -d "$TARGET_DIR/$name/.git" ]]; then
    echo "[update] $name"
    git -C "$TARGET_DIR/$name" pull --ff-only
  else
    echo "[clone]  $name"
    git clone "$url" "$TARGET_DIR/$name"
  fi
}

# Public implementations useful for WOSAC-oriented study.
clone_or_pull "https://github.com/waymo-research/waymo-open-dataset.git" "waymo-open-dataset"
clone_or_pull "https://github.com/rainmaker22/SMART.git" "SMART"
clone_or_pull "https://github.com/zhejz/TrafficBotsV1.5.git" "TrafficBotsV1.5"
clone_or_pull "https://github.com/Longzhong-Lin/UniMM.git" "UniMM"

echo "Done. Checked public repos under: $TARGET_DIR"
