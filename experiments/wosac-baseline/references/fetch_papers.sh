#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_DIR="$ROOT_DIR/pdfs"
mkdir -p "$PDF_DIR"

fetch_required() {
  local url="$1"
  local out="$2"
  echo "[download] $out"
  curl -L --fail --retry 3 --retry-delay 2 -o "$PDF_DIR/$out" "$url"
}

fetch_optional() {
  local url="$1"
  local out="$2"
  echo "[optional] $out"
  if curl -L --fail --retry 2 --retry-delay 2 -o "$PDF_DIR/$out" "$url"; then
    echo "[ok] $out"
  else
    echo "[warn] unavailable: $url"
    rm -f "$PDF_DIR/$out"
  fi
}

# Benchmark and simulator foundations
fetch_required "https://papers.nips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf" "wosac_challenge_2023_neurips_db.pdf"
fetch_required "https://arxiv.org/pdf/2310.08710.pdf" "waymax_2023.pdf"

# Public method papers with open code relevance
fetch_required "https://arxiv.org/pdf/2405.15677.pdf" "smart_2024.pdf"
fetch_required "https://arxiv.org/pdf/2303.04116.pdf" "trafficbots_2023.pdf"
fetch_required "https://arxiv.org/pdf/2406.10898.pdf" "trafficbots_v1_5_2024.pdf"
fetch_required "https://arxiv.org/pdf/2501.17015.pdf" "unimm_2025_arxiv.pdf"
fetch_required "https://arxiv.org/pdf/2506.21618.pdf" "trajtok_2025_arxiv.pdf"

# Additional recent simulation papers useful for framing gap analysis
fetch_required "https://arxiv.org/pdf/2405.17372.pdf" "behaviorgpt_2024.pdf"
fetch_required "https://arxiv.org/pdf/2306.11868.pdf" "multiverse_transformer_2023.pdf"
fetch_required "https://arxiv.org/pdf/2505.03344.pdf" "rift_2025.pdf"
fetch_required "https://arxiv.org/pdf/2502.14706.pdf" "reliable_simulated_driving_agents_2025.pdf"
fetch_required "https://arxiv.org/pdf/2602.01916.pdf" "forsim_2026.pdf"

# WOSAC 2025 technical reports (direct links are partially public)
fetch_optional "https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Sim%20Agents%20Challenge%20-%20First%20Place%20-%20TrajTok.pdf" "wosac_2025_trajtok_technical_report.pdf"
fetch_optional "https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Sim%20Agents%20Challenge%20-%20Second%20Place%20-%20RLFTSim.pdf" "wosac_2025_rlftsim_technical_report.pdf"
fetch_optional "https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Sim%20Agents%20Challenge%20-%20Third%20Place%20-%20comBOT.pdf" "wosac_2025_combot_technical_report.pdf"
fetch_optional "https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Sim%20Agents%20Challenge%20-%20Honorable%20Mention%20-%20UniMM.pdf" "wosac_2025_unimm_technical_report.pdf"

echo "Done. PDFs in: $PDF_DIR"
