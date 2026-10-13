#!/usr/bin/env bash
# Step 0 gate: expand config → 6 (cam_id, url) rows.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PP="${ROOT}/deepstream/plate_pipeline"
python3 "${PP}/print_streams.py" \
  --config "${PP}/config.yaml" \
  --write-labels "${PP}/cam_labels.txt" \
  --write-urls "${PP}/rtsp_urls_expanded.txt"
