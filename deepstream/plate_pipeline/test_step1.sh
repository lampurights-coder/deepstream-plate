#!/usr/bin/env bash
# Step 1 gate: 6-stream nvstreammux frame gather in DeepStream 7.1.
set -euo pipefail
set +H
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_RUN="${RUN_SECONDS:-20}"

echo "=== Step 0 (refresh labels) ==="
bash "${PP}/test_step0.sh"

echo "=== Step 1: mux-only (${SECONDS_RUN}s) ==="
ds_run '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
  apt-get update -qq >/dev/null
  apt-get install -y -qq gir1.2-gstreamer-1.0 >/dev/null
  pip3 install -q pyyaml 2>/dev/null || true
  export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}
  python3 /workspace/deepstream/plate_pipeline/run_mux_only.py \
    --config /workspace/deepstream/plate_pipeline/config.yaml \
    --seconds '"${SECONDS_RUN}"'
'
