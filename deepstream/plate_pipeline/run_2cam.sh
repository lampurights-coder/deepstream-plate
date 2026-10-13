#!/usr/bin/env bash
# 2 RTSP → plate detect + OCR → save frames to ./output/
set -euo pipefail
set +H
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_RUN="${RUN_SECONDS:-45}"
mkdir -p "${ROOT}/output"

echo "=== 2-cam detect+OCR → output/ (${SECONDS_RUN}s) ==="
ds_run '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
  apt-get update -qq >/dev/null
  apt-get install -y -qq gir1.2-gstreamer-1.0 libpython3.10 python3-gi python3-gst-1.0 \
    python3-dev build-essential >/dev/null
  pip3 install -q --force-reinstall /workspace/deepstream/plate_pipeline/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl >/dev/null
  pip3 install -q "numpy<2" pyyaml opencv-python-headless >/dev/null
  python3 -c "import tensorrt" 2>/dev/null || pip3 install -q "tensorrt==10.3.0" >/dev/null
  python3 -c "import cupy" 2>/dev/null || pip3 install -q cupy-cuda12x >/dev/null
  python3 -c "import pycuda.driver" 2>/dev/null || pip3 install -q pycuda >/dev/null || true
  export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream
  export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/workspace/license-plate:/workspace/deepstream/plate_pipeline
  export ENABLE_CAMLABEL=0
  python3 -u /workspace/deepstream/plate_pipeline/run_2cam_save.py \
    --config /workspace/deepstream/plate_pipeline/config_2cam.yaml \
    --seconds '"${SECONDS_RUN}"'
'
echo "=== Host output ==="
ls -lh "${ROOT}/output" | head -50
echo "Saved under: ${ROOT}/output"
