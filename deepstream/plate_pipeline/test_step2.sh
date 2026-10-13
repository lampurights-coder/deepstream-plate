#!/usr/bin/env bash
# Step 2 gate: export DS plate engine (if needed) + detect-only run.
set -euo pipefail
set +H
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_RUN="${RUN_SECONDS:-40}"

echo "=== Export plate DeepStream engine (if missing) ==="
FORCE="${FORCE:-0}" SUDO_PASSWORD="${SUDO_PASSWORD}" \
  bash "${ROOT}/scripts/export_yolo12_plate_deepstream.sh"

echo "=== Ensure DeepStream-Yolo parser .so ==="
SO="${ROOT}/deepstream/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
if [[ ! -f "${SO}" ]]; then
  ds_run '
    set -euo pipefail
    export PATH="/usr/local/cuda-12.1/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
    LIB=/workspace/deepstream/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
    if [[ ! -f "$LIB" ]]; then
      if [[ ! -d /workspace/deepstream/DeepStream-Yolo/.git ]]; then
        git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git /workspace/deepstream/DeepStream-Yolo
      fi
      cd /workspace/deepstream/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
      CUDA_VER=12.1 make -j"$(nproc)"
    fi
    ls -la "$LIB"
  '
fi

# Ensure pyds 1.2.0 wheel is present
WHEEL="${PP}/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl"
if [[ ! -f "${WHEEL}" ]]; then
  SUDO_PASSWORD="${SUDO_PASSWORD}" bash "${PP}/setup_pyds.sh"
fi

echo "=== Step 2: detect-only (${SECONDS_RUN}s) ==="
ds_run '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
  apt-get update -qq >/dev/null
  apt-get install -y -qq gir1.2-gstreamer-1.0 libpython3.10 python3-gi python3-gst-1.0 >/dev/null
  pip3 install -q --force-reinstall /workspace/deepstream/plate_pipeline/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl >/dev/null
  pip3 install -q "numpy<2" pyyaml >/dev/null
  export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream:/workspace/deepstream/gst-camlabel
  export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
  # Do not put license-plate on PYTHONPATH for detect-only (avoids import side effects)
  export PYTHONPATH=/workspace/deepstream/plate_pipeline
  export ENABLE_CAMLABEL=0
  python3 -u /workspace/deepstream/plate_pipeline/run_plate_ocr.py \
    --config /workspace/deepstream/plate_pipeline/config.yaml \
    --mode detect \
    --seconds '"${SECONDS_RUN}"'
'
