#!/usr/bin/env bash
# DeepStream 7.1 — 6 RTSP (emu) → nvstreammux → plate nvinfer → 2x3 tiler → OSD → EGL
# This is the real DeepStream path (NOT OpenCV run_2cam_lpr.py).
set -euo pipefail
set +H

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_RUN="${RUN_SECONDS:-600}"
MODE="${MODE:-detect}"   # detect | ocr
CONFIG_HOST="${CONFIG_HOST:-${ROOT}/deepstream/plate_pipeline/config_6cam_emu.yaml}"
CONFIG="${CONFIG:-/workspace/deepstream/plate_pipeline/config_6cam_emu.yaml}"
USE_EMU="${USE_EMU:-1}"
SINK="${SINK:-file}"   # file | egl | fakesink
PREVIEW_HOST="${ROOT}/logs/plate_6cam_ds_preview.ts"

if [[ "${USE_EMU}" == "1" ]]; then
  NUM_CAMS=6 bash "${ROOT}/scripts/rtsp_emulator/start_rtsp_emu.sh"
fi

# Patch sink mode without rewriting the whole yaml by hand
python3 - <<PY
import yaml
from pathlib import Path
p = Path("${CONFIG_HOST}")
cfg = yaml.safe_load(p.read_text()) or {}
cfg.setdefault("preview", {})["sink"] = "${SINK}"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"preview.sink=${SINK}")
PY

# Allow Docker containers to open the EGL window on the host display
DISPLAY="${DISPLAY:-:1}"
if [[ "${SINK}" == "egl" ]] && command -v xhost >/dev/null 2>&1; then
  xhost +local: >/dev/null 2>&1 || true
  xhost +local:root >/dev/null 2>&1 || true
fi
rm -f "${PREVIEW_HOST}"

WHEEL="${PP}/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl"
if [[ ! -f "${WHEEL}" ]]; then
  SUDO_PASSWORD="${SUDO_PASSWORD}" bash "${PP}/setup_pyds.sh"
fi

mkdir -p "${ROOT}/logs" "${ROOT}/output"
# Write cam labels for optional overlay (source_id order)
python3 "${PP}/print_streams.py" \
  --config "${CONFIG_HOST}" \
  --write-labels "${PP}/cam_labels.txt" \
  --write-urls "${PP}/rtsp_urls_expanded.txt" || true

echo "=== DeepStream 6-cam (${MODE}) → 2x3 tiler sink=${SINK} for ${SECONDS_RUN}s ==="
echo "Image: ${IMAGE}"
echo "Config: ${CONFIG}"
echo "Preview: ${PREVIEW_HOST}"

ds_run '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
  apt-get update -qq >/dev/null
  apt-get install -y -qq gir1.2-gstreamer-1.0 libpython3.10 python3-gi python3-gst-1.0 \
    python3-dev build-essential >/dev/null
  pip3 install -q --force-reinstall /workspace/deepstream/plate_pipeline/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl >/dev/null
  pip3 install -q "numpy<2" pyyaml opencv-python-headless >/dev/null
  if [[ "'"${MODE}"'" == "ocr" ]]; then
    python3 -c "import tensorrt" 2>/dev/null || pip3 install -q "tensorrt==10.3.0" >/dev/null || true
    python3 -c "import cupy" 2>/dev/null || pip3 install -q cupy-cuda12x >/dev/null || true
    python3 -c "import pycuda.driver" 2>/dev/null || \
      C_INCLUDE_PATH=/usr/local/cuda/include CPLUS_INCLUDE_PATH=/usr/local/cuda/include \
      LIBRARY_PATH=/usr/local/cuda/lib64 CUDA_ROOT=/usr/local/cuda \
      pip3 install -q pycuda >/dev/null || true
  fi
  export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream:/workspace/deepstream/gst-camlabel
  export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/workspace/license-plate:/workspace/deepstream/plate_pipeline
  export ENABLE_CAMLABEL=0
  echo "DISPLAY=${DISPLAY:-unset}"
  python3 -u /workspace/deepstream/plate_pipeline/run_plate_ocr.py \
    --config "'"${CONFIG}"'" \
    --mode "'"${MODE}"'" \
    --seconds '"${SECONDS_RUN}"' \
    --preview /workspace/logs/plate_6cam_ds_preview.ts
'

echo "=== DeepStream 6-cam run finished ==="
if [[ -f "${PREVIEW_HOST}" ]]; then
  ls -lh "${PREVIEW_HOST}"
  echo "Play mosaic: ffplay -autoexit ${PREVIEW_HOST}"
fi
