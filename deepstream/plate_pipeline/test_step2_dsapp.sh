#!/usr/bin/env bash
# Step 2 alternate gate using deepstream-app (no pyds) — plate PGIE + tiler + file preview.
set -euo pipefail
set +H
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_RUN="${RUN_SECONDS:-40}"

# Ensure engine
FORCE="${FORCE:-0}" SUDO_PASSWORD="${SUDO_PASSWORD}" \
  bash "${ROOT}/scripts/export_yolo12_plate_deepstream.sh"

# Expand URLs + labels
bash "${PP}/test_step0.sh"
cp -f "${PP}/rtsp_urls_expanded.txt" "${ROOT}/deepstream/rtsp_urls.txt"
cp -f "${PP}/cam_labels.txt" "${ROOT}/deepstream/cam_labels.txt"

# Point primary YOLO config at plate engine for this run
PLATE_INFER="${PP}/config_infer_primary_plate.txt"
YOLO_INFER="${ROOT}/deepstream/config_infer_primary_yolo.txt"
BACKUP="${ROOT}/deepstream/config_infer_primary_yolo.txt.bak_plate"
if [[ ! -f "${BACKUP}" ]]; then
  cp -f "${YOLO_INFER}" "${BACKUP}"
fi
cp -f "${PLATE_INFER}" "${YOLO_INFER}"

cleanup() {
  if [[ -f "${BACKUP}" ]]; then
    mv -f "${BACKUP}" "${YOLO_INFER}"
  fi
}
trap cleanup EXIT

# Ensure YOLO parser .so
SO="${ROOT}/deepstream/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
if [[ ! -f "${SO}" ]]; then
  ds_run '
    set -euo pipefail
    export PATH="/usr/local/cuda-12.1/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
    if [[ ! -d /workspace/deepstream/DeepStream-Yolo/.git ]]; then
      git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git /workspace/deepstream/DeepStream-Yolo
    fi
    cd /workspace/deepstream/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
    CUDA_VER=12.1 make -j"$(nproc)"
  '
fi

# Ensure camlabel plugin
if [[ ! -f "${ROOT}/deepstream/gst-camlabel/libnvdsgst_dsexample.so" ]]; then
  ds_run '
    set -euo pipefail
    mkdir -p /etc/apt/sources.list.d/disabled
    for f in /etc/apt/sources.list.d/*cuda* /etc/apt/sources.list.d/*nvidia*; do
      [ -e "$f" ] || continue
      mv "$f" /etc/apt/sources.list.d/disabled/ 2>/dev/null || true
    done
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      g++ pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev >/dev/null
    make -C /workspace/deepstream/gst-camlabel CUDA_VER=12.1 clean all
  '
fi

echo "=== Step 2 (deepstream-app): ${SECONDS_RUN}s preview ==="
export PREVIEW=1 NO_DISPLAY=1 RUN_SECONDS="${SECONDS_RUN}" NUM_STREAMS=6
export RTSP_URLS_FILE=/workspace/deepstream/rtsp_urls.txt
export OUTPUT_FILE=/workspace/logs/plate_step2_preview.mkv
export CAM_LABELS_FILE=/workspace/deepstream/cam_labels.txt

# Use existing runner
SUDO_PASSWORD="${SUDO_PASSWORD}" PREVIEW=1 NO_DISPLAY=1 RUN_SECONDS="${SECONDS_RUN}" \
  RTSP_URLS="$(paste -sd' ' "${PP}/rtsp_urls_expanded.txt")" \
  OUTPUT_FILE=/workspace/logs/plate_step2_preview.mkv \
  bash "${ROOT}/scripts/06_run_deepstream.sh"

# Gate: preview artifact exists and is non-trivial size
HOST_MKV="${ROOT}/logs/plate_step2_preview.mkv"
HOST_JPG="${ROOT}/logs/deepstream_preview_mosaic.jpg"
if [[ -f "${HOST_MKV}" ]] && [[ $(stat -c%s "${HOST_MKV}") -gt 10000 ]]; then
  echo "PASS: Step 2 — deepstream-app plate PGIE produced preview (${HOST_MKV})"
  ls -lh "${HOST_MKV}" "${HOST_JPG}" 2>/dev/null || true
  exit 0
fi
if [[ -f "${HOST_JPG}" ]] && [[ $(stat -c%s "${HOST_JPG}") -gt 5000 ]]; then
  echo "PASS: Step 2 — mosaic jpeg present (${HOST_JPG})"
  exit 0
fi
echo "FAIL: Step 2 — no preview artifact" >&2
exit 1
