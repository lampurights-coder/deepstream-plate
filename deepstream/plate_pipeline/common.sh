#!/usr/bin/env bash
# Shared helpers for plate_pipeline step tests (DeepStream 7.1 image).
# shellcheck disable=SC2034

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PP="${ROOT}/deepstream/plate_pipeline"
IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:7.1-samples-multiarch}"
TRT_IMAGE="${TRT_IMAGE:-gpu-decoding:nvdec-trt}"
export SUDO_PASSWORD="${SUDO_PASSWORD:-1}"

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    echo "${SUDO_PASSWORD}" | sudo -S -E docker "$@"
  fi
}

ds_run() {
  # Usage: ds_run <bash -lc body>
  # DISPLAY + X11 (+ /dev/dri when present) so nveglglessink can show the tiler mosaic.
  local display="${DISPLAY:-:1}"
  local dri_args=()
  if [[ -e /dev/dri ]]; then
    dri_args+=(--device /dev/dri)
  fi
  docker_cmd run --rm --runtime nvidia --network host \
    "${dri_args[@]}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display \
    -e DISPLAY="${display}" \
    -e QT_X11_NO_MITSHM=1 \
    -e GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream:/workspace/deepstream/gst-camlabel \
    -e GST_DEBUG=0 \
    -e CAM_LABELS_FILE=/workspace/deepstream/plate_pipeline/cam_labels.txt \
    -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib \
    -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${ROOT}/deepstream:/workspace/deepstream" \
    -v "${ROOT}/license-plate:/workspace/license-plate" \
    -v "${ROOT}/models:/workspace/models" \
    -v "${ROOT}/logs:/workspace/logs" \
    -v "${ROOT}/output:/workspace/output" \
    -v "${ROOT}/scripts:/workspace/scripts:ro" \
    -w /workspace \
    "${IMAGE}" \
    bash -lc "$*"
}

trt_run() {
  docker_cmd run --rm --runtime nvidia --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -e PYTHONPATH=/workspace/license-plate \
    -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -e MPLBACKEND=Agg \
    -e QT_QPA_PLATFORM=offscreen \
    -v "${ROOT}/license-plate:/workspace/license-plate" \
    -v "${ROOT}/deepstream:/workspace/deepstream" \
    -v "${ROOT}/models:/workspace/models" \
    -v "${ROOT}/logs:/workspace/logs" \
    -v "${ROOT}/scripts:/workspace/scripts:ro" \
    -w /workspace/license-plate \
    "${TRT_IMAGE}" \
    bash -lc "$*"
}
