#!/usr/bin/env bash
# Run gpu-decoding:ds-lpr-service against host RTSP emu + MinIO (host network).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
IMAGE="${IMAGE_NAME:-gpu-decoding:ds-lpr-service}"
NAME="${CONTAINER_NAME:-ds-lpr-service}"
CONFIG_HOST="${CONFIG_HOST:-${DIR}/cameras.yaml}"
export SUDO_PASSWORD="${SUDO_PASSWORD:-1}"

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    echo "${SUDO_PASSWORD}" | sudo -S docker "$@"
  fi
}

if ! docker_cmd image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Missing image ${IMAGE}. Run: ${DIR}/build.sh" >&2
  exit 1
fi

docker_cmd rm -f "${NAME}" >/dev/null 2>&1 || true

mkdir -p "${ROOT}/output"

# Mount the whole package dir so multiprocessing spawn can re-import ocr_proc.
# /workspace/output holds preview TS (boxes drawn by nvdsosd) when preview.sink=file.
docker_cmd run -d --name "${NAME}" --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e PYTHONPATH=/workspace:/workspace/license-plate \
  -e DS_LPR_CONFIG=/workspace/ds_lpr_service/cameras.yaml \
  -e LICENSE_PLATE_ROOT=/workspace/license-plate \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -e DISPLAY="${DISPLAY:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${ROOT}/deepstream:/workspace/deepstream" \
  -v "${ROOT}/license-plate:/workspace/license-plate" \
  -v "${ROOT}/models:/workspace/models" \
  -v "${ROOT}/output:/workspace/output" \
  -v "${DIR}:/workspace/ds_lpr_service" \
  -v "${CONFIG_HOST}:/workspace/ds_lpr_service/cameras.yaml" \
  -v "${DIR}/entrypoint.sh:/usr/local/bin/ds-lpr-entrypoint.sh:ro" \
  -w /workspace \
  "${IMAGE}"

echo "Started ${NAME} → http://127.0.0.1:8003  (config: ${CONFIG_HOST})"
echo "Preview (if sink=file): ${ROOT}/output/ds_lpr_preview.ts"
docker_cmd ps --filter "name=${NAME}" --format 'table {{.Names}}\t{{.Status}}'
