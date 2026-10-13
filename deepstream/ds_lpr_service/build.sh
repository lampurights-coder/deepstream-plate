#!/usr/bin/env bash
# Build gpu-decoding:ds-lpr-service from a *local* DeepStream 7.1 base.
# Never docker-pulls NGC. Aborts if base is missing. Prefers wheelhouse (no PyPI).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
IMAGE="${IMAGE_NAME:-gpu-decoding:ds-lpr-service}"
BASE_PRIMARY="${DS_BASE:-nvcr.io/nvidia/deepstream:7.1-samples-multiarch}"
BASE_ALT="deepstream:7.1-samples"
BASE_TAR="${DS_BASE_TAR:-/home/mohmd/Documents/deepstream-7.1-samples-multiarch.tar}"
FORCE="${FORCE_REBUILD:-0}"
ALLOW_PIP_DOWNLOAD="${ALLOW_PIP_DOWNLOAD:-0}"
export SUDO_PASSWORD="${SUDO_PASSWORD:-1}"

WHEELHOUSE="${DIR}/wheelhouse"
DEBS="${DIR}/vendor/debs"
PYDS_WHEEL="${ROOT}/deepstream/plate_pipeline/pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl"
PYDEPS="${ROOT}/deepstream/plate_pipeline/pydeps"
STAGE="${DIR}/.build_ctx"

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}" docker "$@"
  else
    echo "${SUDO_PASSWORD}" | sudo -S env DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}" docker "$@"
  fi
}

resolve_base() {
  if docker_cmd image inspect "${BASE_PRIMARY}" >/dev/null 2>&1; then
    echo "${BASE_PRIMARY}"
    return 0
  fi
  if docker_cmd image inspect "${BASE_ALT}" >/dev/null 2>&1; then
    echo "${BASE_ALT}"
    return 0
  fi
  return 1
}

ensure_wheelhouse() {
  mkdir -p "${WHEELHOUSE}"
  local marker="${WHEELHOUSE}/numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
  local fastapi_whl
  fastapi_whl="$(ls -1 "${WHEELHOUSE}"/fastapi-*.whl 2>/dev/null | head -1 || true)"
  local minio_whl
  minio_whl="$(ls -1 "${WHEELHOUSE}"/minio-*.whl 2>/dev/null | head -1 || true)"
  local cv_whl
  cv_whl="$(ls -1 "${WHEELHOUSE}"/opencv_python_headless-*.whl 2>/dev/null | head -1 || true)"

  if [[ -f "${marker}" && -n "${fastapi_whl}" && -n "${minio_whl}" && -n "${cv_whl}" ]]; then
    echo "Wheelhouse OK ($(ls -1 "${WHEELHOUSE}"/*.whl 2>/dev/null | wc -l) wheels)"
    return 0
  fi

  if [[ "${ALLOW_PIP_DOWNLOAD}" != "1" ]]; then
    cat >&2 <<EOF
Missing cp310 wheels in ${WHEELHOUSE}.
Place wheels there, or re-run with ALLOW_PIP_DOWNLOAD=1 (PyPI only; no docker pull / no cupy).

  ALLOW_PIP_DOWNLOAD=1 ${DIR}/build.sh
EOF
    exit 1
  fi

  echo "ALLOW_PIP_DOWNLOAD=1: fetching cp310 manylinux wheels into ${WHEELHOUSE} ..."
  pip3 download \
    --dest "${WHEELHOUSE}" \
    --python-version 310 \
    --only-binary=:all: \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --abi cp310 \
    -r "${DIR}/requirements.txt"
}

ensure_libpython_debs() {
  mkdir -p "${DEBS}"
  local n gir gst
  n="$(ls -1 "${DEBS}"/libpython3.10*.deb 2>/dev/null | wc -l)"
  gir="$(ls -1 "${DEBS}"/gir1.2-gstreamer-1.0_*.deb 2>/dev/null | head -1 || true)"
  gst="$(ls -1 "${DEBS}"/python3-gst-1.0_*.deb 2>/dev/null | head -1 || true)"
  if [[ "${n}" -ge 3 && -n "${gir}" && -n "${gst}" ]]; then
    echo "Ubuntu debs OK (${n} libpython + GStreamer GI in ${DEBS})"
    return 0
  fi
  cat >&2 <<EOF
Missing Ubuntu debs in ${DEBS} (need libpython for pyds + gir1.2-gstreamer / python3-gst for Gst GI).
Populate offline (Ubuntu archive only, not NGC), e.g. from a local DS container:

  docker run --rm -v ${DEBS}:/out ${BASE_PRIMARY} bash -lc \\
    'sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true; \\
     apt-get update -qq && cd /out && apt-get download \\
       libpython3.10 libpython3.10-stdlib libpython3.10-minimal \\
       gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 python3-gst-1.0'

Then re-run: ${DIR}/build.sh
EOF
  exit 1
}

stage_build_context() {
  rm -rf "${STAGE}"
  mkdir -p "${STAGE}/pyds" "${STAGE}/wheelhouse" "${STAGE}/vendor/debs" \
           "${STAGE}/ocr_src" "${STAGE}/pkg"

  if [[ ! -f "${PYDS_WHEEL}" ]]; then
    echo "Missing pyds wheel: ${PYDS_WHEEL}" >&2
    echo "Run deepstream/plate_pipeline/setup_pyds.sh first." >&2
    exit 1
  fi
  cp -f "${PYDS_WHEEL}" "${STAGE}/pyds/"

  cp -f "${DIR}/requirements.txt" "${STAGE}/requirements.txt"
  cp -f "${DIR}/Dockerfile" "${STAGE}/Dockerfile"
  cp -a "${WHEELHOUSE}/." "${STAGE}/wheelhouse/"
  cp -a "${DEBS}/." "${STAGE}/vendor/debs/"

  if [[ ! -d "${PYDEPS}/pycuda" || ! -d "${PYDEPS}/pycuda-2024.1.2.dist-info" ]]; then
    echo "Missing vendored pycuda under ${PYDEPS}" >&2
    exit 1
  fi
  if [[ ! -d "${PYDEPS}/tensorrt" || ! -d "${PYDEPS}/tensorrt-10.3.0.dist-info" ]]; then
    echo "Missing vendored tensorrt under ${PYDEPS}" >&2
    exit 1
  fi
  cp -a "${PYDEPS}/pycuda" "${STAGE}/vendor/pycuda"
  cp -a "${PYDEPS}/pycuda-2024.1.2.dist-info" "${STAGE}/vendor/pycuda-2024.1.2.dist-info"
  cp -a "${PYDEPS}/tensorrt" "${STAGE}/vendor/tensorrt"
  cp -a "${PYDEPS}/tensorrt-10.3.0.dist-info" "${STAGE}/vendor/tensorrt-10.3.0.dist-info"

  local ocr
  for ocr in __init__.py recognizer_gpu.py decoder.py trt_loader.py; do
    if [[ ! -f "${ROOT}/license-plate/src/${ocr}" ]]; then
      echo "Missing OCR source: license-plate/src/${ocr}" >&2
      exit 1
    fi
    cp -f "${ROOT}/license-plate/src/${ocr}" "${STAGE}/ocr_src/${ocr}"
  done

  # Package modules only (not wheelhouse / Dockerfile / build artifacts)
  shopt -s nullglob
  local py
  for py in "${DIR}"/*.py; do
    cp -f "${py}" "${STAGE}/pkg/"
  done
  shopt -u nullglob
  if [[ ! -f "${STAGE}/pkg/app.py" ]]; then
    echo "Missing ${DIR}/app.py" >&2
    exit 1
  fi
  if [[ ! -f "${DIR}/entrypoint.sh" ]]; then
    echo "Missing ${DIR}/entrypoint.sh" >&2
    exit 1
  fi
  cp -f "${DIR}/entrypoint.sh" "${STAGE}/entrypoint.sh"
}

# --- main ---
if [[ "${FORCE}" != "1" ]] && docker_cmd image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image already built: ${IMAGE} (set FORCE_REBUILD=1 to rebuild)"
  docker_cmd images "${IMAGE}"
  exit 0
fi

BASE="$(resolve_base || true)"
if [[ -z "${BASE}" ]]; then
  cat >&2 <<EOF
Missing DeepStream 7.1 base image (checked ${BASE_PRIMARY} and ${BASE_ALT}).
Load the local tar (do not docker pull), e.g.:

  sudo docker load -i ${BASE_TAR}

Then re-run: ${DIR}/build.sh
EOF
  exit 1
fi
echo "Using local base: ${BASE}"

ensure_wheelhouse
ensure_libpython_debs
stage_build_context

# Retag alt base to the FROM name in Dockerfile when needed
if [[ "${BASE}" != "nvcr.io/nvidia/deepstream:7.1-samples-multiarch" ]]; then
  echo "Tagging ${BASE} as nvcr.io/nvidia/deepstream:7.1-samples-multiarch for Dockerfile FROM"
  docker_cmd tag "${BASE}" "nvcr.io/nvidia/deepstream:7.1-samples-multiarch"
fi

echo "Building ${IMAGE} (--pull=false) ..."
docker_cmd build --pull=false \
  -f "${STAGE}/Dockerfile" \
  -t "${IMAGE}" \
  "${STAGE}"

rm -rf "${STAGE}"
echo "Ready: ${IMAGE}"
docker_cmd images "${IMAGE}"
