#!/usr/bin/env bash
# Install pyds 1.2.0 bindings for DeepStream 7.1 (prebuilt wheel).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

HOST_PYDS="${PP}/pyds"
WHEEL_NAME="pyds-1.2.0-cp310-cp310-linux_x86_64.whl"
WHEEL_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/${WHEEL_NAME}"
HOST_WHEEL="${HOST_PYDS}/${WHEEL_NAME}"

mkdir -p "${HOST_PYDS}"
if [[ ! -f "${HOST_WHEEL}" ]]; then
  echo "Downloading ${WHEEL_URL} ..."
  curl -fsSL -o "${HOST_WHEEL}" "${WHEEL_URL}"
fi

echo "Installing pyds 1.2.0 into DS 7.1 container (staged under ${HOST_PYDS}) ..."
ds_run '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sed -i "/developer.download.nvidia.com/s/^deb/#deb/" /etc/apt/sources.list.d/cuda*.list 2>/dev/null || true
  apt-get update -qq >/dev/null
  apt-get install -y -qq libpython3.10 python3-pip >/dev/null
  pip3 install -q --force-reinstall "/workspace/deepstream/plate_pipeline/pyds/'"${WHEEL_NAME}"'"
  # Also extract .so next to wheel for PYTHONPATH import without site-packages
  python3 - <<PY
import pyds, os, shutil, pathlib
src = pathlib.Path(pyds.__file__)
dst = pathlib.Path("/workspace/deepstream/plate_pipeline/pyds")
print("pyds from", src)
# Copy module file(s)
if src.suffix == ".so":
    shutil.copy2(src, dst / "pyds.so")
else:
    # package dir
    for p in src.parent.glob("pyds*"):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
print("staged", list(dst.iterdir()))
PY
  python3 -c "import sys; sys.path.insert(0,\"/workspace/deepstream/plate_pipeline/pyds\"); import pyds; print(\"pyds OK\", getattr(pyds,\"__version__\", \"?\"), pyds)"
'
echo "pyds 1.2.0 ready"
ls -la "${HOST_PYDS}"
