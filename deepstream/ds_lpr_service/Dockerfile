# DeepStream LPR service image — local base only (never pull NGC at build time).
# Build via: deepstream/ds_lpr_service/build.sh
FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace:/workspace/license-plate \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /workspace

# --- Layer 0: libpython + GStreamer GI typelibs (Ubuntu debs, offline; not NVIDIA) ---
COPY vendor/debs/ /tmp/debs/
RUN dpkg -i /tmp/debs/*.deb \
 && ldconfig \
 && rm -rf /tmp/debs \
 && test -e /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 \
 && test -e /usr/lib/x86_64-linux-gnu/girepository-1.0/Gst-1.0.typelib \
 && python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('Gst GI OK', Gst.version_string())"

# --- Layer 1: pyds 1.2.0 bindings for DeepStream 7.1 (local wheel) ---
# Full import needs libcuda (host GPU at runtime); build only checks the wheel + libpython.
COPY pyds/pyds-1.2.0-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip3 install --no-cache-dir /tmp/pyds-1.2.0-cp310-cp310-linux_x86_64.whl \
 && rm -f /tmp/pyds-1.2.0-cp310-cp310-linux_x86_64.whl \
 && python3 - <<'PY'
from pathlib import Path
import subprocess
so = Path("/usr/local/lib/python3.10/dist-packages/pyds.so")
assert so.is_file(), so
out = subprocess.check_output(["ldd", str(so)], text=True)
missing = [l for l in out.splitlines() if "not found" in l]
# libcuda.so.1 is provided by the NVIDIA container runtime at run time
ok_missing = all("libcuda.so" in l for l in missing)
assert ok_missing, missing
print("pyds wheel OK; runtime will supply libcuda")
PY

# --- Layer 2: FastAPI / MinIO / OpenCV / numpy from prefetched wheelhouse ---
COPY requirements.txt /tmp/requirements.txt
COPY wheelhouse/ /tmp/wheelhouse/
RUN pip3 install --no-cache-dir --no-index --find-links=/tmp/wheelhouse \
      -r /tmp/requirements.txt \
 && rm -rf /tmp/wheelhouse /tmp/requirements.txt \
 && python3 -c "import fastapi, minio, cv2, numpy, yaml; print('deps OK', numpy.__version__, cv2.__version__)"

# --- Layer 3: pycuda + tensorrt Python (prebuilt from plate_pipeline/pydeps; no cupy) ---
COPY vendor/pycuda /usr/local/lib/python3.10/dist-packages/pycuda
COPY vendor/pycuda-2024.1.2.dist-info /usr/local/lib/python3.10/dist-packages/pycuda-2024.1.2.dist-info
COPY vendor/tensorrt /usr/local/lib/python3.10/dist-packages/tensorrt
COPY vendor/tensorrt-10.3.0.dist-info /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info
RUN test -e /usr/local/lib/python3.10/dist-packages/pycuda/_driver.cpython-310-x86_64-linux-gnu.so \
 && test -e /usr/local/lib/python3.10/dist-packages/tensorrt/tensorrt.so \
 && python3 -c "import importlib.util; assert importlib.util.find_spec('pycuda'); assert importlib.util.find_spec('tensorrt'); print('pycuda+tensorrt present')"

# --- Layer 4: thin OCR sources (recognizer_gpu + TRT session + decoder) ---
COPY ocr_src/ /workspace/license-plate/src/

# --- Layer 5: service package (code last so dep layers stay cached) ---
COPY pkg/ /workspace/ds_lpr_service/

COPY entrypoint.sh /usr/local/bin/ds-lpr-entrypoint.sh
RUN chmod +x /usr/local/bin/ds-lpr-entrypoint.sh

EXPOSE 8003
ENTRYPOINT ["/usr/local/bin/ds-lpr-entrypoint.sh"]
CMD ["uvicorn", "ds_lpr_service.app:app", "--host", "0.0.0.0", "--port", "8003"]
