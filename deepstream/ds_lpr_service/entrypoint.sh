#!/usr/bin/env bash
# Runtime entry: ensure DeepStream plugins are discoverable (CUDA present), then exec CMD.
set -euo pipefail

export GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream}"
export GST_PLUGIN_SYSTEM_PATH_1_0="${GST_PLUGIN_SYSTEM_PATH_1_0:-/usr/lib/x86_64-linux-gnu/gstreamer-1.0}"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# Avoid GIO→libproxy "Unable to read configuration" abort on RTSP (known DS issue).
export GIO_USE_PROXY=no
export no_proxy="${no_proxy:-*}"
export NO_PROXY="${NO_PROXY:-*}"

# Build-time Gst.init (no libcuda) blacklists DeepStream .so files; drop that cache.
rm -rf /root/.cache/gstreamer-1.0

# libgiolibproxy throws uncaught C++ from libproxy when RTSP resolves URLs.
for gio_mod in \
  /usr/lib/x86_64-linux-gnu/gio/modules/libgiolibproxy.so \
  /usr/lib/x86_64-linux-gnu/gio/modules/libgiolibproxy.so.*; do
  if [[ -e "${gio_mod}" ]]; then
    rm -f "${gio_mod}" || true
  fi
done
# Refresh GIO module cache if present (ignore failures).
if command -v gio-querymodules >/dev/null 2>&1; then
  gio-querymodules /usr/lib/x86_64-linux-gnu/gio/modules >/dev/null 2>&1 || true
fi

exec "$@"
