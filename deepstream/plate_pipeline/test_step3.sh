#!/usr/bin/env bash
# Step 3 gate: OCR unit test in gpu-decoding:nvdec-trt (has TensorRT Python).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

IMAGE_ARG="${1:-}"
trt_run '
  set -euo pipefail
  pip3 install -q cupy-cuda12x opencv-python-headless 2>/dev/null || pip install -q cupy-cuda12x opencv-python-headless
  python3 /workspace/deepstream/plate_pipeline/test_ocr_unit.py \
    --engine /workspace/license-plate/models/rb_scaner_fp16.engine \
    --dict /workspace/license-plate/dictionary_plate/dict90.txt \
    '"${IMAGE_ARG:+--image $IMAGE_ARG}"'
'
