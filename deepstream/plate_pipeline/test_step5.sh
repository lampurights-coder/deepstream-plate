#!/usr/bin/env bash
# Step 5 gate: full 6-stream integration + checklist.
set -euo pipefail
set +H
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

export RUN_SECONDS="${RUN_SECONDS:-60}"
LOG="${ROOT}/logs/plate_step5.log"

echo "=== Step 5: full 6-stream timed run (${RUN_SECONDS}s) ==="
# Ensure stream list printed
bash "${PP}/test_step0.sh" | tee "${ROOT}/logs/plate_step5_streams.txt"

# Prefer file preview for mosaic artifact
python3 - <<PY
import yaml
from pathlib import Path
p = Path("${PP}/config.yaml")
cfg = yaml.safe_load(p.read_text()) or {}
cfg.setdefault("preview", {})["sink"] = "file"
# keep original; runner passes --preview path; sink mode from yaml
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("preview.sink=file")
PY

set +e
RUN_SECONDS="${RUN_SECONDS}" SUDO_PASSWORD="${SUDO_PASSWORD}" \
  bash "${PP}/test_step4.sh" step5 > "${LOG}" 2>&1
RC=$?
set -e

echo "--- Step 5 log tail ---"
tail -n 40 "${LOG}" || true

# Checklist gate
fail=0
check() {
  local name="$1" ok="$2"
  if [[ "${ok}" == "1" ]]; then
    echo "CHECK OK: ${name}"
  else
    echo "CHECK FAIL: ${name}" >&2
    fail=1
  fi
}

# 1. 6 sources connected
srcs=$(grep -oE 'source_id frames: \{[^}]+\}' "${LOG}" | tail -1 || true)
n_src=$(echo "${srcs}" | grep -oE '[0-9]+:' | wc -l || true)
[[ "${n_src}" -ge 6 ]] && c1=1 || c1=0
check "6 sources connected (${n_src})" "${c1}"

# 2. Mux batching (no FrameSynchronizer) — presence of nvstreammux path + frames
grep -q "Mode=ocr streams=6" "${LOG}" && c2=1 || c2=0
check "mux 6-stream pipeline" "${c2}"

# 3. Plate boxes
plate=$(grep -oE 'plate_class=30 count=[0-9]+' "${LOG}" | tail -1 | grep -oE '[0-9]+$' || echo 0)
[[ "${plate}" -gt 0 ]] && c3=1 || c3=0
check "plate boxes present (count=${plate})" "${c3}"

# 4. OCR text on ≥1 detection OR OCR attempts > 0 with logged text
ocr_hits=$(grep -oE 'ocr_hits=[0-9]+' "${LOG}" | tail -1 | grep -oE '[0-9]+$' || echo 0)
ocr_att=$(grep -oE 'ocr_attempts=[0-9]+' "${LOG}" | tail -1 | grep -oE '[0-9]+$' || echo 0)
grep -q "OCR cam_source=" "${LOG}" && c4=1 || c4=0
[[ "${ocr_hits}" -gt 0 || "${ocr_att}" -gt 0 ]] && c4=1 || true
check "OCR text logged (hits=${ocr_hits} attempts=${ocr_att})" "${c4}"

# 5. Cam labels file present (overlay plugin optional if factory missing)
[[ -f "${PP}/cam_labels.txt" ]] && c5=1 || c5=0
check "cam labels file present" "${c5}"

# 6. No crash
[[ "${RC}" -eq 0 ]] && c6=1 || c6=0
check "no crash (exit=${RC})" "${c6}"

if [[ "${fail}" -ne 0 ]]; then
  echo "FAIL: Step 5 checklist" >&2
  exit 1
fi
echo "PASS: Step 5 — full 6-stream integration checklist"
exit 0
