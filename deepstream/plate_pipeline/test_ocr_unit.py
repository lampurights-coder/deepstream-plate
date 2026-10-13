#!/usr/bin/env python3
"""Step 3: OCR unit test — SAR preprocess + rb_scaner_fp16.engine + SARLabelDecode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engine",
        default="/workspace/license-plate/models/rb_scaner_fp16.engine",
    )
    p.add_argument(
        "--dict",
        default="/workspace/license-plate/dictionary_plate/dict90.txt",
    )
    p.add_argument(
        "--image",
        default="",
        help="Plate crop JPEG/PNG. If empty, synthesize a simple plate-like crop.",
    )
    args = p.parse_args()

    # Ensure license-plate package imports work
    sys.path.insert(0, "/workspace/license-plate")
    from src.recognizer_gpu import LicensePlateRecognition

    if args.image and Path(args.image).is_file():
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"FAIL: cannot read {args.image}", file=sys.stderr)
            return 1
        print(f"Loaded crop {args.image} shape={img.shape}")
    else:
        # Synthetic BGR crop so the gate can run without a live plate crop
        img = np.full((48, 160, 3), 220, dtype=np.uint8)
        cv2.putText(
            img,
            "12ABC34",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        out = Path("/workspace/logs/ocr_unit_synth_crop.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"Wrote synth crop → {out}")

    rec = LicensePlateRecognition(
        model_path=args.engine,
        character_dict_path=args.dict,
        use_gpu=True,
        gpu_id=0,
    )
    results = rec.process_images([img])
    if not results:
        print("FAIL: empty OCR results", file=sys.stderr)
        return 2

    text, conf = results[0][0], float(results[0][1])
    print(f"OCR result: text={text!r} conf={conf:.4f}")
    if not (0.0 < conf <= 1.0):
        print(f"FAIL: conf out of range {conf}", file=sys.stderr)
        return 3

    print("PASS: Step 3 — RobustScanner TRT preprocess/infer/decode works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
