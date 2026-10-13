#!/usr/bin/env python3
"""Expand plate_pipeline config into 6 (cam_id, source_id, url) rows — Step 0 gate."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def expand_streams(cfg: dict) -> list[dict]:
    rows: list[dict] = []
    source_id = 0
    for cam in cfg.get("rtsp_bases", []):
        cam_id = int(cam["id"])
        url = str(cam["url"]).strip()
        n = int(cam.get("replicate", 1))
        if n < 1:
            raise ValueError(f"cam {cam_id}: replicate must be >= 1")
        if not url.lower().startswith("rtsp://"):
            raise ValueError(f"cam {cam_id}: expected rtsp:// url")
        for replica in range(n):
            rows.append(
                {
                    "source_id": source_id,
                    "cam_id": cam_id,
                    "replica": replica,
                    "label": f"CAM{cam_id}r{replica}",
                    "url": url,
                }
            )
            source_id += 1
    return rows


def mask_url(url: str) -> str:
    return re.sub(r"(rtsp://[^:]+):([^@]+)@", r"\1:***@", url)


def write_cam_labels(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One line per source_id (index = source_id) for gst-camlabel
    lines = [r["label"] for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_urls_file(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r["url"] for r in rows) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
    )
    p.add_argument(
        "--write-labels",
        default="",
        help="Optional path for cam_labels.txt (source_id order)",
    )
    p.add_argument(
        "--write-urls",
        default="",
        help="Optional path for expanded URLs (one per line)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"FAIL: missing config {cfg_path}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    rows = expand_streams(cfg)
    if len(rows) != 6:
        print(f"FAIL: expected 6 streams, got {len(rows)}", file=sys.stderr)
        return 2

    print("source_id  cam_id  replica  label        url")
    for r in rows:
        print(
            f"{r['source_id']:<10} {r['cam_id']:<7} {r['replica']:<8} "
            f"{r['label']:<12} {mask_url(r['url'])}"
        )

    if args.write_labels:
        write_cam_labels(rows, Path(args.write_labels))
        print(f"Wrote labels → {args.write_labels}")
    if args.write_urls:
        write_urls_file(rows, Path(args.write_urls))
        print(f"Wrote urls → {args.write_urls}")

    print(f"PASS: Step 0 — {len(rows)} streams from 2 RTSP bases × replicate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
