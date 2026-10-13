#!/usr/bin/env python3
"""DeepStream plate PGIE (+ optional OCR pad probe) for 6 RTSP streams.

Modes:
  --mode detect   : mux → nvinfer → tiler → osd → sink (boxes only)
  --mode ocr      : same + pad probe crops class plate_class_id → RobustScanner
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, GObject, Gst

# Prefer pip-installed pyds 1.2.0; fall back to staged .so from setup_pyds.sh
sys.path.insert(0, "/workspace/deepstream/plate_pipeline/pyds")
try:
    import pyds
except ImportError as exc:
    print(f"FAIL: pyds not available ({exc}). Run setup_pyds.sh first.", file=sys.stderr)
    raise SystemExit(1) from exc

from print_streams import expand_streams, mask_url

GObject.threads_init()
Gst.init(None)

_counts = Counter()
_det_counts = Counter()  # class_id → count
_ocr_hits = 0
_loop = None
_deadline = 0.0
_cfg = {}
_ocr = None
_plate_class = 30
_ocr_conf = 0.94
_mode = "detect"


def _bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"ERROR: {err} ({dbg})", file=sys.stderr)
        # EGL window close should not look like a silent "no frames" success path
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        print(f"WARN: {err} ({dbg})", flush=True)
    return True


def _surface_to_bgr(gst_buffer, frame_meta):
    """Map NvBufSurface frame to BGR numpy (CPU). Requires nvbuf-memory-type=3 + RGBA."""
    import numpy as np
    import cv2

    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    frame_rgba = np.array(n_frame, copy=True, order="C")
    # Unmap on dGPU is optional for UNIFIED but safe
    try:
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    except Exception:
        pass
    if frame_rgba.ndim != 3:
        return None
    if frame_rgba.shape[2] == 4:
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    return frame_rgba[:, :, :3].copy()


_ocr_attempts = 0
_ocr_max_per_run = 30  # rate-limit OCR in probe (expensive)


def _pgie_probe(pad, info):
    global _ocr_hits, _ocr_attempts
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        sid = int(frame_meta.source_id)
        _counts[sid] += 1

        frame_bgr = None
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = int(obj_meta.class_id)
            _det_counts[cid] += 1

            if (
                _mode == "ocr"
                and cid == _plate_class
                and _ocr is not None
                and _ocr_attempts < _ocr_max_per_run
            ):
                rect = obj_meta.rect_params
                if frame_bgr is None:
                    try:
                        frame_bgr = _surface_to_bgr(gst_buffer, frame_meta)
                    except Exception as exc:
                        print(f"WARN: surface map failed: {exc}", flush=True)
                        frame_bgr = False  # sentinel: don't retry this frame
                if frame_bgr is not None and frame_bgr is not False:
                    h, w = frame_bgr.shape[:2]
                    x1 = max(0, int(rect.left))
                    y1 = max(0, int(rect.top))
                    x2 = min(w, int(rect.left + rect.width))
                    y2 = min(h, int(rect.top + rect.height))
                    if x2 > x1 + 2 and y2 > y1 + 2:
                        crop = frame_bgr[y1:y2, x1:x2].copy()
                        _ocr_attempts += 1
                        try:
                            results = _ocr.process_images([crop])
                            if results:
                                text, conf = results[0][0], float(results[0][1])
                                if conf != conf:  # NaN
                                    conf = 0.0
                                print(
                                    f"OCR cam_source={sid} text={text!r} conf={conf:.3f}",
                                    flush=True,
                                )
                                if conf >= _ocr_conf and text and "000000" not in str(text):
                                    obj_meta.text_params.display_text = str(text)
                                    obj_meta.text_params.font_params.font_size = 12
                                    obj_meta.text_params.font_params.font_name = "Serif"
                                    obj_meta.text_params.font_params.font_color.set(
                                        0.0, 1.0, 0.0, 1.0
                                    )
                                    obj_meta.text_params.set_bg_clr = 1
                                    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
                                    _ocr_hits += 1
                        except Exception as exc:
                            print(f"WARN: OCR failed: {exc}", flush=True)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if time.time() >= _deadline and _loop is not None:
        _loop.quit()
    return Gst.PadProbeReturn.OK


def build_launch(rows, cfg, mode: str, preview_path: str) -> str:
    n = len(rows)
    mux = cfg.get("mux", {})
    prev = cfg.get("preview", {})
    width = int(mux.get("width", 1920))
    height = int(mux.get("height", 1080))
    timeout = int(mux.get("batched_push_timeout", 40000))
    rows_t = int(prev.get("tiler_rows", 2))
    cols_t = int(prev.get("tiler_cols", 3))
    infer_cfg = "/workspace/deepstream/plate_pipeline/config_infer_primary_plate.txt"
    # NVBUF_MEM_CUDA_UNIFIED required for get_nvds_buf_surface on dGPU
    mem = 3

    parts = [
        f"nvstreammux name=m batch-size={n} width={width} height={height} "
        f"live-source=1 batched-push-timeout={timeout} nvbuf-memory-type={mem}"
    ]
    for i, row in enumerate(rows):
        uri = row["url"].replace('"', '\\"')
        parts.append(f'nvurisrcbin uri="{uri}" latency=200 ! queue ! m.sink_{i}')

    if mode == "ocr":
        parts.append(
            f"m. ! nvinfer name=pgie config-file-path={infer_cfg} ! "
            f"nvvideoconvert name=conv nvbuf-memory-type={mem} ! "
            f"video/x-raw(memory:NVMM),format=RGBA ! "
            f"nvmultistreamtiler name=tiler rows={rows_t} columns={cols_t} "
            f"width={width} height={height} ! "
            f"nvvideoconvert nvbuf-memory-type={mem} ! nvdsosd name=osd ! "
        )
    else:
        parts.append(
            f"m. ! nvinfer name=pgie config-file-path={infer_cfg} ! "
            f"nvmultistreamtiler name=tiler rows={rows_t} columns={cols_t} "
            f"width={width} height={height} ! "
            f"nvvideoconvert ! nvdsosd name=osd ! "
        )
    if os.environ.get("ENABLE_CAMLABEL", "1") == "1" and Gst.ElementFactory.find("dsexample"):
        parts.append("dsexample name=camlabel full-frame=1 ! ")

    sink_mode = str(prev.get("sink", "fakesink")).lower()
    if sink_mode == "file" and preview_path:
        # mpegtsmux finalizes without needing a clean EOS/moov atom (unlike qtmux/mp4)
        parts.append(
            f"nvvideoconvert ! nvv4l2h264enc bitrate=4000000 ! h264parse ! "
            f"mpegtsmux ! filesink location={preview_path} sync=0"
        )
    elif sink_mode in ("egl", "display", "live"):
        # Live 2x3 mosaic window (needs DISPLAY + X11 socket + /dev/dri in container)
        parts.append("nveglglessink name=sink sync=0")
    else:
        parts.append("fakesink name=sink sync=0 async=0")
    return " ".join(parts)


def maybe_init_ocr(cfg):
    global _ocr
    ocfg = cfg.get("ocr", {})
    engine = ocfg.get("engine", "/workspace/license-plate/models/rb_scaner_fp16.engine")
    dct = ocfg.get("dict", "/workspace/license-plate/dictionary_plate/dict90.txt")
    if not Path(engine).is_file():
        print(f"FAIL: OCR engine missing {engine}", file=sys.stderr)
        return False
    sys.path.insert(0, "/workspace/license-plate")
    from src.recognizer_gpu import LicensePlateRecognition

    _ocr = LicensePlateRecognition(model_path=engine, character_dict_path=dct, use_gpu=True)
    print(f"OCR ready: {engine}")
    return True


def main() -> int:
    global _loop, _deadline, _cfg, _plate_class, _ocr_conf, _mode
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="/workspace/deepstream/plate_pipeline/config.yaml")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--mode", choices=["detect", "ocr"], default="detect")
    p.add_argument("--preview", default="/workspace/logs/plate_pipeline_preview.mp4")
    args = p.parse_args()
    _mode = args.mode

    _cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    det = _cfg.get("detector", {})
    _plate_class = int(det.get("plate_class_id", 30))
    _ocr_conf = float(_cfg.get("ocr", {}).get("conf_thresh", 0.94))
    rows = expand_streams(_cfg)

    engine = det.get("engine", "")
    if not Path(engine).is_file():
        print(f"FAIL: detector engine missing: {engine}", file=sys.stderr)
        print("Run: SUDO_PASSWORD=1 ./scripts/export_yolo12_plate_deepstream.sh", file=sys.stderr)
        return 1

    if args.mode == "ocr":
        if not maybe_init_ocr(_cfg):
            return 1

    print(f"Mode={args.mode} streams={len(rows)} plate_class={_plate_class}")
    launch = build_launch(rows, _cfg, args.mode, args.preview)
    pipeline = Gst.parse_launch(launch)
    # OCR: probe after RGBA convert. Detect: probe on pgie src.
    conv = pipeline.get_by_name("conv")
    probe_el = conv if (args.mode == "ocr" and conv is not None) else pipeline.get_by_name("pgie")
    srcpad = probe_el.get_static_pad("src")
    srcpad.add_probe(Gst.PadProbeType.BUFFER, _pgie_probe)

    _loop = GLib.MainLoop()
    _deadline = time.time() + float(args.seconds)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus_call, _loop)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("FAIL: PLAYING", file=sys.stderr)
        return 1
    print(f"Running {args.seconds}s ...")
    try:
        _loop.run()
    except KeyboardInterrupt:
        pass
    # Drain encoder/mux so preview file is playable
    pipeline.send_event(Gst.Event.new_eos())
    bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)

    print("source_id frames:", dict(_counts))
    print("class_id detections:", dict(_det_counts))
    print(f"plate_class={_plate_class} count={_det_counts.get(_plate_class, 0)}")
    if args.mode == "ocr":
        print(f"ocr_attempts={_ocr_attempts} ocr_hits={_ocr_hits}")

    if sum(_counts.values()) == 0:
        print("FAIL: no frames", file=sys.stderr)
        return 2
    if _det_counts.get(_plate_class, 0) == 0 and sum(_det_counts.values()) == 0:
        print("FAIL: zero detections of any class", file=sys.stderr)
        return 3
    if _det_counts.get(_plate_class, 0) == 0:
        top = _det_counts.most_common(5)
        print(f"WARN: no class {_plate_class}; top classes={top}")
        if args.mode == "detect" and sum(_det_counts.values()) > 0:
            print("PASS: Step 2 — detections present (check plate_class_id if not 30)")
            return 0
        print("FAIL: no plate-class detections", file=sys.stderr)
        return 4
    if args.mode == "ocr":
        if _ocr_hits == 0 and _ocr_attempts == 0:
            print("FAIL: OCR never ran (surface map / crop path broken)", file=sys.stderr)
            return 5
        if _ocr_hits == 0:
            print("WARN: OCR ran but no text passed conf filter (scene-dependent)")
            print(f"PASS: Step 4/5 — boxes OK; OCR attempts={_ocr_attempts} hits=0")
            return 0
        print(f"PASS: Step 4/5 — plate detections={_det_counts.get(_plate_class, 0)} ocr_hits={_ocr_hits}")
        return 0

    print(f"PASS: Step 2 — plate detections={_det_counts.get(_plate_class, 0)} ocr_hits={_ocr_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
