#!/usr/bin/env python3
"""2 RTSP → detect plates → OCR → save annotated frames under output/."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, GObject, Gst

sys.path.insert(0, "/workspace/deepstream/plate_pipeline/pyds")
import pyds  # noqa: E402

from print_streams import expand_streams, mask_url  # noqa: E402

GObject.threads_init()
Gst.init(None)

_counts: Counter = Counter()
_det_counts: Counter = Counter()
_ocr_hits = 0
_saved = 0
_loop = None
_deadline = 0.0
_ocr = None
_plate_class = 30
_ocr_conf = 0.25
_save_dir = Path("/workspace/output")
_every_n = 15
_max_save = 40
_frame_tick: Counter = Counter()


def _bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"ERROR: {err} ({dbg})", file=sys.stderr)
        loop.quit()
    return True


def _surface_to_bgr(gst_buffer, frame_meta):
    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    frame_rgba = np.array(n_frame, copy=True, order="C")
    try:
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    except Exception:
        pass
    if frame_rgba.ndim != 3:
        return None
    if frame_rgba.shape[2] == 4:
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    return frame_rgba[:, :, :3].copy()


def _draw_and_maybe_save(frame_bgr, sid, plates):
    """plates: list of (x1,y1,x2,y2,text,conf)."""
    global _saved
    out = frame_bgr.copy()
    for x1, y1, x2, y2, text, conf in plates:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text} {conf:.2f}" if text else f"plate {conf:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"cam={sid}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    _frame_tick[sid] += 1
    if _frame_tick[sid] % _every_n != 0:
        return
    if _saved >= _max_save:
        if _loop is not None:
            _loop.quit()
        return
    _save_dir.mkdir(parents=True, exist_ok=True)
    path = _save_dir / f"cam{sid}_{_saved:04d}.jpg"
    cv2.imwrite(str(path), out)
    _saved += 1
    texts = [p[4] for p in plates if p[4]]
    print(f"SAVED {path.name} plates={len(plates)} ocr={texts}", flush=True)
    if _saved >= _max_save and _loop is not None:
        _loop.quit()


def _probe(pad, info):
    global _ocr_hits
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
        plates = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = int(obj_meta.class_id)
            _det_counts[cid] += 1
            if cid != _plate_class:
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                continue

            rect = obj_meta.rect_params
            x1 = max(0, int(rect.left))
            y1 = max(0, int(rect.top))
            x2 = int(rect.left + rect.width)
            y2 = int(rect.top + rect.height)
            text = ""
            conf = float(obj_meta.confidence)

            if _ocr is not None:
                if frame_bgr is None:
                    try:
                        frame_bgr = _surface_to_bgr(gst_buffer, frame_meta)
                    except Exception as exc:
                        print(f"WARN: surface map failed: {exc}", flush=True)
                        frame_bgr = False
                if frame_bgr is not None and frame_bgr is not False:
                    h, w = frame_bgr.shape[:2]
                    x2c = min(w, x2)
                    y2c = min(h, y2)
                    if x2c > x1 + 2 and y2c > y1 + 2:
                        crop = frame_bgr[y1:y2c, x1:x2c].copy()
                        try:
                            results = _ocr.process_images([crop])
                            if results:
                                text, oconf = results[0][0], float(results[0][1])
                                if oconf != oconf:
                                    oconf = 0.0
                                if oconf >= _ocr_conf and text and "000000" not in str(text):
                                    conf = oconf
                                    _ocr_hits += 1
                                    obj_meta.text_params.display_text = str(text)
                                    print(
                                        f"OCR cam={sid} text={text!r} conf={oconf:.3f}",
                                        flush=True,
                                    )
                                else:
                                    text = ""
                        except Exception as exc:
                            print(f"WARN: OCR failed: {exc}", flush=True)

            plates.append((x1, y1, x2, y2, text, conf))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        if frame_bgr is not None and frame_bgr is not False:
            _draw_and_maybe_save(frame_bgr, sid, plates)
        elif plates:
            # still try map once for save
            try:
                frame_bgr = _surface_to_bgr(gst_buffer, frame_meta)
                if frame_bgr is not None:
                    _draw_and_maybe_save(frame_bgr, sid, plates)
            except Exception:
                pass

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if time.time() >= _deadline and _loop is not None:
        _loop.quit()
    return Gst.PadProbeReturn.OK


def build_launch(rows, cfg) -> str:
    n = len(rows)
    mux = cfg.get("mux", {})
    width = int(mux.get("width", 1920))
    height = int(mux.get("height", 1080))
    timeout = int(mux.get("batched_push_timeout", 40000))
    infer = "/workspace/deepstream/plate_pipeline/config_infer_primary_plate.txt"
    mem = 3  # CUDA unified — required for get_nvds_buf_surface

    parts = [
        f"nvstreammux name=m batch-size={n} width={width} height={height} "
        f"live-source=1 batched-push-timeout={timeout} nvbuf-memory-type={mem}"
    ]
    for i, row in enumerate(rows):
        uri = row["url"].replace('"', '\\"')
        parts.append(f'nvurisrcbin uri="{uri}" latency=200 ! queue ! m.sink_{i}')

    # 1x2 tiler for 2 cams; RGBA before tiler so probe can map per-source frames
    parts.append(
        f"m. ! nvinfer name=pgie config-file-path={infer} ! "
        f"nvvideoconvert name=conv nvbuf-memory-type={mem} ! "
        f"video/x-raw(memory:NVMM),format=RGBA ! "
        f"nvmultistreamtiler rows=1 columns={n} width={width} height={height} ! "
        f"nvvideoconvert nvbuf-memory-type={mem} ! nvdsosd ! "
        f"fakesink sync=0 async=0"
    )
    return " ".join(parts)


def main() -> int:
    global _loop, _deadline, _ocr, _plate_class, _ocr_conf
    global _save_dir, _every_n, _max_save

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="/workspace/deepstream/plate_pipeline/config_2cam.yaml")
    p.add_argument("--seconds", type=float, default=45.0)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    rows = expand_streams(cfg)
    det = cfg.get("detector", {})
    ocfg = cfg.get("ocr", {})
    scfg = cfg.get("save", {})
    _plate_class = int(det.get("plate_class_id", 30))
    _ocr_conf = float(ocfg.get("conf_thresh", 0.25))
    _save_dir = Path(scfg.get("dir", "/workspace/output"))
    _every_n = int(scfg.get("every_n_frames", 15))
    _max_save = int(scfg.get("max_frames", 40))

    engine = det.get("engine", "")
    if not Path(engine).is_file():
        print(f"FAIL: missing detector engine {engine}", file=sys.stderr)
        return 1
    ocr_engine = ocfg.get("engine", "")
    if not Path(ocr_engine).is_file():
        print(f"FAIL: missing OCR engine {ocr_engine}", file=sys.stderr)
        return 1

    sys.path.insert(0, "/workspace/license-plate")
    from src.recognizer_gpu import LicensePlateRecognition

    _ocr = LicensePlateRecognition(
        model_path=ocr_engine,
        character_dict_path=ocfg.get("dict"),
        use_gpu=True,
    )
    print(f"OCR ready: {ocr_engine}")

    print(f"Streams={len(rows)} save={_save_dir} every_n={_every_n} max={_max_save}")
    for r in rows:
        print(f"  source_id={r['source_id']} {r['label']} {mask_url(r['url'])}")

    launch = build_launch(rows, cfg)
    pipeline = Gst.parse_launch(launch)
    conv = pipeline.get_by_name("conv")
    conv.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, _probe)

    _loop = GLib.MainLoop()
    _deadline = time.time() + float(args.seconds)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus_call, _loop)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("FAIL: PLAYING", file=sys.stderr)
        return 1
    print(f"Running up to {args.seconds}s (or {_max_save} saved frames) ...")
    try:
        _loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)

    print("source_id frames:", dict(_counts))
    print("class_id detections:", dict(_det_counts))
    print(f"plate_class={_plate_class} count={_det_counts.get(_plate_class, 0)}")
    print(f"ocr_hits={_ocr_hits} saved={_saved} → {_save_dir}")
    if sum(_counts.values()) == 0:
        return 2
    if _saved == 0:
        print("FAIL: no frames saved", file=sys.stderr)
        return 3
    print(f"PASS: 2-cam run saved {_saved} frames to {_save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
