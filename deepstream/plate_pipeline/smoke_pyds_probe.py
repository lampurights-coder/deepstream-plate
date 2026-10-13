#!/usr/bin/env python3
"""Minimal pyds + pad-probe smoke test."""
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, GObject, Gst

import pyds

print("pyds", pyds.__file__, getattr(pyds, "__version__", "?"), flush=True)
GObject.threads_init()
Gst.init(None)

uri = sys.argv[1] if len(sys.argv) > 1 else (
    "rtsp://user:pass@CAMERA_HOST:554/cam/realmonitor?channel=1&subtype=0"
)
cfg = sys.argv[2] if len(sys.argv) > 2 else "/workspace/deepstream/config_infer_primary_yolo.txt"
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

launch = (
    f"nvstreammux name=m batch-size=1 width=1280 height=720 live-source=1 batched-push-timeout=40000 "
    f'nvurisrcbin uri="{uri}" latency=200 ! queue ! m.sink_0 '
    f"m. ! nvinfer name=pgie config-file-path={cfg} ! "
    f"fakesink sync=0"
)
print("launch ok", flush=True)
pipe = Gst.parse_launch(launch)
n = [0]
dets = {}


def probe(pad, info):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        n[0] += 1
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        n[0] += 1
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = int(obj_meta.class_id)
            dets[cid] = dets.get(cid, 0) + 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    if n[0] <= 3:
        print(f"probe frames={n[0]} dets={dets}", flush=True)
    return Gst.PadProbeReturn.OK


pipe.get_by_name("pgie").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
loop = GLib.MainLoop()


def _stop():
    print(f"DONE frames={n[0]} dets={dets}", flush=True)
    loop.quit()
    return False


GLib.timeout_add_seconds(int(seconds), _stop)
bus = pipe.get_bus()
bus.add_signal_watch()


def on_msg(bus, msg):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print("ERR", err, dbg, flush=True)
        loop.quit()


bus.connect("message", on_msg)
print(pipe.set_state(Gst.State.PLAYING), flush=True)
loop.run()
pipe.set_state(Gst.State.NULL)
if n[0] == 0:
    raise SystemExit(2)
print("PASS", flush=True)
