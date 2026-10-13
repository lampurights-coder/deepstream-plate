#!/usr/bin/env python3
"""Step 1: 6× RTSP → nvurisrcbin → nvstreammux → fakesink (no pyds required).

Gate: each mux sink_%u pad must receive buffers (per-source frame gather).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from print_streams import expand_streams, mask_url

Gst.init(None)

_counts: Counter = Counter()
_loop = None
_deadline = 0.0


def _bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("EOS")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"ERROR: {err} ({dbg})", file=sys.stderr)
        loop.quit()
    return True


def _make_sink_probe(source_id: int):
    def _probe(pad, info):
        _counts[source_id] += 1
        if time.time() >= _deadline and _loop is not None:
            _loop.quit()
        return Gst.PadProbeReturn.OK

    return _probe


def build_launch(rows: list[dict], mux_cfg: dict) -> str:
    n = len(rows)
    width = int(mux_cfg.get("width", 1920))
    height = int(mux_cfg.get("height", 1080))
    timeout = int(mux_cfg.get("batched_push_timeout", 40000))
    parts = [
        f"nvstreammux name=m batch-size={n} width={width} height={height} "
        f"live-source=1 batched-push-timeout={timeout}"
    ]
    for i, row in enumerate(rows):
        uri = row["url"].replace('"', '\\"')
        parts.append(f'nvurisrcbin uri="{uri}" latency=200 ! queue ! m.sink_{i}')
    parts.append("m. ! fakesink name=sink sync=0 async=0")
    return " ".join(parts)


def main() -> int:
    global _loop, _deadline
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="/workspace/deepstream/plate_pipeline/config.yaml")
    p.add_argument("--seconds", type=float, default=20.0)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    rows = expand_streams(cfg)
    print(f"Building mux-only pipeline for {len(rows)} streams ...")
    for r in rows:
        print(f"  source_id={r['source_id']} {r['label']} {mask_url(r['url'])}")

    launch = build_launch(rows, cfg.get("mux", {}))
    pipeline = Gst.parse_launch(launch)
    mux = pipeline.get_by_name("m")
    for i in range(len(rows)):
        pad = mux.get_static_pad(f"sink_{i}")
        if pad is None:
            # request pads may only appear after linking; try get_request style name
            pad = mux.get_static_pad(f"sink_{i}")
        if pad is None:
            # After parse_launch, sinks are usually already requested
            pad = mux.request_pad_simple(f"sink_{i}") if hasattr(mux, "request_pad_simple") else None
        # Enumerate pads
    # Prefer iterating mux sink pads
    it = mux.iterate_sink_pads()
    pads = []
    while True:
        res, pad = it.next()
        if res != Gst.IteratorResult.OK:
            break
        pads.append(pad)
    if not pads:
        # Fallback: get_request_pad names sink_%u after link
        for i in range(len(rows)):
            pad = mux.get_static_pad(f"sink_{i}")
            if pad:
                pads.append(pad)
    print(f"Mux sink pads found: {len(pads)}")
    for i, pad in enumerate(pads):
        name = pad.get_name()
        # sink_0 → 0
        sid = i
        if name.startswith("sink_"):
            try:
                sid = int(name.split("_", 1)[1])
            except ValueError:
                sid = i
        pad.add_probe(Gst.PadProbeType.BUFFER, _make_sink_probe(sid))
        print(f"  probe on {name} → source_id={sid}")

    _loop = GLib.MainLoop()
    _deadline = time.time() + float(args.seconds)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus_call, _loop)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("FAIL: could not set PLAYING", file=sys.stderr)
        return 1
    print(f"Running {args.seconds}s ...")
    try:
        _loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)

    print("source_id frame_counts:")
    missing = []
    for i in range(len(rows)):
        c = _counts.get(i, 0)
        print(f"  {i}: {c}")
        if c <= 0:
            missing.append(i)

    if missing:
        print(f"FAIL: no frames for source_id {missing}", file=sys.stderr)
        return 2

    print(f"PASS: Step 1 — mux delivered frames for all {len(rows)} source_ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
