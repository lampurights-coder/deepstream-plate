"""Long-lived DeepStream plate detect → OCR pipeline (start/stop, N cams).

Evolved from ``plate_pipeline/run_plate_ocr.py`` for the FastAPI control plane:
build mux→nvinfer→RGBA→probe→tiler→OSD→fakesink from ``cameras.yaml``, run until
``stop()``, and push OCR hits into a ring buffer (optionally via a MinIO uploader).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence

from .ocr_proc import OcrProcessClient, ocr_client_from_config

logger = logging.getLogger("ds_lpr.pipeline")

# Container defaults; host/dev may override via PYTHONPATH.
_PYDS_FALLBACK = "/workspace/deepstream/plate_pipeline/pyds"


class HitUploader(Protocol):
    """Optional MinIO (or other) sink — implemented by ``minio_sink``."""

    def upload_hit(
        self,
        crop: Any,
        *,
        cam_id: str,
        bucket: str,
        ocr: str,
        conf: float,
        bbox: Sequence[float],
        ts: float,
    ) -> Optional[str]:
        """Upload plate crop + JSON; return ``bucket/key`` or ``s3://...``."""
        ...


@dataclass
class PlateHit:
    cam_id: str
    ocr: str
    conf: float
    bucket: str
    object_path: Optional[str]
    bbox: List[float]
    ts: float
    source_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CameraRow:
    source_id: int
    cam_id: str
    url: str
    bucket: str


@dataclass
class _Runtime:
    """Mutable state owned by the GStreamer thread / pad probe."""

    plate_class: int = 30
    ocr_conf: float = 0.25
    ocr: Optional[OcrProcessClient] = None
    uploader: Optional[HitUploader] = None
    source_map: Dict[int, CameraRow] = field(default_factory=dict)
    frame_counts: Counter = field(default_factory=Counter)
    det_counts: Counter = field(default_factory=Counter)
    ocr_attempts: int = 0
    ocr_hits: int = 0
    # Rate-limit OCR in the pad probe (keeps streaming thread responsive).
    ocr_min_interval_s: float = 0.2
    last_ocr_ts: float = 0.0
    # When a plate OCR hits: draw box+text on the frame and write JPEG here.
    save_dir: Optional[Path] = None
    min_box_w: int = 40
    min_box_h: int = 15
    min_det_conf: float = 0.15


def _annotate_plate(
    frame_bgr,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    conf: float,
):
    """Copy of frame with green plate box + OCR label (for save / MinIO)."""
    import cv2

    out = frame_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{text} ({conf:.2f})"
    ty = max(0, y1 - 8)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (x1 + 2, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def _save_annotated_local(
    save_dir: Path,
    cam_id: str,
    annotated,
    text: str,
    ts: float,
) -> Optional[str]:
    """Write ``{save_dir}/{cam_id}/{timestamp}_{ocr}.jpg``. Returns path or None."""
    import cv2

    try:
        dest = save_dir / cam_id
        dest.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        ms = int((ts % 1) * 1_000_000)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:48] or "plate"
        path = dest / f"{stamp}_{ms:06d}_{safe}.jpg"
        if not cv2.imwrite(
            str(path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        ):
            logger.warning("cv2.imwrite failed: %s", path)
            return None
        return str(path)
    except Exception as exc:
        logger.warning("local save failed: %s", exc)
        return None


def cameras_from_config(cfg: dict) -> List[CameraRow]:
    """Parse ``cameras:`` list into mux sink order (source_id = index)."""
    raw = cfg.get("cameras") or []
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.cameras must be a non-empty list")
    rows: List[CameraRow] = []
    buckets: set[str] = set()
    ids: set[str] = set()
    for i, cam in enumerate(raw):
        if not isinstance(cam, dict):
            raise ValueError(f"cameras[{i}] must be a mapping")
        cam_id = str(cam.get("id", "")).strip()
        url = str(cam.get("url", "")).strip()
        bucket = str(cam.get("bucket", "")).strip()
        if not cam_id or not url or not bucket:
            raise ValueError(f"cameras[{i}] requires id, url, and bucket")
        if not url.lower().startswith(("rtsp://", "file://", "http://", "https://")):
            raise ValueError(f"cameras[{i}].url unsupported scheme: {url!r}")
        if cam_id in ids:
            raise ValueError(f"duplicate camera id: {cam_id}")
        if bucket in buckets:
            raise ValueError(f"duplicate camera bucket: {bucket}")
        ids.add(cam_id)
        buckets.add(bucket)
        rows.append(CameraRow(source_id=i, cam_id=cam_id, url=url, bucket=bucket))
    return rows


def build_launch(rows: Sequence[CameraRow], cfg: dict) -> str:
    """OCR-path launch string: unified mem + RGBA before probe (avoids surface segfault).

    Optional ``preview`` in config:
      sink: fakesink | file | egl
      path: /workspace/output/ds_lpr_preview.ts   # for sink=file
      tiler_rows / tiler_cols: override mosaic layout
    """
    n = len(rows)
    mux = cfg.get("mux") or {}
    prev = cfg.get("preview") or {}
    width = int(mux.get("width", 1280))
    height = int(mux.get("height", 720))
    timeout = int(mux.get("batched_push_timeout", 40000))
    det = cfg.get("detector") or {}
    infer_cfg = str(
        det.get(
            "infer_config",
            "/workspace/deepstream/plate_pipeline/config_infer_primary_plate.txt",
        )
    )
    cols = int(prev.get("tiler_cols") or max(1, int(math.ceil(math.sqrt(n)))))
    rows_t = int(prev.get("tiler_rows") or max(1, int(math.ceil(n / cols))))
    mem = 3  # NVBUF_MEM_CUDA_UNIFIED — required for get_nvds_buf_surface on dGPU
    sink_mode = str(prev.get("sink", "fakesink")).lower()
    preview_path = str(
        prev.get("path", "/workspace/output/ds_lpr_preview.ts")
    ).replace('"', "")

    # Prefer dropping late frames over buffering (reduces lag / decode ghosts).
    parts = [
        f"nvstreammux name=m batch-size={n} width={width} height={height} "
        f"live-source=1 batched-push-timeout={timeout} nvbuf-memory-type={mem} "
        f"drop-pipeline-eos=1"
    ]
    for i, row in enumerate(rows):
        uri = row.url.replace('"', '\\"')
        # TCP RTSP (select-rtp-protocol=4) avoids UDP packet loss → macroblock/jitter.
        # leaky queue drops old buffers so the probe sees fresher frames.
        parts.append(
            f'nvurisrcbin uri="{uri}" select-rtp-protocol=4 latency=400 '
            f"drop-on-latency=1 ! "
            f"queue max-size-buffers=2 max-size-time=0 max-size-bytes=0 "
            f"leaky=downstream ! m.sink_{i}"
        )

    parts.append(
        f"m. ! nvinfer name=pgie config-file-path={infer_cfg} ! "
        f"nvvideoconvert name=conv nvbuf-memory-type={mem} ! "
        f"video/x-raw(memory:NVMM),format=RGBA ! "
        f"nvmultistreamtiler name=tiler rows={rows_t} columns={cols} "
        f"width={width} height={height} ! "
        f"nvvideoconvert nvbuf-memory-type={mem} ! nvdsosd name=osd ! "
    )
    # nvdsosd draws YOLO boxes + OCR display_text; choose how you watch it.
    if sink_mode == "file":
        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        parts.append(
            f"nvvideoconvert ! nvv4l2h264enc bitrate=4000000 ! h264parse ! "
            f"mpegtsmux ! filesink name=sink location={preview_path} sync=0"
        )
        logger.info("preview sink=file → %s", preview_path)
    elif sink_mode in ("egl", "display", "live"):
        parts.append("nveglglessink name=sink sync=0")
        logger.info("preview sink=egl (needs DISPLAY + X11 in container)")
    else:
        parts.append("fakesink name=sink sync=0 async=0")
    return " ".join(parts)


def _ensure_gst_pyds():
    """Import Gst / pyds (DeepStream image). Raises ImportError if unavailable."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, GObject, Gst

    if _PYDS_FALLBACK not in sys.path and Path(_PYDS_FALLBACK).is_dir():
        sys.path.insert(0, _PYDS_FALLBACK)
    try:
        import pyds
    except ImportError:
        raise
    GObject.threads_init()
    if not Gst.is_initialized():
        Gst.init(None)
    return Gst, GLib, pyds


def _surface_to_bgr(pyds, gst_buffer, frame_meta):
    import cv2
    import numpy as np

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


class DsLprPipeline:
    """Thread-backed DeepStream runner with start/stop and a results ring buffer."""

    def __init__(self, result_maxlen: int = 200) -> None:
        self._lock = threading.RLock()
        self._results: Deque[PlateHit] = deque(maxlen=max(1, int(result_maxlen)))
        self._thread: Optional[threading.Thread] = None
        self._loop = None  # GLib.MainLoop
        self._gst_pipeline = None
        self._running = False
        self._starting = False
        self._last_error: Optional[str] = None
        self._cameras: List[CameraRow] = []
        self._rt = _Runtime()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._Gst = None
        self._GLib = None
        self._pyds = None
        self._cfg: Optional[dict] = None

    # --- public API (used by FastAPI control plane) ---

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    def health(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "cameras": len(self._cameras),
            "last_error": self._last_error,
            "ocr_attempts": self._rt.ocr_attempts,
            "ocr_hits": self._rt.ocr_hits,
            "frames": dict(self._rt.frame_counts),
            "plate_dets": int(self._rt.det_counts.get(self._rt.plate_class, 0)),
        }

    def get_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        with self._lock:
            items = list(self._results)[-lim:]
        return [h.to_dict() for h in reversed(items)]

    def start(
        self,
        cfg: dict,
        uploader: Optional[HitUploader] = None,
        *,
        join_timeout: float = 0.0,
    ) -> Dict[str, Any]:
        """Build and PLAY the pipeline from config. Returns ``{status, cameras}``."""
        with self._lock:
            if self._running or self._starting:
                raise RuntimeError("pipeline already running")
            self._starting = True
            self._last_error = None
            self._stop_event.clear()
            self._started_event.clear()

        try:
            rows = cameras_from_config(cfg)
            det = cfg.get("detector") or {}
            engine = det.get("engine", "")
            if not engine or not Path(engine).is_file():
                raise FileNotFoundError(f"detector engine missing: {engine}")
            infer_cfg = det.get(
                "infer_config",
                "/workspace/deepstream/plate_pipeline/config_infer_primary_plate.txt",
            )
            if not Path(str(infer_cfg)).is_file():
                raise FileNotFoundError(f"infer config missing: {infer_cfg}")

            Gst, GLib, pyds = _ensure_gst_pyds()

            # OCR runs in a spawn child so pycuda never touches this process.
            ocr_client = ocr_client_from_config(cfg)
            ocr_client.start(ready_timeout=90.0)

            save_cfg = cfg.get("save") or {}
            save_dir: Optional[Path] = None
            if save_cfg.get("enabled", True):
                raw_dir = str(
                    save_cfg.get("dir", "/workspace/output/plates")
                ).strip()
                if raw_dir:
                    save_dir = Path(raw_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)

            ocr_cfg = cfg.get("ocr") or {}
            save_filters = save_cfg
            rt = _Runtime(
                plate_class=int(det.get("plate_class_id", 30)),
                ocr_conf=float(ocr_cfg.get("conf_thresh", 0.60)),
                ocr=ocr_client,
                uploader=uploader,
                source_map={r.source_id: r for r in rows},
                save_dir=save_dir,
                min_box_w=int(save_filters.get("min_box_w", 40)),
                min_box_h=int(save_filters.get("min_box_h", 15)),
                min_det_conf=float(
                    det.get("min_confidence", save_filters.get("min_det_conf", 0.15))
                ),
                ocr_min_interval_s=float(ocr_cfg.get("min_interval_s", 0.2)),
            )
            launch = build_launch(rows, cfg)

            with self._lock:
                self._Gst = Gst
                self._GLib = GLib
                self._pyds = pyds
                self._cameras = rows
                self._rt = rt
                self._cfg = cfg
                self._thread = threading.Thread(
                    target=self._thread_main,
                    args=(launch,),
                    name="ds-lpr-pipeline",
                    daemon=True,
                )
                self._thread.start()

            # Wait until PLAYING or failure
            if not self._started_event.wait(timeout=60.0):
                self._request_force_stop()
                raise TimeoutError("pipeline failed to reach PLAYING within 60s")
            if self._last_error and not self._running:
                raise RuntimeError(self._last_error)

            if join_timeout > 0:
                self._thread.join(timeout=join_timeout)

            return {"status": "started", "cameras": len(rows)}
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("pipeline start failed")
            self._cleanup_after_failure()
            raise
        finally:
            with self._lock:
                self._starting = False

    def stop(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Send EOS, quit main loop, join worker thread."""
        with self._lock:
            thread = self._thread
            if not self._running and not self._starting and thread is None:
                return {"status": "stopped"}
            self._stop_event.set()
            loop = self._loop
            gst_pipeline = self._gst_pipeline
            GLib = self._GLib
            Gst = self._Gst

        if gst_pipeline is not None and Gst is not None:
            try:
                if GLib is not None:
                    GLib.idle_add(self._idle_send_eos)
                else:
                    gst_pipeline.send_event(Gst.Event.new_eos())
            except Exception as exc:
                logger.warning("EOS send failed: %s", exc)
                self._request_force_stop()

        # Hard deadline: force quit if EOS never arrives
        deadline = time.time() + max(1.0, float(timeout))

        def _force_quit_later():
            while time.time() < deadline:
                if not self._running and (thread is None or not thread.is_alive()):
                    return
                time.sleep(0.2)
            if self._running or (thread is not None and thread.is_alive()):
                logger.warning("stop timeout — forcing main-loop quit")
                self._request_force_stop()

        watchdog = threading.Thread(
            target=_force_quit_later, name="ds-lpr-stop-watchdog", daemon=True
        )
        watchdog.start()

        if thread is not None:
            thread.join(timeout=timeout + 2.0)

        ocr = self._rt.ocr
        if ocr is not None:
            try:
                ocr.stop()
            except Exception as exc:
                logger.warning("OCR worker stop: %s", exc)
            self._rt.ocr = None

        with self._lock:
            self._running = False
            self._thread = None
            self._gst_pipeline = None
            self._loop = None
        return {"status": "stopped"}

    # --- internal ---

    def _idle_send_eos(self) -> bool:
        Gst = self._Gst
        pipe = self._gst_pipeline
        if pipe is not None and Gst is not None:
            pipe.send_event(Gst.Event.new_eos())
        return False

    def _request_force_stop(self) -> None:
        loop = self._loop
        GLib = self._GLib
        pipe = self._gst_pipeline
        Gst = self._Gst
        if loop is not None and GLib is not None:
            try:
                GLib.idle_add(loop.quit)
            except Exception:
                pass
        if pipe is not None and Gst is not None:
            try:
                pipe.set_state(Gst.State.NULL)
            except Exception:
                pass

    def _cleanup_after_failure(self) -> None:
        self._stop_event.set()
        self._request_force_stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        ocr = self._rt.ocr
        if ocr is not None:
            try:
                ocr.stop()
            except Exception:
                pass
            self._rt.ocr = None
        with self._lock:
            self._running = False
            self._starting = False
            self._thread = None
            self._gst_pipeline = None
            self._loop = None
            self._cameras = []

    def _thread_main(self, launch: str) -> None:
        Gst = self._Gst
        GLib = self._GLib
        pyds = self._pyds
        assert Gst and GLib and pyds

        try:
            pipeline = Gst.parse_launch(launch)
            conv = pipeline.get_by_name("conv")
            if conv is None:
                raise RuntimeError("pipeline missing nvvideoconvert name=conv")
            srcpad = conv.get_static_pad("src")
            srcpad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_probe)

            loop = GLib.MainLoop()
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._bus_call, loop)

            with self._lock:
                self._gst_pipeline = pipeline
                self._loop = loop

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                self._last_error = "failed to set PLAYING"
                self._started_event.set()
                pipeline.set_state(Gst.State.NULL)
                return

            with self._lock:
                self._running = True
            self._started_event.set()
            logger.info(
                "pipeline PLAYING cameras=%d plate_class=%d ocr_conf=%.3f (OCR subprocess)",
                len(self._cameras),
                self._rt.plate_class,
                self._rt.ocr_conf,
            )

            try:
                loop.run()
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("main loop error")
            finally:
                try:
                    pipeline.send_event(Gst.Event.new_eos())
                    bus.timed_pop_filtered(
                        1 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
                    )
                except Exception:
                    pass
                pipeline.set_state(Gst.State.NULL)
                logger.info(
                    "pipeline stopped frames=%s plate_dets=%s ocr_attempts=%d ocr_hits=%d",
                    dict(self._rt.frame_counts),
                    self._rt.det_counts.get(self._rt.plate_class, 0),
                    self._rt.ocr_attempts,
                    self._rt.ocr_hits,
                )
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("pipeline thread failed")
            self._started_event.set()
        finally:
            with self._lock:
                self._running = False
                self._gst_pipeline = None
                self._loop = None

    def _bus_call(self, bus, message, loop) -> bool:
        Gst = self._Gst
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("bus EOS")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            self._last_error = f"{err} ({dbg})"
            logger.error("bus ERROR: %s (%s)", err, dbg)
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("bus WARN: %s (%s)", err, dbg)
        return True

    def _enqueue_hit(self, hit: PlateHit) -> None:
        with self._lock:
            self._results.append(hit)

    def _pgie_probe(self, pad, info):
        """Pad probe after RGBA convert: crop plate ROI → OCR → enqueue (+ upload)."""
        pyds = self._pyds
        Gst = self._Gst
        rt = self._rt
        if pyds is None or Gst is None:
            return Gst.PadProbeReturn.OK if Gst else 0

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
            rt.frame_counts[sid] += 1
            cam = rt.source_map.get(sid)
            frame_bgr = None

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                cid = int(obj_meta.class_id)
                rt.det_counts[cid] += 1

                # Make plate boxes obvious on nvdsosd preview (even before OCR text).
                if cid == rt.plate_class:
                    try:
                        rp = obj_meta.rect_params
                        rp.border_width = 3
                        rp.border_color.set(0.0, 1.0, 0.0, 1.0)
                        rp.has_bg_color = 0
                        if not (obj_meta.text_params.display_text or "").strip():
                            obj_meta.text_params.display_text = "plate"
                            obj_meta.text_params.font_params.font_size = 12
                            obj_meta.text_params.font_params.font_name = "Serif"
                            obj_meta.text_params.font_params.font_color.set(
                                0.0, 1.0, 0.0, 1.0
                            )
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
                    except Exception:
                        pass

                det_conf = float(getattr(obj_meta, "confidence", 0.0) or 0.0)
                if (
                    cam is not None
                    and cid == rt.plate_class
                    and rt.ocr is not None
                    and det_conf >= rt.min_det_conf
                    and (time.time() - rt.last_ocr_ts) >= rt.ocr_min_interval_s
                ):
                    rect = obj_meta.rect_params
                    if frame_bgr is None:
                        try:
                            frame_bgr = _surface_to_bgr(
                                pyds, gst_buffer, frame_meta
                            )
                        except Exception as exc:
                            logger.warning("surface map failed: %s", exc)
                            frame_bgr = False
                    if frame_bgr is not None and frame_bgr is not False:
                        self._ocr_one(
                            rt, cam, obj_meta, frame_bgr, rect, sid
                        )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _ocr_one(self, rt: _Runtime, cam: CameraRow, obj_meta, frame_bgr, rect, sid: int) -> None:
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(rect.left))
        y1 = max(0, int(rect.top))
        x2 = min(w, int(rect.left + rect.width))
        y2 = min(h, int(rect.top + rect.height))
        bw, bh = x2 - x1, y2 - y1
        # Tiny boxes are almost always false positives (PC bezels, UI chrome, etc.).
        if bw < rt.min_box_w or bh < rt.min_box_h:
            return

        import numpy as np

        # Contiguous uint8 for multiprocessing pickle + OpenCV/OCR preprocessors.
        crop = np.ascontiguousarray(frame_bgr[y1:y2, x1:x2], dtype=np.uint8)
        rt.ocr_attempts += 1
        rt.last_ocr_ts = time.time()
        try:
            results = rt.ocr.process_images([crop])
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return
        if not results:
            return

        text, conf = results[0][0], float(results[0][1])
        if conf != conf:  # NaN
            conf = 0.0
        text_s = str(text) if text is not None else ""
        logger.debug("OCR cam=%s text=%r conf=%.3f", cam.cam_id, text_s, conf)

        # Only save / upload when OCR confidence ≥ threshold (default 60%).
        if conf < rt.ocr_conf or not text_s or "000000" in text_s:
            return

        try:
            obj_meta.text_params.display_text = text_s
            obj_meta.text_params.font_params.font_size = 12
            obj_meta.text_params.font_params.font_name = "Serif"
            obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            obj_meta.text_params.set_bg_clr = 1
            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
        except Exception:
            pass

        ts = time.time()
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        # Full frame with green box + OCR text — what we save / upload (not bare crop).
        annotated = _annotate_plate(frame_bgr, x1, y1, x2, y2, text_s, conf)

        local_path = None
        if rt.save_dir is not None:
            local_path = _save_annotated_local(
                rt.save_dir, cam.cam_id, annotated, text_s, ts
            )
            if local_path:
                logger.info("saved annotated plate %s", local_path)

        object_path: Optional[str] = None
        if rt.uploader is not None:
            try:
                object_path = rt.uploader.upload_hit(
                    annotated,
                    cam_id=cam.cam_id,
                    bucket=cam.bucket,
                    ocr=text_s,
                    conf=conf,
                    bbox=bbox,
                    ts=ts,
                )
            except Exception as exc:
                logger.warning("upload failed cam=%s: %s", cam.cam_id, exc)

        # Prefer local path in API when MinIO path missing (still useful offline).
        if object_path is None and local_path:
            object_path = local_path

        rt.ocr_hits += 1
        self._enqueue_hit(
            PlateHit(
                cam_id=cam.cam_id,
                ocr=text_s,
                conf=conf,
                bucket=cam.bucket,
                object_path=object_path,
                bbox=bbox,
                ts=ts,
                source_id=sid,
            )
        )


# Process-wide runner for the FastAPI control plane.
pipeline_runner = DsLprPipeline(
    result_maxlen=int(os.environ.get("DS_LPR_RESULT_MAXLEN", "200"))
)
