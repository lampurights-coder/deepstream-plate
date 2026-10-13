"""Out-of-process OCR worker.

DeepStream + in-process pycuda/TensorRT share a CUDA context and SIGSEGV under
uvicorn's threaded GLib loop. Running RobustScanner in a child process keeps
CUDA contexts isolated.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ds_lpr.ocr_proc")

# (req_id, crop_ndarray) → child; None = shutdown
# (req_id, ok, text, conf, err) ← parent
ReqItem = Tuple[int, Any]
RespItem = Tuple[int, bool, str, float, str]


def _worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    engine: str,
    dictionary: str,
    ready_evt: mp.Event,
    license_plate_root: str,
) -> None:
    """Child entry: load TRT OCR once, serve crops until None."""
    try:
        if license_plate_root and license_plate_root not in sys.path:
            sys.path.insert(0, license_plate_root)
        from src.recognizer_gpu import LicensePlateRecognition

        ocr = LicensePlateRecognition(
            model_path=engine, character_dict_path=dictionary, use_gpu=True
        )
        ready_evt.set()
    except Exception as exc:
        # Parent waits on ready_evt with timeout; put a sentinel error via stderr.
        print(f"OCR worker init failed: {exc}", file=sys.stderr, flush=True)
        ready_evt.set()
        resp_q.put((-1, False, "", 0.0, str(exc)))
        return

    try:
        while True:
            item = req_q.get()
            if item is None:
                break
            req_id, crop = item
            try:
                if not isinstance(crop, np.ndarray):
                    raise TypeError(f"crop must be ndarray, got {type(crop)}")
                crop = np.ascontiguousarray(crop, dtype=np.uint8)
                results = ocr.process_images([crop])
                if not results:
                    resp_q.put((req_id, True, "", 0.0, ""))
                    continue
                text, conf = results[0][0], float(results[0][1])
                if conf != conf:  # NaN
                    conf = 0.0
                resp_q.put(
                    (req_id, True, str(text) if text is not None else "", conf, "")
                )
            except Exception as exc:
                resp_q.put((req_id, False, "", 0.0, str(exc)))
    finally:
        # Avoid pycuda abort on interpreter shutdown.
        try:
            import pycuda.driver as cuda

            while True:
                try:
                    cuda.Context.pop()
                except Exception:
                    break
        except Exception:
            pass


@dataclass
class OcrProcessClient:
    """Parent-side handle: start/stop child and run synchronous OCR calls."""

    engine: str
    dictionary: str
    license_plate_root: str = "/workspace/license-plate"
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._req: Optional[mp.Queue] = None
        self._resp: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self._ready: Optional[Any] = None
        self._next_id = 1
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, ready_timeout: float = 60.0) -> None:
        if self.alive:
            return
        if not Path(self.engine).is_file():
            raise FileNotFoundError(f"OCR engine missing: {self.engine}")
        if not Path(self.dictionary).is_file():
            raise FileNotFoundError(f"OCR dict missing: {self.dictionary}")

        self._req = self._ctx.Queue(maxsize=8)
        self._resp = self._ctx.Queue(maxsize=8)
        self._ready = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(
                self._req,
                self._resp,
                self.engine,
                self.dictionary,
                self._ready,
                self.license_plate_root,
            ),
            name="ds-lpr-ocr-worker",
            daemon=True,
        )
        self._proc.start()
        if not self._ready.wait(timeout=ready_timeout):
            self.stop()
            raise TimeoutError("OCR worker failed to become ready")
        if not self._proc.is_alive():
            raise RuntimeError("OCR worker exited during init")
        # Drain optional init-failure sentinel
        try:
            while not self._resp.empty():
                rid, ok, _t, _c, err = self._resp.get_nowait()
                if rid == -1 and not ok:
                    self.stop()
                    raise RuntimeError(f"OCR worker init failed: {err}")
        except Exception:
            if not self.alive:
                raise
        logger.info("OCR worker ready pid=%s", self._proc.pid)

    def stop(self, join_timeout: float = 5.0) -> None:
        proc, req = self._proc, self._req
        self._proc = None
        if req is not None:
            try:
                req.put_nowait(None)
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=join_timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        self._req = None
        self._resp = None
        self._ready = None

    def process_images(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """API-compatible with LicensePlateRecognition.process_images for one crop."""
        if not images:
            return []
        if not self.alive or self._req is None or self._resp is None:
            raise RuntimeError("OCR worker not running")
        crop = images[0]
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            self._req.put((req_id, crop))
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                try:
                    rid, ok, text, conf, err = self._resp.get(timeout=0.2)
                except Exception:
                    if not self.alive:
                        raise RuntimeError("OCR worker died")
                    continue
                if rid != req_id:
                    # Unexpected; re-queue conceptually by ignoring (should not happen
                    # with single-threaded probe callers + lock).
                    continue
                if not ok:
                    raise RuntimeError(err or "OCR worker error")
                return [(text, conf)]
            raise TimeoutError("OCR worker timed out")


def ocr_client_from_config(cfg: dict) -> OcrProcessClient:
    ocfg = cfg.get("ocr") or {}
    engine = ocfg.get("engine", "/workspace/license-plate/models/rb_scaner_fp16.engine")
    dct = ocfg.get("dict", "/workspace/license-plate/dictionary_plate/dict90.txt")
    root = os.environ.get("LICENSE_PLATE_ROOT", "/workspace/license-plate")
    return OcrProcessClient(engine=str(engine), dictionary=str(dct), license_plate_root=root)
