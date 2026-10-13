"""Per-camera MinIO sink: ensure buckets, upload plate crop JPG + JSON, return path.

Thin wrapper over patterns in ``license-plate/src/minio_uplaoder.py``, shaped for
``pipeline.HitUploader`` (``upload_hit``) and the ds_lpr ``cameras.yaml`` minio block.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger("ds_lpr.minio_sink")

# Safe fragment for object keys (OCR text / cam ids).
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_fragment(text: str, *, fallback: str = "plate", max_len: int = 64) -> str:
    cleaned = _SAFE_RE.sub("_", (text or "").strip()).strip("._-")
    if not cleaned:
        return fallback
    return cleaned[:max_len]


def _encode_jpg(crop: Union[np.ndarray, bytes, bytearray]) -> bytes:
    if isinstance(crop, (bytes, bytearray)):
        return bytes(crop)
    if isinstance(crop, np.ndarray):
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            raise ValueError("cv2.imencode(.jpg) failed")
        return buf.tobytes()
    raise TypeError(f"Unsupported crop type: {type(crop)}")


class MinioHitSink:
    """Upload OCR hits to per-camera MinIO buckets.

    Key layout: ``{cam_id}/{YYYY-MM-DD}/{ts}_{ocr}.jpg`` plus matching ``.json``.
    Returns ``s3://{bucket}/{key}`` (image object).
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        secure: bool = False,
        region: Optional[str] = None,
        path_style: str = "s3",
    ):
        """
        Args:
            endpoint: ``host:port`` (no scheme).
            access_key / secret_key: MinIO credentials.
            secure: HTTPS when True.
            region: optional MinIO region.
            path_style: ``\"s3\"`` → ``s3://bucket/key``; ``\"plain\"`` → ``bucket/key``.
        """
        if not endpoint or not str(endpoint).strip():
            raise ValueError("minio.endpoint is required")
        self.endpoint = str(endpoint).strip()
        self.secure = bool(secure)
        self.path_style = path_style if path_style in ("s3", "plain") else "s3"
        self.client = Minio(
            self.endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=self.secure,
            region=region,
        )
        self._ensured: set[str] = set()

    @classmethod
    def from_config(cls, minio_cfg: Optional[Dict[str, Any]]) -> "MinioHitSink":
        """Build from the ``minio:`` block of cameras.yaml / config.yaml."""
        cfg = dict(minio_cfg or {})
        return cls(
            endpoint=str(cfg.get("endpoint", "127.0.0.1:9000")),
            access_key=str(cfg.get("access_key", "minioadmin")),
            secret_key=str(cfg.get("secret_key", "minioadmin")),
            secure=bool(cfg.get("secure", False)),
            region=cfg.get("region"),
            path_style=str(cfg.get("path_style", "s3")),
        )

    def ensure_buckets(self, buckets: Iterable[str]) -> List[str]:
        """Create each bucket if missing. Idempotent. Returns ensured bucket names."""
        ensured: List[str] = []
        for raw in buckets:
            bucket = str(raw or "").strip()
            if not bucket:
                continue
            if bucket in self._ensured:
                ensured.append(bucket)
                continue
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info("Created MinIO bucket: %s", bucket)
                else:
                    logger.debug("MinIO bucket exists: %s", bucket)
            except S3Error:
                logger.exception("Failed to ensure MinIO bucket %s", bucket)
                raise
            self._ensured.add(bucket)
            ensured.append(bucket)
        return ensured

    def ensure_camera_buckets(self, cameras: Sequence[Dict[str, Any]]) -> List[str]:
        """Ensure ``bucket`` for each camera mapping (``id`` / ``url`` / ``bucket``)."""
        names = []
        for cam in cameras:
            if isinstance(cam, dict):
                names.append(str(cam.get("bucket", "")).strip())
        return self.ensure_buckets(names)

    def _object_path(self, bucket: str, key: str) -> str:
        if self.path_style == "plain":
            return f"{bucket}/{key}"
        return f"s3://{bucket}/{key}"

    def _make_keys(self, cam_id: str, ocr: str, ts: float) -> tuple[str, str]:
        when = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        date = when.strftime("%Y-%m-%d")
        ts_frag = when.strftime("%Y%m%d_%H%M%S_%f")
        ocr_frag = _safe_fragment(ocr, fallback="plate")
        cam_frag = _safe_fragment(str(cam_id), fallback="cam")
        base = f"{cam_frag}/{date}/{ts_frag}_{ocr_frag}"
        return f"{base}.jpg", f"{base}.json"

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
        """Upload plate crop + JSON metadata; return ``s3://bucket/key`` (or plain).

        JSON body: ``ocr``, ``conf``, ``cam_id``, ``bbox``, ``ts``.
        """
        bucket = str(bucket or "").strip()
        if not bucket:
            raise ValueError("bucket is required")
        self.ensure_buckets([bucket])

        jpg_key, json_key = self._make_keys(cam_id, ocr, ts)
        jpg_bytes = _encode_jpg(crop)
        meta = {
            "ocr": ocr,
            "conf": float(conf),
            "cam_id": cam_id,
            "bbox": [float(x) for x in bbox],
            "ts": float(ts),
        }
        meta_bytes = json.dumps(meta, default=str).encode("utf-8")

        try:
            self.client.put_object(
                bucket,
                jpg_key,
                io.BytesIO(jpg_bytes),
                length=len(jpg_bytes),
                content_type="image/jpeg",
            )
            self.client.put_object(
                bucket,
                json_key,
                io.BytesIO(meta_bytes),
                length=len(meta_bytes),
                content_type="application/json",
            )
        except S3Error:
            logger.exception(
                "S3Error uploading hit cam=%s bucket=%s key=%s", cam_id, bucket, jpg_key
            )
            raise

        path = self._object_path(bucket, jpg_key)
        logger.debug("Uploaded OCR hit %s", path)
        return path
