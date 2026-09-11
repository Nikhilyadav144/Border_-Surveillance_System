"""Event-driven evidence capture.

Writes JPEGs only when something happened, never continuously. Three gates
keep the disk usage sane:

1. a minimum threat level, so LOW-scored noise writes nothing
2. a per-track cooldown, so one person in a restricted zone produces a
   handful of images rather than one per frame
3. an optional retention sweep that deletes files older than N days

Layout on disk::

    data/evidence/
      2026-09-04/
        CAM-01_0042_full_143012.jpg
        CAM-01_0042_crop_143012.jpg
        CAM-01_0042_plate_143014.jpg

The date folder makes manual review and retention easy, and the incident id in
the filename means a stray file can always be traced back to its record.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from core.paths import ensure_dir
from database.models import ThreatLevel

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFile:
    """One written image."""

    path: Path
    kind: str            # full_frame | crop | plate
    width: int
    height: int
    size_bytes: int
    ts: float

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "bytes": self.size_bytes,
            "ts": self.ts,
        }


class EvidenceManager:
    """Decide when to capture, then write the files."""

    def __init__(
        self,
        evidence_dir: Path,
        enabled: bool = True,
        min_threat_level: str = "MEDIUM",
        save_full_frame: bool = True,
        save_crop: bool = True,
        crop_padding: float = 0.25,
        jpeg_quality: int = 90,
        cooldown_seconds: float = 10.0,
        retention_days: int = 30,
    ) -> None:
        self.root = Path(evidence_dir)
        self.enabled = bool(enabled)
        self.min_threat_level = str(min_threat_level).upper()
        self.save_full_frame = bool(save_full_frame)
        self.save_crop = bool(save_crop)
        self.crop_padding = float(crop_padding)
        self.jpeg_quality = int(np.clip(jpeg_quality, 50, 100))
        self.cooldown_seconds = float(cooldown_seconds)
        self.retention_days = int(retention_days)

        self._last_capture: dict[str, float] = {}
        self.files_written = 0
        self.bytes_written = 0

        if self.enabled:
            ensure_dir(self.root)

    # -- gating -----------------------------------------------------------

    def should_capture(
        self,
        threat_level: str,
        track_key: str,
        now: float | None = None,
    ) -> bool:
        """Whether an event deserves evidence right now."""
        if not self.enabled:
            return False
        if not ThreatLevel.at_least(threat_level, self.min_threat_level):
            return False
        moment = now if now is not None else time.time()
        last = self._last_capture.get(track_key)
        # A track that has never had evidence captured is not the same as one
        # captured at time zero. Conflating them suppressed the very first
        # capture whenever the clock started near the epoch.
        if last is None:
            return True
        return (moment - last) >= self.cooldown_seconds

    def note_capture(self, track_key: str, now: float | None = None) -> None:
        self._last_capture[track_key] = now if now is not None else time.time()

    # -- writing ----------------------------------------------------------

    def _day_dir(self, ts: float) -> Path:
        folder = self.root / datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return ensure_dir(folder)

    def _write(self, image: np.ndarray, path: Path, ts: float,
               kind: str) -> Optional[EvidenceFile]:
        if image is None or image.size == 0:
            return None
        try:
            ok = cv2.imwrite(
                str(path), image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                logger.error("cv2 refused to write evidence file %s", path)
                return None
            size = path.stat().st_size
        except (OSError, cv2.error) as exc:
            logger.error("Could not write evidence to %s: %s", path, exc)
            return None

        self.files_written += 1
        self.bytes_written += size
        return EvidenceFile(
            path=path,
            kind=kind,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            size_bytes=size,
            ts=ts,
        )

    def capture(
        self,
        frame: np.ndarray,
        camera_id: str,
        incident_id: int | None,
        track_number: int | None,
        box: Optional[Sequence[float]] = None,
        annotated: Optional[np.ndarray] = None,
        ts: float | None = None,
    ) -> list[EvidenceFile]:
        """Write the evidence set for one incident.

        Parameters
        ----------
        frame:
            The clean frame.
        annotated:
            The frame with overlays. Preferred for the full-frame shot,
            because it shows what the system saw and why.
        box:
            Pixel box of the object, used for the close-up crop.
        """
        if not self.enabled:
            return []

        moment = ts if ts is not None else time.time()
        folder = self._day_dir(moment)
        stamp = datetime.fromtimestamp(moment).strftime("%H%M%S")
        incident_tag = f"{incident_id:04d}" if incident_id is not None else "none"
        prefix = f"{camera_id}_{incident_tag}"

        written: list[EvidenceFile] = []

        if self.save_full_frame:
            source = annotated if annotated is not None else frame
            path = folder / f"{prefix}_full_{stamp}.jpg"
            item = self._write(source, path, moment, "full_frame")
            if item:
                written.append(item)

        if self.save_crop and box is not None:
            crop = self.crop_object(frame, box)
            if crop is not None:
                path = folder / f"{prefix}_crop_{stamp}.jpg"
                item = self._write(crop, path, moment, "crop")
                if item:
                    written.append(item)

        if written:
            logger.debug("Captured %d evidence files for incident %s",
                         len(written), incident_id)
        return written

    def capture_plate(
        self,
        plate_image: np.ndarray,
        camera_id: str,
        track_number: int | None,
        ts: float | None = None,
    ) -> Optional[EvidenceFile]:
        """Save a plate crop.

        Written whether or not OCR succeeded: an unreadable plate is exactly
        the case where a human needs the picture.
        """
        if not self.enabled or plate_image is None or plate_image.size == 0:
            return None
        moment = ts if ts is not None else time.time()
        folder = self._day_dir(moment)
        stamp = datetime.fromtimestamp(moment).strftime("%H%M%S_%f")[:-3]
        track_tag = f"{track_number:04d}" if track_number is not None else "none"
        path = folder / f"{camera_id}_{track_tag}_plate_{stamp}.jpg"
        return self._write(plate_image, path, moment, "plate")

    def crop_object(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> Optional[np.ndarray]:
        """Crop around a box with padding, clamped to the frame."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        pad_x = (x2 - x1) * self.crop_padding
        pad_y = (y2 - y1) * self.crop_padding
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_y))
        cx2 = int(min(w, x2 + pad_x))
        cy2 = int(min(h, y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size else None

    # -- retention --------------------------------------------------------

    def cutoff_timestamp(self) -> Optional[float]:
        """Epoch time before which evidence may be deleted."""
        if self.retention_days <= 0:
            return None
        return (datetime.now() - timedelta(days=self.retention_days)).timestamp()

    def purge(self, paths: list[str]) -> int:
        """Delete the given files. Returns how many were removed."""
        removed = 0
        for raw in paths:
            try:
                path = Path(raw)
                if path.is_file():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", raw, exc)
        # Tidy up any day folders left empty.
        try:
            for folder in self.root.iterdir():
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
        except OSError:
            pass
        if removed:
            logger.info("Retention sweep removed %d evidence files", removed)
        return removed

    def usage_bytes(self) -> int:
        """Total bytes currently stored under the evidence root."""
        try:
            return sum(
                p.stat().st_size for p in self.root.rglob("*.jpg") if p.is_file()
            )
        except OSError:
            return 0
