"""Webcam source.

A webcam is a live source: it never ends, so a failed read is a hiccup to
retry rather than an end of stream. On macOS the first open triggers the
system camera-permission prompt, and the first two or three reads often fail
while the sensor warms up, which is why the retry budget is generous.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from inputs.base import SourceInfo, VideoSource, VideoSourceError

logger = logging.getLogger(__name__)


class WebcamSource(VideoSource):
    """Capture from a local camera device index."""

    max_consecutive_failures = 60

    def __init__(
        self,
        index: int = 0,
        source_id: str = "webcam",
        target_fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_seconds: float = 2.0,
    ) -> None:
        super().__init__(source_id=source_id, target_fps=target_fps)
        self.index = int(index)
        self.width = width
        self.height = height
        self.warmup_seconds = warmup_seconds
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(
                f"Cannot open camera index {self.index}.\n"
                f"Check that the device exists, that no other application is "
                f"using it, and on macOS that this application has been granted "
                f"Camera access in System Settings > Privacy & Security."
            )

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        # Warm-up. Early frames from a cold sensor are often black or absent.
        deadline = time.time() + self.warmup_seconds
        probe = None
        while time.time() < deadline:
            ok, probe = cap.read()
            if ok and probe is not None and probe.size > 0:
                break
            time.sleep(0.1)
        else:
            cap.release()
            raise VideoSourceError(
                f"Camera {self.index} opened but produced no frames within "
                f"{self.warmup_seconds:.0f}s. It may be in use by another "
                f"application."
            )

        self._cap = cap
        self._opened = True
        self._info = SourceInfo(
            kind="webcam",
            detail=f"camera index {self.index}",
            width=int(probe.shape[1]) if probe is not None else 0,
            height=int(probe.shape[0]) if probe is not None else 0,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=0,
            is_live=True,
        )
        logger.info("Opened webcam %d (%dx%d)", self.index,
                    self._info.width, self._info.height)

    def _grab(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, image = self._cap.read()
        return bool(ok), image

    def _on_read_failure(self) -> bool:
        """A dropped frame is normal on a live device. Pause briefly, retry."""
        self._last_error = f"dropped frame ({self._failures})"
        time.sleep(0.05)
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
