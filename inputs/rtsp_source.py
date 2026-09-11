"""RTSP network camera source.

No RTSP camera is required for development. Define the camera in
``config/cameras.yaml`` with ``enabled: false`` and the application lists it
as CONFIGURED without ever dialling it. Flip the flag when a real camera is
available and nothing downstream changes.

What this class handles that a bare ``cv2.VideoCapture(url)`` does not:

* a bounded open timeout, so a wrong address fails in seconds rather than
  blocking the worker thread indefinitely
* forced TCP transport, because UDP RTSP loses packets and produces smeared
  frames on anything but a quiet LAN
* a small buffer, so the operator sees the present rather than a growing
  backlog of stale frames
* automatic reconnection with a delay, because network cameras drop
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from inputs.base import SourceInfo, VideoSource, VideoSourceError

logger = logging.getLogger(__name__)


def redact_url(url: str) -> str:
    """Hide credentials before a URL reaches a log file or the GUI."""
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://***:***@{host}{port}{parsed.path}"
        return url
    except ValueError:
        return "rtsp://<unparseable>"


class RtspSource(VideoSource):
    """Read an RTSP stream with reconnection."""

    max_consecutive_failures = 150

    def __init__(
        self,
        url: str,
        source_id: str = "rtsp",
        transport: str = "tcp",
        open_timeout: float = 10.0,
        reconnect: bool = True,
        reconnect_delay: float = 5.0,
        target_fps: Optional[float] = None,
    ) -> None:
        super().__init__(source_id=source_id, target_fps=target_fps)
        self.url = url
        self.transport = transport.lower()
        self.open_timeout = float(open_timeout)
        self.reconnect = reconnect
        self.reconnect_delay = float(reconnect_delay)
        self._cap: Optional[cv2.VideoCapture] = None
        self._reconnects = 0
        self._last_reconnect_attempt = 0.0

    # -- helpers ----------------------------------------------------------

    def _apply_ffmpeg_options(self) -> None:
        """Configure the FFmpeg backend through its environment variable.

        OpenCV exposes no API for RTSP transport, so this is the supported
        route. Set before the capture is constructed.
        """
        timeout_us = int(self.open_timeout * 1_000_000)
        options = [f"stimeout;{timeout_us}"]
        if self.transport in ("tcp", "udp"):
            options.append(f"rtsp_transport;{self.transport}")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(options)

    def _connect(self) -> Optional[cv2.VideoCapture]:
        self._apply_ffmpeg_options()
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            # Keep only the newest frame. Without this the operator watches
            # an ever-growing delay.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - backend dependent
            pass
        return cap

    # -- VideoSource ------------------------------------------------------

    def open(self) -> None:
        safe = redact_url(self.url)
        logger.info("Connecting to %s over %s", safe, self.transport.upper())

        deadline = time.time() + self.open_timeout
        cap = None
        while time.time() < deadline:
            cap = self._connect()
            if cap is not None:
                break
            time.sleep(0.5)

        if cap is None:
            raise VideoSourceError(
                f"Cannot connect to {safe} within {self.open_timeout:.0f}s.\n"
                f"Check the address, the credentials, that the camera is on "
                f"the same network, and that the stream path is correct for "
                f"this camera model."
            )

        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise VideoSourceError(
                f"Connected to {safe} but received no video. The stream path "
                f"may point at an audio-only or unsupported profile."
            )

        self._cap = cap
        self._opened = True
        self._info = SourceInfo(
            kind="rtsp",
            detail=safe,
            width=int(probe.shape[1]),
            height=int(probe.shape[0]),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=0,
            is_live=True,
        )
        logger.info("RTSP stream live: %s (%dx%d)", safe,
                    self._info.width, self._info.height)

    def _grab(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, image = self._cap.read()
        return bool(ok), image

    def _on_read_failure(self) -> bool:
        if not self.reconnect:
            self._last_error = "stream ended and reconnect is disabled"
            return False

        now = time.time()
        if now - self._last_reconnect_attempt < self.reconnect_delay:
            time.sleep(0.1)
            return True

        self._last_reconnect_attempt = now
        self._reconnects += 1
        safe = redact_url(self.url)
        logger.warning("RTSP %s dropped; reconnect attempt %d",
                       safe, self._reconnects)
        self._last_error = f"reconnecting (attempt {self._reconnects})"

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cap = self._connect()
        if cap is not None:
            self._cap = cap
            self._failures = 0
            self._last_error = None
            logger.info("RTSP %s reconnected", safe)
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def reconnect_count(self) -> int:
        return self._reconnects
