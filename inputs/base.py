"""The common video source interface.

Everything downstream of this file - detection, tracking, zones, incidents -
consumes :class:`Frame` objects and never touches OpenCV capture objects.
That is what makes swapping a local file for a webcam or an RTSP stream a
configuration change rather than a code change.

    VideoSource (abstract)
        |-- FileVideoSource    data/input/*.mp4
        |-- WebcamSource       device index
        `-- RtspSource         rtsp://...

A source is a context manager and an iterator::

    with build_source(config) as src:
        for frame in src:
            ...
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed unrecoverably."""


@dataclass
class Frame:
    """One image plus the metadata the pipeline needs.

    ``ts`` is the **analysis clock**, and which clock that is depends on the
    source. For a live camera it is wall time, because the camera and the wall
    agree. For a video file it is *media time*: the moment this frame depicts,
    derived from the frame index and the file's frame rate.

    That distinction matters more than it looks. Dwell time, speed and
    loitering are all measured against this clock. Using wall time on a file
    would mean a clip analysed at half speed reports everyone loitering twice
    as long as they really did, and a clip analysed faster than real time
    would never detect loitering at all.

    ``wall_ts`` keeps the real arrival time for FPS and latency accounting.
    """

    index: int
    image: np.ndarray
    ts: float
    source_id: str
    wall_ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.wall_ts:
            self.wall_ts = time.time()

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class SourceInfo:
    """What a source can tell us about itself before or during playback."""

    kind: str
    detail: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0          # 0 for live sources, which never end
    is_live: bool = False

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class VideoSource(abc.ABC):
    """Base class for every input type.

    Subclasses implement :meth:`open`, :meth:`_grab` and :meth:`release`.
    Retry policy, throttling, frame counting and health accounting live here
    so every source behaves the same way.
    """

    #: Consecutive failed reads tolerated before the source is declared dead.
    max_consecutive_failures: int = 30

    def __init__(self, source_id: str, target_fps: Optional[float] = None) -> None:
        self.source_id = source_id
        self.target_fps = target_fps
        self._frame_index = 0
        self._opened = False
        self._failures = 0
        self._last_emit = 0.0
        self._last_error: Optional[str] = None
        self._info = SourceInfo(kind="unknown", detail=source_id)

    # -- required of subclasses ------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the underlying stream. Raise :class:`VideoSourceError` on failure."""

    @abc.abstractmethod
    def _grab(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one raw image. ``(False, None)`` means no frame this time."""

    @abc.abstractmethod
    def release(self) -> None:
        """Release the underlying stream. Must be safe to call twice."""

    # -- optional hooks ---------------------------------------------------

    def _on_read_failure(self) -> bool:
        """Called after a failed read.

        Returns True if the caller should try again, False to end the stream.
        The default gives up immediately, which is right for files. Live
        sources override it to reconnect.
        """
        return False

    # -- shared behaviour -------------------------------------------------

    @property
    def info(self) -> SourceInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def frames_read(self) -> int:
        return self._frame_index

    def _throttle(self) -> None:
        """Sleep just enough to respect ``target_fps``.

        Without this, a video file is decoded as fast as the CPU allows and a
        30 second clip flashes past in two seconds. Live sources normally
        leave ``target_fps`` unset because the camera sets the pace.
        """
        if not self.target_fps or self.target_fps <= 0:
            return
        interval = 1.0 / float(self.target_fps)
        elapsed = time.perf_counter() - self._last_emit
        if 0 < elapsed < interval:
            time.sleep(interval - elapsed)

    def analysis_timestamp(self, frame_index: int) -> float:
        """The clock value for a frame. Live sources use wall time.

        File sources override this to return media time, so behaviour
        measured in seconds means seconds of footage, not seconds of CPU.
        """
        return time.time()

    def read(self) -> Optional[Frame]:
        """Return the next :class:`Frame`, or ``None`` when the stream ends."""
        if not self._opened:
            raise VideoSourceError(
                f"{self.source_id}: read() called before open()"
            )

        while True:
            ok, image = self._grab()
            if ok and image is not None and image.size > 0:
                self._failures = 0
                self._throttle()
                self._last_emit = time.perf_counter()
                self._frame_index += 1
                return Frame(
                    index=self._frame_index,
                    image=image,
                    ts=self.analysis_timestamp(self._frame_index),
                    source_id=self.source_id,
                    wall_ts=time.time(),
                )

            self._failures += 1
            if not self._on_read_failure():
                return None
            if self._failures >= self.max_consecutive_failures:
                self._last_error = (
                    f"{self._failures} consecutive read failures; giving up"
                )
                logger.error("%s: %s", self.source_id, self._last_error)
                return None

    # -- protocol ---------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<{type(self).__name__} id={self.source_id!r} "
                f"open={self._opened} frames={self._frame_index}>")
