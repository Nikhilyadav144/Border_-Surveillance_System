"""Local video file source.

Reads ``.mp4``, ``.avi``, ``.mov`` and ``.mkv`` through OpenCV. Which
containers actually decode depends on the FFmpeg build behind your OpenCV
wheel, so an unreadable file is reported as an error the operator can see
rather than as an empty stream.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from inputs.base import SourceInfo, VideoSource, VideoSourceError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm")


class FileVideoSource(VideoSource):
    """Play a video file, optionally looping it.

    Parameters
    ----------
    path:
        Video file to read.
    loop:
        Restart from the first frame when the file ends. Useful for demos
        where a 20 second clip must drive a longer walkthrough.
    target_fps:
        Throttle playback. ``None`` decodes as fast as possible, which is
        right for batch processing and wrong for a live-looking demo.
    """

    def __init__(
        self,
        path: str | Path,
        source_id: str = "file",
        loop: bool = False,
        target_fps: Optional[float] = None,
    ) -> None:
        super().__init__(source_id=source_id, target_fps=target_fps)
        self.path = Path(path).expanduser()
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._loops_done = 0
        self._epoch = time.time()
        self._media_fps = 25.0

    def analysis_timestamp(self, frame_index: int) -> float:
        """Media time: when this frame happens *in the footage*.

        Anchored to the moment playback started so timestamps still look like
        real clock times in the database and the GUI, but advance at the
        video's own rate rather than the CPU's.
        """
        return self._epoch + (frame_index - 1) / self._media_fps

    def open(self) -> None:
        if not self.path.exists():
            raise VideoSourceError(
                f"Video file not found: {self.path}\n"
                f"Place your test videos in data/input/ and pick one from the "
                f"Cameras screen."
            )
        if not self.path.is_file():
            raise VideoSourceError(f"Not a file: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning(
                "%s has an unusual extension (%s). Trying anyway; supported "
                "types are %s",
                self.path.name, self.path.suffix, ", ".join(SUPPORTED_SUFFIXES),
            )

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(
                f"OpenCV could not open {self.path.name}. The file may be "
                f"corrupt, or your OpenCV build may lack a decoder for this "
                f"codec. Try re-encoding it to H.264 MP4."
            )

        # Confirm the file actually yields a frame before declaring success.
        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise VideoSourceError(
                f"{self.path.name} opened but produced no frames. It is most "
                f"likely truncated or zero-length."
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._cap = cap
        self._opened = True
        self._epoch = time.time()
        # A file with no usable FPS metadata still needs a timeline. 25 is a
        # neutral assumption and is reported so the operator can see it.
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if reported_fps <= 0.1 or reported_fps > 240:
            logger.warning(
                "%s reports an implausible frame rate (%.2f). Assuming 25 fps "
                "for timing; dwell and speed figures depend on this.",
                self.path.name, reported_fps,
            )
            self._media_fps = 25.0
        else:
            self._media_fps = reported_fps

        self._info = SourceInfo(
            kind="video",
            detail=str(self.path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(probe.shape[1]),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(probe.shape[0]),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)),
            is_live=False,
        )
        logger.info(
            "Opened video %s (%dx%d, %.1f fps, %d frames)",
            self.path.name, self._info.width, self._info.height,
            self._info.fps, self._info.frame_count,
        )

    def _grab(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, image = self._cap.read()
        return bool(ok), image

    def _on_read_failure(self) -> bool:
        """End of file. Rewind when looping, otherwise stop."""
        if self.loop and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._loops_done += 1
            self._failures = 0
            logger.debug("%s: looping (%d)", self.source_id, self._loops_done)
            return True
        return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def progress(self) -> float:
        """Playback position 0.0-1.0, or 0.0 when the length is unknown."""
        if self._info.frame_count <= 0:
            return 0.0
        return min(1.0, self._frame_index / self._info.frame_count)
