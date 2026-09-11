"""Build a :class:`VideoSource` from a camera configuration block.

This is the single place that knows how a YAML ``source:`` block maps onto a
concrete source class. Adding a new input type means adding one branch here
and one module beside it; nothing in ``core/`` or ``app/`` needs to change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.paths import resolve
from inputs.base import VideoSource, VideoSourceError
from inputs.rtsp_source import RtspSource, redact_url
from inputs.video_source import FileVideoSource
from inputs.webcam_source import WebcamSource

logger = logging.getLogger(__name__)


def build_source(
    camera: dict[str, Any],
    input_dir: Path | None = None,
) -> VideoSource:
    """Create the source described by a camera entry.

    Parameters
    ----------
    camera:
        One item from ``cameras.yaml``, or an equivalent dict.
    input_dir:
        Where bare filenames are looked up. Normally ``data/input``.

    Raises
    ------
    VideoSourceError
        If the source type is unknown or its required fields are missing.
    """
    camera_id = str(camera.get("id", "CAM"))
    source = camera.get("source") or {}
    kind = str(source.get("type", "")).lower()
    target_fps = source.get("target_fps")

    if kind == "video":
        raw = source.get("path")
        if not raw:
            raise VideoSourceError(f"{camera_id}: video source needs a 'path'")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            base = input_dir or resolve("data/input")
            candidate = base / path
            path = candidate if candidate.exists() else resolve(path)
        return FileVideoSource(
            path=path,
            source_id=camera_id,
            loop=bool(source.get("loop", False)),
            target_fps=target_fps,
        )

    if kind == "webcam":
        return WebcamSource(
            index=int(source.get("index", 0)),
            source_id=camera_id,
            target_fps=target_fps,
            width=source.get("width"),
            height=source.get("height"),
        )

    if kind == "rtsp":
        url = source.get("url")
        if not url:
            raise VideoSourceError(f"{camera_id}: rtsp source needs a 'url'")
        return RtspSource(
            url=str(url),
            source_id=camera_id,
            transport=str(source.get("transport", "tcp")),
            open_timeout=float(source.get("open_timeout", 10)),
            reconnect=bool(source.get("reconnect", True)),
            reconnect_delay=float(source.get("reconnect_delay", 5)),
            target_fps=target_fps,
        )

    raise VideoSourceError(
        f"{camera_id}: unknown source type {kind!r}. "
        f"Expected 'video', 'webcam' or 'rtsp'."
    )


def describe_source(camera: dict[str, Any]) -> str:
    """A short, credential-free description for the GUI and the database."""
    source = camera.get("source") or {}
    kind = str(source.get("type", "")).lower()
    if kind == "video":
        return str(source.get("path", "?"))
    if kind == "webcam":
        return f"Camera {source.get('index', 0)}"
    if kind == "rtsp":
        return redact_url(str(source.get("url", "")))
    return "unknown"
