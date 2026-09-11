"""The camera view widget.

Displays annotated frames handed over by the pipeline worker. Two details
matter for a surveillance display:

* **Aspect ratio is preserved.** Stretching the picture to fill the panel
  would move every object away from where it really is in the scene, which
  makes the zone overlays lie.
* **The incoming array is copied before it becomes a QImage.** ``QImage`` does
  not take ownership of the buffer it wraps, and the worker thread reuses its
  frame buffers. Without the copy the widget occasionally paints a
  half-overwritten frame, or crashes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from app.gui.theme import Palette, ui_font


def ndarray_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR array to a QImage that owns its memory."""
    if frame is None or frame.size == 0:
        return QImage()

    if frame.ndim == 2:
        image = QImage(frame.data, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_Grayscale8)
        return image.copy()

    contiguous = np.ascontiguousarray(frame)
    height, width, channels = contiguous.shape
    if channels == 4:
        image = QImage(contiguous.data, width, height,
                       contiguous.strides[0], QImage.Format_ARGB32)
    else:
        image = QImage(contiguous.data, width, height,
                       contiguous.strides[0], QImage.Format_BGR888)
    return image.copy()


class VideoView(QLabel):
    """A letterboxed video surface with an idle state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(480, 270))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background-color: #05070b; border: 1px solid {Palette.BORDER};"
            f"border-radius: 8px;"
        )
        self.setFont(ui_font(11))
        self._pixmap: Optional[QPixmap] = None
        self._placeholder = "NO SIGNAL"
        self._placeholder_hint = "Select a camera and press Start"
        self.frames_shown = 0

    # -- content ----------------------------------------------------------

    def show_frame(self, frame: np.ndarray) -> None:
        """Display one BGR frame."""
        image = ndarray_to_qimage(frame)
        if image.isNull():
            return
        self._pixmap = QPixmap.fromImage(image)
        self.frames_shown += 1
        self.update()

    def clear_frame(self, message: str = "NO SIGNAL", hint: str = "") -> None:
        self._pixmap = None
        self._placeholder = message
        self._placeholder_hint = hint
        self.update()

    def has_frame(self) -> bool:
        return self._pixmap is not None

    def current_pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.black)

        if self._pixmap is None:
            painter.setPen(Palette.qcolor(Palette.TEXT_DIM))
            painter.setFont(ui_font(15, bold=True, spacing=2.0))
            rect = self.rect().adjusted(0, -14, 0, -14)
            painter.drawText(rect, Qt.AlignCenter, self._placeholder)
            if self._placeholder_hint:
                painter.setFont(ui_font(10))
                painter.drawText(self.rect().adjusted(0, 26, 0, 26),
                                 Qt.AlignCenter, self._placeholder_hint)
            return

        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
