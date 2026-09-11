"""Reusable presentation widgets.

Small, dumb components. None of them know about the database, the pipeline or
the configuration: they are handed values and they display them. That keeps
the screens readable and makes the whole GUI layer safe to construct in a
test without any backend running.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.theme import Palette, mono_font, ui_font


class Card(QFrame):
    """A bordered panel. The basic building block of every screen."""

    def __init__(self, title: str = "", parent: QWidget | None = None,
                 flat: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat" if flat else "Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(10)

        self.title_label: Optional[QLabel] = None
        if title:
            self.title_label = QLabel(title.upper())
            self.title_label.setObjectName("SectionTitle")
            self._layout.addWidget(self.title_label)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def set_title(self, text: str) -> None:
        if self.title_label is not None:
            self.title_label.setText(text.upper())


class StatusDot(QWidget):
    """A small filled circle used for online / offline / threat state."""

    def __init__(self, colour: str = Palette.OFFLINE, size: int = 9,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour = QColor(colour)
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def set_colour(self, colour: str) -> None:
        self._colour = QColor(colour)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self._colour)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, self._size, self._size)


class Badge(QLabel):
    """A coloured pill: threat levels, statuses, counts."""

    def __init__(self, text: str = "", colour: str = Palette.TEXT_MUTED,
                 parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_colour(colour)
        self.setFont(ui_font(9, bold=True, spacing=0.6))
        # Wide enough for "CRITICAL" so a level label is never clipped.
        self.setMinimumWidth(76)
        self.setFixedHeight(20)

    def set_colour(self, colour: str) -> None:
        self.setStyleSheet(
            f"background-color: {colour}22; color: {colour};"
            f"border: 1px solid {colour}66; border-radius: 4px;"
            f"padding: 2px 8px;"
        )

    def set_threat(self, level: str) -> None:
        self.setText(str(level).upper())
        self.set_colour(Palette.threat(level))


class KpiCard(Card):
    """A headline number with a label and an optional hint line."""

    clicked = Signal()

    def __init__(
        self,
        label: str,
        value: str = "0",
        hint: str = "",
        accent: str = Palette.ACCENT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._layout.setContentsMargins(15, 13, 15, 13)
        self._layout.setSpacing(3)

        self.label = QLabel(label.upper())
        self.label.setObjectName("KpiLabel")

        self.value = QLabel(value)
        self.value.setObjectName("KpiValue")
        self.value.setStyleSheet(f"color: {accent}; font-size: 27px;"
                                 f"font-weight: 700;")

        self.hint = QLabel(hint)
        self.hint.setObjectName("KpiHint")

        self._layout.addWidget(self.label)
        self._layout.addWidget(self.value)
        self._layout.addWidget(self.hint)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value, hint: str | None = None,
                  accent: str | None = None) -> None:
        self.value.setText(str(value))
        if hint is not None:
            self.hint.setText(hint)
        if accent is not None:
            self.value.setStyleSheet(
                f"color: {accent}; font-size: 27px; font-weight: 700;"
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)


class StatRow(QWidget):
    """A label on the left, a value on the right. Used for detail panels."""

    def __init__(self, label: str, value: str = "-",
                 value_colour: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        self.label = QLabel(label)
        self.label.setObjectName("Muted")
        self.value = QLabel(str(value))
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setFont(mono_font(10))
        if value_colour:
            self.value.setStyleSheet(f"color: {value_colour};")

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.value)

    def set_value(self, value, colour: str | None = None) -> None:
        self.value.setText(str(value))
        if colour:
            self.value.setStyleSheet(f"color: {colour};")


class BarChart(QWidget):
    """A minimal horizontal bar chart.

    Hand-drawn rather than pulled from a charting library: the data here is a
    handful of labelled counts, and a full chart engine would add a dependency
    and a visual style that does not match the rest of the application.
    """

    def __init__(self, parent: QWidget | None = None,
                 accent: str = Palette.ACCENT) -> None:
        super().__init__(parent)
        self._data: list[tuple[str, float, str]] = []
        self._accent = accent
        self.setMinimumHeight(120)

    def set_data(self, data: Iterable[tuple[str, float]] | Iterable[
            tuple[str, float, str]]) -> None:
        normalised: list[tuple[str, float, str]] = []
        for item in data:
            if len(item) == 3:
                normalised.append((str(item[0]), float(item[1]), str(item[2])))
            else:
                normalised.append((str(item[0]), float(item[1]), self._accent))
        self._data = normalised
        self.setMinimumHeight(max(90, 26 * len(normalised) + 16))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(ui_font(9))

        if not self._data:
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data yet")
            return

        peak = max((value for _, value, _ in self._data), default=1.0) or 1.0
        label_w = 132
        value_w = 44
        bar_x = label_w + 8
        bar_w = max(24, self.width() - bar_x - value_w - 8)
        row_h = 26

        for index, (label, value, colour) in enumerate(self._data):
            y = 8 + index * row_h

            painter.setPen(QColor(Palette.TEXT_MUTED))
            text = label if len(label) <= 22 else label[:21] + "…"
            painter.drawText(0, y, label_w, 16, Qt.AlignLeft | Qt.AlignVCenter,
                             text)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(Palette.SURFACE_ALT))
            painter.drawRoundedRect(bar_x, y + 3, bar_w, 11, 5, 5)

            filled = int(bar_w * (value / peak))
            if filled > 0:
                painter.setBrush(QColor(colour))
                painter.drawRoundedRect(bar_x, y + 3, max(4, filled), 11, 5, 5)

            painter.setPen(QColor(Palette.TEXT))
            painter.setFont(mono_font(9))
            painter.drawText(bar_x + bar_w + 6, y, value_w, 16,
                             Qt.AlignLeft | Qt.AlignVCenter,
                             f"{int(value)}")
            painter.setFont(ui_font(9))


class Sparkline(QWidget):
    """A compact line chart for a time series."""

    def __init__(self, parent: QWidget | None = None,
                 colour: str = Palette.ACCENT) -> None:
        super().__init__(parent)
        self._values: list[float] = []
        self._labels: list[str] = []
        self._colour = colour
        self.setMinimumHeight(110)

    def set_data(self, points: Iterable[tuple[str, float]]) -> None:
        items = list(points)
        self._labels = [str(p[0]) for p in items]
        self._values = [float(p[1]) for p in items]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._values:
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.setFont(ui_font(9))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data yet")
            return

        if len(self._values) == 1:
            # One day of history is not a trend, but it is a fact. Show it as
            # a single labelled value rather than pretending there is nothing.
            painter.setPen(QColor(self._colour))
            painter.setFont(ui_font(24, bold=True))
            painter.drawText(self.rect().adjusted(0, -10, 0, -10),
                             Qt.AlignCenter, f"{int(self._values[0])}")
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.setFont(ui_font(9))
            painter.drawText(self.rect().adjusted(0, 26, 0, 26), Qt.AlignCenter,
                             f"on {self._labels[0]} - the only day with data")
            return

        margin_l, margin_r = 30, 8
        margin_t, margin_b = 10, 20
        w = self.width() - margin_l - margin_r
        h = self.height() - margin_t - margin_b
        peak = max(self._values) or 1.0

        # Horizontal guides
        painter.setPen(QPen(QColor(Palette.BORDER), 1, Qt.DotLine))
        painter.setFont(mono_font(8))
        for fraction in (0.0, 0.5, 1.0):
            y = margin_t + h - int(h * fraction)
            painter.drawLine(margin_l, y, margin_l + w, y)
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.drawText(0, y - 7, margin_l - 5, 14,
                             Qt.AlignRight | Qt.AlignVCenter,
                             f"{int(peak * fraction)}")
            painter.setPen(QPen(QColor(Palette.BORDER), 1, Qt.DotLine))

        step = w / max(1, len(self._values) - 1)
        points = [
            (margin_l + i * step,
             margin_t + h - (value / peak) * h)
            for i, value in enumerate(self._values)
        ]

        painter.setPen(QPen(QColor(self._colour), 2))
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]),
                             int(points[i + 1][0]), int(points[i + 1][1]))

        painter.setBrush(QColor(self._colour))
        painter.setPen(Qt.NoPen)
        for x, y in points:
            painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)

        painter.setPen(QColor(Palette.TEXT_DIM))
        painter.setFont(mono_font(8))
        if self._labels:
            painter.drawText(margin_l, self.height() - 14, 90, 12,
                             Qt.AlignLeft, self._labels[0][-5:])
            painter.drawText(margin_l + w - 90, self.height() - 14, 90, 12,
                             Qt.AlignRight, self._labels[-1][-5:])


class Toast(QFrame):
    """A transient alert banner shown over the main window."""

    def __init__(self, parent: QWidget, message: str, level: str = "HIGH",
                 timeout_ms: int = 6000) -> None:
        super().__init__(parent)
        colour = Palette.threat(level)
        self.setStyleSheet(
            f"background-color: {Palette.SURFACE_ALT};"
            f"border: 1px solid {colour}; border-left: 4px solid {colour};"
            f"border-radius: 7px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(11)

        badge = Badge(str(level).upper(), colour)
        text = QLabel(message)
        text.setWordWrap(True)
        text.setStyleSheet("border: none; background: transparent;")

        close = QPushButton("✕")
        close.setFixedSize(22, 22)
        close.setStyleSheet(
            f"border: none; background: transparent; color: {Palette.TEXT_DIM};"
        )
        close.clicked.connect(self.close)

        layout.addWidget(badge)
        layout.addWidget(text, 1)
        layout.addWidget(close)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 170))
        self.setGraphicsEffect(shadow)

        self.setFixedWidth(400)
        self.adjustSize()
        QTimer.singleShot(timeout_ms, self.close)


def divider() -> QFrame:
    """A one-pixel horizontal rule."""
    line = QFrame()
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    return line


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionTitle")
    return label


class ColumnChart(QWidget):
    """Vertical bars, for a fixed set of buckets such as hours of the day.

    A horizontal bar chart with 24 rows does not fit a dashboard panel. This
    draws the same data as columns, which does.
    """

    def __init__(self, parent: QWidget | None = None,
                 colour: str = Palette.ACCENT) -> None:
        super().__init__(parent)
        self._data: list[tuple[str, float]] = []
        self._colour = colour
        self.setMinimumHeight(150)

    def set_data(self, data: Iterable[tuple[str, float]]) -> None:
        self._data = [(str(label), float(value)) for label, value in data]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._data:
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.setFont(ui_font(9))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data yet")
            return

        peak = max((value for _, value in self._data), default=1.0) or 1.0
        margin_l, margin_r = 26, 6
        margin_t, margin_b = 12, 22
        width = self.width() - margin_l - margin_r
        height = self.height() - margin_t - margin_b
        count = len(self._data)
        slot = width / count
        bar_w = max(3.0, slot * 0.66)

        painter.setPen(QPen(QColor(Palette.BORDER), 1, Qt.DotLine))
        painter.setFont(mono_font(8))
        for fraction in (0.0, 0.5, 1.0):
            y = margin_t + height - int(height * fraction)
            painter.drawLine(margin_l, y, margin_l + int(width), y)
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.drawText(0, y - 7, margin_l - 4, 14,
                             Qt.AlignRight | Qt.AlignVCenter,
                             f"{int(peak * fraction)}")
            painter.setPen(QPen(QColor(Palette.BORDER), 1, Qt.DotLine))

        for index, (label, value) in enumerate(self._data):
            x = margin_l + index * slot + (slot - bar_w) / 2
            bar_h = (value / peak) * height
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(Palette.SURFACE_ALT))
            painter.drawRoundedRect(int(x), margin_t, int(bar_w),
                                    int(height), 2, 2)
            if value > 0:
                painter.setBrush(QColor(self._colour))
                painter.drawRoundedRect(
                    int(x), int(margin_t + height - bar_h),
                    int(bar_w), max(2, int(bar_h)), 2, 2,
                )

            if count <= 12 or index % 3 == 0:
                painter.setPen(QColor(Palette.TEXT_DIM))
                painter.setFont(mono_font(7))
                painter.drawText(int(x) - 6, self.height() - 16,
                                 int(bar_w) + 12, 14,
                                 Qt.AlignCenter, label)
