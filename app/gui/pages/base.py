"""Shared page scaffolding."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.common import divider


class Page(QWidget):
    """A screen with a title block and a content area.

    Subclasses build their UI in :meth:`build` and refresh their data in
    :meth:`refresh`. The main window calls ``refresh()`` whenever the page
    becomes visible and on a slow timer, so a page never has to poll.
    """

    title: str = "Page"
    subtitle: str = ""

    def __init__(self, context: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = context

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(12)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("PageTitle")
        self.subtitle_label = QLabel(self.subtitle)
        self.subtitle_label.setObjectName("PageSubtitle")
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)

        header.addLayout(titles)
        header.addStretch(1)

        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(8)
        header.addLayout(self.header_actions)

        outer.addLayout(header)
        outer.addWidget(divider())

        self.content = QVBoxLayout()
        self.content.setSpacing(12)
        outer.addLayout(self.content, 1)

        self.build()

    # -- to override -------------------------------------------------------

    def build(self) -> None:
        """Construct the page contents."""

    def refresh(self) -> None:
        """Reload data from the database. Called when the page is shown."""

    # -- helpers -----------------------------------------------------------

    def add_action(self, widget: QWidget) -> QWidget:
        self.header_actions.addWidget(widget)
        return widget

    def set_subtitle(self, text: str) -> None:
        self.subtitle_label.setText(text)

    @staticmethod
    def scrollable(inner: QWidget) -> QScrollArea:
        """Wrap a widget in a vertical scroll area."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setWidget(inner)
        return area
