"""Dark command-centre theme.

One place for every colour and every piece of styling, so the application
reads as one product rather than nine screens that happen to share a window.

The palette is deliberately restrained: a near-black background, two greys for
surfaces, one cyan accent, and four semantic colours reserved exclusively for
threat levels. Reserving colour for meaning is what lets an operator find a
CRITICAL row at a glance.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase


class Palette:
    """Named colours. Never write a hex value anywhere else."""

    # Surfaces, darkest to lightest
    BG = "#0b0e14"
    SURFACE = "#11151f"
    SURFACE_ALT = "#161b27"
    SURFACE_HOVER = "#1d2432"
    BORDER = "#232a3a"
    BORDER_LIGHT = "#2e3648"

    # Text
    TEXT = "#e6ecf5"
    TEXT_MUTED = "#8794ab"
    TEXT_DIM = "#5b6880"

    # Brand
    ACCENT = "#22d3ee"
    ACCENT_DIM = "#0e7490"
    ACCENT_SOFT = "#164e5c"

    # Threat levels. These four are reserved: do not use them decoratively.
    LOW = "#3fb950"
    MEDIUM = "#d29922"
    HIGH = "#f0883e"
    CRITICAL = "#f85149"

    # Status
    ONLINE = "#3fb950"
    OFFLINE = "#5b6880"
    ERROR = "#f85149"
    WARNING = "#d29922"

    # Object classes, matched to the video overlay colours
    PERSON = "#5ADC78"
    VEHICLE = "#FAAA28"
    OTHER = "#c8c8c8"

    @classmethod
    def threat(cls, level: str) -> str:
        return {
            "LOW": cls.LOW, "MEDIUM": cls.MEDIUM,
            "HIGH": cls.HIGH, "CRITICAL": cls.CRITICAL,
        }.get(str(level).upper(), cls.TEXT_MUTED)

    @classmethod
    def status(cls, status: str) -> str:
        return {
            "ONLINE": cls.ONLINE, "OFFLINE": cls.OFFLINE,
            "ERROR": cls.ERROR, "CONNECTING": cls.WARNING,
            "CONFIGURED": cls.TEXT_DIM,
        }.get(str(status).upper(), cls.TEXT_MUTED)

    @classmethod
    def qcolor(cls, value: str) -> QColor:
        return QColor(value)


def mono_font(size: int = 11, bold: bool = False) -> QFont:
    """A monospace face, whatever the platform actually has."""
    for family in ("SF Mono", "Menlo", "JetBrains Mono", "DejaVu Sans Mono",
                   "Consolas", "Monaco", "Courier New"):
        if family in QFontDatabase.families():
            font = QFont(family, size)
            font.setBold(bold)
            return font
    font = QFont()
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(size)
    font.setBold(bold)
    return font


def ui_font(size: int = 11, bold: bool = False, spacing: float = 0.0) -> QFont:
    """The interface face."""
    for family in ("SF Pro Text", "Inter", "Segoe UI", "Helvetica Neue",
                   "DejaVu Sans", "Arial"):
        if family in QFontDatabase.families():
            font = QFont(family, size)
            font.setBold(bold)
            if spacing:
                font.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
            return font
    font = QFont()
    font.setPointSize(size)
    font.setBold(bold)
    if spacing:
        font.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return font


STYLESHEET = f"""
/* ---------------------------------------------------------------- base */
QWidget {{
    background-color: {Palette.BG};
    color: {Palette.TEXT};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background-color: {Palette.BG}; }}

/* -------------------------------------------------------------- panels */
QFrame#Card {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
}}
QFrame#CardFlat {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER};
    border-radius: 6px;
}}
QFrame#Divider {{
    background-color: {Palette.BORDER};
    max-height: 1px;
    border: none;
}}

/* ------------------------------------------------------------- sidebar */
QFrame#Sidebar {{
    background-color: {Palette.SURFACE};
    border-right: 1px solid {Palette.BORDER};
}}
QPushButton#NavButton {{
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    color: {Palette.TEXT_MUTED};
    text-align: left;
    padding: 11px 16px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#NavButton:hover {{
    background-color: {Palette.SURFACE_HOVER};
    color: {Palette.TEXT};
}}
QPushButton#NavButton:checked {{
    background-color: {Palette.SURFACE_ALT};
    border-left: 3px solid {Palette.ACCENT};
    color: {Palette.ACCENT};
    font-weight: 600;
}}

/* --------------------------------------------------------------- title */
QFrame#TitleBar {{
    background-color: {Palette.SURFACE};
    border-bottom: 1px solid {Palette.BORDER};
}}
QLabel#AppTitle {{
    font-size: 17px; font-weight: 700;
    color: {Palette.TEXT}; letter-spacing: 1px;
}}
QLabel#AppSubtitle {{ font-size: 11px; color: {Palette.TEXT_DIM}; }}
QLabel#PageTitle {{
    font-size: 20px; font-weight: 700; color: {Palette.TEXT};
}}
QLabel#PageSubtitle {{ font-size: 12px; color: {Palette.TEXT_MUTED}; }}
QLabel#SectionTitle {{
    font-size: 11px; font-weight: 700; color: {Palette.TEXT_MUTED};
    letter-spacing: 1.2px;
}}
QLabel#Muted   {{ color: {Palette.TEXT_MUTED}; }}
QLabel#Dim     {{ color: {Palette.TEXT_DIM}; font-size: 11px; }}
QLabel#KpiValue {{ font-size: 28px; font-weight: 700; color: {Palette.TEXT}; }}
QLabel#KpiLabel {{
    font-size: 10px; font-weight: 600; color: {Palette.TEXT_MUTED};
    letter-spacing: 1.1px;
}}
QLabel#KpiHint {{ font-size: 10px; color: {Palette.TEXT_DIM}; }}

/* ------------------------------------------------------------- buttons */
QPushButton {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER_LIGHT};
    border-radius: 6px;
    padding: 7px 15px;
    color: {Palette.TEXT};
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {Palette.SURFACE_HOVER};
    border-color: {Palette.ACCENT_DIM};
}}
QPushButton:pressed {{ background-color: {Palette.BORDER}; }}
QPushButton:disabled {{
    color: {Palette.TEXT_DIM};
    background-color: {Palette.SURFACE};
    border-color: {Palette.BORDER};
}}
QPushButton#Primary {{
    background-color: {Palette.ACCENT_DIM};
    border: 1px solid {Palette.ACCENT};
    color: #eafdff; font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: {Palette.ACCENT}; color: #04212a; }}
QPushButton#Danger {{
    background-color: #5c1d1d;
    border: 1px solid {Palette.CRITICAL};
    color: #ffe8e6;
}}
QPushButton#Danger:hover {{ background-color: #7d2626; }}

/* -------------------------------------------------------------- inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER_LIGHT};
    border-radius: 6px;
    padding: 6px 9px;
    color: {Palette.TEXT};
    selection-background-color: {Palette.ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {Palette.ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {Palette.TEXT_MUTED}; margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER_LIGHT};
    selection-background-color: {Palette.ACCENT_DIM};
    outline: none;
}}
QCheckBox {{ spacing: 8px; color: {Palette.TEXT}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {Palette.BORDER_LIGHT};
    background-color: {Palette.SURFACE_ALT};
}}
QCheckBox::indicator:checked {{
    background-color: {Palette.ACCENT}; border-color: {Palette.ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px; background: {Palette.BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {Palette.ACCENT}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {Palette.ACCENT_DIM}; border-radius: 2px;
}}

/* -------------------------------------------------------------- tables */
QTableWidget, QTableView, QTreeWidget, QListWidget {{
    background-color: {Palette.SURFACE};
    alternate-background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    gridline-color: {Palette.BORDER};
    selection-background-color: {Palette.ACCENT_SOFT};
    selection-color: {Palette.TEXT};
    outline: none;
}}
QTableWidget::item, QListWidget::item {{ padding: 5px 7px; border: none; }}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {Palette.ACCENT_SOFT};
}}
QHeaderView::section {{
    background-color: {Palette.SURFACE_ALT};
    color: {Palette.TEXT_MUTED};
    padding: 8px 7px;
    border: none;
    border-bottom: 1px solid {Palette.BORDER_LIGHT};
    border-right: 1px solid {Palette.BORDER};
    font-size: 10px; font-weight: 700; letter-spacing: 0.9px;
}}
QTableCornerButton::section {{
    background-color: {Palette.SURFACE_ALT}; border: none;
}}

/* ------------------------------------------------------------ scrollbar */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {Palette.BORDER_LIGHT}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {Palette.TEXT_DIM}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {Palette.BORDER_LIGHT}; border-radius: 5px; min-width: 28px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ----------------------------------------------------------------- tabs */
QTabWidget::pane {{
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    background-color: {Palette.SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {Palette.TEXT_MUTED};
    padding: 8px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    color: {Palette.ACCENT};
    border-bottom: 2px solid {Palette.ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {Palette.TEXT}; }}

/* --------------------------------------------------------------- misc */
QGroupBox {{
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {Palette.TEXT_MUTED};
    font-size: 11px; letter-spacing: 1px;
}}
QProgressBar {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER};
    border-radius: 5px; height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background-color: {Palette.ACCENT}; border-radius: 4px; }}
QSplitter::handle {{ background-color: {Palette.BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QStatusBar {{
    background-color: {Palette.SURFACE};
    border-top: 1px solid {Palette.BORDER};
    color: {Palette.TEXT_MUTED};
}}
QToolTip {{
    background-color: {Palette.SURFACE_ALT};
    color: {Palette.TEXT};
    border: 1px solid {Palette.BORDER_LIGHT};
    padding: 5px 8px; border-radius: 4px;
}}
QMenu {{
    background-color: {Palette.SURFACE_ALT};
    border: 1px solid {Palette.BORDER_LIGHT};
    border-radius: 6px; padding: 4px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {Palette.ACCENT_SOFT}; }}
QScrollArea {{ border: none; background: transparent; }}
"""


def apply_theme(app) -> None:
    """Install the palette and stylesheet on a QApplication."""
    app.setStyleSheet(STYLESHEET)
    app.setFont(ui_font(11))
