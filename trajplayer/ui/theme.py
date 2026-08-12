from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QPalette


_LIGHT_STYLESHEET = """
QMainWindow, QWidget#centralWidget {
    background: #f4f6f8;
    color: #20242a;
    font-size: 10pt;
}
QFrame#topBar {
    background: #ffffff;
    border-bottom: 1px solid #dfe3e8;
}
QFrame#transportBar {
    background: #ffffff;
    border-top: 1px solid #dfe3e8;
}
QFrame#analysisPanel {
    background: #ffffff;
    border-top: 1px solid #dfe3e8;
}
QLabel#analysisResultLabel {
    color: #20242a;
    font-size: 9pt;
    font-weight: 600;
}
QLabel#analysisWarningLabel {
    color: #626b76;
    background: #eef2f6;
    border-left: 3px solid #2f855a;
    border-radius: 3px;
    padding: 6px 8px;
    font-size: 8.5pt;
}
QSplitter#contentSplitter {
    background: #f4f6f8;
}
QSplitter#contentSplitter::handle {
    width: 1px;
    background: #dfe3e8;
}
QScrollArea#inspectorScroll, QScrollArea#inspectorScroll > QWidget > QWidget {
    background: #ffffff;
    border: 0;
}
QFrame#inspectorPanel {
    background: #ffffff;
    border-left: 1px solid #dfe3e8;
}
QLabel#fileLabel {
    color: #20242a;
    font-size: 10pt;
    font-weight: 600;
}
QLabel#infoLabel, QLabel#fpsLabel {
    color: #68717d;
    font-size: 9pt;
}
QLabel#selectionSummaryLabel, QLabel#measurementDraftLabel {
    color: #38414b;
    font-size: 9pt;
    padding: 3px 0;
}
QLabel#frameLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QLabel#controlLabel {
    color: #59636f;
    font-size: 9pt;
}
QLabel#controlLabel:disabled, QLabel#sizeValueLabel:disabled {
    color: #9ca3ab;
}
QLabel#sectionLabel {
    color: #20242a;
    font-size: 9pt;
    font-weight: 600;
    padding-top: 2px;
}
QLabel#sizeValueLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QPushButton, QToolButton {
    min-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #ffffff;
    color: #20242a;
    padding: 0 10px;
}
QPushButton:hover, QToolButton:hover {
    border-color: #8aa8c7;
    background: #eef4fa;
}
QPushButton:pressed, QToolButton:pressed {
    background: #dfeaf5;
}
QPushButton:disabled, QToolButton:disabled {
    color: #9ca3ab;
    border-color: #e3e6ea;
    background: #f7f8f9;
}
QPushButton#openButton {
    background: #1769aa;
    border-color: #1769aa;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#openButton:hover {
    background: #125b94;
    border-color: #125b94;
}
QToolButton#playButton {
    min-width: 38px;
    min-height: 38px;
    background: #1769aa;
    border-color: #1769aa;
}
QToolButton#playButton:hover {
    background: #125b94;
    border-color: #125b94;
}
QToolButton#inspectorToggleButton {
    min-width: 32px;
    max-width: 32px;
    padding: 0;
}
QToolButton#recentButton {
    min-width: 32px;
    padding: 0 7px;
}
QLabel#dropFeedbackLabel {
    margin: 28px;
    border: 2px dashed #6f9bc5;
    border-radius: 6px;
    background: rgba(255, 255, 255, 225);
    color: #20242a;
    font-size: 12pt;
    font-weight: 600;
    padding: 24px;
}
QToolButton#inspectorToggleButton:checked {
    background: #e7f0f8;
    border-color: #8aa8c7;
}
QToolButton#advancedToggle {
    min-height: 26px;
    border: 0;
    border-radius: 3px;
    background: transparent;
    color: #4f5965;
    padding: 0 2px;
    text-align: left;
}
QToolButton#advancedToggle:hover {
    background: #eef2f6;
    color: #20242a;
}
QFrame#advancedContent {
    background: transparent;
    border: 0;
}
QCheckBox {
    color: #38414b;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
}
QComboBox {
    min-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #ffffff;
    color: #20242a;
    padding: 0 7px;
}
QComboBox:hover {
    border-color: #8aa8c7;
}
QComboBox:disabled {
    color: #9ca3ab;
    border-color: #e3e6ea;
    background: #f7f8f9;
}
QLineEdit#chainSelectionEdit {
    min-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #ffffff;
    color: #20242a;
    padding: 0 8px;
    selection-background-color: #1769aa;
    selection-color: #ffffff;
}
QLineEdit#chainSelectionEdit:hover, QLineEdit#chainSelectionEdit:focus {
    border-color: #6f9bc5;
}
QLineEdit#chainSelectionEdit[invalid="true"] {
    border-color: #c43d4b;
    background: #fff7f8;
}
QFrame#filterModeSegment {
    min-height: 30px;
    max-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #edf1f5;
}
QToolButton#filterModeButton {
    min-height: 28px;
    max-height: 28px;
    border: 0;
    border-radius: 3px;
    background: transparent;
    color: #4f5965;
    padding: 0 7px;
}
QToolButton#filterModeButton:hover {
    background: #e2e9f0;
    color: #20242a;
}
QToolButton#filterModeButton:checked {
    background: #1769aa;
    color: #ffffff;
    font-weight: 600;
}
QToolButton#filterModeButton:disabled {
    background: transparent;
    color: #a5acb4;
}
QLabel#filterValueLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QLabel#filterValueLabel:disabled {
    color: #9ca3ab;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #d7dce2;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #2b6fae;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border: 2px solid #2b6fae;
    border-radius: 7px;
    background: #ffffff;
}
QSlider:disabled::handle:horizontal {
    border-color: #aeb5bd;
}
QSlider#sizeSlider::groove:horizontal {
    height: 3px;
}
QSlider#sizeSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-width: 1px;
    border-radius: 6px;
}
QProgressBar {
    min-height: 3px;
    max-height: 3px;
    border: 0;
    background: #e6e9ed;
}
QProgressBar::chunk {
    background: #2f855a;
}
QStatusBar {
    background: #f7f8fa;
    color: #626b76;
    border-top: 1px solid #dfe3e8;
}
"""


@dataclass(frozen=True)
class ThemePalette:
    theme_id: str
    window_bg: str
    panel_bg: str
    elevated_bg: str
    text: str
    secondary_text: str
    border: str
    accent: str
    accent_hover: str
    disabled: str
    danger: str
    success: str
    selection: str
    plot_grid: str
    viewport_bg: tuple[float, float, float]


LIGHT_PALETTE = ThemePalette(
    theme_id="light",
    window_bg="#f4f6f8",
    panel_bg="#ffffff",
    elevated_bg="#eef2f6",
    text="#20242a",
    secondary_text="#68717d",
    border="#dfe3e8",
    accent="#1769aa",
    accent_hover="#125b94",
    disabled="#9ca3ab",
    danger="#c43d4b",
    success="#2f855a",
    selection="#d9485f",
    plot_grid="#dfe3e8",
    viewport_bg=(1.0, 1.0, 1.0),
)

DARK_PALETTE = ThemePalette(
    theme_id="dark",
    window_bg="#1f2328",
    panel_bg="#292d32",
    elevated_bg="#343a40",
    text="#f1f3f5",
    secondary_text="#adb5bd",
    border="#495057",
    accent="#4c9ad6",
    accent_hover="#6aaddd",
    disabled="#737b84",
    danger="#ff6b6b",
    success="#69b58a",
    selection="#ff8a5b",
    plot_grid="#495057",
    viewport_bg=(0.075, 0.082, 0.09),
)


def resolve_theme(theme_id: str, system_palette: QPalette | None = None) -> ThemePalette:
    normalized = str(theme_id).lower()
    if normalized == "dark":
        return DARK_PALETTE
    if normalized == "light":
        return LIGHT_PALETTE
    if normalized != "system":
        raise ValueError(f"Unsupported theme: {theme_id}")
    palette = QPalette() if system_palette is None else system_palette
    return (
        DARK_PALETTE
        if palette.color(QPalette.ColorRole.Window).lightness() < 128
        else LIGHT_PALETTE
    )


def build_stylesheet(palette: ThemePalette) -> str:
    if palette.theme_id == "light":
        return _LIGHT_STYLESHEET
    replacements = {
        "#f4f6f8": palette.window_bg,
        "#ffffff": palette.panel_bg,
        "#20242a": palette.text,
        "#303740": palette.text,
        "#38414b": palette.text,
        "#59636f": palette.secondary_text,
        "#68717d": palette.secondary_text,
        "#626b76": palette.secondary_text,
        "#4f5965": palette.secondary_text,
        "#dfe3e8": palette.border,
        "#cdd3da": palette.border,
        "#e3e6ea": palette.border,
        "#d7dce2": palette.border,
        "#e6e9ed": palette.border,
        "#9ca3ab": palette.disabled,
        "#a5acb4": palette.disabled,
        "#aeb5bd": palette.disabled,
        "#f7f8f9": palette.elevated_bg,
        "#f7f8fa": palette.elevated_bg,
        "#eef2f6": palette.elevated_bg,
        "#eef4fa": palette.elevated_bg,
        "#edf1f5": palette.elevated_bg,
        "#e2e9f0": palette.elevated_bg,
        "#dfeaf5": palette.elevated_bg,
        "#e7f0f8": palette.elevated_bg,
        "#1769aa": palette.accent,
        "#2b6fae": palette.accent,
        "#125b94": palette.accent_hover,
        "#8aa8c7": palette.accent_hover,
        "#6f9bc5": palette.accent_hover,
        "#c43d4b": palette.danger,
        "#fff7f8": palette.elevated_bg,
        "#2f855a": palette.success,
    }
    stylesheet = _LIGHT_STYLESHEET
    for source, target in replacements.items():
        stylesheet = stylesheet.replace(source, target)
    stylesheet = stylesheet.replace(
        "background: rgba(255, 255, 255, 225);",
        "background: rgba(41, 45, 50, 235);",
    )
    stylesheet += f"""
QPushButton#openButton, QToolButton#filterModeButton:checked {{
    color: {palette.text};
}}
QLineEdit#chainSelectionEdit {{
    selection-color: {palette.text};
}}
"""
    return stylesheet


APP_STYLESHEET = build_stylesheet(LIGHT_PALETTE)
