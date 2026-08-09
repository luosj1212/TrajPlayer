from __future__ import annotations


APP_STYLESHEET = """
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
QLabel#fileLabel {
    color: #20242a;
    font-size: 10pt;
    font-weight: 600;
}
QLabel#infoLabel, QLabel#fpsLabel {
    color: #68717d;
    font-size: 9pt;
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

