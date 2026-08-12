from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

from trajplayer.ui.theme import DARK_PALETTE, LIGHT_PALETTE, build_stylesheet, resolve_theme


def test_light_theme_preserves_white_viewport() -> None:
    assert LIGHT_PALETTE.viewport_bg == (1.0, 1.0, 1.0)
    assert "#ffffff" in build_stylesheet(LIGHT_PALETTE)


def test_new_install_defaults_to_the_white_light_theme() -> None:
    source = open("trajplayer/ui/main_window.py", encoding="utf-8").read()
    assert 'self._settings.value("ui/theme", "light")' in source


def test_system_theme_follows_palette_brightness() -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101216"))
    assert resolve_theme("system", palette) is DARK_PALETTE
    palette.setColor(QPalette.ColorRole.Window, QColor("#fafafa"))
    assert resolve_theme("system", palette) is LIGHT_PALETTE


def test_dark_stylesheet_has_dark_semantic_backgrounds() -> None:
    stylesheet = build_stylesheet(DARK_PALETTE)
    assert DARK_PALETTE.window_bg in stylesheet
    assert DARK_PALETTE.panel_bg in stylesheet
    assert DARK_PALETTE.text in stylesheet
    assert "QTabWidget#inspectorTabs" in stylesheet
