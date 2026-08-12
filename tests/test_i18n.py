from __future__ import annotations

from PySide6.QtCore import QTranslator

from trajplayer.i18n import translation_file


def test_qtranslator_catalogs_cover_new_analysis_ui() -> None:
    translator = QTranslator()
    assert translator.load(str(translation_file("zh")))
    assert translator.translate("TrajPlayer", "run_analysis") == "运行"
    assert translator.translate("TrajPlayer", "measurement") == "测量"
