from __future__ import annotations

from pathlib import Path


def translation_file(language: str) -> Path:
    locale = {"en": "en_US", "zh": "zh_CN"}.get(str(language), "en_US")
    return Path(__file__).resolve().parent / f"trajplayer_{locale}.qm"


__all__ = ["translation_file"]
