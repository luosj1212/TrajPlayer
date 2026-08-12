from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "trajplayer" / "i18n"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from trajplayer.ui.main_window import UI_TEXT

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for language, locale in (("en", "en_US"), ("zh", "zh_CN")):
        root = ET.Element("TS", version="2.1", language=locale)
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = "TrajPlayer"
        for key in sorted(UI_TEXT):
            message = ET.SubElement(context, "message")
            ET.SubElement(message, "source").text = key
            ET.SubElement(message, "translation").text = UI_TEXT[key][language]
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        source_path = OUTPUT / f"trajplayer_{locale}.ts"
        tree.write(source_path, encoding="utf-8", xml_declaration=True)
        subprocess.run(
            ["pyside6-lrelease", str(source_path), "-qm", str(source_path.with_suffix(".qm"))],
            check=True,
        )


if __name__ == "__main__":
    main()
