import unittest

from PySide6.QtGui import QColor, QImage

from trajplayer.gui_smoke import _validate_framebuffer, framebuffer_metrics


class GuiSmokeTests(unittest.TestCase):
    def test_framebuffer_metrics_find_content_against_white_background(self) -> None:
        image = QImage(20, 20, QImage.Format.Format_RGBA8888)
        image.fill(QColor("white"))
        for x in range(4):
            for y in range(4):
                image.setPixelColor(x + 5, y + 5, QColor("red"))

        metrics = framebuffer_metrics(image)

        self.assertEqual(metrics["non_background_pixels"], 16)
        _validate_framebuffer(metrics)

    def test_uniform_black_framebuffer_is_rejected(self) -> None:
        image = QImage(20, 20, QImage.Format.Format_RGBA8888)
        image.fill(QColor("black"))

        with self.assertRaisesRegex(RuntimeError, "background"):
            _validate_framebuffer(framebuffer_metrics(image))


if __name__ == "__main__":
    unittest.main()
