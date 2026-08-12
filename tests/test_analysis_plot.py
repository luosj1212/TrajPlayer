from __future__ import annotations

import numpy as np

from trajplayer.interaction.models import AnalysisResult
from trajplayer.ui.analysis_plot import heatmap_values_for_range, minmax_decimate


def test_minmax_decimation_preserves_narrow_extrema() -> None:
    x = np.arange(100_000, dtype=np.float64)
    y = np.zeros_like(x)
    y[50_001] = 17.0
    y[72_345] = -9.0
    drawn_x, drawn_y = minmax_decimate(x, y, 800)
    assert drawn_x.size <= 1600
    assert float(drawn_y.max()) == 17.0
    assert float(drawn_y.min()) == -9.0


def test_heatmap_view_uses_only_the_visible_x_range() -> None:
    result = AnalysisResult(
        kind="density_profile",
        x=np.arange(5, dtype=np.float64),
        y=np.arange(15, dtype=np.float64).reshape(5, 3),
        x_unit="frame",
        y_unit="g/cm3",
        source_frames=(0, 5, 1),
        selection_revision=0,
        trajectory_generation=1,
    )
    visible = heatmap_values_for_range(result, (1.0, 3.0))
    np.testing.assert_array_equal(visible, result.y[1:4].T[::-1])
