from __future__ import annotations

from uuid import UUID

from trajplayer.timeline import TimelineModel


def test_timeline_range_markers_and_dynamic_frame_count() -> None:
    model = TimelineModel()
    model.reset(100, final=False)
    model.set_range(10, 80)
    marker = model.add_marker(42, "event")
    model.set_analysis_cursor(41)

    assert (model.range_start, model.range_end) == (10, 80)
    assert model.markers[0].frame_index == 42
    assert model.analysis_cursor == 41

    model.set_frame_count(50, final=True)
    assert (model.range_start, model.range_end) == (10, 49)
    assert model.markers[0].frame_index == 42
    model.remove_marker(UUID(str(marker.marker_id)))
    assert model.markers == ()


def test_timeline_full_range_tracks_progressive_indexing() -> None:
    model = TimelineModel()
    model.reset(2, final=False)
    model.set_frame_count(20, final=False)
    assert (model.range_start, model.range_end) == (0, 19)
    model.set_preview_frame(12)
    assert model.preview_frame == 12
    model.set_preview_frame(None)
    assert model.preview_frame is None
