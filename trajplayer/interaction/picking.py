from __future__ import annotations

from dataclasses import dataclass


MAX_RGBA8_PICK_ID = 0xFFFFFF


@dataclass(frozen=True)
class PickResult:
    atom_index: int
    depth: float
    frame_revision: int
    backend: str

    def __post_init__(self) -> None:
        if int(self.atom_index) < 0:
            raise ValueError("atom_index must be non-negative")
        if self.backend not in {"gpu", "cpu"}:
            raise ValueError(f"Unknown picking backend: {self.backend}")
        object.__setattr__(self, "atom_index", int(self.atom_index))
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "frame_revision", int(self.frame_revision))


def physical_pick_pixel(
    logical_x: float,
    logical_y: float,
    logical_width: int,
    logical_height: int,
    device_pixel_ratio: float,
) -> tuple[int, int, int, int]:
    if logical_width <= 0 or logical_height <= 0:
        raise ValueError("Viewport dimensions must be positive")
    scale = max(float(device_pixel_ratio), 1.0e-6)
    width = max(1, int(float(logical_width) * scale + 0.5))
    height = max(1, int(float(logical_height) * scale + 0.5))
    x = min(width - 1, max(0, int(float(logical_x) * scale)))
    top_y = min(height - 1, max(0, int(float(logical_y) * scale)))
    return x, height - 1 - top_y, width, height


def encode_pick_id_rgba8(atom_index: int) -> tuple[int, int, int, int]:
    pick_id = int(atom_index) + 1
    if pick_id <= 0 or pick_id > MAX_RGBA8_PICK_ID:
        raise ValueError("atom_index is outside the RGBA8 picking range")
    return (
        pick_id & 0xFF,
        (pick_id >> 8) & 0xFF,
        (pick_id >> 16) & 0xFF,
        0xFF,
    )


def decode_pick_id_rgba8(rgba: tuple[int, int, int, int] | list[int]) -> int | None:
    if len(rgba) != 4:
        raise ValueError("RGBA picking values must contain four channels")
    channels = tuple(int(value) for value in rgba)
    if any(value < 0 or value > 0xFF for value in channels):
        raise ValueError("RGBA picking channels must be bytes")
    pick_id = channels[0] | (channels[1] << 8) | (channels[2] << 16)
    return None if pick_id == 0 else pick_id - 1
