import unittest

from trajplayer.memory_budget import (
    MAX_FRAME_CACHE_BYTES,
    MIB,
    MemoryBudgetManager,
    choose_frame_cache_budget,
)


class MemoryBudgetTests(unittest.TestCase):
    def test_budget_scales_with_available_memory(self) -> None:
        small = choose_frame_cache_budget(
            frame_bytes=12 * MIB,
            frame_count=1_000,
            prefetch_radius=200,
            available_bytes=2 * 1024 * MIB,
        )
        medium = choose_frame_cache_budget(
            frame_bytes=12 * MIB,
            frame_count=1_000,
            prefetch_radius=200,
            available_bytes=8 * 1024 * MIB,
        )
        large = choose_frame_cache_budget(
            frame_bytes=12 * MIB,
            frame_count=1_000,
            prefetch_radius=200,
            available_bytes=32 * 1024 * MIB,
        )

        self.assertEqual(small.bytes, 64 * MIB)
        self.assertEqual(medium.bytes, 192 * MIB)
        self.assertEqual(large.bytes, MAX_FRAME_CACHE_BYTES)

    def test_renderer_and_topology_working_set_reduces_cache_budget(self) -> None:
        unreserved = choose_frame_cache_budget(
            frame_bytes=MIB,
            frame_count=1_000,
            prefetch_radius=200,
            available_bytes=8 * 1024 * MIB,
        )
        reserved = choose_frame_cache_budget(
            frame_bytes=MIB,
            frame_count=1_000,
            prefetch_radius=200,
            reserved_working_set_bytes=96 * MIB,
            available_bytes=8 * 1024 * MIB,
        )

        self.assertEqual(unreserved.bytes, 192 * MIB)
        self.assertEqual(reserved.bytes, 96 * MIB)

    def test_explicit_frame_requirement_never_exceeds_global_cap(self) -> None:
        decision = choose_frame_cache_budget(
            frame_bytes=100 * MIB,
            frame_count=100,
            prefetch_radius=200,
            available_bytes=64 * 1024 * MIB,
        )

        self.assertEqual(decision.bytes, MAX_FRAME_CACHE_BYTES)

    def test_manager_accounts_for_all_viewer_working_sets(self) -> None:
        allocation = MemoryBudgetManager(
            atom_count=1_000_000,
            available_bytes=8 * 1024 * MIB,
        ).allocate(
            frame_bytes=12 * MIB,
            frame_count=1_000,
            prefetch_radius=200,
            persistent_writer_bytes=8 * MIB,
        )

        self.assertEqual(allocation.renderer_bytes, 32_000_000)
        self.assertEqual(allocation.topology_bytes, 16_000_000)
        self.assertEqual(allocation.persistent_writer_bytes, 8 * MIB)
        self.assertEqual(
            allocation.total_bytes,
            allocation.frame_cache.bytes + allocation.reserved_working_set_bytes,
        )


if __name__ == "__main__":
    unittest.main()
