import tempfile
import unittest
from pathlib import Path

from trajplayer.memory_budget import (
    MAX_FRAME_CACHE_BYTES,
    MIB,
    MemoryBudgetPolicy,
    MemoryBudgetManager,
    MemorySnapshot,
    _cgroup_memory_available_bytes,
    choose_frame_cache_budget,
)


class MemoryBudgetTests(unittest.TestCase):
    def test_dynamic_policy_shrinks_immediately_under_memory_pressure(self) -> None:
        policy = MemoryBudgetPolicy(frame_bytes=12 * MIB, ceiling_bytes=256 * MIB)
        decision = policy.decide(
            MemorySnapshot(
                available_bytes=256 * MIB,
                process_rss_bytes=512 * MIB,
                cache_hit_rate=0.5,
                decode_latency_ms=2.0,
                decode_mb_s=1000.0,
                playback_fps=60.0,
                interactive=False,
            ),
            current_bytes=256 * MIB,
            now_s=1.0,
        )

        self.assertEqual(decision.target_cache_bytes, 48 * MIB)
        self.assertEqual(decision.reason, "memory-pressure")

    def test_dynamic_policy_requires_sustained_low_hit_rate_before_growth(self) -> None:
        policy = MemoryBudgetPolicy(frame_bytes=MIB, ceiling_bytes=256 * MIB)
        snapshot = MemorySnapshot(
            available_bytes=16 * 1024 * MIB,
            process_rss_bytes=256 * MIB,
            cache_hit_rate=0.2,
            decode_latency_ms=1.0,
            decode_mb_s=500.0,
            playback_fps=60.0,
            interactive=False,
        )

        waiting = policy.decide(snapshot, current_bytes=64 * MIB, now_s=10.0)
        grown = policy.decide(snapshot, current_bytes=64 * MIB, now_s=15.1)

        self.assertEqual(waiting.target_cache_bytes, 64 * MIB)
        self.assertEqual(waiting.reason, "grow-hysteresis")
        self.assertEqual(grown.target_cache_bytes, 96 * MIB)
        self.assertEqual(grown.reason, "low-hit-rate-grow")

    def test_cgroup_v2_available_memory_respects_limit_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text(str(1024 * MIB), encoding="ascii")
            (root / "memory.current").write_text(str(256 * MIB), encoding="ascii")

            self.assertEqual(_cgroup_memory_available_bytes(root), 768 * MIB)

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
