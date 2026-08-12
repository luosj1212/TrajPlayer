from __future__ import annotations

import numpy as np

from trajplayer.analysis.alignment import align_positions, rmsd
from trajplayer.analysis.density import AMU_PER_ANGSTROM3_TO_G_CM3, bulk_density, density_profile
from trajplayer.analysis.geometry import angle, center_of_mass, dihedral, distance, radius_of_gyration
from trajplayer.analysis.msd import (
    bounded_fft_atom_chunk,
    msd_from_origin,
    windowed_msd_direct,
    windowed_msd_fft,
)
from trajplayer.analysis.pbc import (
    cartesian_to_fractional,
    make_whole_relative_to_anchor,
    minimum_image_displacement,
)
from trajplayer.analysis.rms import rmsf
from trajplayer.interaction.measurements import Measurement, MeasurementKind, evaluate_measurement


TRICLINIC_CELL = np.array(
    [[4.0, 0.0, 0.0], [1.1, 3.5, 0.0], [0.4, 0.7, 5.0]],
    dtype=np.float64,
)


def test_triclinic_minimum_image_uses_fractional_coordinates() -> None:
    fractional_delta = np.array([0.9, -0.8, 0.6])
    cartesian = fractional_delta @ TRICLINIC_CELL
    wrapped = minimum_image_displacement(cartesian, TRICLINIC_CELL)
    expected = np.array([-0.1, 0.2, -0.4]) @ TRICLINIC_CELL
    np.testing.assert_allclose(wrapped, expected, atol=1.0e-12)


def test_measurement_kernels_and_manager_path_share_values() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    )
    assert distance(positions[0], positions[1]) == 1.0
    assert abs(angle(positions[0], positions[1], positions[2]) - 90.0) < 1.0e-12
    assert abs(abs(dihedral(*positions)) - 90.0) < 1.0e-12

    measurement = Measurement(MeasurementKind.ANGLE, (0, 1, 2))
    value = evaluate_measurement(measurement, positions, None)
    assert value.value == angle(positions[0], positions[1], positions[2])
    assert value.unit == "deg"


def test_measurement_minimum_image_crosses_box() -> None:
    cell = np.diag([10.0, 10.0, 10.0])
    assert abs(distance(np.array([9.8, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]), cell) - 0.4) < 1.0e-12


def test_density_bulk_and_profile_conserve_particle_count() -> None:
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.1, 0.0, 0.0], [2.5, 0.0, 0.0], [7.5, 0.0, 0.0]])
    number = bulk_density([cell], 3.0, mass_density=False)
    np.testing.assert_allclose(number, [0.003])
    mass = bulk_density([cell], 3.0, mass_density=True)
    np.testing.assert_allclose(mass, [0.003 * AMU_PER_ANGSTROM3_TO_G_CM3])

    profile = density_profile(positions, cell, axis=0, bins=5)
    slice_volume = 1000.0 / 5
    assert abs(float(profile.sum()) * slice_volume - 3.0) < 1.0e-12


def test_kabsch_removes_arbitrary_rigid_transform_without_reflection() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.2, 1.3, 0.4], [0.1, -0.4, 1.1]]
    )
    angle_radians = np.radians(37.0)
    rotation = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0.0],
            [np.sin(angle_radians), np.cos(angle_radians), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = reference @ rotation + np.array([4.0, -2.0, 0.7])
    assert rmsd(mobile, reference) < 1.0e-12
    np.testing.assert_allclose(align_positions(mobile, reference), reference, atol=1.0e-12)


def test_rmsf_static_and_symmetric_motion() -> None:
    static = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rmsf([static, static, static]), 0.0)
    moving = [np.array([[value, 0.0, 0.0]]) for value in (-1.0, 1.0)]
    np.testing.assert_allclose(rmsf(moving), [1.0])


def test_msd_linear_motion_and_pbc_unwrap() -> None:
    frames = []
    for step in range(5):
        positions = np.array([[float(step), 2.0 * step, 0.0], [float(step), 2.0 * step, 0.0]])
        frames.append((positions, None))
    np.testing.assert_allclose(
        msd_from_origin(frames, dimensions="xy", unwrap_pbc=False),
        5.0 * np.arange(5, dtype=np.float64) ** 2,
    )

    cell = np.diag([10.0, 10.0, 10.0])
    wrapped = [9.0, 0.5, 2.0, 3.5]
    pbc_frames = [(np.array([[x, 0.0, 0.0]]), cell) for x in wrapped]
    np.testing.assert_allclose(
        msd_from_origin(pbc_frames, dimensions="x", unwrap_pbc=True),
        np.array([0.0, 1.5, 3.0, 4.5]) ** 2,
    )


def test_windowed_fft_msd_matches_direct_reference() -> None:
    rng = np.random.default_rng(20260812)
    positions = np.cumsum(rng.normal(size=(32, 7, 3)), axis=0)
    direct = windowed_msd_direct(positions, dimensions="xyz", max_lag=16)
    fast = windowed_msd_fft(positions, dimensions="xyz", max_lag=16, atom_chunk=3)
    np.testing.assert_allclose(fast, direct, rtol=1.0e-11, atol=1.0e-11)


def test_windowed_fft_keeps_long_trajectory_chunks_bounded(tmp_path) -> None:
    assert bounded_fft_atom_chunk(100_000, "xyz", 4096) < 16
    rng = np.random.default_rng(17)
    source = np.cumsum(rng.normal(size=(40, 9, 3)), axis=0).astype(np.float32)
    mapped = np.memmap(
        tmp_path / "msd.bin",
        mode="w+",
        dtype=np.float32,
        shape=source.shape,
    )
    mapped[:] = source
    mapped.flush()
    expected = windowed_msd_direct(mapped, max_lag=12)
    actual = windowed_msd_fft(mapped, max_lag=12, atom_chunk=4096)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_com_rg_and_make_whole_are_physically_consistent() -> None:
    cell = np.diag([10.0, 10.0, 10.0])
    wrapped = np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]])
    whole = make_whole_relative_to_anchor(wrapped, cell)
    fractional = cartesian_to_fractional(whole, cell)
    assert abs(float(fractional[1, 0] - fractional[0, 0]) - 0.04) < 1.0e-12
    np.testing.assert_allclose(center_of_mass(whole), [10.0, 0.0, 0.0])
    assert abs(radius_of_gyration(whole) - 0.2) < 1.0e-12
