from __future__ import annotations

import math

import numpy as np
import pytest

from fr2dynage.constants import C_LIGHT, KPC_M, MYR_S
from fr2dynage.kinematics import (
    INVALID_DISTANCE, INVALID_SPEED, INVALID_UNCERTAINTY, NUMERICAL_RANGE,
    KinematicMapError, compute_kinematic_age,
)


CONVERSION = KPC_M / (1000.0 * MYR_S)


def flow(distance=1.0, speed=1000.0, **kwargs):
    return compute_kinematic_age(distance, speed, mode="flow", distance_frame="intrinsic", **kwargs)


def test_flow_clock_uses_physical_host_frame_units() -> None:
    result = flow()
    assert result.age_myr.shape == ()
    assert result.age_myr.item() == pytest.approx(0.9777922216807892)
    assert result.error_myr is None
    assert result.valid.item() is True
    assert result.flags.item() == 0


def test_expansion_clock_requires_and_applies_explicit_expansion_law() -> None:
    result = compute_kinematic_age([1.0, 2.0], 1000.0, mode="expansion", expansion_index=0.6,
                                  distance_frame="intrinsic")
    np.testing.assert_allclose(result.age_myr, np.array([1.0, 2.0]) * CONVERSION / 1000.0 * 0.6)
    with pytest.raises(KinematicMapError, match="explicit expansion_index"):
        compute_kinematic_age(1.0, 1000.0, mode="expansion")
    with pytest.raises(KinematicMapError, match="only valid for expansion"):
        flow(expansion_index=1.0)


def test_projection_frames_and_angle_convention() -> None:
    baseline = flow().age_myr.item()
    projected_radius = compute_kinematic_age(1.0, 1000.0, mode="flow", inclination_deg=30.0)
    assert projected_radius.age_myr.item() == pytest.approx(2.0 * baseline)
    projected_speed = compute_kinematic_age(1.0, 1000.0, mode="flow", distance_frame="intrinsic",
                                           speed_frame="projected", inclination_deg=30.0)
    assert projected_speed.age_myr.item() == pytest.approx(0.5 * baseline)
    both_projected = compute_kinematic_age(1.0, 1000.0, mode="flow", distance_frame="projected",
                                         speed_frame="projected", inclination_deg=30.0,
                                         inclination_error_deg=2.0)
    assert both_projected.age_myr.item() == pytest.approx(baseline)
    assert both_projected.error_myr.item() == 0.0
    with pytest.raises(KinematicMapError, match="inclination_deg is required"):
        compute_kinematic_age(1.0, 1000.0, mode="flow")


def test_independent_first_order_uncertainties_match_analytic_derivatives() -> None:
    result = compute_kinematic_age(20.0, 2000.0, mode="expansion", expansion_index=0.6,
        inclination_deg=60.0, distance_error_kpc=2.0, speed_error_km_s=100.0,
        inclination_error_deg=3.0, expansion_index_error=0.03)
    age = 0.6 * 20.0 / 2000.0 * CONVERSION / math.sin(math.radians(60.0))
    expected = age * math.sqrt((2.0 / 20.0)**2 + (100.0 / 2000.0)**2
        + (math.cos(math.radians(60.0)) / math.sin(math.radians(60.0)) * math.radians(3.0))**2
        + (0.03 / 0.6)**2)
    assert result.age_myr.item() == pytest.approx(age)
    assert result.error_myr.item() == pytest.approx(expected)


def test_zero_distance_has_finite_absolute_distance_error() -> None:
    result = flow([0.0, 1.0], distance_error_kpc=0.1, speed_error_km_s=20.0)
    assert result.age_myr[0] == 0.0
    assert result.error_myr[0] == pytest.approx(0.1 * CONVERSION / 1000.0)
    assert np.all(result.valid)


def test_broadcasting_and_invalid_pixels_are_explicit() -> None:
    result = flow([[1.0], [2.0]], [1000.0, -1.0, np.nan], distance_error_kpc=[[0.1], [-0.1]])
    assert result.age_myr.shape == (2, 3)
    assert result.valid.tolist() == [[True, False, False], [False, False, False]]
    assert result.flags[0, 1] & INVALID_SPEED
    assert result.flags[1, 0] & INVALID_UNCERTAINTY
    assert np.isnan(result.age_myr[~result.valid]).all()
    assert np.isnan(result.error_myr[~result.valid]).all()


def test_invalid_distances_and_physical_speed_limit() -> None:
    result = flow([1.0, -1.0, np.inf, 1.0], [C_LIGHT / 1000.0, 1000.0, 1000.0, C_LIGHT / 1000.0 + 1.0])
    assert result.valid.tolist() == [True, False, False, False]
    assert result.flags[1] == INVALID_DISTANCE
    assert result.flags[2] == INVALID_DISTANCE
    assert result.flags[3] == INVALID_SPEED


def test_projected_speed_cannot_deproject_to_a_superluminal_physical_speed() -> None:
    result = compute_kinematic_age([1.0, 1.0], [1000.0, 200000.0], mode="flow",
        distance_frame="intrinsic", speed_frame="projected", inclination_deg=30.0)
    assert result.valid.tolist() == [True, False]
    assert result.flags[1] == INVALID_SPEED
    same_frame = compute_kinematic_age([1.0, 1.0], [1000.0, 200000.0], mode="flow",
        distance_frame="projected", speed_frame="projected", inclination_deg=30.0)
    assert same_frame.valid.tolist() == [True, False]


@pytest.mark.parametrize("kwargs", [
    {"mode": np.array(["flow"])}, {"distance_frame": np.array(["intrinsic"])},
    {"speed_frame": np.array(["intrinsic"])},
])
def test_enumerated_scalar_settings_reject_array_values(kwargs) -> None:
    settings = {"mode": "flow", "distance_frame": "intrinsic", **kwargs}
    with pytest.raises(KinematicMapError):
        compute_kinematic_age(1.0, 1000.0, **settings)


@pytest.mark.parametrize("kwargs", [
    {"inclination_deg": 0}, {"inclination_deg": 91}, {"inclination_deg": True},
    {"inclination_deg": float("nan")}, {"inclination_deg": 5e-324},
    {"inclination_error_deg": 1.0}, {"inclination_deg": 30.0, "inclination_error_deg": -1.0},
    {"inclination_deg": 30.0, "inclination_error_deg": True},
    {"expansion_index": True, "mode": "expansion"},
    {"expansion_index": 0, "mode": "expansion"},
    {"expansion_index": 1.0, "expansion_index_error": -1.0, "mode": "expansion"},
])
def test_invalid_scalar_clock_assumptions(kwargs) -> None:
    settings = {"mode": "flow", **kwargs}
    with pytest.raises(KinematicMapError):
        compute_kinematic_age(1.0, 1000.0, **settings)


@pytest.mark.parametrize("distance, speed", [(True, 1000.0), (1.0, False), (["1.0"], 1000.0),
    ([1+2j], 1000.0), (pytest.param(10**10000, 1000.0, id="oversized-integer")),
    ([[1.0, 2.0], [3.0]], 1000.0), ([], []), ([1.0, 2.0], [100.0, 200.0, 300.0])])
def test_non_numeric_empty_or_incompatible_maps_are_rejected(distance, speed) -> None:
    with pytest.raises(KinematicMapError):
        flow(distance, speed)


def test_errors_cannot_expand_the_science_map_shape() -> None:
    with pytest.raises(KinematicMapError, match="map shape"):
        flow([1.0, 2.0], distance_error_kpc=[[0.1], [0.2]])


@pytest.mark.parametrize("speed", [0.0, -1.0, np.inf, np.nan])
def test_entirely_invalid_maps_do_not_report_success(speed) -> None:
    with pytest.raises(KinematicMapError, match="no valid pixels"):
        flow([1.0, 2.0], speed)


def test_large_finite_ratio_does_not_overflow_intermediate_multiplication() -> None:
    result = flow(1e308, 1e5)
    assert result.valid.item()
    assert result.age_myr.item() == pytest.approx(1e308 / 1e5 * CONVERSION, rel=2e-13)


def test_unrepresentable_age_or_error_is_flagged_instead_of_zero_or_infinity() -> None:
    result = flow([1.0, 5e-324, 1e308], [1000.0, 1e5, 1e-100])
    assert result.valid.tolist() == [True, False, False]
    assert result.flags[1] == NUMERICAL_RANGE
    assert result.flags[2] == NUMERICAL_RANGE
    error_result = flow([1.0, 1.0], 1000.0, speed_error_km_s=[1.0, 5e-324])
    assert error_result.valid.tolist() == [True, False]
    assert error_result.flags[1] == NUMERICAL_RANGE
