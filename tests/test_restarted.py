from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from fr2dynage.constants import C_LIGHT, MYR_S
from fr2dynage.fitting import FitBounds, FitResult
from fr2dynage.io import load_configuration
from fr2dynage.model import (
    Geometry,
    KDAForwardModel,
    ModelAssumptions,
    Observation,
    SpectrumPoint,
)
from fr2dynage.remnant import RemnantFitBounds
from fr2dynage.restarted import (
    RestartedFitError,
    _fit_inner_with_fixed_jet_power,
    _power_law_diagnostic,
    fit_restarted,
)


ROOT = Path(__file__).resolve().parents[1]


def _component(name: str, redshift: float, size_kpc: float, count: int) -> Observation:
    frequencies = (74.0, 151.0, 325.0, 610.0, 1400.0, 3000.0, 6000.0)[:count]
    return Observation.from_points(
        name,
        redshift,
        Geometry(size_kpc, 4.0, 90.0, "two_sided"),
        [
            SpectrumPoint(frequency, 1.0e27 * frequency**-0.7, 0.05)
            for frequency in frequencies
        ],
    )


def _fit_result(*, remnant: bool) -> FitResult:
    if remnant:
        return FitResult(
            source_name="Restarted source",
            method="remnant_joint_log_likelihood",
            success=True,
            age_myr=12.0,
            alpha_inj=0.6,
            jet_power_w=2.0e38,
            rho0_kg_m3=1.0e-23,
            objective=1.0,
            objective_name="chi2_log_power",
            derived={
                "active_duration_myr": 8.0,
                "remnant_duration_myr": 4.0,
            },
            spectrum=[],
            activity_history={"state": "remnant"},
        )
    return FitResult(
        source_name="Restarted source",
        method="joint_log_likelihood",
        success=True,
        age_myr=2.0,
        alpha_inj=0.58,
        jet_power_w=1.0e38,
        rho0_kg_m3=1.0e-24,
        objective=1.0,
        objective_name="chi2_log_power",
        derived={},
        spectrum=[],
    )


def test_restarted_chronology_includes_old_jet_travel_time(monkeypatch) -> None:
    outer = _component("Restarted source", 0.2, 600.0, 5)
    inner = _component("Restarted source", 0.2, 60.0, 4)
    monkeypatch.setattr(
        "fr2dynage.restarted.fit_remnant_joint",
        lambda *args, **kwargs: _fit_result(remnant=True),
    )
    monkeypatch.setattr(
        "fr2dynage.restarted.fit_joint",
        lambda *args, **kwargs: _fit_result(remnant=False),
    )

    result = fit_restarted(
        outer,
        ModelAssumptions(),
        RemnantFitBounds(),
        inner,
        ModelAssumptions(beta=0.0),
        FitBounds(),
        old_jet_speed_fraction_c=0.5,
        allow_underconstrained_inner=True,
    )
    expected_travel = outer.geometry.lobe_length_m / (0.5 * C_LIGHT) / MYR_S
    assert result.chronology["old_jet_travel_time_myr"] == pytest.approx(
        expected_travel
    )
    assert result.chronology["lobe_clock_gap_myr"] == pytest.approx(2.0)
    assert result.chronology["central_engine_quiescent_interval_myr"] == pytest.approx(
        2.0 + expected_travel
    )
    assert result.chronology["jet_power_ratio_new_to_old"] == pytest.approx(0.5)
    assert any("single power law" in warning for warning in result.warnings)


def test_power_law_inner_requires_an_external_constraint() -> None:
    with pytest.raises(RestartedFitError, match="no unique dynamical age"):
        fit_restarted(
            _component("Restarted source", 0.2, 600.0, 5),
            ModelAssumptions(),
            RemnantFitBounds(),
            _component("Restarted source", 0.2, 60.0, 4),
            ModelAssumptions(beta=0.0),
            FitBounds(),
        )


def test_restarted_rejects_mismatched_source_redshifts() -> None:
    with pytest.raises(RestartedFitError, match="same source redshift"):
        fit_restarted(
            _component("Outer", 0.2, 600.0, 5),
            ModelAssumptions(),
            RemnantFitBounds(),
            _component("Inner", 0.3, 60.0, 4),
            ModelAssumptions(beta=0.0),
            FitBounds(),
        )


def test_fixed_jet_power_recovers_synthetic_inner_age() -> None:
    observation, assumptions, bounds, _ = load_configuration(
        ROOT / "examples/restarted_inner_synthetic.json"
    )
    result = _fit_inner_with_fixed_jet_power(
        KDAForwardModel(observation, assumptions),
        bounds,
        1.0e38,
    )
    assert result.method == "joint_fixed_jet_power_log_likelihood"
    assert result.age_myr == pytest.approx(3.0, rel=2.0e-5)
    assert result.alpha_inj == pytest.approx(0.57, rel=2.0e-5)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("power", [-1.0, float("nan"), float("inf"), True, 1.0e50,
    pytest.param(10**10000, id="oversized-integer")])
def test_invalid_fixed_power_is_rejected_before_outer_fit(power, monkeypatch) -> None:
    def unexpected_fit(*args, **kwargs):
        pytest.fail("outer fit must not run for invalid inner power")

    monkeypatch.setattr("fr2dynage.restarted.fit_remnant_joint", unexpected_fit)
    with pytest.raises(RestartedFitError, match="inner jet power"):
        fit_restarted(
            _component("Outer", 0.2, 600.0, 5), ModelAssumptions(), RemnantFitBounds(),
            _component("Inner", 0.2, 60.0, 4), ModelAssumptions(beta=0.0), FitBounds(),
            inner_jet_power_w=power,
        )


def test_restarted_compares_lengths_with_the_lobe_convention(monkeypatch) -> None:
    outer = _component("Outer", 0.2, 200.0, 5)  # one-lobe length is 100 kpc
    inner = _component("Inner", 0.2, 150.0, 4)
    inner = replace(inner, geometry=replace(inner.geometry, mode="one_lobe"))

    def unexpected_fit(*args, **kwargs):
        pytest.fail("an inner lobe longer than the outer lobe must be rejected")

    monkeypatch.setattr("fr2dynage.restarted.fit_remnant_joint", unexpected_fit)
    with pytest.raises(RestartedFitError, match="one-lobe length"):
        fit_restarted(
            outer, ModelAssumptions(), RemnantFitBounds(),
            inner, ModelAssumptions(beta=0.0), FitBounds(),
            allow_underconstrained_inner=True,
        )


def test_power_law_diagnostic_rejects_unrepresentable_chi_square() -> None:
    observation = _component("Inner", 0.2, 60.0, 4)
    observation = replace(observation, spectrum=tuple(
        replace(point, fractional_error=1e-200,
                power_w_hz_sr=point.power_w_hz_sr * (1.1 if index == 0 else 1.0))
        for index, point in enumerate(observation.spectrum)
    ))
    with pytest.raises(RestartedFitError, match="floating-point range"):
        _power_law_diagnostic(observation)
