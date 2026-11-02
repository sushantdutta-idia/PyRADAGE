from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

import numpy as np

from fr2dynage.fitting import (
    FitBounds,
    FitError,
    IntersectionSolution,
    _age_solution_for_alpha,
    _fractional_error_sigma_ln,
    _root_log_jet_power,
    _spectrum_records,
    fit_classic,
    fit_joint,
    parametric_bootstrap,
)
from fr2dynage.io import load_configuration
from fr2dynage.model import KDAForwardModel, SpectrumPoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def configured_model():
    observation, assumptions, bounds, _ = load_configuration(ROOT / "examples/cygnus_a.json")
    return KDAForwardModel(observation, assumptions), bounds


@pytest.mark.slow
def test_classic_cygnus_regression(configured_model) -> None:
    model, bounds = configured_model
    result = fit_classic(model, bounds)
    # Table inputs are rounded and the original intersection aggregation is
    # undocumented, so this is intentionally a physical, not bitwise, check.
    assert result.age_myr == pytest.approx(6.78, rel=0.12)
    assert result.alpha_inj == pytest.approx(0.607, abs=0.025)
    assert result.jet_power_w > 0
    assert result.rho0_kg_m3 > 0


@pytest.mark.slow
def test_joint_fit_is_finite(configured_model) -> None:
    model, bounds = configured_model
    result = fit_joint(model, bounds)
    assert result.success
    assert bounds.age_myr[0] <= result.age_myr <= bounds.age_myr[1]
    assert bounds.alpha_inj[0] <= result.alpha_inj <= bounds.alpha_inj[1]
    assert result.jet_power_w > 0


def test_joint_fit_rejects_penalty_only_solution(configured_model) -> None:
    reference, bounds = configured_model

    class ZeroSpectrumModel:
        observation = reference.observation

        @staticmethod
        def spectrum(frequencies, jet_power_w, age_s, alpha_inj):
            return np.zeros(len(frequencies))

    with pytest.raises(FitError, match="no converged solution"):
        fit_joint(ZeroSpectrumModel(), bounds)  # type: ignore[arg-type]


def test_power_root_scan_finds_all_branches_and_selects_lowest() -> None:
    class ThreeRootModel:
        @staticmethod
        def radio_power(frequency_hz, jet_power_w, age_s, alpha_inj):
            log_q = np.log10(jet_power_w)
            log_power = (log_q - 35.0) * (log_q - 36.0) * (log_q - 37.0)
            return 10.0**log_power

    bounds = FitBounds(
        log10_jet_power_w=(34.0, 38.0),
        power_scan_size=17,
    )
    root, count = _root_log_jet_power(
        ThreeRootModel(), 1.0e9, 1.0, 1.0, 0.6, bounds  # type: ignore[arg-type]
    )
    assert root == pytest.approx(35.0)
    assert count == 3


def test_fractional_error_uses_lognormal_coefficient_of_variation() -> None:
    assert _fractional_error_sigma_ln(0.5) == pytest.approx(np.sqrt(np.log1p(0.5**2)))
    assert np.isfinite(_fractional_error_sigma_ln(1.0e155))


def test_uncertainty_propagation_rejects_unsuccessful_fits(configured_model) -> None:
    model, bounds = configured_model

    class Unsuccessful:
        success = False

    def failed_fit(varied_model, varied_bounds):
        return Unsuccessful()

    with pytest.raises(FitError, match="every bootstrap realization failed"):
        parametric_bootstrap(
            model.observation,
            lambda observation: object(),  # type: ignore[arg-type,return-value]
            failed_fit,  # type: ignore[arg-type]
            bounds,
            samples=2,
            seed=7,
        )


def test_power_root_can_be_bracketed_by_zero_emission() -> None:
    class CutoffModel:
        @staticmethod
        def radio_power(frequency_hz, jet_power_w, age_s, alpha_inj):
            return max(0.0, math.log10(jet_power_w) - 34.0)

    root, count = _root_log_jet_power(
        CutoffModel(), 1e9, 0.01, 1.0, 0.6,
        FitBounds(log10_jet_power_w=(34.0, 38.0), power_scan_size=9),
    )
    assert root == pytest.approx(34.01, abs=2e-6)
    assert count == 1


def test_exact_power_root_survives_invalid_neighbor() -> None:
    class IsolatedGridRoot:
        @staticmethod
        def radio_power(frequency_hz, jet_power_w, age_s, alpha_inj):
            return 1.0 if math.log10(jet_power_w) == 34.0 else math.nan

    root, count = _root_log_jet_power(
        IsolatedGridRoot(), 1e9, 1.0, 1.0, 0.6,
        FitBounds(log10_jet_power_w=(34.0, 38.0), power_scan_size=9),
    )
    assert root == 34.0
    assert count == 1


def _intersection(age: float, alpha: float, delta: float = 0.0, log_q: float = 38.0):
    return IntersectionSolution(age, alpha, (log_q,) * 3, (-23.0,) * 3, delta, log_q, -23.0)


def test_classic_age_refinement_keeps_better_grid_solution(monkeypatch) -> None:
    monkeypatch.setattr(
        "fr2dynage.fitting.intersection_at_age",
        lambda model, age, alpha, bounds: (_intersection(age, alpha, abs(math.log10(age) - 1.0)), 1),
    )
    monkeypatch.setattr(
        "fr2dynage.fitting.minimize_scalar", lambda *args, **kwargs: SimpleNamespace(success=True, x=1.1)
    )
    result, _ = _age_solution_for_alpha(object(), 0.6, FitBounds(age_myr=(1.0, 100.0), age_grid_size=9))
    assert result.age_myr == 10.0
    assert result.delta == 0.0


def test_classic_alpha_refinement_keeps_better_grid_solution(monkeypatch) -> None:
    monkeypatch.setattr(
        "fr2dynage.fitting._age_solution_for_alpha",
        lambda model, alpha, bounds: (_intersection(10.0, alpha, log_q=38.0 + 100.0 * (alpha - 0.6) ** 2), {}),
    )
    monkeypatch.setattr(
        "fr2dynage.fitting.minimize_scalar", lambda *args, **kwargs: SimpleNamespace(success=True, x=0.64)
    )
    monkeypatch.setattr("fr2dynage.fitting._spectrum_records", lambda *args: ([], None))
    model = SimpleNamespace(
        observation=SimpleNamespace(name="Grid minimum", spectrum=[SimpleNamespace(frequency_hz=1e9)] * 3),
        derived_quantities=lambda *args: {},
        quadrature_relative_error=lambda *args: 0.0,
    )
    result = fit_classic(model, FitBounds(alpha_grid_size=7))
    assert result.alpha_inj == pytest.approx(0.6)
    assert all(row["log10_rho0_kg_m3"] == -23.0 for row in result.diagnostics["classic_alpha_scan"])


def test_spectrum_records_use_difference_of_logs(configured_model) -> None:
    reference, _ = configured_model
    points = [SpectrumPoint(frequency, 1e-300) for frequency in (100.0, 1000.0, 10000.0)]
    model = SimpleNamespace(observation=replace(reference.observation, spectrum=tuple(points)), radio_power=lambda *args: 1e300)
    records, chi2 = _spectrum_records(model, 1e38, 10.0, 0.6)
    assert [point["log10_residual_model_over_data"] for point in records] == [600.0] * 3
    assert chi2 is None


@pytest.mark.parametrize("power", [0.0, -1.0, float("nan"), float("inf")])
def test_spectrum_records_reject_invalid_predictions(configured_model, power) -> None:
    reference, _ = configured_model
    model = SimpleNamespace(observation=reference.observation, radio_power=lambda *args: power)
    with pytest.raises(FitError, match="finite, positive powers"):
        _spectrum_records(model, 1e38, 10.0, 0.6)


def _successful_draw(**changes):
    result = dict(success=True, age_myr=10.0, alpha_inj=0.6, jet_power_w=1e38, rho0_kg_m3=1e-23)
    result.update(changes)
    return SimpleNamespace(**result)


def test_bootstrap_counts_invalid_draw_without_aborting(configured_model, monkeypatch) -> None:
    reference, bounds = configured_model

    class ExtremeFirstDraw:
        calls = 0
        def normal(self, *args):
            self.calls += 1
            return 10000.0 if self.calls == 1 else 0.0

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ExtremeFirstDraw())
    result = parametric_bootstrap(
        reference.observation, lambda observation: observation,
        lambda *args: _successful_draw(), bounds, samples=2, seed=7,
    )
    assert result["successful_samples"] == 1
    assert result["failed_samples"] == 1
    assert not result["percentile_intervals_valid"]


@pytest.mark.parametrize("age", [0.0, -1.0, float("nan"), 10000.0])
def test_bootstrap_counts_invalid_fitted_ages(configured_model, age) -> None:
    reference, bounds = configured_model
    with pytest.raises(FitError, match="every bootstrap realization failed"):
        parametric_bootstrap(
            reference.observation, lambda observation: observation,
            lambda *args: _successful_draw(age_myr=age), bounds, samples=2, seed=7,
        )


@pytest.mark.parametrize("samples, seed", [(True, 7), (1.5, 7), (0, 7), (1, True), (1, -1), (1, 1.5)])
def test_bootstrap_rejects_invalid_count_and_seed(configured_model, samples, seed) -> None:
    reference, bounds = configured_model
    with pytest.raises(ValueError):
        parametric_bootstrap(reference.observation, lambda observation: observation, lambda *args: _successful_draw(), bounds, samples=samples, seed=seed)


def test_bootstrap_mean_median_and_draw_provenance(configured_model):
    reference, bounds = configured_model
    ages = iter(range(5, 15))
    def fitter(*args):
        draw = _successful_draw()
        draw.age_myr = float(next(ages))
        return draw
    result = parametric_bootstrap(reference.observation, lambda o:o, fitter, bounds, samples=10, seed=13)
    assert result['percentile_intervals_valid']
    assert result['sample_summaries']['age_myr']['mean'] == pytest.approx(9.5)
    assert result['sample_summaries']['age_myr']['median'] == pytest.approx(9.5)
    assert [row[0] for row in result['successful_draws']] == list(range(5,15))
    assert result['sample_parameter_order'][0] == 'age_myr'


@pytest.mark.parametrize("changes", [
    pytest.param({"age_myr": (1, 10**10000)}, id="huge-integer"),
    {"age_myr": (1.0, 1e308)},
    {"log10_jet_power_w": (34.0, 400.0)},
    {"log10_jet_power_w": (-400.0, 43.0)},
])
def test_fit_bounds_reject_unrepresentable_values(changes) -> None:
    with pytest.raises(ValueError):
        FitBounds(**changes)


def test_joint_fit_continues_after_one_start_fails(configured_model, monkeypatch) -> None:
    import fr2dynage.fitting as fitting
    reference, bounds = configured_model
    actual_solver = fitting.least_squares
    calls = 0

    def flaky_solver(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("unusable numerical start")
        return actual_solver(*args, **kwargs)

    monkeypatch.setattr(fitting, "least_squares", flaky_solver)
    result = fit_joint(reference, bounds)
    assert result.success
    assert result.diagnostics["failed_start_count"] == 1
    assert calls == 7
