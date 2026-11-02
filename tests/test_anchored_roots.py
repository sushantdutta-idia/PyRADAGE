"""Independent numerical checks of the fixed-field luminosity root identity."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fr2dynage.constants import MU_0, MYR_S
from fr2dynage.fitting import FitBounds, FitError, _root_log_jet_power
from fr2dynage.io import load_configuration
from fr2dynage.model import KDAForwardModel, ModelAssumptions


class GeneralScanModel(KDAForwardModel):
    """Same physical forward model, deliberately bypassing the exact-type fast path."""


@pytest.fixture(scope="module")
def observation():
    path = Path(__file__).resolve().parents[1]/"examples/cygnus_a.json"
    return load_configuration(path)[0]


def _threshold(model, age_s):
    state = model.dynamics_state(1e40, age_s)
    ub = (.4*model.assumptions.equipartition_field_ug*1e-10)**2/(2*MU_0)
    return 1e40*ub/(state.pressure_pa/(model.assumptions.gamma_c-1))


@pytest.mark.parametrize("age,alpha,field,frequency,gamma,k_prime", [
    (1., .5, 1., 50e6, 5/3, 0.),
    (10., .62, 3., 1.4e9, 5/3, 0.),
    (100., .8, 20., 1e10, 5/3, 100.),
    (3., .9, 20., 100e6, 4/3, 0.),
    (30., .55, 5., 5e9, 4/3, 100.),
])
def test_anchored_root_recovers_known_power_and_matches_general_scan(
    observation, age, alpha, field, frequency, gamma, k_prime,
):
    assumptions = ModelAssumptions(
        equipartition_field_ug=field, gamma_c=gamma, gamma_b=gamma,
        k_prime=k_prime, integration_order=64,
    )
    fast = KDAForwardModel(observation, assumptions)
    scan = GeneralScanModel(observation, assumptions)
    age_s = age*MYR_S
    # The full scan uses a range with no invalid energy-capacity segment.
    capacity = _threshold(fast, age_s)
    lower_q = max(1e37, capacity*1.05)
    true_q = lower_q*20.
    bounds = FitBounds(log10_jet_power_w=(math.log10(lower_q), math.log10(true_q*20)),
                       power_scan_size=25)
    measured = fast.radio_power(frequency, true_q, age_s, alpha)
    assert measured > 0
    optimized_root, optimized_count = _root_log_jet_power(
        fast, frequency, measured, age_s, alpha, bounds)
    general_root, general_count = _root_log_jet_power(
        scan, frequency, measured, age_s, alpha, bounds)
    assert optimized_root == pytest.approx(math.log10(true_q), abs=2e-9)
    assert general_root == pytest.approx(optimized_root, abs=3e-6)
    assert optimized_count == general_count == 1
    # Test the affine relation separately from the inverse solver's probes.
    q_values = capacity + np.array([.01, .2, 1., 7.])*true_q
    normalized = [fast.radio_power(frequency, q, age_s, alpha)/(q-capacity)
                  for q in q_values]
    np.testing.assert_allclose(normalized, normalized[0], rtol=2e-9)


@pytest.mark.parametrize("side", ["below", "above"])
def test_anchored_power_root_outside_requested_bounds_is_rejected(observation, side):
    model = KDAForwardModel(observation, ModelAssumptions(
        equipartition_field_ug=3., integration_order=32))
    q = 1e38 if side == "below" else 1e41
    observed = model.radio_power(151e6, q, 10*MYR_S, .6)
    with pytest.raises(FitError, match="outside the jet-power bounds"):
        _root_log_jet_power(model, 151e6, observed, 10*MYR_S, .6,
                           FitBounds(log10_jet_power_w=(39., 40.)))


def test_no_physical_power_within_bounds_has_explicit_capacity_error(observation):
    model = KDAForwardModel(observation, ModelAssumptions(
        equipartition_field_ug=1e5, integration_order=32))
    with pytest.raises(FitError, match="cannot support the adopted magnetic-field energy"):
        _root_log_jet_power(model, 151e6, 1e25, 10*MYR_S, .6,
                           FitBounds(log10_jet_power_w=(37., 39.)))


def test_anchored_root_remains_valid_when_lower_bound_cannot_support_field(observation):
    model = KDAForwardModel(observation, ModelAssumptions(
        equipartition_field_ug=20., integration_order=64))
    age = 1*MYR_S
    capacity = _threshold(model, age)
    q = capacity*1.001
    observed = model.radio_power(151e6, q, age, .6)
    root, count = _root_log_jet_power(
        model, 151e6, observed, age, .6,
        FitBounds(log10_jet_power_w=(math.log10(capacity)-2, math.log10(capacity)+2)),
    )
    assert root == pytest.approx(math.log10(q), abs=2e-9)
    assert count == 1


def test_absent_anchor_retains_general_root_scan(observation):
    assumptions = ModelAssumptions(integration_order=64)
    model = KDAForwardModel(observation, assumptions)
    scan = GeneralScanModel(observation, assumptions)
    q, age, alpha, frequency = 1e39, 5*MYR_S, .6, 151e6
    observed = model.radio_power(frequency, q, age, alpha)
    bounds = FitBounds(log10_jet_power_w=(38., 40.))
    actual = _root_log_jet_power(model, frequency, observed, age, alpha, bounds)
    reference = _root_log_jet_power(scan, frequency, observed, age, alpha, bounds)
    assert actual == reference
    assert actual[0] == pytest.approx(39., abs=3e-6)


def test_failed_affine_check_falls_back_to_general_scan(observation, monkeypatch):
    model = KDAForwardModel(observation, ModelAssumptions(
        equipartition_field_ug=3., integration_order=32))
    physical_power = model.radio_power

    def deliberately_nonlinear(frequency, q, age, alpha):
        return physical_power(frequency, q, age, alpha)*(q/1e39)**.2

    monkeypatch.setattr(model, "radio_power", deliberately_nonlinear)
    q, age, alpha, frequency = 1e39, 5*MYR_S, .6, 151e6
    observed = model.radio_power(frequency, q, age, alpha)
    root, count = _root_log_jet_power(
        model, frequency, observed, age, alpha,
        FitBounds(log10_jet_power_w=(38., 40.)),
    )
    assert root == pytest.approx(39., abs=3e-6)
    assert count == 1
