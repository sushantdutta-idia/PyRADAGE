from __future__ import annotations

import math
import json

import numpy as np
import pytest

from fr2dynage.constants import MYR_S
from fr2dynage.fitting import FitBounds
from fr2dynage.model import (
    Geometry,
    KDAForwardModel,
    ModelAssumptions,
    Observation,
    SpectrumPoint,
)
from fr2dynage.remnant import KDAExtForwardModel, RemnantFitBounds, fit_remnant_joint


FREQUENCIES_MHZ = (74.0, 151.0, 325.0, 610.0, 1400.0, 3000.0, 6000.0)


def _template_observation() -> Observation:
    return Observation.from_points(
        "Synthetic remnant",
        0.1,
        Geometry(300.0, 4.0, 70.0, "two_sided"),
        [SpectrumPoint(frequency, 1.0e25, 0.05) for frequency in FREQUENCIES_MHZ],
    )


def test_kdaext_active_limit_matches_active_kda() -> None:
    observation = _template_observation()
    assumptions = ModelAssumptions(integration_order=64, gamma_b=4.0 / 3.0)
    active = KDAForwardModel(observation, assumptions)
    remnant = KDAExtForwardModel(observation, assumptions)
    age = 8.0 * MYR_S
    jet_power = 1.0e39
    alpha = 0.6

    active_state = active.dynamics_state(jet_power, age)
    remnant_state = remnant.dynamics_state(jet_power, age, age)
    assert remnant_state.lobe_length_m == pytest.approx(
        active_state.lobe_length_m, rel=1.0e-14
    )
    assert remnant_state.pressure_pa == pytest.approx(
        active_state.pressure_pa, rel=1.0e-14
    )
    active_power = active.spectrum(
        [frequency * 1.0e6 for frequency in FREQUENCIES_MHZ],
        jet_power,
        age,
        alpha,
    )
    remnant_power = remnant.spectrum(
        [frequency * 1.0e6 for frequency in FREQUENCIES_MHZ],
        jet_power,
        age,
        age,
        alpha,
    )
    assert remnant_power == pytest.approx(active_power, rel=2.0e-13)


def test_post_shutdown_dynamics_and_energy_accounting() -> None:
    observation = _template_observation()
    model = KDAExtForwardModel(
        observation,
        ModelAssumptions(integration_order=48),
    )
    total_age = 10.0 * MYR_S
    active_duration = 7.0 * MYR_S
    jet_power = 1.0e39
    state = model.dynamics_state(jet_power, total_age, active_duration)

    expected_length_ratio = (total_age / active_duration) ** (
        model.post_shutdown_length_exponent
    )
    expected_pressure_ratio = (total_age / active_duration) ** (
        -model.post_shutdown_pressure_exponent
    )
    assert state.lobe_length_m / state.lobe_length_at_shutdown_m == pytest.approx(
        expected_length_ratio
    )
    assert state.pressure_pa / state.pressure_at_shutdown_pa == pytest.approx(
        expected_pressure_ratio
    )
    assert state.lobe_length_m == pytest.approx(
        observation.geometry.lobe_length_m, rel=2.0e-14
    )

    derived = model.derived_quantities(
        jet_power,
        total_age,
        active_duration,
        0.6,
    )
    assert derived["one_jet_q_times_active_duration_j"] == pytest.approx(
        jet_power * active_duration
    )
    assert derived["injected_jet_energy_j"] == pytest.approx(
        observation.geometry.emitting_lobes * jet_power * active_duration
    )


def test_switch_off_fades_and_steepens_the_spectrum() -> None:
    observation = _template_observation()
    model = KDAExtForwardModel(
        observation,
        ModelAssumptions(integration_order=64, gamma_b=4.0 / 3.0),
    )
    frequencies = np.asarray([151.0e6, 6000.0e6])
    active_duration = 7.0 * MYR_S
    active = model.spectrum(
        frequencies,
        1.0e39,
        active_duration,
        active_duration,
        0.6,
    )
    faded = model.spectrum(
        frequencies,
        1.0e39,
        10.0 * MYR_S,
        active_duration,
        0.6,
    )
    assert np.all(faded < active)
    assert faded[1] / active[1] < faded[0] / active[0]


@pytest.mark.slow
def test_synthetic_remnant_fit_recovers_both_phase_ages() -> None:
    template = _template_observation()
    assumptions = ModelAssumptions(integration_order=48, gamma_b=4.0 / 3.0)
    generator = KDAExtForwardModel(template, assumptions)
    powers = generator.spectrum(
        [frequency * 1.0e6 for frequency in FREQUENCIES_MHZ],
        1.0e39,
        8.0 * MYR_S,
        7.0 * MYR_S,
        0.6,
    )
    observation = Observation.from_points(
        template.name,
        template.redshift,
        template.geometry,
        [
            SpectrumPoint(frequency, float(power), 0.05)
            for frequency, power in zip(FREQUENCIES_MHZ, powers, strict=True)
        ],
    )
    result = fit_remnant_joint(
        observation,
        assumptions,
        FitBounds(
            age_myr=(5.0, 12.0),
            alpha_inj=(0.5, 0.7),
            log10_jet_power_w=(38.5, 39.5),
            age_grid_size=9,
            alpha_grid_size=7,
            power_scan_size=7,
        ),
        active_fraction_bounds=(0.7, 0.98),
    )
    assert result.method == "remnant_joint_log_likelihood"
    assert result.age_myr == pytest.approx(8.0, rel=2.0e-5)
    assert result.derived["active_duration_myr"] == pytest.approx(7.0, rel=2.0e-5)
    assert result.alpha_inj == pytest.approx(0.6, rel=2.0e-5)
    assert math.log10(result.jet_power_w) == pytest.approx(39.0, abs=2.0e-5)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"total_age_myr": (1.0, 10**10000)},
    {"total_age_myr": (1.0, 1e300)},
    {"log10_jet_power_w": (-400.0, 40.0)},
    {"log10_jet_power_w": (34.0, 400.0)},
])
def test_remnant_bounds_reject_unrepresentable_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        RemnantFitBounds(**kwargs)
