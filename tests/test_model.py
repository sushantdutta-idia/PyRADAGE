from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
import math

import pytest

from fr2dynage.constants import KPC_M, MYR_S
from fr2dynage.model import (
    Geometry,
    KDAForwardModel,
    ModelDomainError,
    ModelAssumptions,
    Observation,
    SpectrumPoint,
    electron_energy_integral,
)


def cygnus_model(order: int = 128) -> KDAForwardModel:
    observation = Observation.from_points(
        "Cygnus A",
        0.0564,
        Geometry(135.0, 3.9, 80.0, "two_sided"),
        [
            SpectrumPoint(151.0, 10**27.796),
            SpectrumPoint(408.0, 10**27.465),
            SpectrumPoint(1400.0, 10**26.963),
            SpectrumPoint(4800.0, 10**26.358),
        ],
    )
    return KDAForwardModel(
        observation,
        ModelAssumptions(gamma_b=5 / 3, integration_order=order),
    )


def test_cygnus_geometry_and_coefficients() -> None:
    model = cygnus_model()
    assert model.observation.geometry.lobe_axial_ratio == pytest.approx(1.95)
    assert model.observation.geometry.lobe_length_m / KPC_M == pytest.approx(68.5412963)
    assert model.pressure_ratio == pytest.approx(4.134724108, rel=1e-9)
    assert model.c1 == pytest.approx(1.618908033, rel=1e-9)


def test_public_model_inputs_reject_non_finite_or_wrong_types() -> None:
    with pytest.raises(ModelDomainError, match="size_error_kpc"):
        Geometry(135.0, 3.9, size_error_kpc=math.nan)
    with pytest.raises(ModelDomainError, match="gamma_min"):
        ModelAssumptions(gamma_max=math.inf)
    with pytest.raises(ModelDomainError, match="magnetic_ratio"):
        ModelAssumptions(magnetic_ratio=math.nan)
    with pytest.raises(ModelDomainError, match="integration_order"):
        ModelAssumptions(integration_order=64.0)  # type: ignore[arg-type]
    with pytest.raises(ModelDomainError, match="source name"):
        Observation(None, 0.1, Geometry(10.0, 2.0), ())  # type: ignore[arg-type]
    with pytest.raises(ModelDomainError, match="frequency, jet power"):
        cygnus_model().radio_power(math.nan, 1.0e39, 1.0e14, 0.6)


def test_kaiser2000_coefficient_is_explicitly_distinct() -> None:
    legacy = cygnus_model()
    modern = KDAForwardModel(
        legacy.observation,
        ModelAssumptions(c1_prescription="kaiser2000", integration_order=64),
    )
    assert modern.c1 == pytest.approx(1.838692068, rel=1e-9)
    assert modern.c1 != pytest.approx(legacy.c1, rel=0.01)


def test_dynamics_forward_inverse_round_trip() -> None:
    model = cygnus_model()
    q = 10**39.19
    age = 6.78 * MYR_S
    density_parameter = model.density_parameter_from_length(q, age)
    length = model.lobe_length(q, density_parameter, age)
    assert length == pytest.approx(model.observation.geometry.lobe_length_m, rel=2e-14)


def test_published_cygnus_dynamics_regression() -> None:
    model = cygnus_model()
    q = 10**39.19
    rho0 = 10**-22.45
    age = 6.78 * MYR_S
    density_parameter = rho0 * model.assumptions.core_radius_m ** model.assumptions.beta
    pressure = model.cocoon_pressure(q, density_parameter, age)
    length = model.lobe_length(q, density_parameter, age)
    assert math.log10(pressure) == pytest.approx(-10.59, abs=0.003)
    assert length / KPC_M == pytest.approx(68.54, rel=0.01)


def test_whole_source_power_needs_factor_two() -> None:
    model = cygnus_model(order=192)
    q = 10**39.19
    age = 6.78 * MYR_S
    total = model.radio_power(151e6, q, age, 0.607)
    one_lobe = model.radio_power_one_lobe(151e6, q, age, 0.607)
    assert total == pytest.approx(2 * one_lobe, rel=1e-14)
    # Rounded published parameters and powers only agree to a few percent.
    assert math.log10(total) == pytest.approx(27.796, abs=0.05)


def test_electron_integral_continuous_at_p_two() -> None:
    exact = electron_energy_integral(2.0, 1.0, 1e7)
    below = electron_energy_integral(2.0 - 1e-7, 1.0, 1e7)
    above = electron_energy_integral(2.0 + 1e-7, 1.0, 1e7)
    assert below == pytest.approx(exact, rel=2e-6)
    assert above == pytest.approx(exact, rel=2e-6)


def test_quadrature_convergence_at_published_solution() -> None:
    model = cygnus_model(order=96)
    error = model.quadrature_relative_error(4.8e9, 10**39.19, 6.78 * MYR_S, 0.607)
    assert error < 3e-3


def test_narrow_lorentz_cutoff_interval_is_not_missed() -> None:
    reference = cygnus_model()
    assumptions = replace(
        reference.assumptions,
        gamma_min=100.0,
        gamma_max=101.0,
        max_log_injection_span=8.0,
    )
    model = KDAForwardModel(reference.observation, assumptions)
    q = 10**39.19
    age = 6.78 * MYR_S
    frequency_hz = 1_749_065.341859284
    intervals = model._valid_intervals(  # noqa: SLF001 - numerical regression
        frequency_hz,
        model.dynamics_state(q, age),
        2.0 * 0.607 + 1.0,
    )
    assert intervals
    assert intervals[0][0] == pytest.approx(0.04676, abs=5.0e-4)
    assert intervals[0][1] == pytest.approx(0.06957, abs=5.0e-4)
    assert model.radio_power_one_lobe(frequency_hz, q, age, 0.607) > 0.0
    convergence = model.quadrature_relative_error(frequency_hz, q, age, 0.607)
    assert 0.0 <= convergence < 1.0e-6


def test_case2_one_lobe_and_two_sided_normalizations_agree() -> None:
    whole = cygnus_model(order=64)
    half_observation = Observation.from_points(
        "Cygnus A east lobe",
        whole.observation.redshift,
        Geometry(67.5, 1.95, 80.0, "one_lobe"),
        [
            SpectrumPoint(point.frequency_mhz, 0.5 * point.power_w_hz_sr)
            for point in whole.observation.spectrum
        ],
    )
    assumptions = replace(whole.assumptions, gamma_b=4.0 / 3.0, integration_order=64)
    whole_case2 = KDAForwardModel(whole.observation, assumptions)
    one_lobe_case2 = KDAForwardModel(half_observation, assumptions)
    q = 10**39.19
    age = 6.78 * MYR_S
    assert one_lobe_case2.observation.geometry.lobe_length_m == pytest.approx(
        whole_case2.observation.geometry.lobe_length_m
    )
    one_lobe_power = one_lobe_case2.radio_power(151.0e6, q, age, 0.607)
    whole_power = whole_case2.radio_power(151.0e6, q, age, 0.607)
    assert whole_power == pytest.approx(2.0 * one_lobe_power, rel=1.0e-12)


def test_legacy_energy_identity_at_cygnus() -> None:
    model = cygnus_model()
    derived = model.derived_quantities(10**39.19, 6.78 * MYR_S, 0.607)
    assert derived["legacy_energy_identity_ratio"] == pytest.approx(4.486310332)
    assert derived["injected_to_stored_energy_ratio"] == pytest.approx(4.486310332, rel=2e-12)
    assert derived["one_jet_qt_to_stored_energy_ratio"] == pytest.approx(
        2.243155166, rel=2e-12
    )


@pytest.mark.parametrize("p", [1.0 + 1e-9, 2.0 - 1e-9, 2.0, 2.0 + 1e-9, 2.2, 4.9])
@pytest.mark.parametrize(
    ("gamma_min", "gamma_max"),
    [(1.0, 1.0 + 1e-8), (100.0, 100.0 + 1e-9), (1.0, 1e7)],
)
def test_electron_integral_stable_for_narrow_ranges_and_limiting_indices(
    p: float, gamma_min: float, gamma_max: float
) -> None:
    # Evaluate the exact same analytic integral at higher precision.  Narrow
    # ranges near gamma=1 previously cancelled to zero or a negative value.
    with localcontext() as context:
        context.prec = 75
        index = Decimal(p)
        lower = Decimal(gamma_min)
        upper = Decimal(gamma_max)
        if index == 2:
            expected = (upper / lower).ln() + 1 / upper - 1 / lower
        else:
            expected = (
                (upper ** (2 - index) - lower ** (2 - index)) / (2 - index)
                - (upper ** (1 - index) - lower ** (1 - index)) / (1 - index)
            )
    assert electron_energy_integral(p, gamma_min, gamma_max) == pytest.approx(
        float(expected), rel=2e-12, abs=0.0
    )


def test_oversized_integer_model_inputs_raise_domain_errors() -> None:
    with pytest.raises(ModelDomainError, match="frequency_mhz"):
        SpectrumPoint(10**1000, 1e25)
    with pytest.raises(ModelDomainError, match="projected_size_kpc"):
        Geometry(10**1000, 3.9)
    with pytest.raises(ModelDomainError, match="gamma_min"):
        electron_energy_integral(2.2, 1.0, 10**1000)
