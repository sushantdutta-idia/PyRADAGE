"""Observed-snapshot magnetic anchoring, not an unaged r conversion."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from fr2dynage.constants import MU_0, MYR_S
from fr2dynage.io import load_configuration, normalized_input
from fr2dynage.model import (Geometry, KDAForwardModel, ModelAssumptions,
                            ModelDomainError, Observation, SpectrumPoint)
from fr2dynage.remnant import KDAExtForwardModel


def _observation():
    return Observation.from_points(
        "Synthetic anchored FR II", .0564,
        Geometry(135., 3.9, 80., "two_sided"),
        [SpectrumPoint(frequency, 1e26, .05)
         for frequency in [151., 408., 1400., 4800.]],
    )


def _model(field=5., **kwargs):
    return KDAForwardModel(_observation(), ModelAssumptions(
        equipartition_field_ug=field, integration_order=96, **kwargs))


@pytest.mark.parametrize("gamma", [4/3, 5/3])
@pytest.mark.parametrize("k_prime", [0., 100.])
@pytest.mark.parametrize("field", [1., 5., 20.])
def test_anchor_sets_all_present_cohort_fields_and_energy_closure(gamma, k_prime, field):
    model = _model(field, gamma_c=gamma, gamma_b=gamma, k_prime=k_prime)
    state = model.dynamics_state(10**39.19, 6.78*MYR_S)
    x = np.geomspace(1e-8, 1., 73)
    cohort = model._cohort_kinematics(x, 151e6, state, 2.2)
    current_ub = cohort[2]
    expected_field_t = .4*field*1e-10
    np.testing.assert_allclose(np.sqrt(2*MU_0*current_ub), expected_field_t, rtol=4e-14)
    derived = model.derived_quantities(10**39.19, 6.78*MYR_S, .6)
    assert derived['newest_cohort_magnetic_field_t'] == pytest.approx(expected_field_t, rel=2e-14)
    assert derived['supplied_equipartition_field_ug'] == field
    assert derived['adopted_field_fraction_of_equipartition'] == .4
    assert derived['adopted_current_magnetic_field_ug'] == pytest.approx(.4*field)
    ub = expected_field_t**2/(2*MU_0)
    uc = state.pressure_pa/(gamma-1.)
    assert derived['current_magnetic_energy_density_j_m3'] == pytest.approx(ub)
    assert derived['current_particle_energy_density_j_m3'] == pytest.approx(uc-ub)
    assert derived['magnetic_to_particle_energy_ratio_at_injection'] == pytest.approx(ub/(uc-ub))


def test_anchor_repartitions_energy_and_loss_history_consistently():
    anchored = _model(6.)
    q, age, alpha = 10**39.19, 6.78*MYR_S, .607
    state = anchored.dynamics_state(q, age)
    ratio = anchored._magnetic_ratio(2*alpha+1, state)
    frozen = KDAForwardModel(anchored.observation, replace(
        anchored.assumptions, equipartition_field_ug=None, magnetic_ratio=ratio))
    frequencies = np.array([151e6, 408e6, 1.4e9, 4.8e9])
    prediction = anchored.spectrum(frequencies, q, age, alpha)
    assert np.all(np.isfinite(prediction)) and np.all(prediction > 0.)
    np.testing.assert_allclose(prediction, frozen.spectrum(frequencies, q, age, alpha), rtol=2e-13)
    assert anchored.quadrature_relative_error(1.4e9, q, age, alpha) < 3e-3
    later = anchored.dynamics_state(q, 2*age)
    # A new anchored snapshot may infer a different injection partition.
    # A physical evolution track must instead freeze the best-fit partition.
    assert anchored._magnetic_ratio(2*alpha+1, later) != pytest.approx(ratio, rel=.1)


def test_absent_anchor_preserves_published_reference_partition():
    model = _model(None)
    assert model._magnetic_ratio(2.2) == pytest.approx(.8)
    assert model._magnetic_ratio(2.) == .75
    derived = model.derived_quantities(10**39.19, 6.78*MYR_S, .607)
    assert 'supplied_equipartition_field_ug' not in derived
    assert 'adopted_field_fraction_of_equipartition' not in derived
    explicit = KDAForwardModel(model.observation, replace(model.assumptions, magnetic_ratio=.8035))
    assert model.radio_power(151e6, 10**39.19, 6.78*MYR_S, .607) == pytest.approx(
        explicit.radio_power(151e6, 10**39.19, 6.78*MYR_S, .607), rel=1e-14)


@pytest.mark.parametrize("value", [0., -1., True, float('nan'), float('inf'), 1e-300, 1e300])
def test_invalid_or_unrepresentable_field_anchor_rejected(value):
    with pytest.raises(ModelDomainError, match='equipartition_field_ug'):
        ModelAssumptions(equipartition_field_ug=value)


def test_anchor_rejects_ambiguous_partition_and_nonuniform_evolution():
    with pytest.raises(ModelDomainError, match='mutually exclusive'):
        ModelAssumptions(equipartition_field_ug=5., magnetic_ratio=.1)
    with pytest.raises(ModelDomainError, match='gamma_b == gamma_c'):
        ModelAssumptions(equipartition_field_ug=5., gamma_b=4/3)
    with pytest.raises(ModelDomainError, match='dynamics snapshot'):
        _model()._magnetic_ratio(2.2)


@pytest.mark.parametrize("pressure", [0., -1., float('nan'), float('inf'), 1e-30])
def test_anchor_rejects_inadequate_or_invalid_snapshot_energy(pressure):
    model = _model(5.)
    state = replace(model.dynamics_state(1e39, 5*MYR_S), pressure_pa=pressure)
    with pytest.raises(ModelDomainError, match='total cocoon energy'):
        model._magnetic_ratio(2.2, state)


def test_unphysical_anchor_is_rejected_by_actual_emission_and_derived_calls():
    model = _model(1e6)
    with pytest.raises(ModelDomainError, match='total cocoon energy'):
        model.radio_power(151e6, 1e39, 5*MYR_S, .6)
    with pytest.raises(ModelDomainError, match='total cocoon energy'):
        model.derived_quantities(1e39, 5*MYR_S, .6)


def test_field_cannot_exhaust_exactly_all_snapshot_energy():
    model = _model(5., gamma_c=2., gamma_b=2.)
    ub = (.4*5.*1e-10)**2/(2*MU_0)
    state = replace(model.dynamics_state(1e39, 5*MYR_S), pressure_pa=ub)
    with pytest.raises(ModelDomainError, match='total cocoon energy'):
        model._magnetic_ratio(2.2, state)


def test_new_optional_anchor_preserves_old_positional_constructor_order():
    assumptions = ModelAssumptions(1.5, 10., 1., 1e7, 5/3, 5/3, 5/3,
                                  0., None, 2.7255, 'dynage2007', 96, 32.)
    assert assumptions.integration_order == 96
    assert assumptions.max_log_injection_span == 32.
    assert assumptions.equipartition_field_ug is None


def test_remnant_model_rejects_active_snapshot_anchor():
    with pytest.raises(ModelDomainError, match='not supported for remnant'):
        KDAExtForwardModel(_observation(), ModelAssumptions(equipartition_field_ug=5.))


@pytest.mark.parametrize("field", [None, 5.])
def test_anchor_json_and_normalized_input_round_trip(tmp_path, field):
    example = Path(__file__).resolve().parents[1]/'examples/cygnus_a.json'
    document = json.loads(example.read_text())
    document['model']['equipartition_field_ug'] = field
    first = tmp_path/'first.json'
    first.write_text(json.dumps(document))
    observation, assumptions, bounds, runtime = load_configuration(first)
    normalized = normalized_input(observation, assumptions, bounds, runtime)
    assert normalized['model']['equipartition_field_ug'] == field
    second = tmp_path/'second.json'
    second.write_text(json.dumps(normalized))
    reloaded = load_configuration(second)
    assert reloaded[0] == observation
    assert reloaded[1] == assumptions
    assert reloaded[2] == bounds
