"""Independent diagnostics, fixed-environment evolution and failure checks."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from fr2dynage.constants import KPC_M, MYR_S
from fr2dynage.dynage_diagnostics import (
    DiagnosticError, evolution_track, observed_spectrum_diagnostics,
    sample_intersection_tracks, summarize_fit_quality,
)
from fr2dynage.fitting import FitBounds, FitError, FitResult, IntersectionSolution
from fr2dynage.model import Geometry, KDAForwardModel, ModelAssumptions, ModelDomainError, Observation, SpectrumPoint


def _observation(*, curvature=0., errors=True, frequencies=None):
    frequencies = np.geomspace(50., 10000., 7) if frequencies is None else np.asarray(frequencies)
    x = np.log(frequencies) - np.mean(np.log(frequencies))
    powers = np.exp(math.log(1e25) - .7*x + curvature*x*x)
    return Observation.from_points('Synthetic diagnostic FR II', .15,
        Geometry(120., 4., inclination_deg=60.),
        [SpectrumPoint(float(nu), float(power), .03 if errors else None)
         for nu, power in zip(frequencies, powers)])


def _model(*, anchor=True):
    assumptions = ModelAssumptions(gamma_c=4/3, gamma_b=4/3,
        equipartition_field_ug=3. if anchor else None, integration_order=32,
        gamma_min=10., gamma_max=1e7, max_log_injection_span=8.)
    return KDAForwardModel(_observation(), assumptions)


def _result(model, **changes):
    defaults = dict(source_name=model.observation.name, method='classic_dynage', success=True,
        age_myr=10., alpha_inj=.7, jet_power_w=1e38,
        rho0_kg_m3=model.dynamics_state(1e38, 10*MYR_S).rho0_kg_m3,
        objective=0., objective_name='mean_pairwise_Delta_logQ_logrho',
        derived=model.derived_quantities(1e38, 10*MYR_S, .7), spectrum=[], diagnostics={})
    defaults.update(changes)
    return FitResult(**defaults)


def test_powerlaw_diagnostic_recovers_slope_without_inventing_a_break():
    result = observed_spectrum_diagnostics(_observation())
    assert result['power_law']['alpha_at_reference'] == pytest.approx(.7, abs=1e-12)
    assert result['power_law']['chi2'] < 1e-20
    assert result['quadratic']['curvature_c_ln'] == pytest.approx(0., abs=1e-12)
    assert not result['curvature_required_by_heuristic']
    assert result['age_identifiability'] == 'not_determined_by_this_diagnostic'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('curvature,steepening', [(-.15, True), (.15, False)])
def test_quadratic_curvature_has_explicit_sign_and_known_error_scale(curvature, steepening):
    result = observed_spectrum_diagnostics(_observation(curvature=curvature))
    assert result['quadratic']['curvature_c_ln'] == pytest.approx(curvature, abs=1e-12)
    assert result['quadratic']['alpha_at_reference'] == pytest.approx(.7, abs=1e-12)
    assert result['quadratic']['chi2'] < 1e-20
    assert result['curvature_chi2_improvement'] > 100
    assert result['curvature_required_by_heuristic']
    assert result['steepening_required_by_heuristic'] is steepening
    assert np.sign(result['curvature_z']) == np.sign(curvature)
    # Independent covariance from the unscaled polynomial design.
    frequency = np.array(result['frequencies_mhz_rest'])
    x = np.log(frequency) - np.mean(np.log(frequency))
    matrix = np.column_stack((np.ones_like(x), x, x*x))
    sigma = math.sqrt(math.log1p(.03**2))
    covariance = np.linalg.inv(matrix.T @ matrix) * sigma**2
    np.testing.assert_allclose(result['quadratic']['coefficient_covariance'], covariance, rtol=1e-10, atol=1e-16)


def test_missing_errors_do_not_produce_calibrated_curvature_significance():
    observation = _observation(curvature=-.15)
    points = list(observation.spectrum)
    points[2] = replace(points[2], fractional_error=None)
    result = observed_spectrum_diagnostics(replace(observation, spectrum=tuple(points)))
    assert not result['all_errors_supplied']
    assert result['power_law']['chi2'] is None
    assert result['quadratic']['chi2'] is None
    assert result['quadratic']['coefficient_covariance'] is None
    assert result['curvature_z'] is None
    assert result['curvature_chi2_improvement'] is None
    assert not result['curvature_required_by_heuristic']


def test_three_points_cannot_supply_quadratic_residual_diagnostics():
    result = observed_spectrum_diagnostics(_observation(frequencies=[100., 300., 1000.]))
    assert result['quadratic'] is None
    assert result['curvature_z'] is None
    assert any('four' in warning for warning in result['warnings'])


def test_extreme_powers_and_fractional_errors_remain_json_safe():
    observation = _observation()
    points = tuple(SpectrumPoint(float(nu), float(power), 1e155)
                   for nu, power in zip(np.geomspace(100., 10000., 7), np.geomspace(1e-300, 1e300, 7)))
    result = observed_spectrum_diagnostics(replace(observation, spectrum=points))
    assert math.isfinite(result['power_law']['alpha_at_reference'])
    json.dumps(result, allow_nan=False)


def test_active_evolution_holds_environment_and_follows_analytic_scalings():
    model = _model()
    result = _result(model)
    document = evolution_track(model, result, [5., 10., 20.], [100., 1000.])
    rows = document['tracks']
    assert all(row['valid'] for row in rows)
    exponent = 3 / (5 - model.assumptions.beta)
    pressure_exponent = -(4 + model.assumptions.beta) / (5 - model.assumptions.beta)
    assert rows[-1]['lobe_length_kpc'] / rows[0]['lobe_length_kpc'] == pytest.approx(4**exponent)
    assert rows[-1]['cocoon_pressure_pa'] / rows[0]['cocoon_pressure_pa'] == pytest.approx(4**pressure_exponent)
    density = [row['density_parameter_rho0_a0_beta_si'] for row in rows]
    np.testing.assert_allclose(density, document['density_parameter_rho0_a0_beta_si'], rtol=1e-12)
    assert rows[1]['lobe_length_kpc'] == pytest.approx(model.observation.geometry.lobe_length_m/KPC_M)
    assert rows[1]['projected_size_kpc'] == pytest.approx(120.)
    assert all(len(row['radio_power_w_hz_sr_by_frequency']) == 2 for row in rows)
    json.dumps(document, allow_nan=False)


def test_field_anchor_is_resolved_only_at_fit_epoch_then_evolves():
    model = _model()
    result = _result(model)
    rows = evolution_track(model, result, [5., 10., 20.], [1000.])['tracks']
    assert rows[1]['magnetic_field_t'] == pytest.approx(.4*3e-10)
    assert rows[0]['magnetic_field_t'] > rows[1]['magnetic_field_t'] > rows[2]['magnetic_field_t']
    assert rows[-1]['magnetic_field_t'] / rows[0]['magnetic_field_t'] == pytest.approx(
        math.sqrt(rows[-1]['cocoon_pressure_pa'] / rows[0]['cocoon_pressure_pa']))


def test_one_lobe_evolution_preserves_projection_convention():
    model = _model(anchor=False)
    observation = replace(model.observation, geometry=Geometry(120., 2., inclination_deg=30., mode='one_lobe'))
    model = KDAForwardModel(observation, model.assumptions)
    result = _result(model)
    row = evolution_track(model, result, [10.], [1000.])['tracks'][0]
    assert row['valid']
    assert row['projected_size_kpc'] == pytest.approx(120.)
    assert row['lobe_length_kpc'] == pytest.approx(240.)


def test_evolution_domain_failure_is_recorded_without_losing_valid_epochs(monkeypatch):
    model = _model()
    result = _result(model)
    original = KDAForwardModel.radio_power
    def radio_power(self, frequency, power, age, alpha):
        if age < 8*MYR_S:
            raise ModelDomainError('test early-time invalid domain')
        return original(self, frequency, power, age, alpha)
    monkeypatch.setattr(KDAForwardModel, 'radio_power', radio_power)
    document = evolution_track(model, result, [5., 10., 20.], [1000.])
    assert [row['valid'] for row in document['tracks']] == [False, True, True]
    assert 'early-time' in document['tracks'][0]['reason']
    assert document['valid_points'] == 2
    json.dumps(document, allow_nan=False)


def test_intersection_tracks_hold_observed_size_and_report_root_counts(monkeypatch):
    model = _model(anchor=False)
    result = _result(model)
    frequencies = [point.frequency_mhz for point in model.observation.spectrum]
    # Independent luminosity model: exact Q(t)=1e38*(10 Myr/t) at all bands.
    by_frequency = {point.frequency_hz: point.power_w_hz_sr for point in model.observation.spectrum}
    monkeypatch.setattr(model, 'radio_power', lambda frequency, q, time, alpha:
                        by_frequency[frequency] * (q/1e38) * (time/(10*MYR_S)))
    document = sample_intersection_tracks(model, result,
        FitBounds(age_myr=(1., 100.), log10_jet_power_w=(36., 40.), power_scan_size=9), [5., 10., 20.])
    assert document['frequencies_mhz_rest'] == frequencies
    assert document['track_kind'] == 'intersection_family_at_fixed_observed_size'
    assert document['sampled_age_selection']['status'] == 'flat_or_numerically_unresolved_delta_profile'
    assert document['sampled_age_selection']['sampled_minimum_age_myr'] is None
    for row in document['tracks']:
        assert row['valid']
        np.testing.assert_allclose(row['log10_q_by_frequency'], 38+math.log10(10/row['age_myr']), atol=2e-6)
        assert row['delta'] < 1e-5
        assert row['maximum_roots_in_power_equation'] == 1


def test_failed_intersection_epochs_remain_explicit_and_json_safe(monkeypatch):
    model = _model()
    result = _result(model)
    def intersection(model, age, alpha, bounds):
        if age == 5.:
            raise FitError('no power bracket')
        n = len(model.observation.spectrum)
        return IntersectionSolution(age, alpha, (38.,)*n, (-23.,)*n, .1, 38., -23.), 2
    monkeypatch.setattr('fr2dynage.dynage_diagnostics.intersection_at_age', intersection)
    document = sample_intersection_tracks(model, result, FitBounds(), [5., 10.])
    assert not document['tracks'][0]['valid']
    assert document['tracks'][0]['log10_q_by_frequency'] is None
    assert document['tracks'][1]['maximum_roots_in_power_equation'] == 2
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize('ages', [[0., 1.], [1., 1.], [2., 1.], [True], [float('nan')], [], [[1., 2.]]])
def test_track_age_grids_fail_closed(ages):
    model = _model()
    with pytest.raises(DiagnosticError):
        evolution_track(model, _result(model), ages, [1000.])


def test_intersection_grid_cannot_silently_extend_fit_domain():
    model = _model()
    with pytest.raises(DiagnosticError, match='inside'):
        sample_intersection_tracks(model, _result(model), FitBounds(age_myr=(1., 100.)), [.5, 10.])


@pytest.mark.parametrize('changes', [{'success': False}, {'activity_history': {'active_duration_myr': 5.}},
                                      {'method': 'remnant_joint_log_likelihood'}])
def test_active_tracks_reject_failed_or_nonactive_results(changes):
    model = _model()
    with pytest.raises(DiagnosticError):
        evolution_track(model, _result(model, **changes), [10.], [1000.])


def test_non_kda_model_is_not_silently_treated_as_frii():
    model = _model()
    with pytest.raises(DiagnosticError, match='active FR II'):
        evolution_track(object(), _result(model), [10.], [1000.])


def test_quality_summary_preserves_heuristic_scope_and_collects_failure_flags():
    model = _model()
    result = _result(model, diagnostics={'weighted_log_power_chi2_at_consensus': 100.,
        'age_search': {'age_grid_boundary': True, 'maximum_roots_in_power_equation': 2},
        'quadrature_relative_error_order_vs_double': [.01]})
    result.derived['instantaneous_head_speed_over_c'] = 1.1
    summary = summarize_fit_quality(model, result, FitBounds())
    for flag in ['poor_consensus_spectrum_fit', 'multiple_luminosity_roots',
                 'quadrature_error_above_0p5_percent', 'superluminal_head_speed',
                 'parameter_or_classic_root_near_search_boundary']:
        assert flag in summary['flags']
    assert summary['classical_objective_is_likelihood'] is False
    assert summary['reduced_chi2'] == 25.
    json.dumps(summary, allow_nan=False)


def test_joint_quality_reports_unavailable_or_ill_conditioned_covariance():
    model = _model()
    result = _result(model, method='joint_log_likelihood', diagnostics={
        'chi2': 1., 'information_matrix_condition_number': 1e9, 'local_covariance': None})
    summary = summarize_fit_quality(model, result, FitBounds())
    assert 'ill_conditioned_local_information' in summary['flags']
    assert 'local_covariance_unavailable' in summary['flags']


def test_flat_classic_energy_selection_is_not_reported_as_a_confidence_interval():
    model = _model()
    result = _result(model, diagnostics={'classic_alpha_scan': [
        {'alpha_inj': alpha, 'log10_q_times_t_j': 52.+offset}
        for alpha, offset in [(.5, .003), (.6, .001), (.7, .002)]]})
    summary = summarize_fit_quality(model, result, FitBounds())
    assert 'flat_classic_energy_selection_profile' in summary['flags']
    assert summary['classic_energy_selection_dynamic_range_dex'] == pytest.approx(.002)
    assert 'not a confidence' in summary['classic_energy_flatness_rule']


def test_known_fixed_power_is_not_counted_as_a_fitted_parameter():
    model = _model()
    result = _result(model, method='joint_fixed_jet_power_log_likelihood',
                     diagnostics={'chi2': 10., 'local_covariance': [[1., 0.], [0., 1.]]})
    summary = summarize_fit_quality(model, result, FitBounds())
    assert summary['free_parameter_count'] == 2
    assert summary['degrees_of_freedom'] == 5
    assert summary['reduced_chi2'] == 2.
