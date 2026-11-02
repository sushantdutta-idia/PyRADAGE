"""Data-only PyRATS interchange, units and conditional-uncertainty checks."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from fr2dynage.pyrats_bridge import PyRATSPropertyError, load_pyrats_properties


def _report():
    return {
        'pyrats_version': '2.1.0', 'generated_utc': '2026-10-28T12:00:00Z',
        'configuration': {'redshift': .1469, 'cosmology': {'H0_km_s_Mpc': 69.6, 'Om0': .286}},
        'input_maps': [{'label': 'band_1000', 'path': 'raw-radio.fits', 'frequency_Hz': 1e9, 'OBJECT': 'Synthetic fixture'}],
        'geometry_physical': {'volume_m3': 1e65, 'filling_factor': 1.,
                              'volume_fractional_error': .4, 'largest_linear_size_kpc': 400.,
                              'assumption_note': 'Line-of-sight depth is assumed equal to projected width.'},
        'self_consistent_solution': {
            'converged': True, 'electron_spectrum': 'powerlaw', 'spectrum_fallback': False,
            'alpha_inj_positive': .5, 'alpha_inj_err': .05, 'alpha_source': 'user_override',
            'sub_equipartition': {'fraction_of_B_eq': .371543, 'B_uG': 2.229258, 'applied_to_analysis': True},
            'equipartition': {
                'inputs': {'redshift': .1469, 'kappa_proton_electron_energy_ratio': 100.,
                           'zeta_uB_over_uparticles': 101., 'filling_factor': 1.,
                           'gamma_min': 10., 'gamma_max': 1e7, 'volume_m3': 1e65},
                'direct': {
                    'equipartition': {'B_uG': 6., 'B_T': 6e-10, 'volume_m3': 1e65,
                                     'total_energy_J': 6e52, 'pressure_Pa': 1e-13},
                    'minimum_energy': {'B_uG': 5.5, 'B_T': 5.5e-10, 'volume_m3': 1e65,
                                       'total_energy_J': 5.9e52, 'pressure_Pa': 9.9e-14}},
                'monte_carlo': {'status': 'ok', 'n_success': 1000,
                    'sampled': ['flux', 'alpha', 'volume'], 'not_sampled': ['kappa', 'filling factor'],
                    'B_eq_uG': {'median': 6.3, 'p16': 5.6, 'p84': 7.1, 'p2.5': 5., 'p97.5': 8.},
                    'B_me_uG': {'median': 5.8, 'p16': 5.2, 'p84': 6.5}}}}}


def _write(tmp_path, document=None):
    path = tmp_path / 'PyRATS_results.json'
    path.write_text(json.dumps(_report() if document is None else document))
    return path


def test_field_uses_mc_median_and_explicit_systematic_not_adopted_override(tmp_path):
    path = _write(tmp_path)
    result = load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.35)
    prior = result['priors']['magnetic_field_ug']
    assert math.exp(prior['log_mean']) == pytest.approx(0.4 * 6.3)
    assert prior['log_sigma'] == pytest.approx(math.hypot(.5 * math.log(7.1/5.6), .35))
    assert result['properties']['field_direct_uG'] == 6.
    assert result['properties']['field_input_direct_uG'] == pytest.approx(2.4)
    assert result['provenance']['field_scale_assumption'] == 0.4
    assert result['properties']['injection_index'] == {
        'value_positive': .5, 'error': .05, 'source': 'user_override', 'imported_as_prior': False}
    assert set(result['priors']) == {'magnetic_field_ug'}
    assert result['provenance']['same_radio_data_reuse_approximation']
    assert result['provenance']['report_adopted_field_not_automatically_used']['B_uG'] == 2.229258
    assert result['provenance']['assumptions']['kappa_proton_electron_energy_ratio'] == 100
    assert result['provenance']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not result['provenance']['image_association']['pixel_identity_verified']
    json.dumps(result, allow_nan=False)


def test_minimum_energy_and_explicit_field_scaling_preserve_native_budgets(tmp_path):
    result = load_pyrats_properties(_write(tmp_path), expected_redshift=.1469,
        systematic_log_sigma=.2, field_basis='minimum_energy', field_scale=.4)
    assert math.exp(result['priors']['magnetic_field_ug']['log_mean']) == pytest.approx(5.8*.4)
    assert result['properties']['native_field_budgets']['minimum_energy']['pressure_Pa'] == 9.9e-14
    assert result['provenance']['field_scale_assumption'] == .4
    assert any('not rescaled' in warning for warning in result['provenance']['warnings'])


def test_prepared_map_label_association_is_not_hash_identity(tmp_path):
    report = _write(tmp_path)
    target = tmp_path / 'band_1000.fits'
    target.write_bytes(b'prepared test data; the adapter does not execute or parse the image')
    result = load_pyrats_properties(report, expected_redshift=.1469, systematic_log_sigma=.3,
        image_paths=[{'path': target, 'frequency_mhz': 1000.}])
    association = result['provenance']['image_association']
    assert association['status'] == 'associated_with_reported_inputs'
    assert association['matches'][0]['association'] == 'prepared_map_label_only'
    assert association['matches'][0]['frequency_checked']
    assert association['matches'][0]['target_sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not association['pixel_identity_verified']


def test_exact_input_hash_is_verified_if_report_supplies_it(tmp_path):
    target = tmp_path / 'raw-radio.fits'
    target.write_bytes(b'raw radio data')
    document = _report()
    document['input_maps'][0]['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    path = _write(tmp_path, document)
    result = load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.3,
                                    image_paths=[target])
    assert result['provenance']['image_association']['pixel_identity_verified']
    target.write_bytes(b'changed image')
    with pytest.raises(PyRATSPropertyError, match='SHA-256'):
        load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.3, image_paths=[target])


def test_report_prose_and_command_line_are_never_executed(tmp_path):
    sentinel = tmp_path / 'must-not-exist'
    document = _report()
    document['command_line'] = f"touch '{sentinel}'"
    document['instructions'] = f'run {document["command_line"]}'
    load_pyrats_properties(_write(tmp_path, document), expected_redshift=.1469, systematic_log_sigma=.3)
    assert not sentinel.exists()


@pytest.mark.parametrize('mutate,match', [
    (lambda d: d.update(pyrats_version='3.0.0'), 'version'),
    (lambda d: d['configuration'].update(redshift=.2), 'redshift'),
    (lambda d: d['self_consistent_solution'].update(converged=False), 'converged'),
    (lambda d: d['self_consistent_solution']['equipartition']['inputs'].update(redshift=.3), 'redshifts'),
    (lambda d: d['self_consistent_solution']['equipartition']['inputs'].update(filling_factor=2), 'filling'),
    (lambda d: d['self_consistent_solution']['equipartition']['inputs'].update(zeta_uB_over_uparticles=1), 'zeta'),
    (lambda d: d['self_consistent_solution']['equipartition']['inputs'].update(gamma_max=1), 'electron'),
    (lambda d: d['self_consistent_solution']['equipartition']['direct']['equipartition'].update(B_T=6), 'unit'),
    (lambda d: d['self_consistent_solution']['equipartition']['monte_carlo'].update(status='failed'), 'Monte Carlo'),
    (lambda d: d['self_consistent_solution']['equipartition']['monte_carlo'].update(n_success=True), 'successful'),
    (lambda d: d['self_consistent_solution']['equipartition']['monte_carlo']['B_eq_uG'].update(p16=7), 'p16'),
    (lambda d: d['geometry_physical'].update(volume_m3=2e65), 'volumes'),
    (lambda d: d['geometry_physical'].update(filling_factor=.5), 'filling factors'),
    (lambda d: d['configuration'].update(kappa_proton_electron_energy_ratio=0.), 'assumptions disagree'),
    (lambda d: d['self_consistent_solution']['equipartition']['direct']['minimum_energy'].update(volume_m3=2e65), 'volumes'),
    (lambda d: d['input_maps'].append(deepcopy(d['input_maps'][0])), 'duplicate'),
])
def test_malformed_or_inconsistent_scientific_results_fail(tmp_path, mutate, match):
    document = _report()
    mutate(document)
    with pytest.raises(PyRATSPropertyError, match=match):
        load_pyrats_properties(_write(tmp_path, document), expected_redshift=.1469, systematic_log_sigma=.3)


@pytest.mark.parametrize('options', [
    {'systematic_log_sigma': -1}, {'systematic_log_sigma': True},
    {'field_scale': 0}, {'field_scale': float('nan')},
    {'field_basis': 'analysis_field'}, {'expected_redshift': True},
])
def test_invalid_adapter_options_fail(tmp_path, options):
    arguments = {'expected_redshift': .1469, 'systematic_log_sigma': .3, **options}
    with pytest.raises(PyRATSPropertyError):
        load_pyrats_properties(_write(tmp_path), **arguments)


def test_image_frequency_and_source_association_mismatch_fail(tmp_path):
    path = _write(tmp_path)
    image = tmp_path / 'band_1000.fits'
    image.write_bytes(b'data')
    with pytest.raises(PyRATSPropertyError, match='frequency'):
        load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.3,
                              image_paths=[{'path': image, 'frequency_mhz': 1400.}])
    other = tmp_path / 'different_source.fits'
    other.write_bytes(b'data')
    with pytest.raises(PyRATSPropertyError, match='no unique'):
        load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.3, image_paths=[other])


@pytest.mark.parametrize('text', ['{"pyrats_version":"2.1.0","pyrats_version":"2.1.1"}',
                                '{"pyrats_version":NaN}', '[]'])
def test_strict_json_loader(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises(PyRATSPropertyError):
        load_pyrats_properties(path, expected_redshift=.1469, systematic_log_sigma=.3)


def test_extra_systematic_is_required_by_api(tmp_path):
    with pytest.raises(TypeError, match='systematic_log_sigma'):
        load_pyrats_properties(_write(tmp_path), expected_redshift=.1469)


@pytest.mark.parametrize('counts', [
    {'n_requested':2000}, {'n_requested':999}, {'n_requested':True},
    {'n_requested':1000,'n_rejected':1}, {'n_rejected':-1}, {'n_rejected':True}])
def test_bad_monte_carlo_accounting_rejected(tmp_path, counts):
    document = _report()
    document['self_consistent_solution']['equipartition']['monte_carlo'].update(counts)
    with pytest.raises(PyRATSPropertyError, match='Monte Carlo'):
        load_pyrats_properties(_write(tmp_path, document), expected_redshift=.1469, systematic_log_sigma=0.)
