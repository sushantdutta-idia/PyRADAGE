"""PyRATS field-input integration with the actual radio inference workflow."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

pytest.importorskip('astropy')


def _generator():
    source = Path(__file__).resolve().parents[1] / 'examples/generate_radio_demo.py'
    spec = importlib.util.spec_from_file_location('pyrats_integration_demo', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _properties_report(folder, configuration):
    """Synthetic physical-property fixture, explicitly not an observed source."""
    return {
        'pyrats_version': '2.1.0', 'generated_utc': '2026-10-28T00:00:00Z',
        'configuration': {'redshift': configuration['source']['redshift'],
                          'cosmology': {'H0_km_s_Mpc': 67.66, 'Om0': .30966}},
        'input_maps': [
            {'label': Path(record['path']).stem, 'path': record['path'],
             'frequency_Hz': record['frequency_mhz'] * 1e6,
             'sha256': hashlib.sha256((folder / record['path']).read_bytes()).hexdigest(),
             'OBJECT': 'Synthetic integration fixture'} for record in configuration['images']],
        'geometry_physical': {'volume_m3': 1e65, 'filling_factor': 1.,
                              'assumption_note': 'Synthetic adopted line-of-sight depth, not measured.'},
        'self_consistent_solution': {
            'converged': True, 'electron_spectrum': 'powerlaw', 'spectrum_fallback': False,
            'alpha_inj_positive': .5, 'alpha_inj_err': .05, 'alpha_source': 'user_override',
            'sub_equipartition': {'fraction_of_B_eq': .4, 'B_uG': 1.12, 'applied_to_analysis': True},
            'equipartition': {
                'inputs': {'redshift': configuration['source']['redshift'],
                           'kappa_proton_electron_energy_ratio': 0., 'zeta_uB_over_uparticles': 1.,
                           'filling_factor': 1., 'gamma_min': 10., 'gamma_max': 1e7,
                           'volume_m3': 1e65},
                # Direct and MC centres intentionally differ to catch importing
                # the wrong field estimate or the adopted sub-equipartition B.
                'direct': {'equipartition': {'B_uG': 2.8, 'B_T': 2.8e-10},
                           'minimum_energy': {'B_uG': 2.6, 'B_T': 2.6e-10}},
                'monte_carlo': {'status': 'ok', 'n_success': 1000,
                    'sampled': ['normalization flux', 'adopted volume'],
                    'not_sampled': ['filling factor', 'kappa'],
                    'B_eq_uG': {'median': 3., 'p16': 2.94, 'p84': 3.06},
                    'B_me_uG': {'median': 2.8, 'p16': 2.74, 'p84': 2.86}}}}}


@pytest.fixture(scope='module')
def pyrats_radio_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp('pyrats-radio')
    _generator().generate(folder, particles=128)
    configuration = json.loads((folder / 'radio.json').read_text())
    properties = _properties_report(folder, configuration)
    report = folder / 'PyRATS_results.json'
    report.write_text(json.dumps(properties))
    configuration['priors'].pop('magnetic_field_ug')
    configuration['pyrats_properties'] = {
        'path': report.name, 'field_basis': 'equipartition',
        'systematic_log_sigma': .03}
    configuration['inference']['replicates'] = 2
    configuration['prior_description'] += ' PyRATS input is a synthetic conditional-property fixture.'
    config_path = folder / 'with-pyrats.json'
    config_path.write_text(json.dumps(configuration))
    return folder, config_path, configuration, properties


@pytest.fixture(scope='module')
def pyrats_radio_result(pyrats_radio_case):
    from fr2dynage.radio_inference import compute_radio_inference
    return compute_radio_inference(pyrats_radio_case[1])


def _modified(pyrats_radio_case, tmp_path, mutation, *, report_mutation=None):
    folder, _, original, report = pyrats_radio_case
    config = deepcopy(original)
    for image in config['images']:
        image['path'] = str(folder / image['path'])
    config['selection']['mask'] = str(folder / config['selection']['mask'])
    for key in ('distance_map', 'branch_map'):
        config['geometry'][key] = str(folder / config['geometry'][key])
    if report_mutation is None:
        config['pyrats_properties']['path'] = str(folder / config['pyrats_properties']['path'])
    else:
        report = deepcopy(report)
        # Raw-image paths in a moved report retain the original image origin.
        for record in report['input_maps']:
            record['path'] = str(folder / record['path'])
        report_mutation(report)
        property_path = tmp_path / 'properties.json'
        property_path.write_text(json.dumps(report))
        config['pyrats_properties']['path'] = str(property_path)
    mutation(config)
    path = tmp_path / 'modified.json'
    path.write_text(json.dumps(config))
    return path


def test_pyrats_field_prior_is_imported_without_copying_fixed_injection_index(pyrats_radio_result):
    result = pyrats_radio_result
    imported = result.metadata['pyrats_properties']
    prior = result.metadata['priors']['magnetic_field_ug']
    assert math.exp(prior['log_mean']) == pytest.approx(0.4 * 3.)
    assert prior['log_sigma'] == pytest.approx(math.hypot(.5*math.log(3.06/2.94), .03))
    assert imported['properties']['field_direct_uG'] == 2.8
    assert imported['properties']['injection_index']['value_positive'] == .5
    assert not imported['properties']['injection_index']['imported_as_prior']
    assert result.metadata['priors']['alpha_inj']['mean'] == .65
    assert result.metadata['global_dynamical_age_myr']['mean'] is None
    assert result.metadata['global_dynamical_age_myr']['median'] is None


def test_pyrats_lineage_and_data_reuse_warning_reach_result(pyrats_radio_case, pyrats_radio_result):
    folder, _, _, _ = pyrats_radio_case
    metadata = pyrats_radio_result.metadata
    imported = metadata['pyrats_properties']['provenance']
    assert imported['same_radio_data_reuse_approximation']
    association = imported['image_association']
    assert len(association['matches']) == len(metadata['frequencies_mhz'])
    assert association['pixel_identity_verified']
    assert all(record['frequency_checked'] for record in association['matches'])
    assert all(record['association'] == 'exact_input_path' for record in association['matches'])
    assert any('full joint flux-field covariance' in warning for warning in metadata['warnings'])
    assert any('does not jointly refit' in warning for warning in metadata['warnings'])
    report = (folder / 'PyRATS_results.json').resolve()
    assert report in pyrats_radio_result.input_paths
    record = next(record for record in metadata['input_files'] if record['role'] == 'pyrats_properties')
    assert record['sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()


def test_pyrats_field_and_explicit_field_prior_are_mutually_exclusive(pyrats_radio_case, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified(pyrats_radio_case, tmp_path, lambda d: d['priors'].update(
        magnetic_field_ug={'distribution': 'log_uniform', 'low': 1., 'high': 10.}))
    with pytest.raises(ValueError, match='not both'):
        compute_radio_inference(path)


def test_pyrats_target_redshift_mismatch_fails_before_inference(pyrats_radio_case, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified(pyrats_radio_case, tmp_path, lambda d: None,
        report_mutation=lambda d: d['configuration'].update(redshift=.3))
    with pytest.raises(ValueError, match='redshift'):
        compute_radio_inference(path)


def test_pyrats_systematic_width_is_an_explicit_required_input(pyrats_radio_case, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified(pyrats_radio_case, tmp_path,
                     lambda d: d['pyrats_properties'].pop('systematic_log_sigma'))
    with pytest.raises(ValueError, match='systematic_log_sigma'):
        compute_radio_inference(path)


def test_output_cannot_replace_pyrats_properties_report(pyrats_radio_case, pyrats_radio_result, tmp_path):
    from fr2dynage.radio_inference import write_radio_products
    folder, _, _, _ = pyrats_radio_case
    report = folder / 'PyRATS_results.json'
    original = report.read_bytes()
    summary = tmp_path / 'should-not-write.json'
    with pytest.raises(ValueError, match='overwrite|input'):
        write_radio_products(pyrats_radio_result, report, summary, overwrite=True)
    assert report.read_bytes() == original
    assert not summary.exists()


def test_pyrats_prior_overrides_catalogue_field_with_recorded_provenance(pyrats_radio_case, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    folder, _, _, _ = pyrats_radio_case
    path = _modified(pyrats_radio_case, tmp_path, lambda d: d.update(reference_catalogue={
        'path': str(folder / 'reference-synthetic.json'), 'allow_synthetic': True}))
    result = compute_radio_inference(path)
    assert result.metadata['pyrats_overrides_reference_field']
    assert result.metadata['reference_catalogue']['contains_synthetic_constraints']
    assert math.exp(result.metadata['priors']['magnetic_field_ug']['log_mean']) == pytest.approx(0.4 * 3.)


@pytest.mark.parametrize('scale', [1., .371543, .5])
def test_source_field_fraction_cannot_override_fixed_prescription(pyrats_radio_case, tmp_path, scale):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified(pyrats_radio_case, tmp_path,
                     lambda d: d['pyrats_properties'].update(field_scale=scale))
    with pytest.raises(ValueError, match='fixed at B = 0.4 B_eq'):
        compute_radio_inference(path)


def test_source_field_recipe_requires_equipartition_basis(pyrats_radio_case, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified(pyrats_radio_case, tmp_path,
                     lambda d: d['pyrats_properties'].update(field_basis='minimum_energy'))
    with pytest.raises(ValueError, match='fixed at B = 0.4 B_eq'):
        compute_radio_inference(path)


def test_fixed_fraction_reaches_samples_and_fits(pyrats_radio_result, tmp_path):
    from astropy.io import fits
    import numpy as np
    from fr2dynage.radio_inference import write_radio_products
    result = pyrats_radio_result
    assert result.metadata['field_prescription']['ratio_is_fixed']
    assert result.metadata['field_prescription']['fraction'] == .4
    np.testing.assert_allclose(result.samples['magnetic_field_ug'],
                               .4 * result.samples['equipartition_field_ug'], rtol=1e-15)
    target, summary = tmp_path/'field.fits', tmp_path/'summary.json'
    write_radio_products(result, target, summary)
    with fits.open(target) as hdus:
        assert hdus[0].header['BFRACEQ'] == .4
