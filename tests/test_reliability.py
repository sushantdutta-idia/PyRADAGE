"""Reject inconsistent physics, false reporting support and changed artifacts."""
from copy import deepcopy
import json

import pytest

from fr2dynage.constants import MU_0
from fr2dynage.reliability import (validate_adopted_budget, radio_parameter_quality,
    sha256_file, verify_input_records, verify_analysis_manifest)


def physical_budget():
    report = {'self_consistent_solution': {'equipartition': {
        'inputs': {'volume_m3': 1e65, 'kappa_proton_electron_energy_ratio': 100.},
        'direct': {'equipartition': {'B_uG': 6.}}}}}
    ue = 6e-15
    up = 101*ue
    ub = (2.4e-10)**2/(2*MU_0)
    u = up+ub
    budget = {'budget': {'fraction_of_B_eq': .4, 'B_uG': 2.4, 'B_T': 2.4e-10,
        'B_eq_uG': 6., 'B_eq_T': 6e-10, 'electron_energy_density_J_m3': ue,
        'particle_energy_density_J_m3': up, 'magnetic_energy_density_J_m3': ub,
        'total_energy_density_J_m3': u, 'total_energy_J': u*1e65, 'pressure_Pa': u/3}}
    return budget, report


def test_consistent_physical_budget_accounting_is_conditional():
    budget, report = physical_budget()
    result = validate_adopted_budget(budget, report)
    assert result['status'] == 'accounting_verified_conditional_budget'
    assert len(result['checks']) == 10
    assert 'not independently verified' in result['scope']


@pytest.mark.parametrize('key', ['B_T', 'B_eq_uG', 'B_eq_T', 'total_energy_J', 'pressure_Pa',
    'electron_energy_density_J_m3', 'particle_energy_density_J_m3',
    'magnetic_energy_density_J_m3', 'total_energy_density_J_m3'])
def test_positive_but_inconsistent_budget_is_rejected(key):
    budget, report = physical_budget()
    budget['budget'][key] *= 10
    with pytest.raises(ValueError, match='budget'):
        validate_adopted_budget(budget, report)


def metadata():
    return {'fit': {'adequate_at_threshold_4': True},
        'sampling': {'status': 'replicate_consistent'},
        'identifiability': {'curvature': 'required_by_heuristic'},
        'posterior_parameters': {'alpha_inj': {'median': .7}, 'magnetic_field_ug': {'median': 2.4}},
        'map_statistics': {'area_weighted_mean': {'median': 40.}, 'area_weighted_median': {'median': 35.}}}


def test_supported_clock_is_still_prior_conditioned_not_global_lifetime():
    result = radio_parameter_quality(metadata())
    assert result['transport']['mean_myr']['median'] == 40.
    assert result['parameters']['alpha_inj']['estimate']['median'] == .7
    assert result['global_dynamical_age_myr']['mean'] is None


def test_upstream_budget_index_mismatch_is_explicit_not_a_new_budget_fit():
    data = metadata()
    data['posterior_parameters']['alpha_inj']['credible_interval_68'] = [.68,.72]
    data['pyrats_properties'] = {'properties': {'injection_index': {'value_positive': .5}}}
    result = radio_parameter_quality(data, budget_validation={'status':'accounting_verified_conditional_budget'})
    check = result['source_budget']['injection_spectrum_consistency']
    assert check['status'] == 'upstream_index_outside_transport_68'
    assert 'not refitted' in check['interpretation']


@pytest.mark.parametrize('section,key,value', [
    ('fit', 'adequate_at_threshold_4', False), ('sampling', 'status', 'single_run_unverified'),
    ('identifiability', 'curvature', 'not_required_by_radio_data'),
    ('distance_scale_consistency', 'status', 'inconsistent')])
def test_unsupported_source_parameters_withheld_but_diagnostics_retained(section,key,value):
    data = metadata(); data.setdefault(section,{})[key] = value
    result = radio_parameter_quality(data)
    assert result['transport']['status'] == 'diagnostic_only'
    assert result['transport']['mean_myr'] is None
    assert result['transport']['median_myr'] is None
    assert result['parameters']['alpha_inj']['estimate'] is None
    assert result['parameters']['alpha_inj']['diagnostic_summary']['median'] == .7
    assert result['transport']['reasons']


def test_input_mutation_and_conflicting_reads_are_rejected(tmp_path):
    path = tmp_path/'image'; path.write_bytes(b'image one')
    record = {'path': str(path), 'sha256': sha256_file(path)}
    verify_input_records([record, record])
    path.write_bytes(b'image two')
    with pytest.raises(ValueError, match='changed during'):
        verify_input_records([record])
    with pytest.raises(ValueError, match='between reads'):
        verify_input_records([record, {'path': str(path), 'sha256': sha256_file(path)}])


def manifest(tmp_path):
    (tmp_path/'science.fits').write_bytes(b'science data')
    data = {'schema_version': 'fr2dynage-analysis-manifest-2', 'files': ['science.fits'],
        'sha256': {'science.fits': sha256_file(tmp_path/'science.fits')}}
    (tmp_path/'run_manifest.json').write_text(json.dumps(data))
    return data


def test_verify_outputs_detects_mutation_and_cli_nonzero_exit(tmp_path):
    from fr2dynage.cli import main
    manifest(tmp_path)
    assert verify_analysis_manifest(tmp_path)['output_files_verified'] == 1
    assert main(['verify-analysis', str(tmp_path)]) == 0
    (tmp_path/'science.fits').write_bytes(b'corrupted')
    assert main(['verify-analysis', str(tmp_path)]) == 2


@pytest.mark.parametrize('relative', ['../outside', '/tmp/outside', 'run_manifest.json'])
def test_manifest_cannot_reach_outside_or_hash_itself(tmp_path, relative):
    data = manifest(tmp_path)
    data.update(files=[relative], sha256={relative:'a'*64})
    (tmp_path/'run_manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='paths'):
        verify_analysis_manifest(tmp_path)


def test_inputs_changing_during_render_abort_publication(tmp_path, monkeypatch):
    import fr2dynage.source_analysis as analysis
    import fr2dynage.analysis_products as products
    source = tmp_path/'input.json'; source.write_text('{}')
    target = tmp_path/'results'
    document = {'schema_version':'fr2dynage-source-analysis-1', 'kind':'fr2_dynage'}
    monkeypatch.setattr(analysis, '_fr2_document', lambda *a,**k:(deepcopy(document), {source}, None))
    def writer(document, stage, **kwargs):
        (stage/'analysis.json').write_text(json.dumps(document))
        source.write_text('{"changed":true}')
        return {}
    monkeypatch.setattr(products, 'write_analysis_products', writer)
    with pytest.raises(ValueError, match='changed during analysis'):
        analysis.analyze_source(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob('.results.*'))
