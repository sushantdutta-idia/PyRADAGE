"""A user-supplied B is neither silently rescaled nor replaced by 0.4 B_eq."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from fr2dynage.constants import MU_0
from fr2dynage.reliability import validate_adopted_budget, radio_parameter_quality
from fr2dynage.source_analysis import _radio_document


def fixed_budget():
    report = {'self_consistent_solution': {'equipartition': {
        'inputs': {'volume_m3': 1e65, 'kappa_proton_electron_energy_ratio': 100.},
        'direct': {'equipartition': {'B_uG': 6.}}}}}
    field = 2.24e-10
    electron = 6e-15
    particle = 101*electron
    magnetic = field**2/(2*MU_0)
    total = particle+magnetic
    document = {'budget': {'fraction_of_B_eq': 2.24/6., 'B_uG': 2.24, 'B_T':field,
        'B_eq_uG':6., 'B_eq_T':6e-10, 'electron_energy_density_J_m3':electron,
        'particle_energy_density_J_m3':particle, 'magnetic_energy_density_J_m3':magnetic,
        'total_energy_density_J_m3':total, 'total_energy_J':total*1e65, 'pressure_Pa':total/3}}
    return document, report


def test_explicit_fixed_field_budget_keeps_native_eq_field_and_true_ratio():
    document, report = fixed_budget()
    result = validate_adopted_budget(document, report, fixed_field_t=2.24e-10)
    assert result['status'] == 'accounting_verified_conditional_budget'
    assert result['adopted_field_t'] == 2.24e-10
    assert result['adopted_field_ug'] == 2.24
    assert result['fraction_of_nominal_B_eq'] == pytest.approx(2.24/6.)
    assert 'only as a comparison' in result['field_basis']
    assert document['budget']['B_eq_uG'] == 6.
    with pytest.raises(ValueError, match='fixed B/B_eq'):
        validate_adopted_budget(document,report)


@pytest.mark.parametrize('change', [
    lambda d:d['budget'].update(fraction_of_B_eq=.4),
    lambda d:d['budget'].update(B_eq_uG=5.6, B_eq_T=5.6e-10, fraction_of_B_eq=.4),
    lambda d:d['budget'].update(B_T=2.24000224e-10),
    lambda d:d['budget'].update(total_energy_J=1e99),
    lambda d:d['budget'].update(pressure_Pa=1e99)])
def test_fixed_input_does_not_relax_budget_physics_or_change_the_eq_reference(change):
    document, report = fixed_budget()
    change(document)
    with pytest.raises(ValueError,match='budget'):
        validate_adopted_budget(document,report,fixed_field_t=2.24e-10)


@pytest.mark.parametrize('field',[True,0.,-1.,float('nan'),float('inf'),1e-20,1.])
def test_invalid_fixed_field_value_rejected(field):
    document, report = fixed_budget()
    with pytest.raises(ValueError,match='fixed field'):
        validate_adopted_budget(document,report,fixed_field_t=field)


def test_user_fixed_quality_does_not_call_it_a_pyrats_field_measurement():
    metadata = {'field_prescription':{'kind':'user_fixed','value_t':2.24e-10},
        'posterior_parameters':{'magnetic_field_ug':{'median':2.24}},
        'fit':{'adequate_at_threshold_4':False}}
    row = radio_parameter_quality(metadata)['parameters']['magnetic_field_ug']
    assert row['status'] == 'user_fixed_input'
    assert row['estimate']['median'] == 2.24
    assert 'user-supplied fixed' in row['reasons'][0]
    assert 'no uncertainty' in row['reasons'][0]


def _files(tmp_path):
    document, report = fixed_budget()
    report_path = tmp_path/'PyRATS.json'; report_path.write_text(json.dumps(report))
    document['input_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    budget_path = tmp_path/'budget.json'; budget_path.write_text(json.dumps(document))
    config = {'pyrats_properties':{'path':report_path.name},
        'fixed_magnetic_field':{'value_t':2.24e-10,'description':'Explicit user request'}}
    path = tmp_path/'radio.json'; path.write_text(json.dumps(config))
    return path,budget_path,config,report_path


def _result(path, report_path, *, field=2.24, declaration=2.24e-10):
    metadata = {'source':{'name':'Synthetic user-fixed source'},
        'field_prescription':{'kind':'user_fixed','value_t':declaration,'value_ug':declaration*1e10},
        'branches':{}, 'clock':{'model':{'kind':'steady_jp_tribble_advection'}},
        'pyrats_properties':{'provenance':{'sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()}},
        'fitted_regional_spectra':{}, 'status':'curvature_unconstrained',
        'sampling':{'status':'replicate_consistent'}, 'warnings':[],
        'configuration_path':str(path),'configuration_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'input_files':[]}
    return SimpleNamespace(metadata=metadata,samples={'magnetic_field_ug':np.full(8,field)},
        input_paths=frozenset({path,report_path}),resolved_results=None)


@pytest.mark.parametrize('field',[2.24,2.24e-10/1e-10])
def test_source_document_accepts_valid_fixed_budget_and_preserves_clock_scope(tmp_path,monkeypatch,field):
    import fr2dynage.radio_inference as radio
    path,budget_path,config,report_path = _files(tmp_path)
    result = _result(path,report_path,field=field)
    monkeypatch.setattr(radio,'compute_radio_inference',lambda *a:result)
    document,protected,_ = _radio_document(path,budget_path=budget_path)
    assert document['field_constraint']['value_t'] == 2.24e-10
    assert document['parameter_quality']['source_budget']['adopted_field_t'] == 2.24e-10
    assert document['parameter_quality']['global_dynamical_age_myr']['mean'] is None
    assert budget_path in protected


def test_wrong_fixed_budget_rejected_before_any_expensive_fit(tmp_path,monkeypatch):
    import fr2dynage.radio_inference as radio
    path,budget_path,config,_ = _files(tmp_path)
    config['fixed_magnetic_field']['value_t'] = 2.4e-10
    path.write_text(json.dumps(config))
    monkeypatch.setattr(radio,'compute_radio_inference',lambda *a:pytest.fail('wrong fixed budget reached inference'))
    with pytest.raises(ValueError,match='requested fixed magnetic field'):
        _radio_document(path,budget_path=budget_path)


@pytest.mark.parametrize('field,declaration',[(2.3,2.24e-10),(2.24,2.4e-10)])
def test_source_document_rejects_sampler_or_metadata_ignoring_fixed_input(tmp_path,monkeypatch,field,declaration):
    import fr2dynage.radio_inference as radio
    path,budget_path,_,report_path = _files(tmp_path)
    monkeypatch.setattr(radio,'compute_radio_inference',lambda *a:_result(path,report_path,field=field,declaration=declaration))
    with pytest.raises(ValueError,match='preserve.*fixed magnetic field'):
        _radio_document(path,budget_path=budget_path)


def test_even_tiny_field_prior_scatter_is_not_an_exact_fixed_input(tmp_path,monkeypatch):
    import fr2dynage.radio_inference as radio
    path,budget_path,_,report_path = _files(tmp_path)
    result = _result(path,report_path)
    result.samples['magnetic_field_ug'] += np.linspace(-1e-13,1e-13,8)
    monkeypatch.setattr(radio,'compute_radio_inference',lambda *a:result)
    with pytest.raises(ValueError,match='exact fixed magnetic field'):
        _radio_document(path,budget_path=budget_path)
