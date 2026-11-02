"""Exact conditioned fields remove a sampler dimension without inventing B_eq."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('astropy')
from astropy.io import fits

from fr2dynage.radio_inference import compute_radio_inference, write_radio_products


@pytest.fixture(scope='module')
def fixed_demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp('fixed-field-radio')
    script = Path(__file__).resolve().parents[1]/'examples/generate_radio_demo.py'
    specification = importlib.util.spec_from_file_location('fixed_field_demo',script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    truth = module.generate(folder,particles=128)
    config = json.loads((folder/'radio.json').read_text())
    config['priors'].pop('magnetic_field_ug')
    config['fixed_magnetic_field'] = {'value_t':3e-10,
        'description':'Declared exact generating field for a synthetic numerical recovery check.'}
    config['inference'].update(replicates=2,mcmc_steps=8)
    (folder/'fixed-radio.json').write_text(json.dumps(config,indent=2)+'\n')
    return folder, truth, config


@pytest.fixture(scope='module')
def fixed_result(fixed_demo):
    return compute_radio_inference(fixed_demo[0]/'fixed-radio.json')


def config_file(fixed_demo, tmp_path, mutate):
    folder, _, original = fixed_demo
    config = json.loads(json.dumps(original))
    for record in config['images']:
        record['path'] = str(folder/record['path'])
    config['selection']['mask'] = str(folder/config['selection']['mask'])
    for key in ['distance_map','branch_map']:
        config['geometry'][key] = str(folder/config['geometry'][key])
    mutate(config)
    path = tmp_path/'radio.json'
    path.write_text(json.dumps(config))
    return path


def test_fixed_field_is_exact_and_is_not_a_free_or_equipartition_parameter(fixed_result):
    result = fixed_result
    np.testing.assert_array_equal(result.samples['magnetic_field_ug'],
                                  np.full(256,3.))
    assert result.metadata['fixed_parameters'] == {'magnetic_field_ug':3.}
    assert 'magnetic_field_ug' not in result.metadata['free_parameters']
    assert 'magnetic_field_ug' not in result.metadata['priors']
    assert 'equipartition_field_ug' not in result.samples
    prescription = result.metadata['field_prescription']
    assert prescription['kind'] == 'user_fixed'
    assert prescription['value_t'] == 3e-10
    assert prescription['value_ug'] == 3.
    field = result.metadata['posterior_parameters']['magnetic_field_ug']
    assert field['mean'] == field['median'] == 3.
    assert field['credible_interval_95'] == [3.,3.]
    assert len(result.metadata['sampling']['runs']) == 2
    assert all(run['stages'][-1]['beta'] == 1. for run in result.metadata['sampling']['runs'])


def test_fixed_parameter_not_counted_in_fit_degrees_of_freedom(fixed_result):
    regions = fixed_result.metadata['regions']
    free = fixed_result.metadata['free_parameters']
    assert len(free) == 3  # Two branch speeds and shared injection index.
    expected = sum(region['bands'] for region in regions)-len(regions)-len(free)
    assert fixed_result.metadata['fit']['residual_dof_after_shared_parameters'] == expected


def test_fixed_known_generating_field_recovers_transport_clock(fixed_demo,fixed_result):
    folder, truth, _ = fixed_demo
    assert fixed_result.metadata['fit']['adequate_at_threshold_4']
    for label, expected in truth['flow_speed_km_s_by_branch'].items():
        recovered = fixed_result.metadata['posterior_parameters'][f'flow_speed_km_s_branch_{label}']
        # Data are fully Gaussian-convolved; the fit uses finite path
        # quadrature and uniform local emissivity, so exact equality is not
        # a physically meaningful test of this approximation.
        assert recovered['median'] == pytest.approx(expected,rel=.15)
    age = fits.getdata(folder/'truth-age.fits')
    selected = fixed_result.maps['AGE_VALID'].astype(bool)
    actual_mean = np.mean(age[selected])
    recovered_mean = fixed_result.metadata['map_statistics']['area_weighted_mean']['median']
    assert recovered_mean == pytest.approx(actual_mean,rel=.15)
    assert fixed_result.metadata['global_dynamical_age_myr']['mean'] is None
    assert fixed_result.metadata['global_dynamical_age_myr']['median'] is None


def test_fixed_field_fits_headers_do_not_claim_point_four_equipartition(fixed_result,tmp_path):
    output, summary = tmp_path/'transport.fits', tmp_path/'summary.json'
    write_radio_products(fixed_result,output,summary)
    with fits.open(output,checksum=True) as hdus:
        assert hdus[0].header['BINPUT'] == 'USERFIX'
        assert hdus[0].header['BFIX_T'] == 3e-10
        assert hdus[0].header['BFIX_UG'] == 3.
        assert 'BFRACEQ' not in hdus[0].header
        for hdu in hdus:
            assert hdu.verify_checksum() == hdu.verify_datasum() == 1
        encoded = json.loads(bytes(hdus['PROVENANCE'].data['JSON_UTF8'][0]))
        assert encoded == json.loads(summary.read_text())
        assert encoded['fixed_parameters']['magnetic_field_ug'] == 3.


def test_fixed_field_with_sampled_deceleration_counts_only_free_dimensions(fixed_demo,tmp_path):
    def modify(config):
        config['model'].pop('deceleration')
        config['priors']['deceleration'] = {'distribution':'uniform','low':0.,'high':1.}
        config['inference'].update(particles=128,replicates=2)
    result = compute_radio_inference(config_file(fixed_demo,tmp_path,modify))
    assert len(result.metadata['free_parameters']) == 4
    assert 'deceleration' in result.metadata['free_parameters']
    assert np.all(result.samples['magnetic_field_ug'] == 3.)
    assert np.all((result.samples['deceleration'] >= 0.) & (result.samples['deceleration'] <= 1.))
    assert result.metadata['fit']['adequate_at_threshold_4']
    for label, expected in fixed_demo[1]['flow_speed_km_s_by_branch'].items():
        assert result.metadata['posterior_parameters'][f'flow_speed_km_s_branch_{label}']['median'] == pytest.approx(expected,rel=.2)


@pytest.mark.parametrize('value', [True,False,0.,-3e-10,1e-15,1e-4,'3e-10',None])
def test_invalid_fixed_field_rejected_before_sampling(fixed_demo,tmp_path,monkeypatch,value):
    import fr2dynage.radio_inference as radio
    path = config_file(fixed_demo,tmp_path,
        lambda config:config['fixed_magnetic_field'].update(value_t=value))
    monkeypatch.setattr(radio,'tempered_smc',lambda *a,**k:pytest.fail('bad fixed field reached sampler'))
    with pytest.raises(ValueError,match='fixed_magnetic_field|field|value_t'):
        compute_radio_inference(path)


def test_fixed_field_rejects_a_second_explicit_field_prior(fixed_demo,tmp_path,monkeypatch):
    import fr2dynage.radio_inference as radio
    path = config_file(fixed_demo,tmp_path,lambda config:config['priors'].update(
        magnetic_field_ug={'distribution':'uniform','low':2.,'high':4.}))
    monkeypatch.setattr(radio,'tempered_smc',lambda *a,**k:pytest.fail('conflicting field reached sampler'))
    with pytest.raises(ValueError,match='fixed|field|prior'):
        compute_radio_inference(path)


def test_maximum_supported_move_count_is_valid_for_fixed_field(fixed_demo,tmp_path,monkeypatch):
    import fr2dynage.radio_inference as radio
    actual_sampler = radio.tempered_smc
    def cheap_sampler(likelihood, dimension, **options):
        # Exercise the actual sampler's argument validator and move count
        # cheaply; scientific recovery is tested against the real likelihood
        # in the shared fixed_result fixture above.
        return actual_sampler(lambda row:0.,dimension,**options)
    monkeypatch.setattr(radio,'tempered_smc',cheap_sampler)
    path = config_file(fixed_demo,tmp_path,
        lambda config:config['inference'].update(mcmc_steps=40))
    result = compute_radio_inference(path)
    assert np.all(result.samples['magnetic_field_ug'] == 3.)


def pyrats_fixture(config, folder):
    return {'pyrats_version':'2.1.0','configuration':{'redshift':.1469},
        'input_maps':[{'label':Path(record['path']).stem,'path':'raw-'+Path(record['path']).name,
                       'frequency_Hz':record['frequency_mhz']*1e6} for record in config['images']],
        'geometry_physical':{'volume_m3':1e65,'filling_factor':1.},
        'self_consistent_solution':{'converged':True,'electron_spectrum':'powerlaw',
            'alpha_inj_positive':.65,'alpha_inj_err':.02,'alpha_source':'synthetic_fixture',
            'equipartition':{
                'inputs':{'redshift':.1469,'kappa_proton_electron_energy_ratio':100.,
                    'zeta_uB_over_uparticles':101.,'filling_factor':1.,
                    'gamma_min':10.,'gamma_max':1e7,'volume_m3':1e65},
                'direct':{'equipartition':{'B_uG':6.,'B_T':6e-10,'volume_m3':1e65},
                          'minimum_energy':{'B_uG':5.5,'B_T':5.5e-10,'volume_m3':1e65}},
                'monte_carlo':{'status':'ok','n_success':1000,'B_eq_uG':{'median':6.3,'p16':5.6,'p84':7.1},
                               'B_me_uG':{'median':5.8,'p16':5.2,'p84':6.5}}}}}


def test_pyrats_provenance_does_not_turn_fixed_field_into_equipartition_draws(fixed_demo,tmp_path):
    def modify(config):
        report = tmp_path/'PyRATS.json'
        report.write_text(json.dumps(pyrats_fixture(config,tmp_path)))
        config['pyrats_properties'] = {'path':str(report),'systematic_log_sigma':0.}
        config['inference'].update(particles=128,replicates=2)
    result = compute_radio_inference(config_file(fixed_demo,tmp_path,modify))
    assert np.all(result.samples['magnetic_field_ug'] == 3.)
    assert 'equipartition_field_ug' not in result.samples
    assert result.metadata['pyrats_properties']['properties']['field_direct_uG'] == 6.
    assert result.metadata['pyrats_properties']['provenance']['field_scale_assumption'] == 1.
    assert result.metadata['pyrats_properties']['field_constraint_applied'] is False
    assert result.metadata['fixed_parameters'] == {'magnetic_field_ug':3.}


def test_fixed_field_and_pyrats_scale_cannot_compete(fixed_demo,tmp_path,monkeypatch):
    import fr2dynage.radio_inference as radio
    path = config_file(fixed_demo,tmp_path,lambda config:config.update(
        pyrats_properties={'path':'not-read.json','systematic_log_sigma':0.,'field_scale':.4}))
    monkeypatch.setattr(radio,'tempered_smc',lambda *a,**k:pytest.fail('conflicting scale reached sampler'))
    with pytest.raises(ValueError,match='fixed|field_scale|conflict'):
        compute_radio_inference(path)
