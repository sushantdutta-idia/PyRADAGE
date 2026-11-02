"""Full-source spectral products preserve core fits, clocks and provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from fr2dynage.advanced_spectra import spectral_log_emissivity
from fr2dynage.maps_io import read_fits_plane
from fr2dynage.radio_inference import RadioInferenceResult
from fr2dynage.resolved_products import parse_resolved_settings, compute_resolved_products, write_resolved_products


@pytest.fixture
def saved_source(tmp_path):
    frequencies = [150., 600., 1400., 3000.]
    wcs = WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crpix=[2.,2.];wcs.wcs.crval=[53.,-27.];wcs.wcs.cdelt=[-.001,.001]
    header=wcs.to_header();header['BUNIT']='Jy/beam';header['BMAJ']=header['BMIN']=.003;header['BPA']=0.
    flux=np.exp(spectral_log_emissivity(frequencies,40.,2.5,.65,.1469,model='jp_tribble'))*.05
    records,images=[],[]
    for index, frequency in enumerate(frequencies):
        path=tmp_path/f'band{index}.fits'
        fits.PrimaryHDU(np.full((3,3),flux[index]),header=header).writeto(path)
        records.append({'path':str(path),'hdu':0,'role':f'images[{index}]','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        images.append({'path':path.name,'frequency_mhz':frequency,'rms_jy_per_beam':.0001,'calibration_fraction':.02})
    config={'images':images,'selection':{'snr_min':5,'min_bands':4}}
    source=np.ones((3,3),dtype=np.uint8);core=np.zeros_like(source);core[1,1]=1
    reference=read_fits_plane(tmp_path/'band0.fits')
    result=RadioInferenceResult(reference,{'SOURCE_MASK':source,'SOURCE_VALID':source,'CORE':core},
        {'source':{'name':'Synthetic core-inclusive spectral products','redshift':.1469},'configuration_path':str(tmp_path/'radio.json'),
         'input_configuration':config,'configuration_sha256':'synthetic','input_files':records},
        {'magnetic_field_ug':np.full(32,2.5),'alpha_inj':np.full(32,.65)},frozenset(Path(item['path']) for item in records))
    return result


def test_both_models_include_core_and_export_distinct_physical_clocks(saved_source,tmp_path):
    settings=parse_resolved_settings({})
    results=compute_resolved_products(saved_source,settings)
    assert set(results)=={'jp_tribble','ci'}
    folder=tmp_path/'output';folder.mkdir()
    summaries=write_resolved_products(results,saved_source,folder)
    for model in results:
        assert results[model]['maps']['SPEC_VALID'].sum()==9
        assert results[model]['metadata']['core_modelled_pixels']==1
        assert results[model]['metadata']['fixed_circular_exclusion'] is False
        assert np.isfinite(results[model]['maps']['SPECTRAL_AGE'][1,1])
        np.testing.assert_allclose(results[model]['maps']['SPECTRAL_AGE'],results[model]['maps']['AGE_B16'])
        with fits.open(folder/f'resolved_{model}.fits',checksum=True) as hdus:
            assert hdus[0].header['AGEKIND']==('CI_DURATION' if model=='ci' else 'RADIATIVE')
            assert hdus[0].header['COREPOL']=='INCLUDE'
            assert hdus[0].header['BFRACEQ']==.4
            assert hdus['REF_INTENSITY'].header['BUNIT']=='Jy/beam'
            assert hdus['AGE_FLAGS'].header['BUNIT']=='1'
            for hdu in hdus:
                assert hdu.verify_checksum()==1
                assert hdu.verify_datasum()==1
        assert (folder/f'resolved_{model}.pdf').read_bytes().startswith(b'%PDF-')
        assert '<svg' in (folder/f'resolved_{model}.svg').read_text()
        json.dumps(summaries[model],allow_nan=False)


def test_changed_image_rejected_between_transport_and_spectral_fits(saved_source):
    path=Path(saved_source.metadata['input_files'][0]['path'])
    with fits.open(path,mode='update') as hdus:
        hdus[0].data[1,1]*=2
    with pytest.raises(ValueError,match='image changed'):
        compute_resolved_products(saved_source,parse_resolved_settings({}))


@pytest.mark.parametrize('settings',[{'models':['jp']},{'models':['ci','ci']},{'models':[[]]},
    {'age_bounds_myr':[True,100.]},{'age_bounds_myr':[1.,0.]},{'age_grid_size':True},
    {'alpha_inj':True},{'alpha_inj':.3},{'alpha_inj':'unknown'},{'unknown':1}])
def test_invalid_resolved_settings_fail_before_any_fit(settings):
    with pytest.raises(ValueError):
        parse_resolved_settings(settings)


@pytest.mark.parametrize('value', [True, 'bad', [.6], [.7,.6], [.4,.7], [.5,True], [[],.7]])
def test_invalid_alpha_sensitivity(value):
    with pytest.raises(ValueError, match='alpha_sensitivity'):
        parse_resolved_settings({'alpha_sensitivity':value})


def test_alpha_refits_match_independent_fits_and_keep_core(saved_source):
    from fr2dynage.resolved_spectra import fit_resolved_spectra
    settings = parse_resolved_settings({'models':['jp_tribble'], 'alpha_sensitivity':[.55,.75]})
    result = compute_resolved_products(saved_source, settings)['jp_tribble']
    cube = np.stack([fits.getdata(path['path']) for path in saved_source.metadata['input_files']])
    for alpha, label in [(.55,'ALOW'),(.75,'AHIGH')]:
        independent = fit_resolved_spectra([150.,600.,1400.,3000.],cube,.0001,.02,np.eye(4),
            np.ones((3,3)),2.5,.1469,alpha,age_grid_size=241)
        np.testing.assert_allclose(result['maps']['AGE_'+label], independent['maps']['SPECTRAL_AGE'])
        np.testing.assert_equal(result['maps']['FLAG_'+label], independent['maps']['AGE_FLAGS'])
        assert np.isfinite(result['maps']['AGE_'+label][1,1])
    assert result['metadata']['whole_detected_source_summary']['mean_myr'] is None
    assert result['metadata']['core_modelled_pixels'] == 1
    assert not np.any(result['maps']['AGE_SCENARIO_OK'])


def test_unassessed_sensitivity_never_creates_scenario_support(saved_source):
    result = compute_resolved_products(saved_source, parse_resolved_settings({
        'models':['jp_tribble'], 'alpha_sensitivity':None}))['jp_tribble']
    assert not np.any(result['maps']['AGE_SCENARIO_OK'])
    assert result['metadata']['whole_detected_source_summary']['status'] == 'withheld'


def test_sensitivity_endpoints_must_bracket_nominal_before_fitting(saved_source, monkeypatch):
    import fr2dynage.resolved_spectra as spectra
    monkeypatch.setattr(spectra,'fit_resolved_spectra',lambda *a,**kw:pytest.fail('must reject before fit'))
    with pytest.raises(ValueError, match='bracket'):
        compute_resolved_products(saved_source,parse_resolved_settings({'alpha_sensitivity':[.7,.8]}))


@pytest.mark.parametrize('model',['jp_tribble','ci'])
def test_known_curved_source_passes_scenarios_and_recovers_conditional_mean(saved_source,model):
    frequencies = [80.,350.,1400.,10000.]
    brightness = .03*np.exp(spectral_log_emissivity(frequencies,50.,2.5,.65,.1469,model=model))
    for frequency, value, record, image in zip(frequencies,brightness,
            saved_source.metadata['input_files'],saved_source.metadata['input_configuration']['images']):
        fits.PrimaryHDU(np.full((3,3),value),header=saved_source.reference.header).writeto(record['path'],overwrite=True)
        record['sha256'] = hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
        image.update(frequency_mhz=frequency,rms_jy_per_beam=.01*value,calibration_fraction=.01)
    result = compute_resolved_products(saved_source,parse_resolved_settings({
        'models':[model], 'alpha_sensitivity':[.649,.651]}))[model]
    assert np.all(result['maps']['AGE_SCENARIO_OK'])
    assert result['metadata']['whole_detected_source_summary']['status'] == 'conditional_scenario_supported'
    assert result['metadata']['whole_detected_source_summary']['mean_myr'] == pytest.approx(50.,rel=.002)
    assert result['metadata']['core_modelled_pixels'] == 1


def test_fixed_field_products_preserve_native_beq_and_omit_artificial_field_errors(saved_source,tmp_path):
    saved_source.metadata['field_prescription'] = {'kind':'user_fixed','value_t':2.5e-10,
        'value_ug':2.5,'fraction':None,'uncertainty':'conditioned exactly on the supplied field'}
    saved_source.metadata['pyrats_properties'] = {'properties':{
        'native_field_budgets':{'equipartition':{'B_uG':6.}}}}
    result = compute_resolved_products(saved_source,parse_resolved_settings({'models':['jp_tribble']}))
    metadata = result['jp_tribble']['metadata']
    assert metadata['field_fraction'] is None
    assert metadata['equipartition_field_ug'] == 6.
    assert metadata['equipartition_field_ug'] != 2.5/.4
    assert metadata['nominal_B_over_Beq'] == pytest.approx(2.5/6.)
    assert metadata['field_sensitivity'] == {}
    assert 'AGE_B16' not in result['jp_tribble']['maps']
    assert 'AGE_B84' not in result['jp_tribble']['maps']
    folder=tmp_path/'fixed-output';folder.mkdir()
    write_resolved_products(result,saved_source,folder)
    with fits.open(folder/'resolved_jp_tribble.fits') as hdus:
        assert 'BFRACEQ' not in hdus[0].header
        assert hdus[0].header['BINPUT']=='USERFIX'
        assert hdus[0].header['BFIX_T']==2.5e-10
        assert hdus[0].header['B_UG']==2.5
        assert hdus['SPEC_VALID'].data[1,1]==1
