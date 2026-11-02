"""Cancellation is honoured per pixel and before atomic publication.

The controller tests stub expensive spectral/field calculations and renderers,
while using actual FITS, staging, provenance and publication. Independent
numerical tests cover their scientific calculations separately.
"""
import threading
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def test_resolved_cancel_stops_between_actual_pixel_refinements(monkeypatch):
    import fr2dynage.advanced_spectra as kernels
    from fr2dynage.resolved_spectra import fit_resolved_spectra
    frequencies=np.array([80.,150.,350.,650.,1400.,3000.,10000.])
    original=kernels.spectral_log_emissivity
    brightness=.03*np.exp(original(frequencies,50.,2.4,.65,.15,model='jp_tribble'))
    cube=np.broadcast_to(brightness[:,None,None],(len(frequencies),2,2)).copy()
    before=cube.copy()
    event=threading.Event()
    calls=[]
    def cancel_after_first_pixel(*args,**kwargs):
        value=original(*args,**kwargs)
        calls.append(True)
        if len(calls)==2:
            event.set() # First call builds the common grid; second refines pixel one.
        return value
    monkeypatch.setattr(kernels,'spectral_log_emissivity',cancel_after_first_pixel)
    with pytest.raises(InterruptedError,match='cancelled'):
        fit_resolved_spectra(frequencies,cube,.01*cube,.01,np.eye(len(frequencies)),
            np.ones((2,2)),2.4,.15,.65,model='jp_tribble',age_grid_size=81,cancel_event=event)
    assert len(calls)==2
    np.testing.assert_array_equal(cube,before)


@pytest.fixture
def controller_case(tmp_path,monkeypatch):
    import fr2dynage.image_preparation as preparation
    import fr2dynage.automatic_field as field
    import fr2dynage.automatic_spectra as shared
    import fr2dynage.resolved_spectra as resolved
    import fr2dynage.automatic_summary as summary
    from fr2dynage.maps_io import FitsPlane
    from fr2dynage.reliability import sha256_file
    frequencies=np.array([150.,600.,1400.,3000.])
    shape=(6,6)
    wcs=WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crval=[53.,-27.];wcs.wcs.crpix=[3.,3.];wcs.wcs.cdelt=[-.001,.001]
    header=wcs.to_header();header['BUNIT']='Jy/beam';header['BMAJ']=header['BMIN']=.003;header['BPA']=0.
    images=[]
    for index,frequency in enumerate(frequencies):
        path=tmp_path/f'input{index}.fits'
        band_header=header.copy();band_header['FREQ']=float(frequency)*1e6
        fits.PrimaryHDU(np.ones(shape),header=band_header).writeto(path)
        images.append(path)
    reference=FitsPlane(np.ones(shape),header,wcs,images[0],'synthetic')
    geometry={'volume_m3':1e65,'volume_fractional_error':.5,
              'largest_linear_size_kpc':100.,'minor_axis_kpc':50.,
              'volume_estimator':'synthetic controller geometry assumption'}
    sed=[{'frequency_mhz':float(frequency),'flux_jy':.03*(float(frequency)/1000.)**-.65,
          'thermal_error_jy':.0001,'zero_level_error_jy':.00001,
          'calibration_error_jy':.0015,'total_error_jy':float(np.sqrt(.0001**2+.00001**2+.0015**2)),
          'scope':'synthetic controller photometry'} for frequency in frequencies]
    prepared=SimpleNamespace(reference=reference,frequencies_mhz=frequencies,
        flux_cube=np.ones((4,*shape)),rms_cube=np.full((4,*shape),.01),
        calibration_fractions=np.full(4,.05),calibration_correlation=np.eye(4),
        source_mask=np.ones(shape,dtype=bool),
        geometry=geometry,
        image_provenance=[{'path':str(path),'sha256':sha256_file(path)} for path in images],
        metadata={'integrated_sed':sed,'geometry':geometry,'warnings':[]})
    def prepare(*args,**kwargs):
        folder=Path(args[2]);folder.mkdir()
        (folder/'preparation.json').write_text(json.dumps(prepared.metadata))
        return prepared
    def fit_shared(*args,**kwargs):
        chi2=.3 if kwargs['model']=='jp_tribble' else .6
        return {'model':kwargs['model'],'alpha_inj':.65,'alpha_local_sigma':.03,
                'alpha_sensitivity':[.6,.7],'chi2':chi2,'degrees_of_freedom':3,
                'reduced_chi2':chi2/3,'flags':[],
                'status':'conditional_shared_index_fit'}
    def compute_field(*args,**kwargs):
        folder=Path(args[3]);folder.mkdir()
        result={'adopted_field_t':3e-10,'prescription':'user_fixed','field_mc':None,
                'adopted_budget':None,'scope':'Synthetic controller test field'}
        (folder/'field.json').write_text(json.dumps(result))
        return result
    def fit_pixels(*args,**kwargs):
        return {'maps':{'SPECTRAL_AGE':np.full(shape,50.),'SPEC_VALID':np.ones(shape,dtype=np.uint8),
            'AGE_FLAGS':np.full(shape,32,dtype=np.uint16),'AGE_CONSTRAINED':np.zeros(shape,dtype=np.uint8)},
            'metadata':{'model':kwargs['model'],'field_ug':3.,'alpha_inj_fixed':float(args[8]),
                        'detected_candidate_pixels':int(np.prod(shape)),'constrained_age_pixels':0}}
    def render(*args,**kwargs):
        stem=Path(args[3])
        for extension in ['png','pdf','svg']:
            stem.with_suffix('.'+extension).write_bytes(b'synthetic controller rendering')
    monkeypatch.setattr(preparation,'prepare_images',prepare)
    monkeypatch.setattr(field,'compute_automatic_field',compute_field)
    monkeypatch.setattr(shared,'fit_shared_injection',fit_shared)
    monkeypatch.setattr(resolved,'fit_resolved_spectra',fit_pixels)
    monkeypatch.setattr(summary,'render_automatic_summary',render)
    return images,tmp_path/'result'


def test_cancel_requested_by_precommit_progress_leaves_no_visible_analysis(controller_case):
    from fr2dynage.automatic import run_automatic
    images,target=controller_case
    before=[path.read_bytes() for path in images]
    event=threading.Event()
    def progress(message,fraction):
        if message=='Publishing the completed analysis':
            event.set()
    with pytest.raises(InterruptedError,match='cancelled'):
        run_automatic(images,.15,target,field_t=3e-10,progress=progress,cancel_event=event)
    assert not target.exists()
    assert not list(target.parent.glob('.result.*'))
    assert [path.read_bytes() for path in images]==before


def test_cancellation_after_committed_rename_does_not_discard_a_complete_result(controller_case,monkeypatch):
    import fr2dynage.automatic as workflow
    from fr2dynage.reliability import verify_analysis_manifest
    images,target=controller_case
    event=threading.Event();messages=[]
    rename=workflow.os.rename
    def commit(source,destination):
        result=rename(source,destination)
        if Path(destination)==target:
            event.set()
        return result
    monkeypatch.setattr(workflow.os,'rename',commit)
    result=workflow.run_automatic(images,.15,target,field_t=3e-10,cancel_event=event,
        progress=lambda message,fraction:messages.append((message,fraction)))
    assert Path(result['output_dir'])==target
    assert verify_analysis_manifest(target,check_inputs=True)['passed']
    assert messages[-1][1]==1.
    assert messages[-1][0].startswith('Completed')
    assert not list(target.parent.glob('.result.*'))
