"""Recover common injection shape without inventing paths or ages."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from astropy.wcs import WCS

from fr2dynage.advanced_spectra import spectral_log_emissivity
from fr2dynage.automatic_spectra import fit_shared_injection
from fr2dynage.maps_io import FitsPlane


@pytest.mark.parametrize('model',['jp_tribble','ci'])
def test_shared_index_and_independent_clocks_recover_known_shapes(tmp_path,model):
    freq=np.array([80.,150.,350.,650.,1400.,3000.,10000.])
    ages=np.array([20.,40.,60.,80.])
    truth=.03*np.exp(spectral_log_emissivity(freq,ages,3.,.65,.15,model=model))
    flux=np.zeros((len(freq),5,20));noise=np.full_like(flux,1e-5)
    mask=np.zeros((5,20),dtype=bool)
    for i,x in enumerate([2,7,12,17]):
        flux[:,2,x]=truth[i];noise[:,2,x]=.01*truth[i];mask[2,x]=True
    wcs=WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crval=[53.,-27.];wcs.wcs.crpix=[10.,3.];wcs.wcs.cdelt=[-.001,.001]
    header=wcs.to_header();header['BUNIT']='Jy/beam';header['BMAJ']=header['BMIN']=.002;header['BPA']=0.
    ref=FitsPlane(flux[0],header,wcs,tmp_path/'synthetic.fits','synthetic')
    prepared=SimpleNamespace(reference=ref,flux_cube=flux,rms_cube=noise,source_mask=mask,
        frequencies_mhz=freq,calibration_fractions=np.full(len(freq),.01),calibration_correlation=np.eye(len(freq)))
    fitted=fit_shared_injection(prepared,.15,model=model)
    assert fitted['alpha_inj']==pytest.approx(.65,abs=.004)
    np.testing.assert_allclose(fitted['regional_best_ages_myr'],ages,rtol=.015)
    assert fitted['reduced_chi2']<.01
    assert fitted['degrees_of_freedom']==4*len(freq)-2*4-1
    assert 'shared band-calibration covariance' in fitted['scope']


def test_cancelled_shape_fit_stops_before_consuming_data():
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):fit_shared_injection(None,.15,cancel_event=event)
