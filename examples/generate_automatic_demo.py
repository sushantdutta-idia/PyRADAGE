"""Generate a self-contained GUI demonstration; these are not observations."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter


def generate(output, *, seed=731):
    from astropy.io import fits
    from astropy.wcs import WCS
    from fr2dynage.advanced_spectra import spectral_log_emissivity
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frequencies=[150.,350.,650.,1400.,3000.,6000.]
    yy,xx=np.indices((64,64),dtype=float)
    amplitude=.03*np.exp(-.5*((xx-32)/12.)**2-.5*((yy-32)/7.)**2)
    age=10.+1.15*xx
    wcs=WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crval=[53.,-27.];wcs.wcs.crpix=[32.,32.];wcs.wcs.cdelt=[-3/3600.,3/3600.]
    header=wcs.to_header();header['BUNIT']='Jy/beam';header['BMAJ']=header['BMIN']=7.5/3600.;header['BPA']=0.
    header['OBJECT']='SYNTHETIC GUI DEMONSTRATION'
    rng=np.random.default_rng(seed)
    sigma=2.5/np.sqrt(8*np.log(2))
    for frequency in frequencies:
        model=amplitude*np.exp(spectral_log_emissivity([frequency],age,3.,.65,.15,model='jp_tribble')[...,0])
        model=gaussian_filter(model,sigma)
        noise=gaussian_filter(rng.normal(size=model.shape),sigma)
        noise*=3e-5/noise.std()
        current=header.copy();current['FREQ']=frequency*1e6;current['FREQUNIT']='Hz'
        fits.PrimaryHDU(model+noise,header=current).writeto(output/f'demo_{int(frequency)}MHz.fits',checksum=True)
    truth={'origin':'synthetic, not observed','redshift':.15,'field_t':3e-10,'alpha_inj':.65,
        'age_equation_myr':'10+1.15*x before Gaussian beam convolution',
        'frequencies_mhz':frequencies,'seed':seed,'global_dynamical_age':None}
    (output/'truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    return truth


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(generate(parser.parse_args().output),indent=2))
