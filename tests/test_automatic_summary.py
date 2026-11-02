"""A single title-free summary keeps distinct clocks, source pixels and flags."""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip('astropy')
pytest.importorskip('matplotlib')
from astropy.io import fits
from astropy.wcs import WCS

from fr2dynage.maps_io import read_fits_plane
from fr2dynage.automatic_summary import render_automatic_summary


@pytest.fixture
def summary_case(tmp_path):
    n=48
    yy,xx=np.indices((n,n))
    intensity=.03*np.exp(-((xx-23.)**2+(yy-24.)**2)/130.)
    mask=(xx-23.)**2+(yy-24.)**2<20.**2
    wcs=WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crpix=[24.,24.];wcs.wcs.crval=[53.,-27.];wcs.wcs.cdelt=[-.001,.001]
    header=wcs.to_header();header['BUNIT']='Jy/beam';header['BMAJ']=.005;header['BMIN']=.003;header['BPA']=35.
    image=tmp_path/'radio.fits';fits.PrimaryHDU(intensity,header=header).writeto(image)
    reference=read_fits_plane(image)
    valid=mask.copy();valid[24,24]=False
    flags=np.where(mask,32,1).astype(np.uint16)
    flags[24,24]=2
    flags[25,24]=0
    flags[26,24]=64
    fitted={}
    for model,factor in [('jp_tribble',1.),('ci',4.)]:
        age=(20.+xx)*factor
        fitted[model]={'maps':{'SPECTRAL_AGE':np.where(valid,age,np.nan),
            'AGE_UPPER95':np.where(valid,age+10.,np.nan),
            'REDCHI2':np.where(valid,1.,np.nan),
            'SPEC_VALID':valid.astype(np.uint8),
            'AGE_CONSTRAINED':(flags==0).astype(np.uint8),'AGE_FLAGS':flags.copy()},
            'metadata':{'source':{'name':'Synthetic source with an included central region'}}}
    return reference,intensity,fitted,mask


def test_summary_exports_one_combined_figure_with_distinct_clocks_and_no_titles(summary_case,tmp_path,monkeypatch):
    import fr2dynage.automatic_summary as module
    reference,intensity,fitted,mask=summary_case
    before=deepcopy(fitted)
    captured=[]
    original=module.save_publication_figure
    def capture(figure,*args,**kwargs):
        sky=[axis for axis in figure.axes if hasattr(axis,'coords')]
        assert len(sky)==6
        width,height=figure.get_size_inches()
        assert height>width
        assert len({round(axis.get_position().x0,5) for axis in sky})==2
        assert len({round(axis.get_position().y0,5) for axis in sky})==3
        assert all(not axis.title.get_text() for axis in sky)
        classes=np.asarray(sky[-1].images[0].get_array())
        assert classes[24,24]==1
        assert classes[25,24]==3
        assert classes[26,24]==4
        assert classes[24,25]==2
        np.testing.assert_allclose(sky[1].images[0].get_array()[24,25],45.)
        np.testing.assert_allclose(sky[2].images[0].get_array()[24,25],180.)
        captured.append(True)
        return original(figure,*args,**kwargs)
    monkeypatch.setattr(module,'save_publication_figure',capture)
    products=render_automatic_summary(reference,intensity,fitted,tmp_path/'age_summary.png',source_mask=mask)
    assert captured
    assert set(products)=={'png','pdf','svg'}
    assert all(path.is_file() for path in products.values())
    assert products['pdf'].read_bytes().startswith(b'%PDF-')
    svg=ElementTree.parse(products['svg'])
    text=' '.join(''.join(node.itertext()) for node in svg.findall('.//{http://www.w3.org/2000/svg}text'))
    assert 'JP–Tribble time (Myr)' in text
    assert 'CI injection duration (Myr)' in text
    assert '(a)' in text and '(f)' in text
    assert 'Synthetic source' not in text
    assert 'Conditional' in text and 'Poor fit' in text
    for model in fitted:
        for name,values in fitted[model]['maps'].items():
            np.testing.assert_equal(values,before[model]['maps'][name])


def test_summary_preserves_existing_companion_before_any_write(summary_case,tmp_path):
    reference,intensity,fitted,mask=summary_case
    output=tmp_path/'existing'
    output.with_suffix('.pdf').write_bytes(b'user paper')
    with pytest.raises(FileExistsError):
        render_automatic_summary(reference,intensity,fitted,output,source_mask=mask)
    assert output.with_suffix('.pdf').read_bytes()==b'user paper'
    assert not output.with_suffix('.png').exists()
    assert not output.with_suffix('.svg').exists()
    assert not list(tmp_path.glob('.age-summary-*'))


def test_summary_rolls_back_earlier_formats_if_a_destination_appears(summary_case,tmp_path,monkeypatch):
    import fr2dynage.automatic_summary as module
    reference,intensity,fitted,mask=summary_case
    output=tmp_path/'race'
    link=module.os.link
    def race(source,target):
        if Path(target).suffix=='.pdf':
            Path(target).write_bytes(b'concurrent user file')
            raise FileExistsError('destination appeared')
        return link(source,target)
    monkeypatch.setattr(module.os,'link',race)
    with pytest.raises(FileExistsError):
        render_automatic_summary(reference,intensity,fitted,output,source_mask=mask)
    assert output.with_suffix('.pdf').read_bytes()==b'concurrent user file'
    assert not output.with_suffix('.png').exists()
    assert not output.with_suffix('.svg').exists()
    assert not list(tmp_path.glob('.age-summary-*'))


def test_all_open_profiles_remain_empty_without_invented_upper_limits(summary_case,tmp_path):
    reference,intensity,fitted,mask=summary_case
    fitted['jp_tribble']['maps']['AGE_UPPER95'][:]=np.nan
    products=render_automatic_summary(reference,intensity,fitted,tmp_path/'open',source_mask=mask)
    assert 'No finite upper limits' in products['svg'].read_text()
    assert np.isnan(fitted['jp_tribble']['maps']['AGE_UPPER95']).all()


@pytest.mark.parametrize('mutate',[
    lambda fitted,mask:fitted.pop('ci'),
    lambda fitted,mask:fitted['jp_tribble']['maps'].pop('AGE_FLAGS'),
    lambda fitted,mask:fitted['jp_tribble']['maps'].update(SPEC_VALID=np.full(mask.shape,2)),
    lambda fitted,mask:fitted['jp_tribble']['maps'].update(SPECTRAL_AGE=np.full(mask.shape,-1.)),
])
def test_invalid_summary_inputs_do_not_create_an_output_directory(summary_case,tmp_path,mutate):
    reference,intensity,fitted,mask=summary_case
    mutate(fitted,mask)
    with pytest.raises(ValueError):
        render_automatic_summary(reference,intensity,fitted,tmp_path/'new'/'summary',source_mask=mask)
    assert not (tmp_path/'new').exists()
