"""Independent surface-brightness, Gaussian-beam and frequency checks."""
from pathlib import Path
import json
import math

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip('astropy')
from astropy.io import fits
from astropy.wcs import WCS

from fr2dynage.image_preparation import (FrequencyMetadataError, _correlated_flux_error,
    _geometry, _jacobian, _local_transverse_volume, _resample, _smooth, image_frequency_mhz, prepare_images)
from fr2dynage.maps_io import MapInputError, read_fits_plane, science_image


def header(*, frequency=1000., pixels=64, beam=6., unit='Jy/beam', offset=0.):
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=[pixels/2+.5+offset,pixels/2+.5]
    wcs.wcs.crval=[53.,-27.]
    wcs.wcs.cdelt=[-1/3600,1/3600]
    wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    result=wcs.to_header();result['BMAJ']=result['BMIN']=beam/3600
    result['BPA']=0.;result['BUNIT']=unit
    if frequency is not None:result['FREQ']=frequency*1e6
    return result


def image(tmp_path,name='radio.fits',*,data=None,**kwargs):
    target=tmp_path/name
    fits.PrimaryHDU(np.zeros((64,64)) if data is None else data,header=header(**kwargs)).writeto(target)
    return target


def source_images(tmp_path, *, islands=False, metadata=True):
    rng=np.random.default_rng(501)
    frequencies=[150.,600.,1400.,3000.]
    yy,xx=np.indices((80,80));paths=[]
    expected=[]
    for index,(frequency,beam) in enumerate(zip(frequencies,[4.,5.,6.,6.])):
        total=.02*(frequency/1000.)**-.65
        expected.append(total)
        sigma=beam/math.sqrt(8*math.log(2))
        if islands:
            profile=.5*np.exp(-((xx-24.)**2+(yy-40.)**2)/(2*sigma**2))
            profile+=.5*np.exp(-((xx-56.)**2+(yy-40.)**2)/(2*sigma**2))
        else:
            profile=np.exp(-((xx-39.5)**2+(yy-39.5)**2)/(2*sigma**2))
        data=total*profile+rng.normal(0.,1e-5,(80,80))+.00003
        unit='mJy/beam' if index==1 else 'Jy/beam'
        if index==1:data*=1000
        target=image(tmp_path,f'band{index}.fits',data=data,
            frequency=frequency if metadata else None,pixels=80,beam=beam,unit=unit)
        paths.append(target)
    return paths,np.array(expected),frequencies


def test_observed_header_frequency_units_and_filename_independence(tmp_path):
    path=image(tmp_path,'not-the-150MHz-band.fits',frequency=1400.)
    assert image_frequency_mhz(path)==1400.
    with fits.open(path,mode='update') as hdus:
        hdus[0].header['FREQ']=1.4;hdus[0].header['FREQUNIT']='GHz'
    assert image_frequency_mhz(path)==1400.


def test_singleton_frequency_wcs_uses_actual_plane_not_reference_pixel(tmp_path):
    data=np.zeros((1,64,64));h=header(frequency=None)
    h['WCSAXES']=3;h['CTYPE3']='FREQ';h['CUNIT3']='Hz'
    h['CRVAL3']=1.4e9;h['CRPIX3']=2.;h['CDELT3']=1e6
    path=tmp_path/'cube.fits';fits.PrimaryHDU(data,header=h).writeto(path)
    assert image_frequency_mhz(path)==pytest.approx(1399.)


def test_rest_frequency_alone_is_not_an_observing_frequency(tmp_path):
    path=image(tmp_path,'radio-1400MHz.fits',frequency=None)
    with fits.open(path,mode='update') as hdus:hdus[0].header['RESTFRQ']=1.420405751e9
    with pytest.raises(FrequencyMetadataError,match='supply its observing frequency') as error:
        image_frequency_mhz(path)
    assert error.value.path==path


def test_conflicting_observed_frequency_metadata_requires_override(tmp_path):
    path=image(tmp_path,frequency=1000.)
    with fits.open(path,mode='update') as hdus:hdus[0].header['REFFREQ']=1.4e9
    with pytest.raises(FrequencyMetadataError,match='disagree') as error:
        image_frequency_mhz(path)
    assert set(error.value.candidates_mhz)=={'FREQ','REFFREQ'}


def test_coupled_spectral_coordinate_is_not_one_frequency(tmp_path):
    h=header(frequency=None);h['WCSAXES']=3;h['CTYPE3']='FREQ'
    h['CRVAL3']=1e9;h['CRPIX3']=1.;h['CDELT3']=1e6;h['PC3_1']=.1
    path=tmp_path/'coupled.fits';fits.PrimaryHDU(np.zeros((1,64,64)),header=h).writeto(path)
    with pytest.raises(FrequencyMetadataError,match='varies across'):
        image_frequency_mhz(path)


def test_bilinear_reprojection_preserves_constant_surface_brightness(tmp_path):
    source=WCS(header(offset=.4)).celestial
    reference=WCS(header()).celestial
    result,reprojected=_resample(np.full((64,64),7.3),source,reference,(64,64))
    assert reprojected
    np.testing.assert_allclose(result[np.isfinite(result)],7.3,rtol=1e-12)
    assert np.isfinite(result).sum()<64*64


def test_same_grid_never_interpolates_or_fills_a_nan():
    wcs=WCS(header()).celestial
    data=np.arange(64*64,dtype=float).reshape(64,64);data[30,30]=np.nan
    result,reprojected=_resample(data,wcs,wcs,data.shape)
    assert not reprojected
    np.testing.assert_equal(result,data)


def test_gaussian_common_beam_convolution_conserves_flux_and_covariance():
    yy,xx=np.indices((128,128))
    source_cov=np.array([[4.,1.],[1.,2.]])
    target_cov=np.eye(2)*9.
    displacement=np.stack([xx-63.5,yy-63.5],axis=-1)
    profile=np.exp(-.5*np.einsum('...i,ij,...j->...',displacement,np.linalg.inv(source_cov),displacement))
    profile/=profile.sum()
    smoothed=_smooth(profile,target_cov-source_cov)
    finite=np.isfinite(smoothed)
    assert np.sum(smoothed[finite])==pytest.approx(1.,rel=1e-7)
    normalized=np.where(finite,smoothed,0.)
    measured=np.einsum('xyi,xyj,xy->ij',displacement,displacement,normalized)
    np.testing.assert_allclose(measured,target_cov,atol=1e-6)


def test_common_beam_never_deconvolves():
    with pytest.raises(MapInputError,match='deconvolution'):
        _smooth(np.ones((20,20)),np.diag([1.,-1.]))


def test_nan_gaps_and_edges_are_not_normalized_into_full_beam_flux():
    data=np.ones((64,64));data[30:34,30:34]=np.nan
    result=_smooth(data,np.eye(2)*4.)
    assert np.isnan(result[31,31])
    assert np.isnan(result[0,0])
    assert result[15,15]==pytest.approx(1.,rel=1e-6)


def test_integrated_noise_accounts_for_correlated_pixels():
    mask=np.zeros((9,9),dtype=bool);mask[2:7,3:6]=True
    covariance=np.array([[2.,.2],[.2,1.]])
    rms=.003;pixels_per_beam=2*math.pi*math.sqrt(np.linalg.det(covariance))
    y,x=np.nonzero(mask);points=np.column_stack([x,y])
    delta=points[:,None,:]-points[None,:,:]
    rho=np.exp(-.25*np.einsum('...i,ij,...j->...',delta,np.linalg.inv(covariance),delta))
    expected=rms*math.sqrt(rho.sum())/pixels_per_beam
    actual=_correlated_flux_error(mask,rms,covariance,pixels_per_beam)
    assert actual==pytest.approx(expected,rel=1e-12)
    naive=rms*math.sqrt(mask.sum())/pixels_per_beam
    assert actual>naive*2


def test_preparation_preserves_flux_units_central_emission_and_common_beam(tmp_path):
    paths,expected,frequencies=source_images(tmp_path)
    output=tmp_path/'prepared'
    original=[path.read_bytes() for path in paths]
    messages=[]
    result=prepare_images(paths,.1469,output,progress=messages.append)
    np.testing.assert_array_equal(result.frequencies_mhz,frequencies)
    assert result.flux_cube.shape==(4,80,80)
    assert result.source_mask[40,40]
    assert result.metadata['fixed_circular_exclusion'] is False
    assert result.metadata['peak_is_not_automatically_identified_as_core'] is True
    assert result.geometry['volume_m3']>0
    assert result.geometry['volume_fractional_error']>=.5
    assert result.geometry_support_mask.sum()==result.metadata['geometry_support_pixels']
    assert np.all(result.source_mask[result.geometry_support_mask])
    assert result.rms_cube.shape==result.flux_cube.shape
    assert np.all(result.rms_cube>0)
    np.testing.assert_allclose([item['flux_jy'] for item in result.metadata['integrated_sed']],expected,rtol=.01)
    for record in result.image_records:
        plane=read_fits_plane(record['path']);_,beam=science_image(plane)
        assert beam['bmaj_deg']==pytest.approx(6/3600)
        assert beam['bmin_deg']==pytest.approx(6/3600)
        with fits.open(record['path'],checksum=True) as hdus:
            assert hdus[0].verify_checksum()==1
    assert original==[path.read_bytes() for path in paths]
    assert messages
    assert json.loads((output/'preparation.json').read_text())['redshift']==.1469


def test_disconnected_lobes_are_retained_instead_of_selecting_brightest_island(tmp_path):
    paths,_,_=source_images(tmp_path,islands=True)
    result=prepare_images(paths,.1469,tmp_path/'prepared')
    assert result.source_mask[40,24] and result.source_mask[40,56]
    assert result.metadata['source_islands_retained']==2
    assert any('unrelated field sources' in text for text in result.metadata['warnings'])


def test_explicit_frequency_values_allow_missing_headers_without_filename_guesses(tmp_path):
    paths,_,frequencies=source_images(tmp_path,metadata=False)
    result=prepare_images(paths,.1469,tmp_path/'prepared',frequencies_mhz=frequencies)
    assert all(item['frequency_origin']=='explicit_user_value' for item in result.image_provenance)


def test_shifted_celestial_grids_align_the_same_source_without_losing_integrated_flux(tmp_path):
    rng=np.random.default_rng(550)
    yy,xx=np.indices((80,80));sigma=6/math.sqrt(8*math.log(2));paths=[]
    for index in range(4):
        offset=.3*index
        profile=np.exp(-((xx-39.5-offset)**2+(yy-39.5)**2)/(2*sigma**2))
        paths.append(image(tmp_path,f'shift{index}.fits',data=.03*profile+rng.normal(0.,1e-5,(80,80)),
            pixels=80,beam=6.,offset=offset,frequency=100.+index*400.))
    result=prepare_images(paths,.1469,tmp_path/'prepared')
    assert sum(item['reprojected'] for item in result.image_provenance)>=3
    np.testing.assert_allclose([item['flux_jy'] for item in result.metadata['integrated_sed']],.03,rtol=.01)
    centroids=[]
    for image_values in result.flux_cube:
        weights=np.where(result.source_mask,np.maximum(image_values,0.),0.)
        centroids.append([np.nansum(xx*weights)/np.nansum(weights),np.nansum(yy*weights)/np.nansum(weights)])
    assert np.max(np.ptp(centroids,axis=0))<.1


@pytest.mark.parametrize('redshift',[0.,-1.,True,float('nan'),21.])
def test_invalid_source_distance_redshift_fails_before_writing(tmp_path,redshift):
    with pytest.raises(MapInputError,match='redshift'):
        prepare_images([tmp_path/f'image{i}' for i in range(4)],redshift,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_duplicate_images_and_frequencies_are_not_independent_measurements(tmp_path):
    paths,_,_=source_images(tmp_path)
    with pytest.raises(MapInputError,match='one input image'):
        prepare_images([paths[0]]*4,.1469,tmp_path/'prepared')
    with pytest.raises(MapInputError,match='frequencies must be distinct'):
        prepare_images(paths,.1469,tmp_path/'prepared',frequencies_mhz=[1000.]*4)
    assert not (tmp_path/'prepared').exists()


def test_existing_preparation_directory_is_left_untouched(tmp_path):
    output=tmp_path/'prepared';output.mkdir();(output/'keep').write_text('user data')
    with pytest.raises(FileExistsError):
        prepare_images([tmp_path/f'image{i}' for i in range(4)],.1469,output)
    assert (output/'keep').read_text()=='user data'


def test_inconsistent_restoring_beam_metadata_is_rejected_before_output(tmp_path):
    paths,_,_=source_images(tmp_path)
    with fits.open(paths[0],mode='update') as hdus:
        hdus[0].header['BMIN']=hdus[0].header['BMAJ']*2
    with pytest.raises(MapInputError,match='BMIN exceeds BMAJ'):
        prepare_images(paths,.1469,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_blank_radio_noise_does_not_generate_a_source(tmp_path):
    paths=[];rng=np.random.default_rng(991)
    for index in range(4):
        paths.append(image(tmp_path,f'noise{index}.fits',data=rng.normal(0.,1e-5,(64,64)),frequency=100.+index*100))
    with pytest.raises(MapInputError,match='no beam-scale source'):
        prepare_images(paths,.1469,tmp_path/'prepared',source_seed_sigma=7.)
    assert not (tmp_path/'prepared').exists()


def test_local_transverse_cylinder_volume_has_independent_analytic_normalization():
    mask=np.zeros((80,80),dtype=bool);mask[10:60,25:45]=True
    result=_local_transverse_volume(mask,np.eye(2)/3600.,4.)
    # A uniform projected tube: 50 arcsec long, corrected diameter
    # sqrt(20^2-4^2) arcsec, and a circular assumed depth/cross-section.
    expected=math.pi/4*(20.**2-4.**2)*50.
    assert result['status']=='conditional_resolved_slices'
    assert result['volume_arcsec3']==pytest.approx(expected,rel=1e-10)


def test_local_ellipse_volume_and_physical_scale_follow_three_dimensional_scaling():
    yy,xx=np.indices((160,160))
    major,minor=40.,20.
    mask=((yy-79.5)/major)**2+((xx-79.5)/minor)**2<=1
    measured=_local_transverse_volume(mask,np.eye(2)/3600.,0.)
    expected=4*math.pi/3*major*minor**2
    assert measured['volume_arcsec3']==pytest.approx(expected,rel=.025)
    doubled=_local_transverse_volume(mask,np.eye(2)*2/3600.,0.)
    assert doubled['volume_arcsec3']==pytest.approx(8*measured['volume_arcsec3'],rel=1e-10)


def test_disjoint_transverse_branches_do_not_include_empty_space_between_them():
    mask=np.zeros((100,100),dtype=bool)
    mask[15:85,15:25]=True;mask[15:85,75:85]=True
    result=_local_transverse_volume(mask,np.eye(2)/3600.,2.)
    # PCA's long direction can follow the combined two-branch separation;
    # use a sufficiently longer longitudinal extent so the slices follow y.
    long=np.zeros((240,100),dtype=bool)
    long[10:230,15:25]=True;long[10:230,75:85]=True
    result=_local_transverse_volume(long,np.eye(2)/3600.,2.)
    expected=2*math.pi/4*(10.**2-2.**2)*220.
    assert result['volume_arcsec3']==pytest.approx(expected,rel=1e-10)


def test_unresolved_transverse_widths_use_explicit_positive_ellipsoid_fallback():
    from astropy.cosmology import Planck18
    mask=np.zeros((64,64),dtype=bool);mask[10:50,30:33]=True
    local=_local_transverse_volume(mask,np.eye(2)/3600.,6.)
    assert local['volume_arcsec3'] is None
    wcs=WCS(header()).celestial;j=_jacobian(wcs,mask.shape)
    beam_area=math.pi*(6/3600)**2/(4*math.log(2))
    result=_geometry(mask,wcs,j,beam_area,.1469,Planck18,6.)
    assert result['volume_status']=='conditional_ellipsoid_fallback'
    assert result['volume_m3']>0
    assert result['local_transverse_volume_m3'] is None


def test_bright_beam_sized_gaussian_is_not_resolved_by_its_wide_isophotal_mask(tmp_path):
    from astropy.cosmology import Planck18
    yy,xx=np.indices((64,64));sigma=6/math.sqrt(8*math.log(2))
    brightness=np.exp(-((xx-31.5)**2+(yy-31.5)**2)/(2*sigma**2))
    mask=brightness>1e-4
    wcs=WCS(header()).celestial;j=_jacobian(wcs,mask.shape)
    beam_area=math.pi*(6/3600)**2/(4*math.log(2))
    result=_geometry(mask,wcs,j,beam_area,.1469,Planck18,6.,brightness)
    assert result['transverse_resolution']['status']=='transverse_size_unresolved_by_moment_heuristic'
    assert result['volume_status']=='conditional_ellipsoid_fallback'
