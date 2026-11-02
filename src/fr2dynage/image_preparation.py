"""Explicit, auditable preparation of cropped multi-frequency radio images.

This module does not infer a compact core, a flow path or a source lifetime.
It registers surface brightness, matches Gaussian restoring beams, measures
off-source noise, and retains every significant island in a source crop. An
isophotal footprint and an assumed line-of-sight depth supply *conditional*
geometry for an equipartition calculation, never a measured 3-D volume.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
from scipy import fft, ndimage, signal

from .constants import KPC_M
from .maps_io import FitsPlane, MapInputError, read_fits_plane, science_image
from .reliability import sha256_file


class FrequencyMetadataError(MapInputError):
    """An observed frequency needs an explicit user value, in MHz."""
    def __init__(self, path, reason, candidates_mhz=None):
        self.path = Path(path)
        self.reason = reason
        self.candidates_mhz = candidates_mhz or {}
        super().__init__(f'{self.path.name}: {reason}; supply its observing frequency in MHz')


@dataclass
class PreparedImages:
    reference: FitsPlane
    frequencies_mhz: np.ndarray
    flux_cube: np.ndarray
    rms_cube: np.ndarray
    calibration_fractions: np.ndarray
    calibration_correlation: np.ndarray
    source_mask: np.ndarray
    footprint_mask: np.ndarray
    geometry: dict
    image_provenance: list[dict]
    image_records: list[dict]
    metadata: dict
    output_dir: Path
    geometry_support_mask: np.ndarray | None = None


def _finite_number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MapInputError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise MapInputError(f'{name} must be finite'+(' and positive' if positive else ''))
    return value


def image_frequency_mhz(path, hdu=0):
    """Read an observing frequency; never derive it from a filename or RESTFRQ.

    A FREQ spectral WCS is evaluated at the actual singleton plane (pixel 0),
    so a non-unit CRPIX is respected. REFFREQ/FREQ keywords use FREQUNIT when
    supplied, otherwise Hz. RESTFRQ alone is insufficient: for line images it
    describes an emitted transition, not the observed continuum frequency.
    Conflicting observed-frequency candidates require an explicit override.
    """
    from astropy import units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    path = Path(path).expanduser().resolve()
    candidates = {}
    with fits.open(path, memmap=False) as hdus:
        header = hdus[hdu].header.copy()
    try:
        for key in ('REFFREQ', 'FREQ', 'FREQCENT'):
            if key in header:
                value = _finite_number(header[key], f'{path.name} {key}', positive=True)
                unit = u.Unit(header.get('FREQUNIT', 'Hz'))
                candidates[key] = float((value*unit).to_value(u.MHz))
        full = WCS(header)
        frequency_axes = [index for index, kind in enumerate(full.wcs.ctype)
                          if str(kind).strip().upper().startswith('FREQ')]
        if len(frequency_axes) > 1:
            raise FrequencyMetadataError(path, 'more than one FREQ axis is present')
        if frequency_axes:
            index = frequency_axes[0]
            if index < 2:
                raise FrequencyMetadataError(path, 'a celestial image axis is labelled FREQ')
            # Coupled celestial/frequency axes cannot describe one common
            # observing frequency across a 2-D continuum image.
            matrix = full.axis_correlation_matrix
            if np.any(matrix[index, :2]):
                raise FrequencyMetadataError(path, 'observing frequency varies across celestial pixels')
            coordinates = [0. for _ in range(full.pixel_n_dim)]
            world = full.pixel_to_world_values(*coordinates)
            value = float(np.asarray(world[index]))
            unit = u.Unit(full.world_axis_units[index] or 'Hz')
            candidates['FREQ_WCS'] = float((value*unit).to_value(u.MHz))
    except FrequencyMetadataError:
        raise
    except (ValueError, TypeError, OverflowError, u.UnitConversionError) as exc:
        raise FrequencyMetadataError(path, f'invalid frequency metadata ({exc})', candidates) from exc
    if not candidates:
        raise FrequencyMetadataError(path, 'no unambiguous observed-frequency metadata is present')
    values = np.array(list(candidates.values()))
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise FrequencyMetadataError(path, 'frequency metadata is nonfinite or nonpositive', candidates)
    if not np.allclose(values, values[0], rtol=1e-3, atol=0.):
        raise FrequencyMetadataError(path, 'observed-frequency metadata candidates disagree by more than 0.1%', candidates)
    return float(values[0])


def _jacobian(wcs, shape):
    """Local east/north degrees per x/y pixel, including rotation and cos(dec)."""
    y, x = (np.array(shape)-1)/2
    origin = wcs.pixel_to_world(x, y)
    east, north = origin.spherical_offsets_to(wcs.pixel_to_world([x+1, x], [y, y+1]))
    matrix = np.array([east.deg, north.deg])
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-20:
        raise MapInputError('image has a singular or invalid local celestial pixel scale')
    return matrix


def _robust_noise(data, selection=None):
    values = np.asarray(data)[np.isfinite(data) & (True if selection is None else selection)]
    if values.size < 32:
        raise MapInputError('too little finite off-source sky to estimate background/noise; use a larger source crop')
    for _ in range(5):
        median = float(np.median(values))
        sigma = float(1.482602218505602*np.median(np.abs(values-median)))
        if not math.isfinite(sigma) or sigma <= 0:
            raise MapInputError('off-source noise is zero or unmeasurable; supply real noisy images or an explicit RMS constraint')
        keep = np.abs(values-median) <= 4.*sigma
        if np.all(keep):
            break
        values = values[keep]
        if len(values) < 32:
            raise MapInputError('too few sky samples remain after robust background clipping')
    return median, sigma, int(len(values))


def _border(shape):
    width = max(2, int(math.ceil(min(shape)*.15)))
    selected = np.zeros(shape,dtype=bool)
    selected[:width] = selected[-width:] = True
    selected[:,:width] = selected[:,-width:] = True
    return selected


def _resample(data, source_wcs, reference_wcs, shape):
    """Bilinear surface-brightness interpolation with strict finite coverage."""
    if data.shape == shape:
        corners = np.array([[0.,0.],[shape[1]-1.,shape[0]-1.],[(shape[1]-1)/2,(shape[0]-1)/2]])
        sky = reference_wcs.pixel_to_world(corners[:,0],corners[:,1])
        x,y = source_wcs.world_to_pixel(sky)
        if np.all(np.isfinite([x,y])) and np.max(np.hypot(x-corners[:,0],y-corners[:,1])) < 1e-6:
            return data.copy(), False
    result = np.full(shape,np.nan)
    finite = np.isfinite(data)
    safe = np.where(finite,data,0.)
    count = math.prod(shape)
    for start in range(0,count,65536):
        flat = np.arange(start,min(start+65536,count))
        y,x = flat//shape[1],flat%shape[1]
        sky = reference_wcs.pixel_to_world(x,y)
        sx,sy = source_wcs.world_to_pixel(sky)
        positions = np.array([sy,sx])
        values = ndimage.map_coordinates(safe,positions,order=1,mode='constant',cval=0.,prefilter=False)
        weight = ndimage.map_coordinates(finite.astype(float),positions,order=1,mode='constant',cval=0.,prefilter=False)
        good = np.isfinite(sx)&np.isfinite(sy)&(weight >= 1.-1e-6)
        result[y[good],x[good]] = values[good]
    return result, True


def _smooth(data, covariance_pixels):
    """Apply a Gaussian transfer function; zero padding prevents wraparound."""
    covariance = (covariance_pixels+covariance_pixels.T)/2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < -1e-6*max(1.,eigenvalues.max()):
        raise MapInputError('common-beam kernel would require deconvolution')
    if eigenvalues.max() <= 1e-8:
        return data.copy()
    covariance += np.eye(2)*max(0.,-eigenvalues.min())
    padding = max(2,int(math.ceil(6.*math.sqrt(eigenvalues.max()))))
    shape = tuple(fft.next_fast_len(size+2*padding) for size in data.shape)
    kx = 2*np.pi*fft.rfftfreq(shape[1])[None,:]
    ky = 2*np.pi*fft.fftfreq(shape[0])[:,None]
    transfer = np.exp(-.5*(covariance[0,0]*kx*kx+2*covariance[0,1]*kx*ky+covariance[1,1]*ky*ky))
    finite = np.isfinite(data)
    values = np.pad(np.where(finite,data,0.),padding)
    weights = np.pad(finite.astype(float),padding)
    smooth = fft.irfftn(fft.rfftn(values,s=shape)*transfer,s=shape)
    coverage = fft.irfftn(fft.rfftn(weights,s=shape)*transfer,s=shape)
    part = (slice(padding,padding+data.shape[0]),slice(padding,padding+data.shape[1]))
    result = smooth[part].copy()
    # Never renormalise a partly missing source/edge into a full-beam flux.
    result[np.abs(coverage[part]-1.) > 1e-3] = np.nan
    return result


def _source_mask(cube, noises, footprint, seed_sigma, grow_sigma, min_pixels):
    weights = 1./np.square(noises)
    detection = np.sum(cube*weights[:,None,None],axis=0)/math.sqrt(float(weights.sum()))
    seed = footprint & (detection >= seed_sigma)
    grow = footprint & (detection >= grow_sigma)
    labels, count = ndimage.label(grow,structure=np.ones((3,3)))
    retained=[]
    for index in range(1,count+1):
        selected = labels==index
        if np.any(selected & seed) and selected.sum() >= min_pixels:
            retained.append(index)
    mask = np.isin(labels,retained)
    if not np.any(mask):
        raise MapInputError('no beam-scale source passes automatic detection; check units/frequencies/noise and source cropping')
    mask = ndimage.binary_fill_holes(mask)&footprint
    return mask, len(retained), detection


def _correlated_flux_error(mask, rms, covariance_pixels, pixels_per_beam):
    """Integrated thermal error under a Gaussian-beam autocorrelation model."""
    radius = max(2,int(math.ceil(6*math.sqrt(2*np.linalg.eigvalsh(covariance_pixels).max()))))
    yy,xx = np.mgrid[-radius:radius+1,-radius:radius+1]
    inverse = np.linalg.inv(covariance_pixels)
    correlation = np.exp(-.25*(inverse[0,0]*xx**2+2*inverse[0,1]*xx*yy+inverse[1,1]*yy**2))
    effective_pairs = float(np.sum(mask*signal.fftconvolve(mask.astype(float),correlation,mode='same')))
    return rms*math.sqrt(max(effective_pairs,0.))/pixels_per_beam


def _local_transverse_volume(mask, jacobian, beam_fwhm_arcsec):
    """Conditional circular-cross-section volume in a PCA-aligned raster.

    Every disjoint transverse run is treated separately, so the empty space
    between bent branches is not filled by one global bounding ellipse.
    The isophotal run width is corrected by sqrt(width^2-beam_FWHM^2).
    That correction is a declared geometric/Gaussian-width approximation,
    not an inference of an intrinsic emissivity profile or measured depth.
    """
    y,x=np.nonzero(mask)
    pixel_arcsec=math.sqrt(abs(np.linalg.det(jacobian)))*3600.
    if not len(x):
        return {'status':'empty_support','volume_arcsec3':None}
    center=np.array([np.mean(x),np.mean(y)])
    world=(np.column_stack([x,y])-center)@(jacobian*3600.).T
    covariance=np.cov(world.T) if len(x)>2 else np.eye(2)
    _,basis=np.linalg.eigh(covariance)
    positions=world@basis
    lower=np.min(positions,axis=0)
    spans=np.ptp(positions,axis=0)
    dimensions=np.ceil(spans/pixel_arcsec).astype(int)+1
    swap=np.array([[0.,1.],[1.,0.]])
    inverse=np.linalg.inv(jacobian*3600.)
    matrix=swap@inverse@basis@swap*pixel_arcsec
    offset=swap@(center+inverse@basis@lower)
    raster=ndimage.affine_transform(mask.astype(float),matrix,offset,
        output_shape=(int(dimensions[1]),int(dimensions[0])),order=1,
        mode='constant',cval=0.,prefilter=False)>=.5
    observed=0.;intrinsic=0.;resolved_pixels=0;resolved_slices=0;runs=0
    resolved_length=0.
    for row in raster:
        transitions=np.diff(np.r_[False,row,False].astype(np.int8))
        starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
        row_resolved=False
        for first,last in zip(starts,ends):
            width=(last-first)*pixel_arcsec;runs+=1
            observed+=math.pi/4*width**2*pixel_arcsec
            square=width**2-beam_fwhm_arcsec**2
            if square>0:
                intrinsic+=math.pi/4*square*pixel_arcsec
                resolved_pixels+=last-first;row_resolved=True
                resolved_length+=pixel_arcsec
        resolved_slices+=int(row_resolved)
    fraction=resolved_pixels/max(int(raster.sum()),1)
    stable=intrinsic>0 and resolved_slices>=3 and fraction>=.5
    return {'status':'conditional_resolved_slices' if stable else 'unresolved_or_insufficient_transverse_support',
        'volume_arcsec3':float(intrinsic) if stable else None,
        'observed_volume_arcsec3':float(observed),
        'resolved_area_fraction':float(fraction),'resolved_slices':resolved_slices,
        'transverse_runs':runs,'raster_pixel_arcsec':pixel_arcsec,
        'effective_intrinsic_width_arcsec':math.sqrt(intrinsic/(math.pi/4*resolved_length)) if resolved_length>0 else None,
        'beam_width_correction':'sqrt(isophotal_width^2 - common_beam_FWHM^2); geometric Gaussian-width approximation, not a resolved-profile measurement'}


def _geometry(mask, wcs, jacobian, beam_area_deg2, redshift, cosmology, beam_fwhm_arcsec, intensity=None):
    y,x = np.nonzero(mask)
    sky = wcs.pixel_to_world(x,y)
    center = wcs.pixel_to_world(float(np.mean(x)),float(np.mean(y)))
    east,north = center.spherical_offsets_to(sky)
    positions = np.column_stack([east.arcsec,north.arcsec])
    covariance = np.cov(positions.T) if len(x)>2 else np.eye(2)
    _,basis = np.linalg.eigh(covariance)
    projected = positions@basis
    pixel_arcsec = math.sqrt(abs(np.linalg.det(jacobian)))*3600.
    spans = np.ptp(projected,axis=0)+pixel_arcsec
    minor,major = float(np.min(spans)),float(np.max(spans))
    scale = float(cosmology.kpc_proper_per_arcmin(redshift).value/60.)
    major_kpc,minor_kpc = major*scale,minor*scale
    area_deg2 = abs(float(np.linalg.det(jacobian)))*int(mask.sum())
    ellipsoid_kpc3 = math.pi/6*major_kpc*minor_kpc**2
    transverse=_local_transverse_volume(mask,jacobian,beam_fwhm_arcsec)
    resolution={'status':'not_assessed_without_intensity',
        'scope':'descriptive global intensity-moment check, not a deconvolution confidence test'}
    if intensity is not None:
        weights=np.maximum(np.asarray(intensity)[mask],0.)
        weights/=weights.sum()
        centered=positions-np.sum(positions*weights[:,None],axis=0)
        weighted_covariance=(centered*weights[:,None]).T@centered
        observed_minor=float(np.linalg.eigvalsh(weighted_covariance).min())
        beam_variance=beam_fwhm_arcsec**2/(8*math.log(2))
        intrinsic_minor=observed_minor-beam_variance
        resolved=intrinsic_minor>.1*beam_variance
        resolution.update(status='transverse_extension_by_moment_heuristic' if resolved else 'transverse_size_unresolved_by_moment_heuristic',
            observed_minor_variance_arcsec2=observed_minor,beam_variance_arcsec2=beam_variance,
            nominal_intrinsic_minor_variance_arcsec2=intrinsic_minor,
            rule='deconvolved minor moment variance > 0.1 beam variance; descriptive resolution gate, not a calibrated detection significance')
        if not resolved:
            transverse['status']='unresolved_in_intensity_moments'
            transverse['volume_arcsec3']=None
    local_kpc3=transverse['volume_arcsec3']*scale**3 if transverse['volume_arcsec3'] is not None else None
    volume_kpc3=local_kpc3 if local_kpc3 is not None else ellipsoid_kpc3
    spread=abs(ellipsoid_kpc3-volume_kpc3)/volume_kpc3
    return {'geometry_model':'principal-axis local transverse circular sections' if local_kpc3 is not None else 'projected-mask ellipsoid fallback',
        'volume_estimator':'local transverse circular cross-section integration' if local_kpc3 is not None else 'ellipsoid fallback: local widths unresolved/unstable',
        'volume_status':'conditional_local_transverse_model' if local_kpc3 is not None else 'conditional_ellipsoid_fallback',
        'geometry_support_pixels':int(mask.sum()),
        'largest_linear_size_kpc':major_kpc,'minor_axis_kpc':minor_kpc,
        'major_axis_arcsec':major,'minor_axis_arcsec':minor,
        'volume_kpc3':volume_kpc3,'volume_m3':volume_kpc3*KPC_M**3,
        'ellipsoid_volume_m3':ellipsoid_kpc3*KPC_M**3,
        'local_transverse_volume_m3':local_kpc3*KPC_M**3 if local_kpc3 is not None else None,
        'volume_estimator_fractional_spread':spread,'local_transverse_model':transverse,
        'transverse_resolution':resolution,
        'effective_local_width_kpc':(transverse['effective_intrinsic_width_arcsec']*scale
                                     if transverse.get('effective_intrinsic_width_arcsec') is not None else None),
        'projected_area_kpc2':area_deg2*(3600*scale)**2,
        'axis_ratio':major/minor,'beams_per_source':area_deg2/beam_area_deg2,
        'kpc_per_arcsec':scale,'volume_fractional_error':max(.5,spread),'filling_factor':1.,
        'assumption_note':'A conditional volume model, not a measured 3-D shape or intrinsic deconvolved dimensions. Principal-axis transverse runs assume circular local cross-sections/depth equal to the width, with a geometric sqrt(width^2-beam_FWHM^2) correction. Disconnected transverse runs are separate. The whole-support ellipsoid is retained as a comparison/fallback. Adopted volume scatter is at least 50% and at least the estimator spread; it is not a measured confidence interval or a guarantee of geometric coverage.'}


def prepare_images(paths, redshift, output_dir, *, frequencies_mhz=None,
                   calibration_fraction=.05, source_seed_sigma=5.,
                   source_grow_sigma=3., min_bands=4, cosmology=None,
                   progress: Callable[[str],None] | None=None):
    """Prepare a cropped single-source image set and write an atomic bundle.

    Four to 64 independent frequencies, celestial WCS, Jy/beam-convertible
    units and valid restoring beams are required. Frequencies may be supplied
    explicitly in input order when observing-frequency metadata is missing or
    ambiguous. Images are sorted by frequency after validation. The finest
    pixel grid is retained; matching never implies new independent samples.
    Empirical post-preparation off-source RMS supplies the fit errors.
    ``output_dir`` must not already exist. A callback receives status strings.
    """
    from astropy.cosmology import Planck18
    from astropy.io import fits
    z = _finite_number(redshift,'redshift',positive=True)
    if z>20:
        raise MapInputError('redshift must lie in (0,20]')
    calibration = _finite_number(calibration_fraction,'calibration_fraction')
    if not 0 <= calibration <= 1:
        raise MapInputError('calibration_fraction must lie in [0,1]')
    seed = _finite_number(source_seed_sigma,'source_seed_sigma',positive=True)
    grow = _finite_number(source_grow_sigma,'source_grow_sigma',positive=True)
    if grow>=seed or grow<2:
        raise MapInputError('source thresholds require 2 <= grow < seed')
    if isinstance(min_bands,bool) or not isinstance(min_bands,int) or min_bands<4:
        raise MapInputError('min_bands must be an integer of at least 4')
    if not isinstance(paths,(list,tuple)) or not 4<=len(paths)<=64 or min_bands>len(paths):
        raise MapInputError('automatic ageing preparation requires 4–64 images and min_bands <= image count')
    paths = [Path(path).expanduser().resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise MapInputError('one input image cannot stand for multiple independent frequency measurements')
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'{target} already exists; choose a new preparation directory')
    if any(path.is_relative_to(target) for path in paths):
        raise MapInputError('preparation output must not contain or replace an input')
    if frequencies_mhz is not None and (not isinstance(frequencies_mhz,(list,tuple,np.ndarray)) or len(frequencies_mhz)!=len(paths)):
        raise MapInputError('frequencies_mhz must provide one observing frequency per image, in input order')
    cosmology = Planck18 if cosmology is None else cosmology
    notify = progress or (lambda message:None)
    planes,fluxes,beams,frequencies,provenance=[],[],[],[],[]
    for index,path in enumerate(paths):
        notify(f'Reading image {index+1}/{len(paths)}: {path.name}')
        plane = read_fits_plane(path)
        flux,beam = science_image(plane)
        if beam['bmin_deg']>beam['bmaj_deg']*(1+1e-8):
            raise MapInputError(f'{path.name}: BMIN exceeds BMAJ; correct the restoring-beam metadata')
        explicit = frequencies_mhz[index] if frequencies_mhz is not None else None
        frequency = (_finite_number(explicit,'observing frequency MHz',positive=True)
                     if explicit is not None else image_frequency_mhz(path))
        if not .01 <= frequency <= 1e7:
            raise FrequencyMetadataError(path,'frequency falls outside 0.01–10000000 MHz; check header frequency units')
        planes.append(plane);fluxes.append(flux);beams.append(beam);frequencies.append(frequency)
        provenance.append({'path':str(path),'hdu':plane.hdu_index,'sha256':plane.sha256,
            'observed_frequency_mhz':frequency,'frequency_origin':'explicit_user_value' if explicit is not None else 'observed_frequency_header',
            'input_beam_major_deg':beam['bmaj_deg'],'input_beam_minor_deg':beam['bmin_deg'],
            'input_beam_pa_deg':beam['bpa_deg'],'input_shape':list(flux.shape)})
    if len(set(frequencies)) != len(frequencies):
        raise MapInputError('observing frequencies must be distinct; duplicated bands are not independent spectra')
    order=np.argsort(frequencies)
    planes=[planes[i] for i in order];fluxes=[fluxes[i] for i in order];beams=[beams[i] for i in order]
    provenance=[provenance[i] for i in order];frequencies=np.array(frequencies)[order]
    jacobians=[_jacobian(plane.wcs,plane.data.shape) for plane in planes]
    reference_index=int(np.argmin([abs(np.linalg.det(j)) for j in jacobians]))
    reference=planes[reference_index];shape=reference.data.shape;jacobian=jacobians[reference_index]
    if len(paths)*math.prod(shape)>64_000_000:
        raise MapInputError('image cube is too large for the automatic desktop workflow; crop around one source first')
    inverse=np.linalg.inv(jacobian)
    target_major=max(beam['bmaj_deg'] for beam in beams)
    common_covariance=np.eye(2)*target_major**2/(8*math.log(2))
    common_area=math.pi*target_major**2/(4*math.log(2))
    pixel_area=abs(float(np.linalg.det(jacobian)));pixels_per_beam=common_area/pixel_area
    prepared=[]
    for index,(plane,flux,beam) in enumerate(zip(planes,fluxes,beams)):
        notify(f'Aligning and matching beam {index+1}/{len(planes)}')
        background,native_rms,_=_robust_noise(flux,_border(flux.shape))
        source_area=math.pi*beam['bmaj_deg']*beam['bmin_deg']/(4*math.log(2))
        brightness,reprojected=_resample((flux-background)/source_area,plane.wcs,reference.wcs,shape)
        kernel_covariance=inverse@(common_covariance-beam['covariance_deg2'])@inverse.T
        image=_smooth(brightness,kernel_covariance)*common_area
        prepared.append(image)
        provenance[index].update(background_subtracted_jy_per_input_beam=background,
            input_robust_rms_jy_per_beam=native_rms,reprojected=reprojected,
            beam_area_conversion=common_area/source_area)
    cube=np.stack(prepared)
    footprint=np.all(np.isfinite(cube),axis=0)
    if footprint.sum()<128:
        raise MapInputError('too little finite common sky footprint after registration and beam matching')
    noises=np.array([_robust_noise(image,footprint)[1] for image in cube])
    minimum_pixels=max(3,int(math.ceil(.25*pixels_per_beam)))
    mask,components,detection=_source_mask(cube,noises,footprint,seed,grow,minimum_pixels)
    common_cov_pixels=inverse@common_covariance@inverse.T
    exclusion_iterations=max(1,int(math.ceil(target_major/math.sqrt(pixel_area))))
    for _ in range(2):
        off_source=footprint&~ndimage.binary_dilation(mask,iterations=exclusion_iterations)
        if off_source.sum()<max(64,3*pixels_per_beam):
            raise MapInputError('too little off-source sky after source detection; select a larger crop around one source')
        for index,image in enumerate(cube):
            background,noises[index],count=_robust_noise(image,off_source)
            cube[index]-=background
            provenance[index]['residual_background_subtracted_jy_per_common_beam']=background
            provenance[index]['final_robust_rms_jy_per_beam']=float(noises[index])
            provenance[index]['off_source_clipped_pixels']=count
        mask,components,detection=_source_mask(cube,noises,footprint,seed,grow,minimum_pixels)
    geometry_support=mask&(np.sum(cube/noises[:,None,None]>=grow,axis=0)>=min_bands)
    if geometry_support.sum()<minimum_pixels:
        raise MapInputError('too little multi-band source support for conditional volume; a field/volume constraint is needed in the expert workflow')
    geometry=_geometry(geometry_support,reference.wcs,jacobian,common_area,z,cosmology,target_major*3600.,detection)
    sed=[]
    off_source=footprint&~ndimage.binary_dilation(mask,iterations=exclusion_iterations)
    background_beams=max(float(off_source.sum())/(2*pixels_per_beam),1.)
    for frequency,image,noise in zip(frequencies,cube,noises):
        integrated=float(np.sum(image[mask])/pixels_per_beam)
        thermal=_correlated_flux_error(mask,float(noise),common_cov_pixels,pixels_per_beam)
        zero=1.2533141373155*float(noise)/math.sqrt(background_beams)*mask.sum()/pixels_per_beam
        flux_cal=calibration*abs(integrated)
        sed.append({'frequency_mhz':float(frequency),'flux_jy':integrated,
            'thermal_error_jy':thermal,'zero_level_error_jy':float(zero),'calibration_error_jy':flux_cal,
            'total_error_jy':math.sqrt(thermal**2+zero**2+flux_cal**2),
            'scope':'integrated selected-footprint flux; Gaussian-beam-correlated thermal errors, approximate median zero-level error and supplied calibration fraction'})
    warnings=[
        'Inputs are assumed cropped around one radio source; morphology, membership, matched uv coverage, flux recovery, variability and absorption are not independently verified.',
        'Reprojection is bilinear surface-brightness interpolation, not a reconstruction of interferometric uv coverage. It adds correlated smoothing; prepared pixels are not independent observations.',
        'Empirical post-preparation off-source RMS is treated as spatially constant per band. Actual mosaics/primary-beam noise gradients require supplied RMS maps in the expert workflow.',
        'Calibration fractions are user/default assumptions, not inferred instrument calibration errors; independent inter-band calibration is assumed here.',
        'The isophotal source mask retains all significant islands and fills interior holes, including central emission. It neither isolates a compact core nor assigns central ages.',
        'Source volume, filling factor and line-of-sight depth remain explicit modelling assumptions; no dynamical path, velocity or global source lifetime is inferred by image preparation.']
    if components>1:
        warnings.append(f'{components} significant islands retained; disconnected radio lobes are preserved, but unrelated field sources cannot be identified automatically.')
    truncated=bool(np.any(mask & ndimage.binary_dilation(~footprint)) or np.any(mask[[0,-1],:]) or np.any(mask[:,[0,-1]]))
    if truncated:
        warnings.append('Selected emission touches the finite footprint/crop boundary; integrated flux and source geometry may be truncated.')
    if geometry['volume_status']=='conditional_ellipsoid_fallback':
        warnings.append('Local transverse volume is unresolved or unstable; the positive ellipsoid fallback is an explicit size/depth assumption, not a measured source volume.')
    if geometry['volume_estimator_fractional_spread']>.5:
        warnings.append('Local-transverse and whole-support ellipsoid volumes differ by more than 50%; this model spread is included in the adopted volume scatter and neither estimator is an independent 3-D measurement.')
    if pixels_per_beam<4:
        warnings.append('Common beam is poorly sampled by the reference pixels; resolution/interpolation systematics may dominate.')
    peak_y,peak_x=np.unravel_index(np.nanargmax(np.where(footprint,detection,np.nan)),shape)
    peak=reference.wcs.pixel_to_world(peak_x,peak_y).icrs
    image_records=[{'path':str(target/f'band_{index:02d}.fits'),'frequency_mhz':float(frequency),
        'rms_jy_per_beam':float(noise),'calibration_fraction':calibration}
        for index,(frequency,noise) in enumerate(zip(frequencies,noises))]
    metadata={'schema_version':'fr2dynage-prepared-images-1','redshift':z,
        'cosmology':{'name':cosmology.name,'H0_km_s_Mpc':float(cosmology.H0.value),
                     'Om0':float(cosmology.Om0),'Tcmb0_K':float(cosmology.Tcmb0.value)},
        'reference_input':str(reference.path),'shape':list(shape),'pixels_per_beam':pixels_per_beam,
        'common_beam_major_deg':target_major,'common_beam_minor_deg':target_major,'common_beam_pa_deg':0.,
        'common_finite_pixels':int(footprint.sum()),'selected_source_pixels':int(mask.sum()),
        'source_islands_retained':components,'source_seed_sigma':seed,'source_grow_sigma':grow,
        'geometry_support_pixels':int(geometry_support.sum()),'geometry_support_min_bands':min_bands,
        'geometry_support_threshold_sigma':grow,'source_truncated':truncated,
        'intensity_peak_icrs_deg':[float(peak.ra.deg),float(peak.dec.deg)],
        'peak_is_not_automatically_identified_as_core':True,'fixed_circular_exclusion':False,
        'observed_frequencies_mhz':frequencies.tolist(),'image_provenance':provenance,
        'images':image_records,'integrated_sed':sed,'geometry':geometry,'warnings':warnings}
    json.dumps(metadata,allow_nan=False)
    for plane in planes:
        if sha256_file(plane.path)!=plane.sha256:
            raise MapInputError(f'input changed during preparation: {plane.path}')
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'.',dir=target.parent))
    try:
        header=reference.wcs.to_header(relax=True)
        header['BUNIT']='Jy/beam';header['BMAJ']=target_major;header['BMIN']=target_major;header['BPA']=0.
        header['REDSHIFT']=z
        for index,(frequency,image) in enumerate(zip(frequencies,cube)):
            current=header.copy();current['FREQ']=float(frequency)*1e6;current['FREQUNIT']='Hz'
            current['HISTORY']='Automatic preparation: background subtraction, finite WCS registration and common Gaussian beam.'
            fits.PrimaryHDU(image,header=current).writeto(stage/f'band_{index:02d}.fits',checksum=True)
        for name,data in [('source_mask.fits',mask),('footprint_mask.fits',footprint),('geometry_support_mask.fits',geometry_support)]:
            current=header.copy();current['BUNIT']='1'
            fits.PrimaryHDU(data.astype(np.uint8),header=current).writeto(stage/name,checksum=True)
        (stage/'preparation.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        if target.exists():
            raise FileExistsError(f'{target} appeared during preparation; it was left untouched')
        os.rename(stage,target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    prepared_reference=read_fits_plane(target/f'band_{reference_index:02d}.fits')
    notify('Image registration, beam matching, source detection and conditional geometry complete')
    return PreparedImages(prepared_reference,frequencies,cube,
        np.broadcast_to(noises[:,None,None],cube.shape).copy(),np.full(len(paths),calibration),
        np.eye(len(paths)),mask,footprint,geometry,provenance,image_records,metadata,target,geometry_support)
