"""Descriptive central-beam spectra, kept distinct from plume-age inference.

A common-beam image does not deblend a compact nucleus from inner jets or
returning plume emission. This module includes that emission in the analysis
without assigning a spectral age or a transport age to an unresolved core.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real

import numpy as np

from .spectral_maps import _real_array, _sigma_ln


class CoreDiagnosticError(ValueError):
    """Core diagnostics cannot be constructed from these arrays/settings."""


def _boolean_mask(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'biuf' or not np.all((array == 0) | (array == 1)):
        raise CoreDiagnosticError(f'{label} must be a binary mask matching the image shape')
    return array.astype(bool)


def _strict(document):
    json.dumps(document, allow_nan=False)
    return document


def analyze_core_region(frequencies_mhz, flux_cube, rms_cube,
                        calibration_fractions, calibration_correlation,
                        core_mask, source_valid, reference,
                        core_position_icrs_deg=None, *, snr_min=5., min_bands=4):
    """Return finite JSON for one representative central-beam power-law SED.

    ``flux_cube``/``rms_cube`` have shape (frequency, y, x), in Jy/beam.
    The selected pixel is nearest to the supplied ICRS position among detected
    core pixels. Without a position, select the highest-SNR pixel in the band
    closest to the geometric mean frequency (falling back to median SNR).
    The intensity is a restoring-beam brightness, never aperture-integrated
    flux or a deblended nuclear measurement. No pixel spectra are combined as
    independent measurements. The GLS error is conditional on the supplied
    thermal errors and full inter-band calibration correlation.
    """
    frequency = _real_array(frequencies_mhz, 'frequencies_mhz')
    flux = _real_array(flux_cube, 'flux_cube')
    rms = _real_array(rms_cube, 'rms_cube')
    calibration = _real_array(calibration_fractions, 'calibration_fractions')
    correlation = _real_array(calibration_correlation, 'calibration_correlation')
    if (frequency.ndim != 1 or not 4 <= frequency.size <= 64 or
            not np.all(np.isfinite(frequency)) or np.any(frequency <= 0) or
            np.unique(frequency).size != frequency.size):
        raise CoreDiagnosticError('frequencies_mhz requires 4–64 distinct finite positive values')
    if flux.ndim != 3 or flux.shape[0] != len(frequency) or flux.shape != rms.shape or min(flux.shape[1:]) < 1:
        raise CoreDiagnosticError('flux_cube and rms_cube must share shape (frequency,y,x)')
    if calibration.shape != frequency.shape or not np.all(np.isfinite(calibration)) or np.any(calibration < 0):
        raise CoreDiagnosticError('calibration_fractions must contain one finite nonnegative value per band')
    if (correlation.shape != (len(frequency), len(frequency)) or
            not np.all(np.isfinite(correlation)) or
            not np.allclose(correlation, correlation.T, rtol=0., atol=1e-10) or
            not np.allclose(np.diag(correlation), 1., rtol=0., atol=1e-10) or
            np.linalg.eigvalsh(correlation).min() < -1e-9):
        raise CoreDiagnosticError('calibration_correlation must be a symmetric positive-semidefinite correlation matrix')
    if (not isinstance(snr_min, Real) or isinstance(snr_min, (bool, np.bool_)) or
            not math.isfinite(snr_min) or snr_min <= 0):
        raise CoreDiagnosticError('snr_min must be finite and positive')
    if (not isinstance(min_bands, Integral) or isinstance(min_bands, (bool, np.bool_)) or
            not 4 <= min_bands <= len(frequency)):
        raise CoreDiagnosticError('min_bands must be an integer from 4 to the number of bands')
    core = _boolean_mask(core_mask, flux.shape[1:], 'core_mask')
    source = _boolean_mask(source_valid, flux.shape[1:], 'source_valid')
    position = None
    if core_position_icrs_deg is not None:
        if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(core_position_icrs_deg, dtype=object).flat):
            raise CoreDiagnosticError('core_position_icrs_deg must contain real coordinates, not booleans')
        position = _real_array(core_position_icrs_deg, 'core_position_icrs_deg')
        if (position.shape != (2,) or not np.all(np.isfinite(position)) or
                not 0 <= position[0] < 360 or not -90 <= position[1] <= 90):
            raise CoreDiagnosticError('core_position_icrs_deg requires RA in [0,360), Dec in [-90,90]')
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        snr = flux/rms
    usable = np.isfinite(flux) & (flux > 0) & np.isfinite(rms) & (rms > 0) & (snr >= snr_min)
    detected = core & source & (usable.sum(axis=0) >= min_bands)
    result = {'status': 'unresolved_age', 'core_mask_pixels': int(core.sum()),
              'source_selected_core_pixels': int((core & source).sum()),
              'detected_pixels': int(detected.sum()),
              'spectral_age_myr': None, 'transport_age_myr': None,
              'age_status': 'unresolved_core_geometry_and_population_mixing',
              'interpretation': 'central restoring-beam brightness; not integrated core flux or deblended nuclear emission',
              'central_spectrum': None,
              'warnings': ['The common restoring beam can blend the nucleus, inner jets and projected plume emission; a unique core age is not identified.',
                           'A descriptive power-law fit does not test a physical ageing or absorption model.',
                           'Core spectra are reported separately and do not enter the optically thin JP plume-age likelihood.',
                           'The geometrical zero-time origin is a model reference, not a measured zero-age region.']}
    if not np.any(detected):
        result['status'] = 'no_detected_core_spectrum'
        return _strict(result)
    yy, xx = np.nonzero(detected)
    try:
        sky = reference.wcs.pixel_to_world(xx, yy).icrs
        if position is not None:
            from astropy.coordinates import SkyCoord
            coordinate = SkyCoord(position[0], position[1], unit='deg', frame='icrs')
            index = int(np.argmin(sky.separation(coordinate).deg))
            selection = 'nearest detected core pixel to supplied ICRS position'
        else:
            ref_band = int(np.argmin(np.abs(np.log(frequency)-np.mean(np.log(frequency)))))
            scores = np.where(usable[ref_band, yy, xx], snr[ref_band, yy, xx], -np.inf)
            if not np.any(np.isfinite(scores)):
                scores = np.nanmedian(np.where(usable[:, yy, xx], snr[:, yy, xx], np.nan), axis=0)
            index = int(np.argmax(scores))
            selection = 'highest-SNR detected core pixel near reference frequency; median SNR fallback'
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoreDiagnosticError(f'reference must provide a valid celestial WCS: {exc}') from exc
    y, x = int(yy[index]), int(xx[index])
    use = usable[:, y, x]
    observed, noise, freq = flux[use, y, x], rms[use, y, x], frequency[use]
    thermal_sigma = _sigma_ln(noise/observed)
    cal_sigma = _sigma_ln(calibration[use])
    covariance = (np.diag(thermal_sigma**2) +
                  correlation[np.ix_(use, use)]*cal_sigma[:, None]*cal_sigma[None, :])
    reference_frequency = float(math.exp(np.mean(np.log(freq))))
    design = np.column_stack((np.ones(len(freq)), -np.log(freq/reference_frequency)))
    spectrum = {'pixel_xy': [x, y], 'sky_icrs_deg': [float(sky[index].ra.deg), float(sky[index].dec.deg)],
                'selection': selection, 'frequencies_mhz': frequency.tolist(),
                'observed_jy_per_beam': [float(value) if good else None for value, good in zip(flux[:, y, x], use)],
                'rms_jy_per_beam': [float(value) if good else None for value, good in zip(rms[:, y, x], use)],
                'model_jy_per_beam': [None]*len(frequency),
                'usable_bands': int(use.sum()), 'power_law': None}
    try:
        root = np.linalg.cholesky(covariance)
        weighted_design = np.linalg.solve(root, design)
        weighted_data = np.linalg.solve(root, np.log(observed))
        coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_data, rcond=None)
        if rank != 2:
            raise np.linalg.LinAlgError('rank-deficient spectral design')
        information = weighted_design.T@weighted_design
        parameters_covariance = np.linalg.inv(information)
        residual = weighted_data-weighted_design@coefficients
        chi2 = float(residual@residual)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            predicted = np.exp(coefficients[0]-coefficients[1]*np.log(frequency/reference_frequency))
            normalization = float(np.exp(coefficients[0]))
        if (not np.all(np.isfinite(parameters_covariance)) or
                parameters_covariance[1, 1] <= 0 or not math.isfinite(chi2) or
                not np.all(np.isfinite(predicted)) or np.any(predicted <= 0) or
                not math.isfinite(normalization) or normalization <= 0):
            raise np.linalg.LinAlgError('unrepresentable descriptive spectrum')
        spectrum['model_jy_per_beam'] = predicted.tolist()
        spectrum['power_law'] = {'status': 'descriptive_gls_fit', 'alpha_positive_convention': float(coefficients[1]),
            'alpha_error_conditional': float(math.sqrt(parameters_covariance[1, 1])),
            'reference_frequency_mhz': reference_frequency, 'reference_intensity_jy_per_beam': normalization,
            'chi2': chi2, 'degrees_of_freedom': len(freq)-2, 'reduced_chi2': chi2/(len(freq)-2),
            'log_intensity_covariance': covariance.tolist(),
            'parameter_covariance_log_intensity_alpha': parameters_covariance.tolist(),
            'error_scope': 'conditional GLS with thermal errors and supplied inter-band calibration covariance; no error rescaling by residual chi-square'}
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        spectrum['power_law'] = {'status': 'numerical_failure', 'reason': str(exc)}
    result['central_spectrum'] = spectrum
    return _strict(result)
