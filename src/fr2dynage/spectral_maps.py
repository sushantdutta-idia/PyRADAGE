"""Power-law spectral diagnostics for registered radio images.

These fits measure ``I_nu = I_ref (nu/nu_ref)**(-alpha)``. They are not
radiative ages, dynamical ages, or a replacement for an electron-loss model.
Thermal noise and per-band calibration errors are assumed independent between
bands. Pixels within a restoring beam and calibration errors across pixels
are correlated; the returned pixel errors do not remove those correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


MASKED = 1
INVALID_FLUX = 2
INVALID_RMS = 4
LOW_SNR = 8
INSUFFICIENT_BANDS = 16
NUMERICAL_FAILURE = 32
POWER_LAW_MISMATCH = 64
INVALID_WEIGHT = 128

SPECTRAL_FLAGS = {
    MASKED: "excluded by the supplied mask",
    INVALID_FLUX: "at least one band has nonfinite or nonpositive intensity",
    INVALID_RMS: "at least one band has nonfinite or nonpositive local RMS",
    LOW_SNR: "at least one band is below the signal-to-noise threshold",
    INSUFFICIENT_BANDS: "too few usable detected bands for the requested fit",
    NUMERICAL_FAILURE: "fit or error cannot be represented numerically",
    POWER_LAW_MISMATCH: "power-law reduced chi-square exceeds 4",
    INVALID_WEIGHT: "at least one log-intensity error is unrepresentable",
}


class SpectralMapError(ValueError):
    """Invalid array shapes or settings for resolved spectral diagnostics."""


@dataclass(frozen=True)
class SpectralMapResult:
    """Two-dimensional diagnostics; float products are NaN outside ``valid``.

    ``alpha_error`` is conditional on the supplied independent per-band
    sigmas and is not rescaled by reduced chi-square. ``reference_flux`` has
    the input flux units (Jy/beam in the FITS mapping workflow). ``n_bands``
    counts usable detections; quality flags may also be set on a valid fit.
    In particular, a poor power-law fit remains a useful diagnostic and does
    not invalidate an independently constrained kinematic age.
    """

    alpha: np.ndarray
    alpha_error: np.ndarray
    chi2: np.ndarray
    reduced_chi2: np.ndarray
    reference_flux: np.ndarray
    n_bands: np.ndarray
    valid: np.ndarray
    flags: np.ndarray
    reference_frequency_mhz: float


def _real_array(value: object, label: str) -> np.ndarray:
    try:
        original = np.asarray(value)
        if original.dtype.kind == "O":
            if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                   for v in original.flat):
                raise SpectralMapError(f"{label} must contain real numbers")
        elif original.dtype.kind not in "iuf":
            raise SpectralMapError(f"{label} must contain real numbers, not booleans or strings")
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, SpectralMapError):
            raise
        raise SpectralMapError(f"{label} must contain representable real numbers") from exc


def _positive_scalar(value: object, label: str) -> float:
    try:
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise SpectralMapError(f"{label} must be a finite positive number")
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SpectralMapError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise SpectralMapError(f"{label} must be a finite positive number")
    return result


def _sigma_ln(fraction: np.ndarray) -> np.ndarray:
    """Vectorized stable sqrt(log1p(f**2)), including subnormal f."""

    result = np.empty_like(fraction)
    small = fraction < 1e-8
    middle = (fraction >= 1e-8) & (fraction < 1.0)
    large = fraction >= 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result[small] = fraction[small]
        result[middle] = np.sqrt(np.log1p(fraction[middle] ** 2))
        result[large] = np.sqrt(2.0 * np.log(fraction[large])
                                + np.log1p(fraction[large] ** -2))
    return result


def fit_power_law_maps(
    frequencies_mhz,
    flux_cube,
    rms_cube,
    *,
    calibration_fractions=None,
    snr_min=5.0,
    min_bands=3,
    require_all_bands=True,
    mask=None,
) -> SpectralMapResult:
    """Fit spectral-index maps from ``[frequency, y, x]`` image arrays.

    At least three distinct frequencies are required, so chi-square has at
    least one residual degree of freedom. Noise is a scalar, one value per
    band, or a full cube; invalid local RMS pixels are excluded, while invalid
    scalar/per-band RMS settings are rejected. Calibration fractions are a
    scalar or one nonnegative value per band. Thermal and calibration terms
    add in quadrature before the lognormal-coefficient conversion:

    ``sigma_ln = sqrt(log1p((rms / I)**2 + calibration_fraction**2))``.

    The thermal log-space approximation assumes high-SNR positive detections;
    ``snr_min`` must be at least 3 (the default is 5). No upper limits are fit.
    A common normalized weight scale prevents spurious overflow when formal
    measurement errors are very small. Missing bands are allowed only when
    ``require_all_bands=False`` and at least ``min_bands`` remain at a pixel.
    """

    frequencies = _real_array(frequencies_mhz, "frequencies_mhz")
    if (frequencies.ndim != 1 or len(frequencies) < 3
            or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0)
            or len(np.unique(frequencies)) != len(frequencies)):
        raise SpectralMapError("frequencies_mhz must contain at least three unique finite positive frequencies")
    # A mixed Python list otherwise silently turns True into 1.0.
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(frequencies_mhz, dtype=object).flat):
        raise SpectralMapError("frequencies_mhz cannot contain booleans")
    flux = _real_array(flux_cube, "flux_cube")
    if flux.ndim != 3 or flux.shape[0] != len(frequencies) or any(size == 0 for size in flux.shape):
        raise SpectralMapError("flux_cube must have nonempty shape [frequency, y, x] matching frequencies_mhz")
    nfreq, ny, nx = flux.shape
    snr_min = _positive_scalar(snr_min, "snr_min")
    if snr_min < 3.0:
        raise SpectralMapError("snr_min must be at least 3 for log-intensity fitting")
    if isinstance(min_bands, (bool, np.bool_)) or not isinstance(min_bands, Integral) or not 3 <= min_bands <= nfreq:
        raise SpectralMapError("min_bands must be an integer between 3 and the number of frequencies")
    if not isinstance(require_all_bands, (bool, np.bool_)):
        raise SpectralMapError("require_all_bands must be a boolean")

    noise = _real_array(rms_cube, "rms_cube")
    if noise.ndim == 0 or noise.shape == (nfreq,):
        if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(rms_cube, dtype=object).flat):
            raise SpectralMapError("scalar or per-band RMS cannot contain booleans")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise SpectralMapError("scalar or per-band RMS must be finite and positive")
        noise = np.broadcast_to(noise if noise.ndim == 0 else noise[:, None, None], flux.shape)
    elif noise.shape != flux.shape:
        raise SpectralMapError("rms_cube must be a scalar, shape [frequency], or match flux_cube exactly")

    if calibration_fractions is None:
        calibration = np.zeros((nfreq, 1, 1), dtype=float)
    else:
        calibration = _real_array(calibration_fractions, "calibration_fractions")
        if calibration.ndim != 0 and calibration.shape != (nfreq,):
            raise SpectralMapError("calibration_fractions must be a scalar or shape [frequency]")
        if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(calibration_fractions, dtype=object).flat):
            raise SpectralMapError("calibration_fractions cannot contain booleans")
        if not np.all(np.isfinite(calibration)) or np.any(calibration < 0.0):
            raise SpectralMapError("calibration_fractions must be finite and nonnegative")
        if calibration.ndim == 1:
            calibration = calibration[:, None, None]

    if mask is None:
        included = np.ones((ny, nx), dtype=bool)
    else:
        supplied_mask = np.asarray(mask)
        if supplied_mask.shape != (ny, nx):
            raise SpectralMapError("mask must have the image shape [y, x]")
        if supplied_mask.dtype.kind not in "biuf" or not np.all(np.isfinite(supplied_mask)) or not np.all((supplied_mask == 0) | (supplied_mask == 1)):
            raise SpectralMapError("mask must contain only booleans or finite binary 0/1 values")
        included = supplied_mask.astype(bool)

    flags = np.zeros((ny, nx), dtype=np.uint16)
    flags[~included] |= MASKED
    valid_flux = np.isfinite(flux) & (flux > 0.0)
    valid_noise = np.isfinite(noise) & (noise > 0.0)
    flags[np.any(~valid_flux, axis=0)] |= INVALID_FLUX
    flags[np.any(~valid_noise, axis=0)] |= INVALID_RMS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_flux = np.where(valid_flux, np.log(flux), 0.0)
        relative_noise = np.divide(noise, flux, out=np.zeros_like(flux), where=valid_flux & valid_noise)
        detected = valid_flux & valid_noise & (relative_noise <= 1.0 / snr_min)
        relative_noise = np.where(detected, relative_noise, 0.0)
        fractional_error = np.hypot(relative_noise, calibration)
        sigma = _sigma_ln(fractional_error)
    flags[np.any(valid_flux & valid_noise & ~detected, axis=0)] |= LOW_SNR
    valid_weight = np.isfinite(sigma) & (sigma > 0.0)
    flags[np.any(detected & ~valid_weight, axis=0)] |= INVALID_WEIGHT
    usable = detected & valid_weight & included[None, :, :]
    n_bands = np.sum(usable, axis=0, dtype=np.int32)
    required = nfreq if require_all_bands else int(min_bands)
    ready = included & (n_bands >= required)
    flags[n_bands < required] |= INSUFFICIENT_BANDS

    log_frequency = np.log(frequencies)
    log_reference = float(np.mean(log_frequency))
    reference_frequency = min(float(np.max(frequencies)), max(float(np.min(frequencies)), math.exp(log_reference)))
    x = (log_frequency - log_reference)[:, None, None]
    sigma_min = np.min(np.where(usable, sigma, np.inf), axis=0)
    sigma_min = np.where(ready, sigma_min, 1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        weights = np.divide(sigma_min[None, :, :], sigma, out=np.zeros_like(flux), where=usable) ** 2
        total_weight = np.sum(weights, axis=0)
        safe_weight = np.where(total_weight > 0.0, total_weight, 1.0)
        mean_x = np.sum(weights * x, axis=0) / safe_weight
        mean_y = np.sum(weights * log_flux, axis=0) / safe_weight
        dx = x - mean_x
        dy = log_flux - mean_y
        sxx = np.sum(weights * dx**2, axis=0)
        sxy = np.sum(weights * dx * dy, axis=0)
        slope = np.divide(sxy, sxx, out=np.zeros_like(sxy), where=sxx > 0.0)
        alpha = -slope
        alpha_error = sigma_min / np.sqrt(sxx)
        reference_flux = np.exp(mean_y - slope * mean_x)
        residual = dy - slope * dx
        standardized = np.divide(residual, sigma, out=np.zeros_like(flux), where=usable)
        chi2 = np.sum(standardized**2, axis=0)
        reduced_chi2 = chi2 / np.maximum(n_bands - 2, 1)

    numerical = ((sxx > 0.0) & np.isfinite(alpha) & np.isfinite(alpha_error)
                 & (alpha_error > 0.0) & np.isfinite(reference_flux) & (reference_flux > 0.0)
                 & np.isfinite(chi2) & np.isfinite(reduced_chi2))
    valid = ready & numerical
    flags[ready & ~numerical] |= NUMERICAL_FAILURE
    flags[valid & (reduced_chi2 > 4.0)] |= POWER_LAW_MISMATCH

    def masked(values):
        return np.where(valid, values, np.nan)

    return SpectralMapResult(
        alpha=masked(alpha), alpha_error=masked(alpha_error),
        chi2=masked(chi2), reduced_chi2=masked(reduced_chi2),
        reference_flux=masked(reference_flux), n_bands=n_bands,
        valid=valid, flags=flags, reference_frequency_mhz=reference_frequency,
    )
