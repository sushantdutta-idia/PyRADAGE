"""Resolved TJP/CI radiative-age profiles, without a core exclusion.

These are conditional spectral clocks. A CI age is an injection duration;
neither model turns a fitted electron-loss time into a source dynamical age.
All selected detected pixels are fitted, including central emission. A finite
best fit is retained with quality/identifiability flags rather than silently
inventing a zero-age central aperture or hiding a poorly constrained age.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real

import numpy as np

from .spectral_maps import _real_array, _sigma_ln


MASKED = 1
INSUFFICIENT_BANDS = 2
NUMERICAL_FAILURE = 4
LOWER_BOUND = 8
UPPER_BOUND = 16
CURVATURE_NOT_ESTABLISHED = 32
POOR_FIT = 64
OPEN_68_PROFILE = 128
OPEN_UPPER95_PROFILE = 256
DISCONNECTED_PROFILE = 512
UNDERRESOLVED_PROFILE = 1024

AGE_FLAGS = {
    MASKED: "outside the supplied source footprint",
    INSUFFICIENT_BANDS: "too few usable positive detections",
    NUMERICAL_FAILURE: "invalid or ill-conditioned spectral likelihood",
    LOWER_BOUND: "minimum on the lower age search boundary",
    UPPER_BOUND: "minimum on the upper age search boundary",
    CURVATURE_NOT_ESTABLISHED: "independent power-law comparison does not establish curvature by the stated heuristic",
    POOR_FIT: "ageing-model reduced chi-square exceeds 4",
    OPEN_68_PROFILE: "at least one central profile endpoint does not cross Delta chi-square=1 within the search bounds",
    OPEN_UPPER95_PROFILE: "upper profile does not cross Delta chi-square=2.71 within the search bounds",
    DISCONNECTED_PROFILE: "profile acceptance set has disconnected components; reported endpoints describe its hull",
    UNDERRESOLVED_PROFILE: "68-percent profile occupies at most one sampled age; refine the age grid",
}


class ResolvedSpectrumError(ValueError):
    """Resolved spectra or their settings cannot support this analysis."""


def _array(value, label):
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ResolvedSpectrumError(f"{label} cannot contain booleans")
    return _real_array(value, label)


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ResolvedSpectrumError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ResolvedSpectrumError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _crossing(ages, delta, left, right, threshold):
    fraction = (threshold-delta[left])/(delta[right]-delta[left])
    if ages[left] > 0:
        return float(math.exp(math.log(ages[left])+fraction*math.log(ages[right]/ages[left])))
    return float(ages[left]+fraction*(ages[right]-ages[left]))


def _profile_interval(ages, delta, threshold):
    """Return threshold-set hull; an uncrossed endpoint is explicitly NaN."""
    included = np.flatnonzero(delta <= threshold)
    if not len(included):
        return math.nan, math.nan, False, 0
    first, last = int(included[0]), int(included[-1])
    lower = _crossing(ages, delta, first-1, first, threshold) if first > 0 else math.nan
    upper = _crossing(ages, delta, last, last+1, threshold) if last < len(ages)-1 else math.nan
    disconnected = bool(np.any(np.diff(included) != 1))
    return lower, upper, disconnected, len(included)


def fit_resolved_spectra(frequencies_mhz, flux_cube, rms_cube,
                         calibration_fractions, calibration_correlation,
                         source_mask, field_ug, redshift, alpha_inj, *,
                         model="jp_tribble", age_bounds_myr=(0., 1000.),
                         snr_min=5., min_bands=4, age_grid_size=241,
                         cancel_event=None):
    """Profile radiative age and log-amplitude for every detected source pixel.

    Arrays use ``[frequency,y,x]`` and Jy/beam. ``field_ug`` and positive
    ``alpha_inj`` are fixed common inputs. The JP--Tribble field is the RMS
    field used by the free-streaming kernel; the CI age is the duration of
    continuous injection. No path map, distance cut, or circular core mask is
    applied here. Missing bands are omitted, not treated as upper limits.

    The profile likelihood uses the configured inter-band calibration
    correlation plus independent thermal log-flux errors. Each pixel has its
    own normalization. Ages are sampled on a shared logarithmic grid plus an
    exact zero-age point when the lower bound is zero, then 65 actual model
    evaluations refine the two grid cells surrounding each coarse minimum.
    Endpoints interpolate
    the sampled profile, using Delta chi-square=1 and 2.71. These are
    conditional, asymptotic profile conventions, not posterior credible
    intervals or a formal detection significance at a parameter boundary.

    ``SPEC_VALID`` means numerically fitted, not a well identified age.
    ``AGE_CONSTRAINED`` additionally requires a closed, sufficiently sampled
    profile, no boundary/poor-fit flag, and the descriptive curvature check.
    Best ages remain available with flags even when unconstrained. Uncrossed
    interval endpoints stay NaN; no Gaussian or bound-valued errors are made
    up. Shared calibration/field/beam correlations prohibit treating pixel
    age errors as independent in a source average.
    """
    from .advanced_spectra import spectral_log_emissivity
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Analysis cancelled')
    frequencies = _array(frequencies_mhz, "frequencies_mhz")
    if (frequencies.ndim != 1 or not 4 <= frequencies.size <= 64 or
            not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0) or
            len(np.unique(frequencies)) != len(frequencies) or
            any(isinstance(v, (bool, np.bool_)) for v in np.asarray(frequencies_mhz, dtype=object).flat)):
        raise ResolvedSpectrumError("frequencies_mhz requires 4–64 distinct finite positive values")
    flux = _array(flux_cube, "flux_cube")
    if flux.ndim != 3 or flux.shape[0] != len(frequencies) or min(flux.shape[1:]) < 1:
        raise ResolvedSpectrumError("flux_cube must have nonempty shape (frequency,y,x)")
    rms = _array(rms_cube, "rms_cube")
    if rms.ndim == 0 or rms.shape == frequencies.shape:
        if np.any(~np.isfinite(rms)) or np.any(rms <= 0):
            raise ResolvedSpectrumError("scalar/per-band rms_cube must be finite and positive")
        rms = np.broadcast_to(rms if rms.ndim == 0 else rms[:, None, None], flux.shape)
    elif rms.shape != flux.shape:
        raise ResolvedSpectrumError("rms_cube must be scalar, per-band, or match flux_cube")
    calibration = _array(calibration_fractions, "calibration_fractions")
    if calibration.ndim == 0:
        calibration = np.full(len(frequencies), float(calibration))
    if calibration.shape != frequencies.shape or np.any(~np.isfinite(calibration)) or np.any(calibration < 0):
        raise ResolvedSpectrumError("calibration_fractions must be finite nonnegative scalar or per-band values")
    correlation = _array(calibration_correlation, "calibration_correlation")
    if (correlation.shape != (len(frequencies), len(frequencies)) or
            np.any(~np.isfinite(correlation)) or
            not np.allclose(correlation, correlation.T, atol=1e-10, rtol=0.) or
            not np.allclose(np.diag(correlation), 1., atol=1e-10, rtol=0.) or
            np.linalg.eigvalsh(correlation).min() < -1e-9):
        raise ResolvedSpectrumError("calibration_correlation must be a symmetric positive-semidefinite correlation matrix")
    raw_mask = np.asarray(source_mask)
    if raw_mask.shape != flux.shape[1:] or raw_mask.dtype.kind not in "biuf" or not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise ResolvedSpectrumError("source_mask must be a finite binary mask matching the images")
    mask = raw_mask.astype(bool)
    field = _number(field_ug, "field_ug", positive=True)
    z = _number(redshift, "redshift")
    alpha = _number(alpha_inj, "alpha_inj")
    if not .5 <= alpha <= 1.5:
        raise ResolvedSpectrumError("alpha_inj must lie in [.5,1.5]")
    if model not in {"jp_tribble", "ci"}:
        raise ResolvedSpectrumError("model must be jp_tribble or ci")
    bounds = _array(age_bounds_myr, "age_bounds_myr")
    if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or not 0 <= bounds[0] < bounds[1]:
        raise ResolvedSpectrumError("age_bounds_myr must be increasing finite nonnegative bounds")
    if isinstance(age_grid_size, (bool, np.bool_)) or not isinstance(age_grid_size, Integral) or not 41 <= age_grid_size <= 4096:
        raise ResolvedSpectrumError("age_grid_size must be an integer in [41,4096]")
    snr = _number(snr_min, "snr_min", positive=True)
    if snr < 3:
        raise ResolvedSpectrumError("snr_min must be at least 3 for log-flux fitting")
    if isinstance(min_bands, (bool, np.bool_)) or not isinstance(min_bands, Integral) or not 4 <= min_bands <= len(frequencies):
        raise ResolvedSpectrumError("min_bands must be an integer from 4 to the number of frequencies")
    ages = (np.r_[0., np.geomspace(bounds[1]*1e-6, bounds[1], age_grid_size-1)]
            if bounds[0] == 0 else np.geomspace(*bounds, age_grid_size))
    reference_frequency = 1000.
    templates = np.asarray(spectral_log_emissivity(np.r_[frequencies, reference_frequency], ages,
        field, alpha, z, model=model, reference_frequency_mhz=reference_frequency), dtype=float)
    if templates.shape != (len(ages), len(frequencies)+1) or not np.all(np.isfinite(templates)):
        raise ResolvedSpectrumError("spectral model grid must have finite age/frequency emissivities")
    reference_templates, templates = templates[:, -1].copy(), templates[:, :-1]
    shape = flux.shape[1:]
    maps = {name: np.full(shape, np.nan) for name in
            ("SPECTRAL_AGE", "AGE_LO68", "AGE_HI68", "AGE_UPPER95", "CHI2", "REDCHI2",
             "REF_INTENSITY", "POWERLAW_ALPHA", "POWERLAW_REDCHI2", "CURVATURE_DELTA_CHI2")}
    maps.update(SPEC_VALID=np.zeros(shape, dtype=np.uint8), AGE_CONSTRAINED=np.zeros(shape, dtype=np.uint8),
                AGE_FLAGS=np.where(mask, INSUFFICIENT_BANDS, MASKED).astype(np.uint16))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        snr_cube = flux/rms
    usable = np.isfinite(flux) & (flux > 0) & np.isfinite(rms) & (rms > 0) & (snr_cube >= snr)
    maps["N_BANDS"] = usable.sum(axis=0).astype(np.uint8)
    candidate = mask & (maps["N_BANDS"] >= min_bands)
    yy, xx = np.nonzero(candidate)
    # Group equal missing-band patterns, then batch the small GLS systems.
    patterns = usable[:, yy, xx].T
    unique, assignment = np.unique(patterns, axis=0, return_inverse=True)
    for group_index, use in enumerate(unique):
        members = np.flatnonzero(assignment == group_index)
        model_logs = templates[:, use]
        model_centered = model_logs-model_logs.mean(axis=1, keepdims=True)
        frequency_log = np.log(frequencies[use]/reference_frequency)
        cal_sigma = _sigma_ln(calibration[use])
        cal_covariance = correlation[np.ix_(use, use)]*cal_sigma[:, None]*cal_sigma[None, :]
        for start in range(0, len(members), 128):
            selected = members[start:start+128]
            py, px = yy[selected], xx[selected]
            observed = flux[use][:, py, px].T
            noise = rms[use][:, py, px].T
            thermal = _sigma_ln(noise/observed)
            covariance = np.broadcast_to(cal_covariance, (len(selected), len(cal_sigma), len(cal_sigma))).copy()
            diagonal = np.arange(len(cal_sigma))
            covariance[:, diagonal, diagonal] += thermal**2
            scale = np.max(np.diagonal(covariance, axis1=1, axis2=2), axis=1)
            good = np.isfinite(scale) & (scale > 0) & np.all(np.isfinite(covariance), axis=(1, 2))
            maps["AGE_FLAGS"][py, px] = NUMERICAL_FAILURE
            if not np.any(good):
                continue
            normalized = covariance[good]/scale[good, None, None]
            eigenvalues = np.linalg.eigvalsh(normalized)
            stable = (eigenvalues[:, 0] > 0) & (eigenvalues[:, -1]/np.maximum(eigenvalues[:, 0], 1e-300) <= 1e12)
            good_indices = np.flatnonzero(good)[stable]
            if not len(good_indices):
                continue
            py, px, observed, scale = py[good_indices], px[good_indices], observed[good_indices], scale[good_indices]
            root = np.linalg.cholesky(normalized[stable])
            ones = np.linalg.solve(root, np.ones((len(root), len(cal_sigma), 1))).squeeze(-1)
            norm = np.sum(ones**2, axis=1)
            unit = ones/np.sqrt(norm)[:, None]
            data = np.log(observed)
            centered = data-data.mean(axis=1, keepdims=True)
            white_data = np.linalg.solve(root, centered[..., None]).squeeze(-1)
            projected_data = white_data-unit*np.sum(unit*white_data, axis=1)[:, None]
            white_models = np.linalg.solve(root, np.broadcast_to(model_centered.T, (len(root), len(cal_sigma), len(ages))))
            projected_models = white_models-unit[..., None]*np.einsum("qn,qna->qa", unit, white_models)[:, None, :]
            residual = projected_data[..., None]-projected_models
            profiles = np.sum(residual**2, axis=1)/scale[:, None]
            white_slope = np.linalg.solve(root, np.broadcast_to(frequency_log, (len(root), len(cal_sigma)))[..., None]).squeeze(-1)
            projected_slope = white_slope-unit*np.sum(unit*white_slope, axis=1)[:, None]
            slope_norm = np.sum(projected_slope**2, axis=1)
            pl_alpha = -np.sum(projected_slope*projected_data, axis=1)/slope_norm
            pl_residual = projected_data+pl_alpha[:, None]*projected_slope
            pl_chi2 = np.sum(pl_residual**2, axis=1)/scale
            dof = int(use.sum())-2
            for pixel in range(len(py)):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError('Analysis cancelled')
                y, x = py[pixel], px[pixel]
                profile = profiles[pixel]
                if not np.all(np.isfinite(profile)):
                    continue
                coarse_best = int(np.flatnonzero(profile <= np.min(profile)+1e-8)[0])
                left, right = max(coarse_best-1, 0), min(coarse_best+1, len(ages)-1)
                local_ages = (np.geomspace(ages[left], ages[right], 65) if ages[left] > 0
                              else np.linspace(ages[left], ages[right], 65))
                local_emission = spectral_log_emissivity(np.r_[frequencies[use], reference_frequency], local_ages,
                    field, alpha, z, model=model, reference_frequency_mhz=reference_frequency)
                local_models = local_emission[:, :-1]
                local_centered = local_models-local_models.mean(axis=1, keepdims=True)
                local_white = np.linalg.solve(root[pixel], local_centered.T)
                local_projected = local_white-unit[pixel, :, None]*(unit[pixel]@local_white)[None, :]
                local_residual = projected_data[pixel, :, None]-local_projected
                local_profile = np.sum(local_residual**2, axis=0)/scale[pixel]
                profile_ages, ordering = np.unique(np.r_[ages, local_ages], return_index=True)
                profile = np.r_[profile, local_profile][ordering]
                minimum = float(np.min(profile))
                best = int(np.flatnonzero(profile <= minimum+1e-8)[0])
                delta = np.maximum(profile-minimum, 0.)
                lo, hi, disconnected68, accepted = _profile_interval(profile_ages, delta, 1.)
                _, upper, disconnected95, _ = _profile_interval(profile_ages, delta, 2.71)
                flags = 0
                if best == 0:
                    flags |= LOWER_BOUND
                if best == len(profile_ages)-1:
                    flags |= UPPER_BOUND
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    flags |= OPEN_68_PROFILE
                if not np.isfinite(upper):
                    flags |= OPEN_UPPER95_PROFILE
                if disconnected68 or disconnected95:
                    flags |= DISCONNECTED_PROFILE
                if accepted <= 1:
                    flags |= UNDERRESOLVED_PROFILE
                reduced = minimum/dof
                if reduced > 4:
                    flags |= POOR_FIT
                improvement = float(pl_chi2[pixel]-minimum)
                if improvement < 9. or pl_chi2[pixel]/dof <= 2.:
                    flags |= CURVATURE_NOT_ESTABLISHED
                original_index = ordering[best]
                best_model = model_logs[original_index] if original_index < len(ages) else local_models[original_index-len(ages)]
                best_reference = reference_templates[original_index] if original_index < len(ages) else local_emission[original_index-len(ages), -1]
                whitened_difference = np.linalg.solve(root[pixel], data[pixel]-best_model)
                log_norm = float(ones[pixel]@whitened_difference/norm[pixel])
                log_reference = log_norm+best_reference
                reference = math.exp(log_reference) if math.log(np.finfo(float).tiny) < log_reference < math.log(np.finfo(float).max) else math.nan
                maps["SPECTRAL_AGE"][y, x] = profile_ages[best]
                maps["AGE_LO68"][y, x], maps["AGE_HI68"][y, x], maps["AGE_UPPER95"][y, x] = lo, hi, upper
                maps["CHI2"][y, x], maps["REDCHI2"][y, x] = minimum, reduced
                maps["REF_INTENSITY"][y, x] = reference
                maps["POWERLAW_ALPHA"][y, x], maps["POWERLAW_REDCHI2"][y, x] = pl_alpha[pixel], pl_chi2[pixel]/dof
                maps["CURVATURE_DELTA_CHI2"][y, x] = improvement
                maps["SPEC_VALID"][y, x] = 1
                maps["AGE_FLAGS"][y, x] = flags
                maps["AGE_CONSTRAINED"][y, x] = flags == 0
    counts = {str(flag): {"meaning": meaning, "pixels": int(np.sum((maps["AGE_FLAGS"] & flag) != 0))}
              for flag, meaning in AGE_FLAGS.items()}
    metadata = {"model": model, "clock": "radiative_single_injection_time" if model == "jp_tribble" else "continuous_injection_duration",
        "dynamical_age": "not inferred by this spectral fit", "units": "Myr",
        "field_ug": field, "field_convention": "RMS field for JP-Tribble; homogeneous field for CI",
        "alpha_inj_fixed": alpha, "redshift": z, "observed_frequencies_mhz": frequencies.tolist(),
        "reference_frequency_mhz": reference_frequency, "reference_intensity": "fitted aged model brightness in Jy/beam at the observed reference frequency",
        "age_grid_myr": ages.tolist(), "local_refinement": "65 actual model evaluations across the coarse minimum's adjacent grid cells",
        "selected_pixels": int(mask.sum()), "detected_candidate_pixels": len(yy),
        "fitted_pixels": int(maps["SPEC_VALID"].sum()), "constrained_age_pixels": int(maps["AGE_CONSTRAINED"].sum()),
        "core_treatment": "no central exclusion, distance threshold or circular cut; all supplied selected detections are fitted",
        "degrees_of_freedom": "number of detected bands minus 2 (age and log-amplitude); injection index and field fixed",
        "profile_intervals": {"68_threshold_delta_chi2": 1., "upper95_threshold_delta_chi2": 2.71,
            "scope": "conditional asymptotic likelihood-profile conventions, not posterior credible intervals; boundaries/weak curvature can invalidate nominal coverage",
            "open_endpoints": "NaN if the threshold is not crossed inside the search bounds; physical zero is not substituted as a measured lower error",
            "interpolation": "piecewise linear in log age, linear across the zero-age first segment",
            "disconnected": "endpoints form the acceptance-set hull and the pixel is flagged"},
        "curvature_heuristic": "ageing-model chi-square improves by at least 9 over an independent GLS power law and power-law reduced chi-square >2; nonnested descriptive diagnostic, not significance",
        "flags": counts, "warnings": [
            "Fixed injection index and magnetic field do not cover their uncertainty or joint covariance with the radio flux.",
            "A fitted spectral time or CI injection duration is not an absolute dynamical source lifetime.",
            "Calibration errors are correlated across source pixels and restoring beams correlate neighbouring samples; pixel age errors are not independent.",
            "Positive high-SNR log-flux measurements are fitted; nondetections are omitted rather than used as upper limits.",
            "Absorption, variable particle injection, adiabatic losses, reacceleration, field history and unresolved population mixing are not reconstructed.",
            "The sampled age minimum and profile crossings have finite grid resolution; refine the grid for narrow profiles or final parameter reporting."]}
    json.dumps(metadata, allow_nan=False)
    return {"maps": maps, "metadata": metadata}
