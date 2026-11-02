"""Resolved spectral profiles include central detections and preserve limits."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fr2dynage.resolved_spectra import (
    CURVATURE_NOT_ESTABLISHED, INSUFFICIENT_BANDS, LOWER_BOUND, MASKED,
    OPEN_68_PROFILE, OPEN_UPPER95_PROFILE, POOR_FIT, UPPER_BOUND,
    ResolvedSpectrumError, _profile_interval, fit_resolved_spectra,
)


FREQUENCIES = np.array([80., 150., 350., 650., 1400., 3000., 10000.])


def _case(model="jp_tribble", age=50., *, calibration=.01, fractional_error=.01):
    from fr2dynage.advanced_spectra import spectral_log_emissivity
    observed = .03*np.exp(spectral_log_emissivity(FREQUENCIES, age, 2.4, .65, .1469, model=model))
    flux = np.broadcast_to(observed[:, None, None], (len(observed), 2, 3)).copy()
    rms = fractional_error*flux
    mask = np.ones(flux.shape[1:], dtype=bool)
    return dict(frequencies_mhz=FREQUENCIES, flux_cube=flux, rms_cube=rms,
        calibration_fractions=np.full(len(observed), calibration),
        calibration_correlation=np.eye(len(observed)), source_mask=mask,
        field_ug=2.4, redshift=.1469, alpha_inj=.65, model=model,
        age_bounds_myr=(0., 300.), age_grid_size=401)


@pytest.mark.parametrize("model", ["jp_tribble", "ci"])
def test_known_aged_spectrum_recovery_with_all_selected_central_pixels(model):
    arguments = _case(model)
    result = fit_resolved_spectra(**arguments)
    maps, metadata = result["maps"], result["metadata"]
    assert maps["SPEC_VALID"].sum() == 6
    assert np.all(maps["SPECTRAL_AGE"] == pytest.approx(50., rel=.002))
    assert np.all(maps["REDCHI2"] < 1.)
    assert np.all(maps["REF_INTENSITY"] > 0.)
    assert metadata["fitted_pixels"] == 6
    assert "no central exclusion" in metadata["core_treatment"]
    assert metadata["dynamical_age"].startswith("not inferred")
    assert maps["N_BANDS"][0, 0] == len(FREQUENCIES)
    assert maps["AGE_LO68"][0, 0] < 50 < maps["AGE_HI68"][0, 0]
    assert maps["AGE_UPPER95"][0, 0] > maps["AGE_HI68"][0, 0]
    assert np.all(maps["AGE_CONSTRAINED"])
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("model", ["jp_tribble", "ci"])
def test_zero_age_retained_as_flagged_boundary_and_one_sided_profile(model):
    result = fit_resolved_spectra(**_case(model, 0.))
    maps = result["maps"]
    assert np.all(maps["SPEC_VALID"])
    assert np.all(maps["SPECTRAL_AGE"] == 0.)
    assert np.all((maps["AGE_FLAGS"] & LOWER_BOUND) != 0)
    assert np.all((maps["AGE_FLAGS"] & CURVATURE_NOT_ESTABLISHED) != 0)
    assert np.all((maps["AGE_FLAGS"] & OPEN_68_PROFILE) != 0)
    assert np.all(np.isnan(maps["AGE_LO68"]))
    assert np.all(np.isfinite(maps["AGE_UPPER95"]))
    assert not np.any(maps["AGE_CONSTRAINED"])


def test_missing_bands_are_omitted_and_source_mask_is_only_spatial_exclusion():
    arguments = _case()
    arguments["source_mask"][1, 2] = False
    arguments["flux_cube"][0, 0, 0] = np.nan
    arguments["rms_cube"][0, 0, 0] = np.nan
    arguments["flux_cube"][:4, 1, 0] = np.nan
    result = fit_resolved_spectra(**arguments)["maps"]
    assert result["SPEC_VALID"][0, 0] == 1
    assert result["N_BANDS"][0, 0] == len(FREQUENCIES)-1
    assert result["SPECTRAL_AGE"][0, 0] == pytest.approx(50., rel=.025)
    assert result["AGE_FLAGS"][1, 2] == MASKED
    assert result["AGE_FLAGS"][1, 0] == INSUFFICIENT_BANDS
    assert np.isnan(result["SPECTRAL_AGE"][1, 2])
    assert np.isnan(result["SPECTRAL_AGE"][1, 0])


def test_correlated_calibration_gls_chi_square_matches_direct_profile():
    from fr2dynage.advanced_spectra import spectral_log_emissivity
    from fr2dynage.spectral_maps import _sigma_ln
    arguments = _case(calibration=.04)
    correlation = np.full((len(FREQUENCIES), len(FREQUENCIES)), .6)
    np.fill_diagonal(correlation, 1.)
    arguments["calibration_correlation"] = correlation
    arguments["flux_cube"][:, 0, 0] *= np.exp([.01, -.02, .015, -.01, .005, .01, -.005])
    maps = fit_resolved_spectra(**arguments)["maps"]
    data = arguments["flux_cube"][:, 0, 0]
    error = _sigma_ln(arguments["rms_cube"][:, 0, 0]/data)
    cal = _sigma_ln(arguments["calibration_fractions"])
    covariance = np.diag(error**2)+correlation*cal[:, None]*cal[None, :]
    precision = np.linalg.inv(covariance)
    shape = spectral_log_emissivity(FREQUENCIES, maps["SPECTRAL_AGE"][0, 0], 2.4, .65, .1469, model="jp_tribble")
    difference = np.log(data)-shape
    amplitude = np.ones(len(data))@precision@difference/(np.ones(len(data))@precision@np.ones(len(data)))
    residual = difference-amplitude
    assert maps["CHI2"][0, 0] == pytest.approx(residual@precision@residual, rel=1e-9, abs=1e-10)


def test_power_law_with_different_fixed_injection_slope_does_not_fake_curvature():
    arguments = _case(age=0.)
    arguments["flux_cube"][:] = (.03*(FREQUENCIES/1000.)**-.8)[:, None, None]
    arguments["rms_cube"][:] = .01*arguments["flux_cube"]
    maps = fit_resolved_spectra(**arguments)["maps"]
    assert np.all(maps["SPECTRAL_AGE"] > 0)
    assert np.all((maps["AGE_FLAGS"] & CURVATURE_NOT_ESTABLISHED) != 0)
    assert np.all(maps["POWERLAW_ALPHA"] == pytest.approx(.8, abs=1e-12))
    assert np.all(maps["POWERLAW_REDCHI2"] < 1e-20)
    assert not np.any(maps["AGE_CONSTRAINED"])


def test_bad_spectrum_retains_numeric_fit_with_poor_fit_flag():
    arguments = _case()
    arguments["flux_cube"][3] *= 2.
    maps = fit_resolved_spectra(**arguments)["maps"]
    assert np.all(maps["SPEC_VALID"])
    assert np.all((maps["AGE_FLAGS"] & POOR_FIT) != 0)
    assert not np.any(maps["AGE_CONSTRAINED"])


def test_age_outside_search_is_flagged_without_fabricated_upper_limit():
    arguments = _case(age=80.)
    arguments["age_bounds_myr"] = (0., 10.)
    maps = fit_resolved_spectra(**arguments)["maps"]
    assert np.all(maps["SPECTRAL_AGE"] == 10.)
    assert np.all((maps["AGE_FLAGS"] & UPPER_BOUND) != 0)
    assert np.all((maps["AGE_FLAGS"] & OPEN_UPPER95_PROFILE) != 0)
    assert np.all(np.isnan(maps["AGE_UPPER95"]))


def test_profile_hull_records_disconnected_and_open_sets():
    ages = np.array([0., 1., 2., 4., 8.])
    delta = np.array([3., 0., 4., .5, 3.])
    lower, upper, disconnected, count = _profile_interval(ages, delta, 1.)
    assert lower == pytest.approx(2/3)
    assert 4 < upper < 8
    assert disconnected and count == 2
    assert math.isnan(_profile_interval(ages, np.zeros(5), 1.)[0])
    assert math.isnan(_profile_interval(ages, np.zeros(5), 1.)[1])


@pytest.mark.parametrize("change, match", [
    ({"model": "jp"}, "model"), ({"field_ug": -1.}, "field"),
    ({"alpha_inj": .2}, "alpha"), ({"redshift": True}, "redshift"),
    ({"age_grid_size": True}, "age_grid_size"), ({"age_grid_size": 2}, "age_grid_size"),
    ({"age_bounds_myr": [0., 0.]}, "age_bounds"), ({"min_bands": 3}, "min_bands"),
    ({"snr_min": 1.}, "snr_min"), ({"calibration_fractions": True}, "booleans"),
    ({"source_mask": [[1, 2, 1], [1, 1, 1]]}, "source_mask"),
])
def test_invalid_settings_fail_cleanly(change, match):
    arguments = _case()
    arguments.update(change)
    with pytest.raises(ValueError, match=match):
        fit_resolved_spectra(**arguments)


def test_empty_selected_footprint_returns_finite_metadata_and_no_fits():
    arguments = _case()
    arguments["source_mask"][:] = False
    result = fit_resolved_spectra(**arguments)
    assert not np.any(result["maps"]["SPEC_VALID"])
    assert np.all(result["maps"]["AGE_FLAGS"] == MASKED)
    assert result["metadata"]["fitted_pixels"] == 0
    json.dumps(result["metadata"], allow_nan=False)
