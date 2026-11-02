from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fr2dynage.spectral_maps import (
    INSUFFICIENT_BANDS,
    INVALID_FLUX,
    INVALID_RMS,
    INVALID_WEIGHT,
    LOW_SNR,
    MASKED,
    NUMERICAL_FAILURE,
    POWER_LAW_MISMATCH,
    SpectralMapError,
    fit_power_law_maps,
)


FREQUENCIES = np.array([150.0, 300.0, 600.0, 1200.0])


def _cube(alpha=0.7, amplitude=1.0, shape=(2, 3)):
    reference = np.exp(np.mean(np.log(FREQUENCIES)))
    alpha = np.broadcast_to(alpha, shape)
    amplitude = np.broadcast_to(amplitude, shape)
    return amplitude[None] * (FREQUENCIES[:, None, None] / reference) ** -alpha[None]


def test_recovers_spatially_varying_spectral_index_and_reference_flux():
    alpha = np.array([[0.4, 0.6, 0.8], [1.0, 1.2, 1.4]])
    amplitude = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    cube = _cube(alpha, amplitude)
    result = fit_power_law_maps(FREQUENCIES, cube, 0.01 * cube)
    assert np.all(result.valid)
    assert result.alpha == pytest.approx(alpha, abs=2e-14)
    assert result.reference_flux == pytest.approx(amplitude, rel=2e-14)
    assert np.all(result.n_bands == 4)
    assert np.all(result.chi2 < 1e-24)
    assert result.flags.dtype == np.uint16
    assert result.alpha.shape == result.valid.shape == (2, 3)


def test_known_error_is_not_rescaled_by_near_zero_chi_square():
    cube = _cube()
    result = fit_power_law_maps(FREQUENCIES, cube, 0.05 * cube)
    expected = np.sqrt(np.log1p(0.05**2)) / np.sqrt(np.sum((np.log(FREQUENCIES) - np.mean(np.log(FREQUENCIES))) ** 2))
    assert result.alpha_error == pytest.approx(np.full((2, 3), expected))
    assert np.all(result.alpha_error > 0.0)


def test_heteroscedastic_fit_agrees_with_independent_weighted_design_matrix():
    cube = _cube(shape=(1, 1))
    cube[:, 0, 0] *= np.array([0.99, 1.03, 0.98, 1.01])
    errors = cube * np.array([0.03, 0.06, 0.02, 0.05])[:, None, None]
    calibration = np.array([0.02, 0.01, 0.03, 0.01])
    result = fit_power_law_maps(FREQUENCIES, cube, errors, calibration_fractions=calibration)
    x = np.log(FREQUENCIES / result.reference_frequency_mhz)
    y = np.log(cube[:, 0, 0])
    sigma = np.sqrt(np.log1p((errors[:, 0, 0] / cube[:, 0, 0])**2 + calibration**2))
    design = np.column_stack((np.ones(4), x))
    parameters = np.linalg.lstsq(design / sigma[:, None], y / sigma, rcond=None)[0]
    covariance = np.linalg.inv((design / sigma[:, None]).T @ (design / sigma[:, None]))
    assert result.alpha[0, 0] == pytest.approx(-parameters[1])
    assert result.alpha_error[0, 0] == pytest.approx(np.sqrt(covariance[1, 1]))
    assert result.chi2[0, 0] == pytest.approx(np.sum(((y - design @ parameters) / sigma)**2))


def test_per_band_rms_and_scalar_rms_are_supported():
    cube = _cube()
    per_band = fit_power_law_maps(FREQUENCIES, cube, np.full(4, 0.01))
    scalar = fit_power_law_maps(FREQUENCIES, cube, 0.01)
    assert np.all(per_band.valid)
    assert per_band.alpha == pytest.approx(scalar.alpha)


def test_calibration_error_increases_conditional_error():
    cube = _cube()
    thermal = fit_power_law_maps(FREQUENCIES, cube, cube * 0.01)
    calibration = fit_power_law_maps(FREQUENCIES, cube, cube * 0.01, calibration_fractions=0.1)
    assert np.all(calibration.alpha_error > thermal.alpha_error)
    assert calibration.alpha == pytest.approx(thermal.alpha)


def test_default_requires_every_band_but_optional_partial_fit_is_usable():
    cube = _cube()
    cube[0, 0, 0] = np.nan
    strict = fit_power_law_maps(FREQUENCIES, cube, 0.01)
    partial = fit_power_law_maps(FREQUENCIES, cube, 0.01, require_all_bands=False)
    assert not strict.valid[0, 0]
    assert strict.flags[0, 0] & INVALID_FLUX
    assert strict.flags[0, 0] & INSUFFICIENT_BANDS
    assert partial.valid[0, 0]
    assert partial.n_bands[0, 0] == 3
    assert partial.alpha[0, 0] == pytest.approx(0.7)
    assert partial.flags[0, 0] & INVALID_FLUX
    assert not partial.flags[0, 0] & INSUFFICIENT_BANDS


def test_mask_low_snr_and_invalid_local_noise_have_distinct_flags():
    cube = _cube()
    noise = cube * 0.01
    noise[0, 0, 1] = cube[0, 0, 1]
    noise[0, 1, 0] = 0.0
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    result = fit_power_law_maps(FREQUENCIES, cube, noise, mask=mask)
    assert result.flags[0, 0] & MASKED
    assert result.flags[0, 1] & LOW_SNR
    assert result.flags[1, 0] & INVALID_RMS
    assert not result.valid[0, 0] and not result.valid[0, 1] and not result.valid[1, 0]
    assert result.valid[1, 1]
    for data in (result.alpha, result.alpha_error, result.chi2, result.reduced_chi2, result.reference_flux):
        assert np.all(np.isnan(data[~result.valid]))


def test_signal_exactly_at_snr_threshold_is_retained():
    cube = _cube(amplitude=1e200)
    result = fit_power_law_maps(FREQUENCIES, cube, cube / 5.0)
    assert np.all(result.valid)


def test_curved_spectrum_flags_poor_power_law_without_hiding_kinematic_pixel():
    cube = _cube()
    cube[2] *= 2.0
    result = fit_power_law_maps(FREQUENCIES, cube, cube * 0.01)
    assert np.all(result.valid)
    assert np.all(result.reduced_chi2 > 4)
    assert np.all(result.flags & POWER_LAW_MISMATCH)


@pytest.mark.parametrize("amplitude", [1e-250, 1e250])
def test_extreme_finite_fluxes_do_not_overflow_ratios(amplitude):
    cube = _cube(amplitude=amplitude)
    result = fit_power_law_maps(FREQUENCIES, cube, cube * 0.05)
    assert np.all(result.valid)
    assert result.alpha == pytest.approx(np.full((2, 3), 0.7), abs=2e-12)
    assert np.all(np.isfinite(result.reference_flux))


def test_unrepresentable_relative_error_is_flagged_instead_of_a_false_zero_error():
    cube = np.full((4, 1, 1), 1e300)
    result = fit_power_law_maps(FREQUENCIES, cube, 1e-300)
    assert not result.valid[0, 0]
    assert result.flags[0, 0] & INVALID_WEIGHT
    assert result.flags[0, 0] & INSUFFICIENT_BANDS


def test_unresolved_frequency_leverage_is_flagged_numerically():
    frequencies = np.array([1e300])
    for _ in range(3):
        frequencies = np.append(frequencies, np.nextafter(frequencies[-1], np.inf))
    result = fit_power_law_maps(frequencies, np.ones((4, 1, 1)), 0.01)
    assert not result.valid[0, 0]
    assert result.flags[0, 0] & NUMERICAL_FAILURE


def test_summary_of_valid_diagnostics_is_strict_json_encodable():
    result = fit_power_law_maps(FREQUENCIES, _cube(), 0.01)
    summary = {"reference_frequency_mhz": result.reference_frequency_mhz,
               "valid_pixels": int(np.sum(result.valid)),
               "median_alpha": float(np.median(result.alpha[result.valid]))}
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("frequencies", [[150.0, 300.0], [150.0, 150.0, 600.0, 1200.0],
    [150.0, np.nan, 600.0, 1200.0], [0.0, 300.0, 600.0, 1200.0],
    [True, 300.0, 600.0, 1200.0], ["150", "300", "600", "1200"]])
def test_invalid_frequency_configuration_is_rejected(frequencies):
    with pytest.raises(SpectralMapError):
        fit_power_law_maps(frequencies, _cube(), 0.01)


@pytest.mark.parametrize("kwargs", [
    {"snr_min": True}, {"snr_min": 2.0}, {"snr_min": float("inf")},
    {"min_bands": True}, {"min_bands": 2}, {"min_bands": 5},
    {"require_all_bands": "true"}, {"calibration_fractions": -0.1},
    {"calibration_fractions": True}, {"calibration_fractions": [0.1, 0.1]},
    {"mask": np.ones((3, 2))}, {"mask": np.full((2, 3), 0.5)},
])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(SpectralMapError):
        fit_power_law_maps(FREQUENCIES, _cube(), 0.01, **kwargs)


@pytest.mark.parametrize("rms", [0.0, -1.0, float("nan"), [0.01, 0.01, 0.01, 0.0],
    [True, 0.01, 0.01, 0.01],
    np.ones((2, 3)), np.ones((4, 1, 1))])
def test_invalid_global_rms_settings_are_rejected(rms):
    with pytest.raises(SpectralMapError):
        fit_power_law_maps(FREQUENCIES, _cube(), rms)


def test_wrong_flux_shape_is_rejected():
    with pytest.raises(SpectralMapError, match="shape"):
        fit_power_law_maps(FREQUENCIES, np.ones((3, 2, 3)), 0.01)


def test_bundled_demo_generator_matches_known_spectral_truth_and_refuses_overwrites(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    script = Path(__file__).resolve().parents[1] / "examples/generate_map_demo.py"
    spec = importlib.util.spec_from_file_location("fr2dynage_demo_generator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "demo"
    module.generate(output)
    truth = json.loads((output / "truth.json").read_text())
    for label in ("expansion", "flow"):
        config = json.loads((output / truth[label]["configuration"]).read_text())
        cube = np.stack([fits.getdata(output / point["path"]) for point in config["images"]])
        mask = fits.getdata(output / config["selection"]["mask"]).astype(bool)
        result = fit_power_law_maps([point["frequency_mhz"] for point in config["images"]], cube,
            [point["rms_jy_per_beam"] for point in config["images"]],
            calibration_fractions=[point["calibration_fraction"] for point in config["images"]], mask=mask)
        expected = fits.getdata(output / truth[label]["alpha_truth_fits"])
        assert np.array_equal(result.valid, mask)
        assert result.alpha[result.valid] == pytest.approx(expected[result.valid], abs=2e-14)
        px, py = truth[label]["pixel_xy"]
        assert result.alpha[py, px] == pytest.approx(truth[label]["expected_alpha"])
    flow_truth = truth["flow"]
    x, y = flow_truth["pixel_xy"]
    assert flow_truth["expected_distance_kpc"] > 1.1 * np.hypot(x - 8, y - 10)
    original = (output / "flow.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.generate(output)
    assert (output / "flow.json").read_bytes() == original
