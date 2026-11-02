"""A central beam belongs in source diagnostics without a fabricated core age."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

pytest.importorskip('astropy')
from astropy.wcs import WCS

from fr2dynage.core_diagnostics import CoreDiagnosticError, analyze_core_region


@pytest.fixture
def case():
    frequency = np.array([100., 300., 600., 1000., 3000., 10000.])
    flux = np.broadcast_to(.02*(frequency[:, None, None]/1000.)**-.62, (6, 3, 4)).copy()
    rms = flux*.01
    calibration = np.full(6, .05)
    correlation = np.eye(6)
    core = np.ones((3, 4), bool)
    selected = core.copy()
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2., 2.]
    wcs.wcs.crval = [359.9999, -27.]
    wcs.wcs.cdelt = [-.001, .001]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    reference = SimpleNamespace(wcs=wcs)
    return [frequency, flux, rms, calibration, correlation, core, selected, reference]


def test_powerlaw_core_diagnostic_recovers_spectrum_and_never_returns_age(case):
    result = analyze_core_region(*case, core_position_icrs_deg=[359.9999, -27.])
    spectrum = result['central_spectrum']
    fit = spectrum['power_law']
    assert result['status'] == 'unresolved_age'
    assert result['spectral_age_myr'] is None
    assert result['transport_age_myr'] is None
    assert result['detected_pixels'] == 12
    assert spectrum['pixel_xy'] == [1, 1]
    assert fit['alpha_positive_convention'] == pytest.approx(.62, abs=1e-13)
    assert fit['alpha_error_conditional'] > 0
    assert fit['chi2'] < 1e-20
    assert fit['degrees_of_freedom'] == 4
    np.testing.assert_allclose(spectrum['model_jy_per_beam'], case[1][:, 1, 1], rtol=1e-13)
    assert 'not integrated core flux' in result['interpretation']
    assert 'do not enter' in ' '.join(result['warnings'])
    json.dumps(result, allow_nan=False)


def test_shared_calibration_changes_slope_uncertainty_consistently(case):
    independent = analyze_core_region(*case, core_position_icrs_deg=[359.9999, -27.])
    case[4] = np.ones((6, 6))
    shared = analyze_core_region(*case, core_position_icrs_deg=[359.9999, -27.])
    independent_fit = independent['central_spectrum']['power_law']
    shared_fit = shared['central_spectrum']['power_law']
    assert shared_fit['alpha_error_conditional'] < independent_fit['alpha_error_conditional']/4
    # A completely shared multiplicative scale leaves the slope almost thermal-noise limited.
    x = np.log(case[0]/np.exp(np.mean(np.log(case[0]))))
    expected = np.sqrt(np.log1p(.01**2)/np.sum(x*x))
    assert shared_fit['alpha_error_conditional'] == pytest.approx(expected, rel=1e-12)
    cov = np.asarray(shared_fit['log_intensity_covariance'])
    assert cov[0, 1] == pytest.approx(np.log1p(.05**2))


def test_pixel_selection_uses_spherical_distance_across_ra_wrap(case):
    # Pixel x=0 lies beyond 360deg, which is near the supplied small positive RA.
    coordinate = case[-1].wcs.pixel_to_world(0, 1)
    result = analyze_core_region(*case, core_position_icrs_deg=[coordinate.ra.deg, coordinate.dec.deg])
    assert result['central_spectrum']['pixel_xy'] == [0, 1]
    assert result['central_spectrum']['sky_icrs_deg'][0] < .01


def test_unavailable_bands_are_null_and_model_remains_labelled_description(case):
    case[1][0, :, :] = np.nan
    case[1][1, :, :] = -1
    result = analyze_core_region(*case, core_position_icrs_deg=[359.9999, -27.])
    spectrum = result['central_spectrum']
    assert spectrum['usable_bands'] == 4
    assert spectrum['observed_jy_per_beam'][:2] == [None, None]
    assert spectrum['rms_jy_per_beam'][:2] == [None, None]
    assert all(value > 0 for value in spectrum['model_jy_per_beam'])
    assert spectrum['power_law']['alpha_positive_convention'] == pytest.approx(.62)
    json.dumps(result, allow_nan=False)


def test_source_selection_is_respected_and_no_core_data_remains_explicit(case):
    case[6][:] = False
    result = analyze_core_region(*case)
    assert result['core_mask_pixels'] == 12
    assert result['source_selected_core_pixels'] == 0
    assert result['status'] == 'no_detected_core_spectrum'
    assert result['central_spectrum'] is None
    assert result['transport_age_myr'] is None
    json.dumps(result, allow_nan=False)


def test_fallback_selection_picks_highest_snr_reference_band(case):
    case[2][:, 2, 3] *= .1
    result = analyze_core_region(*case)
    assert result['central_spectrum']['pixel_xy'] == [3, 2]
    assert 'highest-SNR' in result['central_spectrum']['selection']


def test_supplied_core_position_chooses_nearest_detected_pixel(case):
    case[6][1, 1] = False
    result = analyze_core_region(*case, core_position_icrs_deg=[359.9999, -27.])
    assert result['central_spectrum']['pixel_xy'] != [1, 1]
    assert result['source_selected_core_pixels'] == 11


@pytest.mark.parametrize('change', [
    lambda c: c.__setitem__(0, [100., 100., 600., 1000., 3000., 10000.]),
    lambda c: c.__setitem__(0, [100., 300., 600., float('nan'), 3000., 10000.]),
    lambda c: c.__setitem__(2, np.ones((6, 3, 3))),
    lambda c: c.__setitem__(3, [-.1]*6),
    lambda c: c.__setitem__(4, np.ones((5, 5))),
    lambda c: c.__setitem__(4, np.full((6, 6), 2.)),
    lambda c: c.__setitem__(5, np.full((3, 4), .5)),
    lambda c: c.__setitem__(6, np.ones((2, 3))),
])
def test_invalid_inputs_are_rejected(case, change):
    change(case)
    with pytest.raises(ValueError):
        analyze_core_region(*case)


@pytest.mark.parametrize('position', [[360., 0.], [1., 100.], [True, 1.], [1.], ['bad', 1.]])
def test_invalid_core_coordinates_rejected(case, position):
    with pytest.raises(ValueError):
        analyze_core_region(*case, core_position_icrs_deg=position)


@pytest.mark.parametrize('options', [{'snr_min': True}, {'snr_min': 0}, {'min_bands': True},
                                      {'min_bands': 3}, {'min_bands': 7}])
def test_invalid_selection_controls_rejected(case, options):
    with pytest.raises(ValueError):
        analyze_core_region(*case, **options)


def test_numerical_failure_is_reported_without_nonfinite_json(case):
    case[2][:] = np.nextafter(0., 1.)
    case[3][:] = 0
    result = analyze_core_region(*case)
    assert result['central_spectrum']['power_law']['status'] == 'numerical_failure'
    assert result['central_spectrum']['model_jy_per_beam'] == [None]*6
    json.dumps(result, allow_nan=False)
