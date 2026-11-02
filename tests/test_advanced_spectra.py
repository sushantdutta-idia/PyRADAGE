"""Physical limiting forms and independent integrations for TJP and CI."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaincc

from fr2dynage.advanced_spectra import (
    _ci_high_log_suppression, _field_mixture_log_suppression,
    _log_field_moment, _maxwell_quadrature, _model_log_suppression,
    _tribble_high_log_suppression, model_metadata,
    spectral_log_emissivity, spectral_log_spectrum,
)
from fr2dynage.radio_spectra import (
    _log_isotropic_kernel, _log_moment, _log_suppression,
    RadiativeSpectrumError, jp_log_emissivity, jp_loss_coefficient,
)


@pytest.mark.parametrize("model", ["jp_tribble", "ci"])
def test_zero_age_powerlaw_shapes_and_aged_reference_normalization(model):
    frequency = np.array([100., 650., 1000., 3000.])
    ages = np.array([[0., 20.], [60., 200.]])
    emission = spectral_log_emissivity(frequency, ages, 2.5, .7, .1469, model=model)
    shape = spectral_log_spectrum(frequency, ages, 2.5, .7, .1469, model=model)
    assert emission.shape == (2, 2, 4)
    np.testing.assert_allclose(emission[0, 0], -.7*np.log(frequency/1000.), atol=2e-15)
    np.testing.assert_array_equal(shape[..., 2], np.zeros((2, 2)))
    np.testing.assert_allclose(shape, emission-emission[..., 2, None], atol=2e-14)
    assert np.all(np.diff(emission.reshape(4, 4), axis=0) < 0)


def test_maxwell_distribution_has_unit_rms_and_correct_emissivity_moments():
    # alpha=-1 cancels the synchrotron field weighting in this internal
    # quadrature, giving the pure Maxwell probability distribution.
    field, logweights = _maxwell_quadrature(-1., 256)
    weights = np.exp(logweights)
    assert weights.sum() == pytest.approx(1., abs=2e-14)
    assert weights@field**2 == pytest.approx(1., rel=2e-14)
    assert weights@field == pytest.approx(2.*math.sqrt(2./(3.*math.pi)), rel=2e-6)
    for power in [1.5, 1.7, 2., 2.5]:
        assert weights@field**power == pytest.approx(math.exp(_log_field_moment(power)), rel=2e-6)


def test_delta_field_limit_is_exact_common_loss_jp():
    y = np.geomspace(1e-7, 1e4, 40)
    mixture = _field_mixture_log_suppression(y, .73, np.array([1.]), np.array([0.]))
    np.testing.assert_allclose(mixture, _log_suppression(y, .73), atol=1e-13)


@pytest.mark.parametrize("model", ["jp_tribble", "ci"])
def test_cooling_does_not_amplify_common_unaged_emissivity(model):
    y = np.r_[0., np.geomspace(1e-12, 1e9, 1500)]
    for alpha in [.5, .517, .73, 1.17, 1.5]:
        values = _model_log_suppression(y, alpha, model)
        assert np.all(np.isfinite(values))
        assert values.max() <= 0.
        # Sub-grid interpolation/leading-asymptote joins can have errors far
        # below the tested physical integration tolerance; no resolved
        # cooling trend may reverse.
        assert np.max(np.diff(values)) < 5e-6


@pytest.mark.parametrize("alpha,y", [(.5, .001), (.73, .2), (1.11, 2.), (.63, 30.), (1.5, 1000.)])
def test_tjp_table_matches_independent_adaptive_field_integral(alpha, y):
    # This integrates Maxwell volume probability and the B^(alpha+1)
    # synchrotron normalization directly, not the production generalized-
    # Laguerre nodes, adjusted weights or two-dimensional interpolation.
    phase = 4.5*(y/3.)**(2./3.)
    normalization = math.sqrt(2./math.pi)*3.**1.5/math.exp(_log_field_moment(alpha+1.))
    def integrand(field):
        if field == 0:
            return 0.
        return math.exp((alpha+3.)*math.log(field)-1.5*field*field
                        +float(_log_suppression(np.array(y/field), alpha))+phase)
    direct = math.log(normalization*quad(integrand, 0., max(12., 3.*(y/3.)**(1./3.)),
                                          epsabs=1e-8, epsrel=2e-6, limit=400)[0])-phase
    actual = float(_model_log_suppression(np.array(y), alpha, "jp_tribble"))
    assert actual == pytest.approx(direct, abs=6e-4)


@pytest.mark.parametrize("y", [.001, .2, 2., 30.])
def test_tjp_p2_matches_swapped_electron_field_integral_without_jp_table(y):
    # For p=2, swapping the electron and field integrals gives an analytic
    # upper-gamma probability for the Maxwell field. This bypasses every
    # production JP/TJP suppression table and both Gaussian quadratures.
    phase = 4.5*(y/3.)**(2./3.)
    def integrand(x):
        if x == 0:
            return 0.
        field_tail = gammaincc(2.25, 1.5*(y/x)**2)
        if field_tail == 0:
            return 0.
        return math.exp(-.5*math.log(x)+float(_log_isotropic_kernel(np.array(x)))
                        -float(_log_moment(.5))+math.log(field_tail)+phase)
    direct = math.log(quad(integrand, 0., np.inf, epsabs=1e-10, epsrel=2e-9, limit=300)[0])-phase
    actual = float(_model_log_suppression(np.array(y), .5, "jp_tribble"))
    assert actual == pytest.approx(direct, abs=5e-4)


@pytest.mark.parametrize("alpha,y", [(.5, .001), (.73, .2), (1.11, 2.), (.63, 30.), (1.5, 1000.)])
def test_ci_table_matches_analytic_continuous_injection_electron_distribution(alpha, y):
    # The time-integrated electron distribution is analytic. Integrating
    # it against R(x) independently checks the numerical cohort-age integral.
    def integrand(x):
        if x == 0:
            return 0.
        ratio = math.sqrt(y/x)
        if ratio >= 1.:
            population = 1./(2.*alpha*ratio)
        else:
            population = -math.expm1(2.*alpha*math.log1p(-ratio))/(2.*alpha*ratio)
        return math.exp((alpha-1.)*math.log(x)+float(_log_isotropic_kernel(np.array(x)))
                        -float(_log_moment(alpha)))*population
    direct = quad(integrand, 0., y, epsabs=2e-11, epsrel=2e-9, limit=300)[0]
    direct += quad(integrand, y, np.inf, epsabs=2e-11, epsrel=2e-9, limit=300)[0]
    actual = float(_model_log_suppression(np.array(y), alpha, "ci"))
    assert actual == pytest.approx(math.log(direct), abs=6e-4)


@pytest.mark.parametrize("alpha", [.5, .73, 1.1, 1.5])
def test_tjp_saddle_tail_is_continuous_and_matches_adaptive_integral(alpha):
    y = 1e4
    table = float(_model_log_suppression(np.array(y), alpha, "jp_tribble"))
    asymptotic = float(_tribble_high_log_suppression(y, alpha))
    assert table == pytest.approx(asymptotic, abs=4e-4)
    values = _model_log_suppression(np.array([y*(1.-1e-8), y*(1.+1e-8), 1e10]), alpha, "jp_tribble")
    assert abs(values[1]-values[0]) < .001
    assert np.all(np.isfinite(values))
    assert values[2] < values[1]


@pytest.mark.parametrize("alpha", [.5, .73, 1.1, 1.5])
def test_ci_high_frequency_slope_and_tail_coefficient(alpha):
    # A fully cooled active CI spectrum must steepen by exactly 0.5.
    frequency = np.array([1e5, 1e6])
    emission = spectral_log_emissivity(frequency, 300., 3., alpha, .15, model="ci")
    observed_alpha = -(emission[1]-emission[0])/math.log(10.)
    assert observed_alpha == pytest.approx(alpha+.5, abs=2e-14)
    y = np.array([50., 50.00001, 1e10])
    actual = _model_log_suppression(y, alpha, "ci")
    np.testing.assert_allclose(actual, _ci_high_log_suppression(y, alpha), atol=8e-5)


def test_tjp_differs_from_jp_and_extends_its_high_frequency_tail():
    freq = np.array([150., 650., 1400., 6000., 30000.])
    jp = jp_log_emissivity(freq, 100., 2.5, .7, .1469)
    tjp = spectral_log_emissivity(freq, 100., 2.5, .7, .1469, model="jp_tribble")
    assert not np.allclose(jp, tjp, rtol=.01)
    assert tjp[-1] > jp[-1]


@pytest.mark.parametrize("model", ["jp_tribble", "ci"])
def test_loss_frequency_scaling_and_emissivity_mixtures_preserve_fading(model):
    freq, age, b1, b2, z = [150., 650., 1400., 6000.], 60., 2., 8., .1469
    second_age = age*jp_loss_coefficient(b1, z)/jp_loss_coefficient(b2, z)*math.sqrt(b2/b1)
    np.testing.assert_allclose(
        spectral_log_emissivity(freq, age, b1, .73, z, model=model),
        spectral_log_emissivity(freq, second_age, b2, .73, z, model=model), atol=2e-13)
    ages, weights = np.array([0., 150.]), np.array([.2, .8])
    emission = np.exp(spectral_log_emissivity([1000.], ages, b1, .73, z, model=model))
    assert float(weights@emission[:, 0]) < 1.
    normalized = np.exp(spectral_log_spectrum([1000.], ages, b1, .73, z, model=model))
    assert float(weights@normalized[:, 0]) == pytest.approx(1.)


def test_metadata_declares_field_and_clock_conventions_and_is_not_shared():
    tjp = model_metadata("jp_tribble")
    assert "RMS" in tjp["field_convention"]
    assert "B_loss=B_rms" in tjp["loss_history"]
    assert "injection duration" in model_metadata("ci")["clock"]
    tjp["references"].clear()
    assert model_metadata("jp_tribble")["references"]


@pytest.mark.parametrize("override", [
    {"model": "jp"}, {"model": True}, {"model": None},
    {"frequencies_mhz": [100., True]}, {"frequencies_mhz": [0., 1000.]},
    {"frequencies_mhz": [[100., 1000.]]}, {"frequencies_mhz": []},
    {"age_myr": -1.}, {"age_myr": True}, {"age_myr": float("nan")},
    {"magnetic_field_ug": 0.}, {"magnetic_field_ug": 1e-320},
    {"alpha_inj": .49}, {"alpha_inj": 1.51}, {"redshift": -.1},
    {"reference_frequency_mhz": 0.}, {"age_myr": 1e10},
])
def test_invalid_inputs_raise_typed_spectrum_error(override):
    arguments = dict(frequencies_mhz=[100., 1000.], age_myr=30., magnetic_field_ug=2.5,
                     alpha_inj=.7, redshift=.1469)
    arguments.update(override)
    with pytest.raises(RadiativeSpectrumError):
        spectral_log_emissivity(**arguments)
