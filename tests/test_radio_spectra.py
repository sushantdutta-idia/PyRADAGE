"""Independent numerical and physical regressions for the JP emissivity."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from fr2dynage.constants import (C_LIGHT, M_ELECTRON, MU_0, MYR_S,
                                 RADIATION_CONSTANT, SIGMA_T)
from fr2dynage.radio_spectra import (
    CRITICAL_HZ_PER_T, CMB_TEMPERATURE_K, RadiativeSpectrumError,
    _log_isotropic_kernel, _log_moment, _log_suppression,
    jp_cutoff_frequency_mhz, jp_log_emissivity, jp_log_spectrum,
    jp_loss_coefficient, jp_spectrum,
)


@pytest.mark.parametrize("x", [0.001, 0.1, 1.0, 10.0])
def test_isotropic_kernel_matches_independent_pitch_angle_integral(x):
    # This uses the original F(x)=x*integral K_5/3, not the Bessel-product
    # formula or the cached kernel used in production.
    def synchrotron(argument):
        return argument * quad(lambda q: kv(5.0 / 3.0, q), argument, np.inf,
                               epsabs=1e-11, epsrel=1e-9)[0]

    direct = quad(lambda theta: np.sin(theta) ** 2 * synchrotron(x / np.sin(theta)),
                  0.0, np.pi / 2.0, epsabs=1e-10, epsrel=1e-8)[0]
    assert np.exp(_log_isotropic_kernel(np.array(x))) == pytest.approx(direct, rel=2e-6)


@pytest.mark.parametrize("alpha,y", [(.5, 1e-7), (.53, .01), (.73, .2),
                                     (1.11, 2.0), (1.5, 20.0), (.73, 2000.)])
def test_aged_emissivity_agrees_with_adaptive_electron_integral(alpha, y):
    # Change variable x=y+q in the original electron emissivity integral;
    # adaptive quadrature is independent of the production Gaussian-Jacobi
    # integration and its two-dimensional table. Factoring exp(-y) permits
    # independent checks far into the exponentially faint high-frequency tail.
    def integrand(q):
        if q == 0:
            return 0.0
        x = y + q
        cut = -np.expm1(-0.5 * np.log1p(q / y))
        return np.exp((alpha - 1.0) * np.log(x) + (2.0 * alpha - 1.0) * np.log(cut)
                      + _log_isotropic_kernel(np.array(x)) + y)

    direct = np.log(quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-9,
                         limit=300)[0]) - y - _log_moment(alpha)
    assert float(_log_suppression(np.array(y), alpha)) == pytest.approx(direct, abs=3e-4)


def test_zero_age_is_exact_power_law_and_reference_is_exactly_normalized():
    frequencies = np.array([150., 617., 1000., 3000.])
    values = jp_spectrum(frequencies, 0.0, 2.24, .71, .1469)
    assert values == pytest.approx((frequencies / 1000.) ** -.71, rel=2e-15)
    age_grid = np.array([[0., 1.], [100., 300.]])
    output = jp_spectrum(frequencies, age_grid, 2.24, .71, .1469)
    assert output.shape == (2, 2, 4)
    assert np.array_equal(output[..., 2], np.ones((2, 2)))


def test_cooling_steepens_high_frequency_spectrum_with_time():
    ages = np.array([0., 10., 30., 100., 200.])
    values = jp_log_spectrum([150., 3000.], ages, 2.24, .7, .1469)
    high_to_low = values[:, 1] - values[:, 0]
    assert np.all(np.diff(high_to_low) < 0.0)


def test_emissivity_keeps_radiative_fading_and_recovers_normalized_spectrum():
    ages = np.array([0., 20., 60., 150.])
    frequencies = np.array([150., 600., 1000., 3000.])
    emission = jp_log_emissivity(frequencies, ages, 2.24, .7, .1469)
    normalized = jp_log_spectrum(frequencies, ages, 2.24, .7, .1469)
    assert normalized == pytest.approx(emission - emission[:, 2, None], abs=2e-14)
    assert emission[0] == pytest.approx(-.7 * np.log(frequencies / 1000.), abs=2e-15)
    assert np.all(np.diff(emission, axis=0) < 0)
    assert np.all(emission[1:, 2] < 0)


def test_beam_age_mixture_conserves_emissivity_with_injection_weights():
    # A positive linear beam combines emissivities before spectral
    # normalization. Its zero-age limit preserves a common injected power
    # law, while the older component contributes less after cooling.
    frequencies = np.array([150., 600., 1000., 3000.])
    ages = np.array([20., 150.])
    weights = np.array([.3, .7])
    emission = np.exp(jp_log_emissivity(frequencies, ages, 2.24, .7, .1469))
    mixture = np.sum(weights[:, None] * emission, axis=0)
    separate = (weights[0] * np.exp(jp_log_emissivity(frequencies, ages[0], 2.24, .7, .1469))
                + weights[1] * np.exp(jp_log_emissivity(frequencies, ages[1], 2.24, .7, .1469)))
    assert mixture == pytest.approx(separate, rel=1e-14)
    zero = np.exp(jp_log_emissivity(frequencies, np.zeros(2), 2.24, .7, .1469))
    assert weights @ zero == pytest.approx((frequencies / 1000.) ** -.7, rel=1e-14)
    wrongly_normalized = weights @ np.exp(jp_log_spectrum(frequencies, ages, 2.24, .7, .1469))
    assert mixture[2] < 1.0
    assert wrongly_normalized[2] == pytest.approx(1.0)
    assert not np.allclose(mixture / mixture[2], wrongly_normalized)


def test_physical_loss_coefficient_and_redshift_frequency_convention():
    field, z, age = 3.0, .2, 100.0
    u_b = (field * 1e-10) ** 2 / (2.0 * MU_0)
    u_cmb = RADIATION_CONSTANT * (CMB_TEMPERATURE_K * (1 + z)) ** 4
    expected = 4 * SIGMA_T * (u_b + u_cmb) / (3 * M_ELECTRON * C_LIGHT)
    assert jp_loss_coefficient(field, z) == pytest.approx(expected, rel=1e-14)
    gamma_c = 1.0 / (expected * age * MYR_S)
    cutoff = CRITICAL_HZ_PER_T * field * 1e-10 * gamma_c ** 2 / (1e6 * (1 + z))
    assert jp_cutoff_frequency_mhz(age, field, z) == pytest.approx(cutoff, rel=1e-14)
    assert math.isinf(jp_cutoff_frequency_mhz(0., field, z))
    # B_CMB ~= 3.24 microgauss at z=0, and its energy density scales (1+z)^4.
    assert jp_loss_coefficient(1e-6, 1.) / jp_loss_coefficient(1e-6, 0.) == pytest.approx(16., rel=1e-11)


def test_equal_cooling_ratios_give_same_spectrum_for_different_fields():
    z, alpha = .1469, .67
    age1, b1, b2 = 60., 2., 8.
    # y proportional to nu*(a*t)^2/B. Match this at every frequency.
    age2 = age1 * jp_loss_coefficient(b1, z) / jp_loss_coefficient(b2, z) * np.sqrt(b2 / b1)
    f = [150., 650., 1400., 6000.]
    assert jp_log_spectrum(f, age1, b1, alpha, z) == pytest.approx(
        jp_log_spectrum(f, age2, b2, alpha, z), abs=2e-13)


def test_log_spectrum_remains_usable_when_linear_ratios_overflow():
    values = jp_log_spectrum([100., 1000., 10000.], 10000., 3., .7, .15)
    assert np.all(np.isfinite(values))
    assert values[0] > 709 and values[1] == 0 and values[2] < -709
    with pytest.raises(RadiativeSpectrumError, match="use jp_log_spectrum"):
        jp_spectrum([100., 1000., 10000.], 10000., 3., .7, .15)


@pytest.mark.parametrize("override", [
    {"frequencies_mhz": [100., True, 1000.]},
    {"frequencies_mhz": [0., 1000.]},
    {"frequencies_mhz": [[100., 1000.]]},
    {"frequencies_mhz": []},
    {"age_myr": [-1., 10.]},
    {"age_myr": [True, 10.]},
    {"age_myr": float("nan")},
    {"magnetic_field_ug": 0.},
    {"magnetic_field_ug": 1e-320},
    {"alpha_inj": .49},
    {"alpha_inj": 1.51},
    {"alpha_inj": True},
    {"redshift": -.1},
    {"reference_frequency_mhz": 0.},
    {"age_myr": 1e10},
])
def test_invalid_inputs_raise_typed_error(override):
    arguments = dict(frequencies_mhz=[150., 650., 1400.], age_myr=50.,
                     magnetic_field_ug=2.24, alpha_inj=.7, redshift=.1469)
    arguments.update(override)
    with pytest.raises(RadiativeSpectrumError):
        jp_spectrum(**arguments)
