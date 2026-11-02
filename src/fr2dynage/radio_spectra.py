"""Single-injection Jaffe--Perola (JP) synchrotron spectral shapes.

This is a radiative emissivity kernel, not a source dynamical-age estimator.
It assumes rapid pitch-angle isotropization, a spatially and temporally
constant field, and losses from synchrotron emission and Thomson-regime
inverse-Compton scattering of the CMB only. There is no adiabatic expansion,
mixing, transport, absorption, reacceleration, or finite injection duration.
The injected electron distribution is an unbounded ultrarelativistic power
law with p=2*alpha_inj+1. Supported injection indices are 0.5 <= alpha <= 1.5.

For a cooling coefficient a, the aged distribution below gamma_c=1/(a*t)
is N(gamma) proportional to gamma**(-p)*(1-gamma/gamma_c)**(p-2).
The emission uses the full synchrotron function, isotropically averaged over
pitch angle, rather than a delta-function or exponential-cutoff substitute.
The constants and frequency convention are explicit below. See Jaffe &
Perola (1973), A&A 26, 423; Crusius & Schlickeiser (1986), A&A 164, L16;
and Harwood et al. (2013), https://arxiv.org/abs/1308.4137.

Dimensionless integrals are cached on an (alpha, log-frequency/cutoff) grid.
Interpolating log(suppression)+frequency/cutoff preserves the exponential
tail. Independent adaptive-quadrature comparisons cover the interpolation
and asymptotic regimes in the accompanying numerical tests.
"""

from __future__ import annotations

from functools import lru_cache
import math
from numbers import Real

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln, kve, logsumexp, roots_jacobi

from .constants import (C_LIGHT, E_CHARGE, M_ELECTRON, MHZ_HZ, MU_0,
                        MYR_S, RADIATION_CONSTANT, SIGMA_T)


class RadiativeSpectrumError(ValueError):
    """Invalid inputs or unrepresentable normalized radiative spectrum."""


CMB_TEMPERATURE_K = 2.7255
CRITICAL_HZ_PER_T = 3.0 * E_CHARGE / (4.0 * math.pi * M_ELECTRON)
_ALPHA_MIN, _ALPHA_MAX = 0.5, 1.5
_Y_MIN, _Y_MAX = 1.0e-5, 1.0e3


def _scalar(value, label, *, lower=0.0, strict=False):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise RadiativeSpectrumError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (result <= lower if strict else result < lower):
        relation = "greater than" if strict else "at least"
        raise RadiativeSpectrumError(f"{label} must be finite and {relation} {lower}")
    return result


def _real_array(value, label):
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise RadiativeSpectrumError(f"{label} must contain finite real numbers")
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise RadiativeSpectrumError(f"{label} must not contain booleans")
    result = np.asarray(value, dtype=float)
    if not result.size or not np.all(np.isfinite(result)):
        raise RadiativeSpectrumError(f"{label} must contain finite real numbers")
    return result


def jp_loss_coefficient(magnetic_field_ug: float, redshift: float) -> float:
    """Return a [s^-1] for dgamma/dt=-a*gamma², with isotropic pitch angles.

    The synchrotron loss term uses U_B=B²/(2*mu_0), already averaged over
    isotropic pitch angles. The CMB energy density is a_rad*T_0^4*(1+z)^4.
    No extra sin²(theta) or redshift factor belongs in this loss coefficient.
    """
    field = _scalar(magnetic_field_ug, "magnetic_field_ug", strict=True) * 1e-10
    if not math.isfinite(field) or field <= 0.0:
        raise RadiativeSpectrumError("magnetic_field_ug cannot be represented in tesla")
    z = _scalar(redshift, "redshift")
    try:
        energy = field * field / (2.0 * MU_0) + RADIATION_CONSTANT * (CMB_TEMPERATURE_K * (1.0 + z)) ** 4
        result = 4.0 * SIGMA_T * energy / (3.0 * M_ELECTRON * C_LIGHT)
    except OverflowError as exc:
        raise RadiativeSpectrumError("field/redshift loss coefficient is unrepresentable") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise RadiativeSpectrumError("field/redshift loss coefficient is unrepresentable")
    return result


def jp_cutoff_frequency_mhz(age_myr, magnetic_field_ug: float, redshift: float):
    """Observed critical frequency at gamma_c for pitch angle pi/2 [MHz].

    This explicitly defined cooling-cutoff frequency is not a model-free
    measured spectral break. Zero-age values are +inf. Ages are source-frame
    proper times; redshift enters the observed frequency via (1+z).
    """
    ages = _real_array(age_myr, "age_myr")
    if np.any(ages < 0.0):
        raise RadiativeSpectrumError("age_myr must be nonnegative")
    field = _scalar(magnetic_field_ug, "magnetic_field_ug", strict=True)
    z = _scalar(redshift, "redshift")
    loss = jp_loss_coefficient(field, z)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cutoff = CRITICAL_HZ_PER_T * field * 1e-10 / ((loss * ages * MYR_S) ** 2 * MHZ_HZ * (1.0 + z))
    return float(cutoff) if cutoff.ndim == 0 else cutoff


def _log_moment(alpha):
    """log ∫x^(alpha-1) R(x) dx; valid for alpha > -1/3."""
    return (alpha * np.log(2.0) - np.log(alpha + 1.0)
            + gammaln((alpha + 11.0 / 3.0) / 2.0)
            + gammaln((alpha + 1.0 / 3.0) / 2.0)
            + 0.5 * np.log(np.pi) - np.log(2.0)
            + gammaln((alpha + 3.0) / 2.0)
            - gammaln((alpha + 4.0) / 2.0))


def _log_isotropic_kernel(x):
    """log R(x), R=∫_0^(pi/2) sin²(theta) F(x/sin(theta)) dtheta.

    The Bessel-product expression of Crusius & Schlickeiser is evaluated
    with exponentially scaled Bessel functions to avoid underflow. The
    large-x expansion avoids cancellation between nearly equal products.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = x < 1e-8
    large = x > 200.0
    middle = ~(small | large)
    # F(x) ~ 2^(2/3)*Gamma(2/3)*x^(1/3), with the pitch-angle moment.
    small_constant = (2.0 / 3.0 * np.log(2.0) + gammaln(2.0 / 3.0)
                      + 0.5 * np.log(np.pi) + gammaln(4.0 / 3.0)
                      - np.log(2.0) - gammaln(11.0 / 6.0))
    result[small] = small_constant + np.log(x[small]) / 3.0
    result[large] = np.log(np.pi / 2.0) - x[large] + np.log1p(-11.0 / (18.0 * x[large]))
    current = x[middle]
    a, b = kve(4.0 / 3.0, current / 2.0), kve(1.0 / 3.0, current / 2.0)
    bracket = a * b - 0.3 * current * (a * a - b * b)
    result[middle] = 2.0 * np.log(current) - np.log(2.0) + np.log(bracket) - current
    return result


@lru_cache(maxsize=1)
def _kernel_interpolation():
    grid = np.linspace(np.log(1e-12), np.log(1e7), 12000)
    # Interpolate log(R)+x so the exponential tail stays exact.
    return grid, _log_isotropic_kernel(np.exp(grid)) + np.exp(grid)


def _fast_log_kernel(x):
    grid, values = _kernel_interpolation()
    return np.interp(np.log(x), grid, values) - x


def _quadrature_log_suppression(y, alpha, order=512):
    """Gaussian-Jacobi integral with the cooling endpoint weight explicit."""
    exponent = 2.0 * alpha - 1.0
    nodes, weights = roots_jacobi(order, 0.0, exponent)
    u = (nodes + 1.0) / 2.0
    # ∫_0^1 u^exponent*f(u)du; nodes omit both endpoints.
    logweights = np.log(weights) - (exponent + 1.0) * np.log(2.0)
    y = np.atleast_1d(y)
    x = y[:, None] / (1.0 - u[None]) ** 2
    values = (logweights[None] - (2.0 * alpha + 1.0) * np.log1p(-u[None])
              + _fast_log_kernel(x))
    return (np.log(2.0) + alpha * np.log(y)
            + logsumexp(values, axis=1) - _log_moment(alpha))


@lru_cache(maxsize=1)
def _suppression_interpolation():
    alphas = np.linspace(_ALPHA_MIN, _ALPHA_MAX, 51)
    logs = np.linspace(np.log(_Y_MIN), np.log(_Y_MAX), 501)
    y = np.exp(logs)
    values = np.stack([_quadrature_log_suppression(y, alpha) + y for alpha in alphas])
    return RegularGridInterpolator((alphas, logs), values, bounds_error=True)


def _log_suppression(y, alpha):
    y = np.asarray(y, dtype=float)
    result = np.empty_like(y)
    low, high = y < _Y_MIN, y > _Y_MAX
    middle = ~(low | high)
    # First-order cooling expansion. Its omitted endpoint term is O(y^(alpha+1/3)).
    coefficient = (2.0 * alpha - 1.0) * np.exp(_log_moment(alpha - 0.5) - _log_moment(alpha))
    result[low] = np.log1p(-coefficient * np.sqrt(y[low]))
    # Laplace expansion of the same integral, including its first correction.
    constant = np.log(np.pi / 2.0) + gammaln(2.0 * alpha) - (2.0 * alpha - 1.0) * np.log(2.0) - _log_moment(alpha)
    correction = -alpha * alpha - 0.5 * alpha - 11.0 / 18.0
    result[high] = (-y[high] - alpha * np.log(y[high]) + constant
                    + np.log1p(correction / y[high]))
    if np.any(middle):
        yy = y[middle]
        coordinates = np.column_stack((np.full(yy.size, alpha), np.log(yy)))
        result[middle] = _suppression_interpolation()(coordinates) - yy
    return result


def _jp_log_components(frequencies_mhz, age_myr, magnetic_field_ug: float,
                       alpha_inj: float, redshift: float, *, reference_frequency_mhz):
    frequencies = _real_array(frequencies_mhz, "frequencies_mhz")
    if frequencies.ndim != 1 or np.any(frequencies <= 0.0):
        raise RadiativeSpectrumError("frequencies_mhz must be a nonempty positive one-dimensional array")
    ages = _real_array(age_myr, "age_myr")
    if np.any(ages < 0.0):
        raise RadiativeSpectrumError("age_myr must be nonnegative")
    field = _scalar(magnetic_field_ug, "magnetic_field_ug", strict=True)
    alpha = _scalar(alpha_inj, "alpha_inj")
    if not _ALPHA_MIN <= alpha <= _ALPHA_MAX:
        raise RadiativeSpectrumError("alpha_inj must lie in [0.5, 1.5] for this JP kernel")
    z = _scalar(redshift, "redshift")
    reference = _scalar(reference_frequency_mhz, "reference_frequency_mhz", strict=True)
    loss = jp_loss_coefficient(field, z)
    loss_time = loss * ages * MYR_S
    if not np.all(np.isfinite(loss_time)) or np.any(loss_time >= 1.0):
        raise RadiativeSpectrumError("age/field leaves the ultrarelativistic cooling domain (gamma_c <= 1)")
    coefficient = (1.0 + z) * MHZ_HZ / (CRITICAL_HZ_PER_T * field * 1e-10)
    with np.errstate(over="ignore", invalid="ignore"):
        cooling = coefficient * loss_time ** 2
        y = cooling[..., None] * frequencies
        reference_y = cooling * reference
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(reference_y)):
        raise RadiativeSpectrumError("frequency/cooling cutoff ratio is unrepresentable")
    return (-alpha * np.log(frequencies / reference) + _log_suppression(y, alpha),
            _log_suppression(reference_y, alpha)[..., None])


def jp_log_emissivity(frequencies_mhz, age_myr, magnetic_field_ug: float,
                      alpha_inj: float, redshift: float, *, reference_frequency_mhz=1000.0):
    """Return log(S_nu(t)/S_ref(0)), retaining radiative fading.

    The reference is the *unaged* injection flux at the observed reference
    frequency. This dimensionless emissivity retains the suppression caused
    by cooling; unlike :func:`jp_log_spectrum`, it is suitable for adding
    populations of different ages at a specified relative injection
    normalization, including spatial convolution with a restoring beam.
    Absolute electron-density, field-prefactor, distance and volume factors
    are not included; those belong in the forward model's amplitude.

    Frequencies are observed MHz, ages are source proper Myr and B is in
    microgauss. Shape is age.shape+(number_of_frequencies,). All physical
    assumptions and domain restrictions are those of this module.
    """
    emissivity, _ = _jp_log_components(frequencies_mhz, age_myr, magnetic_field_ug,
        alpha_inj, redshift, reference_frequency_mhz=reference_frequency_mhz)
    return emissivity


def jp_log_spectrum(frequencies_mhz, age_myr, magnetic_field_ug: float,
                    alpha_inj: float, redshift: float, *, reference_frequency_mhz=1000.0):
    """Return log(S_nu(t)/S_ref(t)), normalized to the *aged* reference flux.

    Frequencies and reference frequency are observed MHz; ages are source
    proper Myr; B is in microgauss. B, injection index and z are scalars.
    The normalization is exactly unity at the supplied reference frequency.
    Use this function for fitting very old spectra whose linear fluxes can
    underflow. For sums/convolutions at a common injection normalization,
    use :func:`jp_log_emissivity` instead: normalization of each aged
    spectrum separately would incorrectly remove its radiative fading.
    No Doppler correction is included.
    """
    emissivity, reference = _jp_log_components(frequencies_mhz, age_myr, magnetic_field_ug,
        alpha_inj, redshift, reference_frequency_mhz=reference_frequency_mhz)
    return emissivity - reference


def jp_spectrum(frequencies_mhz, age_myr, magnetic_field_ug: float,
                alpha_inj: float, redshift: float, *, reference_frequency_mhz=1000.0):
    """Return S_nu/S_ref; arguments and shape follow :func:`jp_log_spectrum`.

    Vanishing high-frequency flux can underflow to zero. An overflowing
    ratio raises an error; :func:`jp_log_spectrum` remains usable for it.
    """
    values = jp_log_spectrum(frequencies_mhz, age_myr, magnetic_field_ug,
                             alpha_inj, redshift, reference_frequency_mhz=reference_frequency_mhz)
    if np.any(values > np.log(np.finfo(float).max)):
        raise RadiativeSpectrumError("normalized flux ratio overflows; use jp_log_spectrum")
    with np.errstate(under="ignore"):
        return np.exp(values)
