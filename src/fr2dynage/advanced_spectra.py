"""Free-streaming JP--Tribble and continuous-injection synchrotron spectra.

These are radiative-emissivity kernels, not source dynamical models. The
JP--Tribble (TJP) variant follows Hardcastle (2013), arXiv:1306.1640, equations
4--6 and section 3.2: isotropic Gaussian field components, a Maxwell field
magnitude with sigma=B_rms/sqrt(3), spatially uniform electron density and a
common JP electron-loss history in B_loss=B_rms. Local emitting fields vary;
they are not substituted individually into the cooling coefficient. This is
the rapid-diffusion/free-streaming limit, not a mixture of isolated regions
each cooling in its own local field. The run's adopted field, whether a
prescribed fraction of B_eq or an explicit user value, is interpreted as the
RMS field in this explicitly declared random-field model.

The CI kernel is constant-rate injection from time zero until the observing
epoch in a constant field. It averages JP emissivities over electron ages
0..T; T is the injection duration, not the travel time of a single plasma
parcel. The spectrum approaches alpha_inj+0.5 at high frequency. Neither
kernel models absorption, adiabatic evolution, reacceleration, finite upper
electron cutoffs, Doppler effects or an independently measured AGN lifetime.

The full isotropic synchrotron kernel is supplied by radio_spectra. Cached
dimensionless tables keep repeated spectral fits inexpensive. Independent
quadratures and limiting forms are tested in test_advanced_spectra.py.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln, logsumexp, roots_genlaguerre, roots_legendre

from .constants import MHZ_HZ, MYR_S
from .radio_spectra import (CRITICAL_HZ_PER_T, RadiativeSpectrumError,
                            _log_moment, _log_suppression, _real_array,
                            _scalar, jp_loss_coefficient)

MODELS = ("jp_tribble", "ci")
MODEL_DESCRIPTIONS = {
    "jp_tribble": {
        "name": "JP--Tribble", "variant": "rapid-diffusion/free-streaming",
        "field_convention": "input B is RMS sqrt(<B_local^2>); Gaussian component sigma=B/sqrt(3)",
        "field_prescription": "RMS field supplied by the run configuration; the kernel does not infer B_eq or impose a field fraction",
        "loss_history": "common JP losses in B_loss=B_rms plus inverse-Compton CMB losses",
        "emission": "Maxwell-distributed local field strengths; isotropic pitch angles; uniform electron density",
        "clock": "single-injection radiative elapsed time; source dynamical lifetime requires a separate physical model",
        "references": ["https://arxiv.org/abs/1306.1640", "https://arxiv.org/abs/1308.4137"],
        "warning": "Not the isolated-cell/local-field-loss Tribble variant and not a claim of bitwise agreement with BRATS or synchrofit.",
    },
    "ci": {
        "name": "Continuous injection (CI)", "variant": "constant rate, still active",
        "field_convention": "constant homogeneous input magnetic field",
        "field_prescription": "Homogeneous field supplied by the run configuration; the kernel does not infer B_eq or impose a field fraction",
        "loss_history": "JP synchrotron and inverse-Compton CMB losses for every injected electron cohort",
        "emission": "uniform injection-age integral from 0 to T, normalized per unit total injected population",
        "clock": "continuous-injection duration T; not a single-parcel plasma transport time or independently measured global lifetime",
        "references": ["https://arxiv.org/abs/1612.04390"],
        "warning": "CI represents a superposition of cohort ages; it must not be silently equated to an advective parcel-age kernel.",
    },
}


def model_metadata(model):
    """Return JSON-safe model conventions without mutable shared state."""
    if not isinstance(model, str) or model not in MODELS:
        raise RadiativeSpectrumError("spectral model must be jp_tribble or ci")
    return deepcopy(MODEL_DESCRIPTIONS[model])


def _log_field_moment(power):
    """log E[q**power] for a Maxwell magnitude q with E[q²]=1."""
    return .5*power*math.log(2./3.) + gammaln((power+3.)/2.)-gammaln(1.5)


def _maxwell_quadrature(alpha, order=128):
    """Emission-weighted field nodes q and normalized log weights.

    x=3q²/2 has a gamma distribution. Including q^(alpha+1) in
    the generalized-Laguerre weight avoids resolving a fractional endpoint
    separately; division by its moment normalizes the unaged spectrum.
    """
    nodes, weights = roots_genlaguerre(order, (alpha+2.)/2.)
    with np.errstate(divide="ignore"):
        logweights = np.log(weights)-gammaln((alpha+4.)/2.)
    return np.sqrt(2.*nodes/3.), logweights


def _tribble_quadrature_log_suppression(y, alpha, order=128):
    field, logweights = _maxwell_quadrature(alpha, order)
    return _field_mixture_log_suppression(y, alpha, field, logweights)


def _field_mixture_log_suppression(y, alpha, field_ratios, emission_logweights):
    """Common-loss mixture; weights include the local unaged B^(alpha+1)."""
    y = np.asarray(y, dtype=float)
    return logsumexp(_log_suppression(y[..., None]/field_ratios, alpha)
                     +emission_logweights, axis=-1)


def _ci_quadrature_log_suppression(y, alpha, order=128):
    nodes, weights = roots_legendre(order)
    u = .5*(nodes+1.)
    y = np.asarray(y, dtype=float)
    return logsumexp(_log_suppression(y[..., None]*u**2, alpha)+np.log(.5*weights), axis=-1)


def _tribble_high_log_suppression(y, alpha):
    """Saddle-point expansion through relative order y^(-2/3).

    At large y the field integral has phase 3q²/2+y/q, with
    saddle q=(y/3)^(1/3), curvature 9. Factoring this phase before
    interpolation also avoids interpolating its large exponential term.
    """
    y = np.asarray(y, dtype=float)
    q = np.cbrt(y/3.)
    n = 2.*alpha+3.
    jp_constant = (math.log(math.pi/2.)+gammaln(2.*alpha)
                   -(2.*alpha-1.)*math.log(2.)-_log_moment(alpha))
    correction = (-2.*alpha**2+11.*alpha+8.)/18.
    return (math.log(2.*math.sqrt(3.))+jp_constant-_log_field_moment(alpha+1.)
            -alpha*np.log(y)+n*np.log(q)-4.5*q*q+np.log1p(correction/(q*q)))


def _ci_high_log_suppression(y, alpha):
    # For the cooled CI power-law population, N(gamma)/N_unaged equals
    # 1/(2 alpha a T gamma). The synchrotron moment gives this coefficient.
    return _log_moment(alpha+.5)-_log_moment(alpha)-math.log(2.*alpha)-.5*np.log(y)


_Y_LOW = 1e-7
_Y_TRIBBLE_HIGH = 1e4
_Y_CI_HIGH = 50.
_TRIBBLE_PHASE = 1.5*3.**(1./3.)


@lru_cache(maxsize=2)
def _table(model):
    alphas = np.linspace(.5, 1.5, 51)
    upper = _Y_TRIBBLE_HIGH if model == "jp_tribble" else _Y_CI_HIGH
    logs = np.linspace(math.log(_Y_LOW), math.log(upper), 601)
    y = np.exp(logs)
    values = []
    for alpha in alphas:
        if model == "jp_tribble":
            values.append(_tribble_quadrature_log_suppression(y, alpha)
                          +_TRIBBLE_PHASE*y**(2./3.))
        else:
            values.append(_ci_quadrature_log_suppression(y, alpha))
    return RegularGridInterpolator((alphas, logs), np.asarray(values), bounds_error=True)


def _model_log_suppression(y, alpha, model):
    y = np.asarray(y, dtype=float)
    result = np.empty_like(y)
    upper = _Y_TRIBBLE_HIGH if model == "jp_tribble" else _Y_CI_HIGH
    low, high = y < _Y_LOW, y > upper
    middle = ~(low | high)
    coefficient = (2.*alpha-1.)*np.exp(_log_moment(alpha-.5)-_log_moment(alpha))
    if model == "jp_tribble":
        coefficient *= np.exp(_log_field_moment(alpha+.5)-_log_field_moment(alpha+1.))
    else:
        coefficient *= .5
    result[low] = np.log1p(-coefficient*np.sqrt(y[low]))
    if np.any(high):
        result[high] = (_tribble_high_log_suppression(y[high], alpha) if model == "jp_tribble"
                        else _ci_high_log_suppression(y[high], alpha))
    if np.any(middle):
        coordinates = np.column_stack((np.full(int(middle.sum()), alpha), np.log(y[middle])))
        result[middle] = _table(model)(coordinates)
        if model == "jp_tribble":
            result[middle] -= _TRIBBLE_PHASE*y[middle]**(2./3.)
    # Both kernels average non-amplifying cooling factors. Tiny positive
    # interpolation residuals (~1e-8 near the p=2 zero-age limit) cannot
    # physically create emission above the common unaged normalization.
    return np.minimum(result, 0.)


def spectral_log_emissivity(frequencies_mhz, age_myr, magnetic_field_ug,
                            alpha_inj, redshift, *, model="jp_tribble",
                            reference_frequency_mhz=1000.):
    """Return log(S_nu(t)/S_ref(0)) at a common unaged normalization.

    Observed frequencies/reference are MHz, source-proper ages are Myr,
    and the scalar field is microgauss. Output shape is
    ``age_myr.shape + (len(frequencies_mhz),)``. Alpha, field and redshift
    are scalars; 0.5<=alpha<=1.5. Zero age is the exact injected power law.

    ``jp_tribble`` uses the declared RMS-field free-streaming convention.
    ``ci`` returns the injection-age *average*, not an unnormalized T-fold
    accumulation: its amplitude corresponds to a fixed total injected
    population. A separate fitted regional amplitude absorbs that common
    normalization, while sums across regions/cohorts retain radiative fading.
    """
    if not isinstance(model, str) or model not in MODELS:
        raise RadiativeSpectrumError("spectral model must be jp_tribble or ci")
    frequencies = _real_array(frequencies_mhz, "frequencies_mhz")
    if frequencies.ndim != 1 or np.any(frequencies <= 0):
        raise RadiativeSpectrumError("frequencies_mhz must be a nonempty positive one-dimensional array")
    ages = _real_array(age_myr, "age_myr")
    if np.any(ages < 0):
        raise RadiativeSpectrumError("age_myr must be nonnegative")
    field = _scalar(magnetic_field_ug, "magnetic_field_ug", strict=True)
    alpha = _scalar(alpha_inj, "alpha_inj")
    if not .5 <= alpha <= 1.5:
        raise RadiativeSpectrumError("alpha_inj must lie in [0.5, 1.5]")
    z = _scalar(redshift, "redshift")
    reference = _scalar(reference_frequency_mhz, "reference_frequency_mhz", strict=True)
    loss = jp_loss_coefficient(field, z)
    loss_time = loss*ages*MYR_S
    if not np.all(np.isfinite(loss_time)) or np.any(loss_time >= 1.):
        raise RadiativeSpectrumError("age/field leaves the ultrarelativistic cooling domain (gamma_c <= 1)")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = (1.+z)*MHZ_HZ/(CRITICAL_HZ_PER_T*field*1e-10)*loss_time[..., None]**2*frequencies
    if not np.all(np.isfinite(y)):
        raise RadiativeSpectrumError("frequency/cooling cutoff ratio is unrepresentable")
    result = -alpha*np.log(frequencies/reference)+_model_log_suppression(y, alpha, model)
    if not np.all(np.isfinite(result)):
        raise RadiativeSpectrumError("emissivity is numerically unrepresentable")
    return result


def spectral_log_spectrum(frequencies_mhz, age_myr, magnetic_field_ug,
                          alpha_inj, redshift, *, model="jp_tribble",
                          reference_frequency_mhz=1000.):
    """Normalize the selected model to its *aged* reference-frequency flux."""
    emission = spectral_log_emissivity(frequencies_mhz, age_myr, magnetic_field_ug,
        alpha_inj, redshift, model=model, reference_frequency_mhz=reference_frequency_mhz)
    reference = spectral_log_emissivity([reference_frequency_mhz], age_myr,
        magnetic_field_ug, alpha_inj, redshift, model=model,
        reference_frequency_mhz=reference_frequency_mhz)
    return emission-reference
