"""Radio-only, prior-conditioned inference for steady radiating advective flows.

This deliberately restricted model estimates plasma transport time, not the
birth date of an AGN. A constant shared field scale and single-injection
electron population couple the radio spectrum to the advective clock. The
JP--Tribble route uses Maxwell-distributed emitting fields and a common RMS
loss field; the explicit historical JP route uses a homogeneous field. The field-speed
degeneracy is retained in the posterior. No hydrodynamic WAT evolution, jet
power, external atmosphere, restart history or global source age is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Callable

import numpy as np
import scipy
from scipy.linalg import cholesky, null_space, solve_triangular
from scipy.special import logsumexp, ndtr, ndtri

from .constants import C_LIGHT, KPC_M, MYR_S, CONSTANT_SET
from .io import _unique_object, _reject_json_constant
from .maps import _mapping, _number, _file_spec
from .maps_io import (MapInputError, FitsPlane, read_fits_plane, science_image,
                      ensure_same_beam, validate_registration, auxiliary_data,
                      atomic_write_fits, _astropy)
from .radio_classes import normalize_radio_class
from .spectral_maps import fit_power_law_maps, _sigma_ln

KPC_KMS_TO_MYR = KPC_M / 1000.0 / MYR_S
SUPPORTED_CLASSES = {"fr1", "wat", "nat", "head_tail"}
PHYSICAL_BOUNDS = {"flow_speed_km_s": (1.0, 0.3 * C_LIGHT / 1000.),
                   "magnetic_field_ug": (0.01, 1000.),
                   "alpha_inj": (0.5, 1.5), "deceleration": (0., 10.)}


@dataclass(frozen=True)
class Prior:
    name: str
    specification: dict
    low: float
    high: float
    cdf_low: float
    cdf_high: float

    def inverse_transform(self, values):
        """Map bounded physical values back to standard-normal coordinates."""
        value = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value < self.low) or np.any(value > self.high):
            raise MapInputError(f"{self.name} is outside the prior support")
        spec = self.specification
        kind = spec["distribution"]
        if kind == "uniform":
            probability = (value - self.low) / (self.high - self.low)
        elif kind == "log_uniform":
            probability = np.log(value / self.low) / np.log(self.high / self.low)
        else:
            transformed = value if kind == "normal" else np.log(value)
            mu = spec["mean" if kind == "normal" else "log_mean"]
            sigma = spec["sigma" if kind == "normal" else "log_sigma"]
            if self.cdf_low >= .5:
                low, high = ((self.low, self.high) if kind == "normal" else (np.log(self.low), np.log(self.high)))
                sf_low, sf_high = ndtr(-(low - mu) / sigma), ndtr(-(high - mu) / sigma)
                probability = (sf_low - ndtr(-(transformed - mu) / sigma)) / (sf_low - sf_high)
            else:
                probability = (ndtr((transformed - mu) / sigma) - self.cdf_low) / (self.cdf_high - self.cdf_low)
        return ndtri(np.clip(probability, 1e-12, 1 - 1e-12))

    def log_density_up_to_constant(self, values):
        """Physical density; constants cancel in a within-prior MH ratio."""
        value = np.asarray(values, dtype=float)
        kind, spec = self.specification["distribution"], self.specification
        if kind == "uniform":
            result = np.zeros_like(value)
        elif kind == "log_uniform":
            result = -np.log(value)
        elif kind == "normal":
            result = -.5 * ((value - spec["mean"]) / spec["sigma"]) ** 2
        else:
            result = -.5 * ((np.log(value) - spec["log_mean"]) / spec["log_sigma"]) ** 2 - np.log(value)
        return np.where((value >= self.low) & (value <= self.high), result, -np.inf)

    def transform(self, z):
        p = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
        spec = self.specification
        kind = spec["distribution"]
        if kind == "uniform":
            return self.low + p * (self.high - self.low)
        if kind == "log_uniform":
            return np.exp(np.log(self.low) + p * np.log(self.high / self.low))
        if self.cdf_low >= .5:
            mu = spec["mean" if kind == "normal" else "log_mean"]
            sigma = spec["sigma" if kind == "normal" else "log_sigma"]
            low, high = ((self.low, self.high) if kind == "normal" else (np.log(self.low), np.log(self.high)))
            sf_low, sf_high = ndtr(-(low - mu) / sigma), ndtr(-(high - mu) / sigma)
            q = -ndtri(sf_low - p * (sf_low - sf_high))
        else:
            probability = self.cdf_low + p * (self.cdf_high - self.cdf_low)
            q = ndtri(np.clip(probability, np.nextafter(0., 1.), np.nextafter(1., 0.)))
        if kind == "normal":
            return np.clip(spec["mean"] + spec["sigma"] * q, self.low, self.high)
        return np.clip(np.exp(spec["log_mean"] + spec["log_sigma"] * q), self.low, self.high)


def make_prior(name: str, specification: dict) -> Prior:
    if name not in PHYSICAL_BOUNDS:
        raise MapInputError(f"unsupported prior parameter {name!r}")
    spec = _mapping(specification, f"priors.{name}",
                    {"distribution", "mean", "sigma", "log_mean", "log_sigma", "low", "high"})
    kind = spec.get("distribution")
    if kind not in {"normal", "lognormal", "uniform", "log_uniform"}:
        raise MapInputError(f"priors.{name}.distribution must be normal, lognormal, uniform or log_uniform")
    default_low, default_high = PHYSICAL_BOUNDS[name]
    low = _number(spec.get("low", default_low), f"priors.{name}.low")
    high = _number(spec.get("high", default_high), f"priors.{name}.high")
    if not default_low <= low < high <= default_high:
        raise MapInputError(f"priors.{name} bounds must lie within {PHYSICAL_BOUNDS[name]}")
    if kind == "log_uniform" and low <= 0:
        raise MapInputError("log_uniform requires a positive lower bound")
    parameters = {"mean", "sigma"} if kind == "normal" else {"log_mean", "log_sigma"} if kind == "lognormal" else set()
    if set(spec) - {"distribution", "low", "high"} != parameters:
        raise MapInputError(f"priors.{name}: {kind} requires only {sorted(parameters)} plus optional bounds")
    normalized = {"distribution": kind, "low": low, "high": high}
    normalized.update({key: _number(spec.get(key), f"priors.{name}.{key}", positive=key in {"sigma", "log_sigma"}) for key in parameters})
    cdf_low, cdf_high = 0., 1.
    if kind in {"normal", "lognormal"}:
        mu = normalized["mean" if kind == "normal" else "log_mean"]
        sigma = normalized["sigma" if kind == "normal" else "log_sigma"]
        if kind == "lognormal" and low <= 0:
            raise MapInputError("lognormal requires positive bounds")
        transform = (lambda value: value) if kind == "normal" else np.log
        cdf_low = float(ndtr((transform(low) - mu) / sigma))
        cdf_high = float(ndtr((transform(high) - mu) / sigma))
        if cdf_high - cdf_low < 1e-9:
            raise MapInputError(f"priors.{name}: negligible mass within physical bounds")
    return Prior(name, normalized, low, high, cdf_low, cdf_high)


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise MapInputError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise MapInputError("nonfinite posterior statistic")
    q = np.quantile(values, [.025, .16, .5, .84, .975])
    return {"mean": float(np.mean(values)), "median": float(q[2]),
            "credible_interval_68": [float(q[1]), float(q[3])],
            "credible_interval_95": [float(q[0]), float(q[4])]}


def tempered_smc(log_likelihood: Callable, dimension: int, *, particles=512,
                 seed=42, mcmc_steps=4, ess_target=.7, max_stages=80,
                 rejuvenation=None):
    """Tempered SMC in independent standard-normal prior coordinates.

    Rejuvenation uses symmetric random walks and occasional independence
    proposals from the prior. Reported ESS is BEFORE resampling, never an
    assertion that the final equal-weight particles are independent.
    """
    dimension = _integer(dimension, "dimension", 1, 32)
    particles = _integer(particles, "particles", 64, 4096)
    mcmc_steps = _integer(mcmc_steps, "mcmc_steps", 1, 40)
    max_stages = _integer(max_stages, "max_stages", 2, 300)
    if not isinstance(ess_target, (int, float)) or isinstance(ess_target, bool) or not .3 <= ess_target <= .95:
        raise MapInputError("ess_target must lie in [.3,.95]")
    rng = np.random.default_rng(seed)
    def evaluate(rows):
        values = np.array([log_likelihood(row) for row in rows], dtype=float)
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise MapInputError("likelihood returned NaN or positive infinity")
        return values
    z = rng.normal(size=(particles, dimension))
    likelihood = evaluate(z)
    rejected_domain, attempted_domain = 0, particles
    # Rejection samples the prior conditional on the physical kernel domain.
    # Its rejected prior mass is recorded, rather than silently clamping ages.
    for _ in range(30):
        invalid = ~np.isfinite(likelihood)
        if not np.any(invalid):
            break
        rejected_domain += int(invalid.sum())
        attempted_domain += int(invalid.sum())
        z[invalid] = rng.normal(size=(int(invalid.sum()), dimension))
        likelihood[invalid] = evaluate(z[invalid])
    if not np.all(np.isfinite(likelihood)):
        raise MapInputError("too little prior mass within the physical spectral-model domain")
    beta, stages, evidence = 0., [], 0.
    ancestors = np.arange(particles)
    for stage in range(max_stages):
        def weights_at(next_beta):
            logw = (next_beta - beta) * likelihood
            normalizer = logsumexp(logw)
            weights = np.exp(logw - normalizer)
            return weights, 1. / float(weights @ weights), float(normalizer - np.log(particles))
        weights, ess, log_increment = weights_at(1.)
        next_beta = 1.
        if ess < ess_target * particles:
            low, high = beta, 1.
            for _ in range(35):
                middle = .5 * (low + high)
                if weights_at(middle)[1] < ess_target * particles:
                    high = middle
                else:
                    low = middle
            next_beta = low
            weights, ess, log_increment = weights_at(next_beta)
        if next_beta <= beta + 1e-12:
            raise MapInputError("SMC tempering stalled; narrow or reassess priors/model")
        evidence += log_increment
        center = np.sum(weights[:, None] * z, axis=0)
        delta = z - center
        covariance = (delta * weights[:, None]).T @ delta
        covariance += np.eye(dimension) * 1e-4
        proposal_root = cholesky(covariance, lower=True) * (2.38 / np.sqrt(dimension))
        positions = (rng.random() + np.arange(particles)) / particles
        indices = np.searchsorted(np.cumsum(weights), positions, side="right")
        indices = np.minimum(indices, particles - 1)
        z, likelihood, ancestors = z[indices].copy(), likelihood[indices].copy(), ancestors[indices]
        accepted = 0
        for _ in range(mcmc_steps):
            independent = rng.random(particles) < .15
            proposed = z + rng.normal(size=z.shape) @ proposal_root.T
            proposed[independent] = rng.normal(size=(int(independent.sum()), dimension))
            proposed_likelihood = evaluate(proposed)
            ratio = next_beta * (proposed_likelihood - likelihood)
            ratio[~independent] -= .5 * (np.sum(proposed[~independent] ** 2, axis=1) - np.sum(z[~independent] ** 2, axis=1))
            take = np.log(rng.random(particles)) < ratio
            z[take], likelihood[take] = proposed[take], proposed_likelihood[take]
            accepted += int(take.sum())
        stages.append({"beta": float(next_beta), "pre_resampling_ess": ess,
                       "acceptance": accepted / (particles * mcmc_steps),
                       "surviving_initial_ancestors": int(len(np.unique(ancestors)))})
        if rejuvenation is not None:
            z, likelihood, acceptance = rejuvenation(z, likelihood, next_beta, rng)
            stages[-1]["physical_ridge_acceptance"] = float(acceptance)
            shape_acceptance = getattr(rejuvenation, "last_shape_acceptance", None)
            if shape_acceptance is not None:
                stages[-1]["flow_shape_acceptance"] = float(shape_acceptance)
        beta = next_beta
        if beta == 1.:
            break
    if beta != 1.:
        raise MapInputError("SMC did not reach the posterior; increase max_stages or reassess the model")
    return z, likelihood, {"stages": stages, "log_evidence_up_to_constant": evidence,
                           "evidence_scope": "conditional on physical spectral-kernel support; rejected prior mass normalization is omitted, not a cross-model evidence",
                           "particles": particles, "seed": seed,
                           "physical_domain_prior_rejection_fraction": rejected_domain / attempted_domain,
                           "ess_interpretation": "conditional importance ESS before resampling; final particles are correlated"}


def make_field_speed_move(priors, branch_count, redshift, log_likelihood, *, shape_steps=4):
    """Construct ridge refreshes and invariant-shape Metropolis moves.

    Hold ``u_i = v_i / h(B)`` fixed, where ``h(B)=a(B)/sqrt(B)`` and
    ``a`` is the constant-field JP loss coefficient. Draw the new field from
    its own prior and scale every branch velocity by ``h(new)/h(old)``.
    In ``(u_i, B)`` coordinates the physical-volume Jacobian is ``h(B)**n``;
    the field prior cancels its independence proposal, leaving the velocity
    prior density ratio plus ``n*log(scale)`` and the tempered likelihood
    ratio. In particular, log-uniform velocity priors give acceptance one
    within their bounds. The callback already includes physical prior and
    proposal factors; no extra latent-normal density ratio may be added.

    ``priors`` follows inference ordering: branch speeds, shared field,
    then injection index and any further unchanged parameters. Returned
    After refreshing the field, propose the invariant shape coordinates
    ``(log(v_i/h(B)), z_alpha, z_eta, ...)`` at fixed field. A frozen empirical
    covariance supplies symmetric Gaussian proposals for ``shape_steps``
    sweeps. The target in these coordinates includes ``p_v(v_i)*v_i`` for
    every velocity and standard-normal densities for the remaining latent
    coordinates. This removes the curved field/speed ridge from the proposal
    covariance without discarding injection-index/flow correlations.

    The callback has signature ``(latent, likelihood, beta, rng)`` and updates
    those arrays in place, returning them with the *field-refresh* acceptance
    fraction. Its ``last_shape_acceptance`` attribute reports the separate
    shape-block acceptance. Set ``shape_steps=0`` to isolate the exact ridge
    refresh, for example in analytic invariance checks.
    """
    shape_steps = _integer(shape_steps, "shape_steps", 0, 40)

    def field_speed_move(latent, likelihood, beta, rng):
        # Hold u_i=v_i/[a(B)/sqrt(B)] fixed and independently propose B from
        # its prior. The physical-coordinate Jacobian is scale**nbranch.
        # This move explores the exact field/speed ridge that local Gaussian
        # random walks can traverse poorly. Include beta*dlogL to correct
        # finite numerical differences and preserve the tempered target.
        from .radio_spectra import jp_loss_coefficient
        count, nbranch = len(latent), branch_count
        current = np.column_stack([prior.transform(latent[:, index]) for index, prior in enumerate(priors)])
        proposed = current.copy()
        field_prior = priors[nbranch]
        new_field = field_prior.transform(rng.normal(size=count))
        old_field = current[:, nbranch]
        new_loss = np.array([jp_loss_coefficient(field, redshift) for field in new_field])
        old_loss = np.array([jp_loss_coefficient(field, redshift) for field in old_field])
        scale = (new_loss / old_loss) * np.sqrt(old_field / new_field)
        proposed[:, :nbranch] *= scale[:, None]
        proposed[:, nbranch] = new_field
        allowed = np.ones(count, dtype=bool)
        ratio = nbranch * np.log(scale)
        for index in range(nbranch):
            prior = priors[index]
            allowed &= (proposed[:, index] >= prior.low) & (proposed[:, index] <= prior.high)
            ratio += prior.log_density_up_to_constant(proposed[:, index]) - prior.log_density_up_to_constant(current[:, index])
        proposed_latent = latent.copy()
        for index in range(nbranch + 1):
            proposed_latent[allowed, index] = priors[index].inverse_transform(proposed[allowed, index])
        proposed_likelihood = np.full(count, -np.inf)
        proposed_likelihood[allowed] = np.array([log_likelihood(row) for row in proposed_latent[allowed]])
        ratio += beta * (proposed_likelihood - likelihood)
        take = allowed & (np.log(rng.random(count)) < ratio)
        latent[take], likelihood[take] = proposed_latent[take], proposed_likelihood[take]
        field_acceptance = float(np.mean(take))
        field_speed_move.last_shape_acceptance = None
        if not shape_steps:
            return latent, likelihood, field_acceptance

        # Invariant coordinates isolate spectral-shape uncertainty from the
        # exact B-v degeneracy. The field is fixed throughout these sweeps.
        current = np.column_stack([prior.transform(latent[:, index])
                                   for index, prior in enumerate(priors)])
        field = current[:, nbranch]
        log_h = np.log(np.array([jp_loss_coefficient(value, redshift) for value in field])) - .5 * np.log(field)
        other = np.arange(nbranch + 1, len(priors))
        coordinates = np.column_stack((np.log(current[:, :nbranch]) - log_h[:, None],
                                       latent[:, other]))
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered / count
        covariance += np.eye(coordinates.shape[1]) * 1e-6
        shape_root = cholesky(covariance, lower=True) * (2.38 / np.sqrt(coordinates.shape[1]))

        def shape_prior_density(physical, state):
            density = -.5 * np.sum(state[:, other] ** 2, axis=1)
            for index in range(nbranch):
                values = physical[:, index]
                density += priors[index].log_density_up_to_constant(values) + np.log(values)
            return density

        current_density = shape_prior_density(current, latent)
        shape_accepted = 0
        for _ in range(shape_steps):
            candidate = coordinates + rng.normal(size=coordinates.shape) @ shape_root.T
            log_speed = candidate[:, :nbranch] + log_h[:, None]
            allowed = np.ones(count, dtype=bool)
            for index in range(nbranch):
                allowed &= (log_speed[:, index] >= np.log(priors[index].low)) & (log_speed[:, index] <= np.log(priors[index].high))
            candidate_latent = latent.copy()
            candidate_physical = current.copy()
            for index in range(nbranch):
                candidate_physical[allowed, index] = np.exp(log_speed[allowed, index])
                candidate_latent[allowed, index] = priors[index].inverse_transform(candidate_physical[allowed, index])
            candidate_latent[:, other] = candidate[:, nbranch:]
            candidate_likelihood = np.full(count, -np.inf)
            candidate_likelihood[allowed] = np.array([log_likelihood(row) for row in candidate_latent[allowed]])
            candidate_density = shape_prior_density(candidate_physical, candidate_latent)
            ratio = beta * (candidate_likelihood - likelihood) + candidate_density - current_density
            take = allowed & (np.log(rng.random(count)) < ratio)
            latent[take], likelihood[take] = candidate_latent[take], candidate_likelihood[take]
            current[take], current_density[take] = candidate_physical[take], candidate_density[take]
            coordinates[take] = candidate[take]
            shape_accepted += int(take.sum())
        field_speed_move.last_shape_acceptance = shape_accepted / (count * shape_steps)
        return latent, likelihood, field_acceptance

    field_speed_move.last_shape_acceptance = None
    return field_speed_move


@dataclass
class RadioInferenceResult:
    reference: FitsPlane
    maps: dict[str, np.ndarray]
    metadata: dict[str, Any]
    samples: dict[str, np.ndarray]
    input_paths: frozenset[Path]
    resolved_results: dict | None = None


def compute_radio_inference(config_path: str | Path) -> RadioInferenceResult:
    """Infer a beam-informed parametric transport field from radio images."""
    from . import __version__
    from .radio_spectra import jp_log_emissivity, RadiativeSpectrumError

    path = Path(config_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_json_constant)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise MapInputError(f"cannot load radio inference configuration: {exc}") from exc
    config = _mapping(config, "root", {"schema_version", "source", "images", "selection", "geometry", "model", "priors", "prior_description", "reference_catalogue", "pyrats_properties", "fixed_magnetic_field", "inference", "calibration_correlation", "core", "resolved_analysis"})
    if config.get("schema_version") != "fr2dynage-radio-1":
        raise MapInputError("schema_version must be 'fr2dynage-radio-1'")
    fixed_field = None
    if 'fixed_magnetic_field' in config:
        fixed_field = _mapping(config['fixed_magnetic_field'], 'fixed_magnetic_field', {'value_t', 'description'})
        field_t = _number(fixed_field.get('value_t'), 'fixed_magnetic_field.value_t', positive=True)
        field_ug = field_t*1e10
        if not PHYSICAL_BOUNDS['magnetic_field_ug'][0] <= field_ug <= PHYSICAL_BOUNDS['magnetic_field_ug'][1]:
            raise MapInputError('fixed_magnetic_field.value_t must correspond to 0.01–1000 microgauss')
        if not isinstance(fixed_field.get('description'), str) or not fixed_field['description'].strip():
            raise MapInputError('fixed_magnetic_field.description must explain the user-supplied fixed input')
        fixed_field = {**fixed_field, 'value_t': field_t, 'value_ug': field_ug}
    if "resolved_analysis" in config:
        from .resolved_products import parse_resolved_settings
        parse_resolved_settings(config["resolved_analysis"])
    source = _mapping(config.get("source"), "source", {"name", "redshift", "radio_class", "activity_state"})
    if not isinstance(source.get("name"), str) or not source["name"].strip():
        raise MapInputError("source.name is required")
    radio_class = normalize_radio_class(source.get("radio_class"))
    if radio_class not in SUPPORTED_CLASSES:
        raise MapInputError("infer-radio steady advection supports fr1, wat, nat and head_tail; FR II dynamics use fit, other classes require their own validated model")
    if source.get("activity_state") != "active":
        raise MapInputError("steady JP advection requires activity_state='active'; restarted/remnant histories are not modelled")
    redshift = _number(source.get("redshift"), "source.redshift", positive=True)
    if not isinstance(config.get("prior_description"), str) or not config["prior_description"].strip():
        raise MapInputError("prior_description must explain the origin of priors and whether they are illustrative")
    model = _mapping(config.get("model"), "model", {"kind", "deceleration", "log_emissivity_scatter", "beam_path_nodes"})
    if model.get("kind") not in {"steady_jp_advection", "steady_jp_tribble_advection"}:
        raise MapInputError("model.kind must be 'steady_jp_tribble_advection' (or explicit historical 'steady_jp_advection'); CI injection-duration fits belong to resolved_analysis")
    use_tribble = model["kind"] == "steady_jp_tribble_advection"
    if use_tribble:
        from .advanced_spectra import spectral_log_emissivity, model_metadata
        def emissivity(frequencies, age, field, alpha, redshift, **kwargs):
            return spectral_log_emissivity(frequencies, age, field, alpha, redshift, model="jp_tribble", **kwargs)
        spectral_model = model_metadata("jp_tribble")
    else:
        emissivity = jp_log_emissivity
        spectral_model = {"model": "jp", "label": "JP (historical homogeneous-field reference)",
            "clock": "single-aged electron population; transport time only under the declared advection assumptions"}
    deceleration = _number(model.get("deceleration", 0.), "model.deceleration", nonnegative=True)
    if deceleration > 10.:
        raise MapInputError("model.deceleration must be <=10")
    scatter = _number(model.get("log_emissivity_scatter", .05), "model.log_emissivity_scatter", nonnegative=True)
    nodes = _integer(model.get("beam_path_nodes", 12), "model.beam_path_nodes", 4, 32)
    geometry = _mapping(config.get("geometry"), "geometry", {"distance_map", "branch_map", "description", "kpc_per_arcsec"})
    if not isinstance(geometry.get("description"), str) or not geometry["description"].strip():
        raise MapInputError("geometry.description must explain the paths and zero-time origin")
    path_scale = (_number(geometry['kpc_per_arcsec'], 'geometry.kpc_per_arcsec', positive=True)
                  if 'kpc_per_arcsec' in geometry else None)
    selection = _mapping(config.get("selection", {}), "selection", {"snr_min", "min_bands", "mask"})
    settings = _mapping(config.get("inference", {}), "inference", {"particles", "seed", "mcmc_steps", "ess_target", "max_stages", "replicates"})
    particles = _integer(settings.get("particles", 512), "inference.particles", 64, 4096)
    seed = _integer(settings.get("seed", 42), "inference.seed", 0, 2**32 - 1)
    steps = _integer(settings.get("mcmc_steps", 4), "inference.mcmc_steps", 1, 40)
    generic_steps = min(steps+1, 40) if fixed_field is not None else 1
    max_stages = _integer(settings.get("max_stages", 80), "inference.max_stages", 2, 300)
    replicates = _integer(settings.get("replicates", 2), "inference.replicates", 1, 4)
    ess_target = _number(settings.get("ess_target", .7), "inference.ess_target", positive=True)
    if not .3 <= ess_target <= .95:
        raise MapInputError("inference.ess_target must lie in [.3,.95]")
    inputs, records = {path}, []
    def load(spec, label):
        filename, hdu = _file_spec(spec, path.parent, label)
        plane = read_fits_plane(filename, hdu)
        inputs.add(plane.path)
        records.append({"role": label, "path": str(plane.path), "hdu": hdu, "sha256": plane.sha256})
        return plane
    images = config.get("images")
    if not isinstance(images, list) or not 4 <= len(images) <= 64:
        raise MapInputError("infer-radio requires 4–64 distinct frequency images")
    planes, fluxes, noises, frequencies, calibrations, identities = [], [], [], [], [], set()
    for index, entry in enumerate(images):
        label = f"images[{index}]"
        entry = _mapping(entry, label, {"path", "hdu", "frequency_mhz", "rms_jy_per_beam", "rms_map", "calibration_fraction"})
        plane = load({key: entry[key] for key in ("path", "hdu") if key in entry}, label)
        identity = (plane.path, plane.hdu_index)
        if identity in identities:
            raise MapInputError("one image/HDU cannot represent independent frequency measurements")
        identities.add(identity)
        flux, beam = science_image(plane)
        if planes:
            validate_registration(planes[0], plane)
            ensure_same_beam(planes[0], plane)
        if ("rms_jy_per_beam" in entry) == ("rms_map" in entry):
            raise MapInputError(f"{label} requires exactly one of rms_jy_per_beam or rms_map")
        noise = (auxiliary_data(load(entry["rms_map"], label + ".rms_map"), plane, "Jy / beam")
                 if "rms_map" in entry else np.full(flux.shape, _number(entry["rms_jy_per_beam"], label + ".rms_jy_per_beam", positive=True)))
        planes.append(plane); fluxes.append(flux); noises.append(noise)
        frequencies.append(_number(entry.get("frequency_mhz"), label + ".frequency_mhz", positive=True))
        calibrations.append(_number(entry.get("calibration_fraction", 0.), label + ".calibration_fraction", nonnegative=True))
    if len(set(frequencies)) != len(frequencies):
        raise MapInputError("image frequencies must be distinct")
    reference, flux, rms, frequencies = planes[0], np.stack(fluxes), np.stack(noises), np.array(frequencies)
    distance_plane = load(geometry.get("distance_map"), "geometry.distance_map")
    distance = auxiliary_data(distance_plane, reference, "kpc")
    if 'KPCAS' in distance_plane.header:
        header_scale = _number(distance_plane.header['KPCAS'], 'distance-map KPCAS', positive=True)
        if path_scale is not None and not math.isclose(path_scale, header_scale, rel_tol=1e-6):
            raise MapInputError('geometry.kpc_per_arcsec disagrees with distance-map KPCAS')
        path_scale = header_scale
    branch_data = auxiliary_data(load(geometry.get("branch_map"), "geometry.branch_map"), reference, "1")
    if np.any(~np.isfinite(branch_data)) or np.any(branch_data < 0) or np.any(branch_data != np.floor(branch_data)) or np.max(branch_data) > 32767:
        raise MapInputError("branch_map must contain finite integer labels 0..32767 (0=excluded)")
    branches = branch_data.astype(np.int16)
    source_mask = np.ones(distance.shape, dtype=bool)
    if "mask" in selection:
        supplied = auxiliary_data(load(selection["mask"], "selection.mask"), reference, None)
        if np.any(~np.isfinite(supplied)) or not np.all((supplied == 0) | (supplied == 1)):
            raise MapInputError("selection.mask must be finite binary 0/1")
        source_mask &= supplied.astype(bool)
    core_mask = np.zeros(distance.shape, dtype=bool)
    core_spec = None
    core_policy = "separate"
    if "core" in config:
        core_spec = _mapping(config["core"], "core", {"mask", "position_icrs_deg", "description", "policy"})
        core_policy = core_spec.get("policy", "separate")
        if not isinstance(core_policy, str) or core_policy not in {"include", "separate"}:
            raise MapInputError("core.policy must be include or separate")
        if not isinstance(core_spec.get("description"), str) or not core_spec["description"].strip():
            raise MapInputError("core.description must explain the central footprint and unresolved age treatment")
        supplied_core = auxiliary_data(load(core_spec.get("mask"), "core.mask"), reference, None)
        if np.any(~np.isfinite(supplied_core)) or not np.all((supplied_core == 0) | (supplied_core == 1)) or not np.any(supplied_core):
            raise MapInputError("core.mask must be a nonempty finite binary 0/1 mask")
        core_mask = supplied_core.astype(bool)
        source_mask |= core_mask
    geometry_valid = (branches > 0) & np.isfinite(distance) & (distance >= 0)
    mask = source_mask & geometry_valid
    if core_policy == "separate":
        mask &= ~core_mask
    snr_min = _number(selection.get("snr_min", 5.), "selection.snr_min", positive=True)
    min_bands = _integer(selection.get("min_bands", 4), "selection.min_bands", 4, len(frequencies))
    spectrum = fit_power_law_maps(frequencies, flux, rms, calibration_fractions=calibrations,
                                 snr_min=snr_min, min_bands=min_bands, require_all_bands=False, mask=source_mask)
    source_valid = source_mask & spectrum.valid
    valid = mask & spectrum.valid
    if not np.any(valid):
        raise MapInputError("no pixels pass the spectral and geometry selection")
    labels = np.unique(branches[valid])
    if len(labels) > 8:
        raise MapInputError("at most eight independently modelled branches are supported")
    prior_specs = {}
    catalogue_provenance = None
    if "reference_catalogue" in config:
        from .reference_priors import load_reference_priors
        catalogue = _mapping(config["reference_catalogue"], "reference_catalogue", {"path", "allow_synthetic", "allow_dependent"})
        for boolean in ("allow_synthetic", "allow_dependent"):
            if boolean in catalogue and not isinstance(catalogue[boolean], bool):
                raise MapInputError(f"reference_catalogue.{boolean} must be boolean")
        catalogue_path, _ = _file_spec(catalogue.get("path"), path.parent, "reference_catalogue")
        calibrated = load_reference_priors(catalogue_path, source["name"], radio_class,
                                         allow_synthetic=catalogue.get("allow_synthetic", False),
                                         allow_dependent=catalogue.get("allow_dependent", False))
        prior_specs.update(calibrated["priors"])
        catalogue_provenance = calibrated["provenance"]
        inputs.add(catalogue_path)
        records.append({'role': 'reference_catalogue', 'path': str(catalogue_path), 'sha256': catalogue_provenance['sha256']})
    explicit = _mapping(config.get("priors", {}), "priors", set(PHYSICAL_BOUNDS))
    if fixed_field is not None and 'magnetic_field_ug' in explicit:
        raise MapInputError('fixed_magnetic_field conflicts with an explicit magnetic_field_ug prior')
    pyrats = None
    pyrats_overrides_reference_field = False
    if "pyrats_properties" in config:
        from .pyrats_bridge import load_pyrats_properties, ADOPTED_B_EQ_FRACTION
        spec = _mapping(config["pyrats_properties"], "pyrats_properties", {"path", "field_basis", "field_scale", "systematic_log_sigma"})
        if "magnetic_field_ug" in explicit:
            raise MapInputError("use pyrats_properties or an explicit magnetic_field_ug prior, not both; the PyRATS source prescription is B = 0.4 B_eq")
        if fixed_field is not None:
            if 'field_scale' in spec or 'field_basis' in spec:
                raise MapInputError('fixed_magnetic_field supplies the adopted field; omit PyRATS field_scale/field_basis knobs when its report is used only as provenance')
            requested_scale = 1.
        else:
            requested_scale = _number(spec.get("field_scale", ADOPTED_B_EQ_FRACTION), "pyrats_properties.field_scale", positive=True)
            if spec.get("field_basis", "equipartition") != "equipartition" or not math.isclose(requested_scale, ADOPTED_B_EQ_FRACTION, rel_tol=0., abs_tol=1e-12):
                raise MapInputError("the default source field prescription is fixed at B = 0.4 B_eq; use field_basis='equipartition' and field_scale=0.4, or an explicit fixed_magnetic_field override")
        property_path, _ = _file_spec(spec.get("path"), path.parent, "pyrats_properties")
        pyrats = load_pyrats_properties(property_path, expected_redshift=redshift,
                    systematic_log_sigma=spec.get("systematic_log_sigma"),
                    field_basis="equipartition", field_scale=requested_scale,
                    image_paths=[{"path": str(plane.path), "frequency_mhz": float(frequency)} for plane, frequency in zip(planes, frequencies)])
        pyrats['field_constraint_applied'] = fixed_field is None
        pyrats_overrides_reference_field = fixed_field is None and "magnetic_field_ug" in prior_specs
        if fixed_field is None:
            prior_specs.update(pyrats["priors"])
        inputs.add(property_path)
        records.append({"role": "pyrats_properties", "path": str(property_path), "sha256": pyrats["provenance"]["sha256"]})
    field_scale = pyrats['properties']['geometry'].get('kpc_per_arcsec') if pyrats is not None else None
    scale_consistency = {'status': 'not_cross_checked', 'path_kpc_per_arcsec': path_scale,
                         'pyrats_kpc_per_arcsec': field_scale}
    if path_scale is not None and field_scale is not None:
        scale_consistency.update(fractional_difference=path_scale/field_scale-1.,
            status='consistent' if math.isclose(path_scale, field_scale, rel_tol=.01) else 'inconsistent')
    overridden = sorted(set(prior_specs) & set(explicit))
    prior_specs.update(explicit)
    fixed_overrides_reference_field = fixed_field is not None and 'magnetic_field_ug' in prior_specs
    if fixed_field is not None:
        prior_specs.pop('magnetic_field_ug', None)
    for required in (['flow_speed_km_s', 'alpha_inj'] if fixed_field is not None else ['flow_speed_km_s', 'magnetic_field_ug', 'alpha_inj']):
        if required not in prior_specs:
            raise MapInputError(f"missing prior for {required}; supply explicitly or calibrate with reference_catalogue")
    base_priors = {name: make_prior(name, spec) for name, spec in prior_specs.items()}
    names = [f"flow_speed_km_s_branch_{label}" for label in labels] + ["magnetic_field_ug", "alpha_inj"]
    priors = [base_priors['flow_speed_km_s'] for _ in labels] + ([] if fixed_field is not None else [base_priors['magnetic_field_ug']]) + [base_priors['alpha_inj']]
    if "deceleration" in base_priors:
        if "deceleration" in model:
            raise MapInputError("provide either model.deceleration or priors.deceleration, not both")
        names.append("deceleration"); priors.append(base_priors["deceleration"])
    sampled_names = [name for name in names if not (fixed_field is not None and name == 'magnetic_field_ug')]

    # One representative observation per beam-major square and branch.
    # Oversampling the FITS grid therefore does not multiply likelihood terms.
    from astropy.wcs.utils import proj_plane_pixel_area
    pixel_area = abs(float(proj_plane_pixel_area(reference.wcs)))
    step = max(1, int(math.ceil(beam["bmaj_deg"] / math.sqrt(pixel_area))))
    yy, xx = np.nonzero(valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_cube = flux / rms
    detected = np.isfinite(flux) & np.isfinite(rms) & (flux > 0) & (rms > 0) & (snr_cube >= snr_min)
    score = np.where(detected, snr_cube, -np.inf)
    score = np.sort(score, axis=0)[-min_bands]
    groups = {}
    for y, x in zip(yy, xx):
        key = (int(branches[y, x]), int(y // step), int(x // step))
        if key not in groups or score[y, x] > score[groups[key]]:
            groups[key] = (int(y), int(x))
    positions = np.array(list(groups.values()), dtype=int)
    if not 3 <= len(positions) <= 256:
        raise MapInputError("require 3–256 beam-scale spectral sample regions; adjust image/mask extent")
    ry, rx = positions.T
    region_branch = np.searchsorted(labels, branches[ry, rx])
    if any(np.sum(region_branch == index) < 2 for index in range(len(labels))):
        raise MapInputError("each branch requires at least two beam-scale spectral samples")
    region_bands = detected[:, ry, rx].T
    rid, fid = np.nonzero(region_bands)
    observed = flux[fid, ry[rid], rx[rid]]
    thermal = np.sqrt(np.log1p((rms[fid, ry[rid], rx[rid]] / observed) ** 2))
    # Physical east/north offsets avoid RA wrapping and cos(dec) errors.
    sky = reference.wcs.pixel_to_world(rx, ry)
    origin = reference.wcs.pixel_to_world((distance.shape[1] - 1) / 2, (distance.shape[0] - 1) / 2)
    east, north = origin.spherical_offsets_to(sky)
    offsets = np.column_stack((east.deg, north.deg))
    delta = offsets[:, None, :] - offsets[None, :, :]
    inverse_beam = np.linalg.inv(beam["covariance_deg2"])
    correlation = np.exp(-np.einsum("...i,ij,...j->...", delta, inverse_beam, delta) / 4.)
    covariance = (correlation[rid[:, None], rid[None, :]] * (fid[:, None] == fid[None, :])
                  * thermal[:, None] * thermal[None, :])
    cal_correlation = np.asarray(config.get("calibration_correlation", np.eye(len(frequencies))), dtype=float)
    if (cal_correlation.shape != (len(frequencies), len(frequencies)) or
        not np.all(np.isfinite(cal_correlation)) or
        not np.allclose(cal_correlation, cal_correlation.T, atol=1e-10) or
        not np.allclose(np.diag(cal_correlation), 1.) or
        np.linalg.eigvalsh(cal_correlation).min() < -1e-9):
        raise MapInputError("calibration_correlation must be a finite symmetric positive-semidefinite correlation matrix")
    core_diagnostics = None
    if core_spec is not None:
        from .core_diagnostics import analyze_core_region
        core_diagnostics = analyze_core_region(frequencies, flux, rms, calibrations, cal_correlation,
            core_mask, source_valid, reference, core_spec.get("position_icrs_deg"),
            snr_min=snr_min, min_bands=min_bands)
        core_diagnostics["description"] = core_spec["description"]
        core_diagnostics["policy"] = core_policy
        core_diagnostics["modelled_pixels"] = int(np.sum(core_mask & valid))
        if core_policy == "include":
            core_diagnostics["status"] = "included_in_conditional_transport_model"
            core_diagnostics["age_status"] = "per_pixel_model_conditioned_transport; no single deblended nuclear age"
            core_diagnostics["warnings"] = [
                "Core-region spectra enter the same declared radiative/transport likelihood as other modelled source pixels.",
                "The common beam can mix nucleus, inner jets and projected plume emission; the per-pixel clock is conditional on the supplied paths and radiative model.",
                "Near the origin, the provisional branch/path projection is ambiguous; zero-distance clock values are geometric reference values, not measured zero-age nuclear plasma.",
                "The descriptive central-beam power-law diagnostic does not establish an independent break, age or absorption model."]
    cal_sigma = _sigma_ln(np.array(calibrations))
    covariance += cal_correlation[fid[:, None], fid[None, :]] * cal_sigma[fid[:, None]] * cal_sigma[fid[None, :]]
    covariance += np.eye(len(rid)) * (scatter ** 2 + 1e-12)
    condition = float(np.linalg.cond(covariance))
    if condition > 1e12:
        raise MapInputError("spectral covariance is ill-conditioned; reassess noise/calibration settings")
    root = cholesky(covariance, lower=True)
    inverse_root = solve_triangular(root, np.eye(len(rid)), lower=True)
    amplitudes = np.eye(len(positions))[rid]
    design = inverse_root @ amplitudes
    complement = null_space(design.T)
    projector = complement.T @ inverse_root
    projected_data = projector @ np.log(observed)
    dof = len(rid) - len(positions) - len(sampled_names)
    if dof < 1:
        raise MapInputError("too few independent spectral measurements for the model")
    lengths = np.array([np.max(distance[mask & (branches == label)]) for label in labels])
    if np.any(lengths <= 0):
        raise MapInputError("each branch must have a positive traced length; zero-distance origin pixels alone do not define a flow")
    geometry_y, geometry_x = np.nonzero(mask)
    all_sky = reference.wcs.pixel_to_world(geometry_x, geometry_y)
    ae, an = origin.spherical_offsets_to(all_sky)
    all_offsets = np.column_stack((ae.deg, an.deg))
    path_nodes = np.empty((len(positions), nodes))
    path_weights = np.full(path_nodes.shape, 1. / nodes)
    for index in range(len(positions)):
        same_branch = branches[geometry_y, geometry_x] == labels[region_branch[index]]
        separation = all_offsets - offsets[index]
        radius2 = np.einsum("...i,ij,...j->...", separation, inverse_beam, separation)
        include = same_branch & (radius2 <= 25.)
        distances = distance[geometry_y[include], geometry_x[include]]
        weights = np.exp(-radius2[include] / 2.)
        order = np.argsort(distances)
        cumulative = np.cumsum(weights[order]); cumulative /= cumulative[-1]
        path_nodes[index] = np.interp((np.arange(nodes) + .5) / nodes, cumulative, distances[order])

    def unpack(z):
        parameters = np.array([prior.transform(value) for prior, value in zip(priors, z)])
        return np.insert(parameters, len(labels), fixed_field['value_ug']) if fixed_field is not None else parameters

    def prediction(parameters):
        speed = parameters[:len(labels)]
        field, alpha = parameters[len(labels):len(labels) + 2]
        eta = parameters[-1] if "deceleration" in names else deceleration
        age = KPC_KMS_TO_MYR * (path_nodes + eta * path_nodes ** 2 / (2 * lengths[region_branch, None])) / speed[region_branch, None]
        log_spectra = emissivity(frequencies, age, field, alpha, redshift, reference_frequency_mhz=1000.)
        return logsumexp(log_spectra + np.log(path_weights[..., None]), axis=1)

    def log_likelihood(z):
        try:
            predicted = prediction(unpack(z))[rid, fid]
        except RadiativeSpectrumError:
            return -np.inf
        residual = projected_data - projector @ predicted
        return -.5 * float(residual @ residual)

    field_speed_move = (None if fixed_field is not None else
                        make_field_speed_move(priors, len(labels), redshift, log_likelihood, shape_steps=steps))

    posterior_z, posterior_ll, sampler_diagnostics = [], [], []
    for replicate in range(replicates):
        z, logl, diagnostics = tempered_smc(log_likelihood, len(sampled_names), particles=particles,
             seed=seed + replicate, mcmc_steps=generic_steps, ess_target=ess_target, max_stages=max_stages,
             rejuvenation=field_speed_move)
        posterior_z.append(z); posterior_ll.append(logl); sampler_diagnostics.append(diagnostics)
    z = np.concatenate(posterior_z)
    logl = np.concatenate(posterior_ll)
    physical = np.column_stack([prior.transform(z[:, index]) for index, prior in enumerate(priors)])
    if fixed_field is not None:
        physical = np.insert(physical, len(labels), fixed_field['value_ug'], axis=1)
    samples = {name: physical[:, index] for index, name in enumerate(names)}
    if pyrats is not None and fixed_field is None:
        samples["equipartition_field_ug"] = samples["magnetic_field_ug"] / ADOPTED_B_EQ_FRACTION
    samples["log_likelihood"] = logl
    samples["replicate"] = np.repeat(np.arange(replicates), particles)
    prior_rng = np.random.default_rng(seed + 100000)
    prior_draws = np.column_stack([prior.transform(prior_rng.normal(size=len(z))) for prior in priors])
    if fixed_field is not None:
        prior_draws = np.insert(prior_draws, len(labels), fixed_field['value_ug'], axis=1)
    replica_difference = []
    if replicates > 1:
        for index in range(len(names)):
            if fixed_field is not None and names[index] == 'magnetic_field_ug':
                continue
            medians = [np.median(physical[j * particles:(j + 1) * particles, index]) for j in range(replicates)]
            replica_difference.append(float(np.ptp(medians) / max(np.std(physical[:, index]), 1e-12)))
    sampler_status = ("single_run_unverified" if replicates == 1 else
                      "replicate_disagreement" if max(replica_difference) > .5 else "replicate_consistent")
    best = int(np.argmax(logl))
    best_log_shape = prediction(physical[best])
    amplitude_solution = np.linalg.lstsq(design, inverse_root @ (np.log(observed) - best_log_shape[rid, fid]), rcond=None)[0]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        best_region_flux = np.exp(best_log_shape + amplitude_solution[:, None])
    regional_spectra = []
    for index in range(len(positions)):
        def values(array, use):
            return [float(value) if selected and np.isfinite(value) else None for value, selected in zip(array, use)]
        regional_spectra.append({"id": index + 1, "branch": int(labels[region_branch[index]]),
            "pixel_xy": [int(rx[index]), int(ry[index])],
            "observed_jy_per_beam": values(flux[:, ry[index], rx[index]], region_bands[index]),
            "rms_jy_per_beam": values(rms[:, ry[index], rx[index]], region_bands[index]),
            "model_jy_per_beam": values(best_region_flux[index], np.ones(len(frequencies), dtype=bool))})
    best_reduced_chi2 = float(-2 * logl[best] / dof)
    adequate = best_reduced_chi2 <= 4.
    slope_design = amplitudes * np.log(frequencies[fid, None] / 1000.)
    projected_slopes = projector @ slope_design
    pl_coefficients = np.linalg.lstsq(projected_slopes, projected_data, rcond=None)[0]
    pl_residual = projected_data - projected_slopes @ pl_coefficients
    pl_chi2 = float(pl_residual @ pl_residual)
    pl_dof = len(rid) - 2 * len(positions)
    curvature_improvement = pl_chi2 + 2 * float(logl[best])
    curvature_required = bool(curvature_improvement >= 9. and pl_chi2 / max(pl_dof, 1) > 2.)
    status = ("model_inadequate" if not adequate else "sampling_unverified" if sampler_status != "replicate_consistent"
              else "curvature_unconstrained" if not curvature_required else "model_conditioned")
    age_flags = np.zeros(distance.shape, dtype=np.uint8)
    age_flags[~source_mask] |= 1
    age_flags[source_mask & ~source_valid] |= 2
    if core_policy == "separate":
        age_flags[core_mask] |= 4
    age_flags[source_mask & ~geometry_valid] |= 8
    maps = {"VALID": valid.astype(np.uint8), "AGE_VALID": valid.astype(np.uint8),
            "SOURCE_VALID": source_valid.astype(np.uint8), "SOURCE_MASK": source_mask.astype(np.uint8),
            "CORE": core_mask.astype(np.uint8), "AGE_FLAGS": age_flags,
            "INTENSITY": np.where(source_mask, flux[0], np.nan), "BRANCH": branches,
            "DIST_KPC": distance, "ALPHA": spectrum.alpha, "N_BANDS": spectrum.n_bands,
            "REGION_ID": np.zeros(distance.shape, dtype=np.int16)}
    maps["REGION_ID"][ry, rx] = np.arange(1, len(positions) + 1)
    for name in ("TRANSPORT", "T_LO68", "T_HI68", "T_LO95", "T_HI95"):
        maps[name] = np.full(distance.shape, np.nan)
    map_branch = np.searchsorted(labels, branches[yy, xx])
    speed_draws = physical[:, :len(labels)]
    eta_draws = physical[:, -1] if "deceleration" in names else np.full(len(z), deceleration)
    mapped_distances = distance[yy, xx]
    for start in range(0, len(yy), 2048):
        part = slice(start, start + 2048)
        s = mapped_distances[part][None, :]
        b = map_branch[part]
        age = KPC_KMS_TO_MYR * (s + eta_draws[:, None] * s ** 2 / (2 * lengths[b][None, :])) / speed_draws[:, b]
        quantiles = np.quantile(age, [.5, .16, .84, .025, .975], axis=0)
        for name, values in zip(("TRANSPORT", "T_LO68", "T_HI68", "T_LO95", "T_HI95"), quantiles):
            maps[name][yy[part], xx[part]] = values
    spatial_mean, spatial_median = [], []
    for index in range(len(z)):
        age = KPC_KMS_TO_MYR * (mapped_distances + eta_draws[index] * mapped_distances ** 2 / (2 * lengths[map_branch])) / speed_draws[index, map_branch]
        spatial_mean.append(np.mean(age)); spatial_median.append(np.median(age))
    samples["map_area_mean_myr"], samples["map_area_median_myr"] = np.array(spatial_mean), np.array(spatial_median)
    branch_summaries = {}
    for index, label in enumerate(labels):
        tip = KPC_KMS_TO_MYR * lengths[index] * (1 + eta_draws / 2) / speed_draws[:, index]
        samples[f"endpoint_transport_myr_branch_{label}"] = tip
        branch_summaries[str(label)] = {"traced_projected_length_kpc": float(lengths[index]),
                                      "endpoint_transport_myr": _summarize(tip)}
    beam_area = math.pi * beam["bmaj_deg"] * beam["bmin_deg"] / (4 * math.log(2))
    warnings = [
        ("Magnetic field is conditioned exactly on the user-supplied value; no field uncertainty was supplied or propagated. Flow/transport inferences remain conditional on spectra, geometry and physical priors." if fixed_field is not None else
         "Transport age and magnetic field/flow speed are structurally degenerate; absolute ages depend on the stated priors even when spectral curvature is measured."),
        "One active steady flow, zero age at path origin, constant shared magnetic field and injection index; no adiabatic losses, reacceleration, diffusion, restart or time-varying paths.",
        "Beam-averaged ageing spectra assume uniform intrinsic emissivity within each local branch beam; amplitude gradients and unresolved population mixing are not reconstructed.",
        "Thermal spatial covariance approximates white noise convolved with the Gaussian restoring beam; actual interferometric noise and missing-flux errors may differ.",
        "The mapped field interpolates a shared parametric flow model; image pixels are not independent age measurements.",
        "WCS and common-beam checks do not verify matched uv coverage, flux recovery or calibration correctness.",
        "Geometry is conditioned on the supplied paths; trajectory and deprojection uncertainties are not inferred.",
        "The emitting flow is treated as nonrelativistic; Doppler frequency shifts and fluid-frame time corrections are omitted.",
        "Thermal errors are assumed independent across frequency bands; overlap or shared imaging residuals require separate assessment.",
        "SMC particles are correlated; replicate agreement is a diagnostic, not a proof of posterior convergence.",
    ]
    if not adequate:
        warnings.append("The declared steady radiative-advection model fails the fit-quality threshold. Age summaries are diagnostic only, not a supported scientific estimate.")
    if sampler_status != "replicate_consistent":
        warnings.append("Independent sampler agreement has not been established; increase particles/rejuvenation and compare runs before interpreting intervals.")
    if not curvature_required:
        warnings.append("Independent regional power laws describe the spectra adequately or ageing curvature improves chi-square by less than 9; a radiative break is not established by this heuristic. Transport ages can therefore be set mainly by priors and slope/geometry assumptions.")
    if any(run["physical_domain_prior_rejection_fraction"] > .05 for run in sampler_diagnostics):
        warnings.append("More than 5% of sampled prior draws lie outside the ultrarelativistic spectral-kernel domain. Posterior inference conditions on valid support; prior-comparison summaries retain the original bounded priors.")
    if pyrats is not None:
        if fixed_field is None:
            warnings.extend(pyrats["provenance"]["warnings"])
            warnings.append("The PyRATS field calculation conditions on its reported injection spectrum; this plug-in input does not jointly refit the equipartition field as the transport-model injection index varies.")
        else:
            warnings.append('PyRATS supplies upstream geometry/property provenance only in this run. Its magnetic-field distribution is not applied; the user-fixed value overrides the default B=0.4 B_eq prescription.')
            if not pyrats['provenance']['image_association']['pixel_identity_verified']:
                warnings.append('PyRATS image association is based on reported paths/labels; raw/prepared pixel identity against the report is not independently verified.')
    if scale_consistency['status'] != 'consistent':
        warnings.append('Transport-path and PyRATS distance scales are '+scale_consistency['status']+'; record geometry.kpc_per_arcsec and use the same cosmology for paths, volume and field input. No automatic path rescaling is performed.')
    if core_spec is not None:
        if core_policy == "include":
            warnings.extend(core_diagnostics["warnings"][:3])
        else:
            warnings.append("The central component is included in source intensity, spectral diagnostics and SOURCE_VALID coverage. Its age is unresolved and flagged, rather than forced into the tail JP likelihood or assigned zero across a finite beam. Transport statistics cover AGE_VALID emission only.")
    from astropy import __version__ as astropy_version
    metadata = {"schema_version": "fr2dynage-radio-posterior-1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"name": "PyRADAGE", "pyradage": __version__, "fr2dynage": __version__, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "astropy": astropy_version, "constants": CONSTANT_SET}, "status": status,
        "source": {**source, "radio_class": radio_class},
        "distance_scale_consistency": scale_consistency,
        "spectral_model": spectral_model,
        "clock": {"kind": "model_conditioned_plasma_transport_time", "time_frame": "host proper time",
                  "equation": "v(s)=v0/(1+eta*s/L); t(s)=977.7922*(s+eta*s^2/(2L))/v0 [Myr,kpc,km/s]",
                  "speed_frame": "projected plasma transport", "age_origin_myr": 0., "geometry": geometry,
                  "model": model},
        "global_dynamical_age_myr": {"status": "not_identifiable_in_this_model", "mean": None, "median": None,
            "reason": "Steady transport does not determine source turn-on, unobserved oldest plasma or episode lifetime; map summaries are not global source ages."},
        "map_statistics": {"scope": ("AGE_VALID detected flow emission including the core wherever path geometry is supplied; equal pixel area in tangent-plane approximation" if core_policy == "include" else "AGE_VALID detected flow emission only, excluding the explicitly flagged unresolved central component and missing geometry; equal pixel area in tangent-plane approximation"), "units": "Myr",
            "interpretation": "diagnostic_only" if status != 'model_conditioned' or scale_consistency['status'] == 'inconsistent' else "model_and_prior_conditioned_transport",
            "area_weighted_mean": _summarize(spatial_mean), "area_weighted_median": _summarize(spatial_median),
            "uncertainty": "compute each spatial statistic for every shared posterior parameter draw; geometry and selection systematics excluded"},
        "branches": branch_summaries, "posterior_parameters": {name: _summarize(samples[name]) for name in names},
        "prior_parameters": {name: _summarize(prior_draws[:, index]) for index, name in enumerate(names)
                             if name in sampled_names},
        "priors": {name: prior.specification for name, prior in zip(sampled_names, priors)},
        "free_parameters": sampled_names,
        "fixed_parameters": {'magnetic_field_ug':fixed_field['value_ug']} if fixed_field is not None else {},
        "prior_description": config["prior_description"], "reference_catalogue": catalogue_provenance,
        "pyrats_properties": pyrats,
        "field_prescription": ({'kind':'user_fixed', 'equation':f"B = {fixed_field['value_t']:.12g} T (user supplied)",
                                'value_t':fixed_field['value_t'], 'value_ug':fixed_field['value_ug'],
                                'fraction':None, 'description':fixed_field['description'],
                                'uncertainty':'conditioned exactly on the user value; no field uncertainty was supplied',
                                'overrides_default_equipartition_fraction':True} if fixed_field is not None else
                               {"kind":"equipartition_fraction", "equation": "B = 0.4 B_eq", "fraction": ADOPTED_B_EQ_FRACTION,
                                "ratio_is_fixed": True,
                                "uncertainty": "propagated equipartition-field uncertainty; the fraction is not sampled"} if pyrats is not None else None),
        'fixed_field_overrides_reference_field': fixed_overrides_reference_field,
        "pyrats_overrides_reference_field": pyrats_overrides_reference_field,
        "explicit_prior_overrides": overridden,
        "identifiability": {"absolute_clock": 'fixed_field_conditioned_transport' if fixed_field is not None else "prior_dependent_field_speed_ridge", "global_lifetime": "not_identifiable",
            "prior_comparison_basis": "original physically bounded parameter priors, before any joint cooling-domain rejection",
            "curvature": "required_by_heuristic" if curvature_required else "not_required_by_radio_data",
            "posterior_prior_std_ratios": {name: float(np.std(physical[:, index]) / max(np.std(prior_draws[:, index]), 1e-12)) for index, name in enumerate(names) if name in sampled_names}},
        "fit": {"best_chi2": float(-2 * logl[best]), "residual_dof_after_shared_parameters": dof,
                "best_sampled_parameters": {name: float(physical[best, index]) for index, name in enumerate(names) if name in sampled_names},
                'best_model_parameters': {name: float(physical[best,index]) for index,name in enumerate(names)},
                "best_reduced_chi2": best_reduced_chi2, "adequate_at_threshold_4": adequate,
                "regional_powerlaw_chi2": pl_chi2, "regional_powerlaw_dof": pl_dof,
                "curvature_chi2_improvement": curvature_improvement,
                "curvature_heuristic": "Declared ageing spectrum improves chi2 >=9 over independent regional power laws and power-law reduced chi2 >2; nonnested diagnostic, not a formal detection significance",
                "normalizations": "one flat-prior log-amplitude per observed region analytically marginalized",
                "covariance_condition_number": condition, "calibration_correlation": cal_correlation.tolist()},
        "sampling": {"method": ('tempered SMC with Gaussian/prior Metropolis in free-parameter coordinates; field excluded from sampler' if fixed_field is not None else "tempered SMC with random-walk/prior Metropolis, Jacobian-corrected field-speed ridge and invariant flow-shape moves"), "status": sampler_status,
                     "moves_per_stage": {"generic_gaussian_or_prior": generic_steps, "field_ridge": 0 if fixed_field is not None else 1, "flow_shape": 0 if fixed_field is not None else steps},
                     'replicate_diagnostic_parameters':sampled_names,
                     "replicate_median_range_over_pooled_sd": replica_difference, "runs": sampler_diagnostics},
        "regions": [{"id": index + 1, "pixel_xy": [int(rx[index]), int(ry[index])], "branch": int(labels[region_branch[index]]),
                     "bands": int(region_bands[index].sum()), "path_nodes_kpc": path_nodes[index].tolist()} for index in range(len(positions))],
        "fitted_regional_spectra": {"frequencies_mhz": frequencies.tolist(), "regions": regional_spectra,
            "model_label": "Flow / JP–Tribble" if use_tribble else "Flow / JP",
            "interpretation": "maximum-likelihood sampled physical parameters with conditional GLS regional log-amplitudes; displayed RMS omits calibration/correlation, full fit covariance is documented separately; curves are not posterior predictive intervals"},
        "core": core_diagnostics,
        "source_coverage": {"selected_footprint_pixels": int(source_mask.sum()),
            "source_detected_pixels": int(source_valid.sum()), "age_valid_pixels": int(valid.sum()),
            "core_pixels": int(core_mask.sum()), "core_detected_pixels": int(np.sum(core_mask & source_valid)),
            "core_age_valid_pixels": int(np.sum(core_mask & valid)),
            "detected_pixels_without_age": int(np.sum(source_valid & ~valid))},
        "age_flags": {"0": "eligible for the conditional flow clock", "1": "outside selected source footprint",
            "2": "insufficient spectral detections", "4": "central age unresolved; source data retained",
            "8": "missing or invalid path/branch geometry", "combination": "bitwise OR; VALID/AGE_VALID identify estimable transport pixels"},
        "intensity_reference_frequency_mhz": float(frequencies[0]),
        "valid_pixels": int(valid.sum()), "beam_scale_samples": len(positions), "detected_beam_areas": float(valid.sum() * pixel_area / beam_area),
        "frequencies_mhz": frequencies.tolist(), "configuration_path": str(path), "configuration_sha256": hashlib.sha256(raw).hexdigest(),
        "input_configuration": config, "input_files": records,
        "warnings": warnings}
    from .reliability import radio_parameter_quality, verify_input_records
    metadata['parameter_quality'] = radio_parameter_quality(metadata)
    verify_input_records([{'path': str(path), 'sha256': metadata['configuration_sha256']}, *records])
    json.dumps(metadata, allow_nan=False)
    return RadioInferenceResult(reference, maps, metadata, samples, frozenset(inputs))


def write_radio_products(result: RadioInferenceResult, output_fits, summary_json, *, samples_path=None,
                         preview_path=None, overwrite=False):
    """Preflight all paths, then atomically publish each complete product."""
    from .maps import _preview_targets
    targets = [Path(p).expanduser().resolve() for p in (output_fits, summary_json, samples_path) if p is not None]
    if preview_path is not None:
        targets += [path for _, path in _preview_targets(preview_path)]
    if len(set(targets)) != len(targets):
        raise MapInputError("all radio product paths must differ")
    for target in targets:
        if target in result.input_paths:
            raise MapInputError("radio products must not overwrite any input image, catalogue or configuration")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; request overwrite explicitly")
        target.parent.mkdir(parents=True, exist_ok=True)
    fits, _, _ = _astropy()
    header = result.reference.header.copy()
    header["BUNIT"] = "Myr"
    header["AGEKIND"] = "TRANSPORT"
    header["DYNMODEL"] = "STEADYTJP" if result.metadata["clock"]["model"]["kind"] == "steady_jp_tribble_advection" else "STEADYJP"
    header["AGEBASIS"] = "PRIOR"
    header["AGESTAT"] = result.metadata["status"]
    header["GLOBAGE"] = "UNAVAILABLE"
    header["RADCLASS"] = result.metadata["source"]["radio_class"]
    header["VERSION"] = result.metadata["software"]["fr2dynage"]
    if result.metadata.get("core") is not None:
        policy = result.metadata["core"].get("policy", "separate")
        header["COREPOL"] = policy.upper()
        header["COREAGE"] = ("MODELLED" if policy == "include" else "UNRESOLVED", "Core clock treatment; see provenance")
    prescription = result.metadata.get('field_prescription') or {}
    if prescription.get('kind') == 'user_fixed':
        header['BINPUT'] = 'USERFIX'
        header['BFIX_T'] = prescription['value_t']
        header['BFIX_UG'] = prescription['value_ug']
        header['B_UG'] = prescription['value_ug']
        header['HISTORY'] = 'Field fixed by user; field uncertainty excluded, not measured as zero.'
    elif prescription.get('fraction') is not None:
        header["BFRACEQ"] = (prescription["fraction"], "Adopted fixed B/B_eq")
    header["HISTORY"] = "Prior-conditioned posterior transport field, not a measured AGN lifetime."
    hdus = [fits.PrimaryHDU(result.maps["TRANSPORT"], header=header)]
    for name, data in result.maps.items():
        if name == "TRANSPORT":
            continue
        current = result.reference.header.copy()
        current["BUNIT"] = "Myr" if name.startswith("T_") else "kpc" if name == "DIST_KPC" else "Jy/beam" if name == "INTENSITY" else "1"
        if name == "INTENSITY":
            current["FREQMHZ"] = result.metadata["intensity_reference_frequency_mhz"]
        if name == "AGE_FLAGS":
            for bit, meaning in [(1, "Outside selected source"), (2, "Insufficient spectral detections"),
                                 (4, "Unresolved central age; data retained"), (8, "Missing/invalid flow geometry")]:
                current[f"FLG{bit}"] = meaning
        if name in {"VALID", "AGE_VALID"}:
            current["HISTORY"] = ("1=conditional flow age; core included where geometry supplied." if (result.metadata.get("core") or {}).get("policy") == "include" else "1=eligible conditional flow age; excludes unresolved core.")
        if name == "SOURCE_VALID":
            current["HISTORY"] = "Detected source data including separately classified core."
        hdus.append(fits.ImageHDU(data, header=current, name=name))
    encoded = json.dumps(result.metadata, allow_nan=False, ensure_ascii=False).encode("utf8")
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="JSON_UTF8", format=f"{len(encoded)}B", array=np.frombuffer(encoded, dtype=np.uint8)[None, :])], name="PROVENANCE"))

    def atomic_bytes(target, writer, suffix):
        target = Path(target).expanduser().resolve()
        descriptor, filename = tempfile.mkstemp(prefix="." + target.name + ".", suffix=suffix, dir=target.parent)
        os.close(descriptor)
        temporary = Path(filename)
        try:
            writer(temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    # Prepare the preview first, so a missing plotting dependency leaves no partial bundle.
    if preview_path is not None:
        render_radio_preview(result, preview_path, overwrite=overwrite)
    with fits.HDUList(hdus) as document:
        atomic_write_fits(output_fits, document, overwrite=overwrite)
    atomic_bytes(summary_json, lambda target: target.write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8"), ".json")
    if samples_path is not None:
        atomic_bytes(samples_path, lambda target: np.savez_compressed(target, **result.samples), ".npz")


def render_radio_preview(result: RadioInferenceResult, preview_path, *, overwrite=False):
    """Render publication PNG/PDF/SVG without refitting or rewriting FITS."""
    from .maps import _preview_targets
    from .plotting import (publication_figure, save_publication_figure,
                           AXIS_LABEL_SIZE, TICK_SIZE, LEGEND_SIZE)
    targets = _preview_targets(preview_path)
    for _, target in targets:
        if target in result.input_paths:
            raise MapInputError("radio previews must not overwrite an input image, catalogue or configuration")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; request overwrite explicitly")
        target.parent.mkdir(parents=True, exist_ok=True)
    has_core = result.metadata.get("core") is not None
    separate_core = has_core and result.metadata["core"].get("policy", "separate") == "separate"
    try:
        figure = publication_figure(figsize=(14, 12) if has_core else (20, 6.5), layout="constrained")
    except ImportError as exc:
        raise MapInputError("publication previews require python -m pip install 'pyradage[maps]'") from exc
    panels = [(result.maps["TRANSPORT"], "Posterior median\ntransport time", "Myr"),
              (result.maps["T_HI68"] - result.maps["T_LO68"], "68% interval width", "Myr"),
              (result.maps["ALPHA"], "Power-law spectral\nindex diagnostic", "")]
    if has_core:
        panels.insert(0, (1000 * result.maps["INTENSITY"],
            f"Source including core\n{result.metadata['intensity_reference_frequency_mhz']:.1f} MHz", "mJy/beam"))
    for index, (data, title, unit) in enumerate(panels, 1):
        axis = figure.add_subplot(2, 2, index, projection=result.reference.wcs) if has_core else figure.add_subplot(1, 3, index, projection=result.reference.wcs)
        norm = None
        if has_core and index == 1:
            from matplotlib.colors import AsinhNorm
            values = data[np.isfinite(data)]
            high = float(np.max(values)) if values.size else 1.
            norm = AsinhNorm(linear_width=max(high*.02, 1e-12), vmin=0., vmax=max(high, 1e-12))
        shown = axis.imshow(data, origin="lower", cmap="magma" if unit == "mJy/beam" else "viridis", norm=norm)
        if has_core:
            central = result.maps["CORE"].astype(bool) & result.maps["SOURCE_VALID"].astype(bool)
            if separate_core and index in {2, 3}:
                rgba = np.zeros((*central.shape, 4))
                rgba[central] = (.55, .55, .55, 1.)
                axis.imshow(rgba, origin="lower")
            if separate_core and np.any(central) and not np.all(central):
                axis.contour(central, levels=[.5], colors="cyan", linewidths=1.2)
            position = result.metadata["input_configuration"].get("core", {}).get("position_icrs_deg")
            if position is not None:
                from astropy.coordinates import SkyCoord
                cx, cy = result.reference.wcs.world_to_pixel(SkyCoord(*position, unit="deg", frame="icrs"))
                axis.plot(cx, cy, marker="+", color="cyan", ms=12, mew=2., label="Core")
                axis.legend(fontsize=LEGEND_SIZE, loc="upper right")
        axis.set_title(title)
        for coordinate_index, physical in enumerate(result.reference.wcs.world_axis_physical_types):
            label = {"pos.eq.ra": "Right ascension", "pos.eq.dec": "Declination", "pos.galactic.lon": "Galactic longitude", "pos.galactic.lat": "Galactic latitude"}.get(physical)
            if label:
                coordinate = axis.coords[coordinate_index]
                coordinate.set_axislabel(label, fontsize=AXIS_LABEL_SIZE)
                coordinate.set_ticklabel(size=TICK_SIZE, exclude_overlapping=True)
                coordinate.set_ticks(number=4)
        colorbar = figure.colorbar(shown, ax=axis, label=unit, shrink=.8, pad=.025)
        colorbar.ax.tick_params(labelsize=TICK_SIZE)
        colorbar.set_label(unit, fontsize=AXIS_LABEL_SIZE)
        if has_core and index == 1:
            from matplotlib.ticker import FixedLocator, FuncFormatter
            exponent = int(math.floor(math.log10(max(high, 1e-12))))
            candidates = [step*10.**power for power in range(exponent-2, exponent+1) for step in (1., 3.)]
            ticks = [0.] + [value for value in candidates if high*.025 <= value <= high]
            colorbar.locator = FixedLocator(ticks)
            colorbar.formatter = FuncFormatter(lambda value, position: f"{value:g}")
            colorbar.update_ticks()
    status = result.metadata["status"].replace("_", " ")
    title = result.metadata["source"]["name"] + "\nConditional transport model; " + status
    if has_core:
        title += ("\nGrey: included core emission with unresolved transport age" if separate_core else "\nCore included in the conditional transport model")
    figure.suptitle(title)
    staged = []
    try:
        for kind, target in targets:
            descriptor, filename = tempfile.mkstemp(prefix="."+target.name+".", suffix="."+kind, dir=target.parent)
            os.close(descriptor)
            temporary = Path(filename)
            staged.append((temporary, target))
            save_publication_figure(figure, temporary, formats=(kind,), clear=False)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        for temporary, target in staged:
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
        return [target for _, target in targets]
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        figure.clear()
