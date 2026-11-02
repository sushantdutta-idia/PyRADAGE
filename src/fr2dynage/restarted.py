"""Component-resolved fitting and chronology for restarted FR II sources.

A restarted source is not assigned one dynamical age.  The diffuse outer
double is fitted as a remnant and the inner double as a new active episode.
Their spectra and geometries must be supplied separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .constants import C_LIGHT, MYR_S
from .fitting import (
    FitBounds,
    FitResult,
    _fractional_error_sigma_ln,
    _spectrum_records,
    fit_joint,
)
from .model import (
    KDAForwardModel,
    ModelAssumptions,
    ModelDomainError,
    Observation,
    _is_finite_real,
)
from .numerics import estimate_local_covariance
from .radio_classes import require_fr2_class
from .remnant import (
    KDAExtForwardModel,
    RemnantFitBounds,
    fit_remnant_joint,
)


class RestartedFitError(RuntimeError):
    """Raised when component data cannot define a restarted-source fit."""


@dataclass
class RestartedFitResult:
    """Two-episode result for a component-resolved restarted source."""

    source_name: str
    outer_remnant: FitResult
    inner_active: FitResult
    chronology: dict[str, float | str | None]
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.outer_remnant.success and self.inner_active.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "activity_state": "restarted",
            "method": "component_resolved_outer_kdaext_plus_inner_kda",
            "success": self.success,
            "components": {
                "outer_remnant": self.outer_remnant.to_dict(),
                "inner_active": self.inner_active.to_dict(),
            },
            "chronology": self.chronology,
            "warnings": self.warnings,
        }


def _power_law_diagnostic(observation: Observation) -> dict[str, float | bool | None]:
    """Measure whether a component spectrum is consistent with one power law."""

    log_frequency = np.log10([point.frequency_hz for point in observation.spectrum])
    log_power = np.log10([point.power_w_hz_sr for point in observation.spectrum])
    all_errors = all(point.fractional_error is not None for point in observation.spectrum)
    if all_errors:
        sigma = np.asarray(
            [
                _fractional_error_sigma_ln(point.fractional_error) / math.log(10.0)
                for point in observation.spectrum
                if point.fractional_error is not None
            ]
        )
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise RestartedFitError("fractional errors are too small to represent log-power weights")
        # A common weight scale does not change the fitted line. Normalize it
        # before polyfit squares weights internally.
        coefficients = np.polyfit(log_frequency, log_power, 1, w=np.min(sigma) / sigma)
    else:
        sigma = None
        coefficients = np.polyfit(log_frequency, log_power, 1)
    residual = log_power - np.polyval(coefficients, log_frequency)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        chi2 = (
            float(np.dot(residual / sigma, residual / sigma))
            if sigma is not None
            else None
        )
    if chi2 is not None and not math.isfinite(chi2):
        raise RestartedFitError("power-law diagnostic exceeds floating-point range for the supplied errors")
    degrees_of_freedom = len(log_power) - 2
    reduced_chi2 = (
        chi2 / degrees_of_freedom
        if chi2 is not None and degrees_of_freedom > 0
        else None
    )
    maximum_residual = float(np.max(np.abs(residual)))
    consistent = (
        reduced_chi2 <= 2.0
        if reduced_chi2 is not None
        else maximum_residual < 0.03
    )
    return {
        "best_power_law_spectral_index": float(-coefficients[0]),
        "maximum_residual_dex": maximum_residual,
        "power_law_chi2": chi2,
        "power_law_degrees_of_freedom": float(degrees_of_freedom),
        "power_law_reduced_chi2": reduced_chi2,
        "consistent_with_single_power_law": bool(consistent),
    }


def _validate_fixed_jet_power(jet_power_w: float, bounds: FitBounds) -> None:
    if not _is_finite_real(jet_power_w) or jet_power_w <= 0.0:
        raise RestartedFitError("fixed inner jet power must be finite and positive")
    log_q = math.log10(jet_power_w)
    if not bounds.log10_jet_power_w[0] <= log_q <= bounds.log10_jet_power_w[1]:
        raise RestartedFitError(
            "fixed inner jet power lies outside the inner log10_jet_power_w bounds"
        )


def _fit_inner_with_fixed_jet_power(
    model: KDAForwardModel,
    bounds: FitBounds,
    jet_power_w: float,
) -> FitResult:
    """Fit inner age and injection index conditional on an external Q prior."""

    _validate_fixed_jet_power(jet_power_w, bounds)
    points = model.observation.spectrum
    frequencies = np.asarray([point.frequency_hz for point in points])
    observed_log = np.log10([point.power_w_hz_sr for point in points])
    has_all_errors = all(point.fractional_error is not None for point in points)
    sigma_log = np.asarray(
        [
            (
                _fractional_error_sigma_ln(point.fractional_error)
                / math.log(10.0)
            )
            if point.fractional_error is not None
            else 1.0
            for point in points
        ]
    )
    if np.any(~np.isfinite(sigma_log)) or np.any(sigma_log <= 0.0):
        raise RestartedFitError("fractional errors are too small to represent log-power weights")
    lower = np.asarray(
        [math.log10(bounds.age_myr[0]), bounds.alpha_inj[0]]
    )
    upper = np.asarray(
        [math.log10(bounds.age_myr[1]), bounds.alpha_inj[1]]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_age, alpha = (float(value) for value in parameters)
        try:
            predicted = model.spectrum(
                frequencies,
                jet_power_w,
                10.0**log_age * MYR_S,
                alpha,
            )
        except (
            ModelDomainError,
            ValueError,
            OverflowError,
            FloatingPointError,
        ):
            return np.full(len(points), 1.0e6)
        if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
            return np.full(len(points), 1.0e6)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weighted = (np.log10(predicted) - observed_log) / sigma_log
        return weighted if np.all(np.isfinite(weighted)) else np.full(len(points), 1.0e6)

    midpoint = 0.5 * (lower + upper)
    starts = [midpoint]
    for age_fraction in (0.2, 0.5, 0.8):
        for alpha_fraction in (0.25, 0.75):
            starts.append(
                np.asarray(
                    [
                        lower[0] + age_fraction * (upper[0] - lower[0]),
                        lower[1] + alpha_fraction * (upper[1] - lower[1]),
                    ]
                )
            )
    candidates = []
    start_failures: list[str] = []
    for start in starts:
        try:
            candidate = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                xtol=2.0e-7,
                ftol=2.0e-7,
                gtol=2.0e-7,
                max_nfev=800,
                x_scale="jac",
            )
        except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            start_failures.append(str(exc))
            continue
        candidates.append(candidate)
    valid = [
        candidate
        for candidate in candidates
        if candidate.success
        and np.all(np.isfinite(candidate.fun))
        and np.all(np.abs(candidate.fun) < 5.0e5)
    ]
    if not valid:
        raise RestartedFitError(
            "the fixed-Q inner fit found no finite converged solution"
        )
    best = min(valid, key=lambda item: float(np.dot(item.fun, item.fun)))
    log_age, alpha = (float(value) for value in best.x)
    age_myr = 10.0**log_age
    state = model.dynamics_state(jet_power_w, age_myr * MYR_S)
    spectrum, chi2_check = _spectrum_records(
        model,
        jet_power_w,
        age_myr,
        alpha,
    )
    derived = model.derived_quantities(
        jet_power_w,
        age_myr * MYR_S,
        alpha,
    )
    objective = float(np.dot(best.fun, best.fun))
    degrees_of_freedom = len(points) - 2
    scale = 1.0 if has_all_errors or degrees_of_freedom <= 0 else objective / degrees_of_freedom
    estimate = estimate_local_covariance(
        best.jac, ["log10_age_myr", "alpha_inj"], scale=scale,
    )
    covariance = estimate.covariance
    correlation = estimate.correlation
    standard_errors = estimate.standard_errors
    information_condition = estimate.information_condition_number
    normalized = (best.x - lower) / (upper - lower)
    near_boundary = bool(
        np.any((normalized < 2.0e-3) | (normalized > 1.0 - 2.0e-3))
    )
    warnings = [
        (
            "The inner age is conditional on an externally fixed restarted-jet "
            "power and the adopted effective old-cocoon environment."
        )
    ]
    if not has_all_errors:
        warnings.append(
            "At least one inner spectrum point has no error; the objective is "
            "not a calibrated likelihood."
        )
    if covariance is None:
        warnings.append(
            "The fixed-Q inner covariance is unavailable: " + str(estimate.reason) + "."
        )
    elif information_condition is None or information_condition > 1.0e6:
        warnings.append(
            "The fixed-Q inner information matrix is ill-conditioned."
        )
    if near_boundary:
        warnings.append(
            "At least one fixed-Q inner-fit parameter is on or near a bound."
        )
    return FitResult(
        source_name=model.observation.name,
        method="joint_fixed_jet_power_log_likelihood",
        success=bool(best.success),
        age_myr=age_myr,
        alpha_inj=alpha,
        jet_power_w=jet_power_w,
        rho0_kg_m3=state.rho0_kg_m3,
        objective=objective,
        objective_name=(
            "chi2_log_power"
            if has_all_errors
            else "weighted_sum_squares_log_power"
        ),
        derived=derived,
        spectrum=spectrum,
        diagnostics={
            "optimizer_success": bool(best.success),
            "optimizer_message": str(best.message),
            "optimizer_function_evaluations": int(best.nfev),
            "multi_start_count": len(starts),
            "failed_start_count": len(start_failures),
            "start_failure_examples": start_failures[:3],
            "jet_power_fixed_external_w": jet_power_w,
            "degrees_of_freedom": degrees_of_freedom,
            "chi2": objective if has_all_errors else None,
            "reduced_chi2": (
                objective / degrees_of_freedom
                if has_all_errors and degrees_of_freedom > 0
                else None
            ),
            "covariance_parameter_order": [
                "log10_age_myr",
                "alpha_inj",
            ],
            "local_covariance": covariance,
            "local_correlation": correlation,
            "local_standard_errors": standard_errors,
            "information_matrix_condition_number": information_condition,
            "covariance_unavailable_reason": estimate.reason,
            "parameter_near_boundary": near_boundary,
            "weighted_log_power_chi2_check": chi2_check,
        },
        warnings=warnings,
    )


def fit_restarted(
    outer_observation: Observation,
    outer_assumptions: ModelAssumptions,
    outer_bounds: RemnantFitBounds,
    inner_observation: Observation,
    inner_assumptions: ModelAssumptions,
    inner_bounds: FitBounds,
    *,
    old_jet_speed_fraction_c: float = 1.0,
    inner_jet_power_w: float | None = None,
    allow_underconstrained_inner: bool = False,
) -> RestartedFitResult:
    """Fit the outer and inner doubles independently, then combine their clocks.

    ``old_jet_speed_fraction_c`` converts cessation of particle supply at the
    outer lobe to a central-engine switch-off estimate.  The default, one,
    gives the minimum propagation delay and should be treated as a lower-limit
    assumption rather than a measurement.
    """

    require_fr2_class(outer_observation.radio_class)
    require_fr2_class(inner_observation.radio_class)
    if (
        not _is_finite_real(old_jet_speed_fraction_c)
        or not 0.0 < old_jet_speed_fraction_c <= 1.0
    ):
        raise RestartedFitError("old jet speed must lie in (0, 1] times c")
    if inner_jet_power_w is not None:
        _validate_fixed_jet_power(inner_jet_power_w, inner_bounds)
    redshift_tolerance = 1.0e-8 * max(
        1.0,
        abs(outer_observation.redshift),
        abs(inner_observation.redshift),
    )
    if abs(outer_observation.redshift - inner_observation.redshift) > redshift_tolerance:
        raise RestartedFitError(
            "outer and inner component files must describe the same source redshift"
        )
    if not math.isclose(
        outer_observation.geometry.inclination_deg,
        inner_observation.geometry.inclination_deg,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise RestartedFitError(
            "outer and inner component inclination conventions must agree"
        )
    if (
        outer_observation.geometry.lobe_length_m
        <= inner_observation.geometry.lobe_length_m
    ):
        raise RestartedFitError(
            "outer one-lobe length must exceed the separately measured inner one-lobe length"
        )
    if len(outer_observation.spectrum) < 5:
        raise RestartedFitError(
            "the four-parameter outer-remnant fit requires at least five "
            "component-separated detections"
        )
    if len(inner_observation.spectrum) < 4:
        raise RestartedFitError(
            "the three-parameter inner-active fit requires at least four "
            "component-separated detections"
        )

    inner_power_law = _power_law_diagnostic(inner_observation)
    if (
        bool(inner_power_law["consistent_with_single_power_law"])
        and inner_jet_power_w is None
        and not allow_underconstrained_inner
    ):
        raise RestartedFitError(
            "the inner spectrum is consistent with a single power law and has "
            "no unique dynamical age; provide inner_jet_power_w "
            "(--inner-jet-power-w in the CLI) or explicitly set "
            "allow_underconstrained_inner=True"
        )

    outer_result = fit_remnant_joint(
        KDAExtForwardModel(outer_observation, outer_assumptions),
        outer_bounds,
    )
    inner_model = KDAForwardModel(inner_observation, inner_assumptions)
    inner_result = (
        _fit_inner_with_fixed_jet_power(
            inner_model,
            inner_bounds,
            inner_jet_power_w,
        )
        if inner_jet_power_w is not None
        else fit_joint(inner_model, inner_bounds)
    )

    travel_time_myr = (
        outer_observation.geometry.lobe_length_m
        / (old_jet_speed_fraction_c * C_LIGHT)
        / MYR_S
    )
    outer_active_duration_myr = outer_result.derived["active_duration_myr"]
    outer_remnant_duration_myr = outer_result.derived["remnant_duration_myr"]
    lobe_clock_gap_myr = outer_remnant_duration_myr - inner_result.age_myr
    central_engine_quiescent_myr = lobe_clock_gap_myr + travel_time_myr
    chronology: dict[str, float | str | None] = {
        "outer_total_age_myr": outer_result.age_myr,
        "outer_active_duration_myr": outer_active_duration_myr,
        "outer_lobe_fading_duration_myr": outer_remnant_duration_myr,
        "inner_active_age_myr": inner_result.age_myr,
        "lobe_clock_gap_myr": lobe_clock_gap_myr,
        "old_jet_travel_time_myr": travel_time_myr,
        "assumed_old_jet_speed_fraction_c": old_jet_speed_fraction_c,
        "central_engine_quiescent_interval_myr": central_engine_quiescent_myr,
        "jet_power_ratio_new_to_old": (
            inner_result.jet_power_w / outer_result.jet_power_w
        ),
        "clock_definition": (
            "t_quiet_AGN = t_travel(old jet) + t_fading(outer lobe) "
            "- t_age(inner double)"
        ),
        "uncertainty_propagation": (
            "point estimate only; component covariances are reported separately"
        ),
        "inner_age_constraint": (
            "externally_fixed_jet_power"
            if inner_jet_power_w is not None
            else "unconstrained_joint_spectrum_and_size_fit"
        ),
    }

    inner_result.diagnostics["single_power_law_diagnostic"] = inner_power_law
    warnings = [
        (
            "Restarted fitting requires genuinely component-separated outer- "
            "and inner-double geometries and spectra; no deblending is inferred."
        ),
        (
            "The two episodes have independent jet powers, injection indices, "
            "and conditional environments; the result is not one source age."
        ),
        (
            "The inferred inner density is an effective old-cocoon inertia and "
            "is never copied from the undisturbed outer atmosphere; entrainment "
            "and mixing remain model systematics."
        ),
        (
            "The central-engine interval depends linearly on the assumed old-jet "
            "travel time and ignores orientation/light-travel asymmetry."
        ),
        (
            "Restarted chronology is formed from best-fit clocks; propagate the "
            "joint component distributions before quoting an interval uncertainty."
        ),
    ]
    if inner_assumptions.beta != 0.0:
        warnings.append(
            "The inner component beta is not zero. Inner doubles are commonly "
            "approximated as propagating through a uniform old cocoon; test beta=0 "
            "and an external inner-density or jet-power constraint."
        )
    if bool(inner_power_law["consistent_with_single_power_law"]):
        warning = (
            "The inner spectrum is statistically consistent with a single power "
            "law; its age is conditional on the supplied restarted-jet power."
            if inner_jet_power_w is not None
            else
            "The inner spectrum is statistically consistent with a single power "
            "law; the user explicitly allowed a generally non-identifiable age "
            "without an external inner-density or jet-power constraint."
        )
        warnings.append(warning)
        inner_result.warnings.append(warning)
    if central_engine_quiescent_myr < 0.0:
        warnings.append(
            "The inferred central-engine quiescent interval is negative; the "
            "component clocks and/or assumed old-jet speed are inconsistent with "
            "a non-overlapping restart."
        )
    elif lobe_clock_gap_myr < 0.0:
        warnings.append(
            "The lobe-frame clocks overlap, although the added old-jet travel "
            "delay still permits a positive central-engine quiescent interval."
        )

    common_name = outer_observation.name
    if common_name != inner_observation.name:
        common_name = f"{outer_observation.name} / {inner_observation.name}"
        warnings.append(
            "Outer and inner source names differ; they were paired because their "
            "redshifts agree."
        )
    return RestartedFitResult(
        source_name=common_name,
        outer_remnant=outer_result,
        inner_active=inner_result,
        chronology=chronology,
        warnings=warnings,
    )


__all__ = [
    "RestartedFitError",
    "RestartedFitResult",
    "fit_restarted",
]
