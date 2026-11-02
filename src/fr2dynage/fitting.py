"""Inverse solvers for the KDA forward model.

Two deliberately distinct estimators are provided:

``classic`` reproduces the three-stage DYNAGE procedure published by
Machalski et al. (2007).  Its pairwise intersection statistic is unweighted
and the final minimum-Q*t choice is a physical heuristic, not a likelihood.

``joint`` performs a simultaneous weighted least-squares fit in log power.
It is useful as a statistical cross-check, but it is still conditional on all
fixed KDA model assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from .constants import MYR_S
from .model import KDAForwardModel, ModelDomainError, Observation, _is_finite_real
from .numerics import (
    estimate_local_covariance,
    fractional_error_sigma_ln as _fractional_error_sigma_ln,
)


class FitError(RuntimeError):
    """Raised when the inverse problem has no valid solution in the bounds."""


@dataclass(frozen=True)
class FitBounds:
    age_myr: tuple[float, float] = (0.1, 500.0)
    alpha_inj: tuple[float, float] = (0.45, 0.90)
    log10_jet_power_w: tuple[float, float] = (34.0, 43.0)
    age_grid_size: int = 23
    alpha_grid_size: int = 17
    power_scan_size: int = 17

    def __post_init__(self) -> None:
        for label, pair in (
            ("age_myr", self.age_myr),
            ("alpha_inj", self.alpha_inj),
            ("log10_jet_power_w", self.log10_jet_power_w),
        ):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(_is_finite_real(value) for value in pair)
            ):
                raise ValueError(f"{label} must contain two finite values")
            if pair[0] >= pair[1]:
                raise ValueError(f"{label} lower bound must be below upper bound")
        if self.age_myr[0] <= 0:
            raise ValueError("age lower bound must be positive")
        if not all(math.isfinite(value * MYR_S) and value * MYR_S > 0 for value in self.age_myr):
            raise ValueError("age bounds must be representable in seconds")
        try:
            valid_power_bounds = all(
                math.isfinite(10.0**value) and 10.0**value > 0
                for value in self.log10_jet_power_w
            )
        except OverflowError:
            valid_power_bounds = False
        if not valid_power_bounds:
            raise ValueError("jet-power bounds must represent finite, positive powers")
        if self.alpha_inj[0] <= 0 or self.alpha_inj[1] >= 2:
            raise ValueError("alpha bounds must lie strictly inside (0, 2)")
        for label, value in (
            ("age_grid_size", self.age_grid_size),
            ("alpha_grid_size", self.alpha_grid_size),
            ("power_scan_size", self.power_scan_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
        if self.age_grid_size < 9 or self.alpha_grid_size < 7 or self.power_scan_size < 7:
            raise ValueError("fit grids are too small for a reliable bounded search")


@dataclass(frozen=True)
class IntersectionSolution:
    age_myr: float
    alpha_inj: float
    log10_q_by_frequency: tuple[float, ...]
    log10_rho0_by_frequency: tuple[float, ...]
    delta: float
    consensus_log10_q: float
    consensus_log10_rho0: float

    @property
    def jet_power_w(self) -> float:
        return 10.0**self.consensus_log10_q

    @property
    def rho0_kg_m3(self) -> float:
        return 10.0**self.consensus_log10_rho0


@dataclass
class FitResult:
    source_name: str
    method: Literal[
        "classic_dynage",
        "joint_log_likelihood",
        "joint_fixed_jet_power_log_likelihood",
        "remnant_joint_log_likelihood",
    ]
    success: bool
    age_myr: float
    alpha_inj: float
    jet_power_w: float
    rho0_kg_m3: float
    objective: float
    objective_name: str
    derived: dict[str, float]
    spectrum: list[dict[str, float | None]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] | None = None
    activity_history: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        document = {
            "source_name": self.source_name,
            "method": self.method,
            "success": self.success,
            "best_fit": {
                "age_myr": self.age_myr,
                "alpha_inj": self.alpha_inj,
                "electron_index_p": 2.0 * self.alpha_inj + 1.0,
                "jet_power_w_per_jet": self.jet_power_w,
                "rho0_kg_m3_conditional_on_core_radius": self.rho0_kg_m3,
            },
            "objective": {"name": self.objective_name, "value": self.objective},
            "derived": self.derived,
            "spectrum": self.spectrum,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "uncertainty": self.uncertainty,
        }
        if self.activity_history is not None:
            document["activity_history"] = self.activity_history
        return document


def _pairwise_delta(log_rho: np.ndarray, log_q: np.ndarray) -> float:
    count = len(log_q)
    if count < 2:
        return math.inf
    distances = []
    for left in range(count - 1):
        for right in range(left + 1, count):
            distances.append(
                math.hypot(
                    float(log_rho[left] - log_rho[right]),
                    float(log_q[left] - log_q[right]),
                )
            )
    return float(np.mean(distances))


def _root_log_jet_power(
    model: KDAForwardModel,
    frequency_hz: float,
    observed_power: float,
    age_s: float,
    alpha_inj: float,
    bounds: FitBounds,
) -> tuple[float, int]:
    """Solve one KDA luminosity equation, detecting multiple roots."""

    lower, upper = bounds.log10_jet_power_w

    # With a homogeneous current-field anchor and fixed observed size,
    # p_c is proportional to Q. Cooling and cohort volumes then do not
    # depend on Q, and P_nu=C_nu*(Q-Q_field_capacity). Check that physical
    # identity at two probe powers before using it; other model variants
    # retain the general multi-root scan below.
    if type(model) is KDAForwardModel and model.assumptions.equipartition_field_ug is not None:
        from .constants import MU_0
        high_q = 10.0**upper
        state = model.dynamics_state(high_q, age_s)
        u_c = state.pressure_pa/(model.assumptions.gamma_c-1.)
        b = .4*model.assumptions.equipartition_field_ug*1e-10
        threshold = high_q*(b*b/(2*MU_0))/u_c
        if not math.isfinite(threshold) or threshold >= high_q:
            raise FitError("jet-power bounds cannot support the adopted magnetic-field energy")
        high_power = model.radio_power(frequency_hz, high_q, age_s, alpha_inj)
        if not math.isfinite(high_power) or high_power <= 0:
            raise FitError("anchored spectrum has no positive emission at this frequency")
        probe_q = .5*(high_q+max(threshold, 10.0**lower))
        probe_power = model.radio_power(frequency_hz, probe_q, age_s, alpha_inj)
        expected = high_power*(probe_q-threshold)/(high_q-threshold)
        if math.isclose(probe_power, expected, rel_tol=2e-6, abs_tol=0.):
            solution_q = threshold + observed_power/high_power*(high_q-threshold)
            if not 10.0**lower <= solution_q <= high_q:
                raise FitError("anchored luminosity root is outside the jet-power bounds")
            check = model.radio_power(frequency_hz, solution_q, age_s, alpha_inj)
            if math.isclose(check, observed_power, rel_tol=5e-6, abs_tol=0.):
                return math.log10(solution_q), 1

    def residual(log_q: float) -> float:
        try:
            predicted = model.radio_power(frequency_hz, 10.0**log_q, age_s, alpha_inj)
        except (ModelDomainError, FloatingPointError, OverflowError):
            return math.nan
        if predicted == 0:
            return -math.inf
        if predicted < 0 or not math.isfinite(predicted):
            return math.nan
        return math.log10(predicted) - math.log10(observed_power)

    brackets: list[tuple[float, float]] = []
    grid = np.linspace(lower, upper, bounds.power_scan_size)
    values = [residual(float(value)) for value in grid]
    for index in range(len(grid) - 1):
        f_left, f_right = values[index], values[index + 1]
        if f_left == 0:
            brackets.append((float(grid[index]), float(grid[index])))
        elif not math.isnan(f_left) and not math.isnan(f_right) and (f_left < 0) != (f_right < 0):
            brackets.append((float(grid[index]), float(grid[index + 1])))
    if math.isfinite(values[-1]) and values[-1] == 0:
        brackets.append((float(grid[-1]), float(grid[-1])))

    if not brackets:
        raise FitError(
            f"no jet-power root at {frequency_hz / 1e6:g} MHz, "
            f"age={age_s / MYR_S:g} Myr, alpha={alpha_inj:g}"
        )

    roots: list[float] = []
    for left, right in brackets:
        if left == right:
            root = left
        else:
            try:
                root = float(brentq(residual, left, right, xtol=2.0e-6, rtol=1.0e-10, maxiter=80))
            except (ValueError, RuntimeError, OverflowError, FloatingPointError):
                # A model-domain gap inside one bracket must not discard
                # independently valid roots elsewhere in the scan.
                continue
        if not roots or all(abs(root - old) > 1.0e-4 for old in roots):
            roots.append(root)
    # The published model reports unique solutions.  If numerical/model
    # extensions create several, use the lowest-power branch and report it.
    if not roots:
        raise FitError("jet-power brackets contained no numerically valid root")
    return min(roots), len(roots)


def intersection_at_age(
    model: KDAForwardModel,
    age_myr: float,
    alpha_inj: float,
    bounds: FitBounds,
) -> tuple[IntersectionSolution, int]:
    age_s = age_myr * MYR_S
    log_q_values: list[float] = []
    log_rho_values: list[float] = []
    max_roots = 1
    for point in model.observation.spectrum:
        log_q, root_count = _root_log_jet_power(
            model,
            point.frequency_hz,
            point.power_w_hz_sr,
            age_s,
            alpha_inj,
            bounds,
        )
        max_roots = max(max_roots, root_count)
        state = model.dynamics_state(10.0**log_q, age_s)
        log_q_values.append(log_q)
        log_rho_values.append(math.log10(state.rho0_kg_m3))
    log_q_array = np.asarray(log_q_values)
    log_rho_array = np.asarray(log_rho_values)
    solution = IntersectionSolution(
        age_myr=age_myr,
        alpha_inj=alpha_inj,
        log10_q_by_frequency=tuple(float(value) for value in log_q_array),
        log10_rho0_by_frequency=tuple(float(value) for value in log_rho_array),
        delta=_pairwise_delta(log_rho_array, log_q_array),
        consensus_log10_q=float(np.mean(log_q_array)),
        consensus_log10_rho0=float(np.mean(log_rho_array)),
    )
    return solution, max_roots


def _age_solution_for_alpha(
    model: KDAForwardModel,
    alpha_inj: float,
    bounds: FitBounds,
) -> tuple[IntersectionSolution, dict[str, Any]]:
    log_age_bounds = tuple(math.log10(value) for value in bounds.age_myr)
    cache: dict[float, tuple[IntersectionSolution | None, int]] = {}

    def evaluate(log_age: float) -> tuple[IntersectionSolution | None, int]:
        key = round(float(log_age), 10)
        if key not in cache:
            try:
                cache[key] = intersection_at_age(model, 10.0**log_age, alpha_inj, bounds)
            except (FitError, ModelDomainError, ValueError, OverflowError, FloatingPointError):
                cache[key] = (None, 0)
        return cache[key]

    grid = np.linspace(log_age_bounds[0], log_age_bounds[1], bounds.age_grid_size)
    objective_values = []
    for log_age in grid:
        solution, _ = evaluate(float(log_age))
        objective_values.append(solution.delta if solution is not None else math.inf)
    finite = np.isfinite(objective_values)
    if not np.any(finite):
        raise FitError(f"no valid age solution for alpha_inj={alpha_inj:g}")
    best_index = int(np.nanargmin(objective_values))
    left_index = max(0, best_index - 1)
    right_index = min(len(grid) - 1, best_index + 1)
    hit_grid_bound = best_index in (0, len(grid) - 1)

    def objective(log_age: float) -> float:
        solution, _ = evaluate(log_age)
        return solution.delta if solution is not None else 1.0e6

    if left_index == right_index:
        optimum_log_age = float(grid[best_index])
    else:
        refined = minimize_scalar(
            objective,
            bounds=(float(grid[left_index]), float(grid[right_index])),
            method="bounded",
            options={"xatol": 2.0e-5, "maxiter": 80},
        )
        optimum_log_age = float(refined.x) if refined.success else float(grid[best_index])
    best, max_roots = evaluate(float(grid[best_index]))
    refined_best, refined_roots = evaluate(optimum_log_age)
    if refined_best is not None and (best is None or refined_best.delta < best.delta):
        best, max_roots = refined_best, refined_roots
    if best is None:
        raise FitError(f"age refinement failed for alpha_inj={alpha_inj:g}")
    diagnostics = {
        "age_grid_boundary": hit_grid_bound,
        "age_evaluations": len(cache),
        "maximum_roots_in_power_equation": max_roots,
    }
    return best, diagnostics


def _spectrum_records(
    model: KDAForwardModel,
    jet_power_w: float,
    age_myr: float,
    alpha_inj: float,
) -> tuple[list[dict[str, float | None]], float | None]:
    records: list[dict[str, float | None]] = []
    chi2 = 0.0
    weighted_count = 0
    for point in model.observation.spectrum:
        predicted = model.radio_power(point.frequency_hz, jet_power_w, age_myr * MYR_S, alpha_inj)
        if not math.isfinite(predicted) or predicted <= 0:
            raise FitError("fitted spectrum must contain only finite, positive powers")
        log_residual = math.log10(predicted) - math.log10(point.power_w_hz_sr)
        contribution: float | None = None
        if point.fractional_error is not None:
            sigma_log = _fractional_error_sigma_ln(point.fractional_error) / math.log(10.0)
            if sigma_log <= 0.0:
                raise FitError("fractional errors are too small to represent log-power weights")
            try:
                contribution = (log_residual / sigma_log) ** 2
            except OverflowError as exc:
                raise FitError("fitted spectrum residual exceeds floating-point range") from exc
            if not math.isfinite(contribution):
                raise FitError("fitted spectrum residual exceeds floating-point range")
            chi2 += contribution
            weighted_count += 1
            if not math.isfinite(chi2):
                raise FitError("fitted spectrum chi-square exceeds floating-point range")
        records.append(
            {
                "frequency_mhz_rest": point.frequency_mhz,
                "observed_power_w_hz_sr": point.power_w_hz_sr,
                "predicted_power_w_hz_sr": predicted,
                "log10_residual_model_over_data": log_residual,
                "fractional_error": point.fractional_error,
                "chi2_contribution_log_power": contribution,
            }
        )
    return records, chi2 if weighted_count == len(model.observation.spectrum) else None



def fit_classic(model: KDAForwardModel, bounds: FitBounds | None = None) -> FitResult:
    """Run the published three-stage, DYNAGE-like inverse procedure."""

    bounds = bounds or FitBounds()
    alpha_grid = np.linspace(bounds.alpha_inj[0], bounds.alpha_inj[1], bounds.alpha_grid_size)
    alpha_cache: dict[float, tuple[IntersectionSolution | None, dict[str, Any]]] = {}

    def evaluate(alpha: float) -> tuple[IntersectionSolution | None, dict[str, Any]]:
        key = round(float(alpha), 9)
        if key not in alpha_cache:
            try:
                alpha_cache[key] = _age_solution_for_alpha(model, float(alpha), bounds)
            except FitError:
                alpha_cache[key] = (None, {})
        return alpha_cache[key]

    energy_values: list[float] = []
    for alpha in alpha_grid:
        solution, _ = evaluate(float(alpha))
        energy_values.append(
            solution.consensus_log10_q + math.log10(solution.age_myr) + math.log10(MYR_S)
            if solution is not None else math.inf
        )
    if not np.any(np.isfinite(energy_values)):
        raise FitError("no classic DYNAGE solution exists inside the supplied bounds")
    best_index = int(np.nanargmin(energy_values))
    left_index = max(0, best_index - 1)
    right_index = min(len(alpha_grid) - 1, best_index + 1)
    hit_alpha_bound = best_index in (0, len(alpha_grid) - 1)

    def log_energy(alpha: float) -> float:
        solution, _ = evaluate(alpha)
        if solution is None:
            return 1.0e9
        return solution.consensus_log10_q + math.log10(solution.age_myr) + math.log10(MYR_S)

    if left_index != right_index:
        refined = minimize_scalar(
            log_energy,
            bounds=(float(alpha_grid[left_index]), float(alpha_grid[right_index])),
            method="bounded",
            options={"xatol": 2.0e-5, "maxiter": 60},
        )
        best_alpha = float(refined.x) if refined.success else float(alpha_grid[best_index])
    else:
        best_alpha = float(alpha_grid[best_index])
    solution, age_diagnostics = evaluate(float(alpha_grid[best_index]))
    refined_solution, refined_diagnostics = evaluate(best_alpha)
    if refined_solution is not None and (
        solution is None or log_energy(best_alpha) < log_energy(float(alpha_grid[best_index]))
    ):
        solution, age_diagnostics = refined_solution, refined_diagnostics
    if solution is None:
        raise FitError("classic DYNAGE alpha refinement failed")

    spectrum, chi2 = _spectrum_records(
        model, solution.jet_power_w, solution.age_myr, solution.alpha_inj
    )
    derived = model.derived_quantities(
        solution.jet_power_w, solution.age_myr * MYR_S, solution.alpha_inj
    )
    quadrature_errors = [
        model.quadrature_relative_error(
            point.frequency_hz,
            solution.jet_power_w,
            solution.age_myr * MYR_S,
            solution.alpha_inj,
        )
        for point in model.observation.spectrum
    ]
    warnings = [
        "Classic DYNAGE Delta is unweighted; minimum Q*t is a physical heuristic, not a confidence criterion.",
        "rho0 is conditional on the fixed core radius; the observations constrain rho0*a0^beta.",
    ]
    if hit_alpha_bound:
        warnings.append("The minimum-energy injection index is on the alpha search boundary.")
    if age_diagnostics.get("age_grid_boundary"):
        warnings.append("The Delta minimum is on the age search boundary.")
    if age_diagnostics.get("maximum_roots_in_power_equation", 1) > 1:
        warnings.append(
            "At least one luminosity equation had multiple jet-power roots; "
            "the lower-power branch was used."
        )
    q_lower, q_upper = bounds.log10_jet_power_w
    q_span = q_upper - q_lower
    q_near_boundary = any(
        (log_q - q_lower) / q_span < 2.0e-3
        or (q_upper - log_q) / q_span < 2.0e-3
        for log_q in solution.log10_q_by_frequency
    )
    if q_near_boundary:
        warnings.append(
            "At least one classic luminosity root is on or very near a jet-power bound."
        )
    if chi2 is not None and len(model.observation.spectrum) > 3:
        reduced_chi2 = chi2 / (len(model.observation.spectrum) - 3)
        if reduced_chi2 > 4.0:
            warnings.append(
                "The consensus spectrum is a poor fit relative to the supplied measurement errors "
                "(log-power reduced chi-square > 4)."
            )
    if max(quadrature_errors, default=0.0) > 5.0e-3:
        warnings.append("Quadrature convergence is worse than 0.5%; increase model.integration_order.")

    scan = []
    for alpha in alpha_grid:
        candidate, _ = evaluate(float(alpha))
        if candidate is not None:
            scan.append(
                {
                    "alpha_inj": candidate.alpha_inj,
                    "age_myr": candidate.age_myr,
                    "log10_jet_power_w": candidate.consensus_log10_q,
                        "log10_rho0_kg_m3": candidate.consensus_log10_rho0,
                    "delta": candidate.delta,
                    "log10_q_times_t_j": candidate.consensus_log10_q + math.log10(candidate.age_myr) + math.log10(MYR_S),
                }
            )
    diagnostics: dict[str, Any] = {
        "delta": solution.delta,
        "power_root_solver": ("checked affine anchored-field identity with general root-scan fallback" if type(model) is KDAForwardModel and model.assumptions.equipartition_field_ug is not None else "general bracketed multi-root scan"),
        "log10_q_by_frequency": list(solution.log10_q_by_frequency),
        "log10_rho0_by_frequency": list(solution.log10_rho0_by_frequency),
        "aggregation": "arithmetic mean in log10(Q), log10(rho0)",
        "alpha_grid_boundary": hit_alpha_bound,
        "alpha_evaluations": len(alpha_cache),
        "age_search": age_diagnostics,
        "jet_power_root_near_boundary": q_near_boundary,
        "quadrature_relative_error_order_vs_double": quadrature_errors,
        "classic_alpha_scan": scan,
        "weighted_log_power_chi2_at_consensus": chi2,
    }
    return FitResult(
        source_name=model.observation.name,
        method="classic_dynage",
        success=True,
        age_myr=solution.age_myr,
        alpha_inj=solution.alpha_inj,
        jet_power_w=solution.jet_power_w,
        rho0_kg_m3=solution.rho0_kg_m3,
        objective=solution.delta,
        objective_name="mean_pairwise_Delta_logQ_logrho",
        derived=derived,
        spectrum=spectrum,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def fit_joint(model: KDAForwardModel, bounds: FitBounds | None = None) -> FitResult:
    """Fit all radio powers simultaneously with a log-power likelihood.

    Errors are treated as independent lognormal errors, with each supplied
    fractional error interpreted as a coefficient of variation.  When a point
    has no reported error it receives unit log10-power weight, and the result
    is explicitly marked as an uncalibrated least-squares fit.
    """

    bounds = bounds or FitBounds()
    points = model.observation.spectrum
    frequencies = np.asarray([point.frequency_hz for point in points])
    observed_log = np.log10([point.power_w_hz_sr for point in points])
    has_all_errors = all(point.fractional_error is not None for point in points)
    sigma_log = np.asarray(
        [
            _fractional_error_sigma_ln(point.fractional_error) / math.log(10.0)
            if point.fractional_error is not None
            else 1.0
            for point in points
        ]
    )
    if np.any(~np.isfinite(sigma_log)) or np.any(sigma_log <= 0):
        raise FitError("fractional errors are too small to represent log-power weights")
    lower = np.asarray(
        [math.log10(bounds.age_myr[0]), bounds.alpha_inj[0], bounds.log10_jet_power_w[0]]
    )
    upper = np.asarray(
        [math.log10(bounds.age_myr[1]), bounds.alpha_inj[1], bounds.log10_jet_power_w[1]]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_age, alpha, log_q = (float(value) for value in parameters)
        try:
            predicted = model.spectrum(frequencies, 10.0**log_q, 10.0**log_age * MYR_S, alpha)
        except (ModelDomainError, ValueError, OverflowError, FloatingPointError):
            return np.full(len(points), 1.0e6)
        if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0):
            return np.full(len(points), 1.0e6)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weighted = (np.log10(predicted) - observed_log) / sigma_log
        return weighted if np.all(np.isfinite(weighted)) else np.full(len(points), 1.0e6)

    midpoint = 0.5 * (lower + upper)
    starts = [midpoint]
    for age_fraction in (0.25, 0.5, 0.75):
        for alpha_fraction in (0.25, 0.75):
            start = midpoint.copy()
            start[0] = lower[0] + age_fraction * (upper[0] - lower[0])
            start[1] = lower[1] + alpha_fraction * (upper[1] - lower[1])
            starts.append(start)
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
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.success
        and np.all(np.isfinite(candidate.fun))
        and np.all(np.abs(candidate.fun) < 5.0e5)
    ]
    if not valid_candidates:
        raise FitError(
            "joint fit found no converged solution with a finite, positive model spectrum "
            "inside the supplied bounds"
        )
    best = min(
        valid_candidates,
        key=lambda candidate: float(np.dot(candidate.fun, candidate.fun)),
    )
    log_age, alpha, log_q = (float(value) for value in best.x)
    age_myr = 10.0**log_age
    jet_power = 10.0**log_q
    verification = model.spectrum(
        frequencies,
        jet_power,
        age_myr * MYR_S,
        alpha,
    )
    if np.any(~np.isfinite(verification)) or np.any(verification <= 0):
        raise FitError("joint optimizer returned an invalid model spectrum")
    state = model.dynamics_state(jet_power, age_myr * MYR_S)
    spectrum, chi2 = _spectrum_records(model, jet_power, age_myr, alpha)
    derived = model.derived_quantities(jet_power, age_myr * MYR_S, alpha)
    raw_chi2 = float(np.dot(best.fun, best.fun))
    dof = len(points) - 3
    if has_all_errors:
        scale = 1.0
        covariance_scaling = "known supplied lognormal sigmas; no reduced-chi2 rescaling"
    else:
        scale = raw_chi2 / dof if dof > 0 else 1.0
        covariance_scaling = "uncalibrated residual variance estimate (sum_squares/dof)"
    covariance_estimate = estimate_local_covariance(
        best.jac,
        ["log10_age_myr", "alpha_inj", "log10_jet_power_w"],
        scale=scale,
    )
    covariance = covariance_estimate.covariance
    correlation = covariance_estimate.correlation
    standard_errors = covariance_estimate.standard_errors
    information_condition = covariance_estimate.information_condition_number
    normalized_position = (best.x - lower) / (upper - lower)
    on_boundary = bool(np.any((normalized_position < 2.0e-3) | (normalized_position > 1.0 - 2.0e-3)))
    warnings = [
        "Joint-fit errors are conditional on the fixed KDA assumptions and independent-error model.",
        "rho0 is conditional on the fixed core radius; the observations constrain rho0*a0^beta.",
    ]
    if not has_all_errors:
        warnings.append("At least one spectrum point has no error; objective weights are not a calibrated likelihood.")
    if on_boundary:
        warnings.append("At least one joint-fit parameter is on or very near a supplied bound.")
    if dof <= 0:
        warnings.append("There are no residual degrees of freedom; goodness of fit cannot be assessed.")
    if covariance is None:
        warnings.append("The local joint-fit information matrix is rank deficient or numerically unresolved; covariance is unavailable.")
    elif information_condition is not None and information_condition > 1.0e6:
        warnings.append(
            "The local joint-fit information matrix is ill-conditioned; parameters are strongly "
            "degenerate and covariance errors are not reliable."
        )
    diagnostics: dict[str, Any] = {
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimizer_function_evaluations": int(best.nfev),
        "multi_start_count": len(starts),
        "failed_start_count": len(start_failures),
        "start_failure_examples": start_failures[:5],
        "chi2": raw_chi2 if has_all_errors else None,
        "degrees_of_freedom": dof,
        "reduced_chi2": raw_chi2 / dof if has_all_errors and dof > 0 else None,
        "covariance_parameter_order": ["log10_age_myr", "alpha_inj", "log10_jet_power_w"],
        "local_covariance": covariance,
        "local_correlation": correlation,
        "local_standard_errors": standard_errors,
        "information_matrix_condition_number": information_condition,
        "covariance_scaling": covariance_scaling,
        "covariance_unavailable_reason": covariance_estimate.reason,
        "parameter_near_boundary": on_boundary,
        "weighted_log_power_chi2_check": chi2,
    }
    return FitResult(
        source_name=model.observation.name,
        method="joint_log_likelihood",
        success=bool(best.success),
        age_myr=age_myr,
        alpha_inj=alpha,
        jet_power_w=jet_power,
        rho0_kg_m3=state.rho0_kg_m3,
        objective=raw_chi2,
        objective_name="chi2_log_power" if has_all_errors else "weighted_sum_squares_log_power",
        derived=derived,
        spectrum=spectrum,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def parametric_bootstrap(
    observation: Observation,
    model_factory: Callable[[Observation], KDAForwardModel],
    fit_function: Callable[[KDAForwardModel, FitBounds | None], FitResult],
    bounds: FitBounds,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Refit lognormal measurement-error realizations and summarize survivors.

    This is Monte Carlo propagation around the reported measurements, not a
    conventional bootstrap generated from the fitted model.
    """

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if any(point.fractional_error is None for point in observation.spectrum):
        raise FitError("bootstrap requires fractional_error for every spectrum point")
    rng = np.random.default_rng(seed)
    draws: list[tuple[float, float, float, float]] = []
    failures: list[str] = []
    boundary_hits = 0
    for index in range(samples):
        try:
            varied_points = []
            for point in observation.spectrum:
                assert point.fractional_error is not None
                sigma_ln = _fractional_error_sigma_ln(point.fractional_error)
                power = math.exp(math.log(point.power_w_hz_sr) + float(rng.normal(0.0, sigma_ln)))
                varied_points.append(
                    type(point)(point.frequency_mhz, power, point.fractional_error)
                )
            varied_geometry = observation.geometry
            if (
                observation.geometry.size_error_kpc is not None
                or observation.geometry.axial_ratio_error is not None
            ):
                # The schema exposes independent errors.  If size and width were
                # measured jointly, callers should instead pre-generate correlated
                # realizations; assuming independence here is recorded below.
                for _ in range(100):
                    size = (
                        float(
                            rng.normal(
                                observation.geometry.projected_size_kpc,
                                observation.geometry.size_error_kpc,
                            )
                        )
                        if observation.geometry.size_error_kpc is not None
                        else observation.geometry.projected_size_kpc
                    )
                    axial_ratio = (
                        float(
                            rng.normal(
                                observation.geometry.axial_ratio,
                                observation.geometry.axial_ratio_error,
                            )
                        )
                        if observation.geometry.axial_ratio_error is not None
                        else observation.geometry.axial_ratio
                    )
                    minimum_ratio = 1.0 if observation.geometry.mode == "two_sided" else 0.5
                    if math.isfinite(size) and math.isfinite(axial_ratio) and size > 0 and axial_ratio > minimum_ratio:
                        varied_geometry = replace(
                            observation.geometry,
                            projected_size_kpc=size,
                            axial_ratio=axial_ratio,
                        )
                        break
                else:
                    raise FitError("could not draw a physical geometry")
            varied = Observation.from_points(
                observation.name,
                observation.redshift,
                varied_geometry,
                varied_points,
                radio_class=observation.radio_class,
            )
            fitted = fit_function(model_factory(varied), bounds)
            if not fitted.success:
                raise FitError("optimizer did not converge")
            if (
                not math.isfinite(fitted.age_myr)
                or not math.isfinite(fitted.alpha_inj)
                or not math.isfinite(fitted.jet_power_w)
                or not math.isfinite(fitted.rho0_kg_m3)
                or fitted.age_myr <= 0
                or not 0.0 < fitted.alpha_inj < 2.0
                or fitted.jet_power_w <= 0
                or fitted.rho0_kg_m3 <= 0
            ):
                raise FitError("fit returned non-finite or non-positive parameters")
            if not (
                bounds.age_myr[0] <= fitted.age_myr <= bounds.age_myr[1]
                and bounds.alpha_inj[0] <= fitted.alpha_inj <= bounds.alpha_inj[1]
                and bounds.log10_jet_power_w[0]
                <= math.log10(fitted.jet_power_w)
                <= bounds.log10_jet_power_w[1]
            ):
                raise FitError("fit returned parameters outside the supplied bounds")
            normalized = (
                (math.log10(fitted.age_myr) - math.log10(bounds.age_myr[0]))
                / (math.log10(bounds.age_myr[1]) - math.log10(bounds.age_myr[0])),
                (fitted.alpha_inj - bounds.alpha_inj[0])
                / (bounds.alpha_inj[1] - bounds.alpha_inj[0]),
                (math.log10(fitted.jet_power_w) - bounds.log10_jet_power_w[0])
                / (bounds.log10_jet_power_w[1] - bounds.log10_jet_power_w[0]),
            )
            if any(position < 2.0e-3 or position > 1.0 - 2.0e-3 for position in normalized):
                boundary_hits += 1
            draws.append(
                (
                    fitted.age_myr,
                    fitted.alpha_inj,
                    math.log10(fitted.jet_power_w),
                    math.log10(fitted.rho0_kg_m3),
                )
            )
        except (FitError, ModelDomainError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(f"sample {index}: {exc}")
            continue
    if not draws:
        raise FitError("every bootstrap realization failed")
    array = np.asarray(draws)
    names = ["age_myr", "alpha_inj", "log10_jet_power_w", "log10_rho0_kg_m3"]
    success_fraction = len(draws) / samples
    intervals_valid = len(draws) >= 10 and success_fraction >= 0.8
    intervals: dict[str, dict[str, float]] | None = None
    uncertainty_warnings: list[str] = []
    if intervals_valid:
        percentiles = np.percentile(array, [16.0, 50.0, 84.0], axis=0)
        intervals = {
            name: {
                "p16": float(percentiles[0, column]),
                "p50": float(percentiles[1, column]),
                "p84": float(percentiles[2, column]),
            }
            for column, name in enumerate(names)
        }
    else:
        uncertainty_warnings.append(
            "Percentile intervals were withheld: require at least 10 successful realizations "
            "and an 80% success fraction."
        )
    boundary_fraction = boundary_hits / len(draws)
    if boundary_fraction > 0.1:
        uncertainty_warnings.append(
            "More than 10% of successful realizations lie near a fit bound; intervals may be truncated."
        )
    return {
        "method": "lognormal Monte Carlo measurement-error propagation",
        "requested_samples": samples,
        "successful_samples": len(draws),
        "failed_samples": len(failures),
        "success_fraction": success_fraction,
        "seed": seed,
        "percentile_intervals_valid": intervals_valid,
        "percentile_intervals": intervals,
        "sample_summaries": ({name: {"mean": float(np.mean(array[:, column])), "median": float(np.median(array[:, column])),
                                      "standard_deviation": float(np.std(array[:, column], ddof=1))}
                              for column, name in enumerate(names)} if intervals_valid else None),
        "sample_parameter_order": names,
        "successful_draws": array.tolist(),
        "successful_samples_near_boundary": boundary_hits,
        "boundary_fraction": boundary_fraction,
        "warnings": uncertainty_warnings,
        "failure_examples": failures[:10],
        "scope": "reported spectrum/geometry measurement errors only; fixed-model systematics are excluded",
        "geometry_error_correlation": (
            "independent normal draws"
            if observation.geometry.size_error_kpc is not None
            or observation.geometry.axial_ratio_error is not None
            else "no geometry errors supplied"
        ),
    }
