r"""KDA-EXT forward modelling for switched-off FR II radio lobes.

This module deliberately lives beside, rather than inside, :mod:`model`.
The active :class:`~fr2dynage.model.KDAForwardModel` remains unchanged and
therefore keeps its published DYNAGE regression behaviour.

The remnant dynamics follow the fast-internal-sound-speed, ``Model B`` limit
used by Kuligowska (2017), based on Kaiser & Cotter (2002).  A source follows
the active KA/KDA solution until the supply of particles at the lobe ends at
``active_duration_s``.  Thereafter

.. math::

   L(t) = L(t_on) (t/t_on)^m,
   \quad
   p_c(t) = p_c(t_on) (t/t_on)^{-n},

where

.. math::

   m = {2(\Gamma_c+1) \over
        \Gamma_c(7+3\Gamma_c-2\beta)}, \qquad
   n = {6(\Gamma_c+1) \over 7+3\Gamma_c-2\beta}.

The transition is continuous in length and pressure but instantaneous in
their derivatives.  This is an explicit limiting approximation, not a model
of pressure-balanced or buoyantly rising late-time relics.

The electron calculation is not obtained by simply truncating the active KDA
integral.  For every injection cohort it analytically integrates synchrotron
and CMB losses across the active and remnant pressure histories.  This makes
the ``active_duration_s == total_age_s`` limit numerically equivalent to the
active KDA implementation.

References
----------
Kaiser & Cotter (2002), MNRAS 336, 649.
Kuligowska (2017), A&A 598, A93, doi:10.1051/0004-6361/201629033.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from .constants import (
    C_LIGHT,
    E_CHARGE,
    KPC_M,
    M_ELECTRON,
    MU_0,
    MYR_S,
    RADIATION_CONSTANT,
    SIGMA_T,
)
from .model import (
    KDAForwardModel,
    ModelAssumptions,
    ModelDomainError,
    Observation,
    _is_finite_real,
    _legendre_rule,
    electron_energy_integral,
)
from .numerics import (
    estimate_local_covariance,
    fractional_error_sigma_ln as _fractional_error_sigma_ln,
)

if TYPE_CHECKING:
    from .fitting import FitBounds, FitResult


def _validate_times(total_age_s: float, active_duration_s: float) -> None:
    if (
        not _is_finite_real(total_age_s)
        or not _is_finite_real(active_duration_s)
        or total_age_s <= 0.0
        or active_duration_s <= 0.0
    ):
        raise ModelDomainError(
            "total age and active duration must be finite and positive"
        )
    if active_duration_s > total_age_s:
        raise ModelDomainError("active duration cannot exceed total age")


def _power_integral(ratio: np.ndarray | float, exponent: float) -> np.ndarray:
    """Return integral_1^ratio u**(exponent-1) du stably."""

    values = np.asarray(ratio, dtype=float)
    log_ratio = np.log(values)
    if abs(exponent) < 1.0e-10:
        return log_ratio
    return np.expm1(exponent * log_ratio) / exponent


@dataclass(frozen=True)
class RemnantDynamicsState:
    """Dynamical state at observation time for one switched-off FR II lobe."""

    total_age_s: float
    active_duration_s: float
    remnant_duration_s: float
    jet_power_w: float
    density_parameter_si: float
    rho0_kg_m3: float
    lobe_length_m: float
    lobe_length_at_shutdown_m: float
    lobe_volume_m3: float
    pressure_pa: float
    pressure_at_shutdown_pa: float
    pressure_ratio: float
    c1: float
    c3: float
    post_shutdown_length_exponent: float
    post_shutdown_pressure_exponent: float

    @property
    def age_s(self) -> float:
        """Compatibility alias for code that labels total age simply ``age``."""

        return self.total_age_s

    @property
    def remnant_fraction(self) -> float:
        return self.remnant_duration_s / self.total_age_s


class KDAExtForwardModel:
    """KDA-EXT forward model for an active-then-remnant FR II component.

    ``active_duration_s`` is the time at which fresh particles cease arriving
    at the lobe.  It is therefore not automatically the central-engine
    switch-off time; a jet propagation delay must be treated separately when
    reconstructing an AGN duty cycle.
    """

    equation_set = (
        "KDA97 radiation + DYNAGE2007 geometry/pressure + "
        "KDA-EXT2017/Kaiser-Cotter2002 Model-B post-shutdown dynamics"
    )
    post_shutdown_dynamics = "kdaext2017-kc02-model-b"

    def __init__(
        self,
        observation: Observation,
        assumptions: ModelAssumptions | None = None,
        active_fraction: float | None = None,
    ):
        if active_fraction is not None and (
            not _is_finite_real(active_fraction)
            or not 0.0 < active_fraction <= 1.0
        ):
            raise ModelDomainError(
                "active_fraction must lie in (0, 1] when supplied"
            )
        self.observation = observation
        self.assumptions = assumptions or ModelAssumptions()
        if self.assumptions.equipartition_field_ug is not None:
            raise ModelDomainError(
                "the active snapshot equipartition-field anchor is not supported for remnant dynamics"
            )
        self.active_fraction = (
            float(active_fraction)
            if active_fraction is not None
            else None
        )
        self._active = KDAForwardModel(observation, self.assumptions)

    def _resolve_active_duration(
        self,
        total_age_s: float,
        active_duration_s: float | None,
    ) -> float:
        if active_duration_s is None:
            if self.active_fraction is None:
                raise ModelDomainError(
                    "active_duration_s is required when the model was "
                    "constructed without active_fraction"
                )
            active_duration_s = total_age_s * self.active_fraction
        _validate_times(total_age_s, active_duration_s)
        return float(active_duration_s)

    def _resolve_spectral_parameters(
        self,
        total_age_s: float,
        active_duration_s: float | None,
        alpha_inj: float | None,
    ) -> tuple[float, float]:
        # With a fixed constructor-level active fraction, preserve the active
        # KDA call shape ``(..., age_s, alpha_inj)`` as a convenience.
        if alpha_inj is None:
            if active_duration_s is None or self.active_fraction is None:
                raise ModelDomainError(
                    "alpha_inj and active duration are required"
                )
            alpha_inj = float(active_duration_s)
            active_duration_s = None
        resolved_duration = self._resolve_active_duration(
            total_age_s,
            active_duration_s,
        )
        return resolved_duration, float(alpha_inj)

    @property
    def pressure_ratio(self) -> float:
        return self._active.pressure_ratio

    @property
    def c1(self) -> float:
        return self._active.c1

    @property
    def c3(self) -> float:
        return self._active.c3

    @property
    def post_shutdown_length_exponent(self) -> float:
        a = self.assumptions
        denominator = a.gamma_c * (7.0 + 3.0 * a.gamma_c - 2.0 * a.beta)
        value = 2.0 * (a.gamma_c + 1.0) / denominator
        if value <= 0.0 or not math.isfinite(value):
            raise ModelDomainError("post-shutdown length exponent is undefined")
        return value

    @property
    def post_shutdown_pressure_exponent(self) -> float:
        a = self.assumptions
        denominator = 7.0 + 3.0 * a.gamma_c - 2.0 * a.beta
        value = 6.0 * (a.gamma_c + 1.0) / denominator
        if value <= 0.0 or not math.isfinite(value):
            raise ModelDomainError("post-shutdown pressure exponent is undefined")
        return value

    def density_parameter_from_length(
        self,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
    ) -> float:
        """Infer ``rho0*a0**beta`` from the observed final remnant length."""

        active_duration_s = self._resolve_active_duration(
            total_age_s,
            active_duration_s,
        )
        _validate_times(total_age_s, active_duration_s)
        if not _is_finite_real(jet_power_w) or jet_power_w <= 0.0:
            raise ModelDomainError("jet power must be finite and positive")
        beta = self.assumptions.beta
        length = self.observation.geometry.lobe_length_m
        expansion = (
            total_age_s / active_duration_s
        ) ** self.post_shutdown_length_exponent
        value = (
            jet_power_w
            * active_duration_s**3
            * self.c1 ** (5.0 - beta)
            * expansion ** (5.0 - beta)
            / length ** (5.0 - beta)
        )
        if value <= 0.0 or not math.isfinite(value):
            raise ModelDomainError(
                "density parameter is non-finite for the supplied remnant state"
            )
        return value

    def lobe_length(
        self,
        jet_power_w: float,
        density_parameter_si: float,
        total_age_s: float,
        active_duration_s: float | None = None,
    ) -> float:
        active_duration_s = self._resolve_active_duration(
            total_age_s,
            active_duration_s,
        )
        _validate_times(total_age_s, active_duration_s)
        length_at_shutdown = self._active.lobe_length(
            jet_power_w,
            density_parameter_si,
            active_duration_s,
        )
        return length_at_shutdown * (
            total_age_s / active_duration_s
        ) ** self.post_shutdown_length_exponent

    def cocoon_pressure(
        self,
        jet_power_w: float,
        density_parameter_si: float,
        total_age_s: float,
        active_duration_s: float | None = None,
    ) -> float:
        active_duration_s = self._resolve_active_duration(
            total_age_s,
            active_duration_s,
        )
        _validate_times(total_age_s, active_duration_s)
        pressure_at_shutdown = self._active.cocoon_pressure(
            jet_power_w,
            density_parameter_si,
            active_duration_s,
        )
        return pressure_at_shutdown * (
            total_age_s / active_duration_s
        ) ** (-self.post_shutdown_pressure_exponent)

    def dynamics_state(
        self,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
    ) -> RemnantDynamicsState:
        active_duration_s = self._resolve_active_duration(
            total_age_s,
            active_duration_s,
        )
        _validate_times(total_age_s, active_duration_s)
        density_parameter = self.density_parameter_from_length(
            jet_power_w,
            total_age_s,
            active_duration_s,
        )
        length_at_shutdown = self._active.lobe_length(
            jet_power_w,
            density_parameter,
            active_duration_s,
        )
        pressure_at_shutdown = self._active.cocoon_pressure(
            jet_power_w,
            density_parameter,
            active_duration_s,
        )
        length = self.lobe_length(
            jet_power_w,
            density_parameter,
            total_age_s,
            active_duration_s,
        )
        pressure = self.cocoon_pressure(
            jet_power_w,
            density_parameter,
            total_age_s,
            active_duration_s,
        )
        return RemnantDynamicsState(
            total_age_s=total_age_s,
            active_duration_s=active_duration_s,
            remnant_duration_s=total_age_s - active_duration_s,
            jet_power_w=jet_power_w,
            density_parameter_si=density_parameter,
            rho0_kg_m3=(
                density_parameter
                / self.assumptions.core_radius_m ** self.assumptions.beta
            ),
            lobe_length_m=length,
            lobe_length_at_shutdown_m=length_at_shutdown,
            lobe_volume_m3=self.c3 * length**3,
            pressure_pa=pressure,
            pressure_at_shutdown_pa=pressure_at_shutdown,
            pressure_ratio=self.pressure_ratio,
            c1=self.c1,
            c3=self.c3,
            post_shutdown_length_exponent=self.post_shutdown_length_exponent,
            post_shutdown_pressure_exponent=(
                self.post_shutdown_pressure_exponent
            ),
        )

    def _magnetic_ratio(self, p: float) -> float:
        ratio = self.assumptions.magnetic_ratio
        return (1.0 + p) / 4.0 if ratio is None else ratio

    def _cohort_kinematics(
        self,
        x: np.ndarray,
        frequency_hz: float,
        state: RemnantDynamicsState,
        p: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return exact two-phase loss quantities for injection cohorts.

        ``x`` is ``t_i/t_on``.  The loss equation is solved by using the
        adiabatic scale ``g=(V/V_i)**(1/3)``:

        ``1/gamma_i = 1/(gamma*g_now) - k*integral[(u_B+u_CMB)/g dt]``.

        Both power-law pieces of that integral are evaluated analytically.
        """

        a = self.assumptions
        delta = (4.0 + a.beta) / (5.0 - a.beta)
        a1 = delta / a.gamma_c
        active_volume_scale_exponent = a1 / 3.0
        off_length_exponent = self.post_shutdown_length_exponent
        off_ratio = state.total_age_s / state.active_duration_s
        magnetic_ratio = self._magnetic_ratio(p)

        with np.errstate(
            over="ignore",
            under="ignore",
            divide="ignore",
            invalid="ignore",
        ):
            pressure_i = state.pressure_at_shutdown_pa * x ** (-delta)
            u_e_i = pressure_i / (
                (a.gamma_c - 1.0)
                * (1.0 + a.k_prime)
                * (1.0 + magnetic_ratio)
            )
            u_b_i = (
                magnetic_ratio
                * pressure_i
                / ((a.gamma_c - 1.0) * (1.0 + magnetic_ratio))
            )

            active_ratio = 1.0 / x
            g_at_shutdown = active_ratio**active_volume_scale_exponent
            g_now = g_at_shutdown * off_ratio**off_length_exponent
            u_b_now = u_b_i * g_now ** (-3.0 * a.gamma_b)
            field_t = np.sqrt(2.0 * MU_0 * u_b_now)
            larmor_hz = E_CHARGE * field_t / (
                2.0 * math.pi * M_ELECTRON
            )
            gamma_now = np.sqrt(frequency_hz / larmor_hz)
            u_cmb = (
                RADIATION_CONSTANT
                * a.cmb_temperature_k**4
                * (1.0 + self.observation.redshift) ** 4
            )

            injection_time = state.active_duration_s * x
            active_b_exponent = (
                1.0
                - active_volume_scale_exponent
                * (3.0 * a.gamma_b + 1.0)
            )
            active_cmb_exponent = 1.0 - active_volume_scale_exponent
            magnetic_loss_active = (
                u_b_i
                * injection_time
                * _power_integral(active_ratio, active_b_exponent)
            )
            cmb_loss_active = (
                u_cmb
                * injection_time
                * _power_integral(active_ratio, active_cmb_exponent)
            )

            off_b_exponent = (
                1.0 - off_length_exponent * (3.0 * a.gamma_b + 1.0)
            )
            off_cmb_exponent = 1.0 - off_length_exponent
            magnetic_loss_off = (
                u_b_i
                * g_at_shutdown ** (-(3.0 * a.gamma_b + 1.0))
                * state.active_duration_s
                * _power_integral(off_ratio, off_b_exponent)
            )
            cmb_loss_off = (
                u_cmb
                * g_at_shutdown**-1.0
                * state.active_duration_s
                * _power_integral(off_ratio, off_cmb_exponent)
            )
            integrated_loss = (
                magnetic_loss_active
                + cmb_loss_active
                + magnetic_loss_off
                + cmb_loss_off
            )
            eta = (
                4.0
                * SIGMA_T
                / (3.0 * M_ELECTRON * C_LIGHT)
                * gamma_now
                * g_now
                * integrated_loss
            )
            denominator = 1.0 - eta
            gamma_i = gamma_now * g_now / denominator

        return (
            pressure_i,
            u_e_i,
            u_b_now,
            gamma_now,
            denominator,
            gamma_i,
            g_now,
        )

    def _cohort_values(
        self,
        x: np.ndarray,
        frequency_hz: float,
        state: RemnantDynamicsState,
        p: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return present-day ``dP_nu/dt_i`` and its physical-domain mask."""

        a = self.assumptions
        (
            pressure_i,
            u_e_i,
            u_b_now,
            gamma_now,
            denominator,
            gamma_i,
            g_now,
        ) = self._cohort_kinematics(x, frequency_hz, state, p)
        with np.errstate(
            over="ignore",
            under="ignore",
            divide="ignore",
            invalid="ignore",
        ):
            normalization = electron_energy_integral(
                p,
                a.gamma_min,
                a.gamma_max,
            )
            n0_i = u_e_i / (
                M_ELECTRON * C_LIGHT**2 * normalization
            )
            n_gamma_now = (
                n0_i
                * gamma_i ** (2.0 - p)
                / gamma_now**2
                * g_now**-4.0
            )
            dvolume_now_dti = (
                (a.gamma_c - 1.0)
                * state.jet_power_w
                / pressure_i
                * self.pressure_ratio
                ** ((1.0 - a.gamma_c) / a.gamma_c)
                * g_now**3.0
            )
            dpower_dti = (
                SIGMA_T
                * C_LIGHT
                * u_b_now
                / (6.0 * math.pi * frequency_hz)
                * gamma_now**3
                * n_gamma_now
                * dvolume_now_dti
            )

        valid = (
            np.isfinite(dpower_dti)
            & np.isfinite(gamma_i)
            & np.isfinite(g_now)
            & (denominator > 0.0)
            & (gamma_i >= a.gamma_min)
            & (gamma_i <= a.gamma_max)
            & (dpower_dti >= 0.0)
        )
        return np.where(valid, dpower_dti, 0.0), valid

    def _valid_intervals(
        self,
        frequency_hz: float,
        state: RemnantDynamicsState,
        p: float,
    ) -> list[tuple[float, float]]:
        """Locate valid ``y=-log(t_i/t_on)`` integration intervals."""

        ymax = self.assumptions.max_log_injection_span
        sample = np.linspace(0.0, ymax, 257)

        def kinematics(y: float) -> tuple[float, float]:
            values = self._cohort_kinematics(
                np.asarray([math.exp(-y)]),
                frequency_hz,
                state,
                p,
            )
            return float(values[4][0]), float(values[5][0])

        def denominator(y: float) -> float:
            return kinematics(y)[0]

        def gamma_i(y: float) -> float:
            return kinematics(y)[1]

        def roots_on_grid(
            function,
            grid: np.ndarray,
            sampled_values: np.ndarray | None = None,
        ) -> list[float]:
            values = (
                [float(value) for value in sampled_values]
                if sampled_values is not None
                else [float(function(float(value))) for value in grid]
            )
            roots: list[float] = []
            for index in range(len(grid) - 1):
                left = float(grid[index])
                right = float(grid[index + 1])
                f_left = values[index]
                f_right = values[index + 1]
                if math.isnan(f_left) or math.isnan(f_right):
                    continue
                if f_left == 0.0:
                    roots.append(left)
                elif (f_left < 0.0) != (f_right < 0.0):
                    try:
                        roots.append(
                            float(
                                brentq(
                                    function,
                                    left,
                                    right,
                                    xtol=1.0e-12,
                                    rtol=1.0e-12,
                                    maxiter=100,
                                )
                            )
                        )
                    except ValueError:
                        continue
            if not math.isnan(values[-1]) and values[-1] == 0.0:
                roots.append(float(grid[-1]))
            return roots

        tolerance = 1.0e-10 * (1.0 + ymax)

        def unique(values: list[float]) -> list[float]:
            ordered = sorted(
                value for value in values if 0.0 <= value <= ymax
            )
            result: list[float] = []
            for value in ordered:
                if not result or abs(value - result[-1]) > tolerance:
                    result.append(value)
            return result

        sampled_kinematics = self._cohort_kinematics(
            np.exp(-sample),
            frequency_hz,
            state,
            p,
        )
        denominator_roots = unique(
            roots_on_grid(
                denominator,
                sample,
                sampled_kinematics[4],
            )
        )
        domain_edges = unique([0.0, ymax, *denominator_roots])
        intervals: list[tuple[float, float]] = []

        def predicate(y: float) -> bool:
            _, mask = self._cohort_values(
                np.asarray([math.exp(-y)]),
                frequency_hz,
                state,
                p,
            )
            return bool(mask[0])

        for domain_left, domain_right in zip(
            domain_edges[:-1],
            domain_edges[1:],
            strict=True,
        ):
            if domain_right - domain_left <= tolerance:
                continue
            if denominator(0.5 * (domain_left + domain_right)) <= 0.0:
                continue
            inset = max(
                1.0e-12,
                1.0e-10 * (domain_right - domain_left),
            )
            left_is_root = any(
                abs(domain_left - root) <= tolerance
                for root in denominator_roots
            )
            right_is_root = any(
                abs(domain_right - root) <= tolerance
                for root in denominator_roots
            )
            left = domain_left + inset if left_is_root else domain_left
            right = domain_right - inset if right_is_root else domain_right
            if right <= left:
                continue

            gamma_grid = np.linspace(left, right, 257)
            gamma_values = self._cohort_kinematics(
                np.exp(-gamma_grid),
                frequency_hz,
                state,
                p,
            )[5]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_gamma = np.where(
                    gamma_values > 0.0,
                    np.log(gamma_values),
                    np.nan,
                )
            extrema: list[float] = []
            for index in range(1, len(gamma_grid) - 1):
                previous = log_gamma[index - 1]
                current = log_gamma[index]
                following = log_gamma[index + 1]
                if not all(
                    math.isfinite(value)
                    for value in (previous, current, following)
                ):
                    continue
                is_minimum = current <= previous and current <= following
                is_maximum = current >= previous and current >= following
                if not (is_minimum or is_maximum):
                    continue

                def objective(y: float) -> float:
                    value = gamma_i(y)
                    if value <= 0.0 or not math.isfinite(value):
                        return math.inf
                    log_value = math.log(value)
                    return log_value if is_minimum else -log_value

                refined = minimize_scalar(
                    objective,
                    bounds=(
                        float(gamma_grid[index - 1]),
                        float(gamma_grid[index + 1]),
                    ),
                    method="bounded",
                    options={"xatol": 1.0e-11, "maxiter": 80},
                )
                extrema.append(
                    float(refined.x)
                    if refined.success
                    else float(gamma_grid[index])
                )

            monotonic_edges = unique([left, right, *extrema])
            cutoff_roots: list[float] = []
            for cutoff in (
                self.assumptions.gamma_min,
                self.assumptions.gamma_max,
            ):
                for branch_left, branch_right in zip(
                    monotonic_edges[:-1],
                    monotonic_edges[1:],
                    strict=True,
                ):
                    f_left = gamma_i(branch_left) - cutoff
                    f_right = gamma_i(branch_right) - cutoff
                    if not math.isfinite(f_left) or not math.isfinite(
                        f_right
                    ):
                        continue
                    if f_left == 0.0:
                        cutoff_roots.append(branch_left)
                    elif (f_left < 0.0) != (f_right < 0.0):
                        cutoff_roots.append(
                            float(
                                brentq(
                                    lambda y: gamma_i(y) - cutoff,
                                    branch_left,
                                    branch_right,
                                    xtol=1.0e-12,
                                    rtol=1.0e-12,
                                    maxiter=100,
                                )
                            )
                        )
                if gamma_i(monotonic_edges[-1]) == cutoff:
                    cutoff_roots.append(monotonic_edges[-1])

            validity_edges = unique(
                [domain_left, domain_right, *cutoff_roots]
            )
            for valid_left, valid_right in zip(
                validity_edges[:-1],
                validity_edges[1:],
                strict=True,
            ):
                if valid_right - valid_left <= tolerance:
                    continue
                if predicate(0.5 * (valid_left + valid_right)):
                    if (
                        intervals
                        and abs(valid_left - intervals[-1][1])
                        <= tolerance
                    ):
                        intervals[-1] = (
                            intervals[-1][0],
                            valid_right,
                        )
                    else:
                        intervals.append((valid_left, valid_right))
        return intervals

    def radio_power_one_lobe(
        self,
        frequency_hz: float,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
        alpha_inj: float | None = None,
    ) -> float:
        """Present monochromatic power of one remnant lobe.

        The return convention is W Hz^-1 sr^-1, matching KDA and DYNAGE.
        """

        active_duration_s, alpha_inj = (
            self._resolve_spectral_parameters(
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )
        _validate_times(total_age_s, active_duration_s)
        if (
            not _is_finite_real(frequency_hz)
            or not _is_finite_real(jet_power_w)
            or frequency_hz <= 0.0
            or jet_power_w <= 0.0
        ):
            raise ModelDomainError(
                "frequency and jet power must be finite and positive"
            )
        if (
            not _is_finite_real(alpha_inj)
            or not 0.0 < alpha_inj < 2.0
        ):
            raise ModelDomainError(
                "alpha_inj must be finite and lie between 0 and 2"
            )
        p = 2.0 * alpha_inj + 1.0
        state = self.dynamics_state(
            jet_power_w,
            total_age_s,
            active_duration_s,
        )
        intervals = self._valid_intervals(frequency_hz, state, p)
        if not intervals:
            return 0.0
        nodes, weights = _legendre_rule(
            self.assumptions.integration_order
        )
        total = 0.0
        for lower, upper in intervals:
            half = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            y = midpoint + half * nodes
            x = np.exp(-y)
            dpower_dti, _ = self._cohort_values(
                x,
                frequency_hz,
                state,
                p,
            )
            total += half * float(
                np.dot(
                    weights,
                    dpower_dti * state.active_duration_s * x,
                )
            )
        if not math.isfinite(total) or total < 0.0:
            raise ModelDomainError(
                "remnant radio-power integration returned a non-finite value"
            )
        return total

    def radio_power(
        self,
        frequency_hz: float,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
        alpha_inj: float | None = None,
    ) -> float:
        """Power of the configured one-lobe or two-lobe component."""

        active_duration_s, alpha_inj = (
            self._resolve_spectral_parameters(
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )
        return (
            self.observation.geometry.emitting_lobes
            * self.radio_power_one_lobe(
                frequency_hz,
                jet_power_w,
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )

    def spectrum(
        self,
        frequencies_hz: Iterable[float],
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
        alpha_inj: float | None = None,
    ) -> np.ndarray:
        active_duration_s, alpha_inj = (
            self._resolve_spectral_parameters(
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )
        return np.asarray(
            [
                self.radio_power(
                    float(frequency_hz),
                    jet_power_w,
                    total_age_s,
                    active_duration_s,
                    alpha_inj,
                )
                for frequency_hz in frequencies_hz
            ],
            dtype=float,
        )

    def derived_quantities(
        self,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
        alpha_inj: float | None = None,
    ) -> dict[str, float]:
        active_duration_s, alpha_inj = (
            self._resolve_spectral_parameters(
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )
        if (
            not _is_finite_real(alpha_inj)
            or not 0.0 < alpha_inj < 2.0
        ):
            raise ModelDomainError(
                "alpha_inj must be finite and lie between 0 and 2"
            )
        state = self.dynamics_state(
            jet_power_w,
            total_age_s,
            active_duration_s,
        )
        a = self.assumptions
        p = 2.0 * alpha_inj + 1.0
        magnetic_ratio = self._magnetic_ratio(p)
        cocoon_energy_density = state.pressure_pa / (
            a.gamma_c - 1.0
        )
        magnetic_energy_at_shutdown = (
            magnetic_ratio
            * state.pressure_at_shutdown_pa
            / ((a.gamma_c - 1.0) * (1.0 + magnetic_ratio))
        )
        off_expansion = (
            state.total_age_s / state.active_duration_s
        ) ** self.post_shutdown_length_exponent
        youngest_cohort_u_b = magnetic_energy_at_shutdown * (
            off_expansion ** (-3.0 * a.gamma_b)
        )
        field_t = math.sqrt(2.0 * MU_0 * youngest_cohort_u_b)
        rho_head = (
            state.density_parameter_si
            * state.lobe_length_m ** (-a.beta)
        )
        total_volume = (
            self.observation.geometry.emitting_lobes
            * state.lobe_volume_m3
        )
        stored_energy = cocoon_energy_density * total_volume
        one_jet_injected_energy = jet_power_w * active_duration_s
        injected_energy = (
            self.observation.geometry.emitting_lobes
            * one_jet_injected_energy
        )
        mean_speed = state.lobe_length_m / total_age_s
        if state.remnant_duration_s > 0.0:
            instantaneous_speed = (
                self.post_shutdown_length_exponent
                * state.lobe_length_m
                / total_age_s
            )
        else:
            instantaneous_speed = (
                3.0
                / (5.0 - a.beta)
                * state.lobe_length_m
                / total_age_s
            )
        return {
            "total_age_myr": total_age_s / MYR_S,
            "age_myr": total_age_s / MYR_S,
            "active_duration_myr": active_duration_s / MYR_S,
            "remnant_duration_myr": (
                state.remnant_duration_s / MYR_S
            ),
            "remnant_fraction": state.remnant_fraction,
            "active_duty_fraction": (
                active_duration_s / total_age_s
            ),
            "alpha_inj": alpha_inj,
            "electron_index_p": p,
            "jet_power_w_per_jet_during_active_phase": jet_power_w,
            "density_parameter_rho0_a0_beta_si": (
                state.density_parameter_si
            ),
            "rho0_kg_m3_conditional_on_core_radius": state.rho0_kg_m3,
            "ambient_density_at_head_kg_m3": rho_head,
            "cocoon_pressure_pa": state.pressure_pa,
            "pressure_at_shutdown_pa": state.pressure_at_shutdown_pa,
            "cocoon_energy_density_j_m3": cocoon_energy_density,
            "youngest_surviving_cohort_magnetic_field_t": field_t,
            "lobe_length_kpc_deprojected": (
                state.lobe_length_m / KPC_M
            ),
            "lobe_length_at_shutdown_kpc": (
                state.lobe_length_at_shutdown_m / KPC_M
            ),
            "one_lobe_volume_m3": state.lobe_volume_m3,
            "modelled_total_volume_m3": total_volume,
            "stored_cocoon_energy_j": stored_energy,
            "one_jet_q_times_active_duration_j": (
                one_jet_injected_energy
            ),
            "injected_jet_energy_j": injected_energy,
            "injected_to_stored_energy_ratio": (
                injected_energy / stored_energy
            ),
            "mean_head_speed_over_c": mean_speed / C_LIGHT,
            "instantaneous_head_speed_over_c": (
                instantaneous_speed / C_LIGHT
            ),
            "pressure_ratio_phc": self.pressure_ratio,
            "post_shutdown_length_exponent": (
                self.post_shutdown_length_exponent
            ),
            "post_shutdown_pressure_exponent": (
                self.post_shutdown_pressure_exponent
            ),
            "c1": self.c1,
            "c3": self.c3,
        }

    def quadrature_relative_error(
        self,
        frequency_hz: float,
        jet_power_w: float,
        total_age_s: float,
        active_duration_s: float | None = None,
        alpha_inj: float | None = None,
        higher_order_factor: int = 2,
    ) -> float:
        active_duration_s, alpha_inj = (
            self._resolve_spectral_parameters(
                total_age_s,
                active_duration_s,
                alpha_inj,
            )
        )
        if (
            isinstance(higher_order_factor, bool)
            or not isinstance(higher_order_factor, int)
            or higher_order_factor < 2
        ):
            raise ModelDomainError(
                "higher_order_factor must be an integer of at least 2"
            )
        base = self.radio_power(
            frequency_hz,
            jet_power_w,
            total_age_s,
            active_duration_s,
            alpha_inj,
        )
        higher_assumptions = self.assumptions.with_integration_order(
            self.assumptions.integration_order * higher_order_factor
        )
        higher = KDAExtForwardModel(
            self.observation,
            higher_assumptions,
        ).radio_power(
            frequency_hz,
            jet_power_w,
            total_age_s,
            active_duration_s,
            alpha_inj,
        )
        if higher == 0.0:
            return 0.0 if base == 0.0 else math.inf
        return abs(base - higher) / higher


@dataclass(frozen=True)
class RemnantFitBounds:
    """Bounds for the four-parameter joint KDA-EXT inverse problem."""

    total_age_myr: tuple[float, float] = (0.2, 1000.0)
    remnant_fraction: tuple[float, float] = (0.001, 0.99)
    alpha_inj: tuple[float, float] = (0.45, 0.90)
    log10_jet_power_w: tuple[float, float] = (34.0, 43.0)

    def __post_init__(self) -> None:
        for label, pair in (
            ("total_age_myr", self.total_age_myr),
            ("remnant_fraction", self.remnant_fraction),
            ("alpha_inj", self.alpha_inj),
            ("log10_jet_power_w", self.log10_jet_power_w),
        ):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(
                    not _is_finite_real(value)
                    for value in pair
                )
                or pair[0] >= pair[1]
            ):
                raise ValueError(
                    f"{label} must contain two increasing finite values"
                )
        if self.total_age_myr[0] <= 0.0:
            raise ValueError("total-age lower bound must be positive")
        if not (
            0.0
            < self.remnant_fraction[0]
            < self.remnant_fraction[1]
            < 1.0
        ):
            raise ValueError(
                "remnant-fraction bounds must lie strictly inside (0, 1)"
            )
        if self.alpha_inj[0] <= 0.0 or self.alpha_inj[1] >= 2.0:
            raise ValueError(
                "alpha bounds must lie strictly inside (0, 2)"
            )
        if not _is_finite_real(self.total_age_myr[1] * MYR_S):
            raise ValueError("total-age bounds exceed the representable range in seconds")
        try:
            powers = [10.0**value for value in self.log10_jet_power_w]
        except OverflowError as exc:
            raise ValueError("jet-power bounds exceed the floating-point range") from exc
        if any(not _is_finite_real(value) or value <= 0.0 for value in powers):
            raise ValueError("jet-power bounds must represent finite positive powers")


@dataclass
class RemnantFitResult:
    """Result of a simultaneous log-power KDA-EXT fit."""

    source_name: str
    success: bool
    total_age_myr: float
    active_duration_myr: float
    remnant_duration_myr: float
    remnant_fraction: float
    alpha_inj: float
    jet_power_w: float
    rho0_kg_m3: float
    objective: float
    objective_name: str
    derived: dict[str, float]
    spectrum: list[dict[str, float | None]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def age_myr(self) -> float:
        """Backward-compatible alias: remnant ``age`` means total age."""

        return self.total_age_myr

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "activity_state": "remnant",
            "method": "joint_kdaext_log_likelihood",
            "success": self.success,
            "best_fit": {
                "age_myr": self.total_age_myr,
                "total_age_myr": self.total_age_myr,
                "active_duration_myr": self.active_duration_myr,
                "remnant_duration_myr": self.remnant_duration_myr,
                "remnant_fraction": self.remnant_fraction,
                "active_duty_fraction": (
                    self.active_duration_myr / self.total_age_myr
                ),
                "alpha_inj": self.alpha_inj,
                "electron_index_p": 2.0 * self.alpha_inj + 1.0,
                "jet_power_w_per_jet_during_active_phase": (
                    self.jet_power_w
                ),
                "rho0_kg_m3_conditional_on_core_radius": (
                    self.rho0_kg_m3
                ),
            },
            "objective": {
                "name": self.objective_name,
                "value": self.objective,
            },
            "derived": self.derived,
            "spectrum": self.spectrum,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }


class RemnantFitError(RuntimeError):
    """Raised when no physical remnant solution exists inside the bounds."""


def _logit(value: float) -> float:
    return math.log(value) - math.log1p(-value)


def _expit(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _fit_remnant_joint_model(
    model: KDAExtForwardModel,
    bounds: RemnantFitBounds | None = None,
) -> RemnantFitResult:
    """Fit total age, off fraction, injection index, and active jet power.

    Independent fractional errors are interpreted as lognormal coefficients
    of variation.  Without an error on every point, the objective remains a
    useful log-power least-squares diagnostic but is not a calibrated
    likelihood.  Four detections are the mathematical minimum and leave zero
    residual degrees of freedom; five or more widely spaced frequencies are
    strongly preferred.
    """

    bounds = bounds or RemnantFitBounds()
    points = model.observation.spectrum
    if len(points) < 4:
        raise RemnantFitError(
            "a remnant fit has four free parameters and requires at least "
            "four detected spectrum points"
        )
    frequencies = np.asarray(
        [point.frequency_hz for point in points],
        dtype=float,
    )
    observed_log = np.log10(
        [point.power_w_hz_sr for point in points]
    )
    has_all_errors = all(
        point.fractional_error is not None for point in points
    )
    sigma_log = np.asarray(
        [
            (
                _fractional_error_sigma_ln(point.fractional_error)
                / math.log(10.0)
            )
            if point.fractional_error is not None
            else 1.0
            for point in points
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(sigma_log)) or np.any(sigma_log <= 0.0):
        raise RemnantFitError("fractional errors are too small to represent log-power weights")

    lower = np.asarray(
        [
            math.log10(bounds.total_age_myr[0]),
            _logit(bounds.remnant_fraction[0]),
            bounds.alpha_inj[0],
            bounds.log10_jet_power_w[0],
        ]
    )
    upper = np.asarray(
        [
            math.log10(bounds.total_age_myr[1]),
            _logit(bounds.remnant_fraction[1]),
            bounds.alpha_inj[1],
            bounds.log10_jet_power_w[1],
        ]
    )

    def unpack(
        parameters: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        log_age, off_logit, alpha, log_q = (
            float(value) for value in parameters
        )
        total_age_myr = 10.0**log_age
        off_fraction = _expit(off_logit)
        active_duration_myr = total_age_myr * (1.0 - off_fraction)
        return (
            total_age_myr,
            active_duration_myr,
            off_fraction,
            alpha,
            10.0**log_q,
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            (
                total_age_myr,
                active_duration_myr,
                _,
                alpha,
                jet_power,
            ) = unpack(parameters)
            predicted = model.spectrum(
                frequencies,
                jet_power,
                total_age_myr * MYR_S,
                active_duration_myr * MYR_S,
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
    for age_fraction, off_fraction in (
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    ):
        start = midpoint.copy()
        start[0] = lower[0] + age_fraction * (
            upper[0] - lower[0]
        )
        start[1] = lower[1] + off_fraction * (
            upper[1] - lower[1]
        )
        starts.append(start)
    for alpha_fraction in (0.25, 0.75):
        start = midpoint.copy()
        start[2] = lower[2] + alpha_fraction * (
            upper[2] - lower[2]
        )
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
                max_nfev=1200,
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
        raise RemnantFitError(
            "joint remnant fit found no converged solution with a finite, "
            "positive model spectrum inside the supplied bounds"
        )
    best = min(
        valid_candidates,
        key=lambda candidate: float(
            np.dot(candidate.fun, candidate.fun)
        ),
    )
    (
        total_age_myr,
        active_duration_myr,
        off_fraction,
        alpha,
        jet_power,
    ) = unpack(best.x)
    remnant_duration_myr = total_age_myr - active_duration_myr
    predicted = model.spectrum(
        frequencies,
        jet_power,
        total_age_myr * MYR_S,
        active_duration_myr * MYR_S,
        alpha,
    )
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise RemnantFitError(
            "joint remnant optimizer returned an invalid spectrum"
        )
    state = model.dynamics_state(
        jet_power,
        total_age_myr * MYR_S,
        active_duration_myr * MYR_S,
    )
    derived = model.derived_quantities(
        jet_power,
        total_age_myr * MYR_S,
        active_duration_myr * MYR_S,
        alpha,
    )

    spectrum_records: list[dict[str, float | None]] = []
    for point, prediction in zip(points, predicted, strict=True):
        log_residual = math.log10(float(prediction)) - math.log10(point.power_w_hz_sr)
        contribution: float | None = None
        if point.fractional_error is not None:
            point_sigma = (
                _fractional_error_sigma_ln(point.fractional_error)
                / math.log(10.0)
            )
            try:
                contribution = (log_residual / point_sigma) ** 2
            except OverflowError as exc:
                raise RemnantFitError("fitted spectrum residual exceeds floating-point range") from exc
            if not math.isfinite(contribution):
                raise RemnantFitError("fitted spectrum residual exceeds floating-point range")
        spectrum_records.append(
            {
                "frequency_mhz_rest": point.frequency_mhz,
                "observed_power_w_hz_sr": point.power_w_hz_sr,
                "predicted_power_w_hz_sr": float(prediction),
                "log10_residual_model_over_data": log_residual,
                "fractional_error": point.fractional_error,
                "chi2_contribution_log_power": contribution,
            }
        )

    raw_objective = float(np.dot(best.fun, best.fun))
    degrees_of_freedom = len(points) - 4
    scale = 1.0 if has_all_errors or degrees_of_freedom <= 0 else raw_objective / degrees_of_freedom
    estimate = estimate_local_covariance(
        best.jac,
        ["log10_total_age_myr", "logit_remnant_fraction", "alpha_inj", "log10_jet_power_w"],
        scale=scale,
    )
    covariance = estimate.covariance
    correlation = estimate.correlation
    standard_errors = estimate.standard_errors
    information_condition = estimate.information_condition_number
    covariance_scaling = (
        "known supplied lognormal sigmas; no reduced-chi2 rescaling"
        if has_all_errors
        else "uncalibrated residual variance estimate (sum_squares/dof)"
    ) if covariance is not None else None

    normalized_position = (best.x - lower) / (upper - lower)
    on_boundary = bool(
        np.any(
            (normalized_position < 2.0e-3)
            | (normalized_position > 1.0 - 2.0e-3)
        )
    )
    log_frequency = np.log10(frequencies)
    linear_spectrum = np.polyval(
        np.polyfit(log_frequency, observed_log, 1),
        log_frequency,
    )
    maximum_curvature_residual_dex = float(
        np.max(np.abs(observed_log - linear_spectrum))
    )
    quadrature_errors = [
        model.quadrature_relative_error(
            point.frequency_hz,
            jet_power,
            total_age_myr * MYR_S,
            active_duration_myr * MYR_S,
            alpha,
        )
        for point in points
    ]

    warnings = [
        (
            "KDA-EXT Model B assumes an instantaneous transition to an "
            "overpressured coasting phase; pressure-balanced or buoyant "
            "late remnants are outside this model."
        ),
        (
            "The fitted shutdown time is cessation of fresh-particle supply "
            "at the lobe, not automatically the central-engine switch-off "
            "time."
        ),
        (
            "rho0 is conditional on the fixed core radius; the observations "
            "constrain rho0*a0^beta."
        ),
    ]
    if not has_all_errors:
        warnings.append(
            "At least one spectrum point has no error; objective weights "
            "are not a calibrated likelihood."
        )
    if degrees_of_freedom <= 0:
        warnings.append(
            "The fit has no residual degrees of freedom; goodness of fit "
            "and model selection cannot be assessed."
        )
    elif degrees_of_freedom == 1:
        warnings.append(
            "Only one residual degree of freedom is available; parameter "
            "uncertainties and model comparison are fragile."
        )
    if on_boundary:
        warnings.append(
            "At least one remnant-fit parameter is on or very near a bound."
        )
    if covariance is None:
        warnings.append(
            "The local covariance is unavailable: " + str(estimate.reason) + "."
        )
    elif information_condition is None or information_condition > 1.0e7:
        warnings.append(
            "The local information matrix is ill-conditioned; total age "
            "and shutdown fraction are strongly degenerate."
        )
    if maximum_curvature_residual_dex < 0.02:
        warnings.append(
            "The observed spectrum is nearly a power law (<0.02 dex maximum "
            "curvature); a remnant duration is unlikely to be uniquely "
            "identified without an external density or jet-power constraint."
        )
    frequency_span = float(np.max(frequencies) / np.min(frequencies))
    if frequency_span < 10.0:
        warnings.append(
            "The spectrum spans less than one decade; remnant-age and "
            "injection-index degeneracy may be severe."
        )
    if max(quadrature_errors, default=0.0) > 5.0e-3:
        warnings.append(
            "Quadrature convergence is worse than 0.5%; increase "
            "model.integration_order."
        )

    diagnostics: dict[str, Any] = {
        "post_shutdown_dynamics": model.post_shutdown_dynamics,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimizer_function_evaluations": int(best.nfev),
        "multi_start_count": len(starts),
        "failed_start_count": len(start_failures),
        "start_failure_examples": start_failures[:3],
        "parameterization": [
            "log10_total_age_myr",
            "logit_remnant_fraction",
            "alpha_inj",
            "log10_jet_power_w",
        ],
        "chi2": raw_objective if has_all_errors else None,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi2": (
            raw_objective / degrees_of_freedom
            if has_all_errors and degrees_of_freedom > 0
            else None
        ),
        "local_covariance": covariance,
        "local_correlation": correlation,
        "local_standard_errors": standard_errors,
        "information_matrix_condition_number": information_condition,
        "covariance_scaling": covariance_scaling,
        "covariance_unavailable_reason": estimate.reason,
        "parameter_near_boundary": on_boundary,
        "maximum_observed_power_law_residual_dex": (
            maximum_curvature_residual_dex
        ),
        "quadrature_relative_error_order_vs_double": quadrature_errors,
    }
    return RemnantFitResult(
        source_name=model.observation.name,
        success=bool(best.success),
        total_age_myr=total_age_myr,
        active_duration_myr=active_duration_myr,
        remnant_duration_myr=remnant_duration_myr,
        remnant_fraction=off_fraction,
        alpha_inj=alpha,
        jet_power_w=jet_power,
        rho0_kg_m3=state.rho0_kg_m3,
        objective=raw_objective,
        objective_name=(
            "chi2_log_power"
            if has_all_errors
            else "weighted_sum_squares_log_power"
        ),
        derived=derived,
        spectrum=spectrum_records,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def fit_remnant_joint(
    observation: Observation | KDAExtForwardModel,
    assumptions: ModelAssumptions | RemnantFitBounds | None = None,
    bounds: "FitBounds | RemnantFitBounds | None" = None,
    active_fraction_bounds: tuple[float, float] = (0.01, 0.99),
) -> "FitResult":
    """Fit a remnant and return the package's standard :class:`FitResult`.

    Parameters
    ----------
    observation
        An :class:`~fr2dynage.model.Observation`, or an already constructed
        :class:`KDAExtForwardModel` for advanced callers.
    assumptions
        Fixed model assumptions.  For compatibility with the lower-level API,
        a :class:`RemnantFitBounds` may be passed here when ``observation`` is
        already a model.
    bounds
        Existing :class:`~fr2dynage.fitting.FitBounds` or native
        :class:`RemnantFitBounds`.  ``FitBounds.age_myr`` is interpreted as
        total-age bounds.
    active_fraction_bounds
        Bounds on ``t_on/T``.  They are converted to the internally optimized
        remnant fraction.  Values must lie strictly inside ``(0, 1)``.

    Notes
    -----
    The returned ``FitResult.age_myr`` is the *total* dynamical age.  Active
    and remnant durations are retained in ``derived`` and ``diagnostics``.
    """

    # Imported lazily so this module can remain independent if fitting.py later
    # imports the forward model for CLI dispatch.
    from .fitting import FitBounds, FitResult

    if (
        not isinstance(active_fraction_bounds, (tuple, list))
        or len(active_fraction_bounds) != 2
        or any(
            not _is_finite_real(value)
            for value in active_fraction_bounds
        )
        or not (
            0.0
            < active_fraction_bounds[0]
            < active_fraction_bounds[1]
            < 1.0
        )
    ):
        raise ValueError(
            "active_fraction_bounds must be two increasing values strictly "
            "inside (0, 1)"
        )

    if isinstance(observation, KDAExtForwardModel):
        model = observation
        if isinstance(assumptions, RemnantFitBounds) and bounds is None:
            native_bounds = assumptions
        elif assumptions is not None and not isinstance(
            assumptions,
            ModelAssumptions,
        ):
            raise TypeError(
                "assumptions must be ModelAssumptions or RemnantFitBounds"
            )
        else:
            native_bounds = bounds
    else:
        if assumptions is not None and not isinstance(
            assumptions,
            ModelAssumptions,
        ):
            raise TypeError("assumptions must be ModelAssumptions")
        model = KDAExtForwardModel(observation, assumptions)
        native_bounds = bounds

    if native_bounds is None:
        active_lower = float(active_fraction_bounds[0])
        active_upper = float(active_fraction_bounds[1])
        remnant_bounds = RemnantFitBounds(
            remnant_fraction=(
                1.0 - active_upper,
                1.0 - active_lower,
            )
        )
    elif isinstance(native_bounds, RemnantFitBounds):
        remnant_bounds = native_bounds
    elif isinstance(native_bounds, FitBounds):
        active_lower = float(active_fraction_bounds[0])
        active_upper = float(active_fraction_bounds[1])
        remnant_bounds = RemnantFitBounds(
            total_age_myr=tuple(native_bounds.age_myr),
            remnant_fraction=(
                1.0 - active_upper,
                1.0 - active_lower,
            ),
            alpha_inj=tuple(native_bounds.alpha_inj),
            log10_jet_power_w=tuple(
                native_bounds.log10_jet_power_w
            ),
        )
    else:
        raise TypeError("bounds must be FitBounds or RemnantFitBounds")

    if len(model.observation.spectrum) < 5:
        raise RemnantFitError(
            "the four-parameter remnant fit requires at least five detected "
            "spectrum points so that at least one residual degree of freedom remains"
        )

    detailed = _fit_remnant_joint_model(model, remnant_bounds)
    diagnostics = {
        **detailed.diagnostics,
        "activity_state": "remnant",
        "total_age_myr": detailed.total_age_myr,
        "active_duration_myr": detailed.active_duration_myr,
        "remnant_duration_myr": detailed.remnant_duration_myr,
        "remnant_fraction": detailed.remnant_fraction,
        "active_fraction": 1.0 - detailed.remnant_fraction,
    }
    return FitResult(
        source_name=detailed.source_name,
        method="remnant_joint_log_likelihood",
        success=detailed.success,
        age_myr=detailed.total_age_myr,
        alpha_inj=detailed.alpha_inj,
        jet_power_w=detailed.jet_power_w,
        rho0_kg_m3=detailed.rho0_kg_m3,
        objective=detailed.objective,
        objective_name=detailed.objective_name,
        derived=detailed.derived,
        spectrum=detailed.spectrum,
        diagnostics=diagnostics,
        warnings=detailed.warnings,
        activity_history={
            "state": "remnant",
            "post_shutdown_dynamics": model.post_shutdown_dynamics,
            "total_age_myr": detailed.total_age_myr,
            "active_duration_myr": detailed.active_duration_myr,
            "fresh_particle_supply_cessation_age_myr": (
                detailed.active_duration_myr
            ),
            "remnant_duration_myr": detailed.remnant_duration_myr,
            "active_fraction": 1.0 - detailed.remnant_fraction,
            "remnant_fraction": detailed.remnant_fraction,
        },
    )


__all__ = [
    "KDAExtForwardModel",
    "RemnantDynamicsState",
    "RemnantFitBounds",
    "RemnantFitError",
    "RemnantFitResult",
    "fit_remnant_joint",
]
