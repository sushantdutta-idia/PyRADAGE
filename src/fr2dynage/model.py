"""Kaiser--Alexander/KDA forward model used by the inverse estimator.

The implementation follows the published equations rather than any original
DYNAGE program.  Internally all dimensional values are SI.  Public dataclasses
use unit-bearing field names to make conversions explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
from numbers import Real
from typing import Iterable, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar

from .radio_classes import normalize_radio_class, require_fr2_class

from .constants import (
    C_LIGHT,
    E_CHARGE,
    KPC_M,
    M_ELECTRON,
    MHZ_HZ,
    MU_0,
    MYR_S,
    RADIATION_CONSTANT,
    SIGMA_T,
)


class ModelDomainError(ValueError):
    """Raised when parameters lie outside the physical/model domain."""


def _is_finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


GeometryMode = Literal["two_sided", "one_lobe"]
C1Prescription = Literal["dynage2007", "kaiser2000"]


@dataclass(frozen=True)
class SpectrumPoint:
    """One rest-frame monochromatic radio-power measurement.

    ``power_w_hz_sr`` is power per unit frequency and steradian, matching the
    convention of KDA equation 16 and Machalski et al. (2007).
    """

    frequency_mhz: float
    power_w_hz_sr: float
    fractional_error: float | None = None

    def __post_init__(self) -> None:
        if not _is_finite_real(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ModelDomainError("frequency_mhz must be finite and positive")
        if not _is_finite_real(self.power_w_hz_sr) or self.power_w_hz_sr <= 0:
            raise ModelDomainError("power_w_hz_sr must be finite and positive")
        if self.fractional_error is not None and (
            not _is_finite_real(self.fractional_error)
            or self.fractional_error <= 0
        ):
            raise ModelDomainError("fractional_error must be finite and positive")

    @property
    def frequency_hz(self) -> float:
        return self.frequency_mhz * MHZ_HZ


@dataclass(frozen=True)
class Geometry:
    """Projected source geometry with an explicit lobe convention.

    ``two_sided`` reproduces the legacy DYNAGE convention:

    * projected_size is hotspot-to-hotspot;
    * axial_ratio is that total length divided by lobe width;
    * true one-lobe length is D/(2 sin(theta));
    * KDA lobe axial ratio is R_T = AR/2 (not deprojected).

    ``one_lobe`` uses a core-to-hotspot size and R_T equal to the supplied
    axial ratio.  Its spectrum must refer to that one lobe.
    """

    projected_size_kpc: float
    axial_ratio: float
    inclination_deg: float = 90.0
    mode: GeometryMode = "two_sided"
    size_error_kpc: float | None = None
    axial_ratio_error: float | None = None

    def __post_init__(self) -> None:
        if not _is_finite_real(self.projected_size_kpc) or self.projected_size_kpc <= 0:
            raise ModelDomainError("projected_size_kpc must be finite and positive")
        if not _is_finite_real(self.axial_ratio) or self.axial_ratio <= 0:
            raise ModelDomainError("axial_ratio must be finite and positive")
        if not _is_finite_real(self.inclination_deg) or not 0 < self.inclination_deg <= 90:
            raise ModelDomainError("inclination_deg must be in (0, 90]")
        if self.mode not in ("two_sided", "one_lobe"):
            raise ModelDomainError("geometry mode must be 'two_sided' or 'one_lobe'")
        if self.size_error_kpc is not None and (
            not _is_finite_real(self.size_error_kpc) or self.size_error_kpc <= 0
        ):
            raise ModelDomainError("size_error_kpc must be finite and positive")
        if self.axial_ratio_error is not None and (
            not _is_finite_real(self.axial_ratio_error) or self.axial_ratio_error <= 0
        ):
            raise ModelDomainError("axial_ratio_error must be finite and positive")
        if self.lobe_axial_ratio <= 0.5:
            raise ModelDomainError("derived lobe axial ratio R_T must exceed 0.5")

    @property
    def lobe_length_m(self) -> float:
        projected = self.projected_size_kpc * KPC_M
        deprojection = math.sin(math.radians(self.inclination_deg))
        divisor = 2.0 if self.mode == "two_sided" else 1.0
        return projected / (divisor * deprojection)

    @property
    def lobe_axial_ratio(self) -> float:
        return self.axial_ratio / 2.0 if self.mode == "two_sided" else self.axial_ratio

    @property
    def emitting_lobes(self) -> int:
        return 2 if self.mode == "two_sided" else 1


@dataclass(frozen=True)
class Observation:
    name: str
    redshift: float
    geometry: Geometry
    spectrum: tuple[SpectrumPoint, ...]
    radio_class: str = "fr2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "radio_class", normalize_radio_class(self.radio_class))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelDomainError("source name cannot be empty")
        if not _is_finite_real(self.redshift) or self.redshift < 0:
            raise ModelDomainError("redshift must be finite and non-negative")
        if len(self.spectrum) < 3:
            raise ModelDomainError("at least three radio-power measurements are required")
        frequencies = [p.frequency_mhz for p in self.spectrum]
        if len(set(frequencies)) != len(frequencies):
            raise ModelDomainError("spectrum frequencies must be unique")

    @classmethod
    def from_points(
        cls,
        name: str,
        redshift: float,
        geometry: Geometry,
        points: Iterable[SpectrumPoint],
        *,
        radio_class: str = "fr2",
    ) -> "Observation":
        return cls(name, redshift, geometry, tuple(sorted(points, key=lambda p: p.frequency_mhz)), radio_class)


@dataclass(frozen=True)
class ModelAssumptions:
    """Fixed assumptions of a KDA/DYNAGE calculation."""

    beta: float = 1.5
    core_radius_kpc: float = 10.0
    gamma_min: float = 1.0
    gamma_max: float = 1.0e7
    gamma_c: float = 5.0 / 3.0
    gamma_x: float = 5.0 / 3.0
    # DYNAGE 2007 used KDA Case 3 (Gamma_B=5/3).  The later, commonly used
    # 2009 lobe analysis switched to Case 2 (Gamma_B=4/3); users can select it.
    gamma_b: float = 5.0 / 3.0
    k_prime: float = 0.0
    magnetic_ratio: float | None = None
    cmb_temperature_k: float = 2.7255
    c1_prescription: C1Prescription = "dynage2007"
    integration_order: int = 160
    max_log_injection_span: float = 32.0
    # Appended to preserve the pre-existing positional constructor order.
    # This is a supplied observed-snapshot field constraint, not a KDA
    # equipartition estimate.
    equipartition_field_ug: float | None = None

    def __post_init__(self) -> None:
        if not _is_finite_real(self.beta) or not 0 <= self.beta < 2:
            raise ModelDomainError("beta must satisfy 0 <= beta < 2 for the FR II model")
        if not _is_finite_real(self.core_radius_kpc) or self.core_radius_kpc <= 0:
            raise ModelDomainError("core_radius_kpc must be finite and positive")
        if (
            not _is_finite_real(self.gamma_min)
            or not _is_finite_real(self.gamma_max)
            or not 1 <= self.gamma_min < self.gamma_max
        ):
            raise ModelDomainError("require finite 1 <= gamma_min < gamma_max")
        for label, value in (
            ("gamma_c", self.gamma_c),
            ("gamma_x", self.gamma_x),
            ("gamma_b", self.gamma_b),
        ):
            if not _is_finite_real(value) or value <= 1:
                raise ModelDomainError(f"{label} must be finite and exceed 1")
        if not _is_finite_real(self.k_prime) or self.k_prime < 0:
            raise ModelDomainError("k_prime must be finite and non-negative")
        if self.magnetic_ratio is not None and (
            not _is_finite_real(self.magnetic_ratio) or self.magnetic_ratio <= 0
        ):
            raise ModelDomainError("magnetic_ratio must be finite and positive when supplied")
        if self.equipartition_field_ug is not None:
            if not _is_finite_real(self.equipartition_field_ug) or self.equipartition_field_ug <= 0:
                raise ModelDomainError("equipartition_field_ug must be finite and positive when supplied")
            if self.magnetic_ratio is not None:
                raise ModelDomainError("equipartition_field_ug and magnetic_ratio are mutually exclusive")
            if self.gamma_b != self.gamma_c:
                raise ModelDomainError("equipartition_field_ug requires gamma_b == gamma_c for a homogeneous current field")
            adopted_t = 0.4 * self.equipartition_field_ug * 1e-10
            energy = adopted_t * adopted_t / (2.0 * MU_0)
            if not math.isfinite(energy) or energy <= 0:
                raise ModelDomainError("equipartition_field_ug cannot represent a positive finite magnetic energy density")
        if not _is_finite_real(self.cmb_temperature_k) or self.cmb_temperature_k <= 0:
            raise ModelDomainError("cmb_temperature_k must be finite and positive")
        if self.c1_prescription not in ("dynage2007", "kaiser2000"):
            raise ModelDomainError("unknown c1_prescription")
        if (
            isinstance(self.integration_order, bool)
            or not isinstance(self.integration_order, int)
            or self.integration_order < 32
        ):
            raise ModelDomainError("integration_order must be an integer of at least 32")
        if (
            not _is_finite_real(self.max_log_injection_span)
            or not 8 <= self.max_log_injection_span <= 80
        ):
            raise ModelDomainError(
                "max_log_injection_span must be finite and between 8 and 80"
            )

    @property
    def core_radius_m(self) -> float:
        return self.core_radius_kpc * KPC_M

    def with_integration_order(self, order: int) -> "ModelAssumptions":
        return replace(self, integration_order=order)


@dataclass(frozen=True)
class DynamicsState:
    age_s: float
    jet_power_w: float
    density_parameter_si: float
    rho0_kg_m3: float
    lobe_length_m: float
    lobe_volume_m3: float
    pressure_pa: float
    pressure_ratio: float
    c1: float
    c3: float


@lru_cache(maxsize=16)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    return nodes, weights


def pressure_ratio(beta: float, lobe_axial_ratio: float) -> float:
    """Kaiser (2000) empirical hotspot-to-cocoon pressure ratio."""

    return (2.14 - 0.52 * beta) * lobe_axial_ratio ** (2.04 - 0.25 * beta)


def geometry_coefficient(lobe_axial_ratio: float) -> float:
    """Volume coefficient c3 for a cylindrical one-sided lobe."""

    return math.pi / (4.0 * lobe_axial_ratio**2)


def length_coefficient(
    assumptions: ModelAssumptions,
    lobe_axial_ratio: float,
) -> float:
    """Return c1 for the selected, explicitly versioned dynamics prescription.

    ``dynage2007`` combines KA's energy-conservation coefficient with the
    Kaiser (2000) empirical pressure ratio, as used by Machalski et al. (2007).
    ``kaiser2000`` uses Kaiser's later volume-self-consistent coefficient.
    They are not interchangeable, so the result records the selection.
    """

    beta = assumptions.beta
    gc = assumptions.gamma_c
    gx = assumptions.gamma_x
    phc = pressure_ratio(beta, lobe_axial_ratio)
    c3 = geometry_coefficient(lobe_axial_ratio)
    if assumptions.c1_prescription == "dynage2007":
        energy_term = 9.0 * gc - 4.0 - beta + 2.25 * (gc - 1.0) * phc
        base = (
            phc
            * (gx + 1.0)
            * (gc - 1.0)
            * (5.0 - beta) ** 3
            / (18.0 * c3 * energy_term)
        )
    else:
        base = (
            phc ** (1.0 / gc)
            * gc
            * (gc - 1.0)
            * (gx + 1.0)
            * (5.0 - beta) ** 3
            / (18.0 * c3 * (9.0 * gc - 4.0 - beta))
        )
    if base <= 0 or not math.isfinite(base):
        raise ModelDomainError("c1 is undefined for these assumptions")
    return base ** (1.0 / (5.0 - beta))


def electron_energy_integral(p: float, gamma_min: float, gamma_max: float) -> float:
    """Integral of (gamma-1) gamma^-p with stable p=1 and p=2 limits."""

    if not _is_finite_real(p) or p <= 1:
        raise ModelDomainError("electron index p must exceed 1")
    if (
        not _is_finite_real(gamma_min)
        or not _is_finite_real(gamma_max)
        or not 1 <= gamma_min < gamma_max
    ):
        raise ModelDomainError("require finite 1 <= gamma_min < gamma_max")
    log_min = math.log(gamma_min)
    log_max = math.log(gamma_max)
    # log(max/min) loses narrow intervals when the ratio rounds to one.
    span = math.log1p((gamma_max - gamma_min) / gamma_min)
    first_exponent = 2.0 - p
    second_exponent = 1.0 - p

    def power_integral(exponent: float) -> float:
        if exponent == 0.0:
            return span
        if exponent > 0.0:
            return (
                math.exp(exponent * log_max)
                * -math.expm1(-exponent * span)
                / exponent
            )
        return (
            math.exp(exponent * log_min)
            * -math.expm1(exponent * span)
            / -exponent
        )

    x = first_exponent * span
    y = second_exponent * span
    if max(abs(x), abs(y)) < 0.1:
        # For a narrow interval, subtracting the two power integrals loses
        # precision, especially at gamma_min=1 where the integrand vanishes.
        # Use (exprel(x)-exprel(y))/(x-y)'s convergent series instead.
        divided_power = 1.0
        y_power = y
        factorial = 2.0
        difference_series = divided_power / factorial
        for degree in range(2, 13):
            divided_power = x * divided_power + y_power
            y_power *= y
            factorial *= degree + 1
            difference_series += divided_power / factorial
        first_span_integral = (
            span if first_exponent == 0.0 else math.expm1(x) / first_exponent
        )
        value = math.exp(first_exponent * log_min) * (
            (gamma_min - 1.0) / gamma_min * first_span_integral
            + span**2 / gamma_min * difference_series
        )
    else:
        value = power_integral(first_exponent) - power_integral(second_exponent)
    if value <= 0 or not math.isfinite(value):
        raise ModelDomainError("electron normalization integral is not positive")
    return value


def _power_difference(exponent: float, x: np.ndarray) -> np.ndarray:
    """Return (1-x**exponent)/exponent, including its logarithmic limit."""

    logx = np.log(x)
    if abs(exponent) < 1.0e-10:
        return -logx
    return -np.expm1(exponent * logx) / exponent


class KDAForwardModel:
    """Forward dynamics and synchrotron-loss calculation for one observation."""

    equation_set = "KDA97 radiation + DYNAGE2007 geometry/pressure"

    def __init__(self, observation: Observation, assumptions: ModelAssumptions | None = None):
        require_fr2_class(observation.radio_class)
        self.observation = observation
        self.assumptions = assumptions or ModelAssumptions()
        self._rt = observation.geometry.lobe_axial_ratio
        self._phc = pressure_ratio(self.assumptions.beta, self._rt)
        self._c3 = geometry_coefficient(self._rt)
        self._c1 = length_coefficient(self.assumptions, self._rt)

    @property
    def pressure_ratio(self) -> float:
        return self._phc

    @property
    def c1(self) -> float:
        return self._c1

    @property
    def c3(self) -> float:
        return self._c3

    def density_parameter_from_length(self, jet_power_w: float, age_s: float) -> float:
        """Infer k_rho = rho0*a0**beta from Q, t, and observed length."""

        if (
            not _is_finite_real(jet_power_w)
            or not _is_finite_real(age_s)
            or jet_power_w <= 0
            or age_s <= 0
        ):
            raise ModelDomainError("jet power and age must be finite and positive")
        beta = self.assumptions.beta
        length = self.observation.geometry.lobe_length_m
        return jet_power_w * age_s**3 * self._c1 ** (5.0 - beta) / length ** (5.0 - beta)

    def lobe_length(self, jet_power_w: float, density_parameter_si: float, age_s: float) -> float:
        if (
            not _is_finite_real(jet_power_w)
            or not _is_finite_real(density_parameter_si)
            or not _is_finite_real(age_s)
            or jet_power_w <= 0
            or density_parameter_si <= 0
            or age_s <= 0
        ):
            raise ModelDomainError(
                "jet power, density parameter, and age must be finite and positive"
            )
        beta = self.assumptions.beta
        return (
            self._c1
            * (jet_power_w / density_parameter_si) ** (1.0 / (5.0 - beta))
            * age_s ** (3.0 / (5.0 - beta))
        )

    def cocoon_pressure(self, jet_power_w: float, density_parameter_si: float, age_s: float) -> float:
        if (
            not _is_finite_real(jet_power_w)
            or not _is_finite_real(density_parameter_si)
            or not _is_finite_real(age_s)
            or jet_power_w <= 0
            or density_parameter_si <= 0
            or age_s <= 0
        ):
            raise ModelDomainError(
                "jet power, density parameter, and age must be finite and positive"
            )
        beta = self.assumptions.beta
        gx = self.assumptions.gamma_x
        prefactor = (
            18.0
            * self._c1 ** (2.0 - beta)
            / ((gx + 1.0) * (5.0 - beta) ** 2 * self._phc)
        )
        return (
            prefactor
            * density_parameter_si ** (3.0 / (5.0 - beta))
            * jet_power_w ** ((2.0 - beta) / (5.0 - beta))
            * age_s ** (-(4.0 + beta) / (5.0 - beta))
        )

    def dynamics_state(self, jet_power_w: float, age_s: float) -> DynamicsState:
        density_parameter = self.density_parameter_from_length(jet_power_w, age_s)
        rho0 = density_parameter / self.assumptions.core_radius_m ** self.assumptions.beta
        length = self.lobe_length(jet_power_w, density_parameter, age_s)
        pressure = self.cocoon_pressure(jet_power_w, density_parameter, age_s)
        return DynamicsState(
            age_s=age_s,
            jet_power_w=jet_power_w,
            density_parameter_si=density_parameter,
            rho0_kg_m3=rho0,
            lobe_length_m=length,
            lobe_volume_m3=self._c3 * length**3,
            pressure_pa=pressure,
            pressure_ratio=self._phc,
            c1=self._c1,
            c3=self._c3,
        )

    def _magnetic_ratio(self, p: float, state: DynamicsState | None = None) -> float:
        """Injection energy partition, optionally fixed by a current field.

        In anchor mode ``B_now=0.4*B_eq`` and the candidate snapshot's total
        energy determines r=u_B/(u_c-u_B). Equal magnetic/cocoon adiabatic
        indices make that same current B hold for all injection cohorts.
        This is not the unaged same-emissivity power-law r conversion.
        """
        if self.assumptions.equipartition_field_ug is not None:
            if state is None:
                raise ModelDomainError("equipartition field anchor requires a dynamics snapshot")
            field_t = 0.4 * self.assumptions.equipartition_field_ug * 1e-10
            u_b = field_t * field_t / (2.0 * MU_0)
            u_c = state.pressure_pa / (self.assumptions.gamma_c - 1.0)
            if not math.isfinite(u_c) or not 0.0 < u_b < u_c:
                raise ModelDomainError("adopted magnetic-field energy must be below the total cocoon energy density")
            ratio = u_b / (u_c - u_b)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ModelDomainError("anchored magnetic energy partition is not representable")
            return ratio
        ratio = self.assumptions.magnetic_ratio
        return (1.0 + p) / 4.0 if ratio is None else ratio

    def _cohort_values(
        self,
        x: np.ndarray,
        frequency_hz: float,
        state: DynamicsState,
        p: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return dP_nu/dt_i and a validity mask for injection cohorts."""

        a = self.assumptions
        gc = a.gamma_c
        a1 = ((4.0 + a.beta) / (5.0 - a.beta)) / gc
        (
            pressure_i,
            u_e_i,
            u_b_now,
            gamma_now,
            denominator,
            gamma_i,
        ) = self._cohort_kinematics(x, frequency_hz, state, p)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            norm_integral = electron_energy_integral(p, a.gamma_min, a.gamma_max)
            n0_i = u_e_i / (M_ELECTRON * C_LIGHT**2 * norm_integral)
            n_gamma_now = (
                n0_i
                * gamma_i ** (2.0 - p)
                / gamma_now**2
                * x ** (4.0 * a1 / 3.0)
            )
            dvolume_dti = (
                (gc - 1.0)
                * state.jet_power_w
                / pressure_i
                * self._phc ** ((1.0 - gc) / gc)
                * x ** (-a1)
            )
            dpower_dti = (
                SIGMA_T
                * C_LIGHT
                * u_b_now
                / (6.0 * math.pi * frequency_hz)
                * gamma_now**3
                * n_gamma_now
                * dvolume_dti
            )

        valid = (
            np.isfinite(dpower_dti)
            & np.isfinite(gamma_i)
            & (denominator > 0.0)
            & (gamma_i >= a.gamma_min)
            & (gamma_i <= a.gamma_max)
            & (dpower_dti >= 0.0)
        )
        return np.where(valid, dpower_dti, 0.0), valid

    def _cohort_kinematics(
        self,
        x: np.ndarray,
        frequency_hz: float,
        state: DynamicsState,
        p: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cohort quantities used by both domain location and emissivity."""

        a = self.assumptions
        delta = (4.0 + a.beta) / (5.0 - a.beta)
        a1 = delta / a.gamma_c
        a3 = 1.0 - a1 * (a.gamma_b + 1.0 / 3.0)
        a4 = 1.0 - a1 / 3.0
        r = self._magnetic_ratio(p, state)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            pressure_i = state.pressure_pa * x ** (-delta)
            u_e_i = pressure_i / (
                (a.gamma_c - 1.0) * (1.0 + a.k_prime) * (1.0 + r)
            )
            u_b_i = r * pressure_i / ((a.gamma_c - 1.0) * (1.0 + r))
            u_b_now = u_b_i * x ** (a.gamma_b * a1)
            field_t = np.sqrt(2.0 * MU_0 * u_b_now)
            larmor_hz = E_CHARGE * field_t / (2.0 * math.pi * M_ELECTRON)
            gamma_now = np.sqrt(frequency_hz / larmor_hz)
            u_cmb = (
                RADIATION_CONSTANT
                * a.cmb_temperature_k**4
                * (1.0 + self.observation.redshift) ** 4
            )
            loss_integral = (
                u_b_i * x ** (a.gamma_b * a1) * _power_difference(a3, x)
                + u_cmb * _power_difference(a4, x)
            )
            eta = (
                4.0
                * SIGMA_T
                / (3.0 * M_ELECTRON * C_LIGHT)
                * state.age_s
                * gamma_now
                * loss_integral
            )
            denominator = 1.0 - eta
            gamma_i = gamma_now * x ** (-a1 / 3.0) / denominator
        return pressure_i, u_e_i, u_b_now, gamma_now, denominator, gamma_i

    def _valid_intervals(
        self,
        frequency_hz: float,
        state: DynamicsState,
        p: float,
    ) -> list[tuple[float, float]]:
        """Locate valid y=-log(t_i/t) intervals from continuous domain roots.

        Solving the denominator and Lorentz-factor cutoff boundaries
        independently avoids missing a narrow valid interval whose midpoint is
        absent from a fixed validity-mask grid.
        """

        ymin, ymax = self._injection_log_age_bounds(state)
        sample = np.linspace(ymin, ymax, 257)

        def kinematics(y: float) -> tuple[float, float]:
            values = self._cohort_kinematics(
                np.asarray([math.exp(-y)]), frequency_hz, state, p
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
                left, right = float(grid[index]), float(grid[index + 1])
                f_left, f_right = values[index], values[index + 1]
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

        tolerance = 1.0e-10 * (1.0 + ymax - ymin)

        def unique(values: list[float]) -> list[float]:
            ordered = sorted(value for value in values if ymin <= value <= ymax)
            result: list[float] = []
            for value in ordered:
                if not result or abs(value - result[-1]) > tolerance:
                    result.append(value)
            return result

        sampled_kinematics = self._cohort_kinematics(
            np.exp(-sample), frequency_hz, state, p
        )
        denominator_roots = unique(
            roots_on_grid(denominator, sample, sampled_kinematics[4])
        )
        domain_edges = unique([ymin, ymax, *denominator_roots])
        intervals: list[tuple[float, float]] = []

        def predicate(y: float) -> bool:
            _, mask = self._cohort_values(
                np.asarray([math.exp(-y)]), frequency_hz, state, p
            )
            return bool(mask[0])

        for domain_left, domain_right in zip(domain_edges[:-1], domain_edges[1:], strict=True):
            if domain_right - domain_left <= tolerance:
                continue
            if denominator(0.5 * (domain_left + domain_right)) <= 0.0:
                continue
            inset = max(1.0e-12, 1.0e-10 * (domain_right - domain_left))
            left_is_root = any(
                abs(domain_left - root) <= tolerance for root in denominator_roots
            )
            right_is_root = any(
                abs(domain_right - root) <= tolerance for root in denominator_roots
            )
            left = domain_left + inset if left_is_root else domain_left
            right = domain_right - inset if right_is_root else domain_right
            if right <= left:
                continue

            gamma_grid = np.linspace(left, right, 257)
            gamma_values = self._cohort_kinematics(
                np.exp(-gamma_grid), frequency_hz, state, p
            )[5]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_gamma = np.where(gamma_values > 0.0, np.log(gamma_values), np.nan)
            extrema: list[float] = []
            for index in range(1, len(gamma_grid) - 1):
                previous, current, following = (
                    log_gamma[index - 1],
                    log_gamma[index],
                    log_gamma[index + 1],
                )
                if not all(math.isfinite(value) for value in (previous, current, following)):
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
                    float(refined.x) if refined.success else float(gamma_grid[index])
                )

            monotonic_edges = unique([left, right, *extrema])
            cutoff_roots: list[float] = []
            for cutoff in (self.assumptions.gamma_min, self.assumptions.gamma_max):
                for branch_left, branch_right in zip(
                    monotonic_edges[:-1], monotonic_edges[1:], strict=True
                ):
                    f_left = gamma_i(branch_left) - cutoff
                    f_right = gamma_i(branch_right) - cutoff
                    if not math.isfinite(f_left) or not math.isfinite(f_right):
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

            validity_edges = unique([domain_left, domain_right, *cutoff_roots])
            for valid_left, valid_right in zip(
                validity_edges[:-1], validity_edges[1:], strict=True
            ):
                if valid_right - valid_left <= tolerance:
                    continue
                if predicate(0.5 * (valid_left + valid_right)):
                    if intervals and abs(valid_left - intervals[-1][1]) <= tolerance:
                        intervals[-1] = (intervals[-1][0], valid_right)
                    else:
                        intervals.append((valid_left, valid_right))
        return intervals

    def _injection_log_age_bounds(
        self,
        state: DynamicsState,
    ) -> tuple[float, float]:
        """Integration bounds in ``y=-log(t_i/t)`` for injected cohorts."""

        return 0.0, self.assumptions.max_log_injection_span

    def radio_power_one_lobe(
        self,
        frequency_hz: float,
        jet_power_w: float,
        age_s: float,
        alpha_inj: float,
    ) -> float:
        """KDA monochromatic power of one lobe in W Hz^-1 sr^-1."""

        if (
            not _is_finite_real(frequency_hz)
            or not _is_finite_real(jet_power_w)
            or not _is_finite_real(age_s)
            or frequency_hz <= 0
            or jet_power_w <= 0
            or age_s <= 0
        ):
            raise ModelDomainError(
                "frequency, jet power, and age must be finite and positive"
            )
        if not _is_finite_real(alpha_inj) or not 0 < alpha_inj < 2:
            raise ModelDomainError("alpha_inj must be finite and lie between 0 and 2")
        p = 2.0 * alpha_inj + 1.0
        state = self.dynamics_state(jet_power_w, age_s)
        intervals = self._valid_intervals(frequency_hz, state, p)
        if not intervals:
            return 0.0
        nodes, weights = _legendre_rule(self.assumptions.integration_order)
        total = 0.0
        for lower, upper in intervals:
            half = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            y = midpoint + half * nodes
            x = np.exp(-y)
            dpower_dti, _ = self._cohort_values(x, frequency_hz, state, p)
            # dt_i = t*x*dy under y=-log(t_i/t), with reversed limits.
            total += half * float(np.dot(weights, dpower_dti * state.age_s * x))
        if not math.isfinite(total) or total < 0:
            raise ModelDomainError("radio-power integration returned a non-finite value")
        return total

    def radio_power(
        self,
        frequency_hz: float,
        jet_power_w: float,
        age_s: float,
        alpha_inj: float,
    ) -> float:
        """Power for the observed component (one lobe or both lobes)."""

        return self.observation.geometry.emitting_lobes * self.radio_power_one_lobe(
            frequency_hz, jet_power_w, age_s, alpha_inj
        )

    def spectrum(
        self,
        frequencies_hz: Iterable[float],
        jet_power_w: float,
        age_s: float,
        alpha_inj: float,
    ) -> np.ndarray:
        return np.asarray(
            [self.radio_power(float(nu), jet_power_w, age_s, alpha_inj) for nu in frequencies_hz],
            dtype=float,
        )

    def derived_quantities(self, jet_power_w: float, age_s: float, alpha_inj: float) -> dict[str, float]:
        if not _is_finite_real(alpha_inj) or not 0 < alpha_inj < 2:
            raise ModelDomainError("alpha_inj must be finite and lie between 0 and 2")
        state = self.dynamics_state(jet_power_w, age_s)
        a = self.assumptions
        p = 2.0 * alpha_inj + 1.0
        r = self._magnetic_ratio(p, state)
        u_c = state.pressure_pa / (a.gamma_c - 1.0)
        u_b = r * state.pressure_pa / ((a.gamma_c - 1.0) * (1.0 + r))
        field_t = math.sqrt(2.0 * MU_0 * u_b)
        rho_head = state.density_parameter_si * state.lobe_length_m ** (-a.beta)
        total_volume = self.observation.geometry.emitting_lobes * state.lobe_volume_m3
        total_stored_energy = u_c * total_volume
        one_jet_energy = jet_power_w * age_s
        injected_energy = self.observation.geometry.emitting_lobes * one_jet_energy
        average_speed = state.lobe_length_m / age_s
        instantaneous_speed = 3.0 / (5.0 - a.beta) * average_speed
        energy_identity = (
            (9.0 * a.gamma_c - 4.0 - a.beta) / (5.0 - a.beta)
            + 9.0 * (a.gamma_c - 1.0) * self._phc / (4.0 * (5.0 - a.beta))
        )
        result = {
            "age_myr": age_s / MYR_S,
            "alpha_inj": alpha_inj,
            "electron_index_p": p,
            "jet_power_w_per_jet": jet_power_w,
            "density_parameter_rho0_a0_beta_si": state.density_parameter_si,
            "rho0_kg_m3_conditional_on_core_radius": state.rho0_kg_m3,
            "ambient_density_at_head_kg_m3": rho_head,
            "cocoon_pressure_pa": state.pressure_pa,
            "cocoon_energy_density_j_m3": u_c,
            "newest_cohort_magnetic_field_t": field_t,
            "lobe_length_kpc_deprojected": state.lobe_length_m / KPC_M,
            "one_lobe_volume_m3": state.lobe_volume_m3,
            "modelled_total_volume_m3": total_volume,
            "stored_cocoon_energy_j": total_stored_energy,
            "one_jet_q_times_age_j": one_jet_energy,
            "injected_jet_energy_j": injected_energy,
            "injected_to_stored_energy_ratio": injected_energy / total_stored_energy,
            "one_jet_qt_to_stored_energy_ratio": one_jet_energy / total_stored_energy,
            "legacy_energy_identity_ratio": energy_identity,
            "mean_head_speed_over_c": average_speed / C_LIGHT,
            "instantaneous_head_speed_over_c": instantaneous_speed / C_LIGHT,
            "pressure_ratio_phc": self._phc,
            "c1": self._c1,
            "c3": self._c3,
        }
        if a.equipartition_field_ug is not None:
            result.update({
                "supplied_equipartition_field_ug": float(a.equipartition_field_ug),
                "adopted_field_fraction_of_equipartition": 0.4,
                "adopted_current_magnetic_field_ug": field_t * 1e10,
                "magnetic_to_particle_energy_ratio_at_injection": r,
                "current_magnetic_energy_density_j_m3": u_b,
                "current_particle_energy_density_j_m3": u_c - u_b,
            })
        return result

    def quadrature_relative_error(
        self,
        frequency_hz: float,
        jet_power_w: float,
        age_s: float,
        alpha_inj: float,
        higher_order_factor: int = 2,
    ) -> float:
        base = self.radio_power(frequency_hz, jet_power_w, age_s, alpha_inj)
        higher_assumptions = self.assumptions.with_integration_order(
            self.assumptions.integration_order * higher_order_factor
        )
        higher = KDAForwardModel(self.observation, higher_assumptions).radio_power(
            frequency_hz, jet_power_w, age_s, alpha_inj
        )
        if higher == 0:
            return 0.0 if base == 0 else math.inf
        return abs(base - higher) / higher
