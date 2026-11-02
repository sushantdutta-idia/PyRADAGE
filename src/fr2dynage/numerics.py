"""Numerical diagnostics shared by the bounded inverse solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: list[list[float]] | None = None
    correlation: list[list[float | None]] | None = None
    standard_errors: dict[str, float] | None = None
    information_condition_number: float | None = None
    reason: str | None = None


def estimate_local_covariance(
    jacobian: np.ndarray,
    parameter_names: Sequence[str],
    *,
    scale: float = 1.0,
) -> CovarianceEstimate:
    """Estimate covariance directly from the Jacobian's singular values.

    Forming and inverting ``J.T @ J`` loses precision by squaring the
    condition number.  Rank-deficient or unrepresentable estimates remain
    unavailable without invalidating an otherwise successful fit.  Undefined
    correlations (for example, zero estimated residual variance) are ``None``
    so all returned diagnostics can be encoded as strict JSON.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    count = len(parameter_names)
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != count
        or jacobian.shape[0] < count
        or count == 0
        or not np.all(np.isfinite(jacobian))
        or not math.isfinite(scale)
        or scale < 0.0
    ):
        return CovarianceEstimate(reason="invalid Jacobian or covariance scale")
    try:
        _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return CovarianceEstimate(reason="Jacobian singular-value decomposition failed")
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular_values[0]
    if np.any(singular_values <= threshold):
        return CovarianceEstimate(reason="Jacobian is rank deficient")

    log_condition = 2.0 * (
        math.log(float(singular_values[0])) - math.log(float(singular_values[-1]))
    )
    condition = math.exp(log_condition) if log_condition <= math.log(np.finfo(float).max) else None
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if scale == 0.0:
            covariance_array = np.zeros((count, count))
        else:
            inverse_factor = right_vectors.T * (math.sqrt(scale) / singular_values)
            covariance_array = inverse_factor @ inverse_factor.T
    if not np.all(np.isfinite(covariance_array)) or (
        scale > 0.0 and np.any(np.diag(covariance_array) <= 0.0)
    ):
        return CovarianceEstimate(
            information_condition_number=condition,
            reason="covariance exceeds floating-point range",
        )
    errors = np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        correlations = covariance_array / errors[:, None] / errors[None, :]
    correlation = [
        [float(np.clip(value, -1.0, 1.0)) if math.isfinite(value) else None for value in row]
        for row in correlations
    ]
    return CovarianceEstimate(
        covariance=covariance_array.tolist(),
        correlation=correlation,
        standard_errors={name: float(error) for name, error in zip(parameter_names, errors, strict=True)},
        information_condition_number=condition,
    )


def fractional_error_sigma_ln(fractional_error: float) -> float:
    """Convert a lognormal coefficient of variation to log-space sigma."""

    try:
        valid = (
            isinstance(fractional_error, Real)
            and not isinstance(fractional_error, bool)
            and math.isfinite(fractional_error)
            and fractional_error > 0
        )
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError("fractional_error must be finite and positive")
    if fractional_error < 1.0e-8:
        # sqrt(log1p(f*f)) tends to f; avoid underflow in f*f.
        return fractional_error
    if fractional_error < 1.0:
        log_variance_factor = math.log1p(fractional_error**2)
    else:
        log_variance_factor = (
            2.0 * math.log(fractional_error)
            + math.log1p(fractional_error**-2)
        )
    return math.sqrt(log_variance_factor)

