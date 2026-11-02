from __future__ import annotations

import json

import numpy as np
import pytest

from fr2dynage.numerics import estimate_local_covariance, fractional_error_sigma_ln


def test_covariance_avoids_singular_normal_equations() -> None:
    jacobian = np.array([[1.0, 1.0, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert np.linalg.matrix_rank(jacobian) == 3
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(jacobian.T @ jacobian)
    estimate = estimate_local_covariance(jacobian, ["age", "alpha", "power"])
    assert estimate.reason is None
    assert np.asarray(estimate.covariance) == pytest.approx(
        np.array([[1e20, -1e20, 0.0], [-1e20, 1e20, 0.0], [0.0, 0.0, 1.0]]), rel=1e-12
    )
    assert estimate.information_condition_number > 1e19


@pytest.mark.parametrize("jacobian", [np.ones((4, 3)), np.full((4, 3), np.nan), np.empty((2, 3))])
def test_covariance_unavailable_for_invalid_or_rank_deficient_jacobian(jacobian) -> None:
    estimate = estimate_local_covariance(jacobian, ["age", "alpha", "power"])
    assert estimate.covariance is None
    assert estimate.reason


def test_zero_variance_correlations_are_json_safe() -> None:
    estimate = estimate_local_covariance(np.eye(3), ["age", "alpha", "power"], scale=0.0)
    assert estimate.standard_errors == {"age": 0.0, "alpha": 0.0, "power": 0.0}
    assert estimate.correlation == [[None] * 3 for _ in range(3)]
    json.dumps(estimate.__dict__, allow_nan=False)


def test_covariance_combines_scale_before_inverting_tiny_singular_values() -> None:
    estimate = estimate_local_covariance(np.eye(2) * 1e-310, ["age", "alpha"], scale=1e-320)
    assert estimate.reason is None
    assert np.asarray(estimate.covariance) == pytest.approx(np.eye(2) * (1e-320 / 1e-310 / 1e-310), rel=1e-12)


def test_svd_failure_does_not_abort_fit_diagnostics(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD failed")
    monkeypatch.setattr(np.linalg, "svd", fail)
    estimate = estimate_local_covariance(np.eye(3), ["age", "alpha", "power"])
    assert estimate.covariance is None
    assert "decomposition failed" in estimate.reason


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_unrepresentable_covariance_is_unavailable(factor) -> None:
    estimate = estimate_local_covariance(np.eye(3) * factor, ["age", "alpha", "power"])
    assert estimate.covariance is None
    assert "floating-point range" in estimate.reason


@pytest.mark.parametrize("error", [1e-200, 1e-160, 1e-10])
def test_tiny_fractional_errors_remain_positive(error) -> None:
    assert fractional_error_sigma_ln(error) == pytest.approx(error, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("error", [0, -1, True, float("nan"), float("inf"), pytest.param(10**10000, id="huge-integer")])
def test_invalid_fractional_errors_are_rejected(error) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        fractional_error_sigma_ln(error)
