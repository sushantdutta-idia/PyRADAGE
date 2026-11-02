"""Independent analytic checks for the radio posterior sampler and priors.

The likelihoods here are Gaussian or constant, not radio-kernel replicas.
Finite-particle tolerances are deliberately broader than IID Monte Carlo
errors because rejuvenated SMC particles share ancestry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import truncnorm

from fr2dynage.maps_io import MapInputError
from fr2dynage.radio_inference import make_field_speed_move, make_prior, tempered_smc
from fr2dynage.radio_spectra import jp_loss_coefficient


def test_scalar_gaussian_likelihood_recovers_conjugate_posterior():
    observed, sigma = 0.9, 0.45
    expected_mean = observed / (1 + sigma**2)
    expected_variance = sigma**2 / (1 + sigma**2)
    draws = []
    for seed in (11, 29):
        z, likelihood, diagnostics = tempered_smc(
            lambda row: -0.5 * ((row[0] - observed) / sigma)**2,
            1, particles=512, seed=seed, mcmc_steps=8,
        )
        draws.append(z[:, 0])
        assert np.all(np.isfinite(likelihood))
        assert diagnostics["stages"][-1]["beta"] == 1
        temperatures = [stage["beta"] for stage in diagnostics["stages"]]
        assert np.all(np.diff([0.0] + temperatures) > 0)
        assert all(0 < stage["pre_resampling_ess"] <= 512.000001
                   for stage in diagnostics["stages"])
        assert all(0 <= stage["acceptance"] <= 1 for stage in diagnostics["stages"])
        assert any(stage["pre_resampling_ess"] < 500 for stage in diagnostics["stages"])
    pooled = np.concatenate(draws)
    assert pooled.mean() == pytest.approx(expected_mean, abs=0.07)
    assert pooled.var() == pytest.approx(expected_variance, rel=0.18)
    assert abs(draws[0].mean() - draws[1].mean()) < 0.12


def test_correlated_linear_gaussian_recovers_mean_and_covariance():
    design = np.array([[1.0, 0.6], [-0.3, 1.0]])
    covariance = np.array([[0.2, 0.08], [0.08, 0.5]])
    observed = np.array([1.2, -0.4])
    precision = np.linalg.inv(covariance)
    expected_covariance = np.linalg.inv(np.eye(2) + design.T @ precision @ design)
    expected_mean = expected_covariance @ design.T @ precision @ observed

    def likelihood(row):
        residual = design @ row - observed
        return -0.5 * residual @ precision @ residual

    samples = [tempered_smc(likelihood, 2, particles=512, seed=seed, mcmc_steps=8)[0]
               for seed in (22, 37)]
    pooled = np.concatenate(samples)
    np.testing.assert_allclose(pooled.mean(axis=0), expected_mean, atol=0.08)
    np.testing.assert_allclose(np.cov(pooled.T), expected_covariance, rtol=0.25, atol=0.035)
    assert np.cov(pooled.T)[0, 1] < 0  # catches transpose/sign mistakes


def test_constant_likelihood_preserves_prior_and_reports_real_ess():
    z, likelihood, diagnostics = tempered_smc(
        lambda row: -7.25, 2, particles=1024, seed=103, mcmc_steps=8,
    )
    np.testing.assert_allclose(z.mean(axis=0), np.zeros(2), atol=0.09)
    np.testing.assert_allclose(np.cov(z.T), np.eye(2), atol=0.13)
    assert np.all(likelihood == -7.25)
    assert len(diagnostics["stages"]) == 1
    assert diagnostics["stages"][0]["pre_resampling_ess"] == pytest.approx(1024)
    assert diagnostics["log_evidence_up_to_constant"] == pytest.approx(-7.25)
    assert "correlated" in diagnostics["ess_interpretation"]


def test_likelihood_constant_does_not_change_posterior():
    def likelihood(row):
        return -0.5 * ((row[0] - 0.6) / 0.55)**2

    z1, _, d1 = tempered_smc(likelihood, 1, particles=256, seed=94, mcmc_steps=4)
    z2, _, d2 = tempered_smc(lambda row: likelihood(row) + 37.0, 1,
                            particles=256, seed=94, mcmc_steps=4)
    np.testing.assert_allclose(z1, z2, atol=1e-9)
    assert d2["log_evidence_up_to_constant"] - d1["log_evidence_up_to_constant"] == pytest.approx(37.0)


def test_zero_likelihood_domain_gives_half_normal_posterior():
    z, likelihood, diagnostics = tempered_smc(
        lambda row: 0.0 if row[0] >= 0 else -np.inf,
        1, particles=1024, seed=51, mcmc_steps=8,
    )
    assert np.all(z[:, 0] >= 0)
    assert np.all(likelihood == 0)
    assert z[:, 0].mean() == pytest.approx(math.sqrt(2 / math.pi), abs=0.075)
    assert z[:, 0].var() == pytest.approx(1 - 2 / math.pi, abs=0.08)
    assert 0.4 < diagnostics["physical_domain_prior_rejection_fraction"] < 0.6


@pytest.mark.parametrize("bad_likelihood", [np.nan, np.inf])
def test_invalid_likelihood_is_not_silently_accepted(bad_likelihood):
    with pytest.raises(MapInputError, match="likelihood"):
        tempered_smc(lambda row: bad_likelihood, 1, particles=64)


def test_empty_physical_support_has_clear_error():
    with pytest.raises(MapInputError, match="prior mass"):
        tempered_smc(lambda row: -np.inf, 1, particles=64)


def test_prior_transforms_match_known_distribution_quantiles():
    probabilities = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    latent = ndtri(probabilities)
    uniform = make_prior("alpha_inj", {"distribution": "uniform", "low": 0.5, "high": 1.5})
    np.testing.assert_allclose(uniform.transform(latent), 0.5 + probabilities)
    logarithmic = make_prior("flow_speed_km_s", {"distribution": "log_uniform", "low": 100, "high": 10000})
    np.testing.assert_allclose(logarithmic.transform(latent), 100 * 100**probabilities)
    normal = make_prior("alpha_inj", {"distribution": "normal", "mean": 0.8, "sigma": 0.2,
                                      "low": 0.5, "high": 1.4})
    expected = truncnorm.ppf(probabilities, (0.5 - 0.8) / 0.2, (1.4 - 0.8) / 0.2,
                            loc=0.8, scale=0.2)
    np.testing.assert_allclose(normal.transform(latent), expected, rtol=1e-12)
    positive = make_prior("magnetic_field_ug", {"distribution": "lognormal", "log_mean": math.log(3),
                                                "log_sigma": 0.4, "low": 0.1, "high": 10})
    expected = np.exp(truncnorm.ppf(probabilities, (math.log(0.1) - math.log(3)) / 0.4,
                                   (math.log(10) - math.log(3)) / 0.4,
                                   loc=math.log(3), scale=0.4))
    np.testing.assert_allclose(positive.transform(latent), expected, rtol=1e-12)


def test_tail_truncation_stays_finite_and_inside_physical_bounds():
    # Valid but almost wholly out-of-bounds prior. Direct upper CDF interpolation
    # rounds to exactly 1 at z~6 and used to produce an infinite parameter.
    prior = make_prior("alpha_inj", {"distribution": "normal", "mean": -2.0, "sigma": 0.42})
    values = prior.transform(np.array([-8.0, -4.0, 0.0, 4.0, 6.0, 8.0]))
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.5) & (values <= 1.5))
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("name,spec", [
    ("galaxy_speed", {"distribution": "uniform"}),
    ("flow_speed_km_s", {"distribution": "normal", "mean": 1000, "sigma": 0}),
    ("alpha_inj", {"distribution": "uniform", "mean": 0.7}),
    ("alpha_inj", {"distribution": "uniform", "low": 0.1, "high": 1.0}),
    ("magnetic_field_ug", {"distribution": "lognormal", "log_mean": 1000, "log_sigma": 0.1}),
])
def test_invalid_prior_specifications_fail_closed(name, spec):
    with pytest.raises(MapInputError):
        make_prior(name, spec)


@pytest.mark.parametrize("name,spec", [
    ("flow_speed_km_s", {"distribution": "uniform", "low": 100, "high": 10000}),
    ("flow_speed_km_s", {"distribution": "log_uniform", "low": 100, "high": 10000}),
    ("alpha_inj", {"distribution": "normal", "mean": 0.8, "sigma": 0.2}),
    ("magnetic_field_ug", {"distribution": "lognormal", "log_mean": math.log(3), "log_sigma": 0.6}),
    ("alpha_inj", {"distribution": "normal", "mean": -2.0, "sigma": 0.42}),
])
def test_prior_inverse_cdf_roundtrip(name, spec):
    prior = make_prior(name, spec)
    latent = np.linspace(-4, 4, 33)
    physical = prior.transform(latent)
    recovered = prior.inverse_transform(physical)
    np.testing.assert_allclose(recovered, latent, atol=2e-6)
    np.testing.assert_allclose(prior.transform(recovered), physical, rtol=1e-12)


def _relative_loss_field(field, redshift=0.1469):
    """The exact JP ridge coordinate, independently normalized at 3 microG."""
    normalization = jp_loss_coefficient(3.0, redshift) / math.sqrt(3.0)
    values = np.asarray(field)
    loss = np.array([jp_loss_coefficient(float(value), redshift) for value in values.flat])
    return loss.reshape(values.shape) / np.sqrt(values) / normalization


def _ridge_priors(branch_count, speed_spec=None):
    speed = speed_spec or {"distribution": "log_uniform", "low": 100.0, "high": 30000.0}
    return ([make_prior("flow_speed_km_s", speed) for _ in range(branch_count)]
            + [make_prior("magnetic_field_ug", {"distribution": "log_uniform", "low": 0.3, "high": 15.0}),
               make_prior("alpha_inj", {"distribution": "uniform"})])


def test_exact_two_branch_ridge_move_preserves_shape_and_draws_field_prior():
    # With log-uniform speeds, both physical velocity densities cancel the
    # scale**2 Jacobian. Every in-bounds field proposal must be accepted.
    priors = _ridge_priors(2)
    count = 2048
    initial = np.column_stack((np.full(count, 1000.0), np.full(count, 1500.0),
                               np.full(count, 3.0), np.full(count, 0.8)))
    latent = np.column_stack([prior.inverse_transform(initial[:, index])
                              for index, prior in enumerate(priors)])
    move = make_field_speed_move(priors, 2, 0.1469, lambda row: 0.0, shape_steps=0)
    moved, likelihood, acceptance = move(latent, np.zeros(count), 1.0, np.random.default_rng(309))
    physical = np.column_stack([prior.transform(moved[:, index])
                                for index, prior in enumerate(priors)])
    assert acceptance == 1.0
    assert np.all(likelihood == 0)
    relative = _relative_loss_field(physical[:, 2])
    np.testing.assert_allclose(physical[:, 0] / relative, 1000.0, rtol=2e-12)
    np.testing.assert_allclose(physical[:, 1] / relative, 1500.0, rtol=2e-12)
    np.testing.assert_allclose(physical[:, 3], 0.8, atol=1e-14)
    log_field = np.log(physical[:, 2])
    assert log_field.mean() == pytest.approx(math.log(math.sqrt(0.3 * 15)), abs=0.07)
    assert log_field.var() == pytest.approx(math.log(15 / 0.3)**2 / 12, abs=0.10)


def test_ridge_move_with_lognormal_speed_matches_independent_density_quadrature():
    # Conditional on u=v/h(B), the target in x=log B is proportional to
    # exp[-(log(v(B))-mu_v)^2/(2 sigma_v^2)]. This independent 1D quadrature
    # detects a missing physical-coordinate Jacobian or double-counted prior.
    priors = _ridge_priors(1, {"distribution": "lognormal", "log_mean": math.log(1500),
                              "log_sigma": 0.25, "low": 100.0, "high": 30000.0})
    count = 1024
    initial = np.column_stack((np.full(count, 1000.0), np.full(count, 3.0), np.full(count, 0.8)))
    latent = np.column_stack([prior.inverse_transform(initial[:, index])
                              for index, prior in enumerate(priors)])
    move = make_field_speed_move(priors, 1, 0.1469, lambda row: 0.0, shape_steps=0)
    likelihood = np.zeros(count)
    rng = np.random.default_rng(723)
    for _ in range(35):
        latent, likelihood, _ = move(latent, likelihood, 1.0, rng)
    log_field = np.log(priors[1].transform(latent[:, 1]))
    grid = np.linspace(math.log(0.3), math.log(15), 4097)
    speeds = 1000 * _relative_loss_field(np.exp(grid))
    density = np.exp(-0.5 * ((np.log(speeds) - math.log(1500)) / 0.25)**2)
    density /= np.trapezoid(density, grid)
    expected_mean = np.trapezoid(grid * density, grid)
    expected_variance = np.trapezoid((grid - expected_mean)**2 * density, grid)
    assert log_field.mean() == pytest.approx(expected_mean, abs=0.09)
    assert log_field.var() == pytest.approx(expected_variance, abs=0.13)
    assert np.all(likelihood == 0)


def test_tempered_smc_ridge_oracle_recovers_unidentified_field_prior():
    priors = _ridge_priors(1)
    sigma = 0.07

    def likelihood(row):
        speed = float(priors[0].transform(row[0]))
        field = float(priors[1].transform(row[1]))
        coordinate = math.log(speed / float(_relative_loss_field(field)))
        return -0.5 * ((coordinate - math.log(1000.0)) / sigma)**2

    move = make_field_speed_move(priors, 1, 0.1469, likelihood)
    latent, _, diagnostics = tempered_smc(likelihood, len(priors), particles=512,
                                          seed=803, mcmc_steps=4, rejuvenation=move)
    speed = priors[0].transform(latent[:, 0])
    field = priors[1].transform(latent[:, 1])
    coordinate = np.log(speed / _relative_loss_field(field))
    assert coordinate.mean() == pytest.approx(math.log(1000.0), abs=0.025)
    assert coordinate.var() == pytest.approx(sigma**2, rel=0.30)
    assert np.log(field).mean() == pytest.approx(math.log(math.sqrt(0.3 * 15)), abs=0.12)
    assert np.log(field).var() == pytest.approx(math.log(15 / 0.3)**2 / 12, abs=0.16)
    assert all(stage["physical_ridge_acceptance"] > 0.9 for stage in diagnostics["stages"])


def test_invariant_shape_smc_recovers_correlated_gaussian_flow_and_spectral_target():
    # In (x=log(v/h(B))-log(1000), z_alpha, z_eta) coordinates the log-speed
    # prior is flat and the other two priors are standard normal. A Gaussian
    # observation therefore gives an analytically solvable correlated target.
    # The velocity bounds are many posterior standard deviations away for
    # every B in its support; they do not affect these moments appreciably.
    priors = _ridge_priors(1) + [make_prior("deceleration", {
        "distribution": "uniform", "low": 0.0, "high": 3.0})]
    design = np.array([[1.0, 0.35, -0.22],
                       [0.4, 1.0, 0.3],
                       [-0.25, 0.15, 1.0]])
    error_covariance = np.diag(np.array([0.09, 0.24, 0.28])**2)
    precision = np.linalg.inv(error_covariance)
    observed = np.array([0.12, 0.5, -0.35])
    expected_covariance = np.linalg.inv(
        design.T @ precision @ design + np.diag([0.0, 1.0, 1.0]))
    expected_mean = expected_covariance @ design.T @ precision @ observed

    def coordinates(row):
        speed = float(priors[0].transform(row[0]))
        field = float(priors[1].transform(row[1]))
        return np.array([math.log(speed / float(_relative_loss_field(field)))
                         - math.log(1000.0), row[2], row[3]])

    def likelihood(row):
        residual = design @ coordinates(row) - observed
        return -0.5 * residual @ precision @ residual

    coordinate_draws, field_draws = [], []
    for seed in (910, 911):
        move = make_field_speed_move(priors, 1, 0.1469, likelihood, shape_steps=6)
        latent, _, diagnostics = tempered_smc(
            likelihood, len(priors), particles=512, seed=seed,
            mcmc_steps=1, rejuvenation=move)
        coordinate_draws.append(np.array([coordinates(row) for row in latent]))
        field_draws.append(priors[1].transform(latent[:, 1]))
        assert 0.05 < move.last_shape_acceptance < 0.8
        assert diagnostics["stages"][-1]["beta"] == 1.0
    pooled = np.concatenate(coordinate_draws)
    np.testing.assert_allclose(pooled.mean(axis=0), expected_mean, atol=0.055)
    np.testing.assert_allclose(np.cov(pooled.T), expected_covariance,
                               rtol=0.28, atol=0.008)
    np.testing.assert_allclose(coordinate_draws[0].mean(axis=0),
                               coordinate_draws[1].mean(axis=0), atol=0.08)
    assert np.cov(pooled.T)[0, 1] < 0
    assert np.cov(pooled.T)[1, 2] < 0
    # The correlated spectral likelihood still cannot identify the field.
    log_field = np.log(np.concatenate(field_draws))
    assert log_field.mean() == pytest.approx(math.log(math.sqrt(0.3 * 15)), abs=0.10)
    assert log_field.var() == pytest.approx(math.log(15 / 0.3)**2 / 12, abs=0.15)


def test_shape_rejuvenation_preserves_nonuniform_velocity_and_spectral_priors():
    # Start in the exact prior with a constant likelihood. Every complete
    # rejuvenation must preserve all independent prior marginals. The two
    # different physical speed densities make this sensitive to omission of
    # the dv/dlog(v)=v Jacobian in the invariant-shape block.
    priors = [
        make_prior("flow_speed_km_s", {"distribution": "lognormal",
                   "log_mean": math.log(1500.0), "log_sigma": 0.4,
                   "low": 100.0, "high": 30000.0}),
        make_prior("flow_speed_km_s", {"distribution": "normal",
                   "mean": 5000.0, "sigma": 1000.0,
                   "low": 100.0, "high": 30000.0}),
        make_prior("magnetic_field_ug", {"distribution": "log_uniform",
                   "low": 0.3, "high": 15.0}),
        make_prior("alpha_inj", {"distribution": "normal",
                   "mean": 0.7, "sigma": 0.1}),
        make_prior("deceleration", {"distribution": "uniform",
                   "low": 0.0, "high": 3.0}),
    ]
    rng = np.random.default_rng(812)
    count = 1024
    latent = rng.normal(size=(count, len(priors)))
    likelihood = np.zeros(count)
    move = make_field_speed_move(priors, 2, 0.1469, lambda row: 0.0,
                                 shape_steps=4)
    for _ in range(20):
        latent, likelihood, _ = move(latent, likelihood, 1.0, rng)
    np.testing.assert_allclose(latent.mean(axis=0), np.zeros(len(priors)), atol=0.12)
    np.testing.assert_allclose(np.cov(latent.T), np.eye(len(priors)), atol=0.15)
    assert np.all(likelihood == 0)
    assert 0.05 < move.last_shape_acceptance < 0.8
