"""Numerical diagnostics for integrated, active FR II DYNAGE-style fits.

Intersection families condition on the observed size. Evolutionary tracks
instead keep the inferred environment and jet power fixed. These are different
constructions and must not be plotted or interpreted interchangeably. Neither
the unweighted classical Delta statistic nor minimum Q*t is a confidence rule.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from numbers import Real

import numpy as np

from .constants import KPC_M, MYR_S
from .fitting import FitBounds, FitError, FitResult, intersection_at_age
from .model import KDAForwardModel, ModelDomainError, Observation
from .numerics import estimate_local_covariance, fractional_error_sigma_ln


class DiagnosticError(ValueError):
    """Unsupported scope or invalid settings for a diagnostic calculation."""


def _positive_vector(values, name, *, increasing=True, maximum=512):
    original = np.asarray(values)
    if original.dtype.kind not in 'iuf' or any(isinstance(value, (bool, np.bool_))
                                              for value in np.asarray(values, dtype=object).flat):
        raise DiagnosticError(f'{name} must contain finite positive numbers')
    array = np.asarray(values, dtype=float)
    if (array.ndim != 1 or not 1 <= array.size <= maximum or
            not np.all(np.isfinite(array)) or np.any(array <= 0)):
        raise DiagnosticError(f'{name} must be a nonempty 1D array of at most {maximum} finite positive numbers')
    if len(np.unique(array)) != len(array) or (increasing and np.any(np.diff(array) <= 0)):
        raise DiagnosticError(f'{name} must be distinct' + (' and increasing' if increasing else ''))
    return array


def _active_fit(model, result):
    if not isinstance(model, KDAForwardModel):
        raise DiagnosticError('these tracks require an active FR II KDAForwardModel')
    if result.activity_history is not None or result.method.startswith('remnant'):
        raise DiagnosticError('active FR II tracks cannot represent remnant or restarted histories')
    if not result.success:
        raise DiagnosticError('tracks require a successful fitted result')
    for name in ('age_myr', 'jet_power_w', 'alpha_inj'):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise DiagnosticError(f'result.{name} must be finite and positive')


def _strict(document):
    json.dumps(document, allow_nan=False)
    return document


def observed_spectrum_diagnostics(observation: Observation) -> dict:
    """Fit descriptive linear/quadratic models to ln(P) versus ln(nu).

    The quadratic convention is ln(P)=A-alpha*x+c*x², so negative c means
    steepening. This does not infer a physical break frequency, injection
    index or age. Chi-square and coefficient errors are calibrated only when
    every point has an error, conditional on independent lognormal errors.
    """
    if not isinstance(observation, Observation):
        raise DiagnosticError('observation must be an Observation')
    points = sorted(observation.spectrum, key=lambda point: point.frequency_mhz)
    frequencies = np.array([point.frequency_mhz for point in points])
    log_frequency = np.log(frequencies)
    log_power = np.log([point.power_w_hz_sr for point in points])
    reference_log = float(np.mean(log_frequency))
    x = log_frequency - reference_log
    leverage = float(np.max(np.abs(x)))
    all_errors = all(point.fractional_error is not None for point in points)
    errors = (np.array([fractional_error_sigma_ln(point.fractional_error) for point in points])
              if all_errors else np.ones(len(points)))
    warnings = [
        'Polynomial curvature is a descriptive diagnostic, not a spectral break, injection index or age.',
        'Error calibration assumes independent lognormal spectrum measurements; shared flux-scale covariance is not fitted here.',
    ]
    span = float((log_frequency[-1] - log_frequency[0]) / math.log(10.))
    if span < 1.:
        warnings.append('The spectrum spans less than one decade; frequency leverage may be weak.')
    if not all_errors:
        warnings.append('At least one fractional error is missing; formal chi-square and curvature significance are unavailable.')
    result = {'n_points': len(points), 'frequencies_mhz_rest': frequencies.tolist(),
              'frequency_span_decades': span,
              'reference_frequency_mhz_rest': float(math.exp(reference_log)),
              'all_errors_supplied': all_errors,
              'error_model': 'independent lognormal coefficient-of-variation errors' if all_errors else 'equal-weight descriptive fits only',
              'quadratic_convention': 'ln P = A - alpha*x + c*x^2; x=ln(nu/nu_ref)',
              'age_identifiability': 'not_determined_by_this_diagnostic', 'warnings': warnings}
    if leverage <= 0 or not math.isfinite(leverage):
        result.update(status='unresolved_frequency_leverage', power_law=None, quadratic=None,
                      curvature_chi2_improvement=None, curvature_z=None)
        return _strict(result)

    def fit(degree):
        scaled = x / leverage
        design = np.column_stack([scaled ** order for order in range(degree + 1)])
        sigma_min = float(np.min(errors))
        relative_weights = sigma_min / errors
        whitened = design * relative_weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(whitened, log_power * relative_weights, rcond=None)
        if rank < degree + 1:
            return {'status': 'rank_deficient', 'coefficients': None}
        factor = np.array([leverage ** -order for order in range(degree + 1)])
        coefficients = coefficients * factor
        predicted = sum(coefficients[order] * x ** order for order in range(degree + 1))
        residual = log_power - predicted
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            raw_chi2 = float(np.sum((residual / errors) ** 2))
        calibrated = all_errors and math.isfinite(raw_chi2)
        covariance_scale = sigma_min ** 2
        estimate = (estimate_local_covariance(whitened, [f'c{order}' for order in range(degree + 1)],
                                             scale=covariance_scale)
                    if all_errors and covariance_scale > 0 else None)
        covariance = None
        if estimate is not None and estimate.covariance is not None:
            covariance_array = np.asarray(estimate.covariance) * factor[:, None] * factor[None, :]
            if np.all(np.isfinite(covariance_array)):
                covariance = covariance_array.tolist()
        dof = len(points) - degree - 1
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(predicted)):
            return {'status': 'unrepresentable_coefficients', 'coefficients': None}
        return {'status': 'ok', 'coefficients': coefficients.tolist(),
                'coefficient_covariance': covariance,
                'alpha_at_reference': float(-coefficients[1]),
                'curvature_c_ln': float(coefficients[2]) if degree == 2 else None,
                'chi2': raw_chi2 if calibrated else None,
                'degrees_of_freedom': dof,
                'reduced_chi2': raw_chi2 / dof if calibrated and dof > 0 else None,
                'rms_log_residual': float(np.sqrt(np.mean(residual ** 2))),
                'normalized_design_condition': float(np.linalg.cond(whitened)),
                'covariance_scope': 'known supplied errors, no residual-variance scaling' if all_errors else 'uncalibrated'}

    linear = fit(1)
    quadratic = fit(2) if len(points) >= 4 else None
    delta, significance = None, None
    if (quadratic is not None and linear.get('chi2') is not None and
            quadratic.get('chi2') is not None):
        delta = float(max(0., linear['chi2'] - quadratic['chi2']))
        covariance = quadratic.get('coefficient_covariance')
        if covariance is not None and covariance[2][2] > 0:
            significance = float(quadratic['curvature_c_ln'] / math.sqrt(covariance[2][2]))
    if len(points) < 4:
        warnings.append('At least four detected frequencies are needed for a quadratic fit with residual degrees of freedom.')
    detected = significance is not None and abs(significance) >= 3. and delta is not None and delta >= 9.
    result.update(status='descriptive_fit', power_law=linear, quadratic=quadratic,
                  curvature_chi2_improvement=delta, curvature_z=significance,
                  curvature_required_by_heuristic=bool(detected),
                  steepening_required_by_heuristic=bool(detected and quadratic['curvature_c_ln'] < 0),
                  curvature_heuristic='absolute quadratic-coefficient z>=3 and Delta chi2>=9; conditional on supplied independent errors')
    return _strict(result)


def sample_intersection_tracks(model, result, bounds: FitBounds, ages_myr) -> dict:
    """Sample frequency intersections at fixed size and fitted injection index."""
    _active_fit(model, result)
    ages = _positive_vector(ages_myr, 'ages_myr')
    if not isinstance(bounds, FitBounds):
        raise DiagnosticError('bounds must be FitBounds')
    if ages[0] < bounds.age_myr[0] or ages[-1] > bounds.age_myr[1]:
        raise DiagnosticError('intersection ages must lie inside the supplied fit bounds')
    tracks = []
    for age in ages:
        row = {'age_myr': float(age), 'valid': False}
        try:
            solution, roots = intersection_at_age(model, float(age), result.alpha_inj, bounds)
            row.update(valid=True, delta=float(solution.delta),
                       log10_q_by_frequency=list(solution.log10_q_by_frequency),
                       log10_rho0_by_frequency=list(solution.log10_rho0_by_frequency),
                       consensus_log10_q=float(solution.consensus_log10_q),
                       consensus_log10_rho0=float(solution.consensus_log10_rho0),
                       maximum_roots_in_power_equation=int(roots))
            _strict(row)
        except (FitError, ModelDomainError, ValueError, ArithmeticError) as exc:
            row = {'age_myr': float(age), 'valid': False, 'reason': str(exc)[:500],
                   'delta': None, 'log10_q_by_frequency': None,
                   'log10_rho0_by_frequency': None, 'maximum_roots_in_power_equation': None}
        tracks.append(row)
    valid_rows = [row for row in tracks if row['valid']]
    delta_range = (max(row['delta'] for row in valid_rows) - min(row['delta'] for row in valid_rows)) if valid_rows else None
    flat = delta_range is not None and delta_range <= 1e-5
    selection_status = ('insufficient_valid_samples' if len(valid_rows) < 3 else
                        'flat_or_numerically_unresolved_delta_profile' if flat else 'sampled_delta_minimum')
    best_row = min(valid_rows, key=lambda row: row['delta']) if valid_rows else None
    return _strict({'track_kind': 'intersection_family_at_fixed_observed_size',
                    'frequencies_mhz_rest': [float(point.frequency_mhz) for point in model.observation.spectrum],
                    'alpha_inj': float(result.alpha_inj), 'tracks': tracks,
                    'valid_points': sum(row['valid'] for row in tracks),
                    'sampled_age_selection': {'status': selection_status,
                         'delta_range': delta_range,
                         'sampled_minimum_age_myr': best_row['age_myr'] if selection_status == 'sampled_delta_minimum' else None,
                         'flatness_rule': 'Delta range <=1e-5 over supplied samples; numerical diagnostic only, not a confidence threshold'},
                    'statistic_scope': 'unweighted mean pairwise distance in log10(Q),log10(rho0); not chi-square or confidence',
                    'warning': 'Density and power vary along these inverse curves; they are not a source evolutionary track.'})


def evolution_track(model, result, ages_myr, frequencies_mhz) -> dict:
    """Evolve an active lobe at fixed fitted Q and density normalization.

    The snapshot field anchor, if supplied, is resolved at the fitted epoch.
    Its energy partition is then held fixed during the model evolution, rather
    than incorrectly re-imposing one constant current field at every epoch.
    """
    _active_fit(model, result)
    ages = _positive_vector(ages_myr, 'ages_myr')
    frequencies = _positive_vector(frequencies_mhz, 'frequencies_mhz', increasing=False, maximum=128)
    best_state = model.dynamics_state(result.jet_power_w, result.age_myr * MYR_S)
    density = best_state.density_parameter_si
    ratio = model._magnetic_ratio(2 * result.alpha_inj + 1, best_state)
    assumptions = replace(model.assumptions, equipartition_field_ug=None, magnetic_ratio=ratio)
    geometry = model.observation.geometry
    projection = math.sin(math.radians(geometry.inclination_deg)) * (2 if geometry.mode == 'two_sided' else 1)
    tracks = []
    for age in ages:
        try:
            length = model.lobe_length(result.jet_power_w, density, float(age) * MYR_S)
            varied_geometry = replace(geometry, projected_size_kpc=length / KPC_M * projection)
            varied_observation = replace(model.observation, geometry=varied_geometry)
            varied_model = KDAForwardModel(varied_observation, assumptions)
            derived = varied_model.derived_quantities(result.jet_power_w, float(age) * MYR_S, result.alpha_inj)
            powers = [varied_model.radio_power(float(frequency) * 1e6, result.jet_power_w,
                                               float(age) * MYR_S, result.alpha_inj) for frequency in frequencies]
            if any(not math.isfinite(power) or power < 0 for power in powers):
                raise DiagnosticError('evolutionary radio power must be finite and nonnegative')
            domain_flags = []
            if length < assumptions.core_radius_m:
                domain_flags.append('below_powerlaw_atmosphere_core_radius')
            if derived['instantaneous_head_speed_over_c'] >= 1:
                domain_flags.append('superluminal_head_speed')
            row = {'age_myr': float(age), 'valid': True, 'domain_flags': domain_flags,
                   'lobe_length_kpc': length / KPC_M,
                   'projected_size_kpc': varied_geometry.projected_size_kpc,
                   'density_parameter_rho0_a0_beta_si': derived['density_parameter_rho0_a0_beta_si'],
                   'rho0_kg_m3': derived['rho0_kg_m3_conditional_on_core_radius'],
                   'cocoon_pressure_pa': derived['cocoon_pressure_pa'],
                   'magnetic_field_t': derived['newest_cohort_magnetic_field_t'],
                   'stored_energy_j': derived['stored_cocoon_energy_j'],
                   'injected_energy_j': derived['injected_jet_energy_j'],
                   'mean_head_speed_over_c': derived['mean_head_speed_over_c'],
                   'instantaneous_head_speed_over_c': derived['instantaneous_head_speed_over_c'],
                   'radio_power_w_hz_sr_by_frequency': powers}
            _strict(row)
        except (FitError, ModelDomainError, ValueError, ArithmeticError) as exc:
            row = {'age_myr': float(age), 'valid': False, 'reason': str(exc)[:500]}
        tracks.append(row)
    return _strict({'track_kind': 'active_evolution_at_fixed_environment_and_jet_power',
                    'frequencies_mhz_rest': frequencies.tolist(),
                    'jet_power_w_per_jet': float(result.jet_power_w), 'alpha_inj': float(result.alpha_inj),
                    'density_parameter_rho0_a0_beta_si': float(density),
                    'field_energy_partition_held_fixed': float(ratio),
                    'field_anchor_at_fit_epoch_ug': (float(model.assumptions.equipartition_field_ug)
                                                    if model.assumptions.equipartition_field_ug is not None else None),
                    'magnetic_field_interpretation': 'snapshot anchor resolved at fitted epoch; frozen energy partition evolves with cocoon pressure, not a repeated constant-field fit',
                    'tracks': tracks, 'valid_points': sum(row['valid'] for row in tracks),
                    'warning': 'Conditional active self-similar FR II predictions; not valid for arbitrary WAT/NAT or remnant/restarted histories.'})


def summarize_fit_quality(model, result, bounds: FitBounds) -> dict:
    """Collect numerical, physical and data-leverage flags without new fitting."""
    _active_fit(model, result)
    spectrum = observed_spectrum_diagnostics(model.observation)
    diagnostics = result.diagnostics
    flags = []
    free_parameters = 2 if result.method == 'joint_fixed_jet_power_log_likelihood' else 3
    if not spectrum['all_errors_supplied']:
        flags.append('measurement_error_model_uncalibrated')
    if len(model.observation.spectrum) <= free_parameters:
        flags.append('no_residual_degrees_of_freedom')
    if not spectrum.get('curvature_required_by_heuristic', False):
        flags.append('curvature_not_established_by_polynomial_diagnostic')
    if diagnostics.get('parameter_near_boundary') or diagnostics.get('alpha_grid_boundary') or diagnostics.get('jet_power_root_near_boundary') or diagnostics.get('age_search', {}).get('age_grid_boundary'):
        flags.append('parameter_or_classic_root_near_search_boundary')
    condition = diagnostics.get('information_matrix_condition_number')
    if condition is not None and (not math.isfinite(condition) or condition > 1e6):
        flags.append('ill_conditioned_local_information')
    if result.method.startswith('joint') and diagnostics.get('local_covariance') is None:
        flags.append('local_covariance_unavailable')
    quadrature = diagnostics.get('quadrature_relative_error_order_vs_double', [])
    if quadrature and max(quadrature) > .005:
        flags.append('quadrature_error_above_0p5_percent')
    root_count = diagnostics.get('age_search', {}).get('maximum_roots_in_power_equation', 1)
    if root_count > 1:
        flags.append('multiple_luminosity_roots')
    chi2 = diagnostics.get('chi2', diagnostics.get('weighted_log_power_chi2_at_consensus'))
    if chi2 is None:
        values = [row.get('chi2_contribution_log_power') for row in result.spectrum]
        if values and all(value is not None for value in values):
            chi2 = float(sum(values))
    dof = len(model.observation.spectrum) - free_parameters
    reduced = float(chi2 / dof) if chi2 is not None and dof > 0 else None
    if reduced is not None and reduced > 4:
        flags.append('poor_consensus_spectrum_fit')
    if result.derived.get('instantaneous_head_speed_over_c', 0) >= 1:
        flags.append('superluminal_head_speed')
    energy_scan = diagnostics.get('classic_alpha_scan', [])
    finite_energy = [float(row['log10_q_times_t_j']) for row in energy_scan
                     if isinstance(row, dict) and isinstance(row.get('log10_q_times_t_j'), Real)
                     and math.isfinite(row['log10_q_times_t_j'])]
    energy_span = max(finite_energy) - min(finite_energy) if finite_energy else None
    if len(finite_energy) >= 3 and energy_span <= .01:
        flags.append('flat_classic_energy_selection_profile')
    normalized_positions = {}
    for name, value, pair in [('age_myr', result.age_myr, bounds.age_myr),
                              ('alpha_inj', result.alpha_inj, bounds.alpha_inj),
                              ('log10_jet_power_w', math.log10(result.jet_power_w), bounds.log10_jet_power_w)]:
        normalized_positions[name] = float((value - pair[0]) / (pair[1] - pair[0]))
    if any(value < .002 or value > .998 for value in normalized_positions.values()):
        if 'parameter_or_classic_root_near_search_boundary' not in flags:
            flags.append('parameter_or_classic_root_near_search_boundary')
    return _strict({'scope': 'conditional_active_FRII_fit', 'flags': flags,
                    'chi2': float(chi2) if chi2 is not None else None,
                    'free_parameter_count': free_parameters,
                    'degrees_of_freedom': dof, 'reduced_chi2': reduced,
                    'normalized_bound_positions': normalized_positions,
                    'classic_energy_selection_dynamic_range_dex': energy_span,
                    'classic_energy_flatness_rule': 'range<=0.01 dex over at least three scanned alpha values; descriptive heuristic, not a confidence rule',
                    'observed_spectrum': spectrum,
                    'classical_objective_is_likelihood': False if result.method == 'classic_dynage' else None,
                    'uncertainty_scope': 'fit/data diagnostic flags do not include all geometry, environment, magnetic-history or model uncertainties',
                    'warnings': ['A successful optimizer or a minimum classical Delta does not establish an identifiable or unique physical age.',
                                 'A flat/power-law spectrum can leave age, injection index and environment degenerate; inspect profiles and explicit assumption sensitivities.']})
