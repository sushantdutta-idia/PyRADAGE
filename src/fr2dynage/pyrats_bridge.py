"""Read PyRATS physical-property results as conditional radio-model inputs.

This adapter reads data only. In particular, ``command_line`` and any paths
or prose embedded in the report are provenance, never executable commands.
An equipartition field derived from the same radio images is not an independent
field measurement. The returned lognormal distribution is an explicitly
approximate plug-in constraint, not a joint radio/field likelihood.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .io import _reject_json_constant, _unique_object


class PyRATSPropertyError(ValueError):
    """PyRATS results cannot support the requested conditional field input."""


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise PyRATSPropertyError(f"{label} must be a JSON object")
    return value


def _number(value: Any, label: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PyRATSPropertyError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise PyRATSPropertyError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise PyRATSPropertyError(f"{label} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PyRATSPropertyError(f"{label} must be a nonempty string")
    return value.strip()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _interval(value: Any, label: str) -> dict:
    summary = _object(value, label)
    result = {key: _number(summary.get(key), f'{label}.{key}', positive=True)
              for key in ('median', 'p16', 'p84')}
    if not result['p16'] < result['median'] < result['p84']:
        raise PyRATSPropertyError(f'{label} requires p16 < median < p84')
    for key in ('mean', 'std', 'p2.5', 'p97.5'):
        if key in summary:
            result[key] = _number(summary[key], f'{label}.{key}', positive=True)
    if ('p2.5' in result and result['p2.5'] > result['p16']) or ('p97.5' in result and result['p97.5'] < result['p84']):
        raise PyRATSPropertyError(f'{label}: reported quantiles are unordered')
    return result


def _image_association(records, supplied, base: Path) -> dict:
    """Cross-check supplied target maps against raw inputs or prepared labels.

    Label matches establish lineage only, not pixel identity. PyRATS 2.1 does
    not normally store input hashes, so that limitation is explicit.
    """
    inputs = []
    for index, item in enumerate(records):
        entry = _object(item, f'input_maps[{index}]')
        label = _text(entry.get('label'), f'input_maps[{index}].label')
        filename = _text(entry.get('path'), f'input_maps[{index}].path')
        frequency = _number(entry.get('frequency_Hz'), f'input_maps[{index}].frequency_Hz', positive=True)
        record = {'label': label, 'path': str((base / Path(filename).expanduser()).resolve()),
                  'frequency_Hz': frequency}
        if 'OBJECT' in entry and isinstance(entry['OBJECT'], str):
            record['object'] = entry['OBJECT']
        if 'sha256' in entry:
            sha = entry['sha256']
            if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', sha):
                raise PyRATSPropertyError(f'input_maps[{index}].sha256 must be a SHA-256 hex digest')
            record['sha256'] = sha.casefold()
        inputs.append(record)
    if not inputs:
        raise PyRATSPropertyError('input_maps must contain source image provenance')
    if len({row['label'].casefold() for row in inputs}) != len(inputs):
        raise PyRATSPropertyError('input_maps contains duplicate labels')
    if supplied is None:
        return {'status': 'not_cross_checked_against_target_maps', 'reported_inputs': inputs,
                'matches': [], 'pixel_identity_verified': False}
    if not isinstance(supplied, (list, tuple)) or not supplied:
        raise PyRATSPropertyError('image_paths must be a nonempty list of paths or image records')
    matches = []
    for index, item in enumerate(supplied):
        frequency = None
        if isinstance(item, dict):
            if set(item) - {'path', 'frequency_mhz'}:
                raise PyRATSPropertyError('image_paths records accept only path and frequency_mhz')
            filename = item.get('path')
            if 'frequency_mhz' in item:
                frequency = _number(item['frequency_mhz'], f'image_paths[{index}].frequency_mhz', positive=True) * 1e6
        else:
            filename = item
        if not isinstance(filename, (str, Path)) or not str(filename).strip():
            raise PyRATSPropertyError('image_paths must contain real file paths')
        target = Path(filename).expanduser().resolve()
        exact = [row for row in inputs if Path(row['path']) == target]
        candidates = exact or [row for row in inputs if row['label'].casefold() == target.stem.casefold()]
        if len(candidates) != 1:
            raise PyRATSPropertyError(f'{target}: no unique PyRATS raw-input path or prepared-map label match')
        record = candidates[0]
        if frequency is not None and not math.isclose(frequency, record['frequency_Hz'], rel_tol=1e-5):
            raise PyRATSPropertyError(f'{target}: observing frequency differs from PyRATS provenance')
        try:
            actual_hash = _digest(target)
        except OSError as exc:
            raise PyRATSPropertyError(f'cannot verify target image {target}: {exc}') from exc
        hash_verified = False
        if exact and 'sha256' in record:
            if actual_hash != record['sha256']:
                raise PyRATSPropertyError(f'{target}: input SHA-256 differs from PyRATS report')
            hash_verified = True
        matches.append({'target_path': str(target), 'target_sha256': actual_hash,
                        'pyrats_label': record['label'], 'pyrats_input_path': record['path'],
                        'association': 'exact_input_path' if exact else 'prepared_map_label_only',
                        'pixel_hash_verified_against_report': hash_verified,
                        'frequency_checked': frequency is not None})
    if len({row['target_path'] for row in matches}) != len(matches):
        raise PyRATSPropertyError('image_paths contains duplicate target images')
    return {'status': 'associated_with_reported_inputs', 'reported_inputs': inputs,
            'matches': matches,
            'pixel_identity_verified': all(row['pixel_hash_verified_against_report'] for row in matches)}


ADOPTED_B_EQ_FRACTION = 0.4


def load_pyrats_properties(path: str | Path, *, expected_redshift: float,
                          systematic_log_sigma: float,
                          field_basis: str = 'equipartition', field_scale: float = ADOPTED_B_EQ_FRACTION,
                          image_paths=None) -> dict:
    """Read supported PyRATS 2.x results and construct a conditional B prior.

    The prior centre is the field Monte-Carlo median, not the deterministic
    direct solution. The statistical log width is ``0.5*log(p84/p16)``. Add
    the explicitly supplied model-systematic log width in quadrature. These
    calculations approximate a potentially asymmetric MC distribution by a
    lognormal; raw reported quantiles remain in provenance.

    ``field_scale`` defaults to the adopted B=0.4 B_eq prescription. It
    does not automatically adopt any ``sub_equipartition`` field in the report.
    Energetic properties are retained at their original field basis; no
    scaling of pressure/energy is invented for a different chosen field.
    Fixed or radio-fitted injection indices are reported, never imported as
    independent priors. The report is data only and is never executed.
    """
    filename = Path(path).expanduser().resolve()
    try:
        raw = filename.read_bytes()
        document = json.loads(raw.decode('utf8'), object_pairs_hook=_unique_object,
                              parse_constant=_reject_json_constant)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise PyRATSPropertyError(f'cannot read PyRATS results {filename}: {exc}') from exc
    root = _object(document, 'PyRATS results')
    version = _text(root.get('pyrats_version'), 'pyrats_version')
    if not re.fullmatch(r'2\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?', version):
        raise PyRATSPropertyError('unsupported PyRATS report version; adapter supports the explicit 2.x property schema')
    if field_basis not in {'equipartition', 'minimum_energy'}:
        raise PyRATSPropertyError('field_basis must be equipartition or minimum_energy; adopted fields require an explicit field_scale')
    expected = _number(expected_redshift, 'expected_redshift', nonnegative=True)
    systematic = _number(systematic_log_sigma, 'systematic_log_sigma', nonnegative=True)
    scale = _number(field_scale, 'field_scale', positive=True)
    configuration = _object(root.get('configuration'), 'configuration')
    redshift = _number(configuration.get('redshift'), 'configuration.redshift', nonnegative=True)
    if not math.isclose(redshift, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise PyRATSPropertyError('PyRATS redshift does not match the target redshift')
    solution = _object(root.get('self_consistent_solution'), 'self_consistent_solution')
    if solution.get('converged') is not True:
        raise PyRATSPropertyError('PyRATS self-consistent solution is not reported converged')
    equip = _object(solution.get('equipartition'), 'self_consistent_solution.equipartition')
    assumptions = _object(equip.get('inputs'), 'equipartition.inputs')
    if not math.isclose(_number(assumptions.get('redshift'), 'equipartition.inputs.redshift', nonnegative=True), redshift, rel_tol=1e-6, abs_tol=1e-8):
        raise PyRATSPropertyError('inconsistent redshifts within the PyRATS report')
    kappa = _number(assumptions.get('kappa_proton_electron_energy_ratio'), 'kappa', nonnegative=True)
    zeta = _number(assumptions.get('zeta_uB_over_uparticles'), 'zeta', positive=True)
    filling = _number(assumptions.get('filling_factor'), 'filling_factor', positive=True)
    gamma_min = _number(assumptions.get('gamma_min'), 'gamma_min', positive=True)
    gamma_max = _number(assumptions.get('gamma_max'), 'gamma_max', positive=True)
    if filling > 1. or gamma_max <= gamma_min or not math.isclose(zeta, 1. + kappa, rel_tol=1e-8):
        raise PyRATSPropertyError('inconsistent filling factor, electron limits or zeta=1+kappa assumptions')
    for key, used in (('kappa_proton_electron_energy_ratio', kappa),
                      ('zeta_uB_over_uparticles', zeta), ('filling_factor', filling),
                      ('gamma_min', gamma_min), ('gamma_max', gamma_max)):
        if key in configuration and not math.isclose(
                _number(configuration[key], 'configuration.' + key, nonnegative=True), used,
                rel_tol=1e-8, abs_tol=1e-12):
            raise PyRATSPropertyError(f'configuration and equipartition {key} assumptions disagree')
    direct_all = _object(equip.get('direct'), 'equipartition.direct')
    budgets = {}
    for basis in ('equipartition', 'minimum_energy'):
        direct = _object(direct_all.get(basis), f'equipartition.direct.{basis}')
        field_ug = _number(direct.get('B_uG'), f'{basis}.B_uG', positive=True)
        field_t = _number(direct.get('B_T'), f'{basis}.B_T', positive=True)
        if not math.isclose(field_ug * 1e-10, field_t, rel_tol=1e-6):
            raise PyRATSPropertyError(f'{basis}: B_T and B_uG unit values disagree')
        budgets[basis] = {'B_uG': field_ug, 'B_T': field_t}
        for key in ('volume_m3', 'electron_energy_density_J_m3', 'particle_energy_density_J_m3',
                    'magnetic_energy_density_J_m3', 'total_energy_density_J_m3',
                    'total_energy_J', 'pressure_Pa', 'electron_number_density_m3'):
            if key in direct:
                budgets[basis][key] = _number(direct[key], f'{basis}.{key}', positive=True)
    monte_carlo = _object(equip.get('monte_carlo'), 'equipartition.monte_carlo')
    successful = monte_carlo.get('n_success')
    if isinstance(successful, bool) or not isinstance(successful, int) or successful < 2 or monte_carlo.get('status') != 'ok':
        raise PyRATSPropertyError('PyRATS field Monte Carlo must report status ok and at least two successful samples')
    requested = monte_carlo.get('n_requested')
    rejected = monte_carlo.get('n_rejected')
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < successful:
            raise PyRATSPropertyError('PyRATS Monte Carlo n_requested must be an integer at least n_success')
        if successful/requested < .8:
            raise PyRATSPropertyError('PyRATS Monte Carlo success fraction below 80%; surviving field quantiles may be selection biased')
    if rejected is not None:
        if isinstance(rejected, bool) or not isinstance(rejected, int) or rejected < 0 or (requested is not None and rejected+successful != requested):
            raise PyRATSPropertyError('PyRATS Monte Carlo rejected/success/requested counts disagree')
    mc_key = 'B_eq_uG' if field_basis == 'equipartition' else 'B_me_uG'
    field_mc = _interval(monte_carlo.get(mc_key), 'monte_carlo.' + mc_key)
    statistical_width = .5 * math.log(field_mc['p84'] / field_mc['p16'])
    combined_width = math.hypot(statistical_width, systematic)
    field_median = field_mc['median'] * scale
    if not math.isfinite(field_median) or not math.isfinite(combined_width) or combined_width <= 0.:
        raise PyRATSPropertyError('scaled field or uncertainty is not representable')
    geometry = _object(root.get('geometry_physical'), 'geometry_physical')
    geometry_properties = {}
    for key in ('volume_m3', 'volume_kpc3', 'largest_linear_size_kpc', 'minor_axis_kpc',
                'projected_area_kpc2', 'axis_ratio', 'beams_per_source', 'kpc_per_arcsec',
                'volume_fractional_error', 'filling_factor'):
        if key in geometry:
            geometry_properties[key] = _number(geometry[key], 'geometry_physical.' + key,
                                               nonnegative=key == 'volume_fractional_error',
                                               positive=key != 'volume_fractional_error')
    for key in ('geometry_model', 'volume_estimator', 'assumption_note'):
        if key in geometry:
            geometry_properties[key] = _text(geometry[key], 'geometry_physical.' + key)
    volume = _number(assumptions.get('volume_m3'), 'equipartition.inputs.volume_m3', positive=True)
    if 'volume_m3' in geometry_properties and not math.isclose(volume, geometry_properties['volume_m3'], rel_tol=1e-6):
        raise PyRATSPropertyError('geometry and equipartition volumes disagree')
    if 'filling_factor' in geometry_properties and not math.isclose(filling, geometry_properties['filling_factor'], rel_tol=1e-8):
        raise PyRATSPropertyError('geometry and equipartition filling factors disagree')
    if any('volume_m3' in budget and not math.isclose(volume, budget['volume_m3'], rel_tol=1e-6)
           for budget in budgets.values()):
        raise PyRATSPropertyError('direct field budgets and equipartition input volumes disagree')
    alpha = _number(solution.get('alpha_inj_positive'), 'alpha_inj_positive', positive=True)
    alpha_error = _number(solution.get('alpha_inj_err'), 'alpha_inj_err', nonnegative=True)
    alpha_source = _text(solution.get('alpha_source'), 'alpha_source')
    source_maps = root.get('input_maps')
    if not isinstance(source_maps, list):
        raise PyRATSPropertyError('input_maps must be a list')
    association = _image_association(source_maps, image_paths, filename.parent)
    warnings = [
        'Equipartition/minimum-energy B is conditional on electron limits, non-radiating particle content, filling factor, geometry, injection spectrum and cosmology; it is not an independent magnetic-field measurement.',
        'The field distribution reuses the target radio flux. This plug-in conditional constraint does not propagate the full joint flux-field covariance and must not be described as an independent likelihood or validation.',
        'The explicit additional systematic width is a modelling assumption; adding it in quadrature does not prove coverage of all equipartition departures.',
        'A source-integrated field distribution is used as a shared homogeneous-field input; no spatial field map is inferred by this adapter.',
        'The report injection index is provenance only and is not imported as an independent prior; a fixed/user-override value is not a measurement.',
        'PyRATS Monte-Carlo quantiles are approximated by one lognormal centred on the MC median; reported asymmetric/raw quantiles are preserved.',
    ]
    if not association['pixel_identity_verified']:
        warnings.append('Target image pixel identity cannot be verified from this PyRATS report; matching labels/paths alone do not establish identical image data or preparation.')
    if scale != 1.:
        warnings.append('field_scale is an explicitly adopted departure from the chosen native field basis; listed energies and pressures remain at their original PyRATS field basis and were not rescaled.')
    if systematic == 0.:
        warnings.append('No additional field model-systematic width was supplied beyond the conditional PyRATS Monte-Carlo width.')
    if successful < 100:
        warnings.append('Fewer than 100 successful PyRATS Monte-Carlo samples; field-quantile precision may be poor.')
    provenance = {
        'kind': 'pyrats_conditional_radio_property_input', 'path': str(filename),
        'sha256': hashlib.sha256(raw).hexdigest(), 'pyrats_version': version,
        'generated_utc': root.get('generated_utc'), 'field_basis': field_basis,
        'field_scale_assumption': scale, 'field_mc_summary_uG': field_mc,
        'statistical_log_sigma': statistical_width, 'systematic_log_sigma_assumption': systematic,
        'combined_log_sigma': combined_width,
        'redshift': redshift, 'cosmology': configuration.get('cosmology'),
        'same_radio_data_reuse_approximation': True,
        'uncertainty_scope': 'conditional PyRATS field MC plus explicit independent log-scatter approximation; full joint radio-flux/B covariance is excluded',
        'image_association': association,
        'monte_carlo': {'n_success': successful, 'n_requested': requested, 'n_rejected': rejected,
                        'success_fraction': successful/requested if requested is not None else None,
                        'sampled': monte_carlo.get('sampled', []),
                        'not_sampled': monte_carlo.get('not_sampled', [])},
        'assumptions': {'kappa_proton_electron_energy_ratio': kappa, 'zeta_uB_over_uparticles': zeta,
                        'filling_factor': filling, 'gamma_min': gamma_min, 'gamma_max': gamma_max,
                        'volume_m3': volume, 'electron_spectrum': solution.get('electron_spectrum'),
                        'alpha_inj_positive': alpha, 'alpha_inj_err': alpha_error,
                        'alpha_source': alpha_source, 'solution_converged': True,
                        'spectrum_fallback': solution.get('spectrum_fallback')},
        'warnings': warnings,
    }
    if 'sub_equipartition' in solution:
        adopted = _object(solution['sub_equipartition'], 'sub_equipartition')
        provenance['report_adopted_field_not_automatically_used'] = {
            key: adopted[key] for key in ('fraction_of_B_eq', 'B_uG', 'applied_to_analysis') if key in adopted}
    result = {'priors': {'magnetic_field_ug': {'distribution': 'lognormal',
                       'log_mean': math.log(field_median), 'log_sigma': combined_width}},
              'properties': {'field_basis': field_basis, 'field_input_median_uG': field_median,
                             'field_input_direct_uG': budgets[field_basis]['B_uG'] * scale,
                             'field_direct_uG': budgets[field_basis]['B_uG'],
                             'field_monte_carlo_uG': field_mc,
                             'native_field_budgets': budgets, 'geometry': geometry_properties,
                             'injection_index': {'value_positive': alpha, 'error': alpha_error,
                                                 'source': alpha_source, 'imported_as_prior': False}},
              'provenance': provenance}
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise PyRATSPropertyError('nonfinite or unsupported values in PyRATS provenance') from exc
    return result
