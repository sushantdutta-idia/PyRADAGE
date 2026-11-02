"""Scientific reporting gates and verifiable input/output provenance.

Passing these gates is conditional model support, never a claim of physical
truth, independent validation, or calibrated coverage of model systematics.
"""
from pathlib import Path
import hashlib
import json
import math
import re


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_records(records):
    """Reject an input changed after it was consumed, including duplicate reads."""
    expected = {}
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get('path'), str) or
            not isinstance(record.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise ValueError('input records require a path and SHA-256 digest')
        path, digest = str(Path(record['path']).resolve()), record['sha256']
        if path in expected and digest != expected[path]:
            raise ValueError(f'input changed between reads: {path}')
        expected[path] = digest
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f'input changed during analysis: {path}')
    return expected


def verify_analysis_manifest(folder, *, check_inputs=False):
    """Verify all recorded output hashes without executing result content."""
    from .io import _unique_object, _reject_json_constant
    root = Path(folder).expanduser().resolve()
    document = json.loads((root/'run_manifest.json').read_text(),
                          object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    if not isinstance(document, dict) or document.get('schema_version') != 'fr2dynage-analysis-manifest-2':
        raise ValueError('hash verification requires a version-2 analysis manifest')
    hashes = document.get('sha256')
    if (not isinstance(hashes, dict) or not hashes or
        not isinstance(document.get('files'), list) or
        any(not isinstance(path, str) for path in document['files']) or
        len(document['files']) != len(set(document['files'])) or set(document['files']) != set(hashes)):
        raise ValueError('manifest file list and hashes must agree')
    for relative, digest in hashes.items():
        path = (root/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root/'run_manifest.json':
            raise ValueError('manifest output paths must stay inside the analysis directory')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or sha256_file(path) != digest:
            raise ValueError(f'output checksum mismatch: {relative}')
    if check_inputs:
        inputs = document.get('input_sha256')
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError('manifest does not contain input hashes')
        verify_input_records([{'path': path, 'sha256': digest} for path, digest in inputs.items()])
    return {'passed': True, 'output_files_verified': len(hashes), 'inputs_checked': bool(check_inputs),
            'scope': 'integrity against the supplied manifest; not scientific validation or authentication of the manifest'}


def radio_parameter_quality(metadata, resolved=None, budget_validation=None):
    """Keep numerical diagnostic summaries separate from supported estimates."""
    reasons = []
    if metadata.get('fit', {}).get('adequate_at_threshold_4') is not True:
        reasons.append('radiative/transport model adequacy not established')
    if metadata.get('sampling', {}).get('status') != 'replicate_consistent':
        reasons.append('independent sampling agreement not established')
    if metadata.get('identifiability', {}).get('curvature') != 'required_by_heuristic':
        reasons.append('spectral ageing curvature not established')
    distance = metadata.get('distance_scale_consistency', {})
    if distance.get('status') == 'inconsistent':
        reasons.append('transport and field-input distance scales disagree')
    supported = not reasons
    spatial = metadata.get('map_statistics', {})
    parameters = {}
    fixed_field = metadata.get('field_prescription', {}).get('kind') == 'user_fixed' if metadata.get('field_prescription') else False
    for name, summary in metadata.get('posterior_parameters', {}).items():
        field = name == 'magnetic_field_ug'
        parameters[name] = {
            'status': ('user_fixed_input' if fixed_field else 'conditional_field_input') if field else 'model_and_prior_conditioned' if supported else 'diagnostic_only',
            'estimate': summary if (field or supported) else None,
            'diagnostic_summary': summary,
            'reasons': (['user-supplied fixed magnetic field; no uncertainty was supplied or inferred'] if fixed_field else ['same-radio-data PyRATS input; not an independent field measurement']) if field else reasons.copy()}
    spectral = {}
    for name, fitted in (resolved or {}).items():
        info = fitted.get('metadata', fitted)
        quality = info.get('whole_detected_source_summary', {})
        spectral[name] = quality
    budget = dict(budget_validation or {'status': 'not_supplied'})
    pyrats = metadata.get('pyrats_properties') or {}
    upstream = pyrats.get('properties', {}).get('injection_index', {}).get('value_positive')
    posterior_alpha = metadata.get('posterior_parameters', {}).get('alpha_inj', {})
    interval = posterior_alpha.get('credible_interval_68')
    if upstream is not None and interval is not None:
        budget['injection_spectrum_consistency'] = {
            'upstream_alpha': upstream, 'transport_alpha_68': interval,
            'status': 'upstream_index_outside_transport_68' if not interval[0] <= upstream <= interval[1] else 'overlapping_index',
            'interpretation': 'descriptive assumption comparison using dependent radio data, not an independent significance test; the nominal budget is not refitted to the transport/spectral electron population'}
    return {
        'schema_version': 'fr2dynage-parameter-quality-1',
        'scope': 'conditional reporting gates, not independent physical validation or a coverage guarantee',
        'transport': {
            'status': 'model_and_prior_conditioned' if supported else 'diagnostic_only',
            'reasons': reasons,
            'mean_myr': spatial.get('area_weighted_mean') if supported else None,
            'median_myr': spatial.get('area_weighted_median') if supported else None,
            'diagnostic_map_statistics': spatial},
        'parameters': parameters, 'spectral_models': spectral,
        'global_dynamical_age_myr': {'status': 'not_identifiable', 'mean': None, 'median': None},
        'source_budget': budget}


def validate_adopted_budget(document, report, *, fixed_field_t=None):
    """Check nominal PyRATS accounting at the explicitly adopted field.

    This checks SI units, field basis, composition, volume and energy/pressure
    closure. It does not verify the flux normalization or measure the assumed
    particle content, line-of-sight volume, or aged electron distribution.
    """
    from .constants import MU_0
    values = document.get('budget')
    if not isinstance(values, dict):
        raise ValueError('adopted budget requires a budget object')
    required = ('fraction_of_B_eq', 'B_uG', 'B_T', 'B_eq_uG', 'B_eq_T',
                'electron_energy_density_J_m3', 'particle_energy_density_J_m3',
                'magnetic_energy_density_J_m3', 'total_energy_density_J_m3',
                'total_energy_J', 'pressure_Pa')
    for key in required:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'adopted budget {key} must be finite and positive')
    equip = report['self_consistent_solution']['equipartition']
    inputs, direct = equip['inputs'], equip['direct']['equipartition']
    volume = inputs['volume_m3']
    kappa = inputs['kappa_proton_electron_energy_ratio']
    if fixed_field_t is not None:
        if (isinstance(fixed_field_t, bool) or not isinstance(fixed_field_t, (int, float)) or
            not math.isfinite(fixed_field_t) or not 1e-12 <= fixed_field_t <= 1e-7):
            raise ValueError('fixed field must be a finite number in [1e-12,1e-7] T')
        field_checks = {
            'actual B/B_eq ratio': (values['fraction_of_B_eq'], values['B_uG']/values['B_eq_uG']),
            'requested fixed magnetic field': (values['B_T'], fixed_field_t)}
        basis = 'user-supplied fixed magnetic field; PyRATS B_eq retained only as a comparison'
    else:
        field_checks = {
            'fixed B/B_eq': (values['fraction_of_B_eq'], .4),
            'adopted field': (values['B_uG'], .4*values['B_eq_uG'])}
        basis = 'nominal deterministic PyRATS equipartition field times 0.4'
    checks = {
        **field_checks,
        'nominal equipartition basis': (values['B_eq_uG'], direct['B_uG']),
        'adopted field SI units': (values['B_T'], values['B_uG']*1e-10),
        'equipartition field SI units': (values['B_eq_T'], values['B_eq_uG']*1e-10),
        'magnetic energy density': (values['magnetic_energy_density_J_m3'], values['B_T']**2/(2*MU_0)),
        'particle composition': (values['particle_energy_density_J_m3'], (1+kappa)*values['electron_energy_density_J_m3']),
        'total energy density': (values['total_energy_density_J_m3'], values['particle_energy_density_J_m3']+values['magnetic_energy_density_J_m3']),
        'volume times energy density': (values['total_energy_J'], volume*values['total_energy_density_J_m3']),
        'relativistic pressure': (values['pressure_Pa'], values['total_energy_density_J_m3']/3)}
    for key, (actual, expected) in checks.items():
        tolerance = 1e-12 if key == 'requested fixed magnetic field' else 1e-5
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0.):
            raise ValueError(f'adopted budget fails {key} consistency')
    return {'status': 'accounting_verified_conditional_budget', 'checks': list(checks),
            'field_basis': basis, 'adopted_field_t': values['B_T'], 'adopted_field_ug': values['B_uG'],
            'fraction_of_nominal_B_eq': values['fraction_of_B_eq'],
            'uncertainty': document.get('uncertainty', 'not supplied'),
            'scope': 'SI/field/composition/volume/energy/pressure closure only; flux normalization and physical assumptions are not independently verified'}
