"""Auditable, class-aware source analyses and atomic analysis directories.

Classical DYNAGE diagnostics belong to active FR II components. Bent plumes
retain the separately labelled transport model. No shared report format is
allowed to imply that those models measure the same physical age.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .constants import MYR_S
from .fitting import FitError, fit_classic, fit_joint, parametric_bootstrap
from .io import ConfigurationError, load_configuration, normalized_input, _unique_object, _reject_json_constant
from .model import KDAForwardModel, ModelDomainError


def _read_json(path):
    try:
        raw = Path(path).read_bytes()
        document = json.loads(raw.decode("utf8"), object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigurationError(f"cannot read analysis input {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("analysis input must be a JSON object")
    return document, hashlib.sha256(raw).hexdigest()


def _count(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{label} must be an integer in [{minimum},{maximum}]")
    return value


def _sensitivity_cases(path):
    if path is None:
        return []
    document, _ = _read_json(path)
    if set(document) != {"cases"} or not isinstance(document["cases"], list) or len(document["cases"]) > 12:
        raise ConfigurationError("sensitivity requires only a cases list with at most 12 entries")
    names, checked = set(), []
    for case in document["cases"]:
        if not isinstance(case, dict) or set(case) - {"label", "model", "geometry"}:
            raise ConfigurationError("each sensitivity case accepts label, model, geometry")
        label = case.get("label")
        if not isinstance(label, str) or not label.strip() or label in names:
            raise ConfigurationError("sensitivity labels must be unique nonempty strings")
        names.add(label)
        model, geometry = case.get("model", {}), case.get("geometry", {})
        if not isinstance(model, dict) or set(model) - {"beta", "core_radius_kpc", "gamma_min", "gamma_max", "k_prime", "gamma_c", "gamma_b", "gamma_x", "equipartition_field_ug"}:
            raise ConfigurationError("unsupported model sensitivity parameter")
        if not isinstance(geometry, dict) or set(geometry) - {"inclination_deg", "projected_size_kpc", "axial_ratio"}:
            raise ConfigurationError("unsupported geometry sensitivity parameter")
        for key, value in {**model, **geometry}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigurationError(f"sensitivity {key} must be a finite number")
        checked.append({"label": label, "model": model, "geometry": geometry})
    return checked


def _fr2_document(path, *, method, pyrats_path, sensitivity_path, track_points, evolution_points, bootstrap, seed):
    from .dynage_diagnostics import (observed_spectrum_diagnostics, sample_intersection_tracks,
                                    evolution_track, summarize_fit_quality)
    observation, assumptions, bounds, runtime = load_configuration(path)
    if observation.radio_class != "fr2" or runtime["activity_state"] != "active":
        raise ConfigurationError("classical source analysis requires an active FR II component; use radio transport inference for WAT/NAT/FR I and the separate remnant/restarted routes")
    protected = {path}
    field = {"equation": "B = 0.4 B_eq", "fraction": .4, "scope": "homogeneous current source snapshot, gamma_b=gamma_c; evolution freezes the fitted partition, not a constant field"}
    automatic_sensitivity = []
    if pyrats_path is not None:
        from .pyrats_bridge import load_pyrats_properties
        properties = load_pyrats_properties(pyrats_path, expected_redshift=observation.redshift,
                                           systematic_log_sigma=0., field_basis="equipartition", field_scale=.4)
        p = properties["provenance"]["assumptions"]
        # These energy ratios are not silently equated: the chosen mapping
        # must already agree with the input model, and is explicitly recorded.
        for model_value, reference, label in [(assumptions.k_prime, p["kappa_proton_electron_energy_ratio"], "k_prime / PyRATS kappa"),
                                            (assumptions.gamma_min, p["gamma_min"], "gamma_min"),
                                            (assumptions.gamma_max, p["gamma_max"], "gamma_max")]:
            if not math.isclose(model_value, reference, rel_tol=1e-8, abs_tol=1e-12):
                raise ConfigurationError(f"field-anchor assumptions disagree ({label}); reconcile the component model and PyRATS assumptions explicitly")
        mc = properties["properties"]["field_monte_carlo_uG"]
        beq = mc["median"]
        if assumptions.equipartition_field_ug is not None and not math.isclose(assumptions.equipartition_field_ug, beq, rel_tol=1e-6):
            raise ConfigurationError("input equipartition field conflicts with PyRATS MC median")
        assumptions = replace(assumptions, equipartition_field_ug=beq)
        field.update(input="PyRATS MC median; conditional plug-in anchor", pyrats=properties,
                     composition_mapping="input k_prime is explicitly assumed to represent the reported non-radiating/electron energy ratio; equality alone does not prove identical plasma composition")
        automatic_sensitivity = [{"label": "PyRATS B_eq p16", "model": {"equipartition_field_ug": mc["p16"]}, "geometry": {}},
                                 {"label": "PyRATS B_eq p84", "model": {"equipartition_field_ug": mc["p84"]}, "geometry": {}}]
        protected.add(Path(pyrats_path).resolve())
    if assumptions.equipartition_field_ug is None:
        raise ConfigurationError("production DYNAGE-style analysis requires model.equipartition_field_ug or --pyrats-properties; the published minimum-energy reference does not satisfy B=0.4 B_eq")
    field["equipartition_field_ug"] = assumptions.equipartition_field_ug
    field["adopted_field_ug"] = .4 * assumptions.equipartition_field_ug
    model = KDAForwardModel(observation, assumptions)
    user_cases = _sensitivity_cases(sensitivity_path)
    cases = automatic_sensitivity + user_cases
    if len({case["label"] for case in cases}) != len(cases):
        raise ConfigurationError("sensitivity labels conflict with automatic field cases")
    requested = ["classic", "joint"] if method == "both" else [method]
    results, failures = {}, {}
    for name in requested:
        try:
            results[name] = (fit_classic if name == "classic" else fit_joint)(model, bounds)
        except (FitError, ModelDomainError, ArithmeticError, ValueError) as exc:
            failures[name] = str(exc)
    if not results:
        raise FitError("all requested fits failed: " + json.dumps(failures))
    for result in results.values():
        if runtime.get("core_subtracted") is not True:
            result.warnings.append("Input does not affirm core subtraction; flat-spectrum core emission can bias the fit.")
        if runtime.get("hotspots_subtracted") is not True:
            result.warnings.append("Input does not affirm hotspot subtraction; hotspot emission is outside the lobe integral.")
    # Under the externally fixed field, minimum Q*t is still a heuristic and
    # can prefer a poor spectrum. The weighted joint estimator is the primary
    # when both are requested; classical profiles remain fully reported.
    primary_name = "joint" if "joint" in results else "classic"
    primary = results[primary_name]
    warnings = list(primary.warnings)
    if failures:
        warnings.append("At least one requested fitting method failed; the report is a partial analysis, with failure reasons retained.")
    if bootstrap:
        primary.uncertainty = parametric_bootstrap(observation, lambda o: KDAForwardModel(o, assumptions),
            fit_classic if primary_name == "classic" else fit_joint, bounds, bootstrap, seed)
    runtime.update(bootstrap_samples_effective=bootstrap, seed_effective=seed, selected_method=primary_name)
    ages = np.unique(np.r_[np.geomspace(*bounds.age_myr, track_points), primary.age_myr])
    tracks = sample_intersection_tracks(model, primary, bounds, ages)
    evolution_ages = np.unique(np.r_[np.geomspace(max(primary.age_myr/10, .001), primary.age_myr*2, evolution_points), primary.age_myr])
    frequencies = [point.frequency_mhz for point in observation.spectrum]
    evolution = evolution_track(model, primary, evolution_ages, frequencies)
    if observation.geometry.mode == "two_sided":
        for row in evolution["tracks"]:
            if row.get("valid"):
                row["source_total_length_kpc"] = 2 * row["lobe_length_kpc"]
    quality = {name: summarize_fit_quality(model, result, bounds) for name, result in results.items()}
    spectrum = observed_spectrum_diagnostics(observation)
    if len(results) == 2:
        age_ratio = max(result.age_myr for result in results.values()) / min(result.age_myr for result in results.values())
        if age_ratio > 1.5:
            warnings.append("Classic and joint ages differ by more than 50%; method dependence must be assessed before interpreting a source age.")
    sensitivities = []
    if sensitivity_path is not None:
        protected.add(Path(sensitivity_path).resolve())
    for case in cases:
        try:
            varied_model = KDAForwardModel(replace(observation, geometry=replace(observation.geometry, **case["geometry"])), replace(assumptions, **case["model"]))
            fitted = fit_joint(varied_model, bounds)
            sensitivities.append({"label": case["label"], "status": "conditional_joint_fit", "age_myr": fitted.age_myr,
                "alpha_inj": fitted.alpha_inj, "jet_power_w": fitted.jet_power_w, "rho0_kg_m3": fitted.rho0_kg_m3,
                "model_overrides": case["model"], "geometry_overrides": case["geometry"], "warnings": fitted.warnings})
        except (ValueError, ArithmeticError, FitError, ModelDomainError) as exc:
            sensitivities.append({"label": case["label"], "status": "failed", "reason": str(exc),
                "model_overrides": case["model"], "geometry_overrides": case["geometry"]})
    if not cases:
        warnings.append("No assumption-sensitivity cases were requested; reported fit errors do not cover fixed-model systematics.")
    uncertainty = primary.uncertainty
    summary = {"point_estimate_myr": primary.age_myr, "mean_myr": None, "median_myr": None,
               "scope": "single global FR II model age; not a mean of spatial plasma ages"}
    if uncertainty and uncertainty.get("sample_summaries"):
        summary.update(mean_myr=uncertainty["sample_summaries"]["age_myr"]["mean"], median_myr=uncertainty["sample_summaries"]["age_myr"]["median"])
    else:
        summary["distribution_status"] = "No sufficiently reliable Monte Carlo age distribution; mean/median uncertainty summaries are withheld."
    document = {"schema_version": "fr2dynage-source-analysis-1", "kind": "fr2_dynage", "source": {"name": observation.name, "redshift": observation.redshift, "radio_class": observation.radio_class},
        "primary_method": primary_name, "methods": {name: result.to_dict() for name, result in results.items()},
        "method_failures": failures, "source_age_summary": summary, "field_constraint": field,
        "primary_selection": "weighted joint spectrum fit when available; classical minimum-Q*t is retained as a separate physical-heuristic comparison",
        "diagnostics": {"observed_spectrum": spectrum, "intersections": tracks, "alpha_profile": results.get("classic", primary).diagnostics.get("classic_alpha_scan", []), "evolution": evolution},
        "quality": quality, "sensitivity": sensitivities,
        "sensitivity_interpretation": "conditional alternative assumptions; neither likelihood confidence limits nor independent measurements",
        "normalized_input": normalized_input(observation, assumptions, bounds, runtime), "input_provenance": runtime,
        "warnings": warnings + ["Classical Delta and minimum Q*t are physical heuristics, not statistical confidence criteria.",
            "The explicit fixed-field snapshot is a model variant of published DYNAGE, not the original minimum-energy partition.",
            "The current-field anchor is supplied, not derived from the KDA fit; joint field/radio-data covariance and geometry/model systematics require separate treatment."]}
    if sensitivity_path is not None:
        sensitivity_input, digest = _read_json(sensitivity_path)
        document["sensitivity_input"] = {"path": str(Path(sensitivity_path).resolve()), "sha256": digest, "document": sensitivity_input}
    return document, protected, None


def _radio_document(path, *, budget_path):
    from .radio_inference import compute_radio_inference, KPC_KMS_TO_MYR
    from .resolved_products import parse_resolved_settings
    config, _ = _read_json(path)
    resolved_settings = parse_resolved_settings(config.get("resolved_analysis"))
    fixed_field_t = None
    if 'fixed_magnetic_field' in config:
        fixed_spec = config['fixed_magnetic_field']
        if (not isinstance(fixed_spec, dict) or set(fixed_spec) != {'value_t', 'description'} or
            not isinstance(fixed_spec.get('description'), str) or not fixed_spec['description'].strip()):
            raise ConfigurationError('fixed_magnetic_field requires value_t and a nonempty description')
        fixed_field_t = fixed_spec['value_t']
        if (isinstance(fixed_field_t, bool) or not isinstance(fixed_field_t, (int, float)) or
            not math.isfinite(fixed_field_t) or not 1e-12 <= fixed_field_t <= 1e-7):
            raise ConfigurationError('fixed_magnetic_field.value_t must be a finite number in [1e-12,1e-7] T')
    budget = None
    budget_validation = None
    budget_record = None
    if budget_path is not None:
        budget, budget_hash = _read_json(budget_path)
        budget_record = {'role': 'adopted_budget', 'path': str(Path(budget_path).resolve()), 'sha256': budget_hash}
        values = budget.get("budget")
        if not isinstance(values, dict):
            raise ConfigurationError("adopted budget requires a budget object")
        for key in ("fraction_of_B_eq", "B_uG", "B_eq_uG", "total_energy_J", "pressure_Pa"):
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"adopted budget {key} must be a finite positive number")
        config, _ = _read_json(path)
        field_spec = config.get("pyrats_properties")
        field_path = field_spec.get("path") if isinstance(field_spec, dict) else None
        if not isinstance(field_path, str):
            raise ConfigurationError("adopted budget requires a PyRATS report path")
        report, report_hash = _read_json(Path(path).parent / field_path)
        if budget.get("input_sha256") != report_hash:
            raise ConfigurationError("adopted budget must match the PyRATS input hash")
        if fixed_field_t is None and (values["fraction_of_B_eq"] != .4 or
            not math.isclose(values["B_uG"], .4*values["B_eq_uG"], rel_tol=1e-10)):
            raise ConfigurationError("adopted budget must match the fixed B=0.4 B_eq prescription")
        if fixed_field_t is not None and not math.isclose(values['B_uG']*1e-10, fixed_field_t, rel_tol=1e-12):
            raise ConfigurationError('adopted budget does not match the requested fixed magnetic field')
        from .reliability import validate_adopted_budget
        try:
            budget_validation = validate_adopted_budget(budget, report, fixed_field_t=fixed_field_t)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f'adopted budget validation failed: {exc}') from exc
    result = compute_radio_inference(path)
    if result.metadata.get("field_prescription") is None:
        raise ConfigurationError("observed-source analysis requires the PyRATS B=0.4 B_eq prescription or an explicitly supplied fixed magnetic field")
    if fixed_field_t is not None:
        adopted = result.metadata['field_prescription']
        field_draws = np.asarray(result.samples['magnetic_field_ug'])
        if (adopted.get('kind') != 'user_fixed' or
            not math.isclose(adopted.get('value_t', 0.), fixed_field_t, rel_tol=1e-12) or
            not field_draws.size or not np.all(np.isfinite(field_draws)) or
            not np.all(field_draws == field_draws.flat[0]) or
            not math.isclose(float(field_draws.flat[0])*1e-10, fixed_field_t, rel_tol=1e-12)):
            raise ConfigurationError('inference did not preserve the requested exact fixed magnetic field')
    profiles = []
    for branch, summary in result.metadata["branches"].items():
        distances = np.linspace(0., summary["traced_projected_length_kpc"], 50)
        eta = result.samples.get("deceleration", np.full_like(result.samples["magnetic_field_ug"], result.metadata["clock"]["model"].get("deceleration", 0.)))
        age = KPC_KMS_TO_MYR * (distances[None, :] + eta[:, None]*distances[None, :]**2/(2*distances[-1])) / result.samples[f"flow_speed_km_s_branch_{branch}"][:, None]
        quantiles = np.quantile(age, [.16, .5, .84], axis=0)
        profiles.append({"branch": int(branch), "distance_kpc": distances.tolist(), "age_p16_myr": quantiles[0].tolist(),
                         "age_p50_myr": quantiles[1].tolist(), "age_p84_myr": quantiles[2].tolist()})
    protected = set(result.input_paths)
    if budget_path is not None:
        provenance = (result.metadata.get("pyrats_properties") or {}).get("provenance", {})
        if budget["input_sha256"] != provenance.get("sha256"):
            raise ConfigurationError("PyRATS report changed during analysis")
        protected.add(Path(budget_path).resolve())
        result.metadata['input_files'].append(budget_record)
    document = {"schema_version": "fr2dynage-source-analysis-1", "kind": "radio_transport", "source": result.metadata["source"],
        "primary_method": result.metadata["clock"]["model"]["kind"], "radio_result": result.metadata, "field_constraint": result.metadata.get("field_prescription"),
        "map_products": {"fits": "transport.fits", "preview": "transport.png", "preview_pdf": "transport.pdf",
                         "preview_svg": "transport.svg", "posterior": "posterior.npz"},
        "diagnostics": {"regional_spectra": result.metadata["fitted_regional_spectra"], "transport_profiles": {"branches": profiles}},
        "quality": {"status": result.metadata["status"], "sampling": result.metadata["sampling"]["status"]},
        "adopted_source_budget": budget,
        "unavailable_classical_parameters": {"jet_power": "not identified by the advective spectral model", "rho0": "no ambient-density dynamics are fitted", "global_source_age": "plasma transport time is not a global episode age"},
        "warnings": result.metadata["warnings"] + ["Classical FR II expansion/injection-power fits are not applied to bent plumes; unavailable physical parameters are not manufactured."]}
    document["source_coverage"] = result.metadata.get("source_coverage", {})
    if result.metadata.get("core") is not None:
        document["diagnostics"]["core_spectrum"] = result.metadata["core"]["central_spectrum"]
    if resolved_settings is not None:
        from .resolved_products import compute_resolved_products
        result.resolved_results = compute_resolved_products(result, resolved_settings)
        document["resolved_spectral_analysis"] = {name: {**fitted["metadata"], "products": {
            "fits": "resolved_"+name+".fits", "metadata": "resolved_"+name+".json",
            "preview": "resolved_"+name+".png", "pdf": "resolved_"+name+".pdf", "svg": "resolved_"+name+".svg"}}
            for name, fitted in result.resolved_results.items()}
    from .reliability import radio_parameter_quality
    document['parameter_quality'] = radio_parameter_quality(result.metadata, result.resolved_results, budget_validation)
    document['input_integrity_records'] = [
        {'path': result.metadata['configuration_path'], 'sha256': result.metadata['configuration_sha256']},
        *result.metadata['input_files']]
    return document, protected, result


def analyze_source(input_path, output_dir, *, method="both", pyrats_properties=None, budget_path=None,
                   sensitivity_path=None, track_points=17, evolution_points=17,
                   bootstrap=0, seed=42, overwrite=False):
    """Create a complete reviewable analysis directory or leave it untouched."""
    from .analysis_products import write_analysis_products
    try:
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError("analysis figures require python -m pip install 'pyradage[analysis]'") from exc
    path, target = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if method not in {"classic", "joint", "both"}:
        raise ConfigurationError("analysis method must be classic, joint or both")
    _count(track_points, "track_points", 9, 101); _count(evolution_points, "evolution_points", 5, 101)
    _count(bootstrap, "bootstrap", 0, 10000); _count(seed, "seed", 0, 2**32-1)
    raw, initial_hash = _read_json(path)
    initial_records = [{'path': str(path), 'sha256': initial_hash}]
    from .reliability import sha256_file, verify_input_records
    for optional in (pyrats_properties, budget_path, sensitivity_path):
        if optional is not None:
            initial_records.append({'path': str(Path(optional).expanduser().resolve()), 'sha256': sha256_file(Path(optional).expanduser())})
    if path.is_relative_to(target):
        raise ConfigurationError("analysis directory must not contain or replace its input")
    for optional in (pyrats_properties, budget_path, sensitivity_path):
        if optional is not None and Path(optional).expanduser().resolve().is_relative_to(target):
            raise ConfigurationError("analysis directory must not contain or replace a referenced input")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; use a new directory or --overwrite")
        previous, _ = _read_json(target/"analysis.json")
        if previous.get("schema_version") != "fr2dynage-source-analysis-1":
            raise ConfigurationError("only a recognized source-analysis directory may be replaced")
    if raw.get("schema_version") == "fr2dynage-radio-1":
        if pyrats_properties or sensitivity_path or bootstrap:
            raise ConfigurationError("radio transport uses its own PyRATS/priors/posterior configuration; FR II fitting options do not apply")
        if "pyrats_properties" not in raw and 'fixed_magnetic_field' not in raw:
            raise ConfigurationError("source analysis requires pyrats_properties for B=0.4 B_eq or an explicitly supplied fixed_magnetic_field")
        document, protected, radio = _radio_document(path, budget_path=budget_path)
    else:
        if budget_path is not None:
            raise ConfigurationError("--pyrats-budget belongs to the radio transport analysis")
        document, protected, radio = _fr2_document(path, method=method, pyrats_path=pyrats_properties, sensitivity_path=sensitivity_path,
            track_points=track_points, evolution_points=evolution_points, bootstrap=bootstrap, seed=seed)
    if any(p == target or p.is_relative_to(target) for p in protected):
        raise ConfigurationError("analysis output must not contain or overwrite any referenced input")
    from . import __version__
    integrity_records = [*initial_records, *document.get('input_integrity_records', [])]
    input_digests = verify_input_records(integrity_records)
    document['input_integrity'] = {'status': 'verified_after_computation', 'sha256': input_digests}
    document.update(created_utc=datetime.now(timezone.utc).isoformat(), software_version=__version__,
                    protected_inputs=[str(p) for p in sorted(protected)],
                    analysis_options={"method": method, "track_points": track_points, "evolution_points": evolution_points,
                                      "bootstrap": bootstrap, "seed": seed})
    json.dumps(document, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="."+target.name+".", dir=target.parent))
    try:
        if radio is not None:
            from .radio_inference import write_radio_products
            write_radio_products(radio, stage/"transport.fits", stage/"radio-summary.json", samples_path=stage/"posterior.npz", preview_path=stage/"transport.png")
            if radio.resolved_results is not None:
                from .resolved_products import write_resolved_products
                write_resolved_products(radio.resolved_results, radio, stage)
        products = write_analysis_products(document, stage, overwrite=False, protected_paths=protected)
        verify_input_records(integrity_records)
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        (stage/"run_manifest.json").write_text(json.dumps({"schema_version": "fr2dynage-analysis-manifest-2",
            "files": [str(p.relative_to(stage)) for p in files],
            "sha256": {str(p.relative_to(stage)): sha256_file(p) for p in files},
            "input_sha256": input_digests,
            "scope": "output hashes exclude this manifest; input hashes checked after computation and immediately before publication"}, indent=2)+"\n")
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"{target} appeared during analysis; it was left untouched")
            current, _ = _read_json(target/"analysis.json")
            if current.get("schema_version") != "fr2dynage-source-analysis-1":
                raise ConfigurationError("output changed during analysis; refusing replacement")
            backup = target.with_name(target.name+".backup-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            os.rename(target, backup)
            try:
                os.rename(stage, target)
            except OSError:
                os.rename(backup, target)
                raise
        else:
            os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return document, products
