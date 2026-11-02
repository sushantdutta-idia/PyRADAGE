"""Command-line interface for validation, prediction, and fitting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .constants import MYR_S
from .fitting import FitError, fit_classic, fit_joint, parametric_bootstrap
from .io import (
    ConfigurationError,
    build_restarted_result_document,
    build_result_document,
    load_configuration,
    normalized_input,
    write_json,
)
from .model import KDAForwardModel, ModelDomainError
from .remnant import (
    KDAExtForwardModel,
    RemnantFitBounds,
    RemnantFitError,
    fit_remnant_joint,
)
from .restarted import RestartedFitError, fit_restarted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyradage",
        description=(
            "FR II dynamical fitting, constrained FITS clocks, and "
            "prior-conditioned steady-flow inference from radio images"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser('gui',help='open the image/redshift graphical wizard')
    automatic=subparsers.add_parser('auto',help='automatically prepare radio images and produce one combined radiative-age PDF/PNG')
    automatic.add_argument('images',type=Path,nargs='+')
    automatic.add_argument('--redshift',type=float,required=True)
    automatic.add_argument('--output-dir',type=Path,required=True)
    automatic.add_argument('--field-t',type=float,help='optional explicit field in tesla; default 0.4 B_eq through PyRATS')
    automatic.add_argument('--pyrats-script',type=Path,help='external PyRATS.py; otherwise use environment/PATH discovery')
    automatic.add_argument('--pyrats-python',help='Python executable with PyRATS/pysynch/GSL installed')
    automatic.add_argument('--frequencies-mhz',type=float,nargs='+',help='only needed when FITS observing-frequency metadata are unavailable/ambiguous')
    automatic.add_argument('--calibration-fraction',type=float,default=.05,help='assumed per-band calibration fraction, default 0.05')
    automatic.add_argument('--seed',type=int,default=42)

    validate = subparsers.add_parser("validate", help="validate and normalize an input JSON file")
    validate.add_argument("input", type=Path)
    validate.add_argument("--show-normalized", action="store_true")

    fit = subparsers.add_parser("fit", help="fit age, injection index, jet power, and density")
    fit.add_argument("input", type=Path)
    fit.add_argument(
        "--method",
        choices=("auto", "classic", "joint"),
        default="auto",
        help="auto selects classic for active sources and joint KDA-EXT for remnants",
    )
    fit.add_argument("--output", type=Path, help="write the complete result document to JSON")
    fit.add_argument("--bootstrap", type=int, help="override runtime.bootstrap_samples")
    fit.add_argument("--seed", type=int, help="override runtime.seed")
    fit.add_argument("--print-json", action="store_true", help="also print the result document")

    predict = subparsers.add_parser("predict", help="predict powers for supplied physical parameters")
    predict.add_argument("input", type=Path)
    predict.add_argument("--age-myr", type=float, required=True)
    predict.add_argument("--alpha-inj", type=float, required=True)
    predict.add_argument("--jet-power-w", type=float, required=True)
    phase = predict.add_mutually_exclusive_group()
    phase.add_argument(
        "--active-duration-myr",
        type=float,
        help="remnant only: age when fresh-particle supply to the lobe stopped",
    )
    phase.add_argument(
        "--active-fraction",
        type=float,
        help="remnant only: active duration divided by total age",
    )

    restarted = subparsers.add_parser(
        "fit-restarted",
        help="fit separately measured outer-remnant and inner-active components",
    )
    restarted.add_argument("outer", type=Path, help="outer remnant component JSON")
    restarted.add_argument("inner", type=Path, help="inner active component JSON")
    restarted.add_argument(
        "--old-jet-speed-c",
        type=float,
        default=1.0,
        help="old jet bulk speed as a fraction of c for the duty-cycle correction",
    )
    restarted.add_argument(
        "--inner-jet-power-w",
        type=float,
        help=(
            "external one-jet power constraint for a featureless inner spectrum"
        ),
    )
    restarted.add_argument(
        "--allow-underconstrained-inner",
        action="store_true",
        help=(
            "return a conditional inner age without an external power constraint"
        ),
    )
    restarted.add_argument(
        "--output",
        type=Path,
        help="write the complete two-component result document to JSON",
    )
    restarted.add_argument(
        "--print-json",
        action="store_true",
        help="also print the complete result document",
    )
    mapping = subparsers.add_parser("map-age", help="map conditional expansion ages or flow travel times from aligned FITS images")
    mapping.add_argument("input", type=Path, help="strict map configuration JSON")
    mapping.add_argument("--output", type=Path, required=True, help="output FITS with WCS, diagnostics, and provenance")
    mapping.add_argument("--preview", type=Path, help="optional PNG preview with publication PDF/SVG companions")
    mapping.add_argument("--overwrite", action="store_true", help="replace existing output products")
    map_validate = subparsers.add_parser("validate-map", help="check map configuration, WCS, beams, masks, and supplied constraints")
    map_validate.add_argument("input", type=Path)
    analysis = subparsers.add_parser("analyze", aliases=["analyze-source"], help="complete DYNAGE-style FR II or radio-transport analysis with plots, tables and quality checks")
    analysis.add_argument("input", type=Path)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--method", choices=("classic", "joint", "both"), default="both")
    analysis.add_argument("--pyrats-properties", type=Path, help="FR II field-anchor report; model particle/electron-limit assumptions must agree")
    analysis.add_argument("--pyrats-budget", type=Path, help="radio transport: verified PyRATS budget at the adopted field")
    analysis.add_argument("--sensitivity", type=Path, help="explicit FR II assumption-variation cases JSON")
    analysis.add_argument("--track-points", type=int, default=17)
    analysis.add_argument("--evolution-points", type=int, default=17)
    analysis.add_argument("--bootstrap", type=int, default=0)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--overwrite", action="store_true", help="replace a recognized analysis directory, preserving its previous version as a backup")
    verification = subparsers.add_parser('verify-analysis', help='verify recorded output hashes and optionally original input hashes')
    verification.add_argument('directory', type=Path)
    verification.add_argument('--check-inputs', action='store_true')
    radio = subparsers.add_parser("infer-radio", help="infer prior-conditioned steady-flow transport ages from multi-frequency radio images")
    radio.add_argument("input", type=Path, help="fr2dynage-radio-1 configuration")
    radio.add_argument("--output", type=Path, required=True, help="posterior transport-time FITS")
    radio.add_argument("--summary", type=Path, required=True, help="JSON summaries, diagnostics and provenance")
    radio.add_argument("--samples", type=Path, help="optional compressed NPZ posterior draws")
    radio.add_argument("--preview", type=Path, help="optional WCS PNG preview with publication PDF/SVG companions")
    radio.add_argument("--overwrite", action="store_true")
    calibration = subparsers.add_parser("calibrate-radio-priors", help="build class-specific predictive priors from a reference-source catalogue")
    calibration.add_argument("input", type=Path)
    calibration.add_argument("--target-name", required=True, help="exclude this source from its own calibration")
    calibration.add_argument("--radio-class", required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--allow-synthetic", action="store_true", help="demonstration only: include explicitly synthetic sources")
    calibration.add_argument("--allow-dependent", action="store_true", help="include model-dependent constraints, preserving their provenance")
    calibration.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("classes", help="list available class-aware analysis routes")
    return parser


def _summary(result) -> str:
    max_quad = max(
        result.diagnostics.get("quadrature_relative_error_order_vs_double", [0.0]),
        default=0.0,
    )
    lines = [
        f"{result.source_name}: {result.method}",
        f"  age                 = {result.age_myr:.6g} Myr",
        f"  alpha_inj           = {result.alpha_inj:.6g}",
        f"  p                    = {2.0 * result.alpha_inj + 1.0:.6g}",
        f"  log10 Q_jet [W]      = {math.log10(result.jet_power_w):.6f} (one jet)",
        f"  log10 rho0 [kg/m^3] = {math.log10(result.rho0_kg_m3):.6f} (conditional)",
        f"  {result.objective_name} = {result.objective:.6g}",
    ]
    if result.activity_history:
        lines.extend(
            [
                f"  active duration      = "
                f"{result.activity_history['active_duration_myr']:.6g} Myr",
                f"  remnant duration     = "
                f"{result.activity_history['remnant_duration_myr']:.6g} Myr",
                f"  active fraction      = "
                f"{result.activity_history['active_fraction']:.6g}",
            ]
        )
    if max_quad:
        lines.append(f"  max quadrature check = {max_quad:.3g}")
    if result.uncertainty:
        successful = result.uncertainty.get("successful_samples")
        requested = result.uncertainty.get("requested_samples")
        lines.append(f"  bootstrap            = {successful}/{requested} successful")
    if result.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _run_validate(args: argparse.Namespace) -> int:
    observation, assumptions, bounds, runtime = load_configuration(args.input)
    activity_state = runtime["activity_state"]
    if activity_state == "remnant" and len(observation.spectrum) < 5:
        raise ConfigurationError(
            "the four-parameter remnant fit requires at least five detected spectrum points"
        )
    model = (
        KDAExtForwardModel(observation, assumptions)
        if activity_state == "remnant"
        else KDAForwardModel(observation, assumptions)
    )
    free_parameters = 4 if activity_state == "remnant" else 3
    print(f"Valid input: {observation.name}")
    print(f"  activity state = {activity_state}")
    print(f"  {len(observation.spectrum)} rest-frame power points")
    print(
        f"  fit parameters = {free_parameters}; "
        f"residual dof = {len(observation.spectrum) - free_parameters}"
    )
    print(f"  one-lobe length = {observation.geometry.lobe_length_m / 3.085677581491367e19:.6g} kpc")
    print(f"  R_T = {observation.geometry.lobe_axial_ratio:.6g}")
    print(f"  P_hc = {model.pressure_ratio:.6g}; c1 = {model.c1:.6g}; c3 = {model.c3:.6g}")
    print(f"  input SHA-256 = {runtime['input_sha256']}")
    if args.show_normalized:
        print(
            json.dumps(
                normalized_input(observation, assumptions, bounds, runtime),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


def _append_quality_warnings(result, observation, assumptions) -> None:
    """Attach common input/model-domain warnings to any component fit."""

    if getattr(result, "_quality_warnings_added", False):
        return
    frequency_span = (
        max(point.frequency_mhz for point in observation.spectrum)
        / min(point.frequency_mhz for point in observation.spectrum)
    )
    if frequency_span < 10.0:
        result.warnings.append(
            "The radio spectrum spans less than one decade; age/injection-index "
            "degeneracy may be severe."
        )
    if observation.geometry.lobe_length_m <= assumptions.core_radius_m:
        result.warnings.append(
            "The lobe is not beyond the adopted core radius; the power-law "
            "atmosphere is being extrapolated outside its stated domain."
        )
    head_speed = result.derived["instantaneous_head_speed_over_c"]
    if head_speed >= 1.0:
        result.warnings.append(
            "The fitted instantaneous head speed is superluminal, so the "
            "model/solution is unphysical."
        )
    elif head_speed >= 0.3:
        result.warnings.append(
            "The fitted instantaneous head speed exceeds 0.3c, a common "
            "plausibility ceiling for extended FR II sources."
        )
    if observation.geometry.lobe_axial_ratio < 1.0:
        result.warnings.append(
            "The derived lobe axial ratio R_T is below 1; the empirical elongated-"
            "FR-II pressure-ratio calibration may be unreliable."
        )


def _append_subtraction_warnings(result, runtime, component: str = "Input") -> None:
    if runtime.get("core_subtracted") is not True:
        result.warnings.append(
            f"{component} does not affirm core subtraction; flat-spectrum core "
            "emission can bias the fit."
        )
    if runtime.get("hotspots_subtracted") is not True:
        result.warnings.append(
            f"{component} does not affirm hotspot subtraction; hotspot emission "
            "is outside the lobe integral."
        )


def _run_fit(args: argparse.Namespace) -> int:
    observation, assumptions, bounds, runtime = load_configuration(args.input)
    activity_state = runtime["activity_state"]
    bootstrap_samples = runtime["bootstrap_samples"] if args.bootstrap is None else args.bootstrap
    seed = runtime["seed"] if args.seed is None else args.seed
    if bootstrap_samples < 0:
        raise ConfigurationError("--bootstrap cannot be negative")
    if seed < 0:
        raise ConfigurationError("--seed cannot be negative")
    if bootstrap_samples:
        if activity_state != "active":
            raise ConfigurationError(
                "remnant bootstrap propagation is not implemented in this "
                "release; use the reported local covariance and assumption sweeps"
            )
        if any(point.fractional_error is None for point in observation.spectrum):
            raise ConfigurationError("bootstrap requires fractional_error for every spectrum point")
    if activity_state == "active":
        selected_method = "classic" if args.method == "auto" else args.method
        model = KDAForwardModel(observation, assumptions)
        fit_function = fit_classic if selected_method == "classic" else fit_joint
        result = fit_function(model, bounds)
    else:
        if args.method == "classic":
            raise ConfigurationError(
                "classic DYNAGE has no shutdown parameter and cannot fit a "
                "remnant; use --method auto or joint"
            )
        selected_method = "remnant_joint"
        result = fit_remnant_joint(
            observation,
            assumptions,
            bounds,
            active_fraction_bounds=runtime["active_fraction_bounds"],
        )
    _append_subtraction_warnings(result, runtime)
    _append_quality_warnings(result, observation, assumptions)
    if bootstrap_samples:
        result.uncertainty = parametric_bootstrap(
            observation,
            lambda varied: KDAForwardModel(varied, assumptions),
            fit_function,
            bounds,
            bootstrap_samples,
            seed,
        )
        result.warnings.extend(result.uncertainty.get("warnings", []))
    runtime["selected_method"] = selected_method
    runtime["bootstrap_samples_effective"] = bootstrap_samples
    runtime["seed_effective"] = seed
    document = build_result_document(result, observation, assumptions, bounds, runtime)
    print(_summary(result))
    if args.output:
        write_json(args.output, document)
        print(f"Result written to {args.output.resolve()}")
    if args.print_json or not args.output:
        print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    observation, assumptions, _, runtime = load_configuration(args.input)
    activity_state = runtime["activity_state"]
    frequencies = [point.frequency_hz for point in observation.spectrum]
    if activity_state == "active":
        if args.active_duration_myr is not None or args.active_fraction is not None:
            raise ConfigurationError(
                "--active-duration-myr/--active-fraction are only valid for remnants"
            )
        model = KDAForwardModel(observation, assumptions)
        powers = model.spectrum(
            frequencies,
            args.jet_power_w,
            args.age_myr * MYR_S,
            args.alpha_inj,
        )
        derived = model.derived_quantities(
            args.jet_power_w,
            args.age_myr * MYR_S,
            args.alpha_inj,
        )
        phase_parameters = {}
    else:
        if args.active_duration_myr is None and args.active_fraction is None:
            raise ConfigurationError(
                "remnant prediction requires --active-duration-myr or "
                "--active-fraction"
            )
        active_duration_myr = (
            args.active_duration_myr
            if args.active_duration_myr is not None
            else args.age_myr * args.active_fraction
        )
        model = KDAExtForwardModel(observation, assumptions)
        powers = model.spectrum(
            frequencies,
            args.jet_power_w,
            args.age_myr * MYR_S,
            active_duration_myr * MYR_S,
            args.alpha_inj,
        )
        derived = model.derived_quantities(
            args.jet_power_w,
            args.age_myr * MYR_S,
            active_duration_myr * MYR_S,
            args.alpha_inj,
        )
        phase_parameters = {
            "active_duration_myr": active_duration_myr,
            "remnant_duration_myr": args.age_myr - active_duration_myr,
            "active_fraction": active_duration_myr / args.age_myr,
        }
    output = {
        "source": observation.name,
        "activity_state": activity_state,
        "parameters": {
            "age_myr": args.age_myr,
            "alpha_inj": args.alpha_inj,
            "jet_power_w_per_jet": args.jet_power_w,
            **phase_parameters,
        },
        "derived": derived,
        "predicted_spectrum": [
            {"frequency_mhz_rest": point.frequency_mhz, "power_w_hz_sr": float(power)}
            for point, power in zip(observation.spectrum, powers, strict=True)
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _run_fit_restarted(args: argparse.Namespace) -> int:
    outer_observation, outer_assumptions, outer_active_bounds, outer_runtime = (
        load_configuration(args.outer)
    )
    inner_observation, inner_assumptions, inner_bounds, inner_runtime = (
        load_configuration(args.inner)
    )
    if outer_runtime["bootstrap_samples"] or inner_runtime["bootstrap_samples"]:
        raise ConfigurationError(
            "bootstrap propagation is not implemented for fit-restarted; "
            "set runtime.bootstrap_samples to 0 in both component files"
        )
    if outer_runtime["activity_state"] != "remnant":
        raise ConfigurationError(
            "the outer component file must set activity.state to 'remnant'"
        )
    if inner_runtime["activity_state"] != "active":
        raise ConfigurationError(
            "the inner component file must be active (omit activity or set it to 'active')"
        )
    active_lower, active_upper = outer_runtime["active_fraction_bounds"]
    outer_bounds = RemnantFitBounds(
        total_age_myr=outer_active_bounds.age_myr,
        remnant_fraction=(1.0 - active_upper, 1.0 - active_lower),
        alpha_inj=outer_active_bounds.alpha_inj,
        log10_jet_power_w=outer_active_bounds.log10_jet_power_w,
    )
    result = fit_restarted(
        outer_observation,
        outer_assumptions,
        outer_bounds,
        inner_observation,
        inner_assumptions,
        inner_bounds,
        old_jet_speed_fraction_c=args.old_jet_speed_c,
        inner_jet_power_w=args.inner_jet_power_w,
        allow_underconstrained_inner=args.allow_underconstrained_inner,
    )
    _append_subtraction_warnings(
        result.outer_remnant,
        outer_runtime,
        "Outer component",
    )
    _append_subtraction_warnings(
        result.inner_active,
        inner_runtime,
        "Inner component",
    )
    _append_quality_warnings(
        result.outer_remnant,
        outer_observation,
        outer_assumptions,
    )
    _append_quality_warnings(
        result.inner_active,
        inner_observation,
        inner_assumptions,
    )
    outer_runtime["selected_method"] = "remnant_joint"
    outer_runtime["bootstrap_samples_effective"] = 0
    outer_runtime["seed_effective"] = outer_runtime["seed"]
    inner_runtime["selected_method"] = (
        "joint_fixed_jet_power"
        if args.inner_jet_power_w is not None
        else "joint"
    )
    inner_runtime["fixed_jet_power_w"] = args.inner_jet_power_w
    inner_runtime["allow_underconstrained_inner"] = (
        args.allow_underconstrained_inner
    )
    inner_runtime["bootstrap_samples_effective"] = 0
    inner_runtime["seed_effective"] = inner_runtime["seed"]
    document = build_restarted_result_document(
        result,
        outer_observation,
        outer_assumptions,
        outer_active_bounds,
        outer_runtime,
        inner_observation,
        inner_assumptions,
        inner_bounds,
        inner_runtime,
    )
    chronology = result.chronology
    print(f"{result.source_name}: component-resolved restarted fit")
    print(f"  outer total age      = {chronology['outer_total_age_myr']:.6g} Myr")
    print(f"  outer active phase   = {chronology['outer_active_duration_myr']:.6g} Myr")
    print(f"  outer fading phase   = {chronology['outer_lobe_fading_duration_myr']:.6g} Myr")
    print(f"  inner active age     = {chronology['inner_active_age_myr']:.6g} Myr")
    print(
        "  AGN quiescent time   = "
        f"{chronology['central_engine_quiescent_interval_myr']:.6g} Myr "
        f"(old jet speed={args.old_jet_speed_c:g}c)"
    )
    if result.warnings:
        print("  warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")
    if args.output:
        write_json(args.output, document)
        print(f"Result written to {args.output.resolve()}")
    if args.print_json or not args.output:
        print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command=='gui':
            from .gui import main as gui_main
            return gui_main([])
        if args.command=='auto':
            from .automatic import run_automatic
            result=run_automatic(args.images,args.redshift,args.output_dir,field_t=args.field_t,
                pyrats_script=args.pyrats_script,pyrats_python=args.pyrats_python,
                frequencies_mhz=args.frequencies_mhz,calibration_fraction=args.calibration_fraction,
                seed=args.seed,progress=lambda message,fraction=None:print(message,flush=True))
            print(json.dumps(result,indent=2))
            return 0
        if args.command == 'verify-analysis':
            from .reliability import verify_analysis_manifest
            print(json.dumps(verify_analysis_manifest(args.directory, check_inputs=args.check_inputs), indent=2))
            return 0
        if args.command in ("analyze", "analyze-source"):
            from .source_analysis import analyze_source
            document, products = analyze_source(args.input, args.output_dir, method=args.method,
                pyrats_properties=args.pyrats_properties, budget_path=args.pyrats_budget,
                sensitivity_path=args.sensitivity, track_points=args.track_points,
                evolution_points=args.evolution_points, bootstrap=args.bootstrap,
                seed=args.seed, overwrite=args.overwrite)
            print(f"{document['source']['name']}: {document['kind']}")
            print(f"  analysis directory = {args.output_dir.resolve()}")
            print(f"  primary method = {document['primary_method']}")
            if document["kind"] == "radio_transport":
                print("  Global dynamical age, jet power and ambient density are not identified by the transport model")
            else:
                print(f"  conditional source age [Myr] = {document['source_age_summary']['point_estimate_myr']:.6g}")
            print(f"  products = {len(products.get('files', []))}")
            for warning in document["warnings"]:
                print(f"  note: {warning}")
            return 0
        if args.command == "calibrate-radio-priors":
            from .reference_priors import load_reference_priors
            if args.output.resolve() == args.input.resolve():
                raise ConfigurationError("prior output must not overwrite the reference catalogue")
            if args.output.exists() and not args.overwrite:
                raise ConfigurationError(f"{args.output} exists; use --overwrite")
            document = load_reference_priors(args.input, args.target_name, args.radio_class,
                allow_synthetic=args.allow_synthetic, allow_dependent=args.allow_dependent)
            write_json(args.output, document)
            print(f"Predictive priors written to {args.output.resolve()}")
            return 0
        if args.command == "infer-radio":
            from .radio_inference import compute_radio_inference, write_radio_products
            from .maps import _preview_targets
            targets = [target.expanduser().resolve() for target in (args.output, args.summary, args.samples) if target is not None]
            if args.preview is not None:
                targets += [path for _, path in _preview_targets(args.preview)]
            if len(set(targets)) != len(targets) or args.input.resolve() in targets:
                raise ConfigurationError("radio products must use distinct paths and must not replace the configuration")
            for target in targets:
                if target.exists() and not args.overwrite:
                    raise ConfigurationError(f"{target} exists; use --overwrite")
            from .source_analysis import _read_json
            radio_request, _ = _read_json(args.input)
            if "resolved_analysis" in radio_request:
                raise ConfigurationError("resolved_analysis requests full-source spectral maps; use analyze INPUT --output-dir RESULTS. infer-radio writes transport products only.")
            result = compute_radio_inference(args.input)
            write_radio_products(result, args.output, args.summary, samples_path=args.samples,
                                 preview_path=args.preview, overwrite=args.overwrite)
            print(f"{result.metadata['source']['name']}: {result.metadata['status']}")
            if result.metadata["map_statistics"]["interpretation"] == "diagnostic_only":
                print("  Diagnostic products only: fit adequacy or sampler agreement does not support a scientific age estimate")
            for name in ("area_weighted_mean", "area_weighted_median"):
                statistic = result.metadata["map_statistics"][name]
                print(f"  map {name} [Myr]: posterior median={statistic['median']:.6g}, 68% interval={statistic['credible_interval_68']}")
            print("  Global source dynamical age: not identifiable in this steady transport model")
            print(f"  FITS: {args.output.resolve()}")
            print(f"  Summary: {args.summary.resolve()}")
            for warning in result.metadata["warnings"]:
                print(f"  note: {warning}")
            return 0
        if args.command == "classes":
            from .radio_classes import RADIO_CLASSES
            print("Class           Description                    KDA fit    FITS expansion/flow maps")
            for key, entry in RADIO_CLASSES.items():
                print(f"{key:<15} {entry.label:<30} {'yes' if entry.kda_supported else 'no':<10} yes (supplied constraints)")
            print("Flow maps report travel times; expansion maps condition on the supplied expansion history.")
            print("infer-radio: prior-conditioned steady JP advection for active FR I/WAT/NAT/head-tail sources; no global lifetime fit.")
            return 0
        if args.command in ("map-age", "validate-map"):
            from .maps import compute_age_map, write_age_map, render_age_map, _output_path, _preview_targets
            if args.command == "map-age":
                targets = [args.output.expanduser().resolve()]
                if args.preview is not None:
                    targets += [path for _, path in _preview_targets(args.preview)]
                if len(set(targets)) != len(targets):
                    raise ConfigurationError("FITS output and publication previews must use different paths")
                for target in targets:
                    if target.exists() and not args.overwrite:
                        raise ConfigurationError(f"{target} already exists; use --overwrite to replace it")
            result = compute_age_map(args.input)
            if args.command == "map-age":
                _output_path(result, args.output, args.overwrite)
                if args.preview is not None:
                    for _, target in _preview_targets(args.preview):
                        _output_path(result, target, args.overwrite)
                write_age_map(result, args.output, overwrite=args.overwrite)
                print(f"FITS map written to {args.output.resolve()}")
                if args.preview is not None:
                    render_age_map(result, args.preview, overwrite=args.overwrite)
                    print(f"PNG preview and PDF/SVG companions written to {args.preview.resolve()}")
            else:
                print("Valid map input")
            print(f"  source = {result.metadata['source']['name']}")
            print(f"  radio class = {result.metadata['source']['radio_class']}")
            print(f"  clock = {result.metadata['clock']['kind']}")
            print(f"  valid pixels = {result.metadata['valid_pixels']} / {result.valid.size}")
            print(f"  range [Myr] = {result.metadata['age_range_myr']}")
            for warning in result.metadata["warnings"]:
                print(f"  note: {warning}")
            return 0
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "fit":
            return _run_fit(args)
        if args.command == "predict":
            return _run_predict(args)
        if args.command == "fit-restarted":
            return _run_fit_restarted(args)
        raise AssertionError("unreachable command")
    except (
        ConfigurationError,
        ModelDomainError,
        FitError,
        RemnantFitError,
        RestartedFitError,
        ValueError,
        OSError,
        ArithmeticError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
