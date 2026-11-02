"""Strict, source-level reference calibration for radio transport inference.

This is a conservative empirical random-effects *predictive* calibration, not
a hierarchical posterior sampler or a source-age regression.  References
constrain physical parameters; source-level ages cannot supervise pixel ages.
Natural logarithms are used for positive quantities.  The population scatter
floors below are declared safeguards, not measurements or universal physics.
Catalogue validation cannot establish whether a claimed constraint is truly
independent: the supplied method, citation and independence declaration remain
part of the scientific provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any

import numpy as np

from .io import _reject_json_constant, _unique_object
from .radio_classes import normalize_radio_class


class ReferencePriorError(ValueError):
    """A reference catalogue is malformed or cannot calibrate any parameter."""


SCHEMA_VERSION = "fr2dynage-reference-1"
MIN_SOURCES = 3
# unit, exact physical quantity, transformation, minimum population sigma
PARAMETERS = {
    "flow_speed_km_s": ("km/s", "projected_plasma_transport_speed", "log", 0.35),
    "magnetic_field_ug": ("uG", "emitting_plasma_magnetic_field", "log", 0.30),
    "alpha_inj": ("1", "injection_spectral_index", "identity", 0.05),
}


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReferencePriorError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ReferencePriorError(f"unknown {label} field(s): {', '.join(unknown)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferencePriorError(f"{label} must be a nonempty string")
    return value.strip()


def _identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReferencePriorError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReferencePriorError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ReferencePriorError(f"{label} must be a finite positive number")
    return number


def _checked_catalogue(document: Any) -> list[dict[str, Any]]:
    root = _object(document, {"schema_version", "description", "sources"}, "catalogue")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ReferencePriorError(f"catalogue.schema_version must be '{SCHEMA_VERSION}'")
    if "description" in root:
        _text(root["description"], "catalogue.description")
    sources = root.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ReferencePriorError("catalogue.sources must be a nonempty list")
    checked, identities = [], set()
    for index, item in enumerate(sources):
        label = f"sources[{index}]"
        source = _object(item, {"name", "aliases", "radio_class", "reference", "origin", "constraints"}, label)
        name = _text(source.get("name"), f"{label}.name")
        aliases = source.get("aliases", [])
        if not isinstance(aliases, list):
            raise ReferencePriorError(f"{label}.aliases must be a list of strings")
        names = [name] + [_text(value, f"{label}.aliases") for value in aliases]
        own_ids = {_identity(value) for value in names}
        if len(own_ids) != len(names) or own_ids & identities:
            raise ReferencePriorError(f"{label}: duplicate source identity or alias; references must be independent sources")
        identities.update(own_ids)
        try:
            radio_class = normalize_radio_class(source.get("radio_class"))
        except ValueError as exc:
            raise ReferencePriorError(f"{label}: {exc}") from exc
        reference = _text(source.get("reference"), f"{label}.reference")
        origin = source.get("origin")
        if origin not in ("observed", "synthetic"):
            raise ReferencePriorError(f"{label}.origin must be observed or synthetic")
        constraints = _object(source.get("constraints"), set(PARAMETERS), f"{label}.constraints")
        if not constraints:
            raise ReferencePriorError(f"{label}.constraints must not be empty")
        values = {}
        for parameter, spec in constraints.items():
            clabel = f"{label}.constraints.{parameter}"
            constraint = _object(spec, {"value", "error", "unit", "quantity", "method", "independent"}, clabel)
            unit, quantity, _, _ = PARAMETERS[parameter]
            if constraint.get("unit") != unit:
                raise ReferencePriorError(f"{clabel}.unit must be '{unit}'")
            if constraint.get("quantity") != quantity:
                raise ReferencePriorError(f"{clabel}.quantity must be '{quantity}'")
            value = _positive(constraint.get("value"), f"{clabel}.value")
            error = _positive(constraint.get("error"), f"{clabel}.error")
            if parameter == "flow_speed_km_s" and value >= 299792.458:
                raise ReferencePriorError(f"{clabel}.value must be a subluminal plasma transport speed")
            if parameter == "alpha_inj" and value >= 3:
                raise ReferencePriorError(f"{clabel}.value must be below 3 in the supported injection-index domain")
            method = _text(constraint.get("method"), f"{clabel}.method")
            independent = constraint.get("independent")
            if not isinstance(independent, bool):
                raise ReferencePriorError(f"{clabel}.independent must explicitly be boolean")
            values[parameter] = {"value": value, "error": error, "method": method,
                                 "independent": independent, "unit": unit, "quantity": quantity}
        checked.append({"name": name, "aliases": names[1:], "identities": own_ids,
                        "radio_class": radio_class, "reference": reference,
                        "origin": origin, "constraints": values})
    return checked


def calibrate_reference_priors(
    document: Any,
    target_name: str,
    radio_class: str,
    *,
    allow_synthetic: bool = False,
    allow_dependent: bool = False,
) -> dict[str, Any]:
    """Calibrate declared physical-parameter priors from source-level references.

    Every reference is validated, even if excluded by class. Target matching
    uses NFKC, case folding and collapsed whitespace, including explicit
    aliases. Unknown naming aliases cannot be inferred; catalogue authors must
    provide them. Each returned prior requires at least three distinct sources.
    Parameters with fewer references are omitted and recorded in provenance.

    For each parameter, estimate between-source variance as sample variance
    minus mean measurement variance, floored at the declared scatter floor.
    Use inverse total-variance weights for the population mean. The predictive
    variance is population variance plus uncertainty in that mean. Uncertainty
    in the population variance itself and parameter cross-covariance are not
    sampled; these are empirical, independently applied predictive priors.
    """
    target = _identity(_text(target_name, "target_name"))
    try:
        target_class = normalize_radio_class(radio_class)
    except ValueError as exc:
        raise ReferencePriorError(str(exc)) from exc
    if not isinstance(allow_synthetic, bool) or not isinstance(allow_dependent, bool):
        raise ReferencePriorError("allow_synthetic and allow_dependent must be booleans")
    sources = _checked_catalogue(document)
    candidates = {key: [] for key in PARAMETERS}
    excluded, selected = [], []
    for source in sources:
        reason = None
        if target in source["identities"]:
            reason = "target source excluded to prevent calibration leakage"
        elif source["radio_class"] != target_class:
            reason = "different radio class"
        elif source["origin"] == "synthetic" and not allow_synthetic:
            reason = "synthetic reference requires explicit allow_synthetic"
        if reason:
            excluded.append({"name": source["name"], "reason": reason})
            continue
        for key, constraint in source["constraints"].items():
            if not constraint["independent"] and not allow_dependent:
                excluded.append({"name": source["name"], "parameter": key,
                                 "reason": "dependent constraint requires explicit allow_dependent"})
                continue
            record = {"name": source["name"], "reference": source["reference"],
                      "origin": source["origin"], "parameter": key, **constraint}
            candidates[key].append(record)
            selected.append(record)
    priors, diagnostics = {}, {}
    for parameter, records in candidates.items():
        unit, quantity, transform, floor = PARAMETERS[parameter]
        diagnostic = {"n_sources": len(records), "names": [r["name"] for r in records],
                      "unit": unit, "quantity": quantity, "transform": transform,
                      "minimum_population_sigma": floor}
        diagnostics[parameter] = diagnostic
        if len(records) < MIN_SOURCES:
            diagnostic["status"] = "insufficient_references"
            continue
        values = np.asarray([r["value"] for r in records])
        errors = np.asarray([r["error"] for r in records])
        if transform == "log":
            transformed = np.log(values)
            # Stable sqrt(log(1 + (sigma/value)^2)), even for extreme ratios.
            log_ratio = np.log(errors) - np.log(values)
            measurement_variance = np.logaddexp(0.0, 2.0 * log_ratio)
        else:
            transformed = values
            with np.errstate(over="ignore"):
                measurement_variance = errors ** 2
            if not np.all(np.isfinite(measurement_variance)):
                raise ReferencePriorError(f"{parameter}: uncertainty is outside the representable numerical domain")
        observed_variance = float(np.var(transformed, ddof=1))
        population_variance = max(observed_variance - float(measurement_variance.mean()), floor ** 2)
        weights = 1.0 / (population_variance + measurement_variance)
        total_weight = float(weights.sum())
        mean = float(np.dot(weights, transformed) / total_weight)
        mean_variance = 1.0 / total_weight
        predictive_sigma = math.sqrt(population_variance + mean_variance)
        if not all(math.isfinite(v) for v in (mean, predictive_sigma, observed_variance)):
            raise ReferencePriorError(f"{parameter}: calibration is outside the representable numerical domain")
        priors[parameter] = ({"distribution": "lognormal", "log_mean": mean, "log_sigma": predictive_sigma}
                             if transform == "log" else
                             {"distribution": "normal", "mean": mean, "sigma": predictive_sigma})
        diagnostic.update(status="calibrated", observed_variance=observed_variance,
                          population_sigma=math.sqrt(population_variance),
                          population_mean_sigma=math.sqrt(mean_variance))
    if not priors:
        raise ReferencePriorError(
            "no parameter has at least three eligible independent same-class reference sources "
            "after target, class, origin and dependency exclusions"
        )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "method": "empirical random-effects predictive calibration",
        "target_name": target_name.strip(), "radio_class": target_class,
        "minimum_sources_per_parameter": MIN_SOURCES,
        "allow_synthetic": allow_synthetic, "allow_dependent": allow_dependent,
        "contains_synthetic_constraints": any(r["origin"] == "synthetic" for r in selected),
        "contains_dependent_constraints": any(not r["independent"] for r in selected),
        "selected_constraints": selected, "excluded": excluded,
        "parameter_diagnostics": diagnostics,
        "limitations": [
            "Source and class matching is explicit; unnamed aliases and population selection bias are not inferred.",
            "Scatter floors are declared safeguards, not observed physical population dispersions.",
            "Population-scatter uncertainty and parameter cross-covariance are not sampled.",
            "Reference independence is supplied by the catalogue author and is not proven by validation.",
            "Matching radio class alone does not establish transferability in redshift, size, environment or observing coverage.",
        ],
    }
    result = {"priors": priors, "provenance": provenance}
    json.dumps(result, allow_nan=False)
    return result


def load_reference_priors(
    path: str | Path,
    target_name: str,
    radio_class: str,
    *,
    allow_synthetic: bool = False,
    allow_dependent: bool = False,
) -> dict[str, Any]:
    """Load strict JSON, calibrate eligible references and record file SHA-256."""
    filename = Path(path).expanduser().resolve()
    try:
        raw = filename.read_bytes()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_json_constant)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise ReferencePriorError(f"cannot load reference catalogue {filename}: {exc}") from exc
    result = calibrate_reference_priors(document, target_name, radio_class,
                                        allow_synthetic=allow_synthetic,
                                        allow_dependent=allow_dependent)
    result["provenance"].update(path=str(filename), sha256=hashlib.sha256(raw).hexdigest())
    return result
