"""Strict JSON input parsing and reproducible result serialization."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy
import scipy

from . import __version__
from .constants import CONSTANT_SET, FOUR_PI
from .fitting import FitBounds, FitResult
from .model import Geometry, ModelAssumptions, Observation, SpectrumPoint


class ConfigurationError(ValueError):
    """Raised for missing, ambiguous, or unknown configuration entries."""


_MODEL_FIELDS = {
    "beta",
    "core_radius_kpc",
    "gamma_min",
    "gamma_max",
    "gamma_c",
    "gamma_x",
    "gamma_b",
    "k_prime",
    "magnetic_ratio",
    "equipartition_field_ug",
    "cmb_temperature_k",
    "c1_prescription",
    "integration_order",
    "max_log_injection_span",
}
_FIT_FIELDS = {
    "age_myr",
    "alpha_inj",
    "log10_jet_power_w",
    "active_fraction",
    "age_grid_size",
    "alpha_grid_size",
    "power_scan_size",
}

_POST_SHUTDOWN_MODEL = "kdaext2017-kc02-model-b"


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must be a JSON object")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {path} field(s): {', '.join(unknown)}")


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a JSON number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ConfigurationError(f"{path} is outside the floating-point range") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ConfigurationError(f"JSON numbers must be finite; invalid value: {value}")


def _strict_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be a JSON integer")
    return value


def _optional_finite_float(mapping: dict[str, Any], key: str, path: str) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _finite_float(mapping[key], f"{path}.{key}")


def load_configuration(path: str | Path) -> tuple[Observation, ModelAssumptions, FitBounds, dict[str, Any]]:
    """Load a strict JSON input file.

    Radio powers may be supplied as per-steradian KDA powers or isotropic
    luminosities.  The latter are divided by 4*pi immediately and the
    normalized convention is preserved in the result file.
    """

    input_path = Path(path)
    try:
        raw_bytes = input_path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read {input_path}: {exc}") from exc
    try:
        data = json.loads(
            raw_text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ConfigurationError:
        raise
    except ValueError as exc:
        raise ConfigurationError(f"invalid JSON in {input_path}: {exc}") from exc
    root = _require_mapping(data, "root")
    _reject_unknown(
        root,
        {"source", "geometry", "spectrum", "model", "fit", "runtime", "activity"},
        "root",
    )

    activity_data = _require_mapping(root.get("activity", {}), "activity")
    _reject_unknown(activity_data, {"state", "post_shutdown_dynamics"}, "activity")
    activity_state = activity_data.get("state", "active")
    if activity_state not in ("active", "remnant"):
        raise ConfigurationError("activity.state must be 'active' or 'remnant'")
    post_shutdown_dynamics = activity_data.get("post_shutdown_dynamics")
    if activity_state == "remnant":
        if post_shutdown_dynamics is None:
            post_shutdown_dynamics = _POST_SHUTDOWN_MODEL
        if post_shutdown_dynamics != _POST_SHUTDOWN_MODEL:
            raise ConfigurationError(
                "activity.post_shutdown_dynamics must be "
                f"'{_POST_SHUTDOWN_MODEL}'"
            )
    elif post_shutdown_dynamics is not None:
        raise ConfigurationError(
            "activity.post_shutdown_dynamics is only valid for a remnant"
        )

    source = _require_mapping(root.get("source"), "source")
    _reject_unknown(source, {"name", "redshift", "radio_class"}, "source")
    if "name" not in source or "redshift" not in source:
        raise ConfigurationError("source requires name and redshift")
    if not isinstance(source["name"], str) or not source["name"].strip():
        raise ConfigurationError("source.name must be a non-empty JSON string")
    redshift = _finite_float(source["redshift"], "source.redshift")

    geometry_data = _require_mapping(root.get("geometry"), "geometry")
    _reject_unknown(
        geometry_data,
        {
            "mode",
            "projected_size_kpc",
            "axial_ratio",
            "inclination_deg",
            "size_error_kpc",
            "axial_ratio_error",
        },
        "geometry",
    )
    for required in ("mode", "projected_size_kpc", "axial_ratio", "inclination_deg"):
        if required not in geometry_data:
            raise ConfigurationError(f"geometry requires {required}")
    if not isinstance(geometry_data["mode"], str):
        raise ConfigurationError("geometry.mode must be a JSON string")
    try:
        geometry = Geometry(
            mode=geometry_data["mode"],
            projected_size_kpc=_finite_float(
                geometry_data["projected_size_kpc"], "geometry.projected_size_kpc"
            ),
            axial_ratio=_finite_float(geometry_data["axial_ratio"], "geometry.axial_ratio"),
            inclination_deg=_finite_float(
                geometry_data["inclination_deg"], "geometry.inclination_deg"
            ),
            size_error_kpc=_optional_finite_float(
                geometry_data, "size_error_kpc", "geometry"
            ),
            axial_ratio_error=_optional_finite_float(
                geometry_data, "axial_ratio_error", "geometry"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid geometry: {exc}") from exc

    spectrum_data = _require_mapping(root.get("spectrum"), "spectrum")
    _reject_unknown(
        spectrum_data,
        {"power_convention", "frequency_frame", "core_subtracted", "hotspots_subtracted", "points"},
        "spectrum",
    )
    power_convention = spectrum_data.get("power_convention")
    if power_convention not in ("per_sr", "isotropic"):
        raise ConfigurationError("spectrum.power_convention must be 'per_sr' or 'isotropic'")
    if "frequency_frame" not in spectrum_data:
        raise ConfigurationError("spectrum.frequency_frame is required and must be 'rest'")
    if spectrum_data["frequency_frame"] != "rest":
        raise ConfigurationError(
            "only rest-frame frequencies with K-corrected powers are accepted; "
            "observer-frame conversion is intentionally not guessed"
        )
    for flag in ("core_subtracted", "hotspots_subtracted"):
        if flag in spectrum_data and not isinstance(spectrum_data[flag], bool):
            raise ConfigurationError(f"spectrum.{flag} must be a JSON boolean")
    raw_points = spectrum_data.get("points")
    if not isinstance(raw_points, list):
        raise ConfigurationError("spectrum.points must be a JSON array")
    points = []
    for index, raw_point in enumerate(raw_points):
        point = _require_mapping(raw_point, f"spectrum.points[{index}]")
        _reject_unknown(
            point,
            {"frequency_mhz", "power_w_hz", "log10_power_w_hz", "fractional_error"},
            f"spectrum.points[{index}]",
        )
        if "frequency_mhz" not in point:
            raise ConfigurationError(f"spectrum.points[{index}] requires frequency_mhz")
        supplied = [key for key in ("power_w_hz", "log10_power_w_hz") if key in point]
        if len(supplied) != 1:
            raise ConfigurationError(
                f"spectrum.points[{index}] requires exactly one of power_w_hz or log10_power_w_hz"
            )
        frequency_mhz = _finite_float(
            point["frequency_mhz"], f"spectrum.points[{index}].frequency_mhz"
        )
        try:
            power = (
                _finite_float(point["power_w_hz"], f"spectrum.points[{index}].power_w_hz")
                if "power_w_hz" in point
                else 10.0
                ** _finite_float(
                    point["log10_power_w_hz"],
                    f"spectrum.points[{index}].log10_power_w_hz",
                )
            )
        except OverflowError as exc:
            raise ConfigurationError(
                f"spectrum.points[{index}] power is outside the floating-point range"
            ) from exc
        if power_convention == "isotropic":
            power /= FOUR_PI
        try:
            points.append(
                SpectrumPoint(
                    frequency_mhz=frequency_mhz,
                    power_w_hz_sr=power,
                    fractional_error=_optional_finite_float(
                        point, "fractional_error", f"spectrum.points[{index}]"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid spectrum.points[{index}]: {exc}") from exc
    try:
        observation = Observation.from_points(
            source["name"], redshift, geometry, points,
            radio_class=source.get("radio_class", "fr2"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid observation: {exc}") from exc

    model_data = _require_mapping(root.get("model", {}), "model")
    _reject_unknown(model_data, _MODEL_FIELDS, "model")
    normalized_model: dict[str, Any] = {}
    for key, value in model_data.items():
        if key == "c1_prescription":
            if not isinstance(value, str):
                raise ConfigurationError("model.c1_prescription must be a JSON string")
            normalized_model[key] = value
        elif key == "integration_order":
            normalized_model[key] = _strict_int(value, "model.integration_order")
        elif key in ("magnetic_ratio", "equipartition_field_ug") and value is None:
            normalized_model[key] = None
        else:
            normalized_model[key] = _finite_float(value, f"model.{key}")
    try:
        assumptions = ModelAssumptions(**normalized_model)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid model: {exc}") from exc

    fit_data = _require_mapping(root.get("fit", {}), "fit")
    _reject_unknown(fit_data, _FIT_FIELDS, "fit")
    for pair_name in ("age_myr", "alpha_inj", "log10_jet_power_w", "active_fraction"):
        if pair_name in fit_data:
            value = fit_data[pair_name]
            if not isinstance(value, list) or len(value) != 2:
                raise ConfigurationError(f"fit.{pair_name} must be a two-element JSON array")
            fit_data[pair_name] = (
                _finite_float(value[0], f"fit.{pair_name}[0]"),
                _finite_float(value[1], f"fit.{pair_name}[1]"),
            )
    active_fraction_bounds = fit_data.pop("active_fraction", None)
    if activity_state == "remnant":
        if active_fraction_bounds is None:
            active_fraction_bounds = (0.1, 0.95)
        if not (
            0.0 < active_fraction_bounds[0] < active_fraction_bounds[1] < 1.0
        ):
            raise ConfigurationError(
                "fit.active_fraction must satisfy 0 < lower < upper < 1"
            )
    elif active_fraction_bounds is not None:
        raise ConfigurationError(
            "fit.active_fraction is only valid when activity.state is 'remnant'"
        )
    for size_name in ("age_grid_size", "alpha_grid_size", "power_scan_size"):
        if size_name in fit_data:
            fit_data[size_name] = _strict_int(fit_data[size_name], f"fit.{size_name}")
    try:
        bounds = FitBounds(**fit_data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid fit settings: {exc}") from exc

    runtime = _require_mapping(root.get("runtime", {}), "runtime")
    _reject_unknown(runtime, {"bootstrap_samples", "seed"}, "runtime")
    runtime.setdefault("bootstrap_samples", 0)
    runtime.setdefault("seed", 20260721)
    runtime["bootstrap_samples"] = _strict_int(
        runtime["bootstrap_samples"], "runtime.bootstrap_samples"
    )
    runtime["seed"] = _strict_int(runtime["seed"], "runtime.seed")
    if runtime["bootstrap_samples"] < 0:
        raise ConfigurationError("runtime.bootstrap_samples cannot be negative")
    if runtime["seed"] < 0:
        raise ConfigurationError("runtime.seed cannot be negative")

    runtime["input_path"] = str(input_path.resolve())
    runtime["input_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    runtime["input_power_convention"] = power_convention
    runtime["normalized_power_convention"] = "W Hz^-1 sr^-1"
    runtime["core_subtracted"] = spectrum_data.get("core_subtracted")
    runtime["hotspots_subtracted"] = spectrum_data.get("hotspots_subtracted")
    runtime["activity_state"] = activity_state
    runtime["post_shutdown_dynamics"] = post_shutdown_dynamics
    runtime["active_fraction_bounds"] = active_fraction_bounds
    return observation, assumptions, bounds, runtime


def normalized_input(
    observation: Observation,
    assumptions: ModelAssumptions,
    bounds: FitBounds,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical configuration that ``load_configuration`` can reload."""

    spectrum: dict[str, Any] = {
        "power_convention": "per_sr",
        "frequency_frame": "rest",
        "points": [
            {
                "frequency_mhz": point.frequency_mhz,
                "power_w_hz": point.power_w_hz_sr,
                **(
                    {"fractional_error": point.fractional_error}
                    if point.fractional_error is not None
                    else {}
                ),
            }
            for point in observation.spectrum
        ],
    }
    normalized_runtime: dict[str, int] | None = None
    if runtime is not None:
        for flag in ("core_subtracted", "hotspots_subtracted"):
            if isinstance(runtime.get(flag), bool):
                spectrum[flag] = runtime[flag]
        normalized_runtime = {
            "bootstrap_samples": int(
                runtime.get("bootstrap_samples_effective", runtime["bootstrap_samples"])
            ),
            "seed": int(runtime.get("seed_effective", runtime["seed"])),
        }

    fit_document = asdict(bounds)
    activity_state = runtime.get("activity_state", "active") if runtime else "active"
    if activity_state == "remnant":
        active_fraction_bounds = runtime.get("active_fraction_bounds") if runtime else None
        if active_fraction_bounds is None:
            raise ConfigurationError(
                "normalized remnant input is missing active-fraction bounds"
            )
        fit_document["active_fraction"] = list(active_fraction_bounds)

    document = {
        "source": {"name": observation.name, "redshift": observation.redshift,
                   "radio_class": observation.radio_class},
        "geometry": asdict(observation.geometry),
        "spectrum": spectrum,
        "model": asdict(assumptions),
        "fit": fit_document,
    }
    if activity_state == "remnant":
        document["activity"] = {
            "state": "remnant",
            "post_shutdown_dynamics": (
                runtime.get("post_shutdown_dynamics") or _POST_SHUTDOWN_MODEL
            ),
        }
    if normalized_runtime is not None:
        document["runtime"] = normalized_runtime
    return document


def build_result_document(
    result: FitResult,
    observation: Observation,
    assumptions: ModelAssumptions,
    bounds: FitBounds,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": (
            "fr2dynage-result-2"
            if result.activity_history is not None
            else "fr2dynage-result-1"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "PyRADAGE",
            "version": __version__,
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "physical_constants": CONSTANT_SET,
        },
        "provenance": runtime,
        "normalized_input": normalized_input(observation, assumptions, bounds, runtime),
        "result": result.to_dict(),
    }


def build_restarted_result_document(
    result: Any,
    outer_observation: Observation,
    outer_assumptions: ModelAssumptions,
    outer_bounds: FitBounds,
    outer_runtime: dict[str, Any],
    inner_observation: Observation,
    inner_assumptions: ModelAssumptions,
    inner_bounds: FitBounds,
    inner_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a two-component restarted-source analysis."""

    return {
        "schema_version": "fr2dynage-result-2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "PyRADAGE",
            "version": __version__,
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "physical_constants": CONSTANT_SET,
        },
        "provenance": {
            "outer_remnant": outer_runtime,
            "inner_active": inner_runtime,
        },
        "normalized_input": {
            "activity": {
                "state": "restarted",
                "component_separation_required": True,
            },
            "components": {
                "outer_remnant": normalized_input(
                    outer_observation,
                    outer_assumptions,
                    outer_bounds,
                    outer_runtime,
                ),
                "inner_active": normalized_input(
                    inner_observation,
                    inner_assumptions,
                    inner_bounds,
                    inner_runtime,
                ),
            },
        },
        "result": result.to_dict(),
    }


def write_json(path: str | Path, document: dict[str, Any]) -> None:
    """Atomically replace a result after flushing its complete JSON to disk."""

    target = Path(path)
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # Preserve the original write/replace error if cleanup also fails.
                pass
