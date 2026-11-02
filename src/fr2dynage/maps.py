"""Class-aware, conditional kinematic clocks from registered radio images.

The images select detected emission and supply spectral diagnostics. The
clock comes from an independent expansion/flow constraint and geometry.
No pixel is treated as an independent self-similar FR II radio lobe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from .io import _unique_object, _reject_json_constant
from .kinematics import compute_kinematic_age, KINEMATIC_FLAG_MEANINGS
from .maps_io import (FitsPlane, MapInputError, read_fits_plane, science_image,
                      ensure_same_beam, validate_registration, auxiliary_data,
                      atomic_write_fits, _astropy)
from .radio_classes import normalize_radio_class
from .spectral_maps import fit_power_law_maps, SpectralMapResult, SPECTRAL_FLAGS


@dataclass(frozen=True)
class AgeMapResult:
    age_myr: np.ndarray
    error_myr: np.ndarray | None
    valid: np.ndarray
    kinematic_flags: np.ndarray
    spectral: SpectralMapResult
    distance_kpc: np.ndarray
    speed_km_s: np.ndarray
    reference: FitsPlane
    metadata: dict[str, Any]
    input_paths: frozenset[Path]


def _mapping(value: Any, label: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MapInputError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise MapInputError(f"unknown {label} field(s): {', '.join(sorted(unknown))}")
    return value


def _number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError
        result = float(value)
    except (TypeError, OverflowError, ValueError) as exc:
        raise MapInputError(f"{label} must be a finite JSON number") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise MapInputError(f"{label} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return result


def _file_spec(value: Any, base: Path, label: str) -> tuple[Path, int | str]:
    if isinstance(value, str):
        path, hdu = value, 0
    else:
        spec = _mapping(value, label, {"path", "hdu"})
        path, hdu = spec.get("path"), spec.get("hdu", 0)
    if not isinstance(path, str) or not path.strip():
        raise MapInputError(f"{label}.path must be a nonempty file path")
    if isinstance(hdu, bool) or not isinstance(hdu, (int, str)) or (isinstance(hdu, int) and hdu < 0):
        raise MapInputError(f"{label}.hdu must be a nonnegative integer or extension name")
    return (base / Path(path).expanduser()).resolve(), hdu


def _one_of(mapping: dict[str, Any], names: tuple[str, ...], label: str, *, required: bool = True) -> str | None:
    found = [name for name in names if name in mapping]
    if len(found) > 1 or (required and len(found) != 1):
        raise MapInputError(f"{label} requires exactly one of {', '.join(names)}")
    return found[0] if found else None


def compute_age_map(config_path: str | Path) -> AgeMapResult:
    """Validate input images and calculate an expansion or flow-time map."""

    _astropy()
    from astropy import __version__ as astropy_version
    from . import __version__

    path = Path(config_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_json_constant)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise MapInputError(f"cannot load map configuration {path}: {exc}") from exc
    config = _mapping(config, "root", {"schema_version", "source", "images", "selection", "geometry", "kinematics"})
    if config.get("schema_version", "fr2dynage-map-1") != "fr2dynage-map-1":
        raise MapInputError("schema_version must be 'fr2dynage-map-1'")
    source = _mapping(config.get("source"), "source", {"name", "radio_class", "redshift", "activity_state"})
    if not isinstance(source.get("name"), str) or not source["name"].strip():
        raise MapInputError("source.name is required")
    radio_class = normalize_radio_class(source.get("radio_class"))
    activity = source.get("activity_state", "unknown")
    if activity not in ("active", "remnant", "restarted", "unknown"):
        raise MapInputError("source.activity_state must be active, remnant, restarted, or unknown")
    if "redshift" in source:
        _number(source["redshift"], "source.redshift", nonnegative=True)
    geometry = _mapping(config.get("geometry"), "geometry", {
        "distance_kind", "distance_frame", "distance_map", "origin_pixel", "origin_icrs_deg",
        "kpc_per_arcsec", "path_is_straight", "distance_error_kpc", "distance_error_map"})
    kinetics = _mapping(config.get("kinematics"), "kinematics", {
        "mode", "speed_role", "speed_km_s", "speed_map", "speed_error_km_s", "speed_error_map",
        "speed_frame", "inclination_deg", "inclination_error_deg", "expansion_index",
        "expansion_index_error", "constraint_description"})
    mode = kinetics.get("mode")
    if mode not in ("expansion", "flow"):
        raise MapInputError("kinematics.mode must be expansion or flow")
    expected_kind, expected_role = ("radius", "current_expansion") if mode == "expansion" else ("path_length", "effective_flow")
    if geometry.get("distance_kind") != expected_kind or kinetics.get("speed_role") != expected_role:
        raise MapInputError(f"{mode} requires distance_kind='{expected_kind}' and speed_role='{expected_role}'")
    for mapping, key, label in ((geometry, "distance_frame", "geometry"), (kinetics, "speed_frame", "kinematics")):
        if mapping.get(key) not in ("projected", "intrinsic"):
            raise MapInputError(f"{label}.{key} must explicitly be projected or intrinsic")
    if not isinstance(kinetics.get("constraint_description"), str) or not kinetics["constraint_description"].strip():
        raise MapInputError("kinematics.constraint_description must describe the independent speed constraint")
    origin_key = _one_of(geometry, ("distance_map", "origin_pixel", "origin_icrs_deg"), "geometry")
    if "path_is_straight" in geometry and not isinstance(geometry["path_is_straight"], bool):
        raise MapInputError("geometry.path_is_straight must be boolean")
    if origin_key != "distance_map" and geometry["distance_frame"] != "projected":
        raise MapInputError("WCS-derived distances are projected; use a distance_map for intrinsic distances")
    if origin_key == "distance_map" and ("kpc_per_arcsec" in geometry or "path_is_straight" in geometry):
        raise MapInputError("distance_map already defines distances; omit kpc_per_arcsec/path_is_straight")
    if mode == "flow" and origin_key != "distance_map":
        if radio_class in ("wat", "nat", "head_tail") or geometry.get("path_is_straight") is not True:
            raise MapInputError("flow maps require a path-length distance_map; a straight path may be explicitly affirmed for other classes")
    selection = _mapping(config.get("selection", {}), "selection", {"snr_min", "min_bands", "require_all_bands", "mask"})
    images = config.get("images")
    if not isinstance(images, list) or len(images) < 3:
        raise MapInputError("images requires at least three independently measured frequency images")
    records: list[dict[str, Any]] = []
    inputs = {path}
    base = path.parent

    def load(spec: Any, label: str) -> FitsPlane:
        filename, hdu = _file_spec(spec, base, label)
        plane = read_fits_plane(filename, hdu)
        inputs.add(plane.path)
        records.append({"role": label, "path": str(plane.path), "hdu": hdu, "sha256": plane.sha256})
        return plane

    planes = []
    fluxes, noises, frequencies, calibrations = [], [], [], []
    seen: set[tuple[Path, int]] = set()
    beam = None
    for index, entry in enumerate(images):
        label = f"images[{index}]"
        entry = _mapping(entry, label, {"path", "hdu", "frequency_mhz", "rms_jy_per_beam", "rms_map", "calibration_fraction"})
        spec = {key: entry[key] for key in ("path", "hdu") if key in entry}
        frequency = _number(entry.get("frequency_mhz"), f"{label}.frequency_mhz", positive=True)
        calibration = _number(entry.get("calibration_fraction", 0.0), f"{label}.calibration_fraction", nonnegative=True)
        noise_key = _one_of(entry, ("rms_jy_per_beam", "rms_map"), label)
        plane = load(spec, label)
        identity = (plane.path, plane.hdu_index)
        if identity in seen:
            raise MapInputError("the same image/HDU cannot represent multiple independent frequencies")
        seen.add(identity)
        flux, current_beam = science_image(plane)
        if planes:
            validate_registration(planes[0], plane)
            ensure_same_beam(planes[0], plane)
        else:
            beam = current_beam
        if noise_key == "rms_map":
            noise = auxiliary_data(load(entry[noise_key], f"{label}.rms_map"), plane, "Jy / beam")
        else:
            noise = np.full(plane.data.shape, _number(entry[noise_key], f"{label}.{noise_key}", positive=True))
        planes.append(plane)
        fluxes.append(flux)
        noises.append(noise)
        frequencies.append(frequency)
        calibrations.append(calibration)
    reference = planes[0]
    shape = reference.data.shape
    mask = auxiliary_data(load(selection["mask"], "selection.mask"), reference, None).astype(bool) if "mask" in selection else np.ones(shape, dtype=bool)
    spectrum = fit_power_law_maps(frequencies, np.stack(fluxes), np.stack(noises),
        calibration_fractions=calibrations, mask=mask,
        snr_min=selection.get("snr_min", 5.0), min_bands=selection.get("min_bands", 3),
        require_all_bands=selection.get("require_all_bands", True))
    distance_details: dict[str, Any] = {"method": origin_key}
    if origin_key == "distance_map":
        distance = auxiliary_data(load(geometry[origin_key], "geometry.distance_map"), reference, "kpc")
    else:
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        origin = geometry[origin_key]
        if not isinstance(origin, list) or len(origin) != 2:
            raise MapInputError(f"geometry.{origin_key} must contain two numbers")
        first, second = (_number(value, f"geometry.{origin_key}") for value in origin)
        if origin_key == "origin_pixel":
            sky_origin = reference.wcs.pixel_to_world(first, second)
        else:
            if not 0 <= first < 360 or not -90 <= second <= 90:
                raise MapInputError("origin_icrs_deg requires RA in [0,360) and Dec in [-90,90]")
            sky_origin = SkyCoord(first * u.deg, second * u.deg, frame="icrs")
        if "kpc_per_arcsec" in geometry:
            scale = _number(geometry["kpc_per_arcsec"], "geometry.kpc_per_arcsec", positive=True)
            distance_details["scale_method"] = "supplied angular scale"
        else:
            from astropy.cosmology import Planck18

            redshift = _number(source.get("redshift"), "source.redshift for angular scale", positive=True)
            scale = float(Planck18.kpc_proper_per_arcmin(redshift).value / 60.0)
            distance_details["scale_method"] = "Planck18 proper angular-diameter distance"
        y, x = np.indices(shape, dtype=float)
        distance = reference.wcs.pixel_to_world(x, y).separation(sky_origin).arcsec * scale
        if not np.all(np.isfinite(distance)):
            raise MapInputError("WCS-derived distance is nonfinite; check the origin and image projection")
        distance_details["kpc_per_arcsec"] = scale

    def scalar_or_map(mapping: dict[str, Any], scalar: str, map_key: str, unit: str, label: str, *, required: bool = False):
        chosen = _one_of(mapping, (scalar, map_key), label, required=required)
        if chosen is None:
            return None
        if chosen == scalar:
            return _number(mapping[chosen], f"{label}.{chosen}", positive=required, nonnegative=not required)
        return auxiliary_data(load(mapping[chosen], f"{label}.{chosen}"), reference, unit)

    speed = scalar_or_map(kinetics, "speed_km_s", "speed_map", "km / s", "kinematics", required=True)
    speed_error = scalar_or_map(kinetics, "speed_error_km_s", "speed_error_map", "km / s", "kinematics")
    distance_error = scalar_or_map(geometry, "distance_error_kpc", "distance_error_map", "kpc", "geometry")
    clock_keys = ("inclination_deg", "inclination_error_deg", "expansion_index", "expansion_index_error")
    clock = compute_kinematic_age(distance, speed, mode=mode,
        distance_frame=geometry["distance_frame"], speed_frame=kinetics["speed_frame"],
        distance_error_kpc=distance_error, speed_error_km_s=speed_error,
        **{key: kinetics[key] for key in clock_keys if key in kinetics})
    valid = spectrum.valid & clock.valid & mask
    if not np.any(valid):
        raise MapInputError("no pixels pass both spectral detection and kinematic validity checks")
    age = np.where(valid, clock.age_myr, np.nan)
    error = np.where(valid, clock.error_myr, np.nan) if clock.error_myr is not None else None
    warnings = [
        "Ages condition on supplied speed, geometry, and expansion history; unprovided systematic errors are excluded.",
        "Pixels within a restoring beam and calibration errors across pixels are correlated; pixel counts are not independent sample sizes.",
        "Matching WCS/beam does not establish matched uv coverage, astrometric fidelity, or missing-flux recovery.",
        "Spectral-index diagnostics are not spectral-age estimates and do not determine the kinematic clock.",
    ]
    if mode == "flow":
        warnings.append("Flow travel times use path-effective speeds and are not automatically total source dynamical ages.")
    else:
        warnings.append("Expansion ages assume R proportional to t**m from a common zero-time origin with the supplied current radial speed.")
    if error is None:
        warnings.append("No kinematic measurement errors supplied; an age-error layer is unavailable.")
    metadata = {
        "schema_version": "fr2dynage-age-map-1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"name": "PyRADAGE", "pyradage": __version__, "fr2dynage": __version__, "python": platform.python_version(), "numpy": np.__version__, "astropy": astropy_version},
        "source": {**source, "radio_class": radio_class, "activity_state": activity},
        "clock": {"kind": "conditional_dynamical_expansion_age" if mode == "expansion" else "conditional_flow_travel_time",
                  "mode": mode, "equation": "t = m R / v_current" if mode == "expansion" else "t = L / v_effective",
                  "frame": "host/source proper time", "configuration": kinetics, "geometry": geometry,
                  "distance_details": distance_details,
                  "uncertainty": "independent first-order propagation of supplied errors only; no covariance or model-systematic terms"},
        "spectral_diagnostics": {"model": "I_nu proportional to nu**(-alpha)", "reference_frequency_mhz": spectrum.reference_frequency_mhz,
            "frequencies_mhz": frequencies, "selection": selection,
            "errors": "high-SNR log-power approximation with supplied thermal and independent per-band calibration errors"},
        "input_configuration": config, "configuration_path": str(path), "configuration_sha256": hashlib.sha256(raw).hexdigest(),
        "input_files": records, "restoring_beam": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in beam.items()},
        "shape_yx": list(shape), "valid_pixels": int(valid.sum()), "invalid_pixels": int(valid.size - valid.sum()),
        "age_range_myr": [float(np.nanmin(age)), float(np.nanmax(age))],
        "spectral_flag_meanings": SPECTRAL_FLAGS, "kinematic_flag_meanings": KINEMATIC_FLAG_MEANINGS,
        "warnings": warnings,
    }
    # Reject nonfinite provenance before any output artifact is touched.
    json.dumps(metadata, allow_nan=False)
    return AgeMapResult(age, error, valid, clock.flags, spectrum, distance,
                        np.broadcast_to(speed, shape).copy(), reference, metadata, frozenset(inputs))


def _output_path(result: AgeMapResult, path: str | Path, overwrite: bool) -> Path:
    target = Path(path).expanduser().resolve()
    if target in result.input_paths:
        raise MapInputError("output must not overwrite an input image or configuration")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; use --overwrite to replace it")
    return target


def write_age_map(result: AgeMapResult, path: str | Path, *, overwrite: bool = False) -> None:
    """Write the clock, diagnostics and complete provenance in one atomic FITS."""

    fits, _, _ = _astropy()
    target = _output_path(result, path, overwrite)
    header = result.reference.header.copy()
    mode = result.metadata["clock"]["mode"]
    header["BUNIT"] = "Myr"
    header["EXTNAME"] = "DYNAGE" if mode == "expansion" else "FLOWTIME"
    header["AGEKIND"] = "EXPANSION" if mode == "expansion" else "FLOWTIME"
    header["RADCLASS"] = result.metadata["source"]["radio_class"]
    header["DYNMODEL"] = "KINEMATIC"
    header["VERSION"] = result.metadata["software"]["fr2dynage"]
    header["HISTORY"] = "Conditional clock from independently supplied speed and geometry."
    header["HISTORY"] = "See PROVENANCE for assumptions, input hashes and uncertainty scope."
    hdus = [fits.PrimaryHDU(result.age_myr, header=header)]

    def layer(name, data, unit):
        layer_header = result.reference.header.copy()
        layer_header["BUNIT"] = unit
        hdus.append(fits.ImageHDU(data=data, header=layer_header, name=name))

    if result.error_myr is not None:
        layer("AGE_ERR", result.error_myr, "Myr")
    layer("ALPHA", result.spectral.alpha, "1")
    layer("ALPHA_ERR", result.spectral.alpha_error, "1")
    layer("CHI2", result.spectral.chi2, "1")
    layer("REDCHI2", result.spectral.reduced_chi2, "1")
    layer("N_BANDS", result.spectral.n_bands, "1")
    layer("VALID", result.valid.astype(np.uint8), "1")
    layer("SPEC_FLAGS", result.spectral.flags, "1")
    layer("KIN_FLAGS", result.kinematic_flags, "1")
    layer("DIST_KPC", result.distance_kpc, "kpc")
    layer("SPEED_KMS", result.speed_km_s, "km s-1")
    encoded = json.dumps(result.metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    column = fits.Column(name="JSON_UTF8", format=f"{len(encoded)}B", array=np.frombuffer(encoded, dtype=np.uint8)[None, :])
    hdus.append(fits.BinTableHDU.from_columns([column], name="PROVENANCE"))
    with fits.HDUList(hdus) as document:
        atomic_write_fits(target, document, overwrite=overwrite)


def _preview_targets(path: str | Path) -> list[tuple[str, Path]]:
    """Return the raster preview and publication-vector companion paths."""

    primary = Path(path).expanduser().resolve()
    targets = [("png", primary), ("pdf", primary.with_suffix(".pdf")),
               ("svg", primary.with_suffix(".svg"))]
    if len({target for _, target in targets}) != len(targets):
        raise MapInputError("use a .png preview filename; PDF and SVG companions are written separately")
    return targets


def render_age_map(result: AgeMapResult, path: str | Path, *, overwrite: bool = False) -> None:
    """Render large-type WCS PNG, PDF and SVG age/diagnostic figures."""

    try:
        from .plotting import (publication_figure, save_publication_figure,
                               AXIS_LABEL_SIZE, TICK_SIZE)
    except ImportError as exc:
        raise MapInputError("PNG previews require python -m pip install 'pyradage[maps]'") from exc
    targets = [(kind, _output_path(result, target, overwrite))
               for kind, target in _preview_targets(path)]
    for _, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    flow = result.metadata["clock"]["mode"] == "flow"
    panels = [(result.age_myr, "Flow travel time" if flow else "Conditional expansion age", "Myr", "viridis")]
    if result.error_myr is not None:
        panels.append((result.error_myr, "Supplied-error\npropagation (1σ)", "Myr", "magma"))
    panels.append((np.where(result.valid, result.spectral.alpha, np.nan), "Spectral index α", "", "cividis"))
    try:
        figure = publication_figure(figsize=(6.8 * len(panels), 6.5), layout="constrained")
    except ImportError as exc:
        raise MapInputError("publication previews require python -m pip install 'pyradage[maps]'") from exc
    for index, (data, title, unit, cmap) in enumerate(panels, 1):
        axis = figure.add_subplot(1, len(panels), index, projection=result.reference.wcs)
        shown = axis.imshow(data, origin="lower", cmap=cmap, interpolation="nearest")
        axis.set_title(title)
        labels = {"pos.eq.ra": "Right ascension", "pos.eq.dec": "Declination",
                  "pos.galactic.lon": "Galactic longitude", "pos.galactic.lat": "Galactic latitude"}
        for coordinate_index, physical_type in enumerate(result.reference.wcs.world_axis_physical_types):
            if physical_type in labels:
                coordinate = axis.coords[coordinate_index]
                coordinate.set_axislabel(labels[physical_type], fontsize=AXIS_LABEL_SIZE)
                coordinate.set_ticklabel(size=TICK_SIZE, exclude_overlapping=True)
                coordinate.set_ticks(number=4)
        colorbar = figure.colorbar(shown, ax=axis, label=unit, shrink=0.8, pad=.025)
        colorbar.ax.tick_params(labelsize=TICK_SIZE)
        colorbar.set_label(unit, fontsize=AXIS_LABEL_SIZE)
    figure.suptitle(f"{result.metadata['source']['name']} — {result.metadata['source']['radio_class']}")
    staged = []
    try:
        for kind, target in targets:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix="."+kind, dir=target.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            staged.append((temporary, target))
            save_publication_figure(figure, temporary, formats=(kind,), clear=False)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        for temporary, target in staged:
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        figure.clear()
