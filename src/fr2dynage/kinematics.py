"""Conditional expansion-age and flow-travel-time maps.

For radial self-similar expansion R proportional to t**m, the current
expansion speed gives t=m*R/v. A flow clock instead uses path length divided
by a path-effective speed. These are host-frame kinematic clocks; neither is
an inference of particle spectral age, and a flow clock is not automatically
the total source age. Inclination is measured to the line of sight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import numpy as np

from .constants import C_LIGHT, KPC_M, MYR_S


INVALID_DISTANCE = np.uint16(1)
INVALID_SPEED = np.uint16(2)
INVALID_UNCERTAINTY = np.uint16(4)
NUMERICAL_RANGE = np.uint16(8)

KINEMATIC_FLAG_MEANINGS = {
    int(INVALID_DISTANCE): "distance is nonfinite or negative",
    int(INVALID_SPEED): "speed is nonfinite, nonpositive, or implies a physical speed above c",
    int(INVALID_UNCERTAINTY): "a supplied uncertainty is nonfinite or negative",
    int(NUMERICAL_RANGE): "age or uncertainty is outside representable floating-point range",
}


class KinematicMapError(ValueError):
    """Raised for invalid clock assumptions or an entirely invalid map."""


@dataclass(frozen=True)
class KinematicAgeMap:
    age_myr: np.ndarray
    error_myr: np.ndarray | None
    valid: np.ndarray
    flags: np.ndarray


def _scalar(value: object, label: str, *, positive: bool = False) -> float:
    try:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise KinematicMapError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise KinematicMapError(f"{label} must be {requirement}")
    return result


def _array(value: object, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise TypeError
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise KinematicMapError(f"{label} must contain real numeric values, not booleans or strings") from exc


def compute_kinematic_age(
    distance_kpc: object,
    speed_km_s: object,
    *,
    mode: Literal["expansion", "flow"],
    expansion_index: float | None = None,
    distance_frame: Literal["projected", "intrinsic"] = "projected",
    speed_frame: Literal["projected", "intrinsic"] = "intrinsic",
    inclination_deg: float | None = None,
    distance_error_kpc: object | None = None,
    speed_error_km_s: object | None = None,
    inclination_error_deg: float | None = None,
    expansion_index_error: float | None = None,
) -> KinematicAgeMap:
    """Calculate conditional clocks and independent first-order errors.

    ``distance_kpc`` means radius for expansion and path length for flow.
    ``speed_km_s`` means current radial expansion speed or path-effective flow
    speed respectively. A per-pixel flow speed is only appropriate if it
    represents that pixel's effective speed over the complete path.

    Distance and speed arrays broadcast to a shared shape; uncertainty arrays
    must broadcast to that shape without expanding it. Bad pixels are flagged
    and replaced by NaN. Speed is a physical host-frame speed, not an apparent
    superluminal proper motion. A scalar inclination is required only when
    distance and speed use different projected/intrinsic conventions.
    A projected speed with a supplied inclination is checked against c after
    deprojection, even when the angle cancels from the age ratio.

    If both quantities use the same projection convention, inclination
    cancels exactly. A supplied angle is still validated, but has no effect
    and contributes no uncertainty in that case. Returned errors condition
    on all unsupplied uncertainties being fixed; they are not total errors.
    """

    if not isinstance(mode, str) or mode not in ("expansion", "flow"):
        raise KinematicMapError("mode must be 'expansion' or 'flow'")
    if (
        not isinstance(distance_frame, str) or distance_frame not in ("projected", "intrinsic")
        or not isinstance(speed_frame, str) or speed_frame not in ("projected", "intrinsic")
    ):
        raise KinematicMapError("distance_frame and speed_frame must be 'projected' or 'intrinsic'")
    if mode == "expansion":
        if expansion_index is None:
            raise KinematicMapError("expansion requires an explicit expansion_index in R proportional to t**m")
        exponent = _scalar(expansion_index, "expansion_index", positive=True)
    else:
        if expansion_index is not None or expansion_index_error is not None:
            raise KinematicMapError("expansion_index and expansion_index_error are only valid for expansion")
        exponent = 1.0

    different_frames = distance_frame != speed_frame
    if different_frames and inclination_deg is None:
        raise KinematicMapError("inclination_deg is required when distance and speed use different frames")
    angle = None
    if inclination_deg is not None:
        angle = _scalar(inclination_deg, "inclination_deg", positive=True)
        if angle > 90.0:
            raise KinematicMapError("inclination_deg must lie in (0, 90]")
    if inclination_error_deg is not None and angle is None:
        raise KinematicMapError("inclination_error_deg requires inclination_deg")
    angle_error = None if inclination_error_deg is None else _scalar(inclination_error_deg, "inclination_error_deg")
    if angle_error is not None and angle_error < 0.0:
        raise KinematicMapError("inclination_error_deg cannot be negative")
    exponent_error = None if expansion_index_error is None else _scalar(expansion_index_error, "expansion_index_error")
    if exponent_error is not None and exponent_error < 0.0:
        raise KinematicMapError("expansion_index_error cannot be negative")

    sine = math.sin(math.radians(angle)) if different_frames else 1.0
    # The angle may be positive but too close to zero to survive radians().
    if sine <= 0.0 or not math.isfinite(sine):
        raise KinematicMapError("inclination is too small to represent its projection")
    projection_log = 0.0
    if different_frames:
        projection_log = -math.log(sine) if distance_frame == "projected" else math.log(sine)
    log_coefficient = math.log(KPC_M / (1000.0 * MYR_S)) + math.log(exponent) + projection_log

    distance = _array(distance_kpc, "distance_kpc")
    speed = _array(speed_km_s, "speed_km_s")
    try:
        distance, speed = np.broadcast_arrays(distance, speed)
    except ValueError as exc:
        raise KinematicMapError("distance and speed shapes cannot be broadcast together") from exc
    if distance.size == 0:
        raise KinematicMapError("distance and speed maps cannot be empty")
    shape = distance.shape
    flags = np.zeros(shape, dtype=np.uint16)
    invalid_distance = ~np.isfinite(distance) | (distance < 0.0)
    maximum_speed = C_LIGHT / 1000.0
    if speed_frame == "projected" and angle is not None:
        maximum_speed *= math.sin(math.radians(angle))
    invalid_speed = ~np.isfinite(speed) | (speed <= 0.0) | (speed > maximum_speed)
    flags[invalid_distance] |= INVALID_DISTANCE
    flags[invalid_speed] |= INVALID_SPEED

    uncertainty_fields: dict[str, np.ndarray] = {}
    for label, supplied in (("distance_error_kpc", distance_error_kpc), ("speed_error_km_s", speed_error_km_s)):
        if supplied is None:
            continue
        raw_error = _array(supplied, label)
        try:
            error = np.broadcast_to(raw_error, shape)
        except ValueError as exc:
            raise KinematicMapError(f"{label} must broadcast to the distance/speed map shape") from exc
        flags[~np.isfinite(error) | (error < 0.0)] |= INVALID_UNCERTAINTY
        uncertainty_fields[label] = error
    valid = flags == 0
    positive_distance = valid & (distance > 0.0)
    zero_distance = valid & (distance == 0.0)
    age = np.full(shape, np.nan, dtype=float)
    log_age = np.full(shape, -np.inf, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_age[positive_distance] = log_coefficient + np.log(distance[positive_distance]) - np.log(speed[positive_distance])
        age[positive_distance] = np.exp(log_age[positive_distance])
    age[zero_distance] = 0.0
    flags[valid & (~np.isfinite(age) | ((distance > 0.0) & (age <= 0.0)))] |= NUMERICAL_RANGE

    supplied_any_error = bool(uncertainty_fields) or angle_error is not None or exponent_error is not None
    error_map = None
    if supplied_any_error:
        terms = []
        meaningful = np.zeros(shape, dtype=bool)

        def logarithmic_term(log_prefactor: np.ndarray | float, supplied: np.ndarray | float, relevant: np.ndarray) -> None:
            contribution = np.zeros(shape, dtype=float)
            relevant = relevant & (supplied > 0.0)
            meaningful[...] |= relevant
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                full_log = log_prefactor + np.log(supplied)
                contribution[relevant] = np.exp(np.broadcast_to(full_log, shape)[relevant])
            terms.append(contribution)

        if "distance_error_kpc" in uncertainty_fields:
            logarithmic_term(log_coefficient - np.log(np.where(speed > 0.0, speed, 1.0)), uncertainty_fields["distance_error_kpc"], valid)
        if "speed_error_km_s" in uncertainty_fields:
            logarithmic_term(log_age - np.log(np.where(speed > 0.0, speed, 1.0)), uncertainty_fields["speed_error_km_s"], positive_distance)
        if exponent_error is not None:
            logarithmic_term(log_age - math.log(exponent), exponent_error, positive_distance)
        if angle_error is not None:
            cotangent = 0.0 if not different_frames or angle == 90.0 else abs(math.cos(math.radians(angle)) / sine)
            if cotangent > 0.0 and angle_error > 0.0:
                logarithmic_term(log_age + math.log(cotangent) + math.log(math.pi / 180.0), angle_error, positive_distance)
            else:
                terms.append(np.zeros(shape, dtype=float))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            error_map = np.asarray(np.hypot.reduce(terms), dtype=float)
        flags[valid & (~np.isfinite(error_map) | ((error_map == 0.0) & meaningful))] |= NUMERICAL_RANGE

    valid = flags == 0
    age[~valid] = np.nan
    if error_map is not None:
        error_map[~valid] = np.nan
    if not np.any(valid):
        raise KinematicMapError("no valid pixels remain after checking distances, speeds, uncertainties, and numerical range")
    return KinematicAgeMap(age, error_map, valid, flags)
