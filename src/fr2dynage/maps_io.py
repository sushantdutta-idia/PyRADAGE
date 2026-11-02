"""Validated, optional FITS image I/O for spatial radio-source analysis.

Image preparation remains explicit: this module checks registration and the
restoring beam; it does not reproject or convolve measurements silently.
Astropy is loaded only when a FITS operation is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


class MapInputError(ValueError):
    """An image or its metadata cannot support the requested map analysis."""


def _astropy():
    try:
        from astropy import units
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        raise MapInputError(
            "FITS mapping requires optional dependencies; install with "
            "python -m pip install 'pyradage[maps]'"
        ) from exc
    return fits, WCS, units


@dataclass(frozen=True)
class FitsPlane:
    data: np.ndarray
    header: Any
    wcs: Any
    path: Path
    sha256: str
    hdu_index: int = 0


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fits_plane(path: str | Path, hdu: int | str = 0) -> FitsPlane:
    """Read one celestial plane, allowing only singleton leading cube axes.

    FITS pixel axes 1 and 2 must be the celestial axes. A genuinely
    multichannel/stokes cube requires an explicit plane extraction first.
    Nonfinite science pixels are retained for downstream quality masks.
    """

    fits, WCS, _ = _astropy()
    from astropy.io.fits.verify import VerifyError
    source = Path(path).expanduser().resolve()
    if isinstance(hdu, bool) or not isinstance(hdu, (int, str)) or (
        isinstance(hdu, int) and hdu < 0
    ):
        raise MapInputError("FITS HDU must be a non-negative integer or extension name")
    try:
        before = source.stat()
        checksum = _file_digest(source)
        with fits.open(source, memmap=False) as hdus:
            try:
                selected = hdus[hdu]
                hdu_index = hdus.index_of(hdu)
            except (IndexError, KeyError) as exc:
                raise MapInputError(f"{source}: FITS HDU {hdu!r} does not exist") from exc
            if selected.data is None:
                raise MapInputError(f"{source}: selected FITS HDU has no image")
            data = np.asarray(selected.data)
            if data.ndim < 2 or any(size != 1 for size in data.shape[:-2]):
                raise MapInputError(
                    f"{source}: require a 2D celestial image or leading singleton "
                    "axes; extract a plane from a non-singleton cube first"
                )
            if min(data.shape[-2:]) < 1 or data.dtype.kind not in "iuf":
                raise MapInputError(f"{source}: selected FITS HDU is not a real-valued image")
            original_header = selected.header.copy()
            full_wcs = WCS(original_header)
            if not full_wcs.has_celestial or {
                full_wcs.wcs.lng, full_wcs.wcs.lat
            } != {0, 1}:
                raise MapInputError(
                    f"{source}: FITS pixel axes 1 and 2 require a celestial WCS"
                )
            wcs = full_wcs.celestial
            if wcs.pixel_n_dim != 2:
                raise MapInputError(f"{source}: celestial WCS must have two pixel axes")
            plane = np.array(data.reshape(data.shape[-2:]), dtype=float, copy=True)
            header = wcs.to_header(relax=True)
            for key in ("BUNIT", "BMAJ", "BMIN", "BPA", "OBJECT", "TELESCOP", "INSTRUME",
                        "KPCAS", "COSMO", "REDSHIFT", "DISTFRAM", "H0", "OM0", "TCMB0"):
                if key in original_header:
                    header[key] = (original_header[key], original_header.comments[key])
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise MapInputError(f"{source}: FITS file changed while it was being read")
        # Invalid celestial projections should fail before a derived map is made.
        ny, nx = plane.shape
        points = np.array([[0.0, 0.0], [nx - 1.0, ny - 1.0],
                           [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0]])
        world = wcs.pixel_to_world_values(points[:, 0], points[:, 1])
        if not np.all(np.isfinite(np.asarray(world))):
            raise MapInputError(f"{source}: celestial WCS is undefined across the image")
    except MapInputError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, EOFError, VerifyError) as exc:
        raise MapInputError(f"cannot read FITS plane {source}: {exc}") from exc
    return FitsPlane(plane, header, wcs, source, checksum, hdu_index)


def _unit(value: object, *, path: Path):
    _, _, units = _astropy()
    if not isinstance(value, str) or not value.strip():
        raise MapInputError(f"{path}: BUNIT is required")
    try:
        return units.Unit(value)
    except ValueError:
        # Common radio FITS writers capitalize the Jy and beam tokens. Keep
        # SI prefixes case-sensitive (MJy and mJy differ by a factor of 1e9).
        normalized = value.replace("µ", "u").replace("μ", "u")
        normalized = re.sub("jy", "Jy", normalized, flags=re.IGNORECASE)
        normalized = re.sub("beam", "beam", normalized, flags=re.IGNORECASE)
        try:
            return units.Unit(normalized)
        except ValueError as exc:
            raise MapInputError(f"{path}: unsupported BUNIT {value!r}") from exc


def _finite_header_float(plane: FitsPlane, key: str) -> float:
    value = plane.header.get(key)
    try:
        if isinstance(value, (bool, str)) or value is None:
            raise ValueError("not a numeric value")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("not finite")
    except (TypeError, ValueError, OverflowError) as exc:
        raise MapInputError(f"{plane.path}: {key} must be a finite numeric FITS keyword") from exc
    return result


def science_image(plane: FitsPlane) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a Jy/beam image and its restoring-beam covariance in deg²."""

    _, _, units = _astropy()
    source_unit = _unit(plane.header.get("BUNIT"), path=plane.path)
    try:
        conversion = source_unit.to(units.Jy / units.beam)
    except units.UnitConversionError as exc:
        raise MapInputError(f"{plane.path}: science BUNIT must be convertible to Jy/beam") from exc
    major = _finite_header_float(plane, "BMAJ")
    minor = _finite_header_float(plane, "BMIN")
    if major <= 0.0 or minor <= 0.0:
        raise MapInputError(f"{plane.path}: BMAJ and BMIN must be positive degrees")
    if "BPA" in plane.header:
        position_angle = _finite_header_float(plane, "BPA") % 180.0
    elif math.isclose(major, minor, rel_tol=1e-8):
        position_angle = 0.0
    else:
        raise MapInputError(f"{plane.path}: elliptical restoring beam requires BPA in degrees")
    angle = math.radians(position_angle)
    major_axis = np.array([math.sin(angle), math.cos(angle)])
    minor_axis = np.array([math.cos(angle), -math.sin(angle)])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        covariance = (
            np.outer(major_axis, major_axis) * np.square(major)
            + np.outer(minor_axis, minor_axis) * np.square(minor)
        ) / (8.0 * math.log(2.0))
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
        raise MapInputError(f"{plane.path}: restoring beam is outside the representable range")
    return plane.data * conversion, {
        "bmaj_deg": major,
        "bmin_deg": minor,
        "bpa_deg": position_angle,
        "covariance_deg2": covariance,
    }


def ensure_same_beam(reference: FitsPlane, candidate: FitsPlane) -> None:
    """Require equivalent Gaussian restoring beams, including PA periodicity."""

    _, first = science_image(reference)
    _, second = science_image(candidate)
    covariance = first["covariance_deg2"]
    comparison = second["covariance_deg2"]
    scale = float(max(np.max(np.abs(covariance)), np.max(np.abs(comparison))))
    if not np.allclose(covariance, comparison, rtol=1e-4, atol=scale * 1e-6):
        raise MapInputError(
            f"{candidate.path}: restoring beam differs from {reference.path}; "
            "convolve images to a common beam before mapping"
        )


def validate_registration(
    reference: FitsPlane,
    candidate: FitsPlane,
    tolerance_pixels: float = 0.01,
) -> None:
    """Check every pixel's celestial correspondence in bounded-size chunks."""

    _astropy()
    from astropy.wcs import NoConvergence

    if isinstance(tolerance_pixels, bool) or not isinstance(tolerance_pixels, (int, float)):
        raise MapInputError("registration tolerance must be finite and positive")
    try:
        valid_tolerance = math.isfinite(tolerance_pixels) and tolerance_pixels > 0
    except OverflowError:
        valid_tolerance = False
    if not valid_tolerance:
        raise MapInputError("registration tolerance must be finite and positive")
    if reference.data.shape != candidate.data.shape:
        raise MapInputError(f"{candidate.path}: image shape differs from the reference image")
    ny, nx = reference.data.shape
    count = nx * ny
    try:
        for start in range(0, count, 65536):
            indices = np.arange(start, min(start + 65536, count))
            x = (indices % nx).astype(float)
            y = (indices // nx).astype(float)
            celestial = reference.wcs.pixel_to_world(x, y)
            candidate_x, candidate_y = candidate.wcs.world_to_pixel(celestial)
            difference = np.hypot(candidate_x - x, candidate_y - y)
            if np.any(~np.isfinite(difference)) or np.any(difference > tolerance_pixels):
                raise MapInputError(
                    f"{candidate.path}: celestial WCS is not registered to the reference "
                    f"within {tolerance_pixels:g} pixel; align images before mapping"
                )
    except MapInputError:
        raise
    except (ValueError, TypeError, OverflowError, NoConvergence) as exc:
        raise MapInputError(f"{candidate.path}: cannot compare celestial WCS: {exc}") from exc


def auxiliary_data(plane: FitsPlane, reference: FitsPlane, unit: str | None) -> np.ndarray:
    """Read a registered dimensional map, or a binary inclusion mask.

    NaN mask pixels are excluded; other finite mask values must be exactly
    zero or one. Dimensional-map nonfinite pixels remain available for
    downstream invalid-input flags.
    """

    validate_registration(reference, plane)
    if unit is None:
        finite = np.isfinite(plane.data)
        if np.any(np.isinf(plane.data)) or np.any(finite & (plane.data != 0) & (plane.data != 1)):
            raise MapInputError(f"{plane.path}: inclusion mask must contain only 0, 1, or NaN")
        return finite & (plane.data == 1)
    _, _, units = _astropy()
    source_unit = _unit(plane.header.get("BUNIT"), path=plane.path)
    try:
        factor = source_unit.to(units.Unit(unit))
    except (ValueError, units.UnitConversionError) as exc:
        raise MapInputError(f"{plane.path}: BUNIT must be convertible to {unit}") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        return plane.data * factor


def atomic_write_fits(path: str | Path, hdulist: Any, overwrite: bool = False) -> None:
    """Flush a complete FITS file before publishing it without partial writes.

    The default uses an atomic hard link to refuse replacement even if
    another process creates the destination after the initial check.
    """

    _astropy()
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(f"output already exists: {target}; request overwrite explicitly")
    suffix = "".join(target.suffixes) or ".fits"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix, dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        hdulist.writeto(temporary, overwrite=True, output_verify="exception", checksum=True)
        with temporary.open("rb") as complete:
            os.fsync(complete.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A cleanup failure must not mask the original write failure.
            pass
