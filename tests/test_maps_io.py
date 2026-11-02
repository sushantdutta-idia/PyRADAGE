from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits
from astropy.wcs import WCS

from fr2dynage import maps_io
from fr2dynage.maps_io import (
    MapInputError,
    atomic_write_fits,
    auxiliary_data,
    ensure_same_beam,
    read_fits_plane,
    science_image,
    validate_registration,
)


def _header():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [3.0, 3.0]
    wcs.wcs.crval = [180.0, -30.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = 5.0 / 3600.0
    header["BMIN"] = 3.0 / 3600.0
    header["BPA"] = 15.0
    return header


def _plane(tmp_path, name="radio.fits", *, data=None, header=None):
    target = tmp_path / name
    fits.PrimaryHDU(np.ones((5, 6)) if data is None else data,
                    header=_header() if header is None else header).writeto(target)
    return read_fits_plane(target)


def test_fits_read_preserves_pixels_wcs_and_exact_file_hash(tmp_path):
    data = np.arange(30.0).reshape(5, 6)
    data[1, 2] = np.nan
    plane = _plane(tmp_path, data=data)
    np.testing.assert_equal(plane.data, data)
    assert plane.sha256 == hashlib.sha256(plane.path.read_bytes()).hexdigest()
    assert plane.wcs.has_celestial
    assert plane.header["BUNIT"] == "Jy/beam"
    assert plane.path.is_absolute()


def test_singleton_radio_cube_becomes_a_2d_plane(tmp_path):
    header = _header()
    header["WCSAXES"] = 4
    for axis, kind, value in ((3, "FREQ", 1.4e9), (4, "STOKES", 1.0)):
        header[f"CTYPE{axis}"] = kind
        header[f"CRVAL{axis}"] = value
        header[f"CRPIX{axis}"] = 1.0
        header[f"CDELT{axis}"] = 1.0
    plane = _plane(tmp_path, data=np.ones((1, 1, 5, 6)), header=header)
    assert plane.data.shape == (5, 6)
    assert plane.header["WCSAXES"] == 2
    assert "CTYPE3" not in plane.header


def test_multichannel_cube_is_not_silently_averaged(tmp_path):
    with pytest.raises(MapInputError, match="non-singleton cube"):
        _plane(tmp_path, data=np.ones((2, 5, 6)))


def test_coupled_celestial_and_frequency_axes_are_rejected(tmp_path):
    header = _header()
    header["WCSAXES"] = 3
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = 1.4e9
    header["CRPIX3"] = 1.0
    header["CDELT3"] = 1e6
    header["PC1_3"] = 0.01
    with pytest.raises(MapInputError, match="cannot read FITS"):
        _plane(tmp_path, data=np.ones((1, 5, 6)), header=header)


def test_image_requires_celestial_wcs(tmp_path):
    with pytest.raises(MapInputError, match="celestial WCS"):
        _plane(tmp_path, header=fits.Header())


def test_missing_hdu_has_a_clear_error(tmp_path):
    plane = _plane(tmp_path)
    with pytest.raises(MapInputError, match="does not exist"):
        read_fits_plane(plane.path, hdu=8)


def test_hdu_aliases_have_the_same_canonical_index(tmp_path):
    plane = _plane(tmp_path)
    by_name = read_fits_plane(plane.path, hdu="PRIMARY")
    assert plane.hdu_index == by_name.hdu_index == 0


@pytest.mark.parametrize("unit,factor", [
    ("Jy/beam", 1.0), ("JY/BEAM", 1.0), ("mJy/beam", 1e-3),
    ("uJy/beam", 1e-6), ("MJy/beam", 1e6), ("mJY BEAM-1", 1e-3),
])
def test_science_image_converts_radio_units_without_losing_prefix_case(tmp_path, unit, factor):
    header = _header()
    header["BUNIT"] = unit
    plane = _plane(tmp_path, header=header)
    values, beam = science_image(plane)
    assert values == pytest.approx(np.ones((5, 6)) * factor)
    assert beam["covariance_deg2"].shape == (2, 2)


@pytest.mark.parametrize("mutation", [
    {"BUNIT": "K"}, {"BUNIT": None}, {"BMAJ": -1.0}, {"BMIN": 0.0},
    {"BPA": None}, {"BMAJ": "invalid"},
])
def test_invalid_science_unit_or_beam_is_rejected(tmp_path, mutation):
    header = _header()
    for key, value in mutation.items():
        if value is None:
            del header[key]
        else:
            header[key] = value
    plane = _plane(tmp_path, header=header)
    with pytest.raises(MapInputError):
        science_image(plane)


def test_circular_beam_does_not_require_position_angle(tmp_path):
    header = _header()
    header["BMIN"] = header["BMAJ"]
    del header["BPA"]
    values, beam = science_image(_plane(tmp_path, header=header))
    assert beam["bpa_deg"] == 0.0


@pytest.mark.parametrize("size", [1e200, 1e-200])
def test_unrepresentable_beam_covariance_is_rejected(tmp_path, size):
    header = _header()
    header["BMAJ"] = header["BMIN"] = size
    with pytest.raises(MapInputError, match="representable range"):
        science_image(_plane(tmp_path, header=header))


def test_equivalent_beam_position_angles_and_swapped_axes_match(tmp_path):
    reference = _plane(tmp_path)
    header = _header()
    header["BPA"] += 180.0
    same = _plane(tmp_path, "periodic.fits", header=header)
    ensure_same_beam(reference, same)
    header["BMAJ"], header["BMIN"] = header["BMIN"], header["BMAJ"]
    header["BPA"] += 90.0
    swapped = _plane(tmp_path, "swapped.fits", header=header)
    ensure_same_beam(reference, swapped)


def test_different_restoring_beams_are_rejected(tmp_path):
    reference = _plane(tmp_path)
    header = _header()
    header["BMAJ"] *= 1.05
    candidate = _plane(tmp_path, "different.fits", header=header)
    with pytest.raises(MapInputError, match="common beam"):
        ensure_same_beam(reference, candidate)


def test_registered_images_pass_but_shifted_grid_fails(tmp_path):
    reference = _plane(tmp_path)
    candidate = _plane(tmp_path, "same.fits")
    validate_registration(reference, candidate)
    header = _header()
    header["CRPIX1"] += 0.1
    shifted = _plane(tmp_path, "shifted.fits", header=header)
    with pytest.raises(MapInputError, match="not registered"):
        validate_registration(reference, shifted)


def test_registration_compares_interior_pixels_too(tmp_path):
    reference = _plane(tmp_path)

    class InteriorDistortion:
        def world_to_pixel(self, celestial):
            x, y = reference.wcs.world_to_pixel(celestial)
            return x + np.where((np.abs(x - 2) < 1e-5) & (np.abs(y - 2) < 1e-5), 0.2, 0), y

    candidate = replace(reference, wcs=InteriorDistortion())
    with pytest.raises(MapInputError, match="not registered"):
        validate_registration(reference, candidate)


def test_wcs_inverse_nonconvergence_is_a_clean_input_error(tmp_path):
    from astropy.wcs import NoConvergence
    reference = _plane(tmp_path)

    class FailingInverse:
        def world_to_pixel(self, celestial):
            raise NoConvergence("inverse failed")

    with pytest.raises(MapInputError, match="cannot compare celestial WCS"):
        validate_registration(reference, replace(reference, wcs=FailingInverse()))


def test_registration_rejects_different_shapes(tmp_path):
    reference = _plane(tmp_path)
    candidate = _plane(tmp_path, "shape.fits", data=np.ones((6, 5)))
    with pytest.raises(MapInputError, match="shape differs"):
        validate_registration(reference, candidate)


def test_auxiliary_velocity_and_mask_maps(tmp_path):
    reference = _plane(tmp_path)
    header = _header()
    header["BUNIT"] = "m/s"
    velocity = _plane(tmp_path, "velocity.fits", data=np.ones((5, 6)) * 1000, header=header)
    assert auxiliary_data(velocity, reference, "km/s") == pytest.approx(np.ones((5, 6)))
    data = np.ones((5, 6))
    data[0, 0] = 0
    data[0, 1] = np.nan
    mask = _plane(tmp_path, "mask.fits", data=data)
    selected = auxiliary_data(mask, reference, None)
    assert selected.dtype == bool
    assert not selected[0, 0] and not selected[0, 1]
    assert selected[1, 1]


def test_auxiliary_maps_reject_wrong_units_and_nonbinary_masks(tmp_path):
    reference = _plane(tmp_path)
    with pytest.raises(MapInputError, match="convertible to km/s"):
        auxiliary_data(reference, reference, "km/s")
    mask = _plane(tmp_path, "mask.fits", data=np.full((5, 6), 0.5))
    with pytest.raises(MapInputError, match="only 0, 1"):
        auxiliary_data(mask, reference, None)


def test_fits_writer_publishes_checksum_and_refuses_unrequested_overwrite(tmp_path):
    target = tmp_path / "out.fits"
    hdus = fits.HDUList([fits.PrimaryHDU(np.ones((2, 3)))])
    atomic_write_fits(target, hdus)
    before = target.read_bytes()
    with fits.open(target, checksum=True) as saved:
        assert saved[0].verify_checksum() == 1
    with pytest.raises(FileExistsError):
        atomic_write_fits(target, hdus)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_write_failure_preserves_old_output_and_cleans_temp(tmp_path, monkeypatch, operation):
    target = tmp_path / "out.fits"
    target.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(maps_io.os, operation, fail)
    with pytest.raises(OSError, match="injected"):
        atomic_write_fits(target, fits.HDUList([fits.PrimaryHDU(np.ones((2, 3)))]), overwrite=True)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_no_overwrite_guards_against_a_racing_writer(tmp_path, monkeypatch):
    target = tmp_path / "out.fits"
    real_link = maps_io.os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"racing writer")
        real_link(source, destination)

    monkeypatch.setattr(maps_io.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        atomic_write_fits(target, fits.HDUList([fits.PrimaryHDU(np.ones((2, 3)))]))
    assert target.read_bytes() == b"racing writer"
    assert list(tmp_path.iterdir()) == [target]


def test_missing_optional_dependency_has_install_guidance(monkeypatch):
    import builtins
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "astropy" or name.startswith("astropy."):
            raise ImportError("missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(MapInputError, match=r"pyradage\[maps\]"):
        maps_io._astropy()
