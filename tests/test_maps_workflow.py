"""End-to-end checks of spatial clocks, class routing, and delivered maps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits
from astropy.wcs import WCS

from fr2dynage.constants import KPC_M, MYR_S
from fr2dynage.io import load_configuration, normalized_input
from fr2dynage.model import KDAForwardModel


ROOT = Path(__file__).resolve().parents[1]


def _header():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [3, 3]
    wcs.wcs.crval = [180, -30]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = header["BMIN"] = 3 / 3600
    header["BPA"] = 0
    return header


@pytest.fixture
def map_config(tmp_path):
    images = []
    for frequency in [150, 325, 610, 1400]:
        data = np.full((5, 5), 0.02 * (frequency / 150) ** -0.7)
        data[0, 0] = np.nan
        path = tmp_path / f"{frequency}.fits"
        fits.PrimaryHDU(data, header=_header()).writeto(path)
        images.append({"path": path.name, "frequency_mhz": frequency,
                       "rms_jy_per_beam": 2e-5, "calibration_fraction": 0.01})
    config = {
        "schema_version": "fr2dynage-map-1",
        "source": {"name": "Analytic map", "radio_class": "fr2", "activity_state": "active"},
        "images": images,
        "selection": {"snr_min": 5, "min_bands": 3, "require_all_bands": True},
        "geometry": {"distance_kind": "radius", "distance_frame": "projected",
                     "origin_pixel": [2, 2], "kpc_per_arcsec": 1,
                     "distance_error_kpc": 0.1},
        "kinematics": {"mode": "expansion", "speed_role": "current_expansion",
                       "speed_km_s": 3000, "speed_error_km_s": 300,
                       "speed_frame": "intrinsic", "inclination_deg": 30,
                       "inclination_error_deg": 2, "expansion_index": 0.75,
                       "expansion_index_error": 0.05,
                       "constraint_description": "Independent analytic test motion"},
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps(config))
    return path, config


def _save(path, config):
    path.write_text(json.dumps(config))


def _flow(config):
    config["geometry"].update(distance_kind="path_length", path_is_straight=True)
    config["kinematics"].update(mode="flow", speed_role="effective_flow")
    del config["kinematics"]["expansion_index"]
    del config["kinematics"]["expansion_index_error"]


def test_expansion_map_recovers_analytic_age_and_independent_errors(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    result = compute_age_map(path)
    coefficient = KPC_M / (1000 * MYR_S)
    expected_age = 0.75 * coefficient / (3000 * math.sin(math.radians(30)))
    assert result.distance_kpc[2, 3] == pytest.approx(1.0, rel=2e-7)
    assert result.age_myr[2, 3] == pytest.approx(expected_age, rel=2e-7)
    relative_error = math.sqrt(0.1**2 + 0.1**2
        + (math.radians(2) / math.tan(math.radians(30)))**2 + (0.05 / 0.75)**2)
    assert result.error_myr[2, 3] == pytest.approx(expected_age * relative_error, rel=2e-7)
    assert result.spectral.alpha[2, 3] == pytest.approx(0.7, abs=1e-12)
    assert result.valid[2, 2] and result.age_myr[2, 2] == 0.0
    assert not result.valid[0, 0] and math.isnan(result.age_myr[0, 0])
    json.dumps(result.metadata, allow_nan=False)


def test_fits_output_preserves_wcs_units_and_mask(map_config, tmp_path):
    from fr2dynage.maps import compute_age_map, write_age_map
    result = compute_age_map(map_config[0])
    output = tmp_path / "ages.fits"
    write_age_map(result, output)
    with fits.open(output, checksum=True) as hdus:
        age_hdu = next(hdu for hdu in hdus if hdu.data is not None
                       and getattr(hdu.data, "shape", None) == result.age_myr.shape
                       and hdu.header.get("BUNIT") == "Myr")
        np.testing.assert_allclose(age_hdu.data, result.age_myr, equal_nan=True)
        assert age_hdu.verify_checksum() == 1
        sky = WCS(age_hdu.header).celestial.pixel_to_world_values(3, 2)
        reference = result.reference.wcs.pixel_to_world_values(3, 2)
        assert sky == pytest.approx(reference)
        assert np.isnan(age_hdu.data[0, 0])
        assert len(hdus) >= 5


def test_flow_clock_is_labelled_as_flow_instead_of_expansion(map_config, tmp_path):
    from fr2dynage.maps import compute_age_map, write_age_map
    path, config = map_config
    _flow(config)
    _save(path, config)
    result = compute_age_map(path)
    expected_age = KPC_M / (1000 * MYR_S) / (3000 * 0.5)
    assert result.age_myr[2, 3] == pytest.approx(expected_age, rel=2e-7)
    output = tmp_path / "flow.fits"
    write_age_map(result, output)
    with fits.open(output) as hdus:
        assert any("FLOW" in str(hdu.header.get("AGETYPE", "")).upper()
                   or "FLOW" in str(hdu.header.get("EXTNAME", "")).upper() for hdu in hdus)


def test_existing_map_is_preserved_without_explicit_overwrite(map_config, tmp_path):
    from fr2dynage.maps import compute_age_map, write_age_map
    result = compute_age_map(map_config[0])
    target = tmp_path / "existing.fits"
    target.write_bytes(b"preserve existing result")
    with pytest.raises((FileExistsError, ValueError)):
        write_age_map(result, target)
    assert target.read_bytes() == b"preserve existing result"


@pytest.mark.parametrize("renderer", ["write_age_map", "render_age_map"])
def test_output_cannot_overwrite_an_input_even_when_requested(map_config, renderer):
    from fr2dynage import maps
    result = maps.compute_age_map(map_config[0])
    target = result.reference.path
    original = target.read_bytes()
    with pytest.raises(ValueError, match="input|source"):
        getattr(maps, renderer)(result, target, overwrite=True)
    assert target.read_bytes() == original


def test_map_config_rejects_duplicate_and_unknown_keys(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    config["unexpected"] = 1
    _save(path, config)
    with pytest.raises(ValueError, match="unknown"):
        compute_age_map(path)
    path.write_text('{"source":{},"source":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        compute_age_map(path)


def test_misaligned_band_is_not_silently_reprojected(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    image = path.parent / config["images"][1]["path"]
    with fits.open(image, mode="update") as hdus:
        hdus[0].header["CRPIX1"] += 0.2
    with pytest.raises(ValueError, match="registered|align"):
        compute_age_map(path)


def test_hdu_alias_does_not_allow_reusing_one_image_at_two_frequencies(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    config["images"][1]["path"] = config["images"][0]["path"]
    config["images"][1]["hdu"] = "PRIMARY"
    _save(path, config)
    with pytest.raises(ValueError, match="same image|independent"):
        compute_age_map(path)


def test_tailed_class_requires_curved_path_map(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    config["source"]["radio_class"] = "WAT"
    _flow(config)
    _save(path, config)
    with pytest.raises(ValueError, match="distance_map|path"):
        compute_age_map(path)


def test_curved_path_flow_map_accepts_wat_class(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    _flow(config)
    config["source"]["radio_class"] = "WAT"
    del config["geometry"]["origin_pixel"]
    del config["geometry"]["kpc_per_arcsec"]
    del config["geometry"]["path_is_straight"]
    config["geometry"]["distance_map"] = "path-length.fits"
    header = _header()
    header["BUNIT"] = "kpc"
    fits.PrimaryHDU(np.full((5, 5), 6.0), header=header).writeto(path.parent / "path-length.fits")
    _save(path, config)
    result = compute_age_map(path)
    expected_age = 6 * KPC_M / (1000 * MYR_S) / (3000 * 0.5)
    assert result.age_myr[2, 3] == pytest.approx(expected_age)


def test_speed_map_invalid_pixels_are_nan_without_discarding_good_pixels(map_config):
    from fr2dynage.maps import compute_age_map
    path, config = map_config
    del config["kinematics"]["speed_km_s"]
    config["kinematics"]["speed_map"] = "speed.fits"
    header = _header()
    header["BUNIT"] = "km/s"
    speed = np.full((5, 5), 3000.0)
    speed[1, 1] = -10.0
    fits.PrimaryHDU(speed, header=header).writeto(path.parent / "speed.fits")
    _save(path, config)
    result = compute_age_map(path)
    assert not result.valid[1, 1]
    assert np.isnan(result.age_myr[1, 1]) and np.isnan(result.error_myr[1, 1])
    assert result.valid[2, 3]


def test_non_fr2_integrated_input_roundtrips_but_cannot_use_kda(tmp_path):
    config = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    default_observation, assumptions, _, _ = load_configuration(ROOT / "examples/cygnus_a.json")
    assert default_observation.radio_class == "fr2"
    assert isinstance(KDAForwardModel(default_observation, assumptions), KDAForwardModel)
    config["source"]["radio_class"] = "FR-I"
    target = tmp_path / "fri.json"
    target.write_text(json.dumps(config))
    observation, assumptions, bounds, runtime = load_configuration(target)
    assert observation.radio_class == "fr1"
    canonical = normalized_input(observation, assumptions, bounds, runtime)
    assert canonical["source"]["radio_class"] == "fr1"
    target.write_text(json.dumps(canonical))
    assert load_configuration(target)[0].radio_class == "fr1"
    with pytest.raises(ValueError, match="FR II"):
        KDAForwardModel(observation, assumptions)


def test_measurement_draws_preserve_class_and_cannot_bypass_fr2_guard(tmp_path):
    from fr2dynage.fitting import FitError, parametric_bootstrap
    config = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    config["source"]["radio_class"] = "fr1"
    target = tmp_path / "fri.json"
    target.write_text(json.dumps(config))
    observation, assumptions, bounds, _ = load_configuration(target)
    seen = []

    def factory(varied):
        seen.append(varied.radio_class)
        return KDAForwardModel(varied, assumptions)

    def unexpected_fit(model, bounds):
        pytest.fail("an FR I draw must not reach the FR II fitting route")

    with pytest.raises(FitError, match="every bootstrap realization failed"):
        parametric_bootstrap(observation, factory, unexpected_fit, bounds, samples=1, seed=7)
    assert seen == ["fr1"]


def test_map_cli_validation_and_class_listing(map_config, capsys):
    from fr2dynage.cli import main
    assert main(["validate-map", str(map_config[0])]) == 0
    output = capsys.readouterr()
    assert "Analytic map" in output.out
    assert main(["classes"]) == 0
    output = capsys.readouterr()
    assert "FR I" in output.out and "FR II" in output.out


def test_cli_writes_fits_and_readable_png(map_config, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from fr2dynage.cli import main
    output = tmp_path / "cli-age.fits"
    preview = tmp_path / "cli-age.png"
    assert main(["map-age", str(map_config[0]), "--output", str(output),
                 "--preview", str(preview)]) == 0
    assert output.exists()
    assert preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert preview.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert "<svg" in preview.with_suffix(".svg").read_text()
    assert output.stat().st_size > 0 and preview.stat().st_size > 1000
