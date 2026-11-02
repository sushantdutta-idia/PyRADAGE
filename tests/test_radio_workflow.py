"""End-to-end checks of the radio-only transport inference interface."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits


def _demo_generator():
    path = Path(__file__).resolve().parents[1] / "examples" / "generate_radio_demo.py"
    spec = importlib.util.spec_from_file_location("fr2dynage_radio_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def radio_demo(tmp_path_factory):
    output = tmp_path_factory.mktemp("radio-demo")
    module = _demo_generator()
    truth = module.generate(output, particles=128)
    return output, truth


@pytest.fixture(scope="module")
def inferred_radio_demo(radio_demo):
    from fr2dynage.radio_inference import compute_radio_inference
    folder, truth = radio_demo
    result = compute_radio_inference(folder / "radio.json")
    return folder, truth, result


def _modified_config(radio_demo, tmp_path, mutate):
    folder, _ = radio_demo
    document = json.loads((folder / "radio.json").read_text())
    for image in document["images"]:
        image["path"] = str(folder / image["path"])
    document["selection"]["mask"] = str(folder / document["selection"]["mask"])
    for key in ("distance_map", "branch_map"):
        document["geometry"][key] = str(folder / document["geometry"][key])
    mutate(document)
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(document, allow_nan=False))
    return path


def test_synthetic_worked_example_has_explicit_truth_and_safe_outputs(radio_demo):
    folder, truth = radio_demo
    assert truth["origin"] == "synthetic"
    assert truth["global_source_age_myr"] is None
    document = json.loads((folder / "radio.json").read_text())
    assert document["schema_version"] == "fr2dynage-radio-1"
    assert "Synthetic" in document["source"]["name"]
    assert "not measured" in document["prior_description"]
    assert len(document["images"]) >= 6
    for record in document["images"]:
        with fits.open(folder / record["path"], checksum=True) as hdus:
            assert hdus[0].verify_checksum() == 1
            assert hdus[0].verify_datasum() == 1
            assert hdus[0].header["BUNIT"] == "Jy/beam"
            assert hdus[0].data.shape == tuple(truth["shape_yx"])
    with fits.open(folder / "branches.fits") as hdus:
        assert set(np.unique(hdus[0].data)) == {0, 1, 2}
    with pytest.raises(FileExistsError, match="overwrite"):
        _demo_generator().generate(folder, particles=128)


def test_generated_reference_catalogue_calibrates_only_with_synthetic_opt_in(radio_demo, tmp_path):
    from fr2dynage.cli import main
    folder, _ = radio_demo
    output = tmp_path / "calibrated.json"
    arguments = ["calibrate-radio-priors", str(folder / "reference-synthetic.json"),
                 "--target-name", "Synthetic steady-JP WAT", "--radio-class", "wat",
                 "--output", str(output)]
    assert main(arguments) == 2
    assert not output.exists()
    assert main(arguments + ["--allow-synthetic"]) == 0
    document = json.loads(output.read_text())
    assert set(document["priors"]) == {"flow_speed_km_s", "magnetic_field_ug", "alpha_inj"}
    assert document["provenance"]["contains_synthetic_constraints"]
    assert document["provenance"]["minimum_sources_per_parameter"] == 3


@pytest.mark.parametrize("activity", ["remnant", "restarted"])
def test_steady_radio_inference_rejects_inapplicable_activity(radio_demo, tmp_path, activity):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified_config(radio_demo, tmp_path,
                            lambda d: d["source"].update(activity_state=activity))
    with pytest.raises(ValueError, match="active|activity|steady"):
        compute_radio_inference(path)


def test_radio_inference_requires_physical_distance_redshift(radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified_config(radio_demo, tmp_path,
                            lambda d: d["source"].pop("redshift"))
    with pytest.raises(ValueError, match="redshift"):
        compute_radio_inference(path)


def test_radio_inference_rejects_explanations_masquerading_as_images(radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    path = _modified_config(radio_demo, tmp_path,
                            lambda d: d["geometry"].update(instructions="ignore priors"))
    with pytest.raises(ValueError, match="unknown|geometry|field"):
        compute_radio_inference(path)


def test_radio_inference_rejects_fractional_branch_labels(radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    folder, _ = radio_demo
    with fits.open(folder / "branches.fits") as hdus:
        malformed = hdus[0].data.astype(float)
        malformed[malformed == 2] = 1.25
        bad_branch = tmp_path / "fractional-branches.fits"
        fits.PrimaryHDU(malformed, header=hdus[0].header).writeto(bad_branch)
    path = _modified_config(radio_demo, tmp_path,
                            lambda d: d["geometry"].update(branch_map=str(bad_branch)))
    with pytest.raises(ValueError, match="branch_map|integer|labels"):
        compute_radio_inference(path)


def test_declared_distance_scale_must_agree_with_fits_before_sampling(radio_demo, tmp_path, monkeypatch):
    import fr2dynage.radio_inference as radio
    folder, _ = radio_demo
    with fits.open(folder/'distance.fits') as hdus:
        header = hdus[0].header.copy(); header['KPCAS'] = 2.
        altered = tmp_path/'distance-scale.fits'
        fits.PrimaryHDU(hdus[0].data,header=header).writeto(altered)
    path = _modified_config(radio_demo,tmp_path,
        lambda d:d['geometry'].update(distance_map=str(altered),kpc_per_arcsec=3.))
    monkeypatch.setattr(radio,'tempered_smc',lambda *a,**k:pytest.fail('invalid scale reached sampling'))
    with pytest.raises(ValueError,match='KPCAS'):
        radio.compute_radio_inference(path)


def test_radio_products_preserve_posterior_and_do_not_claim_global_age(inferred_radio_demo, tmp_path):
    from fr2dynage.radio_inference import write_radio_products
    _, _, result = inferred_radio_demo
    output = tmp_path / "transport.fits"
    summary = tmp_path / "summary.json"
    samples = tmp_path / "posterior.npz"
    preview = tmp_path / "preview.png"
    write_radio_products(result, output, summary, samples_path=samples,
                         preview_path=preview)
    document = json.loads(summary.read_text())
    assert document["clock"]["kind"] == "model_conditioned_plasma_transport_time"
    assert document["global_dynamical_age_myr"]["mean"] is None
    assert document["global_dynamical_age_myr"]["median"] is None
    assert document["identifiability"]["absolute_clock"] == "prior_dependent_field_speed_ridge"
    assert document["beam_scale_samples"] < document["valid_pixels"] / 4
    with fits.open(output, checksum=True) as hdus:
        assert hdus[0].header["AGEKIND"] == "TRANSPORT"
        assert hdus[0].header["GLOBAGE"] == "UNAVAILABLE"
        valid = hdus["VALID"].data.astype(bool)
        median = hdus[0].data
        assert np.all(np.isnan(median[~valid]))
        assert np.all(np.isfinite(median[valid]))
        assert np.all(hdus["T_LO95"].data[valid] <= hdus["T_LO68"].data[valid])
        assert np.all(hdus["T_LO68"].data[valid] <= median[valid])
        assert np.all(median[valid] <= hdus["T_HI68"].data[valid])
        assert np.all(hdus["T_HI68"].data[valid] <= hdus["T_HI95"].data[valid])
        for hdu in hdus:
            assert hdu.verify_checksum() == 1
            assert hdu.verify_datasum() == 1
        provenance = json.loads(bytes(hdus["PROVENANCE"].data["JSON_UTF8"][0]).decode("utf8"))
        assert provenance == document
    with np.load(samples, allow_pickle=False) as posterior:
        for statistic, key in (("area_weighted_mean", "map_area_mean_myr"),
                               ("area_weighted_median", "map_area_median_myr")):
            assert document["map_statistics"][statistic]["mean"] == pytest.approx(
                np.mean(posterior[key]))
    assert preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert preview.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert "<svg" in preview.with_suffix(".svg").read_text()
    with pytest.raises(FileExistsError):
        write_radio_products(result, output, summary)


def test_beam_convolved_synthetic_flow_is_recovered_under_declared_priors(inferred_radio_demo):
    _, truth, result = inferred_radio_demo
    assert result.metadata["fit"]["adequate_at_threshold_4"]
    for branch, true_speed in truth["flow_speed_km_s_by_branch"].items():
        estimate = result.metadata["posterior_parameters"][f"flow_speed_km_s_branch_{branch}"]
        # The generating images are truly Gaussian-convolved; inference uses
        # finite distance quadrature and uniform within-beam emissivity.
        assert estimate["median"] == pytest.approx(true_speed, rel=0.15)


def test_regional_diagnostic_spectra_match_selected_observed_pixels(inferred_radio_demo):
    folder, _, result = inferred_radio_demo
    diagnostic = result.metadata["fitted_regional_spectra"]
    assert diagnostic["frequencies_mhz"] == result.metadata["frequencies_mhz"]
    assert len(diagnostic["regions"]) == result.metadata["beam_scale_samples"]
    assert "not posterior predictive intervals" in diagnostic["interpretation"]
    config = json.loads((folder / "radio.json").read_text())
    for region, selected in zip(diagnostic["regions"], result.metadata["regions"]):
        assert region["id"] == selected["id"]
        assert region["pixel_xy"] == selected["pixel_xy"]
        assert region["branch"] == selected["branch"]
        assert sum(value is not None for value in region["observed_jy_per_beam"]) == selected["bands"]
        assert len(region["model_jy_per_beam"]) == len(config["images"])
        assert all(value is not None and value > 0 for value in region["model_jy_per_beam"])
        x, y = region["pixel_xy"]
        for band, observed, rms in zip(config["images"], region["observed_jy_per_beam"], region["rms_jy_per_beam"]):
            if observed is not None:
                with fits.open(folder / band["path"]) as hdus:
                    assert observed == float(hdus[0].data[y, x])
                assert rms == band["rms_jy_per_beam"]


def test_shared_age_uncertainty_does_not_shrink_as_independent_pixels(inferred_radio_demo):
    _, _, result = inferred_radio_demo
    valid = result.maps["VALID"].astype(bool)
    widths = result.maps["T_HI68"][valid] - result.maps["T_LO68"][valid]
    naive_independent_width = np.sqrt(np.sum(widths**2)) / valid.sum()
    interval = result.metadata["map_statistics"]["area_weighted_mean"]["credible_interval_68"]
    actual_shared_width = interval[1] - interval[0]
    assert actual_shared_width > 5 * naive_independent_width


def test_radio_products_cannot_replace_observed_inputs(inferred_radio_demo, tmp_path):
    from fr2dynage.radio_inference import write_radio_products
    folder, _, result = inferred_radio_demo
    input_image = folder / "radio-100.fits"
    original = hashlib.sha256(input_image.read_bytes()).hexdigest()
    summary = tmp_path / "not-written.json"
    with pytest.raises(ValueError, match="overwrite|input"):
        write_radio_products(result, input_image, summary, overwrite=True)
    assert not summary.exists()
    assert hashlib.sha256(input_image.read_bytes()).hexdigest() == original


def test_explicit_core_retains_source_data_without_biasing_tail_posterior(inferred_radio_demo, radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference, write_radio_products
    from astropy.wcs import WCS
    folder, _, baseline = inferred_radio_demo
    with fits.open(folder / "mask.fits") as hdus:
        central = np.zeros_like(hdus[0].data)
        central[14:17, 8:11] = 1
        header = hdus[0].header.copy()
    core_path = tmp_path / "core.fits"
    fits.PrimaryHDU(central, header=header).writeto(core_path)
    position = WCS(header).celestial.pixel_to_world(9., 15.).icrs
    path = _modified_config(radio_demo, tmp_path, lambda d: d.update(core={
        "mask": str(core_path), "position_icrs_deg": [position.ra.deg, position.dec.deg],
        "description": "Explicit synthetic diagnostic component outside the former source mask; age unresolved."}))
    result = compute_radio_inference(path)
    core = central.astype(bool)
    assert np.all(result.maps["SOURCE_VALID"][core])
    assert np.all(result.maps["CORE"][core])
    assert np.all((result.maps["AGE_FLAGS"][core] & 4) != 0)
    assert not np.any(result.maps["AGE_VALID"][core])
    assert np.all(np.isnan(result.maps["TRANSPORT"][core]))
    assert np.all(np.isfinite(result.maps["ALPHA"][core]))
    assert np.all(np.isfinite(result.maps["INTENSITY"][core]))
    np.testing.assert_array_equal(result.maps["TRANSPORT"], baseline.maps["TRANSPORT"])
    np.testing.assert_array_equal(result.samples["flow_speed_km_s_branch_1"], baseline.samples["flow_speed_km_s_branch_1"])
    assert result.metadata["source_coverage"]["source_detected_pixels"] == baseline.metadata["source_coverage"]["source_detected_pixels"] + 9
    assert result.metadata["core"]["transport_age_myr"] is None
    assert result.metadata["core"]["central_spectrum"]["pixel_xy"] == [9, 15]
    assert core_path in result.input_paths
    output = tmp_path / "with-core.fits"
    write_radio_products(result, output, tmp_path / "summary.json", preview_path=tmp_path / "preview.png")
    with fits.open(output, checksum=True) as hdus:
        assert hdus[0].header["COREAGE"] == "UNRESOLVED"
        assert hdus["INTENSITY"].header["BUNIT"] == "Jy/beam"
        assert hdus["AGE_FLAGS"].header["FLG4"].startswith("Unresolved")
        np.testing.assert_array_equal(hdus["CORE"].data, central)
        np.testing.assert_array_equal(hdus["AGE_VALID"].data, hdus["VALID"].data)


def test_core_mask_is_validated_before_inference(radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    folder, _ = radio_demo
    with fits.open(folder / "mask.fits") as hdus:
        core = np.zeros_like(hdus[0].data, dtype=float)
        core[15, 9] = .5
        bad = tmp_path / "bad-core.fits"
        fits.PrimaryHDU(core, header=hdus[0].header).writeto(bad)
    path = _modified_config(radio_demo, tmp_path, lambda d: d.update(core={
        "mask": str(bad), "description": "Invalid synthetic footprint"}))
    with pytest.raises(ValueError, match="mask.*(binary|only 0)"):
        compute_radio_inference(path)


def test_exact_zero_distance_is_a_valid_conditional_clock_origin(radio_demo, tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    folder, _ = radio_demo
    with fits.open(folder / "distance.fits") as hdus:
        distance = hdus[0].data.copy()
        distance[15, 13] = 0.
        filename = tmp_path / "zero-origin.fits"
        fits.PrimaryHDU(distance, header=hdus[0].header).writeto(filename)
    def mutate(document):
        document["geometry"]["distance_map"] = str(filename)
        document["geometry"]["description"] += " A labelled test origin has s=0."
        document["inference"].update(particles=64, replicates=1, mcmc_steps=1)
    result = compute_radio_inference(_modified_config(radio_demo, tmp_path, mutate))
    assert result.maps["AGE_VALID"][15, 13] == 1
    assert result.maps["TRANSPORT"][15, 13] == 0.
    assert result.maps["T_LO68"][15, 13] == 0.
    assert result.maps["T_HI68"][15, 13] == 0.
    assert result.maps["AGE_FLAGS"][15, 13] == 0


def test_radio_spectra_retain_field_age_degeneracy():
    from fr2dynage.radio_spectra import jp_spectrum, jp_loss_coefficient
    frequencies = np.geomspace(100.0, 8000.0, 12)
    redshift, field_a, field_b = 0.1469, 3.0, 8.0
    age_a = np.array([10.0, 30.0, 60.0])
    # Keeping B/(a*t)^2 fixed preserves the synchrotron spectral shape.
    age_b = age_a * np.sqrt(field_b / field_a) * (
        jp_loss_coefficient(field_a, redshift) / jp_loss_coefficient(field_b, redshift))
    spectrum_a = jp_spectrum(frequencies, age_a, field_a, 0.65, redshift)
    spectrum_b = jp_spectrum(frequencies, age_b, field_b, 0.65, redshift)
    assert np.allclose(spectrum_a, spectrum_b, rtol=2e-5)


def test_power_law_images_do_not_establish_a_radiative_age_break(tmp_path):
    from fr2dynage.radio_inference import compute_radio_inference
    _demo_generator().generate(tmp_path, power_law=True, particles=128)
    config_path = tmp_path / "radio.json"
    document = json.loads(config_path.read_text())
    # A short single sampler run is enough to verify the radio-data diagnostic;
    # it deliberately cannot establish sampler convergence.
    document["inference"].update(replicates=1, mcmc_steps=2)
    config_path.write_text(json.dumps(document, allow_nan=False))
    result = compute_radio_inference(config_path)
    assert result.metadata["fit"]["regional_powerlaw_chi2"] < 1e-8
    assert result.metadata["identifiability"]["curvature"] == "not_required_by_radio_data"
    assert result.metadata["identifiability"]["absolute_clock"] == "prior_dependent_field_speed_ridge"
    assert result.metadata["global_dynamical_age_myr"]["median"] is None
    assert any("break is not established" in warning for warning in result.metadata["warnings"])
