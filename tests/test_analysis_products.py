"""Product-level checks: physical labels, missing data, atomicity and guards."""
from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from fr2dynage.analysis_products import AnalysisProductError, write_analysis_products


@pytest.fixture(autouse=True)
def fast_product_rendering(monkeypatch):
    # These tests verify publication bundles and file safety. The dedicated
    # plotting tests check the production 600 dpi raster export separately.
    import fr2dynage.plotting as plotting
    monkeypatch.setattr(plotting, "RASTER_DPI", 72)


def _fit(age=10., power=1e38, method="classic_dynage"):
    spectrum = []
    for frequency, observed, predicted in [(151., 1e27, 1.02e27), (610., 3e26, 2.98e26), (1400., 8e25, 8.1e25)]:
        spectrum.append({"frequency_mhz_rest": frequency, "observed_power_w_hz_sr": observed,
                         "predicted_power_w_hz_sr": predicted, "fractional_error": .05,
                         "log10_residual_model_over_data": float(np.log10(predicted/observed))})
    return {"method": method, "success": True,
            "best_fit": {"age_myr": age, "alpha_inj": .6,
                         "jet_power_w_per_jet": power,
                         "rho0_kg_m3_conditional_on_core_radius": 1e-23},
            "objective": {"name": "Delta" if method == "classic_dynage" else "chi2", "value": .02},
            "spectrum": spectrum,
            "derived": {"newest_cohort_magnetic_field_t": 1e-9, "one_jet_q_times_age_j": 3e52},
            "warnings": ["Synthetic fixture; no observed source inference."]}


def _fr2():
    return {"schema_version": "fr2dynage-analysis-1", "kind": "fr2_dynage",
            "source": {"name": "Synthetic FR II", "radio_class": "fr2", "redshift": .15},
            "primary_method": "classic", "methods": {"classic": _fit(), "joint": _fit(11., 1.1e38, "joint_log_likelihood")},
            "field_constraint": {"equation": "B = 0.4 B_eq", "fraction": .4},
            "diagnostics": {
                "intersections": {"alpha_inj": .6, "frequencies_mhz_rest": [151., 610., 1400.], "tracks": [
                    {"age_myr": 5., "valid": True, "delta": .5, "log10_q_by_frequency": [38., 38.2, 38.4], "log10_rho0_by_frequency": [-23., -22.8, -22.6]},
                    {"age_myr": 7., "valid": False, "delta": None, "log10_q_by_frequency": [None]*3, "log10_rho0_by_frequency": [None]*3, "status": "root unavailable"},
                    {"age_myr": 10., "valid": True, "delta": 0., "log10_q_by_frequency": [38., 38., 38.], "log10_rho0_by_frequency": [-23.]*3}]},
                "alpha_profile": [{"alpha_inj": alpha, "age_myr": age, "log10_jet_power_w": q,
                                   "log10_q_times_t_j": energy, "delta": .02}
                                  for alpha, age, q, energy in [(.55, 9., 38.2, 53.2), (.6, 10., 38., 53.), (.65, 12., 38.1, 53.3)]],
                "evolution": {"frequencies_mhz_rest": [151., 1400.], "tracks": [
                    {"age_myr": 1., "lobe_length_kpc": 30., "cocoon_pressure_pa": 1e-12, "magnetic_field_t": 1e-9,
                     "radio_power_w_hz_sr_by_frequency": [1e27, 1e26]},
                    {"age_myr": 10., "lobe_length_kpc": 100., "cocoon_pressure_pa": 1e-13, "magnetic_field_t": 3e-10,
                     "radio_power_w_hz_sr_by_frequency": [5e26, 2e25]}]}},
            "sensitivity": [{"label": "base", "status": "ok", "result": _fit()},
                            {"label": "invalid domain", "status": "failed", "error": "No solution", "overrides": {"beta": 2.}}],
            "quality": {"status": "conditional", "objective_is_likelihood": False}, "warnings": ["Field prescription is adopted."]}


def _radio():
    return {"schema_version": "fr2dynage-analysis-1", "kind": "radio_transport",
            "source": {"name": "Synthetic WAT", "radio_class": "wat"},
            "primary_method": "steady_jp_advection", "radio_result": {
                "status": "curvature_unconstrained", "map_statistics": {
                    "area_weighted_mean": {"mean": 10., "median": 9., "credible_interval_68": [7., 12.]},
                    "area_weighted_median": {"mean": 8., "median": 7., "credible_interval_68": [5., 10.]}},
                "warnings": ["Shared uncertainty; not independent pixel ages."]},
            "diagnostics": {
                "regional_spectra": {"frequencies_mhz": [150., 610., 1400.], "regions": [
                    {"id": 1, "branch": 1, "pixel_xy": [5, 8], "observed_jy_per_beam": [.01, .003, None],
                     "rms_jy_per_beam": [.0001]*3, "model_jy_per_beam": [.011, .0029, .001]},
                    {"id": 2, "branch": 2, "pixel_xy": [8, 8], "observed_jy_per_beam": [.008, .002, .0007],
                     "rms_jy_per_beam": [.0001]*3, "model_jy_per_beam": [.0081, .0021, .0008]}]},
                "transport_profiles": {"branches": [
                    {"branch": 1, "distance_kpc": [10., 20.], "age_p16_myr": [4., 8.], "age_p50_myr": [5., 10.], "age_p84_myr": [6., 12.]},
                    {"branch": 2, "distance_kpc": [10., 30.], "age_p16_myr": [3., 9.], "age_p50_myr": [4., 12.], "age_p84_myr": [5., 15.]}]}},
            "unavailable_classical_parameters": {"jet_power": "No FR II dynamical model for this WAT", "rho0": "Not constrained by steady transport"},
            "quality": {"status": "curvature_unconstrained"}, "warnings": []}


def _radio_with_core():
    document = _radio()
    spectrum = {"frequencies_mhz": [150., 610., 1400., 3000.],
                "observed_jy_per_beam": [.03, .014, None, .006],
                "rms_jy_per_beam": [.0001, .0001, None, .0002],
                "model_jy_per_beam": [.031, .013, .008, .0061],
                "pixel_xy": [5, 6], "sky_icrs_deg": [53.04, -27.44],
                "power_law": {"status": "descriptive_gls_fit", "alpha_positive_convention": .62,
                              "alpha_error_conditional": .03, "chi2": 2.7, "degrees_of_freedom": 2}}
    document["radio_result"]["core"] = {"status": "unresolved_age", "detected_pixels": 8,
        "core_mask_pixels": 9, "central_spectrum": spectrum,
        "spectral_age_myr": None, "transport_age_myr": None,
        "warnings": ["Core geometry and blended populations do not identify an age."]}
    document["source_coverage"] = {"source_detected_pixels": 120, "age_valid_pixels": 100,
                                   "core_detected_pixels": 8}
    document["diagnostics"]["core_spectrum"] = spectrum
    return document


def test_complete_fr2_products_keep_units_estimators_and_failed_tracks(tmp_path):
    source = _fr2()
    original = deepcopy(source)
    output = tmp_path/"fr2"
    manifest = write_analysis_products(source, output)
    assert source == original
    for filename in manifest["files"]:
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest() == manifest["sha256"][filename]
    for filename in manifest["preview_figures"]:
        assert (output/filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    expected = {"spectrum", "fit_comparison", "intersections", "alpha_profile", "evolution"}
    assert set(manifest["figures"]) == {f"{stem}.{extension}" for stem in expected for extension in ("png", "pdf", "svg")}
    assert set(manifest["vector_figures"]) == {f"{stem}.{extension}" for stem in expected for extension in ("pdf", "svg")}
    for stem in expected:
        assert (output/f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert "<svg" in (output/f"{stem}.svg").read_text()
    document = json.loads((output/"analysis.json").read_text())
    assert document == source
    rows = list(csv.DictReader((output/"intersection_tracks.csv").open()))
    failed = [row for row in rows if row["status"] == "root unavailable"]
    assert len(failed) == 3
    assert all(row["log10_jet_power_w"] == "" for row in failed)
    assert all("nan" not in row.values() for row in rows)
    comparison = list(csv.DictReader((output/"fit_comparison.csv").open()))
    assert {row["objective_name"] for row in comparison} == {"Delta", "chi2"}
    report = (output/"REPORT.md").read_text()
    assert "not a likelihood or confidence criterion" in report
    assert "fixed" in report and "density normalization" in report
    assert "B = 0.4 B_eq" in report
    assert "[PDF](spectrum.pdf)" in report
    assert "[SVG](spectrum.svg)" in report
    assert "![spectrum](spectrum.png)" in report
    assert "![spectrum](spectrum.pdf)" not in report
    sensitivity = list(csv.DictReader((output/"sensitivity.csv").open()))
    assert sensitivity[1]["status"] == "failed"
    assert sensitivity[1]["age_myr"] == ""


def test_radio_products_preserve_existing_fits_and_do_not_invent_dynamics(tmp_path):
    output = tmp_path/"radio"
    output.mkdir()
    fits = output/"transport.fits"
    fits.write_bytes(b"external root-orchestrator FITS product")
    manifest = write_analysis_products(_radio(), output)
    assert fits.read_bytes() == b"external root-orchestrator FITS product"
    assert "transport_profiles.png" in manifest["figures"]
    assert "regional_spectra_01.png" in manifest["figures"]
    assert "fit_comparison.csv" not in manifest["files"]
    radio_rows = list(csv.DictReader((output/"regional_spectra.csv").open()))
    assert len(radio_rows) == 6
    assert radio_rows[2]["observed_jy_per_beam"] == ""
    report = (output/"REPORT.md").read_text()
    assert "global source dynamical age is not inferred" in report
    assert "No FR II dynamical model for this WAT" in report
    assert "rho0" in report
    assert "core_spectrum.png" not in manifest["files"]
    assert "Core inclusion and source coverage" not in report


def _fixed_field_radio():
    document = _radio()
    document['field_constraint'] = {'kind':'user_fixed', 'equation':'B = 2.24e-10 T (user supplied)',
        'value_t':2.24e-10, 'value_ug':2.24, 'fraction':None,
        'uncertainty':'conditioned exactly; no supplied uncertainty'}
    document['radio_result']['field_prescription'] = document['field_constraint']
    document['radio_result']['posterior_parameters'] = {
        'magnetic_field_ug': {'median':2.24, 'credible_interval_68':[2.24,2.24]},
        'alpha_inj': {'median':.65, 'credible_interval_68':[.6,.7]}}
    document['parameter_quality'] = {'parameters': {
        'magnetic_field_ug': {'status':'user_fixed_input'},
        'alpha_inj': {'status':'diagnostic_only'}}}
    return document


def test_fixed_field_report_uses_actual_units_without_inventing_equipartition():
    from fr2dynage.analysis_products import _report
    report = _report(_fixed_field_radio(),{}, {}, [], [])
    assert 'Field: 2.24e-10 T = 2.24 μG' in report
    assert 'no field uncertainty was provided or inferred' in report
    assert 'user_fixed_input' in report and 'diagnostic_only' in report
    assert 'global source dynamical age is not inferred' in report
    assert '5.6' not in report  # B/0.4 is not a measured/reported B_eq.
    assert 'B = 0.4 B_eq' not in report


def test_fixed_field_budget_and_geometry_are_reported_with_true_ratio(tmp_path):
    document = _fixed_field_radio()
    beq = 6.021911561067544
    document['adopted_source_budget'] = {'budget': {
        'B_T':2.24e-10, 'B_uG':2.24, 'B_eq_uG':beq, 'fraction_of_B_eq':2.24/beq,
        'electron_energy_J':1.4e51, 'total_energy_J':1.488953551658181e53,
        'magnetic_energy_J':4.531738162018802e51, 'pressure_Pa':2.1865527945769914e-13}}
    geometry = {'volume_m3':2.2698644724411706e65, 'largest_linear_size_kpc':472.14569268472,
        'filling_factor':1.,'assumption_note':'Depth equals projected width.'}
    document['radio_result']['source'] = {'redshift':.1469}
    document['radio_result']['pyrats_properties'] = {'properties': {'geometry':geometry}}
    original = deepcopy(document)
    target = tmp_path/'fixed-field-report'
    write_analysis_products(document,target)
    assert document == original
    report = (target/'REPORT.md').read_text()
    assert '| Actual B / nominal B eq (comparison) | 0.371975 |' in report
    assert 'electron energy J' in report and 'magnetic energy J' in report
    assert '472.146' in report
    assert 'retained source inputs and conditional PyRATS geometry' in report
    rows = list(csv.DictReader((target/'source_properties.csv').open()))
    retained = {row['quantity']:row for row in rows}
    assert float(retained['volume_m3']['value']) == geometry['volume_m3']
    assert retained['largest_linear_size_kpc']['scope'].startswith('retained conditional')
    budget = {row['quantity']:row['value'] for row in csv.DictReader((target/'adopted_source_budget.csv').open())}
    assert float(budget['fraction_of_B_eq']) == pytest.approx(2.24/beq)


def test_regional_spectra_use_four_readable_panels_per_page(tmp_path):
    document = _radio()
    regions = document["diagnostics"]["regional_spectra"]["regions"]
    template = deepcopy(regions[0])
    document["diagnostics"]["regional_spectra"]["regions"] = [
        dict(deepcopy(template), id=index) for index in range(1, 10)]
    output = tmp_path/"regional-pages"
    manifest = write_analysis_products(document, output)
    previews = [name for name in manifest["preview_figures"] if name.startswith("regional_spectra_")]
    assert previews == ["regional_spectra_01.png", "regional_spectra_02.png", "regional_spectra_03.png"]
    rows = list(csv.DictReader((output/"regional_spectra.csv").open()))
    assert {int(row["region_id"]) for row in rows} == set(range(1, 10))
    assert len(rows) == 27


def test_vector_export_collision_preserves_existing_files(tmp_path):
    output = tmp_path/"vector-collision"
    output.mkdir()
    protected = output/"transport_summary.pdf"
    protected.write_bytes(b"existing vector drawing")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_analysis_products(_radio(), output)
    assert protected.read_bytes() == b"existing vector drawing"
    assert list(output.iterdir()) == [protected]


def test_core_is_included_with_measurements_and_unresolved_age(tmp_path):
    document = _radio_with_core()
    original = deepcopy(document)
    output = tmp_path / "core"
    manifest = write_analysis_products(document, output)
    assert document == original
    assert "core_spectrum.png" in manifest["figures"]
    assert (output / "core_spectrum.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    rows = list(csv.DictReader((output / "core_spectrum.csv").open()))
    assert len(rows) == 4
    assert [float(row["frequency_mhz_observed"]) for row in rows] == [150., 610., 1400., 3000.]
    assert rows[2]["observed_jy_per_beam"] == ""
    assert rows[0]["observed_jy_per_beam"] == "0.03"
    assert all("no age inferred" in row["age_status"] for row in rows)
    assert all("age_myr" not in row for row in rows)
    # Adding an explicitly unresolved core must not fabricate a zero age or
    # change the supplied transport-statistic summaries.
    saved = json.loads((output / "analysis.json").read_text())
    assert saved["radio_result"]["core"]["transport_age_myr"] is None
    assert saved["radio_result"]["map_statistics"] == _radio()["radio_result"]["map_statistics"]
    report = (output / "REPORT.md").read_text()
    assert "core region is included" in report
    assert "transport age is unresolved" in report
    assert "Source spectral coverage | 120" in report
    assert "Included core region | 8" in report
    assert "not a deblended compact-core flux measurement" in report
    assert "does not establish absorption" in report
    assert "0.62" in report and "2.7 / 2" in report
    assert "contributes no invented zero or other age" in report
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest


def test_core_diagnostic_without_a_powerlaw_fit_retains_the_observations(tmp_path):
    document = _radio_with_core()
    document["diagnostics"]["core_spectrum"]["power_law"] = {"status": "numerical_failure", "reason": "Singular covariance"}
    document["diagnostics"]["core_spectrum"]["model_jy_per_beam"] = [None] * 4
    manifest = write_analysis_products(document, tmp_path / "no-core-fit")
    assert "core_spectrum.png" in manifest["figures"]
    rows = list(csv.DictReader((tmp_path / "no-core-fit/core_spectrum.csv").open()))
    assert rows[0]["observed_jy_per_beam"] == "0.03"
    assert all(row["model_jy_per_beam"] == "" for row in rows)
    assert "Conditional α error" not in (tmp_path / "no-core-fit/REPORT.md").read_text()


def test_bad_core_spectrum_cannot_publish_partial_outputs(tmp_path):
    document = _radio_with_core()
    document["diagnostics"]["core_spectrum"]["model_jy_per_beam"] = [.03]
    with pytest.raises(AnalysisProductError, match="core spectrum arrays must match"):
        write_analysis_products(document, tmp_path / "bad-core")
    assert not (tmp_path / "bad-core").exists()
    assert not list(tmp_path.glob(".bad-core.*"))


def test_collisions_fail_without_changing_any_existing_product(tmp_path):
    output = tmp_path/"radio"
    write_analysis_products(_radio(), output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    changed = _radio(); changed["source"]["name"] = "Changed source"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_analysis_products(changed, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original
    (output/"external.dat").write_bytes(b"preserve")
    write_analysis_products(changed, output, overwrite=True)
    assert (output/"external.dat").read_bytes() == b"preserve"
    assert json.loads((output/"analysis.json").read_text())["source"]["name"] == "Changed source"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_provenance_fails_before_output_creation(tmp_path, value):
    document = _radio(); document["bad_value"] = value
    output = tmp_path/"invalid"
    with pytest.raises(AnalysisProductError, match="finite JSON"):
        write_analysis_products(document, output)
    assert not output.exists()


def test_protected_inputs_and_directory_aliases_are_untouched(tmp_path):
    output = tmp_path/"input-folder"; output.mkdir()
    source = output/"raw.json"; source.write_bytes(b"original observations")
    alias = tmp_path/"alias"; alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(AnalysisProductError, match="protected"):
        write_analysis_products(_radio(), alias, overwrite=True, protected_paths=[source])
    assert source.read_bytes() == b"original observations"
    assert len(list(output.iterdir())) == 1


def test_bad_radio_intervals_cannot_publish_partial_products(tmp_path):
    document = _radio()
    document["diagnostics"]["transport_profiles"]["branches"][0]["age_p16_myr"][0] = 100.
    output = tmp_path/"bad-radio"
    with pytest.raises(AnalysisProductError, match="p16"):
        write_analysis_products(document, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".bad-radio.*"))


def test_bad_regional_array_lengths_are_rejected(tmp_path):
    document = _radio()
    document["diagnostics"]["regional_spectra"]["regions"][0]["model_jy_per_beam"] = [1.]
    with pytest.raises(AnalysisProductError, match="match frequencies"):
        write_analysis_products(document, tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_reversed_summary_bounds_are_not_silently_clipped(tmp_path):
    document = _radio()
    document["radio_result"]["map_statistics"]["area_weighted_mean"]["credible_interval_68"] = [12., 7.]
    with pytest.raises(AnalysisProductError, match="bracket"):
        write_analysis_products(document, tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_failed_primary_cannot_be_reported_as_success(tmp_path):
    document = _fr2()
    document["methods"]["classic"]["success"] = False
    with pytest.raises(AnalysisProductError, match="primary_method"):
        write_analysis_products(document, tmp_path/"bad")


@pytest.mark.parametrize("mutation,match", [
    (lambda document: document.update(kind="wat_dynage"), "kind"),
    (lambda document: document.update(primary_method="missing"), "primary_method"),
])
def test_unknown_analysis_shapes_fail_cleanly(tmp_path, mutation, match):
    document = _fr2(); mutation(document)
    with pytest.raises(AnalysisProductError, match=match):
        write_analysis_products(document, tmp_path/"bad")


def test_numpy_data_roundtrip_without_nonfinite_json(tmp_path):
    document = _radio()
    document["quality"]["array"] = np.array([1., 2.])
    document["quality"]["count"] = np.int64(2)
    write_analysis_products(document, tmp_path/"numpy")
    data = json.loads((tmp_path/"numpy/analysis.json").read_text())
    assert data["quality"]["array"] == [1., 2.]
    assert data["quality"]["count"] == 2
