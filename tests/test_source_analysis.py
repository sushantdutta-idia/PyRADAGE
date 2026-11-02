"""End-to-end source analysis plus publication and physical-scope guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from fr2dynage.source_analysis import analyze_source


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    folder = tmp_path_factory.mktemp("source-analysis")
    script = Path(__file__).resolve().parents[1]/"examples/generate_dynage_demo.py"
    spec = importlib.util.spec_from_file_location("dynage_example", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    truth = module.generate(folder)
    return folder, truth


@pytest.fixture(scope="module")
def analyzed(synthetic):
    folder, truth = synthetic
    output = folder/"analysis"
    document, products = analyze_source(folder/"input.json", output, method="joint", track_points=9, evolution_points=5)
    return folder, truth, document, products


def changed(synthetic, tmp_path, mutate):
    data = json.loads((synthetic[0]/"input.json").read_text())
    mutate(data)
    path = tmp_path/"input.json"; path.write_text(json.dumps(data))
    return path


def test_joint_analysis_recovers_known_parameters_and_consistent_evolution(analyzed):
    folder, truth, document, manifest = analyzed
    fit = document["methods"]["joint"]["best_fit"]
    assert fit["age_myr"] == pytest.approx(truth["age_myr"], rel=.03)
    assert fit["alpha_inj"] == pytest.approx(truth["alpha_inj"], abs=.01)
    assert fit["jet_power_w_per_jet"] == pytest.approx(truth["jet_power_w_per_jet"], rel=.03)
    assert document["field_constraint"]["fraction"] == .4
    assert document["source_age_summary"]["mean_myr"] is None
    rows = [row for row in document["diagnostics"]["evolution"]["tracks"] if row["valid"]]
    densities = [row["density_parameter_rho0_a0_beta_si"] for row in rows]
    np.testing.assert_allclose(densities, densities[0], rtol=1e-12)
    assert np.ptp([row["lobe_length_kpc"] for row in rows]) > 0
    assert all(row["source_total_length_kpc"] == 2*row["lobe_length_kpc"] for row in rows)
    assert json.loads((folder/"analysis/analysis.json").read_text()) == json.loads(json.dumps(document))
    assert (folder/"analysis/run_manifest.json").is_file()
    from fr2dynage.reliability import verify_analysis_manifest
    assert verify_analysis_manifest(folder/'analysis', check_inputs=True)['passed']
    assert len(manifest["figures"]) >= 4
    assert document["normalized_input"]["runtime"]["bootstrap_samples"] == 0


def test_existing_output_is_not_overwritten_without_explicit_request(analyzed):
    folder, _, _, _ = analyzed
    before = (folder/"analysis/analysis.json").read_bytes()
    with pytest.raises(FileExistsError):
        analyze_source(folder/"input.json", folder/"analysis", method="joint")
    assert (folder/"analysis/analysis.json").read_bytes() == before


def test_missing_external_field_anchor_is_rejected(synthetic, tmp_path):
    path = changed(synthetic, tmp_path, lambda d:d["model"].pop("equipartition_field_ug"))
    with pytest.raises(ValueError, match="equipartition_field"):
        analyze_source(path, tmp_path/"out")


@pytest.mark.parametrize("change", [lambda d:d["source"].update(radio_class="wat"),
                                     lambda d:d.update(activity={"state":"remnant"})])
def test_classical_analysis_rejects_wrong_morphology_or_history(synthetic, tmp_path, change):
    with pytest.raises(ValueError, match="active FR II"):
        analyze_source(changed(synthetic,tmp_path,change),tmp_path/"out")


def test_output_directory_cannot_contain_inputs(synthetic):
    with pytest.raises(ValueError, match="contain or replace"):
        analyze_source(synthetic[0]/"input.json", synthetic[0], overwrite=True)


def test_bad_sensitivity_fails_before_any_fit(synthetic,tmp_path,monkeypatch):
    import fr2dynage.source_analysis as module
    sentinel=[]
    monkeypatch.setattr(module,"fit_joint",lambda *args:sentinel.append(True))
    cases=tmp_path/"cases.json";cases.write_text(json.dumps({"cases":[{"label":"bad","model":{"execute":"unsafe"}}]}))
    with pytest.raises(ValueError,match="unsupported model"):
        analyze_source(synthetic[0]/"input.json",tmp_path/"out",method="joint",sensitivity_path=cases)
    assert not sentinel


def test_publish_race_preserves_new_directory(synthetic,tmp_path,monkeypatch):
    import fr2dynage.source_analysis as module
    import fr2dynage.analysis_products as products
    target=tmp_path/"out"
    doc={"schema_version":"fr2dynage-source-analysis-1","kind":"fr2_dynage"}
    monkeypatch.setattr(module,"_fr2_document",lambda *a,**k:(deepcopy(doc),{synthetic[0]/"input.json"},None))
    def writer(document,stage,**kwargs):
        (stage/"analysis.json").write_text(json.dumps(document))
        target.mkdir();(target/"user-data.txt").write_text("preserve")
        return {"files":["analysis.json"]}
    monkeypatch.setattr(products,"write_analysis_products",writer)
    with pytest.raises(FileExistsError,match="appeared"):
        analyze_source(synthetic[0]/"input.json",target)
    assert (target/"user-data.txt").read_text()=="preserve"
    assert not list(tmp_path.glob(".out.*"))


def test_failed_render_leaves_previous_analysis_untouched(synthetic,tmp_path,monkeypatch):
    import fr2dynage.source_analysis as module
    import fr2dynage.analysis_products as products
    target=tmp_path/"out";target.mkdir()
    previous={"schema_version":"fr2dynage-source-analysis-1","important":"old"}
    (target/"analysis.json").write_text(json.dumps(previous))
    monkeypatch.setattr(module,"_fr2_document",lambda *a,**k:(deepcopy(previous),set(),None))
    def fail(*args,**kwargs):raise OSError("render failure")
    monkeypatch.setattr(products,"write_analysis_products",fail)
    with pytest.raises(OSError,match="render failure"):
        analyze_source(synthetic[0]/"input.json",target,overwrite=True)
    assert json.loads((target/"analysis.json").read_text())==previous
    assert not list(tmp_path.glob(".out.*"))


def test_contamination_warnings_and_failed_sensitivity_provenance(synthetic, tmp_path):
    path = changed(synthetic, tmp_path, lambda d: d["spectrum"].update(
        core_subtracted=False, hotspots_subtracted=False))
    cases = {"cases": [{"label": "invalid atmosphere", "model": {"beta": 10.}, "geometry": {}}]}
    case_path = tmp_path / "sensitivity.json"
    case_path.write_text(json.dumps(cases))
    document, _ = analyze_source(path, tmp_path / "out", method="joint",
        sensitivity_path=case_path, track_points=9, evolution_points=5)
    warnings = document["methods"]["joint"]["warnings"]
    assert any("core subtraction" in value for value in warnings)
    assert any("hotspot subtraction" in value for value in warnings)
    assert document["sensitivity"][0]["status"] == "failed"
    assert document["sensitivity"][0]["model_overrides"] == {"beta": 10.}
    assert document["sensitivity_input"]["document"] == cases


@pytest.mark.parametrize("change", [lambda d: d.update(budget=[]),
    lambda d: d["budget"].update(B_uG="bad"),
    lambda d: d["budget"].update(B_eq_uG=True),
    lambda d: d["budget"].update(fraction_of_B_eq=1.),
    lambda d: d.update(input_sha256="wrong")])
def test_bad_budget_rejected_before_expensive_radio_fit(tmp_path, monkeypatch, change):
    import hashlib
    import fr2dynage.radio_inference as radio
    from fr2dynage.source_analysis import _radio_document
    report = tmp_path / "PyRATS.json"; report.write_text("{}")
    path = tmp_path / "radio.json"
    path.write_text(json.dumps({"pyrats_properties": {"path": report.name}}))
    budget = {"input_sha256": hashlib.sha256(report.read_bytes()).hexdigest(), "budget": {
        "fraction_of_B_eq": .4, "B_eq_uG": 3., "B_uG": 1.2,
        "total_energy_J": 1e53, "pressure_Pa": 1e-13}}
    change(budget)
    budget_path = tmp_path / "budget.json"; budget_path.write_text(json.dumps(budget))
    def expensive(*args):
        pytest.fail("invalid budget must be rejected before inference")
    monkeypatch.setattr(radio, "compute_radio_inference", expensive)
    with pytest.raises(ValueError, match="budget"):
        _radio_document(path, budget_path=budget_path)
