from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from fr2dynage.reference_priors import (
    ReferencePriorError,
    calibrate_reference_priors,
    load_reference_priors,
)


def _constraint(value=1000.0, error=100.0, parameter="flow_speed_km_s", independent=True):
    units = {"flow_speed_km_s": "km/s", "magnetic_field_ug": "uG", "alpha_inj": "1"}
    quantities = {"flow_speed_km_s": "projected_plasma_transport_speed",
                  "magnetic_field_ug": "emitting_plasma_magnetic_field",
                  "alpha_inj": "injection_spectral_index"}
    return {"value": value, "error": error, "unit": units[parameter],
            "quantity": quantities[parameter], "method": "independent forward-model constraint",
            "independent": independent}


def _catalogue():
    return {"schema_version": "fr2dynage-reference-1", "sources": [
        {"name": f"Reference {i}", "radio_class": "wat", "reference": f"https://example.org/source-{i}",
         "origin": "observed", "constraints": {
             "flow_speed_km_s": _constraint(value),
             "magnetic_field_ug": _constraint(2.0 + i * 0.5, 0.2, "magnetic_field_ug"),
             "alpha_inj": _constraint(0.55 + i * 0.05, 0.02, "alpha_inj")}}
        for i, value in enumerate((800.0, 1000.0, 1250.0))]}


def test_predictive_prior_centres_on_source_population_and_includes_scatter():
    result = calibrate_reference_priors(_catalogue(), "Target", "WAT")
    speed = result["priors"]["flow_speed_km_s"]
    assert speed["distribution"] == "lognormal"
    assert 950 < math.exp(speed["log_mean"]) < 1050
    assert speed["log_sigma"] >= 0.35
    alpha = result["priors"]["alpha_inj"]
    assert alpha["mean"] == pytest.approx(0.6)
    assert alpha["sigma"] > 0.05
    assert result["provenance"]["parameter_diagnostics"]["flow_speed_km_s"]["n_sources"] == 3
    assert not result["provenance"]["contains_synthetic_constraints"]
    json.dumps(result, allow_nan=False)


def test_uncertainty_in_population_mean_is_not_discarded():
    document = _catalogue()
    for source in document["sources"]:
        source["constraints"]["alpha_inj"].update(value=0.6, error=0.3)
    prior = calibrate_reference_priors(document, "Target", "wat")["priors"]["alpha_inj"]
    assert prior["mean"] == pytest.approx(0.6)
    assert prior["sigma"] == pytest.approx(math.sqrt(0.05**2 + (0.05**2 + 0.3**2) / 3))


def test_large_intrinsic_population_scatter_is_retained():
    document = _catalogue()
    for source, speed in zip(document["sources"], (100.0, 1000.0, 10000.0)):
        source["constraints"]["flow_speed_km_s"].update(value=speed, error=1.0)
    result = calibrate_reference_priors(document, "Target", "wat")
    prior = result["priors"]["flow_speed_km_s"]
    assert math.exp(prior["log_mean"]) == pytest.approx(1000.0, rel=1e-4)
    assert prior["log_sigma"] > 2


def test_target_and_aliases_are_excluded_before_calibration():
    document = _catalogue()
    target = deepcopy(document["sources"][0])
    target.update(name="Study target", aliases=["Target alias"])
    target["constraints"]["flow_speed_km_s"]["value"] = 200000
    document["sources"].append(target)
    result = calibrate_reference_priors(document, "  TARGET   ALIAS ", "wat")
    assert len(result["provenance"]["excluded"]) == 1
    assert result["provenance"]["excluded"][0]["name"] == "Study target"
    assert result["provenance"]["parameter_diagnostics"]["flow_speed_km_s"]["n_sources"] == 3


def test_target_exclusion_can_leave_too_few_sources():
    with pytest.raises(ReferencePriorError, match="at least three"):
        calibrate_reference_priors(_catalogue(), "reference 1", "wat")


def test_never_transfers_other_radio_class_to_wat():
    document = _catalogue()
    for source in document["sources"]:
        source["radio_class"] = "fr2"
    with pytest.raises(ReferencePriorError, match="same-class"):
        calibrate_reference_priors(document, "Target", "wat")


def test_synthetic_references_require_explicit_opt_in():
    document = _catalogue()
    for source in document["sources"]:
        source["origin"] = "synthetic"
    with pytest.raises(ReferencePriorError, match="origin"):
        calibrate_reference_priors(document, "Target", "wat")
    result = calibrate_reference_priors(document, "Target", "wat", allow_synthetic=True)
    assert result["provenance"]["contains_synthetic_constraints"]


def test_dependent_flow_estimates_are_excluded_by_default():
    document = _catalogue()
    for source in document["sources"]:
        source["constraints"]["flow_speed_km_s"].update(independent=False, method="spectral-age gradient")
    result = calibrate_reference_priors(document, "Target", "wat")
    assert "flow_speed_km_s" not in result["priors"]
    assert len(result["provenance"]["excluded"]) == 3
    result = calibrate_reference_priors(document, "Target", "wat", allow_dependent=True)
    assert "flow_speed_km_s" in result["priors"]
    assert result["provenance"]["contains_dependent_constraints"]


@pytest.mark.parametrize("quantity", ["galaxy_speed", "intrinsic_plasma_speed", "jet_pattern_speed"])
def test_wrong_velocity_quantity_is_rejected(quantity):
    document = _catalogue()
    document["sources"][0]["constraints"]["flow_speed_km_s"]["quantity"] = quantity
    with pytest.raises(ReferencePriorError, match="projected_plasma_transport_speed"):
        calibrate_reference_priors(document, "Target", "wat")


@pytest.mark.parametrize("field,value", [
    ("value", True), ("value", 0), ("value", float("nan")), ("value", 299792.458),
    ("error", 0), ("error", -1), ("error", "10"),
    ("unit", "m/s"), ("method", ""), ("independent", 1), ("typo", 3),
])
def test_invalid_constraint_fields_fail_closed(field, value):
    document = _catalogue()
    document["sources"][0]["constraints"]["flow_speed_km_s"][field] = value
    with pytest.raises(ReferencePriorError):
        calibrate_reference_priors(document, "Target", "wat")


def test_duplicate_source_or_alias_cannot_boost_reference_count():
    document = _catalogue()
    document["sources"][1]["aliases"] = ["reference 0"]
    with pytest.raises(ReferencePriorError, match="duplicate source identity"):
        calibrate_reference_priors(document, "Target", "wat")


def test_missing_reference_citation_fails_even_for_excluded_class():
    document = _catalogue()
    document["sources"][0]["radio_class"] = "fr2"
    del document["sources"][0]["reference"]
    with pytest.raises(ReferencePriorError, match="reference"):
        calibrate_reference_priors(document, "Target", "wat")


def test_unknown_root_or_missing_schema_is_rejected():
    document = _catalogue()
    document["typo"] = 1
    with pytest.raises(ReferencePriorError, match="unknown catalogue"):
        calibrate_reference_priors(document, "Target", "wat")
    del document["typo"]
    del document["schema_version"]
    with pytest.raises(ReferencePriorError, match="schema_version"):
        calibrate_reference_priors(document, "Target", "wat")


def test_partial_parameter_coverage_is_visible_in_provenance():
    document = _catalogue()
    del document["sources"][0]["constraints"]["magnetic_field_ug"]
    result = calibrate_reference_priors(document, "Target", "wat")
    assert "magnetic_field_ug" not in result["priors"]
    diagnostic = result["provenance"]["parameter_diagnostics"]["magnetic_field_ug"]
    assert diagnostic["status"] == "insufficient_references"
    assert diagnostic["n_sources"] == 2


def test_file_loader_hashes_exact_input(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(_catalogue()))
    result = load_reference_priors(path, "Target", "wat")
    assert result["provenance"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["provenance"]["path"] == str(path.resolve())


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":2}', '{"x":NaN}', '{broken'])
def test_file_loader_rejects_duplicate_nonfinite_or_invalid_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ReferencePriorError, match="cannot load reference catalogue"):
        load_reference_priors(path, "Target", "wat")


def test_extreme_positive_log_uncertainties_remain_finite():
    document = _catalogue()
    for source in document["sources"]:
        source["constraints"]["magnetic_field_ug"].update(value=1e-300, error=1e300)
    result = calibrate_reference_priors(document, "Target", "wat")
    assert np.isfinite(result["priors"]["magnetic_field_ug"]["log_sigma"])


def test_unrepresentable_alpha_error_has_clear_error():
    document = _catalogue()
    document["sources"][0]["constraints"]["alpha_inj"]["error"] = 1e300
    with pytest.raises(ReferencePriorError, match="representable numerical domain"):
        calibrate_reference_priors(document, "Target", "wat")
