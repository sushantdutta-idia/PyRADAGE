from __future__ import annotations

import json
from pathlib import Path

import pytest

import fr2dynage.cli as cli
from fr2dynage.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_resolved_request_is_not_silently_ignored_by_transport_only_cli(tmp_path,monkeypatch,capsys):
    import fr2dynage.radio_inference as radio
    def unexpected(*args,**kwargs):
        pytest.fail("resolved request must be routed before expensive transport inference")
    monkeypatch.setattr(radio,"compute_radio_inference",unexpected)
    path=tmp_path/"radio.json";path.write_text(json.dumps({"resolved_analysis":{}}))
    assert main(["infer-radio",str(path),"--output",str(tmp_path/"age.fits"),"--summary",str(tmp_path/"summary.json")])==2
    assert "use analyze" in capsys.readouterr().err
    assert not (tmp_path/"age.fits").exists()


@pytest.mark.parametrize("command", ["map-age", "infer-radio"])
@pytest.mark.parametrize("extension", ["pdf", "svg"])
def test_preview_companion_collision_stops_before_calculation(tmp_path, monkeypatch, capsys, command, extension):
    import fr2dynage.maps as maps
    import fr2dynage.radio_inference as radio
    def unexpected(*args, **kwargs):
        pytest.fail("a preview companion collision must stop before calculation")
    monkeypatch.setattr(maps, "compute_age_map", unexpected)
    monkeypatch.setattr(radio, "compute_radio_inference", unexpected)
    companion = tmp_path / f"preview.{extension}"
    companion.write_bytes(b"preserve existing drawing")
    output = tmp_path / "out.fits"
    arguments = [command, str(tmp_path/"unused.json"), "--output", str(output), "--preview", str(tmp_path/"preview.png")]
    if command == "infer-radio":
        arguments += ["--summary", str(tmp_path/"summary.json")]
    assert main(arguments) == 2
    assert "overwrite" in capsys.readouterr().err
    assert not output.exists()
    assert companion.read_bytes() == b"preserve existing drawing"


def test_validate_reports_remnant_state(capsys) -> None:
    code = main(["validate", str(ROOT / "examples/remnant_synthetic.json")])
    output = capsys.readouterr()
    assert code == 0
    assert "activity state = remnant" in output.out
    assert "residual dof = 4" in output.out


def test_remnant_predict_requires_phase_clock(capsys) -> None:
    code = main(
        [
            "predict",
            str(ROOT / "examples/remnant_synthetic.json"),
            "--age-myr",
            "30",
            "--alpha-inj",
            "0.6",
            "--jet-power-w",
            "5e38",
        ]
    )
    output = capsys.readouterr()
    assert code == 2
    assert "active-duration-myr" in output.err


def test_classic_method_is_rejected_for_remnant(capsys) -> None:
    code = main(
        [
            "fit",
            str(ROOT / "examples/remnant_synthetic.json"),
            "--method",
            "classic",
        ]
    )
    output = capsys.readouterr()
    assert code == 2
    assert "cannot fit a remnant" in output.err


@pytest.mark.parametrize("option", ["--bootstrap", "--seed"])
def test_invalid_runtime_override_is_rejected_before_fit(option, monkeypatch, capsys) -> None:
    def unexpected_fit(*args, **kwargs):
        pytest.fail("the expensive solver must not run for invalid settings")

    monkeypatch.setattr(cli, "fit_classic", unexpected_fit)
    code = main(["fit", str(ROOT / "examples/cygnus_a.json"), option, "-1"])
    output = capsys.readouterr()
    assert code == 2
    assert f"{option} cannot be negative" in output.err
    assert not output.out


def test_remnant_bootstrap_is_rejected_before_fit(monkeypatch, capsys) -> None:
    def unexpected_fit(*args, **kwargs):
        pytest.fail("the expensive solver must not run for unsupported bootstrap")

    monkeypatch.setattr(cli, "fit_remnant_joint", unexpected_fit)
    code = main([
        "fit", str(ROOT / "examples/remnant_synthetic.json"), "--bootstrap", "10",
    ])
    assert code == 2
    assert "not implemented" in capsys.readouterr().err


def test_bootstrap_requires_errors_before_fit(tmp_path, monkeypatch, capsys) -> None:
    data = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    del data["spectrum"]["points"][0]["fractional_error"]
    path = tmp_path / "missing-error.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def unexpected_fit(*args, **kwargs):
        pytest.fail("the expensive solver must not run without bootstrap errors")

    monkeypatch.setattr(cli, "fit_classic", unexpected_fit)
    assert main(["fit", str(path), "--bootstrap", "10"]) == 2
    assert "requires fractional_error" in capsys.readouterr().err


def test_restarted_bootstrap_is_rejected_before_fit(tmp_path, monkeypatch, capsys) -> None:
    data = json.loads((ROOT / "examples/remnant_synthetic.json").read_text())
    data.setdefault("runtime", {})["bootstrap_samples"] = 10
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def unexpected_fit(*args, **kwargs):
        pytest.fail("the expensive solver must not silently ignore bootstrap")

    monkeypatch.setattr(cli, "fit_restarted", unexpected_fit)
    assert main([
        "fit-restarted", str(path), str(ROOT / "examples/restarted_inner_synthetic.json"),
    ]) == 2
    assert "set runtime.bootstrap_samples to 0" in capsys.readouterr().err


def test_validate_rejects_underdetermined_remnant(tmp_path, capsys) -> None:
    data = json.loads((ROOT / "examples/remnant_synthetic.json").read_text())
    data["spectrum"]["points"] = data["spectrum"]["points"][:4]
    path = tmp_path / "too-few-points.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["validate", str(path)]) == 2
    output = capsys.readouterr()
    assert "at least five" in output.err
    assert not output.out


def test_prediction_numeric_overflow_is_reported_cleanly(capsys) -> None:
    assert main([
        "predict", str(ROOT / "examples/cygnus_a.json"),
        "--age-myr", "1e200", "--alpha-inj", "0.6", "--jet-power-w", "1e38",
    ]) == 2
    output = capsys.readouterr()
    assert output.err.startswith("error:")
    assert "Traceback" not in output.err


def test_output_os_error_is_reported_cleanly(tmp_path, monkeypatch, capsys) -> None:
    from fr2dynage.fitting import FitResult

    result = FitResult(
        source_name="test", method="classic_dynage", success=True,
        age_myr=10.0, alpha_inj=0.6, jet_power_w=1e38, rho0_kg_m3=1e-23,
        objective=0.0, objective_name="delta", spectrum=[],
        derived={"instantaneous_head_speed_over_c": 0.1},
    )
    monkeypatch.setattr(cli, "fit_classic", lambda *args, **kwargs: result)
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("existing", encoding="utf-8")
    assert main([
        "fit", str(ROOT / "examples/cygnus_a.json"),
        "--output", str(parent_file / "result.json"),
    ]) == 2
    output = capsys.readouterr()
    assert output.err.startswith("error:")
    assert "Traceback" not in output.err
    assert parent_file.read_text() == "existing"
