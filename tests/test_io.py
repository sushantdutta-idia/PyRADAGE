from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from fr2dynage.io import ConfigurationError, load_configuration, normalized_input, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_example_loads() -> None:
    observation, assumptions, bounds, runtime = load_configuration(ROOT / "examples/cygnus_a.json")
    assert observation.name == "Cygnus A"
    assert len(observation.spectrum) == 4
    assert assumptions.c1_prescription == "dynage2007"
    assert bounds.age_myr == (1.0, 30.0)
    assert len(runtime["input_sha256"]) == 64


def test_observer_frame_is_rejected(tmp_path: Path) -> None:
    text = (ROOT / "examples/cygnus_a.json").read_text()
    path = tmp_path / "bad.json"
    path.write_text(text.replace('"frequency_frame": "rest"', '"frequency_frame": "observer"'))
    with pytest.raises(ConfigurationError, match="rest-frame"):
        load_configuration(path)


def test_isotropic_power_is_converted_to_per_steradian(tmp_path: Path) -> None:
    data = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    data["spectrum"]["power_convention"] = "isotropic"
    for point in data["spectrum"]["points"]:
        point["log10_power_w_hz"] += math.log10(4.0 * math.pi)
    path = tmp_path / "isotropic.json"
    path.write_text(json.dumps(data))
    converted, _, _, runtime = load_configuration(path)
    reference, _, _, _ = load_configuration(ROOT / "examples/cygnus_a.json")
    assert converted.spectrum[0].power_w_hz_sr == pytest.approx(
        reference.spectrum[0].power_w_hz_sr, rel=1e-14
    )
    assert runtime["input_power_convention"] == "isotropic"


def test_normalized_input_is_reloadable(tmp_path: Path) -> None:
    observation, assumptions, bounds, original_runtime = load_configuration(
        ROOT / "examples/cygnus_a.json"
    )
    path = tmp_path / "normalized.json"
    path.write_text(
        json.dumps(
            normalized_input(observation, assumptions, bounds, original_runtime),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    reloaded_observation, reloaded_assumptions, reloaded_bounds, runtime = load_configuration(path)
    assert reloaded_observation == observation
    assert reloaded_assumptions == assumptions
    assert reloaded_bounds == bounds
    assert runtime["input_power_convention"] == "per_sr"
    assert runtime["core_subtracted"] is True
    assert runtime["hotspots_subtracted"] is True
    assert runtime["bootstrap_samples"] == original_runtime["bootstrap_samples"]
    assert runtime["seed"] == original_runtime["seed"]


def test_normalized_input_uses_effective_cli_runtime_overrides(tmp_path: Path) -> None:
    observation, assumptions, bounds, runtime = load_configuration(ROOT / "examples/cygnus_a.json")
    runtime = {
        **runtime,
        "bootstrap_samples_effective": 23,
        "seed_effective": 104729,
    }
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(normalized_input(observation, assumptions, bounds, runtime)),
        encoding="utf-8",
    )
    _, _, _, reloaded_runtime = load_configuration(path)
    assert reloaded_runtime["bootstrap_samples"] == 23
    assert reloaded_runtime["seed"] == 104729


def test_remnant_activity_round_trip(tmp_path: Path) -> None:
    data = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    data["activity"] = {
        "state": "remnant",
        "post_shutdown_dynamics": "kdaext2017-kc02-model-b",
    }
    data["fit"]["active_fraction"] = [0.25, 0.9]
    source = tmp_path / "remnant.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    observation, assumptions, bounds, runtime = load_configuration(source)
    assert runtime["activity_state"] == "remnant"
    assert runtime["active_fraction_bounds"] == (0.25, 0.9)
    normalized = normalized_input(observation, assumptions, bounds, runtime)
    assert normalized["activity"]["state"] == "remnant"
    assert normalized["fit"]["active_fraction"] == [0.25, 0.9]

    round_trip = tmp_path / "remnant-normalized.json"
    round_trip.write_text(json.dumps(normalized), encoding="utf-8")
    _, _, _, reloaded = load_configuration(round_trip)
    assert reloaded["activity_state"] == "remnant"
    assert reloaded["active_fraction_bounds"] == (0.25, 0.9)


def test_activity_schema_rejects_contradictory_settings(tmp_path: Path) -> None:
    template = json.loads((ROOT / "examples/cygnus_a.json").read_text())

    active_with_fraction = copy.deepcopy(template)
    active_with_fraction["fit"]["active_fraction"] = [0.2, 0.8]
    path = tmp_path / "active-with-remnant-bound.json"
    path.write_text(json.dumps(active_with_fraction), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="only valid"):
        load_configuration(path)

    invalid_fraction = copy.deepcopy(template)
    invalid_fraction["activity"] = {"state": "remnant"}
    invalid_fraction["fit"]["active_fraction"] = [0.9, 0.2]
    path = tmp_path / "bad-remnant-bound.json"
    path.write_text(json.dumps(invalid_fraction), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="0 < lower"):
        load_configuration(path)


def test_strict_schema_rejects_ambiguous_or_non_finite_values(tmp_path: Path) -> None:
    template = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    missing = object()
    cases = [
        (("source", "name"), None, "source.name"),
        (("spectrum", "frequency_frame"), missing, "frequency_frame"),
        (("fit", "age_grid_size"), 23.5, "JSON integer"),
        (("runtime", "bootstrap_samples"), 1.9, "JSON integer"),
        (("model", "magnetic_ratio"), math.nan, "must be finite"),
        (("geometry", "size_error_kpc"), math.nan, "must be finite"),
        (("geometry", "projected_size_kpc"), "135", "JSON number"),
    ]
    for index, (keys, value, message) in enumerate(cases):
        data = copy.deepcopy(template)
        target = data
        for key in keys[:-1]:
            target = target[key]
        if value is missing:
            del target[keys[-1]]
        else:
            target[keys[-1]] = value
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ConfigurationError, match=message):
            load_configuration(path)


@pytest.mark.parametrize("duplicate", [
    '"redshift": 0.1, "redshift": 0.2',
    '"name": "first", "name": "second", "redshift": 0.1',
])
def test_duplicate_json_fields_are_rejected(tmp_path: Path, duplicate: str) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"source": {' + duplicate + '}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate JSON field"):
        load_configuration(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numbers_are_rejected(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"source": {"redshift": ' + constant + '}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be finite"):
        load_configuration(path)


def test_invalid_utf8_is_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid-encoding.json"
    path.write_bytes(b'{"source": "\xff"}')
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_configuration(path)


def test_huge_integer_is_a_configuration_error(tmp_path: Path) -> None:
    data = json.loads((ROOT / "examples/cygnus_a.json").read_text())
    data["source"]["redshift"] = 10**400
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="floating-point range"):
        load_configuration(path)


def test_input_hash_uses_original_bytes(tmp_path: Path) -> None:
    raw = (ROOT / "examples/cygnus_a.json").read_bytes().replace(b"\n", b"\r\n")
    path = tmp_path / "windows-newlines.json"
    path.write_bytes(raw)
    _, _, _, runtime = load_configuration(path)
    assert runtime["input_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("failure_stage", ["fsync", "replace"])
def test_failed_atomic_write_preserves_existing_result(
    tmp_path: Path, monkeypatch, failure_stage: str,
) -> None:
    import fr2dynage.io as io

    path = tmp_path / "result.json"
    previous = b'{"existing": true}\n'
    path.write_bytes(previous)

    def failed_operation(source, target=None):
        if failure_stage == "replace":
            assert json.loads(Path(source).read_text()) == {"new": True}
        raise OSError("simulated output failure")

    monkeypatch.setattr(io.os, failure_stage, failed_operation)
    with pytest.raises(OSError, match="output failure"):
        write_json(path, {"new": True})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_failed_json_serialization_preserves_existing_result(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"existing": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        write_json(path, {"nonfinite": math.nan})
    assert json.loads(path.read_text()) == {"existing": True}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_creates_and_replaces_result(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    write_json(path, {"old": True})
    write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert path.read_bytes().endswith(b"\n")
    assert list(path.parent.iterdir()) == [path]
