"""Canonical product names preserve existing scientific and import contracts."""
from importlib import import_module
from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pyradage
import fr2dynage


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_public_api_agree():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "pyradage"
    assert project["version"] == pyradage.__version__ == fr2dynage.__version__ == "1.0"
    assert pyradage.KDAForwardModel is fr2dynage.KDAForwardModel
    assert set(["pyradage", "pyradage-gui", "fr2dynage", "fr2dynage-gui"]) <= project["scripts"].keys()


def test_canonical_modules_share_legacy_objects_and_state():
    for name in ["cli", "gui", "automatic", "model", "maps", "resolved_spectra", "automatic_field"]:
        assert import_module("pyradage." + name) is import_module("fr2dynage." + name)


def test_checkout_launcher_does_not_shadow_import_package():
    code = "import pyradage, pyradage.automatic; assert hasattr(pyradage, '__path__'); print(pyradage.__version__)"
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "1.0"


def test_new_and_legacy_module_launchers_work_without_a_display(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for name in ["pyradage", "fr2dynage", "pyradage.gui"]:
        completed = subprocess.run([sys.executable, "-m", name, "--help"], cwd=tmp_path,
                                   env=env, capture_output=True, text=True, check=True)
        assert "pyradage" in completed.stdout


def test_canonical_field_environment_precedes_legacy_and_explicit_wins(tmp_path, monkeypatch):
    from pyradage.automatic_field import find_pyrats
    canonical = tmp_path / "canonical.py"; canonical.touch()
    legacy = tmp_path / "legacy.py"; legacy.touch()
    supplied = tmp_path / "supplied.py"; supplied.touch()
    monkeypatch.setenv("PYRADAGE_PYRATS_SCRIPT", str(canonical))
    monkeypatch.setenv("PYRADAGE_PYRATS_PYTHON", sys.executable)
    monkeypatch.setenv("FR2DYNAGE_PYRATS_SCRIPT", str(legacy))
    monkeypatch.setenv("FR2DYNAGE_PYRATS_PYTHON", "a-missing-legacy-python")
    assert find_pyrats()[0] == canonical
    assert find_pyrats(supplied, sys.executable)[0] == supplied
    monkeypatch.delenv("PYRADAGE_PYRATS_SCRIPT")
    assert find_pyrats()[0] == legacy


def test_default_results_directory_uses_product_name(monkeypatch, tmp_path):
    from pyradage.gui import default_output_dir
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_output_dir().parent == tmp_path / "PyRADAGE-results"
