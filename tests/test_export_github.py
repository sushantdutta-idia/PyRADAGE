"""Only reviewable source/synthetic fixtures enter a public repository export."""
from pathlib import Path
import importlib.util
import json
import zipfile

import pytest


def exporter():
    path=Path(__file__).resolve().parents[1]/"tools/export_github.py"
    spec=importlib.util.spec_from_file_location("pyradage_export",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path):
    root=tmp_path/"development";root.mkdir()
    for name in ["README.md","LICENSE","pyproject.toml","gui.py","pyradage.py",
                 "src/pyradage/__init__.py","src/pyradage/__main__.py","src/fr2dynage/__init__.py",
                 "src/fr2dynage/gui.py","tests/test_demo.py","tools/export_github.py",
                 ".github/workflows/tests.yml","examples/synthetic.fits"]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("portable fixture\n")
    for name in ["AGENTS.md","observations.fits","verification/private.json","backups/old.zip",
                 ".venv/private.py","src/fr2dynage/__pycache__/gui.pyc",
                 "src/pyradage/__pycache__/__init__.pyc","results/source.fits"]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("must remain local\n")
    return root


def test_export_excludes_observations_task_context_and_environment(checkout,tmp_path):
    module=exporter();target=tmp_path/"public"
    result=module.export_repository(checkout,target,make_zip=True)
    manifest=json.loads((target/"EXPORT_MANIFEST.json").read_text())
    assert result["published"] is False
    assert manifest["software_name"] == "PyRADAGE"
    assert manifest["schema_version"] == "fr2dynage-github-export-1"
    for name in ["pyradage.py","src/pyradage/__init__.py","src/pyradage/__main__.py",
                 "src/fr2dynage/__init__.py"]:
        assert name in manifest["files"]
    assert "examples/synthetic.fits" in manifest["files"]
    for name in ["AGENTS.md","observations.fits","verification/private.json","backups/old.zip",
                 ".venv/private.py","results/source.fits"]:
        assert not (target/name).exists()
    for relative,digest in manifest["sha256"].items():
        assert module.digest(target/relative)==digest
    with zipfile.ZipFile(result["archive"]) as archive:
        assert "public/EXPORT_MANIFEST.json" in archive.namelist()
        assert "public/examples/synthetic.fits" in archive.namelist()


def test_existing_or_nested_export_is_never_replaced(checkout,tmp_path):
    module=exporter();target=tmp_path/"public";target.mkdir()
    with pytest.raises(FileExistsError):module.export_repository(checkout,target)
    with pytest.raises(ValueError,match="outside"):
        module.export_repository(checkout,checkout/"public")
    with pytest.raises(ValueError,match="outside"):
        module.export_repository(checkout,tmp_path)


def test_symlink_source_is_not_followed(checkout,tmp_path):
    module=exporter();private=tmp_path/"private.py";private.write_text("secret\n")
    try:
        (checkout/"src/fr2dynage/leak.py").symlink_to(private)
    except (OSError,NotImplementedError):
        pytest.skip("This platform/session does not permit creating symlinks")
    with pytest.raises(ValueError,match="symlink"):
        module.export_repository(checkout,tmp_path/"public")


def test_private_home_path_in_data_or_docs_requires_review(checkout,tmp_path):
    (checkout/"README.md").write_text("An accidental local path: "+str(Path.home()/"source-data"))
    with pytest.raises(ValueError,match="private home"):
        exporter().export_repository(checkout,tmp_path/"public")


def test_real_checkout_export_paths_are_portable():
    module=exporter();root=Path(__file__).resolve().parents[1]
    files=module.allowed_paths(root)
    assert root/"gui.py" in files
    assert root/"pyradage.py" in files
    assert root/"src/pyradage/__init__.py" in files
    assert root/"src/fr2dynage/gui.py" in files
    assert all("verification" not in path.relative_to(root).parts for path in files)


def test_export_requires_canonical_and_legacy_packages(checkout,tmp_path):
    (checkout/"src/pyradage/__init__.py").unlink()
    with pytest.raises(ValueError,match="src/pyradage/__init__.py"):
        exporter().export_repository(checkout,tmp_path/"public")
