#!/usr/bin/env python3
"""Create an allowlisted PyRADAGE source tree for a reviewed GitHub upload.

This tool never publishes or pushes. Local data, task instructions, logs,
environments and development-only validation history are deliberately outside
the allowlist. Synthetic example FITS files remain part of the source tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


TOP_LEVEL = {
    ".gitignore", "README.md", "GUI_GUIDE.md", "LICENSE", "CITATION.cff",
    "CHANGELOG.md", "MAP_GUIDE.md", "RADIO_INFERENCE_GUIDE.md",
    "DYNAGE_ANALYSIS_GUIDE.md", "PARAMETER_RELIABILITY.md", "pyproject.toml",
    "MANIFEST.in", "requirements-tested.txt", "dynage.py", "pyradage.py", "gui.py",
    "launch_gui.command", "launch_gui.bat", "CONTRIBUTING.md",
}
REQUIRED = {
    "README.md", "LICENSE", "pyproject.toml", "pyradage.py", "gui.py",
    "src/pyradage/__init__.py", "src/fr2dynage/__init__.py",
}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()


def allowed_paths(source):
    source=Path(source).resolve()
    files={source/name for name in TOP_LEVEL if (source/name).is_file()}
    for folder,extensions in [
        ("src/pyradage",{".py"}),
        ("src/fr2dynage",{".py"}),
        ("tests",{".py"}),
        ("examples",{".py",".json",".fits"}),
        ("tools",{".py"}),
        (".github",{".yml",".yaml",".md"}),
    ]:
        parent=source/folder
        if parent.is_dir():
            files.update(path for path in parent.rglob("*") if path.is_file() and
                path.suffix.lower() in extensions and "__pycache__" not in path.parts)
    relative={str(path.relative_to(source)) for path in files}
    missing=REQUIRED-relative
    if missing:
        raise ValueError("Source checkout is incomplete: "+", ".join(sorted(missing)))
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Export refuses a symlink/outside-source file: {path}")
        # A data/config/document accidentally containing a private absolute
        # location must be reviewed instead of published automatically.
        if path.suffix.lower() in {".json",".md",".toml",".yml",".yaml",".txt",".cff"}:
            text=path.read_text(encoding="utf8")
            if str(Path.home()) in text:
                raise ValueError(f"Review a private home-directory path before export: {path}")
    return sorted(files)


def export_repository(source, output, *, make_zip=False):
    source=Path(source).expanduser().resolve()
    target=Path(output).expanduser().resolve()
    if target==source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Choose an export folder outside, and not containing, the development checkout.")
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a new export folder.")
    archive=target.with_name(target.name+".zip")
    if make_zip and archive.exists():
        raise FileExistsError(f"{archive} exists; preserve the previous archive or choose a new name.")
    files=allowed_paths(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+target.name+".",dir=target.parent))
    try:
        hashes={}
        for path in files:
            relative=path.relative_to(source)
            destination=stage/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            before=digest(path)
            shutil.copy2(path,destination)
            if digest(path)!=before or digest(destination)!=before:
                raise RuntimeError(f"Source changed while being exported: {relative}")
            hashes[str(relative)]=before
        # Keep the existing format identifier so previously written manifests
        # and downstream integrity readers remain compatible after the rename.
        manifest={"schema_version":"fr2dynage-github-export-1", "software_name":"PyRADAGE",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "files":sorted(hashes),"sha256":hashes,
            "scope":"Allowlisted source files only; manifest excludes itself. No data were uploaded or published."}
        (stage/"EXPORT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
        if target.exists():
            raise FileExistsError("The export destination appeared during preparation; it was left untouched.")
        stage.rename(target)
    finally:
        if stage.exists():shutil.rmtree(stage)
    result={"output":str(target),"source_files":len(files),"manifest":str(target/"EXPORT_MANIFEST.json"),
        "published":False}
    if make_zip:
        # Exclusive creation never replaces a pre-existing release archive.
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(target.rglob("*")):
                if path.is_file():bundle.write(path,str(Path(target.name)/path.relative_to(target)))
        result.update(archive=str(archive),archive_sha256=digest(archive))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--zip",action="store_true",dest="make_zip",help="also create a source zip next to the clean export")
    args=parser.parse_args(argv)
    try:
        result=export_repository(args.source,args.output,make_zip=args.make_zip)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Export failed: {exc}\n")
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
