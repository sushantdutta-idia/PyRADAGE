# PyRADAGE

**Current release: v1.0**

Radio-source spectral ageing, physical properties and class-aware dynamics.
Select multi-frequency radio FITS images, enter the source redshift, and run
the guided interface. The automatic workflow prepares the images, fits
JP–Tribble and CI spectra, and produces a combined, publication-ready age
figure as **one PDF and one PNG**, with no top titles.

Source code: [sushantdutta-idia/PyRADAGE](https://github.com/sushantdutta-idia/PyRADAGE).
PyRADAGE is the new name of FR2DYNAGE. Existing `fr2dynage` commands,
Python imports and configuration/output schemas remain compatible. New
workflows use `pyradage`, `pyradage-gui` and `python -m pyradage`.

The automatic figure keeps **JP–Tribble radiative time** and **CI injection
duration** separately labelled. Radio images and redshift alone do not
uniquely determine a global dynamical lifetime. Unconstrained ages, open
limits, poor fits and boundary solutions remain explicitly flagged.

## Install and open the interface

Python **3.11 or newer** is required. From a source checkout:

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux with `source .venv/bin/activate`, or
on Windows with `.venv\Scripts\activate`. Then install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[gui]'
pyradage gui
```

`pyradage-gui` and `python gui.py` open the same interface. The supplied
`launch_gui.command` (macOS) and `launch_gui.bat` (Windows) use the checkout's
virtual environment when present. They do not install dependencies for you.

Tk must be available in the selected Python. Python.org desktop installers
and conda commonly include it; Linux distributions may require `python3-tk`.
Check with `python -m tkinter`. A headless server can use the automatic CLI.

## One-time magnetic-field backend setup

The default source prescription is **B = 0.4 B_eq**, using a documented
PyRATS/PySynch calculation. This requires an installed
[PyRATS](https://github.com/sushantdutta-idia/PyRATS) checkout and its
[PySynch](https://github.com/mhardcastle/pysynch) dependency. PySynch needs
GSL and a C compiler; follow their installation instructions. PyRATS's conda
environment file is a convenient starting point. BRATS is not needed for the
pipeline's native JP–Tribble/CI fitting kernels.

Point to that installation once, through the GUI's Advanced fields or these
environment variables:

```bash
export PYRADAGE_PYRATS_SCRIPT="/path/to/PyRATS/PyRATS.py"
export PYRADAGE_PYRATS_PYTHON="/path/to/the/pysynch-enabled/python"
```

On PowerShell, use `$env:PYRADAGE_PYRATS_SCRIPT = 'C:\path\to\PyRATS.py'`
and `$env:PYRADAGE_PYRATS_PYTHON = 'C:\path\to\python.exe'`. PyRATS is run
as an external, isolated process. Its GPL licence remains separate from this
repository's MIT licence; its source is not bundled here.

The legacy `FR2DYNAGE_PYRATS_SCRIPT` and `FR2DYNAGE_PYRATS_PYTHON`
variables remain accepted; the corresponding `PYRADAGE_` value takes
precedence when both names are set.

After setup, the normal GUI workflow needs only **images and redshift**. If a
source field is independently supplied, the optional fixed-field input can
run radiative maps without PyRATS; unavailable energetics are labelled rather
than invented. A fixed field is conditioned exactly as supplied, with no
field uncertainty inferred.

## Use the automatic workflow

1. Select **4–64 cropped FITS images** of the same source at distinct observing
   frequencies. Images need celestial WCS, units convertible to Jy/beam,
   restoring-beam metadata and observing frequencies.
2. Enter the redshift, such as `0.1469`.
3. Select **Run analysis**. Open the combined PDF or the result folder when
   the background worker finishes.

The default result folder is a fresh timestamped directory under
`~/PyRADAGE-results`. Inputs and earlier results are preserved. If frequency
metadata are missing or wrong, enter one observing frequency in MHz per image
in Advanced, matching the displayed file order. Frequencies are never guessed
from filenames. Cancel requests cooperative cancellation between numerical
steps. [GUI_GUIDE.md](GUI_GUIDE.md) explains the controls and troubleshooting.

Automatic preparation cannot recover missing uv coverage or flux, separate
blended populations, or prove that a selected footprint contains only the
intended object. Crop around one source and inspect the saved masks and
report. No fixed circular core exclusion is applied.

## Run without a graphical display

```bash
pyradage auto band150.fits band350.fits band1400.fits band3000.fits \
  --redshift 0.1469 --output-dir results/source
```

A supplied field is an explicit alternative to the default field backend:

```bash
pyradage auto band150.fits band350.fits band1400.fits band3000.fits \
  --redshift 0.1469 --output-dir results/fixed-field --field-t 2.24e-10
```

Optional `--frequencies-mhz 150 350 1400 3000` repairs frequency metadata in
image-argument order. `--pyrats-script` and `--pyrats-python` configure an
external backend per run. `pyradage auto --help` lists the complete options.

```python
from pyradage.automatic import run_automatic

result = run_automatic(
    ["band150.fits", "band350.fits", "band1400.fits", "band3000.fits"],
    redshift=0.1469,
    output_dir="results/source",
)
print(result["summary_pdf"])
```

## Outputs and interpretation

- A combined PDF/PNG figure with large fonts, WCS axes, units, colour bars,
  separate physical clocks, quality information and **no top titles**.
- FITS maps, parameter/assumption summaries and source masks for review.
- Preparation provenance, open-limit/boundary/poor-fit flags and an auditable
  report. Supported summaries stay distinct from numerical diagnostics.

The source field convention is RMS B for free-streaming JP–Tribble and
homogeneous B for CI. Absorption, adiabatic histories, reacceleration, variable
injection and unresolved mixing are not reconstructed automatically. Core
emission stays eligible; a zero-boundary core fit is not a measured zero age.

Global dynamical inference uses separate physical models and constraints.
Active FR II sources have the classical/joint DYNAGE route; curved
FR I/WAT/NAT plumes have conditional transport with supplied paths and priors.
Changing a morphology label does not make those models interchangeable. The
GUI does not manufacture a global lifetime from an unbroken radio power law.

## Advanced analyses and tests

Existing configuration-driven workflows remain available:

```bash
pyradage analyze input.json --output-dir results/analysis
pyradage verify-analysis results/analysis --check-inputs
```

See the [radio inference guide](RADIO_INFERENCE_GUIDE.md),
[DYNAGE analysis guide](DYNAGE_ANALYSIS_GUIDE.md), [map guide](MAP_GUIDE.md),
and [parameter reliability guide](PARAMETER_RELIABILITY.md).

```bash
python -m pip install -e '.[gui,test]'
python -m pytest -q
python -m build
```

GUI validation/worker tests run without a display. Numerical recovery checks
use labelled synthetic fixtures. External PyRATS is not silently installed
or downloaded during an analysis or during CI tests.

## Try the synthetic GUI example

The six FITS images in `examples/automatic_demo` are explicitly synthetic.
Select them in the GUI and enter redshift **0.15**. To use their known
generating field and run without a PyRATS installation, set the optional
Advanced fixed field to **3e-10 T**. The default equipartition prescription
is a different assumption and need not reproduce that generating field.

Regenerate the fixture in a new directory with:

```bash
python examples/generate_automatic_demo.py my-demo-images
```

## Prepare a clean GitHub upload

This repository includes cross-platform CI, citation metadata, synthetic
examples and a licence. Create an allowlisted source directory:

```bash
python tools/export_github.py --output ../PyRADAGE-GitHub
```

The export excludes observations, local task instructions, verification
archives, backups, caches and environments. Synthetic example FITS fixtures
needed by the tests are included. Review `EXPORT_MANIFEST.json` and run the
tests from the resulting directory before uploading it. The command does
not publish, push or send data anywhere.

## Citation and licence

Cite [CITATION.cff](CITATION.cff) and the model papers appropriate to the
analysis. The declared native kernels are not a claim of bitwise equivalence
to an external BRATS or SynchroFit executable. PyRADAGE uses the
[MIT licence](LICENSE); external tools retain their own licences. See
[CHANGELOG.md](CHANGELOG.md) for release history.
