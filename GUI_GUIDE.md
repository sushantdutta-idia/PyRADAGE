# Guided image-to-map workflow

This guide covers the PyRADAGE v1.0 desktop workflow. Legacy FR2DYNAGE
commands, imports and scientific configuration/output schemas remain supported.

## Starting the desktop interface

Install with `python -m pip install -e '.[gui]'` in a Python 3.11+ environment,
then run `pyradage gui`, `pyradage-gui`, or `python gui.py`. The macOS and
Windows launchers use a checkout-local `.venv` when present; they do not
install packages or modify a Python installation.

The GUI Python needs Tk. Check with `python -m tkinter`. An error mentioning
`_tkinter`, `TclError` or a missing display means that Python/session cannot
open native windows. Use a Tk-enabled Python desktop installation, your OS
Tk package, or the headless automatic CLI. GUI worker tests need no display.

## One-time field backend setup

For the default **B = 0.4 B_eq**, install
[PyRATS](https://github.com/sushantdutta-idia/PyRATS) and
[PySynch](https://github.com/mhardcastle/pysynch) following their instructions.
PySynch requires GSL and a compiler. Configure the script and its Python
executable through Advanced or `PYRADAGE_PYRATS_SCRIPT` and
`PYRADAGE_PYRATS_PYTHON`. The GUI and PyRATS can use different Python
environments. PyRATS runs as an isolated subprocess and is not vendored into
this MIT-licensed repository.

Legacy `FR2DYNAGE_PYRATS_SCRIPT` and `FR2DYNAGE_PYRATS_PYTHON` settings
remain accepted. Corresponding `PYRADAGE_` settings take precedence.

The default field depends on volume, electron limits, composition, injection
spectrum and cosmology. A source-integrated field is not a resolved field
measurement. A missing backend produces an actionable error, not an assumed
field.

An optional source field in tesla can be supplied instead. For example,
`2.24e-10 T = 2.24 μG`. This is a run-specific fixed input, not a universal
default. No B_eq is fabricated from it and no field uncertainty is inferred.
Radiative maps can run without PyRATS in this mode; unavailable energetics
are labelled as unavailable.

## Normal use

1. Select **Add FITS images** and choose 4–64 images of the same cropped source.
   Multiple selection is supported; duplicates are omitted.
2. Enter the source **redshift** as a finite positive number.
3. Select **Run analysis**. A fresh result directory is created under
   `~/PyRADAGE-results`; Choose selects another parent directory.
4. Follow the log, then use **Open summary PDF** or **Open results**.

Images must have celestial WCS, restoring-beam information and units
convertible to Jy/beam. Frequencies are read from FITS metadata, not filenames.
If metadata are missing or wrong, enter space/comma-separated MHz values in
Advanced, one per image in the displayed order. No manual JSON is required.

Delete removes selected images; Ctrl+O adds more. Inputs cannot be changed
during a run. The numerical worker does not block the interface, and repeated
clicks cannot start overlapping analyses. Cancel requests cooperative
cancellation; the current numerical step may need to finish. Closing during
a run requests the same cancellation and waits for the worker to stop.
Inputs and existing result directories are preserved.

## Automatic preparation and model scope

The engine checks/prepares the image grid, beams, noise and footprint, records
assumptions, obtains the field and fits JP–Tribble/CI spectra. Review the saved
source mask: arbitrary images alone cannot reliably distinguish the intended
source from unrelated blended objects. Crop around one source, especially in
crowded fields. WCS/beam checks do not prove matched uv coverage, flux recovery
or calibration correctness.

No fixed circular core exclusion is applied. Detected central emission stays
eligible. A mixed nuclear/inner-jet beam may fail an ageing model or reach its
zero-age boundary; flags stay visible.

The combined PDF/PNG are single figure files with separately labelled clocks.
There are no top figure or panel titles. Large fonts, WCS labels, colour bars,
units and quality information remain. Full captions and assumptions belong
to the report, not a large banner above a publication figure.

JP–Tribble describes radiative time since injection under its declared
free-streaming/RMS-field model. CI describes continuous-injection duration.
A pixelwise mixture of these clocks is not one common dynamical-age map.
When curvature is unestablished, best fits can be diagnostic or limit cases
rather than age measurements. Open-profile, boundary and poor-fit flags are
retained.

The automatic image-only route does not infer AGN turn-on, an external
atmosphere, jet power, three-dimensional trajectories or independently
measured flow/expansion speeds. Global dynamical age needs the appropriate
additional constraints. See the [radio guide](RADIO_INFERENCE_GUIDE.md) and
[DYNAGE guide](DYNAGE_ANALYSIS_GUIDE.md).

## Running on a server

```bash
pyradage auto band1.fits band2.fits band3.fits band4.fits \
  --redshift 0.1469 --output-dir results/source
```

With an explicitly supplied field and observing frequencies:

```bash
pyradage auto band1.fits band2.fits band3.fits band4.fits \
  --redshift 0.1469 --output-dir results/source-fixed \
  --field-t 2.24e-10 --frequencies-mhz 150 350 1400 3000
```

`--pyrats-script` and `--pyrats-python` configure the backend per run. The
scientific engine is the same as in the wizard.

## GitHub distribution

`python tools/export_github.py --output ../PyRADAGE-GitHub` creates a clean
allowlisted source directory. It never uploads or pushes. Code, docs, tests,
synthetic examples and CI are included. Observations, verification output,
backups, old archives, environments and local task instructions are excluded.
Review `EXPORT_MANIFEST.json` and test the export before publishing.
