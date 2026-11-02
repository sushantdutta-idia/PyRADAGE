# Contributing

Contributions are welcome at
[sushantdutta-idia/PyRADAGE](https://github.com/sushantdutta-idia/PyRADAGE).
Use PyRADAGE for the product name and `pyradage` for new commands/imports.
Preserve legacy `fr2dynage` imports, entry points and schema identifiers so
existing analyses remain readable; classical DYNAGE model names do not change.

Install a source checkout with `python -m pip install -e '.[gui,test]'` and run
`python -m pytest -q`. The tests use labelled synthetic sources; they do not
require a graphical display or an external PyRATS installation. Keep
scientific recovery/units/covariance checks separate from UI-state tests.

Use the same `automatic.run_automatic` engine from the GUI and headless CLI.
Tk widgets must only be touched by the main thread. Numerical work reports
progress through the controller queue and accepts cooperative cancellation.
Missing observational metadata should produce a useful correction message,
not guessed values or silently changed units.

Preserve the distinction between JP–Tribble radiative elapsed time, CI
injection duration, plasma transport and global dynamical lifetime. Keep
quality flags, limits, model assumptions, input provenance and publication
exports together. The default source field is B=0.4 B_eq; an explicitly
supplied source field is an auditable alternative, not an implicit new
default for all targets. Detected core emission stays eligible for fitting.

Use the shared publication helpers: large fonts, no top titles, readable
units/colour bars, 600 dpi PNG and PDF/SVG companions where supported. Put
scientific captions and interpretive limitations in the report.

Do not commit observations, private source reports, absolute user paths,
generated results, environments, caches or verification archives. Use
`tools/export_github.py` for a clean allowlisted source tree. It does not
publish or push. External PyRATS/PySynch remain separately installed/licensed;
do not vendor their source without addressing the applicable licence.
