# DYNAGE-style source analysis

Version 0.5 adds `analyze` (alias `analyze-source`), which writes a complete
analysis directory with a report, plots, numerical tables and provenance.
It selects the physical analysis from the input schema. An integrated active
FR II component receives KA/KDA/DYNAGE-style analysis; a prepared radio-image
configuration receives the separately labelled steady-flow transport analysis.
Changing a morphology label does not make FR II dynamics applicable to a WAT.

This is an independent implementation of published equations, not a copy of
the original DYNAGE executable. The original three-stage procedure is described
by [Machalski et al. (2007)](https://arxiv.org/abs/astro-ph/0609680), using the
[KDA model](https://arxiv.org/abs/astro-ph/9710104). The observed-source field
constraint below is an explicit model variant.

## Install and test an active FR II example

From the source checkout, with the Python environment activated:

```bash
python -m pip install -e '.[analysis,maps,test]'
python pyradage.py analyze examples/dynage_demo/input.json \
  --output-dir results/dynage-demo \
  --sensitivity examples/dynage_demo/sensitivity.json
```

After installation, `pyradage` or `python -m pyradage` can replace
`python pyradage.py`. Open `results/dynage-demo/REPORT.md` and inspect the quality
flags as well as the point estimates. These inputs are explicitly synthetic:
the external field reference is a software-verification assumption, not a
measured source field. The joint fit can recover its known generating
parameters while the classical minimum-`Q*t` selection picks a different
solution. That disagreement is retained in the report.

To generate a fresh example without replacing the bundled files:

```bash
python examples/generate_dynage_demo.py --output results/new-dynage-inputs
python pyradage.py analyze results/new-dynage-inputs/input.json \
  --output-dir results/new-dynage-analysis --method both
```

`--method both` is the default. The weighted joint fit is the primary result
when available; the classical fit and its selection profiles remain visible.
`--method classic` or `--method joint` requests just one estimator. If one
requested fit fails, its reason is recorded and the surviving analysis is
marked partial. Failure of every requested estimator stops the run.

## Inputs and the fixed source-field prescription

Use the strict integrated-source JSON schema illustrated in
[`examples/dynage_demo/input.json`](examples/dynage_demo/input.json). Supply
the active FR II component's redshift, projected size, axial ratio,
inclination, rest-frame frequencies, K-corrected powers, and measurement
errors. A two-sided spectrum and size must describe the same double; a
one-lobe spectrum must use its own lobe geometry. The [README](README.md)
defines power units, geometry and particle conventions.

For observed-source analyses the adopted field is always

```text
B_current = 0.4 B_eq
```

There are two input routes:

1. Set `model.equipartition_field_ug` to a documented component-appropriate
   equipartition field in microgauss. Its provenance and uncertainty must be
   established outside this fit. The software cannot verify the physical
   origin of a manually entered number.
2. Supply `--pyrats-properties source_properties.json`, a supported PyRATS
   property report. The field anchor uses the report's Monte Carlo median
   `B_eq`; additional joint fits at its 16th and 84th percentiles describe
   conditional field sensitivity. They are not age confidence limits.

For example, after preparing the integrated FR II configuration and PyRATS
report:

```bash
python pyradage.py analyze source.json --output-dir results/source-analysis \
  --pyrats-properties source_properties.json --method both \
  --sensitivity source_sensitivity.json --bootstrap 100 --seed 314159
```

The redshift, electron Lorentz-factor limits and the explicitly assumed
non-radiating/electron energy-ratio mapping must agree with the PyRATS report.
Conflicting input and report fields are rejected. Numerical equality of
PyRATS `kappa` and model `k_prime` alone does not establish that the plasma
composition definitions are physically identical; that mapping remains an
explicit assumption. An integrated whole-source field should not be imported
as an independently measured field for each unresolved or unequal lobe.

The active snapshot anchor requires `gamma_b == gamma_c` and is mutually
exclusive with `model.magnetic_ratio`. At each trial solution it partitions
the total cocoon energy using `u_B = B_current**2/(2 mu_0)` and
`u_particles = p_c/(gamma_c-1) - u_B`. Trials without positive particle energy
are invalid. This implements the requested present-day field without simply
multiplying the historical magnetic/particle ratio by `0.4**2`.

The lower-level `fit` command still supports the historical unanchored
minimum-energy partition for published benchmarks. It does not automatically
impose this observed-source prescription. Use `analyze` for the guarded
source-analysis workflow. Its FR II branch rejects a missing field anchor.
The new snapshot anchor is not supported by the remnant model; the existing
separate remnant and restarted commands retain their documented scope.

A PyRATS field derived from the same radio measurements is a conditional
plug-in input. It is not an independent magnetic measurement, and this
adapter does not propagate a joint radio-flux/field covariance. Particle
content, electron cutoffs, filling factor, geometry and the adopted
equipartition convention remain material assumptions.

## What the classical analysis includes

For each injection index, the classical estimator solves one luminosity
equation per frequency over trial ages, constructs the corresponding
`(Q, rho0)` intersection families, minimizes their mean pairwise distance
`Delta(t)` in logarithmic coordinates, then selects the injection index by
minimum `Q*t`. Both this fit and the weighted joint spectrum fit are retained.

The report adds:

- Observed and predicted spectra, logarithmic residuals and descriptive
  power-law/quadratic curvature diagnostics.
- Frequency-specific jet-power/density intersection curves and a sampled
  age profile of `Delta`; invalid points retain their failure reasons.
- The classical injection-index scan with `Q*t`, age, power and density.
- A direct comparison of classical and joint parameter estimates.
- Conditional evolutionary tracks for size, pressure, magnetic field,
  energy, head speed and radio power, including a power-size diagram.
- Search-boundary, quadrature, multiple-root, fit-quality, weak-curvature
  and local-conditioning flags, plus explicit assumption-sensitivity fits.

`Delta` and minimum `Q*t` are physical selection heuristics, not likelihoods
or confidence thresholds. Under an externally supplied field, minimum `Q*t`
can favour a poor spectral fit; this is why the weighted joint solution is
primary when both succeed. A successful optimization does not establish a
unique age. Check residuals, search boundaries, profiles and model applicability.

The polynomial curvature diagnostic does not measure a physical break
frequency, injection index or spectral age. Its formal significance assumes
the supplied independent lognormal errors; shared flux-scale covariance is
not fitted by this integrated-spectrum diagnostic.

The environmental quantity directly constrained by this model is
`rho0 * a0**beta`. Quoted `rho0` depends on the fixed core radius `a0` and
atmospheric slope `beta`. Jet power always refers to **one jet**. Energy
tables distinguish total injected energy from stored cocoon energy.

## Evolution, sensitivities and uncertainty

Intersection families hold the measured size fixed while varying the inverse
solution's power and density. An evolutionary track instead holds the fitted
jet power and environmental normalization fixed while allowing size to
change. These are different constructions.

For an anchored fit, the evolutionary track resolves the magnetic/particle
energy partition at the fitted epoch and then freezes that partition. The
field evolves with cocoon pressure; the snapshot's field is not repeatedly
imposed as a constant throughout the source history. Tracks inside the
assumed atmosphere core radius or with superluminal head speed carry
domain flags and need physical scrutiny.

`--sensitivity` accepts a strict JSON object with at most 12 labelled cases:

```json
{
  "cases": [
    {"label": "Shallower atmosphere", "model": {"beta": 1.0}},
    {"label": "Higher electron cutoff", "model": {"gamma_min": 100.0}},
    {"label": "Different inclination", "geometry": {"inclination_deg": 60.0}}
  ]
}
```

Each case performs a joint fit under those alternative assumptions. Allowed
model overrides are `beta`, `core_radius_kpc`, `gamma_min`, `gamma_max`,
`k_prime`, `gamma_c`, `gamma_b`, `gamma_x` and `equipartition_field_ug`.
Allowed geometry overrides are `inclination_deg`, `projected_size_kpc` and
`axial_ratio`. Keep the field-consistency requirements satisfied. When a case
changes an assumption used in the equipartition calculation, its field must
also be recomputed consistently; holding it fixed is only a conditional
experiment. Alternative assumptions do not define a probability distribution
or automatically cover model uncertainty.

`--bootstrap N --seed S` draws measurement-error realizations and refits the
primary estimator. It is measurement-error Monte Carlo, not resampling from
the fitted model. Size and axial-ratio errors, when supplied, are treated as
independent. At least 10 successful fits and an 80% success fraction are
required before interval and sample mean/median summaries are released.
Failures and successful parameter draws remain recorded. The default is
`--bootstrap 0`; mean and median source-age distribution summaries are then
withheld, rather than created from one point estimate. The command-line value
controls this analysis run, and is recorded in the normalized input.

These FR II summaries refer to the distribution of one global model age.
They are different from spatial mean/median plasma times within a map. Local
covariance and measurement Monte Carlo omit fixed-model systematics. The
PyRATS percentile sensitivity fits do not jointly sample field/radio errors.

## WAT and other supported steady-flow sources

For a WAT, use a prepared `fr2dynage-radio-1` image configuration with the
PyRATS bridge set to `field_basis: "equipartition"` and `field_scale: 0.4`.
Version 0.6 uses `model.kind: "steady_jp_tribble_advection"` for the supplied
WAT. Its free-streaming JP–Tribble kernel has Gaussian field components,
Maxwell-distributed magnitudes and component standard deviation
`B_rms/sqrt(3)`. Local fields govern emission; a common JP loss history uses
the spatial average of B², with **B_rms = 0.4 B_eq**. This declared model
variant follows [Hardcastle (2013)](https://arxiv.org/abs/1306.1640); it is not
a local-field-loss cell mixture or a claim of exact BRATS binary equivalence.
The homogeneous-JP option remains for explicit historical compatibility.
The [radio inference guide](RADIO_INFERENCE_GUIDE.md) describes aligned
common-beam FITS inputs, RMS/calibration errors, masks, representative
regions, traced branches, and flow/injection priors. The core coordinate,
redshift and radio images do not by themselves measure the plasma speed.

```bash
python pyradage.py analyze radio.json --output-dir results/wat-analysis \
  --pyrats-budget PyRATS_adopted_field_budget.json
```

The budget option is optional. If supplied, it must match the imported
PyRATS report hash and contain energies/pressure recomputed at `0.4 B_eq`.
Native equipartition budgets are not rescaled merely by changing a field
label. `--pyrats-properties`, `--sensitivity` and `--bootstrap` are FR II
options; radio inference uses the PyRATS and sampling settings in `radio.json`.

The directory includes the posterior median transport-time FITS map with
uncertainty/quality layers, posterior draws, a preview, branch time-distance
profiles, fitted regional spectra, and posterior summaries of spatial
mean/median transport time. Displayed regional error bars are thermal RMS;
the inference retains its full configured covariance, including shared
calibration and spatial correlations. The plotted spectra use fitted
amplitudes at the best sampled parameter vector, not posterior-predictive
credible bands. Image pixels and neighbouring fitted regions are correlated.

Include the core with the optional root-level `core` component documented in
the [radio inference guide](RADIO_INFERENCE_GUIDE.md#include-the-central-region).
Set `core.policy: "include"` to put core-region spectra in the main likelihood
and map conditional transport time wherever the supplied geometry is valid.
Its registered mask is then a diagnostic label, not an exclusion. No circular
outline or grey aperture is drawn. `SOURCE_VALID` records detected source
emission; `CORE` identifies the central diagnostic footprint;
`AGE_VALID`/`VALID` and `AGE_FLAGS` record actual transport coverage. The
supplied WAT has 5,019 geometry-supported transport candidates, including all
805 central pixels. The former separate-core policy is retained only when
explicitly selected or inherited from a legacy configuration with no policy.

When the central-beam spectrum is detected, the report adds
`core_spectrum.csv` and `core_spectrum.png` with a descriptive GLS power law.
This measures brightness within a common restoring beam, which can mix the
nucleus, inner jets and projected plume emission. It is not a deblended
compact-core measurement, a test for absorption or an independent age
estimate. Under the include policy the central data also enter the same
radiative/transport model as other selected emission. Including them does
not resolve path/branch ambiguity, beam mixing or an absent spectral break.
Exact zero-distance points have zero conditional geometric travel time; a
whole finite beam is not assigned zero age from its position. Mean/median
transport summaries use `AGE_VALID` pixels, including age-valid core pixels.

To generate full-source radiative maps as well, add:

```json
{
  "resolved_analysis": {
    "models": ["jp_tribble", "ci"],
    "age_bounds_myr": [0.0, 1000.0],
    "age_grid_size": 241,
    "alpha_inj": "transport_posterior_median"
  }
}
```

The same `analyze radio.json --pyrats-budget ...` command then writes
`resolved_jp_tribble.fits` and `resolved_ci.fits`, each with JSON metadata and
publication PNG/PDF/SVG maps. These fits use the full source mask rather
than the transport paths: all 7,041 detected WAT source pixels, including
the core, are candidates. The JP–Tribble clock is a radiative elapsed time.
The CI clock is the duration of continuous injection; it is not a
single-parcel transport time or a global dynamical lifetime. Both remain
distinct from the posterior transport map.

Resolved profiles hold the common field and injection index fixed. By
default the index is the transport posterior median; a numeric fixed index
is also supported. A TJP-derived index is a dependent input to the CI
comparison, and the PyRATS field is derived using the same radio data.
These fits do not jointly propagate field/index/radio covariance or infer
spatial field changes. Their fixed-field/index profile crossings are
conditional likelihood conventions, not posterior credible intervals.

Inspect `SPEC_VALID`, `AGE_CONSTRAINED`, `REDCHI2` and `AGE_FLAGS` alongside
the best ages. A zero-boundary fit with a finite `AGE_UPPER95` is an
upper-limit case, not an established zero-age measurement. Uncrossed
profile endpoints remain NaN and flagged. At a fixed injection index, CI
can fit old steep tails poorly, and either model can fail to reproduce a
central spectrum flatter than that index; the corresponding best fit is
retained with quality flags rather than hidden behind a core mask.
`AGE_B16`/`AGE_B84` rescale the best time at alternative common-field
percentiles. They are field-sensitivity maps, not age confidence/credible
bounds. See the [radio guide](RADIO_INFERENCE_GUIDE.md#full-source-jptribble-and-ci-spectral-maps)
for the exact layers, flags, assumptions and input validation.

For the supplied 11-band WAT source at `z=0.1469`, the adopted PyRATS field
and flow priors produce a conditional transport map. Weak measured curvature
means these times can be driven mainly by the priors. They must not be
reported as an independently measured absolute dynamical lifetime.
The WAT report explicitly leaves global source age, FR II injection power
and ambient-density normalization unidentified. Dividing a PyRATS energy by
a plasma transport time does not establish jet power.

No universal radio-image-only dynamical model exists in this implementation.
Independent speed/geometry constraints use the separate `map-age` route;
remnant and component-resolved restarted FR II systems use their existing
dedicated fit commands. Reacceleration, entrainment, diffusion, restart and
time-variable flow require models beyond the present steady WAT analysis.

## Products and safe reruns

Every analysis writes `REPORT.md`, finite `analysis.json`, machine-readable
CSV tables, PNG/PDF/SVG figures, `products_manifest.json` with product hashes, and
`run_manifest.json` listing the directory's files. FR II runs include spectrum,
fit comparison, intersection, evolution and derived-quantity tables; the
classical alpha profile is included when available. Radio runs retain
`transport.fits`, `radio-summary.json`, `posterior.npz` and `transport.png`
alongside the added diagnostic products.

All plot routes use the shared publication style introduced in 0.5.2:
18 pt base text, 20 pt axis labels, 17 pt tick labels and 16 pt legends,
with enlarged titles and colour-bar labels. PNGs are exported at 600 dpi;
matching PDF and SVG companions preserve vector text and line work. For
example, `core_spectrum.png` is accompanied by `core_spectrum.pdf` and
`core_spectrum.svg`. The report embeds PNGs and links the vector exports.
All generated figure formats are included in the product hashes and
collision checks.

Regional spectra are paginated four per page. Thus 32 representative
spectra produce eight `regional_spectra_XX` pages, each in the three export
formats. This changes presentation and file counts, not the representative
regions or the inference. Core-age flags, posterior intervals and the
distinction between transport time and global source lifetime are retained.

Use PDF or SVG for manuscript line plots where the journal accepts them.
Inspect type, legends and panel spacing at the final printed column width:
reducing a large figure also reduces the apparent font size. Increasing PNG
dpi improves export sampling; it does not increase the angular resolution
or information content of the input radio images.

The report preserves normalized inputs, physical clock labels, field
provenance, failure reasons and quality flags. A new run stages all products
before publishing the directory. Existing directories are preserved unless
`--overwrite` is specified; only a recognized prior analysis can be replaced,
and its previous directory is retained as a timestamped backup. Input files
cannot live inside the directory being replaced. A failed fit or product
write leaves an existing result intact.

Use `pytest -q` for the automated suite. Numerical tests cover the historical
published benchmark separately from the new anchored synthetic example,
field/energy closure, fixed-environment evolution, JSON/table/figure writing,
and failure/collision handling. Passing tests verifies implemented numerical
behaviour; source-specific morphology and data quality still need assessment.
