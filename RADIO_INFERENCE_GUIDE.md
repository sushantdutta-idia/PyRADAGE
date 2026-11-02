# Radio-only spectral and transport inference

See [parameter reliability checks](PARAMETER_RELIABILITY.md) for v0.7.0
injection-index sensitivity products, reporting gates, physical accounting,
distance-scale provenance and `verify-analysis`.

This workflow estimates a **model-conditioned plasma transport time** from
registered multi-frequency radio images, a redshift, a path-length map, and
explicit physical priors. A new target does not need a measured speed. Its
speed and age estimates therefore retain uncertainty from the reference-source
or user-supplied priors.

Version 0.6 uses JP–Tribble for the supplied WAT transport model and can also
fit full-source JP–Tribble and CI spectral maps. The global source dynamical
age remains explicitly unavailable: a global episode lifetime, a plasma
travel time, a single-injection radiative time and a CI injection duration
are different physical clocks.

## Physical scope

The current WAT configuration uses `steady_jp_tribble_advection`: steady
projected advection with a single-injection JP–Tribble electron spectrum.
Each supported branch has an inlet speed `v0`; a shared deceleration parameter
`eta` gives

```text
v_projected(s) = v0 / (1 + eta * s/L)
t_flow(s) [Myr] = 977.7922 / v0[km/s] * (s[kpc] + eta*s[kpc]^2/(2*L[kpc]))
```

`L` is the maximum distance in the supplied geometry and selection mask for
that branch, before the spectral-detection cut. `eta=0` gives constant speed.
By default `eta` is supplied in the model configuration; an explicit
`deceleration` prior can instead allow the radio likelihood to update it.
The path origin sets zero transport time. The model assumes that particle
injection/last acceleration at that origin also starts the radiative clock.
These assumptions must be reviewed for the target, especially where a core or
jet-to-plume transition is unresolved.

The JP–Tribble kernel follows the rapid-diffusion/free-streaming limit of
[Hardcastle (2013)](https://arxiv.org/abs/1306.1640): isotropic Gaussian field
components give Maxwell-distributed local field magnitudes, with component
standard deviation `sigma = B_rms/sqrt(3)`. Electrons have a common JP loss
history in `B_loss = B_rms`, using the spatial average of B² plus the CMB
inverse-Compton losses. Local emitting fields vary, while electron density
and the injected spectrum are shared. The fixed observed-source prescription
defaults to **B_rms = 0.4 B_eq**. The explicit WAT override described below
instead fixes B_rms to 2.24 μG. This is not a mixture of isolated cells cooling
in its own local field, and no bitwise equivalence to BRATS or synchrofit is
claimed. See also [Harwood et al. (2013)](https://arxiv.org/abs/1308.4137).

The homogeneous-field `steady_jp_advection` option remains an explicit
historical/reference route for existing configurations and numerical tests.
It is not the model selected by the new WAT input. Neither transport option
includes adiabatic losses, reacceleration, evolving fields, arbitrary mixing
histories or restarted activity. A poor fit is evidence against the declared
model; it must not be interpreted as an accurate dynamical age.

The emitting flow is treated as nonrelativistic. Doppler frequency shifts and
fluid-frame time corrections are omitted. The numerical speed bound of 0.3c
does not establish that this approximation is accurate near that bound; use
priors appropriate to the nonrelativistic emitting plume being modelled.

This transport model is a restricted approximation for resolved plumes. It is
not a hydrodynamic model of WAT or NAT bending, jet power, entrainment, or host
motion. The existing KA/KDA commands remain the separate FR II evolution route.
A morphology label does not make an unsupported model physically applicable.

## What the radio images constrain

Spectral curvature constrains a loss scale. In the constant-field model,

```text
nu_break ∝ B / ((B^2 + B_CMB^2)^2 * t^2).
```

Different magnetic-field and flow-speed combinations can therefore predict
the same spectra. More image pixels do not remove this degeneracy. The output
must record which quantities are inferred conditionally and which priors set
their absolute scale. A narrow age posterior alone does not show that the data
independently measured an absolute age.

The frequencies must constrain spectral curvature at the relevant locations.
A nearly unbroken power law may give only a bound or an estimate dominated by
the prior. Count usable frequency coverage and inspect diagnostics rather than
treating the number of input files as a guarantee of identifiability.

## Preparing the images

- Use registered celestial FITS images with a common restoring beam and
  supported Jy/beam brightness units.
- Supply the observing frequencies, post-processing thermal RMS estimates and
  calibration fractions. Redshift sets the CMB losses and the physical scale.
- Verify comparable recoverable angular scales and flux calibration. Matching
  the restoring beam does not establish matched interferometric coverage.
- Supply a reviewed projected path-length map in kpc and identify distinct
  branches. For bent sources, radial distance from the core is insufficient.
- Include the core in the source footprint and use `core.policy: "include"`
  in the optional `core` component below. Keep ambiguous near-origin paths
  explicit. The core's radio spectra enter the declared model rather than a
  fixed circular exclusion. Geometry remains an adopted physical input.

The likelihood samples the emission at beam-scale locations rather than
treating every image pixel as an independent spectrum. Shared band calibration
and a Gaussian approximation to spatial noise covariance are included. Real
image noise can differ from a restoring-beam Gaussian and must be assessed.
Thermal noise is assumed independent between frequency images. Overlapping
subbands or other shared thermal errors are not represented by this model.
The optional calibration-correlation matrix describes calibration uncertainty;
it does not supply a cross-frequency thermal covariance model.
The resulting smooth age field is a parametric model projected onto the image;
sub-beam features are not independent resolved age measurements.

## Example sources and calibration

Reference sources calibrate distributions of physical parameters. They do not
provide ages to copy onto a target. Use a catalogue with source identity,
morphology, provenance, measurement uncertainty and an explicit physical
quantity. A host-galaxy velocity or an intrinsic inner-jet speed cannot be
entered as a projected plume transport speed.

Observed reference calibration requires multiple independent sources of the
same class. The target is excluded by identity. Spectral-derived velocities
cannot provide an independent validation of a spectral-plus-transport model.
Synthetic catalogues are useful for software verification but must be opted
into explicitly and remain labelled synthetic in the output.

The calibration estimates empirical population predictive priors. It is not a
full hierarchical fit to all reference images, and it does not guarantee
transfer to sources outside the reference sample's size, luminosity, redshift,
environment or activity range. Preserve published age definitions and model
assumptions; a published spectral age is not a dynamical-age ground truth.

Validate calibration by holding out entire sources. Randomly splitting pixels
from one object between training and testing creates leakage. A source-level
age cannot supervise a true resolved age map.

## Reported ages and summary statistics

Keep the following outputs distinct:

| Quantity | Interpretation |
|---|---|
| Posterior transport-time map | Time since model plasma entered the adopted path |
| Resolved JP–Tribble spectral time | Single-injection radiative time at fixed field and injection index |
| Resolved CI spectral time | Duration of continuous injection at fixed field and injection index |
| Mean/median mapped transport time | Spatial summary within the explicit valid mask |
| Endpoint transport time | Model travel time along the adopted complete branch |
| Global source dynamical age | Not identified by this steady-flow model |

An area-weighted mean, an area-weighted median and a reference-frequency
flux-weighted mean answer different questions. None is automatically the
source's lifetime or the age of undetected plasma. Report the weighting,
reference frequency and mask with each statistic.

The implemented summaries are area-weighted; flux-weighted ages are not
currently computed. Equal image-pixel areas use the small-field tangent-plane
approximation. Inspect `map_statistics.area_weighted_mean` and
`map_statistics.area_weighted_median` in the JSON report. Each includes the
posterior mean, median and central 68%/95% intervals of that spatial statistic.

Compute spatial summaries for every shared posterior draw, then summarize the
resulting distributions. Averaging marginal error bars or treating pixels as
independent would discard the common velocity and magnetic-field uncertainty.
Use posterior means, medians and credible intervals without describing them
as repeated independent measurements.

## Full-source JP–Tribble and CI spectral maps

The `analyze` command can add a geometry-independent resolved spectral stage.
Merge the following settings into a prepared radio-image configuration:

```json
{
  "model": {
    "kind": "steady_jp_tribble_advection",
    "log_emissivity_scatter": 0.1,
    "beam_path_nodes": 16
  },
  "core": {
    "mask": "core_mask.fits",
    "position_icrs_deg": [53.044708333333325, -27.443166666666666],
    "description": "Central diagnostic footprint; included in model fits, with provisional near-origin geometry.",
    "policy": "include"
  },
  "resolved_analysis": {
    "models": ["jp_tribble", "ci"],
    "age_bounds_myr": [0.0, 1000.0],
    "age_grid_size": 241,
    "alpha_inj": "transport_posterior_median"
  }
}
```

The existing source, images, selection mask, geometry, PyRATS and priors
entries are still required. Set `pyrats_properties.field_basis` to
`"equipartition"` and `field_scale` to `0.4`. Then run:

```bash
python pyradage.py analyze radio.json --output-dir results/wat-tjp-ci \
  --pyrats-budget PyRATS_adopted_field_budget.json
```

`resolved_analysis.models` accepts either or both distinct model names.
Its age bounds require `0 <= lower < upper <= 100000` Myr and its grid size
is an integer from 81 to 2001. When the lower bound is zero, the grid contains
an exact zero plus logarithmically spaced positive times. The supplied WAT
uses 241 points from zero to 1000 Myr; refine the grid when a profile is too
narrow to resolve. `alpha_inj` can be a fixed number from 0.5 to 1.5 or
`"transport_posterior_median"`.

The resolved stage fits every selected pixel meeting the positive-detection
criteria, without using path length, branch assignment or a circular core
cut. The current WAT has **7,041 detected source pixels**, including all
**805 central pixels**. Its separate transport fit covers **5,019 pixels**
with supplied geometry, also including those 805 central pixels. Source
pixels lacking a traced transport path still receive both spectral fits.
Spectral fits do not create missing physical transport geometry.

JP–Tribble uses the free-streaming kernel defined above. CI continuously
injects electrons from time zero to the observing epoch in a homogeneous
field, summing JP-cooled cohorts between ages zero and T. Its fitted T is
an **injection duration**, not an advected parcel's travel time or an
independent global dynamical lifetime. Its high-frequency index approaches
`alpha_inj + 0.5`; old steep tails can therefore fit poorly at the fixed
injection index. Keep that result and its quality flag rather than forcing
the model to fit. For context on integrated CI and resolved clocks, see
[Harwood (2017)](https://arxiv.org/abs/1612.04390).

Both resolved models fix the field at the median of the shared transport
field samples. The JP–Tribble input is the RMS field; the CI input is a
homogeneous field. The selected injection index is also fixed during each
age profile. A JP–Tribble transport-derived index is a dependent input to the
CI comparison. This is a conditional model comparison, not a jointly fitted
field/index distribution or an independent model-selection probability.
The PyRATS field reuses the target radio flux; full flux/field covariance,
spatial field variation, geometry and physical-model systematics are not
jointly propagated by these profiles.

Each model writes `resolved_jp_tribble.*` or `resolved_ci.*`: a FITS product,
finite JSON metadata, and large-font PNG/PDF/SVG maps. The FITS primary array
is the best sampled radiative time or CI duration. Important layers are:

| Layer | Interpretation |
|---|---|
| `SPEC_VALID` | A numerical spectral fit was obtained; this does not establish a unique age |
| `AGE_CONSTRAINED` | The profile and stated quality/curvature checks pass |
| `AGE_LO68`, `AGE_HI68` | Crossings of the conditional age profile at Δχ² = 1 |
| `AGE_UPPER95` | Upper crossing at Δχ² = 2.71, if reached within the bounds |
| `REDCHI2` | Model fit quality with field and injection index held fixed |
| `AGE_FLAGS` | Boundary, curvature, fit-quality, open/disconnected-profile or grid-resolution flags |
| `AGE_B16`, `AGE_B84` | Conditional rescaling at the 16th/84th percentiles of the common field samples |

Open profile endpoints are NaN and flagged; they are not replaced with an
artificial finite error or the search bound. A best age at zero is retained
as a boundary minimum. Where its upper crossing exists, interpret it as an
upper-limit case, not a precisely measured zero-age pixel. If a crossing is
not found, the age is unconstrained within the search range. These asymptotic
profile thresholds are not Bayesian credible intervals and do not guarantee
nominal coverage at boundaries or for weak curvature.

`AGE_B16` and `AGE_B84` hold the spectral shape fixed and rescale time for an
alternative shared field. They are field-sensitivity maps, not lower/upper
age confidence bounds or credible intervals. They need not be ordered like
age bounds because the age/field relation is not monotonic everywhere.
They do not jointly refit field, injection index and radio-flux covariance.
The age coordinate and search bounds are rescaled together, so boundary and
weak-curvature flags persist; these are not new bounded-grid fits.

The retained nominal PyRATS budget uses its upstream homogeneous power-law
spectrum (adopted α=0.5 for this WAT), not a new self-consistent JP–Tribble/CI
energy fit. Its emissivity normalization and particle budget are not jointly
refitted when the ageing model or injection index changes.

Inspect quality maps throughout the source, including the core. If the fixed
injection index is steeper than a central observed spectrum, ageing cannot
flatten that spectrum; a zero-boundary or poor fit can result. No absorption
claim or special central deletion follows from that outcome. Neighbouring
pixels and calibration errors are correlated, so resolved age-profile errors
must not be combined as independent measurements of a whole-source mean.

## Run the historical synthetic reference example

Install the optional image dependencies, then run the bundled synthetic inputs
from the package directory:

```bash
python -m pip install '.[maps]'
python pyradage.py infer-radio examples/radio_demo/radio.json \
  --output examples/radio_demo/transport.fits \
  --summary examples/radio_demo/summary.json \
  --samples examples/radio_demo/posterior.npz \
  --preview examples/radio_demo/preview.png
```

To generate fresh inputs, use
`python examples/generate_radio_demo.py --output results/new-demo` and run
inference against that directory's `radio.json`.

The example contains synthetic data only. It records the true inlet speeds,
field, injection index and transport field in `truth.json` and
`truth-age.fits`. The images are Gaussian-convolved emissivity fields, while
the inference uses an approximate within-region spectrum average. Recovery
therefore also probes the finite-beam approximation. The supplied field and
injection-index priors are deliberately informative and labelled as such.
This bundled homogeneous-JP reference preserves the earlier numerical
benchmark; use the explicit JP–Tribble configuration above for the new WAT
analysis rather than interpreting this legacy fixture as its model choice.

Add `--noise` when generating the demo to add seeded beam-correlated thermal
noise. Generate a separate no-curvature case with `--power-law`; it must not be
described as an independently measured dynamical age. Use a new output
directory or an explicit `--overwrite` when replacing generated files.

For observed images, start from `radio.json` and replace the source metadata,
image list, geometry and priors. Paths are resolved relative to the JSON file.
The schema is `fr2dynage-radio-1`. Every user prior needs a scientific
explanation in `prior_description`. A reference catalogue can supply priors,
and any explicit target overrides remain recorded in provenance. Priors and
model assumptions must accompany the reported ages.

## Inference configuration and output checks

`source` requires `name`, a supported `radio_class` (`fr1`, `wat`, `nat` or
`head_tail`), a positive numeric `redshift`, and `activity_state: "active"`.
The four-to-64 `images` entries follow the existing map schema: `path`,
`frequency_mhz`, one of `rms_jy_per_beam` or `rms_map`, and an optional
`calibration_fraction`. Frequencies must be distinct. An optional `hdu` selects
a FITS extension. Image preparation remains explicit.

`geometry` requires `distance_map`, `branch_map` and `description`. The
distance image has `BUNIT="kpc"`; the registered branch image has `BUNIT="1"`
and finite integer labels from 0 to 32767. Zero excludes pixels from flow
geometry; selected emission can still appear in source spectral coverage. A branch is a
physical flow path, not a fitted region; every branch receives its own inlet
speed. `selection` accepts `mask`, `snr_min` and `min_bands` (at least four).

### Include the central region

The optional root-level `core` component labels a registered central
diagnostic footprint. To include this emission in the model fits, explicitly
set `policy` to `"include"`, as in the current WAT configuration:

```json
{
  "core": {
    "mask": "core_mask.fits",
    "position_icrs_deg": [53.0447083333, -27.4431666667],
    "description": "Core included in model fits; central paths remain provisional at the common beam.",
    "policy": "include"
  }
}
```

The position is optional; if supplied, use numeric ICRS RA/Dec in degrees.
The core mask uses finite binary 0/1 values and the same celestial grid as
the images. Its radius is an explicit data-preparation assumption, not a
measured nuclear size or a boundary imposed by the spectral model. The explicit
core mask expands the selected source footprint by union even if an older
`selection.mask` omitted it. `SOURCE_VALID` still requires the configured
spectral detections. Supply the full source footprint in `selection.mask`;
reviewed branch geometry determines which additional emission has an
interpretable conditional transport time. Under `policy: "include"`, the
central footprint does not remove pixels from the likelihood or transport
maps, and no circular outline or grey overlay is drawn. The coordinate may
still be marked with a cross. FITS records `COREPOL=INCLUDE` and
`COREAGE=MODELLED`, with actual modelled coverage in the JSON metadata.

`policy: "separate"` retains the older treatment that includes the central
image/spectrum while leaving its transport age undefined. A missing policy
retains that legacy behaviour for reproducibility; it is not the setting in
the new WAT configuration. No geometry or age is manufactured for a source
pixel merely by changing this policy.

The core remains in the source brightness and spectral-index products. A
central restoring-beam spectrum is extracted at the detected pixel nearest
the supplied position; without a position the documented SNR rule selects
one representative pixel. A descriptive GLS power law reports its spectral
index, conditional error and residual fit quality with thermal and supplied
inter-band calibration covariance. This is beam brightness in Jy/beam,
not an aperture-integrated or deblended nuclear flux. The fit does not
establish opacity, a break frequency or an age. This descriptive diagnostic
is not an additional independent likelihood. Under the include policy, the
core's representative spectra also enter the main radiative/transport model
on the same basis as other selected emission.

The FITS layers distinguish observed emission from age coverage:

| Layer | Meaning |
|---|---|
| `SOURCE_VALID` | Selected source pixels passing the spectral-detection criteria, including the core |
| `CORE` | A central diagnostic label; not an exclusion under policy `include` |
| `AGE_VALID` and `VALID` | Pixels assigned a conditional plume transport time; these two layers agree |
| `AGE_FLAGS` | Bit flags explaining unavailable or unresolved transport ages |

`AGE_FLAGS` uses flag value 1 for outside the selected source footprint, 2 for
insufficient usable spectral detections, 4 for the legacy separate-core
treatment, and 8 for missing/invalid flow geometry. Flag 4 is not applied
merely because a pixel lies in the included core. Flags combine by bitwise
OR. The JSON `core` object records the policy, selected diagnostic pixel,
detected/modelled central coverage and central spectrum. `analyze` additionally
writes `core_spectrum.csv` and `core_spectrum.png` when a core spectrum is
available, with matching `core_spectrum.pdf` and `core_spectrum.svg` exports.

The core's finite beam can mix the nucleus, inner jets and projected plumes.
Including it permits a model fit; it does not establish a unique physical
core age or resolve those populations. The supplied core coordinate
sets an adopted geometrical zero-time reference, not a measured zero-age
region. Exact zero path values can yield zero conditional travel time at
the adopted reference, but an entire central aperture is not assigned zero
age by position. Mean/median transport summaries include all `AGE_VALID`
emission, including age-valid core pixels under the include policy. They
remain model-conditioned spatial statistics rather than whole-source ages.
The default magnetic-field rule is `B = 0.4 B_eq`, interpreted as the
RMS field for the free-streaming JP–Tribble model. A declared user-fixed field
replaces that input for its run without altering the kernel convention.

### Figure exports

Radio previews and source-analysis plots use the same enlarged publication
style: 18 pt base text, 20 pt axis labels, 17 pt tick labels and 16 pt
legends, including WCS coordinates and colour bars. Supplying
`--preview results/transport.png` creates a 600 dpi PNG plus
`results/transport.pdf` and `results/transport.svg`. The `analyze` report
likewise provides all three formats for its spectra, profiles and summary
figures. Regional spectra use four panels per page; 32 regions therefore
produce eight pages instead of two crowded sixteen-panel grids.

The larger type and export formats change presentation only. Displayed SED
error bars still represent thermal RMS, and the full configured covariance
remains part of the inference. The current include-core configuration has no
circular or grey central mask. Grey central coverage appears only for the
explicit legacy separate-core treatment. Changing plot style creates no
additional independent age measurement.

Prefer PDF/SVG for manuscript text and line work when accepted by the
journal, and inspect each figure at its final column width. Font sizes
shrink with any reduction of the entire figure. Raster radio maps retain
their input angular resolution within vector exports; 600 dpi is a drawing
setting, not extra spatial information. Existing companion files are also
protected by output-collision checks unless overwrite is requested.

The current `model` object uses `kind: "steady_jp_tribble_advection"`.
`"steady_jp_advection"` remains an explicit homogeneous-JP compatibility
choice. CI is selected in `resolved_analysis`, not as a parcel-advection
kernel. Optional transport settings
are `deceleration` (default zero), `log_emissivity_scatter` (default 0.05), and
`beam_path_nodes` (default 12). The scatter is a user-chosen model discrepancy
in natural-log intensity, not a measurement of unresolved mixing.

`priors` must provide `flow_speed_km_s`, `magnetic_field_ug` and `alpha_inj`,
except where a reference catalogue, PyRATS route or explicit fixed-field input
provides the corresponding parameter. Do not supply a field prior together
with `fixed_magnetic_field`. The same speed prior is
applied independently to each branch. Supported distributions are:

```json
{
  "flow_speed_km_s": {
    "distribution": "lognormal", "log_mean": 8.5171931914, "log_sigma": 0.5
  },
  "magnetic_field_ug": {
    "distribution": "log_uniform", "low": 0.5, "high": 10.0
  },
  "alpha_inj": {
    "distribution": "normal", "mean": 0.65, "sigma": 0.1
  }
}
```

These numerical values are syntax examples, not recommended priors for an
observed source. `log_mean` and `log_sigma` use natural logarithms. `uniform`
and `log_uniform` use `low`/`high`; `normal` uses `mean`/`sigma`; `lognormal`
uses `log_mean`/`log_sigma`. All are truncated within the documented physical
bounds: speed 1–89937.7374 km/s, field 0.01–1000 uG, injection index 0.5–1.5,
and deceleration 0–10. Optional tighter bounds must lie within those ranges.
To infer deceleration, supply `priors.deceleration` and omit
`model.deceleration`.

`inference` accepts `particles` (64–4096, default 512), `seed`, `mcmc_steps`
(default four), `ess_target` (default 0.7), `max_stages` (default 80), and
`replicates` (one to four, default two). Independent tempered-SMC runs are
compared; agreement is a convergence diagnostic, not a proof. Use more
particles/rejuvenation and compare independent runs when intervals matter.
Here `mcmc_steps` controls the field-fixed flow-shape sweeps per stage. Each
stage also makes one generic proposal and one field–speed ridge refresh.
The shape block proposes in `log(v/h(B))`, injection-index and deceleration
coordinates, preserving their correlations without the curved B–speed ridge.
Acceptance rates for generic, field and shape moves are reported separately.
These field-refresh descriptions apply to uncertain-field inference. An
explicit fixed-field run holds B at its supplied value and performs no
field-prior or field-speed-ridge refresh; its intervals condition on that B.

Sampler rejuvenation also includes a dedicated move along the field/speed
degeneracy of the declared common-loss spectral model. It holds
`v_branch / h(B)` fixed, where `h(B)=a(B)/sqrt(B)` and `a(B)` is the JP loss
coefficient. A proposed field drawn from its prior rescales all branch speeds
by `h(B_new)/h(B_old)`. The Metropolis acceptance includes the velocity-prior
density ratio, the physical-coordinate Jacobian and any numerical likelihood
change. This move improves exploration of the degenerate posterior; it does
not provide new information that determines the magnetic field or absolute
age. Its acceptance diagnostics are retained in the sampler report.

Additional shape proposals operate on `log(v_branch/h(B))` and the remaining
latent parameters at fixed field. Their covariance can represent correlations
between the spectral shape, injection index and deceleration without spanning
the curved field/speed ridge. The transformed target retains the velocity
prior and its coordinate Jacobian. Field-refresh and shape-proposal acceptance
are reported separately; neither replaces independent-run comparisons.

Optional `calibration_correlation` is a positive-semidefinite correlation
matrix in the exact order of the input images. It describes shared band
calibration uncertainty; omitted entries default to independent bands. The
calibration error in one band is shared across all sampled regions.

The JSON reports `status`, `fit`, `identifiability`, `sampling`, posterior and
prior parameter summaries, region locations, input hashes and assumptions.
Important statuses are:

- `model_conditioned`: the restricted model passes the stated fit and sampler
  diagnostics and curvature heuristic; its absolute scale still depends on
  priors and model assumptions.
- `curvature_unconstrained`: the radio data do not require ageing curvature under
  the implemented diagnostic.
- `sampling_unverified`: independent sampler agreement has not been
  established.
- `model_inadequate`: fit quality rejects the restricted model at the stated
  threshold; age summaries are diagnostic only.

Fit and sampling statuses take precedence over the curvature status. Always
inspect `identifiability.curvature` as well. The curvature check compares
chi-square with independent regional power laws and is explicitly a
non-nested heuristic, not a formal detection significance or Bayes factor.

The FITS primary array is the posterior median `TRANSPORT` field in Myr.
`T_LO68`, `T_HI68`, `T_LO95` and `T_HI95` give central credible bounds.
`VALID`, `BRANCH`, `DIST_KPC`, `REGION_ID`, `N_BANDS`, `ALPHA` and
`PROVENANCE` support inspection. `REGION_ID` marks representative likelihood
sample locations, not independent ages for every pixel. The optional NPZ
contains shared posterior parameter draws and per-draw map statistics.

## Reference catalogue schema and commands

The demo also generates `reference-synthetic.json`, containing three
explicitly synthetic same-class sources. Exercise its calibration with:

```bash
python pyradage.py calibrate-radio-priors examples/radio_demo/reference-synthetic.json \
  --target-name 'Synthetic steady-JP WAT' --radio-class wat \
  --allow-synthetic --output examples/radio_demo/calibrated-priors.json
```

This command writes parameter priors and calibration provenance. It does not
fit an age. Synthetic calibration is for verification only. An observed
reference catalogue uses the same schema with `origin: "observed"` and real
citations/constraints. A complete catalogue has this form; the illustrative
one-source catalogue below intentionally cannot calibrate a population:

```json
{
  "schema_version": "fr2dynage-reference-1",
  "description": "Synthetic schema illustration, not measured source properties",
  "sources": [
    {
      "name": "Synthetic reference 1",
      "aliases": ["Synthetic alias 1"],
      "radio_class": "wat",
      "origin": "synthetic",
      "reference": "Declared example parameters; no observational citation",
      "constraints": {
        "flow_speed_km_s": {
          "value": 5000.0, "error": 500.0, "unit": "km/s",
          "quantity": "projected_plasma_transport_speed",
          "method": "Declared synthetic model input",
          "independent": true
        },
        "magnetic_field_ug": {
          "value": 3.0, "error": 0.2, "unit": "uG",
          "quantity": "emitting_plasma_magnetic_field",
          "method": "Declared synthetic model input",
          "independent": true
        },
        "alpha_inj": {
          "value": 0.65, "error": 0.025, "unit": "1",
          "quantity": "injection_spectral_index",
          "method": "Declared synthetic model input",
          "independent": true
        }
      }
    }
  ]
}
```

Every constraint requires a positive uncertainty, the exact unit/quantity
shown above, a method and an explicit independence declaration. Constraints
can be omitted; at least three eligible independent same-class sources are
needed **per calibrated parameter**. Source names and aliases must identify
distinct sources. Target matches are excluded; unknown aliases cannot be
discovered automatically. Different classes, synthetic sources without an
opt-in, and dependent constraints without an opt-in are excluded and logged.
Allowing dependent constraints does not make the resulting inference an
independent dynamical validation.

To calibrate directly during inference, add the following root configuration
field and omit any explicit priors that should come from the catalogue:

```json
{
  "reference_catalogue": {
    "path": "reference-synthetic.json",
    "allow_synthetic": true,
    "allow_dependent": false
  }
}
```

For observed references, omit the synthetic opt-in. Explicit `priors` entries
override catalogue-derived entries; the report records those overrides.
`prior_description` is still required. The calibrator uses empirical
random-effects predictive distributions with declared scatter floors, not a
fully sampled hierarchical posterior. It does not infer population-selection
bias or covariance between calibrated parameters.

## Using an explicit fixed source field

When a field is supplied for a particular source analysis, record it directly
in tesla and describe its provenance. The WAT rerun requested on 2026-11-02 uses:

```json
{
  "fixed_magnetic_field": {
    "value_t": 2.24e-10,
    "description": "User-supplied WAT field; replaces the default 0.4 B_eq input for this run"
  }
}
```

This is exactly **2.24 μG**. It is the RMS B for the declared free-streaming
JP–Tribble model and homogeneous B for CI. The field is held fixed rather than
represented by an arbitrary narrow prior. With no supplied field uncertainty,
reported time intervals condition on this exact field and omit its uncertainty.
A fixed B does not establish curvature or identify a global dynamical lifetime.

Omit `priors.magnetic_field_ug`. A retained `pyrats_properties` report supplies
source geometry, composition, native equipartition comparisons and provenance;
its field MC distribution is not used as the adopted-field prior in this mode.
The actual B_eq stays the value reported by PyRATS. Never infer a replacement
B_eq as B/0.4 or force the actual comparison ratio to be 0.4. Products identify
the field as user-fixed and do not label the adopted field `BFRACEQ=0.4`.

Recompute the source's electron/particle normalization, magnetic energy, total
energy and pressure at the supplied B while retaining the declared observed
flux, volume, composition and electron-spectrum assumptions. Native budgets
at other fields are comparisons, not the adopted source budget. The accounting
validation checks the requested fixed field and the actual ratio to the
nominal native B_eq. It retains the same unit, composition and energy/pressure
closure checks as the default route.

The detected core remains included. JP–Tribble radiative time, CI injection
duration, geometric transport time and whole-source dynamical lifetime retain
their distinct meanings. Field-fixing alone does not supply jet power,
ambient density or a source turn-on date for a WAT.

## Using PyRATS magnetic fields and source properties

The optional PyRATS bridge reads an existing `PyRATS_results.json` from the
user's [PyRATS](https://github.com/sushantdutta-idia/PyRATS) analysis. PyRATS is
an external program; its source is not bundled or executed by this reader.
Prepare and analyse the radio images with PyRATS first, or reuse a verified
completed analysis of the same source. Then omit an explicit
`priors.magnetic_field_ug` and add this root configuration field:

```json
{
  "pyrats_properties": {
    "path": "PyRATS_results.json",
    "field_basis": "equipartition",
    "field_scale": 0.4,
    "systematic_log_sigma": 0.0
  }
}
```

In the default PyRATS field-input mode, the source prescription is fixed at
**B = 0.4 B_eq**. The ratio is not sampled. The explicit fixed-field mode above
overrides this input; it does not modify or fabricate the native field report.
The zero additional width above propagates the conditional PyRATS B_eq Monte
Carlo uncertainty alone. Any explicitly introduced extra uncertainty must
describe uncertainty in B_eq, not a varying field fraction, and be explained
in `prior_description`. The adapter combines the reported
field Monte-Carlo log-width with that explicit extra width. It centres the
lognormal approximation on the MC median, preserves the raw quantiles, and
records the deterministic field separately. `minimum_energy` is the other
retained native budget for comparison. In the default field-input mode,
source inference requires the
equipartition basis and `field_scale: 0.4`; other fractions/bases are rejected.
An upstream sub-equipartition multiplier is never silently applied. The
default-mode posterior archive includes B_eq draws and the FITS header records
`BFRACEQ=0.4`, so the fixed ratio can be checked.

The reader checks redshift, units and internal assumptions; it associates the
radio inputs by path or label/frequency, and verifies pixel hashes only where
the report actually provides them. It preserves source volume, dimensions,
native magnetic/electron/total energy budgets, pressure when reported,
particle-content ratio, filling factor, electron limits, cosmology and the
injection-index provenance. Those are conditional source properties, not an
independent plasma speed or a global source age. Native energy budgets are
not rescaled when a different field multiplier is adopted.

Equipartition is a physical closure assumption. This bridge is an approximate
plug-in field constraint derived from the same radio fluxes used in the age
fit. It does not implement a joint flux–field likelihood, propagate the full
shared covariance, or remove uncertainty in particle content or geometry.
Adding an arbitrary systematic width does not prove coverage. A PyRATS field
calculation also conditions on its injection spectrum; varying the transport
injection index does not automatically recompute that field. Recheck the
property calculation or test alternate assumptions when these differ.

A fixed upstream injection index is retained as provenance and is never
imported as a measured injection-index prior. The user still supplies or
calibrates the transport injection and speed priors. A PyRATS field replaces
any catalogue-calibrated field input; this precedence is recorded. Supplying
both `pyrats_properties` and an explicit field prior is rejected to avoid an
ambiguous analysis. All approximation warnings accompany the FITS/JSON
products, and the PyRATS JSON is protected against output overwriting.

## Scientific validation

Software tests should include zero-age power laws, an independently checked
synchrotron integral, known transport-law recovery, magnetic-field/speed
degeneracy, uninformative spectra, strict input rejection, output provenance,
source holdout and shared uncertainty propagation. Synthetic recovery checks
verify the implementation within the adopted model; they do not establish
that a real WAT follows that model.

The native JP–Tribble/CI checks additionally cover Maxwell-field moments,
independent emissivity quadrature, the single-field JP limit and the CI
high-frequency `alpha_inj + 0.5` limit. Resolved-profile tests check zero-age
boundaries, open/disconnected intervals, numerical fit coverage and quality
flags. Core-policy tests verify that inclusion changes actual likelihood/map
support and removes the circular display treatment, rather than merely
recolouring an excluded aperture. Numerical agreement with the declared
equations does not establish equality to another code's Tribble variant.

Published methodological precedents include
[RAiSE III](https://arxiv.org/abs/1711.04600),
[RAiSE synthetic synchrotron images](https://arxiv.org/abs/2206.09573), and
[BRATS spectral-age fitting](https://arxiv.org/abs/1308.4137).
The importance of mixed particle populations is illustrated by
[Jerrim et al.](https://doi.org/10.1017/pasa.2025.10114).
