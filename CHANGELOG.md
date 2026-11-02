# Changes

## PyRADAGE v1.0 — 2026-11-02

- Publish PyRADAGE v1.0 with consistent package, runtime and citation versions.
- Rename the project to **PyRADAGE**, with canonical `pyradage` and
  `pyradage-gui` commands, Python imports and GitHub repository metadata.
- Update the GUI, desktop launchers, documentation, citation metadata and CI
  for the new name. New automatic runs use `~/PyRADAGE-results` by default.
- Accept `PYRADAGE_PYRATS_SCRIPT` and `PYRADAGE_PYRATS_PYTHON`, retaining
  legacy environment-variable fallbacks. Keep existing `fr2dynage` commands,
  imports and scientific schema identifiers compatible.
- Preserve the scientific models, field prescriptions, detected-core policy,
  quality flags and title-free publication exports from version 0.8.0.

## 0.8.0 — 2026-11-02

- Add a native graphical image picker/redshift workflow, responsive background
  execution, progress, cancellation, safe new result directories and direct
  summary opening. Provide desktop launchers and a matching `auto` CLI.
- Automatically validate observing frequencies, units, WCS and beams; align
  surface brightness, match resolution, estimate noise, retain significant
  source islands and central emission, and record conditional geometry.
- Fit shared injection indices with beam-thermal/shared-calibration GLS and
  independent regional clocks. Generate full-source JP–Tribble and CI maps,
  profiles, flags, index/field scenarios and auditable statistics. Keep their
  different clocks separate; no global dynamical age or flow is invented.
- Obtain B=0.4 B_eq and nominal source budgets through isolated external
  PyRATS/pysynch execution with strict physical accounting/hash checks. Support
  explicit fixed fields without a backend; unavailable energetics stay absent.
- Export one combined portrait publication summary as PDF/PNG/SVG. Remove top
  titles from all pipeline figure exports by default; preserve readable fonts,
  WCS labels, units, beam/scale indicators and report captions.
- Add a synthetic GUI example, cross-platform/headless tests and a strict
  allowlisted GitHub export tool. Keep personal data, environments, local task
  instructions, backups and development validation history out of public source.

## 0.7.1 — 2026-11-02

- Support an explicit user-fixed magnetic field in tesla, taking precedence
  over the default B=0.4 B_eq prescription for that configuration. Remove the
  fixed field from sampling and free-parameter degrees of freedom; do not
  approximate it with a narrow prior or manufacture B_eq from B/0.4.
- Preserve native PyRATS geometry/property provenance while recomputing the
  nominal electron normalization, energy and internal pressure at the supplied
  field. Check its actual B/B_eq comparison without enforcing 0.4.
- Label user-fixed fields and excluded field uncertainty in JSON/FITS/reports.
  Omit artificial field-quantile sensitivity layers. Retain JP–Tribble/CI,
  core inclusion, profile-quality flags and distinct physical clock meanings.

## 0.7.0 — 2026-10-30

- Refit resolved JP–Tribble/CI profiles at checked injection-index endpoints;
  export age/flag scenarios and a conditional all-scenarios pass mask. Preserve
  every eligible core pixel, including flagged and boundary fits.
- Separate supported conditional parameter estimates from raw posterior/map
  diagnostics. Withhold whole-source spectral summaries if detected pixels
  fail the declared quality/sensitivity gates; retain diagnostic statistics.
- Check nominal adopted-field budget accounting in SI and against PyRATS's
  deterministic B_eq, volume and composition before expensive inference.
- Record/check path and PyRATS angular-distance scales; reject config/header
  contradictions and flag inconsistent model-input scales.
- Validate supplied PyRATS Monte-Carlo attempt/rejection counts and success
  fractions rather than trusting a status string alone.
- Recheck consumed input hashes after fitting and before publication. Add
  version-2 hashed output manifests and `verify-analysis [--check-inputs]`.

## 0.6.0 — 2026-10-30

- Add native JP–Tribble and continuous-injection emissivity kernels with
  independent quadrature and limiting-form checks. The JP–Tribble variant
  uses free-streaming electrons in a Gaussian random field, Maxwell field
  magnitudes and common losses in the RMS field. State the variant explicitly
  rather than claiming equivalence to an external Tribble executable.
- Use JP–Tribble for the supplied WAT transport analysis. Interpret the fixed
  source prescription as B_rms = 0.4 B_eq; CI retains homogeneous B = 0.4 B_eq.
  Keep homogeneous JP only as an explicit historical/reference option.
- Add `core.policy: "include"` to fit core spectra in the transport likelihood
  and map all geometry-supported central pixels without a circular exclusion
  or grey overlay. Preserve the old separate-core policy for legacy inputs.
- Add geometry-independent JP–Tribble and CI spectral-age profiles for every
  selected detected source pixel, including central emission and pixels with
  no traced transport path. Keep CI injection duration distinct from fluid
  travel time and source dynamical lifetime.
- Retain numerical best ages together with boundary, weak-curvature,
  poor-fit, open/disconnected-profile and grid-resolution flags. Do not turn
  a zero-boundary fit into an established zero age or fill an open uncertainty
  endpoint with a search-bound value.
- Export model-specific FITS/JSON and publication PNG/PDF/SVG maps. Record
  fixed field/index assumptions, same-radio-data reuse and conditional
  AGE_B16/AGE_B84 field sensitivity separately from profile intervals.

## 0.5.2 — 2026-10-30

- Apply a shared plotting style to source-analysis figures, radio transport
  maps and conditional kinematic-map previews, with enlarged labels, ticks,
  legends, titles and colour-bar text.
- Export 600 dpi PNG figures and matching vector PDF/SVG companions, with
  output-collision checks and hashes for the generated analysis products.
- Paginate regional spectral diagnostics four per page to preserve readable
  type and room for legends, error bars and axis labels.
- Regenerate current figures from saved scientific products. Preserve fitted
  parameters, data, core inclusion, unresolved-age flags and B = 0.4 B_eq.

## 0.5.1 — 2026-10-29

- Retain explicit core components in radio source footprints and spectral
  diagnostics, including a central-beam GLS spectrum with full configured
  inter-band calibration covariance. Do not infer a deblended nuclear flux
  or unique core age from common-beam images.
- Export source/core/age coverage, reason flags and reference intensity in
  FITS; display the core on intensity, spectral-index and age-preview panels
  with its unresolved-age status. Preserve tail-only transport statistics.
- Allow an exact zero-distance flow reference when otherwise valid; require
  a positive traced length for every modelled branch.
- Restore the WAT's 805 previously cut central pixels to source analysis
  and 132 additional inner-plume pixels to its conditional transport fit.
  Keep B = 0.4 B_eq and the previous data products unchanged.

## 0.5.0 — 2026-10-29

- Add `analyze` / `analyze-source` with atomic analysis directories, Markdown
  reports, CSV tables, figures, provenance and product hashes. Preserve prior
  recognized analyses as backups on explicit overwrite.
- Expose the classical frequency-intersection families, Delta(age) and
  minimum-Q*t injection-index profiles alongside weighted joint fits. Use the
  joint estimator as primary when both succeed, retaining method disagreement
  and failed-method reasons.
- Add descriptive spectrum/curvature diagnostics, numerical and physical
  quality flags, explicit assumption-sensitivity refits and fixed-Q,
  fixed-environment active FR II evolution with power-size diagrams.
- Add an active-source current-field anchor that enforces B = 0.4 B_eq with
  consistent magnetic/particle energy closure. Require gamma_b = gamma_c;
  reject conflicting magnetic ratios and inadequate energy. Preserve the
  historical unanchored partition for published regression benchmarks.
- Import compatible PyRATS FR II field references with recorded composition
  assumptions and conditional percentile sensitivity fits. Keep field/radio
  data reuse and unpropagated joint covariance explicit.
- Record successful measurement Monte Carlo draws and release sample mean
  and median global FR II ages only when the existing reliability threshold
  for uncertainty summaries is met.
- Add fitted regional spectra and branch transport profiles to WAT/radio
  reports while preserving posterior FITS maps and spatial age statistics.
  Keep global episode age, FR II injection power and ambient density explicitly
  unidentified by the steady transport model.
- Bundle a labelled anchored FR II synthetic example and a dedicated source
  analysis guide. The snapshot-field variant is not the original DYNAGE
  minimum-energy partition or a universal model for radio-galaxy classes.

## 0.4.1 — 2026-10-28

- Adopt the user's fixed source-field prescription B = 0.4 B_eq in the
  PyRATS-backed radio inference route. Reject conflicting fractions/bases.
- Propagate equipartition-field uncertainty while keeping the fraction fixed;
  record the prescription in provenance, posterior samples and FITS headers.
- Regenerate the WAT analysis with the PyRATS B_eq Monte Carlo uncertainty
  and remove the previous illustrative factor-of-two field-departure scatter.

## 0.4.0 — 2026-10-28

- Add radio-only, prior-conditioned steady JP-advection inference for active
  FR I/WAT/NAT/head-tail sources, with separate branch speeds and optional
  shared deceleration. Preserve the structural field-speed degeneracy.
- Add an isotropic synchrotron JP kernel checked against independent
  quadrature, beam-averaged spectral predictions and correlated log-flux
  likelihoods with shared calibration errors.
- Add strict reference-source catalogues and empirical predictive priors with
  source exclusion, class checks, uncertainty/provenance requirements and
  explicit synthetic/dependent-constraint opt-ins.
- Add an optional data-only PyRATS property bridge with field Monte-Carlo
  distributions, source energetics, geometry, assumption checks, explicit
  model-systematic uncertainty and shared-radio-data warnings.
- Add tempered SMC posterior draws, replicate diagnostics, curvature/model
  adequacy flags, credible-interval FITS maps, posterior spatial mean/median
  summaries, NPZ samples and PNG previews. Global source lifetime remains
  explicitly unidentified by this transport model.
- Include a convolved synthetic worked example, analytic sampler checks and
  end-to-end product validation. Existing FR II and conditional-map routes
  retain their physical scope.

## 0.3.0 — 2026-10-27

- Add `map-age`, `validate-map`, and `classes` commands for aligned radio
  FITS images and independently constrained motion/geometry.
- Add conditional expansion-age and flow-travel-time maps with explicit
  frames, expansion histories, scalar or mapped speeds/distances, independent
  error propagation, and invalid-pixel flags.
- Add spectral-index/error and power-law fit-quality maps, with thermal and
  per-band calibration errors and SNR/region masks. No spectral ages are
  inferred from spectral indices.
- Validate celestial registration, common restoring beams, brightness and
  auxiliary-map units; support 2D and leading-singleton radio FITS images.
- Store WCS, masks, diagnostics, input hashes, physical assumptions, software
  versions and clock labels together in an atomic FITS product; add PNG previews.
- Accept FR I/II/0, hybrid, WAT/NAT/head-tail, compact/CSS/GPS/CSO, diffuse,
  other and unknown source labels for constrained map analysis. Integrated
  KA/KDA fitting now explicitly rejects non-FR-II classes. Existing inputs
  without a class retain their FR II convention.
- Bundle deterministic expansion and curved-tail examples with analytic truth,
  generator, detailed map guide, regression tests and mapping CI checks.

These additions provide a common analysis interface. They do not implement a
universal dynamical evolution model for every radio-galaxy class; detailed
KA/KDA fitting remains FR II-specific. No observed FITS images or real speed
measurements were supplied for this update.

## 0.2.1 — 2026-10-26

Reliability fixes preserve the published model conventions and public commands.

- Reject duplicate JSON keys, nonstandard nonfinite literals, invalid UTF-8,
  oversized numeric inputs, and unrepresentable fit bounds with clear errors.
- Check runtime overrides and unsupported bootstrap requests before fitting.
  Validation enforces the five-detection requirement for a remnant fit.
- Write results through a flushed temporary file and atomic replacement, so a
  failed write preserves an existing result. Hash the exact input bytes.
- Find classic jet-power roots adjacent to zero-emission intervals and retain
  the better grid solution when local refinement finds a worse solution.
- Estimate covariance from the Jacobian's SVD instead of inverting normal
  equations; report unavailable covariance explicitly. Undefined correlations
  are JSON nulls.
- Recover from numerical failures in individual optimizer starts and Monte
  Carlo realizations, recording failures in diagnostics.
- Stabilize narrow-range electron normalization, very small fractional errors,
  and log-power residual calculation.
- Validate fixed inner power before fitting the outer component and compare
  restarted component lengths using each component's lobe convention.
- Add regression tests and CI for the test suite and installed wheel commands.

Scientific limitations and systematic uncertainties described in the README
still apply. Remnant and restarted Monte Carlo refits remain unsupported.
