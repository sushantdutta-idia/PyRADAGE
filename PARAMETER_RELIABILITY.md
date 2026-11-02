# Reliable source-parameter reporting

Version 0.7.0 adds checks that can reject inconsistent inputs and withhold
unsupported estimates. They do not guarantee the physical accuracy of a model.
The default field prescription is **B = 0.4 B_eq** (RMS B for free-streaming
JP–Tribble), with uncertainty in B_eq and the fraction fixed. An explicitly
requested fixed field can replace that input for a particular run. The WAT
rerun requested on 2026-11-02 conditions on **B = 2.24e-10 T = 2.24 μG**,
without inventing a field uncertainty or a replacement B_eq.

## Run and check

```bash
python pyradage.py analyze radio.json --output-dir results/source \
  --pyrats-budget PyRATS_adopted_field_budget.json
python pyradage.py verify-analysis results/source --check-inputs
python -m pytest -q
```

`verify-analysis` checks SHA-256 hashes of every recorded output. With
`--check-inputs`, it also checks the original absolute input paths, which must
still exist. The manifest is excluded from its own hash list. This detects
changes relative to the supplied manifest; it does not authenticate the
manifest or validate the astrophysics. Earlier manifests without hashes must
not be silently treated as verified. Inputs are checked after computation and
again immediately before publication. Failed checks leave the target analysis
directory untouched; overwrites retain the previous directory as a backup.

## Read the reporting gates

`analysis.json.parameter_quality` distinguishes supported conditional
estimates from numerical diagnostics. Transport mean/median and shared
injection/speed/deceleration estimates are withheld if the model fails its
existing adequacy heuristic, independent sampler replicates disagree, ageing
curvature is not established, or declared distance scales disagree. The raw
posterior summaries remain available as diagnostics. These gates are
descriptive checks, not calibrated detection probabilities or proof of
convergence. A shared field-speed degeneracy persists for uncertain-field
inference even when they pass. Holding a supplied B fixed conditions on that
assumption; it does not independently measure source turn-on time, remove
geometry/history uncertainties or establish a spectral break.

For resolved profiles, `AGE_CONSTRAINED` retains the v0.6.0 conditional
closed-profile/curvature/fit/boundary checks. New products are:

| Layer | Meaning |
|---|---|
| `AGE_ALOW`, `AGE_AHIGH` | Actual spectral refits at two injection indices |
| `FLAG_ALOW`, `FLAG_AHIGH` | Full age-quality flags for those refits |
| `AGE_SCENARIO_OK` | Nominal and both checked index profiles pass |

When `alpha_inj` is `transport_posterior_median`, index sensitivity defaults to
the shared transport posterior's 16th/84th percentiles. A fixed numeric index
defaults to sensitivity not assessed. Supply two explicit bracketing indices
to assess it, or `null` to disable:

```json
"resolved_analysis": {
  "models": ["jp_tribble", "ci"],
  "alpha_inj": "transport_posterior_median",
  "alpha_sensitivity": "transport_posterior_68"
}
```

The scenarios reuse the same data and hold the field fixed. They are not a
joint alpha/B posterior, marginalized confidence intervals, or checks of all
intermediate indices. This default index range may itself be prior/model
dependent when curvature is weak. See the injection-index sensitivity discussed
by [Harwood et al. (2013)](https://arxiv.org/abs/1308.4137).

`whole_detected_source_summary` withholds the source-wide spectral mean/median
unless every detected source pixel passes at the nominal index and both
checked scenarios. It always retains explicitly diagnostic best-fit map
statistics. This avoids reporting the mean of a selected well-fit subset as
the age of the entire source. Beam-correlated pixels and shared calibration
prohibit independent-pixel error averaging. Even a passing mean is a
conditional spectral-map statistic, not a global dynamical lifetime.

Every detected central pixel remains eligible: no circular core exclusion or
overlay is introduced. Poor or boundary core fits remain visible with flags.
Neither forced-zero core ages nor central masking repairs spectral mixing.

## Physical and preparation consistency

For paths derived from angular coordinates, record
`geometry.kpc_per_arcsec`. A distance-map `KPCAS` keyword, when present, must
agree with that value. The pipeline compares it with PyRATS's scale, using a
declared 1% consistency tolerance. It warns when no comparison is possible and
withholds supported transport summaries when the scales disagree. It never
silently rescales a supplied physical path map. Generate paths, source volume,
and luminosities using the same redshift/cosmology; see
[Astropy's proper angular scale](https://docs.astropy.org/en/stable/api/astropy.cosmology.FlatLambdaCDM.html).

An imported nominal adopted-field budget must include B in T and microgauss,
the nominal B_eq, electron/particle/magnetic/total energy densities, total
energy, and pressure. Default checks verify B=0.4 B_eq against the report's
deterministic field. In an explicit fixed-field run, the requested field is
checked instead and the actual ratio to the retained nominal B_eq is reported.
Neither B_eq nor a 0.4 ratio is manufactured. Both routes check
SI conversion, u_B=B²/(2 mu0), u_particles=(1+kappa)u_e,
u_total=u_particles+u_B, E=V*u_total, and relativistic p=u_total/3. Relative
accounting tolerance is 1e-5. These checks neither recompute synchrotron flux
normalization nor prove the assumed composition, volume, filling factor or
electron distribution. A power-law PyRATS budget is not an aged TJP/CI energy
fit. Deterministic budget and MC field median are explicitly different bases.
Fixed-field age intervals exclude field uncertainty because none was supplied;
they must not be read as covering all source uncertainties. If compatibility
AGE_B16/AGE_B84 layers are retained, identical layers express a fixed field,
not newly measured zero uncertainty. Upstream B_eq MC quantiles remain native
comparison results and are not propagated as uncertainty in the user-fixed B.

If PyRATS supplies requested/rejected MC counts, they must be consistent; fewer
than 80% successful draws are rejected because survivor quantiles may be biased.
The 80% threshold is a conservative operational gate, not a coverage theorem.
Absent counts remain explicitly unavailable in provenance.

## What still controls accuracy

The pipeline cannot recover missing observational information. Matched uv
coverage and flux recovery, astrometric alignment, restoring beam, per-band
noise, absolute/calibration covariance, variability, absorption and unresolved
population mixing must be assessed for the source. More points within one
beam are not independent measurements. Lower/higher-frequency detections that
bracket curvature are more useful than many closely spaced bands on a power
law. CI injection duration and TJP loss time are different clocks; the limits
of integrated CI fitting are discussed by
[Harwood (2017)](https://arxiv.org/abs/1612.04390).

The WAT flow model does not identify jet power, ambient density, or source
turn-on time. Active FR II dynamics have their separate model and sensitivity
route. External flow/expansion, inverse-Compton, environmental, or geometry
constraints can reduce the relevant degeneracies; example galaxies provide
explicit population priors, not measurements of this source.
