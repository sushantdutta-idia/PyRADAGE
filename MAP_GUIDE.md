# Spatial age and flow-time maps

The constrained map workflow combines registered radio FITS images with an independent
motion constraint to produce a conditional expansion-age or flow-travel-time
map. It also fits spectral-index and fit-quality maps. The source-integrated
KA/KDA fits remain available for FR II lobes.

These outputs represent different physical clocks:

| Output | Required motion assumption | Interpretation |
|---|---|---|
| Expansion age | Current radial expansion speed and an explicit law `R ∝ t^m` | `t = m R / v_current` |
| Flow travel time | Path length and an effective speed over that complete path | `t_flow = L / v_effective` |
| Spectral index | Detected intensities at three or more frequencies | Slope in `I_nu ∝ nu^(-alpha)`; no age is inferred |

An expansion-age map is conditional on the supplied expansion law, origin,
distance, and speed field. In a single self-similar expansion episode, a
consistent radial velocity field gives the same age throughout the source.
Spatial variation can instead identify separate components or a failure of
the adopted assumptions; it does not establish independent formation times
for individual pixels. A flow travel time is not automatically the total
source age or the time since the radio-emitting particles were accelerated.

Radio spectral ageing and dynamical ageing are also distinct measurements.
Published resolved analyses find discrepancies between their inferred clocks,
including effects associated with particle mixing. This pipeline does not
convert a spectral-index gradient into a radiative or dynamical age.
See [Harwood et al. 2017](https://arxiv.org/abs/1704.01576) and
[Mahatma et al. 2020](https://arxiv.org/abs/1912.01028).

## Supported radio-galaxy classes

Class labels describe the source; they do not supply an expansion law or a
velocity measurement. The same constrained image-analysis interface accepts
all classes below. This is not one universal physical model of radio-galaxy
evolution.

| `radio_class` | Class | Constrained map workflow | Integrated KA/KDA fit |
|---|---|---|---|
| `fr2` | FR II | Expansion or flow | Available within the existing model assumptions |
| `fr1` | FR I | Expansion or flow | Unavailable |
| `fr0` | FR 0 | Expansion or flow, if spatially resolved | Unavailable |
| `hybrid` | Hybrid morphology | Analyse physically distinct components separately | Only a separately identified FR II component |
| `wat`, `nat`, `head_tail` | Wide-angle, narrow-angle, and head-tail sources | Use a path-length map for a curved tail | Unavailable |
| `compact`, `css`, `gps`, `cso` | Compact, compact steep-spectrum, gigahertz-peaked, compact symmetric | Use independently measured component motion and a consistent distance convention | Unavailable |
| `diffuse` | Diffuse radio galaxy | Expansion or flow when independently constrained | Unavailable |
| `other`, `unknown` | Other or unclassified source | Expansion or flow when independently constrained | Unavailable |

Aliases such as `FR-I`, `FR II`, `WAT`, and `HYMORS` are accepted. Use
`activity_state` to record an active, remnant, restarted, or unknown source separately
from its morphological or spectral class. A class such as GPS does not by
itself determine the source age or the cause of its spectral turnover.

For restarted sources, prepare separate component masks and configurations
for the outer structure and inner double. Their clocks need not have the same
origin, speed, or expansion history. Component maps do not replace the
central-engine chronology calculation in `fit-restarted`.

## Install the image dependencies

Use Python 3.11 or newer. In the pipeline directory, with its virtual
environment activated:

```bash
python -m pip install -e '.[maps,test]'
```

The mapping extra supplies FITS/WCS handling and preview plotting. Existing
integrated-spectrum commands can still be installed without the mapping
dependencies.

## Prepare the images

Supply at least three images at distinct positive observing frequencies.
Each image must have:

- The same celestial pixel grid and shape, and a valid celestial FITS WCS.
- A common restoring beam, recorded in `BMAJ`, `BMIN`, and `BPA`; the FITS
  beam-axis units are degrees. An absent position angle is accepted only for
  a circular beam.
- A brightness unit convertible to Jy/beam, such as Jy/beam, mJy/beam, or
  uJy/beam.
- A positive RMS noise estimate, either a scalar in Jy/beam or an aligned RMS
  FITS map with its own brightness unit.

Two-dimensional images and cubes with only singleton leading axes are
accepted. Extract a single image plane from a genuinely multichannel or
multi-Stokes cube before analysis. Image alignment and beam matching are
validated rather than performed silently. If registration or restoring beams
differ, regrid/convolve the images during image preparation and then repeat
validation. Astropy uses zero-based `(x, y)` pixel coordinates; NumPy image
arrays are indexed `[y, x]`. See the
[Astropy WCS interface documentation](https://docs.astropy.org/en/stable/wcs/wcsapi.html).

A matching restoring beam alone does not prove comparable spatial-frequency
coverage or recovery of diffuse flux. Check imaging/uv coverage, flux-scale
consistency, component contamination, and variability between epochs before
interpreting spatial gradients. Pixels within one restoring beam are
correlated and must not be counted as independent age measurements.

## Configuration

Map configurations are strict JSON. File paths are resolved relative to the
configuration file. Replace every example image, coordinate, scale, and
motion constraint below with the corresponding measurement or explicitly
adopted assumption for your source.

An image entry may select its FITS extension using `hdu`, either a
nonnegative integer or an extension name. Auxiliary map fields such as
`speed_map`, `distance_map`, `rms_map`, and `selection.mask` accept a path
string or an object such as `{"path": "motion.fits", "hdu": "SPEED"}`.
Omitted `hdu` selects the primary image (HDU 0).

```json
{
  "source": {
    "name": "Example radio galaxy",
    "radio_class": "fr2",
    "activity_state": "active"
  },
  "images": [
    {
      "path": "image_150MHz.fits",
      "frequency_mhz": 150.0,
      "rms_jy_per_beam": 0.0001,
      "calibration_fraction": 0.05
    },
    {
      "path": "image_325MHz.fits",
      "frequency_mhz": 325.0,
      "rms_jy_per_beam": 0.00008,
      "calibration_fraction": 0.05
    },
    {
      "path": "image_1400MHz.fits",
      "frequency_mhz": 1400.0,
      "rms_jy_per_beam": 0.00003,
      "calibration_fraction": 0.05
    }
  ],
  "selection": {
    "snr_min": 5.0,
    "min_bands": 3,
    "require_all_bands": true
  },
  "geometry": {
    "distance_kind": "radius",
    "distance_frame": "projected",
    "origin_icrs_deg": [180.0, 30.0],
    "kpc_per_arcsec": 1.0,
    "distance_error_kpc": 0.2
  },
  "kinematics": {
    "mode": "expansion",
    "speed_role": "current_expansion",
    "speed_km_s": 3000.0,
    "speed_error_km_s": 300.0,
    "speed_frame": "intrinsic",
    "inclination_deg": 60.0,
    "inclination_error_deg": 5.0,
    "expansion_index": 0.6,
    "expansion_index_error": 0.05,
    "constraint_description": "EXAMPLE ONLY: replace with the provenance and assumptions of an independent expansion-speed constraint."
  }
}
```

`source.redshift` is optional when the physical angular scale is supplied
explicitly. If `geometry.kpc_per_arcsec` is omitted, a positive source
redshift supplies the angular-diameter scale under the recorded Planck18
cosmology. An explicit physical distance map does not need an angular scale.
Use one frequency frame consistently across the images; the spectral-index
fit is invariant to a common factor of `1+z` in all frequency labels.

### Image and pixel selection settings

Each image requires either `rms_jy_per_beam` or `rms_map`, not both.
`calibration_fraction` is an optional nonnegative fractional flux-scale error
for that band. A mask under `selection.mask` is an aligned FITS image with
1 for selected pixels and 0 or NaN for excluded pixels.

By default, every image must have a usable positive detection above five
times its local thermal RMS at a pixel. To allow missing bands, set
`require_all_bands` to `false`; at least `min_bands` usable detections must
still remain, and `min_bands` cannot be below three. `snr_min` cannot be below
three. Upper limits are not treated as detections.

The spectral fit uses

```text
I_nu = I_ref (nu / nu_ref)^(-alpha)
sigma_lnI = sqrt(log(1 + (RMS/I)^2 + calibration_fraction^2)).
```

Thermal noise and calibration contributions are treated as independent
between bands. The log-intensity treatment is a high-SNR approximation for
positive detections. Flux calibration errors can correlate many pixels in a
band; the pixel uncertainty maps do not remove that correlation. A poor
power-law fit is recorded as a spectral diagnostic rather than being
interpreted as an ageing measurement.

### Distance geometry

Use `distance_kind: "radius"` for expansion and `"path_length"` for flow.
Choose one of:

- `origin_pixel: [x, y]`, using zero-based celestial image coordinates, plus
  `kpc_per_arcsec` or the source redshift.
- `origin_icrs_deg: [ra, dec]`, using ICRS degrees, plus a scale or redshift.
- `distance_map`, an aligned FITS map with `BUNIT` convertible to kpc.

Origin-based distances are projected straight-line separations on the sky.
They are suitable for a radial expansion clock, or a genuinely straight
flow path. Origin-based flow requires `path_is_straight: true` as an explicit
geometry assertion. Curved WAT, NAT, and head-tail paths require a distance
map measured along the tail/spine rather than distance directly from the
core. The pipeline does not infer a three-dimensional tail trajectory.

Set `distance_frame` to `"projected"` or `"intrinsic"` to match the supplied
distance map. For a curved path with varying line-of-sight angle, prepare an
intrinsic path-length map from your geometry instead of treating one global
inclination as an exact deprojection.

Optional independent distance errors use either `distance_error_kpc` or
`distance_error_map` in kpc. The distance uncertainty map describes uncertainty
in distance itself; uncertainty in the origin, astrometry, cosmology, or
spine tracing is included only to the extent that it was propagated into
that supplied error map.

### Motion constraints and frame conventions

Specify exactly one of `speed_km_s` or `speed_map`, with a FITS speed map in
units convertible to km/s. Optional errors use exactly one of
`speed_error_km_s` or `speed_error_map` in matching units. Record the
measurement, literature source, or assumption in `constraint_description`;
the images do not measure these speeds on their own.

For expansion, set `mode: "expansion"`, `speed_role: "current_expansion"`, and
an explicit positive `expansion_index`. The speed must be the current radial
expansion speed associated with the measured radius. The exponent is not
automatically inferred from the radio-galaxy class. For example, `m=1` means
constant-speed expansion, whereas a different `m` assumes a different
expansion history.

For flow, set `mode: "flow"` and `speed_role: "effective_flow"`; omit
`expansion_index` and its error. The speed is the effective speed over the
complete path to that pixel. If the local flow speed varies along a path,

```text
t_flow = integral ds / v(s)
v_effective = L / t_flow.
```

Prepare the corresponding effective-speed map before analysis. Dividing the
whole path length by a locally measured terminal speed generally does not
perform that integral. The pipeline does not reconstruct it from a local
velocity map without a supplied trajectory and velocity history.

`speed_frame` uses `"projected"` or `"intrinsic"`. Inclination is measured
from the line of sight, with 90 degrees in the sky plane:

| Distance frame | Speed frame | Clock from the supplied `R`/`L` and `v` |
|---|---|---|
| Projected | Intrinsic | Divide `m R / v` or `L / v` by `sin(theta)` |
| Intrinsic | Projected | Multiply the ratio by `sin(theta)` |
| Same convention | Same convention | The inclination factor cancels |

A finite inclination in `(0, 90]` is required when the frames differ. Speeds
are physical host-frame speeds, not apparent superluminal proper motions.
They cannot exceed the speed of light, including after deprojecting a
projected speed when an inclination is supplied. Relativistic light-travel
and apparent-motion corrections must be made before supplying such a speed.

For compact sources, supply a one-sided core-to-hotspot radius and that
head's physical speed. Convert measured double-source separation and
separation rate to a consistent one-sided pair before analysis. For a
symmetric double, divide both by two; the resulting age ratio is unchanged.
For asymmetric heads, infer the individual distances/speeds rather than
assuming equal halves. A separation rate can exceed `c` even when each
head moves below `c`; it is not an accepted component-speed input here.
The workflow inserts no implicit factor of two. An origin-based radius is
the distance from that origin to each pixel.

### Uncertainty scope

When supplied, independent first-order distance, speed, inclination, and
expansion-index errors are propagated. For expansion with projected radius
and intrinsic speed, the nonzero-radius formula is

```text
(sigma_t/t)^2 = (sigma_R/R)^2 + (sigma_v/v)^2
              + (cot(theta) sigma_theta_rad)^2 + (sigma_m/m)^2.
```

Flow omits the expansion-index term. Same-frame geometry omits the angle
term because it cancels from the clock. At zero distance, absolute
derivatives are used so a nonzero distance error remains meaningful.
No error map is manufactured when no uncertainty is supplied.

These are conditional uncertainties for the supplied error components.
Missing errors are treated as fixed assumptions; geometry/speed covariance,
time-variable motion, unmodelled systematics, and sub-beam correlations are
not inferred. Large or non-Gaussian uncertainties require explicit
realizations of the geometry and motion model beyond first-order errors.

## Validate and run

Validate a configuration before producing results:

```bash
python pyradage.py validate-map /absolute/path/to/source-map.json
```

Run the spectral diagnostics and constrained clock calculation:

```bash
python pyradage.py map-age /absolute/path/to/source-map.json \
  --output results/source-age.fits \
  --preview results/source-age.png
```

`--preview` is optional. The FITS file is the scientific map product; the PNG
is a visual preview with matching PDF/SVG exports. Use `python pyradage.py classes` to list class support and
`python pyradage.py map-age --help` for command options. Output paths are
relative to the current working directory, while input image paths are
relative to the configuration file.

Inspect the output validity mask, quality flags, units, clock type, and
recorded assumptions together with the age values. Invalid or excluded clock
pixels have NaN age/error values rather than artificial zero ages. Spectral
diagnostics can remain valid where a speed constraint is unusable, so use
the joint `VALID` mask when comparing age and spectral maps. A physical
zero-distance pixel may have a valid age of zero. An entirely invalid map
fails with a clear error. Existing outputs are preserved unless `--overwrite`
is explicitly supplied.

Map previews use the shared publication style from version 0.5.2, including
larger WCS coordinate labels, tick labels, titles and colour-bar text.
The defaults use 18 pt base text, 20 pt axis labels, 17 pt tick labels and
16 pt legends. For `--preview results/source-age.png`, the pipeline writes
a 600 dpi PNG and matching `results/source-age.pdf` and
`results/source-age.svg` files. Existing companion files receive the same
collision protection as the requested preview.

Inspect the export at its final manuscript column width, since reducing a
figure reduces its apparent type size. PDF/SVG preserve vector labels and
line work, while radio-image panels remain sampled maps. Higher PNG dpi
does not add angular resolution, change the supplied speed constraint, or
alter the physical clock and its uncertainty.

The output contains these HDUs:

| HDU | Meaning |
|---|---|
| Primary `DYNAGE` or `FLOWTIME` | Conditional expansion age or flow travel time in Myr; `AGEKIND` records the clock type |
| `AGE_ERR` | Supplied-error propagation in Myr; omitted when no kinematic error is supplied |
| `ALPHA`, `ALPHA_ERR` | Power-law spectral index and conditional uncertainty |
| `CHI2`, `REDCHI2` | Spectral power-law chi-square and reduced chi-square |
| `N_BANDS` | Number of usable frequency detections |
| `VALID` | Joint spectral-selection and kinematic validity mask |
| `SPEC_FLAGS`, `KIN_FLAGS` | Spectral and kinematic quality bit masks |
| `DIST_KPC`, `SPEED_KMS` | Supplied/derived distances and supplied speeds in the configured frames |
| `PROVENANCE` | UTF-8 JSON byte vector containing source/model assumptions, input paths and SHA-256 hashes, software versions, flag definitions, and warnings |

For example, inspect the output and its provenance using Astropy:

```python
import json
from astropy.io import fits

with fits.open("results/source-age.fits") as hdus:
    hdus.info()
    age = hdus[0].data
    valid = hdus["VALID"].data.astype(bool)
    metadata = json.loads(
        hdus["PROVENANCE"].data["JSON_UTF8"][0].tobytes().decode("utf-8")
    )
    print(metadata["clock"])
    print(metadata["spectral_flag_meanings"])
    print(metadata["kinematic_flag_meanings"])
```

## Synthetic demonstration and real observations

The bundled map demonstration uses generated images and known kinematic
constraints so the software can be tested without observational data. It
does not establish that those constraints apply to a real radio galaxy.
Analyse the already bundled examples from the pipeline directory. The
optional generator command uses a new directory so it preserves bundled data:

```bash
# Optional: regenerate the fixtures in a new directory.
python examples/generate_map_demo.py --output results/new-map-demo

python pyradage.py validate-map examples/maps_demo/expansion.json
python pyradage.py map-age examples/maps_demo/expansion.json \
  --output results/demo-expansion.fits \
  --preview results/demo-expansion.png

python pyradage.py validate-map examples/maps_demo/flow.json
python pyradage.py map-age examples/maps_demo/flow.json \
  --output results/demo-flow.fits \
  --preview results/demo-flow.png
```

The first example demonstrates an FR II-labelled expansion constraint. The
second demonstrates a WAT-labelled flow clock using a supplied cumulative
path-length map along a curved spine. The generator writes noiseless
analytic spectra with illustrative RMS/calibration uncertainties; it is not
a hydrodynamic simulation. `examples/maps_demo/truth.json` records reference
pixels, exact spectral indices, distances, ages, and propagated errors.
Pixel references there are `[x, y]`, so inspect a corresponding FITS array
using `data[y, x]`.

To regenerate existing demonstration inputs, add `--overwrite` to the
generator. To replace existing results, add `--overwrite` to `map-age`.
After checking the demonstration, replace its image and motion-constraint
entries with your measured inputs and the relevant geometry.

No observational FITS images or measured speed/flow constraints were supplied
with the request that introduced this workflow. A map of your actual source
therefore still requires those files, the origin or path geometry, and the
independent motion constraint described above.
