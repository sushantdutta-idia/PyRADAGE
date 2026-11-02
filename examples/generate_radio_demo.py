#!/usr/bin/env python3
"""Generate a labelled synthetic radio-only JP/advection worked example.

The images are Gaussian-convolved analytic emissivity fields, not hydrodynamic
simulations or observed reference sources. The example deliberately supplies
informative, declared B and injection-index priors to make software recovery
checks tractable. It does not show that real radio images measure B or age
independently. Use --power-law to generate a no-curvature diagnostic case.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


FREQUENCIES_MHZ = (100.0, 150.0, 325.0, 610.0, 1000.0, 1400.0, 3000.0, 6000.0)
REDSHIFT = 0.1469
SIZE = 55
PIXEL_ARCSEC = 2.0
BEAM_FWHM_PIXELS = 5.0
RMS_JY_BEAM = 1.0e-5


def generate(output: Path, *, overwrite: bool = False, noise: bool = False,
             power_law: bool = False, particles: int = 512) -> dict:
    """Write input FITS/configuration and true model fields; return truth."""

    from astropy.cosmology import Planck18
    from astropy.io import fits
    from astropy.wcs import WCS
    from fr2dynage.radio_spectra import jp_log_emissivity

    output = Path(output)
    names = [f"radio-{int(nu)}.fits" for nu in FREQUENCIES_MHZ]
    names += ["distance.fits", "branches.fits", "mask.fits", "truth-age.fits",
              "radio.json", "truth.json", "reference-synthetic.json"]
    collisions = [name for name in names if (output / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError("refusing to overwrite synthetic demo files: "
                              + ", ".join(collisions))
    output.mkdir(parents=True, exist_ok=True)
    scale = float(Planck18.kpc_proper_per_arcmin(REDSHIFT).value / 60.0)
    kpc_pixel = scale * PIXEL_ARCSEC

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [53.044708333333325, -27.443166666666666]
    wcs.wcs.crpix = [8.0, 28.0]
    wcs.wcs.cdelt = [-PIXEL_ARCSEC / 3600.0, PIXEL_ARCSEC / 3600.0]
    header = wcs.to_header()
    header["BMAJ"] = BEAM_FWHM_PIXELS * PIXEL_ARCSEC / 3600.0
    header["BMIN"] = header["BMAJ"]
    header["BPA"] = 0.0
    header["OBJECT"] = "Synthetic steady-JP advection; not an observed source"

    def image(name, data, unit):
        current = header.copy()
        current["BUNIT"] = unit
        fits.PrimaryHDU(np.asarray(data), header=current).writeto(
            output / name, overwrite=overwrite, checksum=True)

    y, x = np.indices((SIZE, SIZE), dtype=float)
    projected_distance = np.maximum(x - 7.0, 0.0) * kpc_pixel
    # The inference defines L by the selected map support, ending at x=46.
    length = (46.0 - 7.0) * kpc_pixel
    deceleration = 0.4
    field, alpha = 3.0, 0.65
    speeds = {1: 4000.0, 2: 6500.0}
    branch_map = np.zeros((SIZE, SIZE), dtype=np.int16)
    truth_age = np.full((SIZE, SIZE), np.nan)
    intrinsic = np.zeros((len(FREQUENCIES_MHZ), SIZE, SIZE))
    for branch, centre_y in ((1, 15.0), (2, 39.0)):
        strip = (x >= 7) & (x <= 48) & (np.abs(y - centre_y) <= 2.0)
        branch_map[strip] = branch
        age = 977.7922216807891 / speeds[branch] * (
            projected_distance + deceleration * projected_distance**2 / (2 * length))
        truth_age[strip] = age[strip]
        amplitude = (0.05 * np.exp(-0.5 * ((y - centre_y) / 1.5)**2)
                     * (0.9 + 0.1 * np.cos((x - 7.0) / 12.0)))
        emitting = (x >= 5.0) & (x <= 50.0) & (np.abs(y - centre_y) < 7)
        if power_law:
            spectrum = np.broadcast_to(
                (np.asarray(FREQUENCIES_MHZ) / 1000.0)**-alpha,
                (int(np.sum(emitting)), len(FREQUENCIES_MHZ)))
        else:
            spectrum = np.exp(jp_log_emissivity(
                FREQUENCIES_MHZ, age[emitting], field, alpha, REDSHIFT,
                reference_frequency_mhz=1000.0))
        intrinsic[:, emitting] += (amplitude[emitting, None] * spectrum).T

    beam_sigma = BEAM_FWHM_PIXELS / math.sqrt(8 * math.log(2))
    rng = np.random.default_rng(20261027)
    for index, frequency in enumerate(FREQUENCIES_MHZ):
        science = gaussian_filter(intrinsic[index], beam_sigma, mode="constant")
        if noise:
            # Smoothed independent noise has Gaussian beam autocorrelation.
            realization = gaussian_filter(rng.normal(size=science.shape),
                                          beam_sigma, mode="constant")
            realization *= RMS_JY_BEAM / np.std(realization)
            science += realization
        image(f"radio-{int(frequency)}.fits", science, "Jy/beam")

    mask = (branch_map > 0) & (x >= 13) & (x <= 46)
    distance = np.where(branch_map > 0, projected_distance, np.nan)
    image("distance.fits", distance, "kpc")
    image("branches.fits", branch_map, "1")
    image("mask.fits", mask.astype(np.uint8), "1")
    image("truth-age.fits", truth_age, "Myr")
    config = {
        "schema_version": "fr2dynage-radio-1",
        "source": {"name": "Synthetic power-law WAT" if power_law else
                   "Synthetic steady-JP WAT", "radio_class": "wat",
                   "redshift": REDSHIFT, "activity_state": "active"},
        "images": [{"path": f"radio-{int(nu)}.fits", "frequency_mhz": nu,
                    "rms_jy_per_beam": RMS_JY_BEAM, "calibration_fraction": 0.01}
                   for nu in FREQUENCIES_MHZ],
        "selection": {"snr_min": 5.0, "min_bands": 4, "mask": "mask.fits"},
        "geometry": {"distance_map": "distance.fits", "branch_map": "branches.fits",
                     "description": "Synthetic projected straight branch paths from x=7; inner beam excluded."},
        "model": {"kind": "steady_jp_advection", "deceleration": deceleration,
                  "log_emissivity_scatter": 0.05},
        "prior_description": "Synthetic verification only: informative B/alpha and broad speed priors are declared assumptions, not measured by these images.",
        "priors": {
            "flow_speed_km_s": {"distribution": "lognormal", "log_mean": math.log(5000.0),
                                "log_sigma": 0.5},
            "magnetic_field_ug": {"distribution": "lognormal", "log_mean": math.log(field),
                                  "log_sigma": 0.025},
            "alpha_inj": {"distribution": "normal", "mean": alpha, "sigma": 0.012},
        },
        "inference": {"particles": particles, "seed": 42, "mcmc_steps": 4,
                      "ess_target": 0.7, "max_stages": 80},
    }
    truth = {
        "schema_version": "fr2dynage-radio-demo-truth-1",
        "origin": "synthetic",
        "description": "Analytic JP/advection emissivity convolved with a Gaussian restoring beam; no hydrodynamic or observed-source truth is claimed.",
        "power_law_case": power_law, "noise_added": noise,
        "shape_yx": [SIZE, SIZE], "redshift": REDSHIFT,
        "frequencies_mhz": list(FREQUENCIES_MHZ),
        "magnetic_field_ug": field, "alpha_inj": alpha,
        "flow_speed_km_s_by_branch": {str(k): v for k, v in speeds.items()},
        "deceleration": deceleration,
        "kpc_per_pixel": kpc_pixel,
        "global_source_age_myr": None,
        "global_source_age_status": "not_identified_by_transport_model",
        "truth_age_is_intrinsic_unconvolved_transport_field": True,
        "configuration": "radio.json",
        "reference_catalogue": "reference-synthetic.json",
    }
    catalogue = {"schema_version": "fr2dynage-reference-1",
                 "description": "Synthetic software-calibration demonstration only; none of these names or constraints denotes an observed galaxy.",
                 "sources": []}
    quantity_metadata = {
        "flow_speed_km_s": ("km/s", "projected_plasma_transport_speed"),
        "magnetic_field_ug": ("uG", "emitting_plasma_magnetic_field"),
        "alpha_inj": ("1", "injection_spectral_index"),
    }
    for index, (speed, field_value, alpha_value) in enumerate(
            ((3500., 2.7, .62), (5000., 3.0, .65), (6500., 3.4, .68)), 1):
        constraints = {}
        for parameter, value, error in (
                ("flow_speed_km_s", speed, 500.),
                ("magnetic_field_ug", field_value, .2),
                ("alpha_inj", alpha_value, .025)):
            unit, quantity = quantity_metadata[parameter]
            constraints[parameter] = {"value": value, "error": error,
                                      "unit": unit, "quantity": quantity,
                                      "method": "Declared synthetic model input; not an observational measurement",
                                      "independent": True}
        catalogue["sources"].append({"name": f"Synthetic reference {index}",
                                     "radio_class": "wat", "origin": "synthetic",
                                     "reference": "examples/generate_radio_demo.py synthetic parameter demonstration",
                                     "constraints": constraints})
    for name, document in (("radio.json", config), ("truth.json", truth),
                           ("reference-synthetic.json", catalogue)):
        (output / name).write_text(json.dumps(document, indent=2, allow_nan=False)
                                  + "\n", encoding="utf-8")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "radio_demo")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--noise", action="store_true", help="add seeded beam-correlated thermal noise")
    parser.add_argument("--power-law", action="store_true", help="generate spectra with no age curvature")
    parser.add_argument("--particles", type=int, default=512)
    args = parser.parse_args()
    try:
        generate(args.output, overwrite=args.overwrite, noise=args.noise,
                 power_law=args.power_law, particles=args.particles)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Synthetic radio-inference inputs written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
