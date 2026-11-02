#!/usr/bin/env python3
"""Generate small, noiseless radio-image fixtures with known map truths.

Requires NumPy and Astropy. The configured thermal RMS and calibration errors
are illustrative supplied uncertainties; random noise is not added, so the
spectral-index gradients can be checked directly. These are conditional
kinematic examples, not hydrodynamic simulations of whole radio sources.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


FREQUENCIES_MHZ = (150.0, 325.0, 610.0, 1400.0)
KPC_M = 3.085677581491367e19
MYR_S = 1e6 * 365.25 * 86400.0
SIZE = 49
RMS_JY_PER_BEAM = 2e-5


def _age_and_error(distance, *, speed, mode, index=None):
    m = index if mode == "expansion" else 1.0
    theta = math.radians(70.0)
    factor = KPC_M / (1000.0 * MYR_S)
    age = factor * m * distance / (speed * math.sin(theta))
    variance = (factor * m * 0.2 / (speed * math.sin(theta))) ** 2
    variance += (age * 0.1) ** 2
    variance += (age / math.tan(theta) * math.radians(3.0)) ** 2
    if mode == "expansion":
        variance += (age * 0.03 / m) ** 2
    return age, math.sqrt(variance)


def generate(output: Path, *, overwrite: bool = False) -> None:
    output = Path(output)
    names = [f"{prefix}-{int(frequency)}.fits" for prefix in ("exp", "flow")
             for frequency in FREQUENCIES_MHZ]
    names += ["exp-mask.fits", "flow-mask.fits", "exp-alpha-truth.fits",
              "flow-alpha-truth.fits", "path-length.fits", "expansion.json",
              "flow.json", "truth.json"]
    collisions = [name for name in names if (output / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"refusing to overwrite demo files: {', '.join(collisions)}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.crpix = [25.0, 25.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    header = wcs.to_header()
    header["BMAJ"] = 6.0 / 3600.0
    header["BMIN"] = 4.0 / 3600.0
    header["BPA"] = 20.0
    header["OBJECT"] = "FR2DYNAGE synthetic mapping demonstration"

    def image(name, data, unit=None):
        image_header = header.copy()
        if unit is not None:
            image_header["BUNIT"] = unit
        fits.PrimaryHDU(np.asarray(data), header=image_header).writeto(output / name, overwrite=overwrite)

    y, x = np.indices((SIZE, SIZE), dtype=float)
    radius = np.hypot(x - 24.0, y - 24.0)
    exp_alpha = 0.55 + 0.35 * np.minimum(radius / 24.0, 1.0)
    exp_amplitude = (0.003 * np.exp(-0.5 * (((x - 12.0) / 5.0)**2 + ((y - 24.0) / 4.0)**2))
                     + 0.003 * np.exp(-0.5 * (((x - 36.0) / 5.0)**2 + ((y - 24.0) / 4.0)**2)))
    exp_mask = exp_amplitude > 5e-4

    # The tail is a curved projected spine. The path map is cumulative arc
    # length along that spine, not radial separation from the core.
    t = np.linspace(0.0, 1.0, 601)
    spine_x = 8.0 + 27.0 * t
    spine_y = 10.0 + 22.0 * t + 14.0 * np.sin(math.pi * t)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(spine_x), np.diff(spine_y)))))
    separation2 = ((x[:, :, None] - spine_x)**2 + (y[:, :, None] - spine_y)**2)
    nearest = np.argmin(separation2, axis=2)
    distance_from_spine = np.sqrt(np.min(separation2, axis=2))
    flow_mask = distance_from_spine <= 2.25
    path_length = np.where(flow_mask, arc[nearest], np.nan)  # one kpc per arcsec/pixel
    flow_alpha = 0.5 + 0.7 * t[nearest]
    flow_amplitude = 0.003 * np.exp(-0.5 * (distance_from_spine / 1.7)**2) * (1.0 - 0.5 * t[nearest])

    reference_frequency = float(np.exp(np.mean(np.log(FREQUENCIES_MHZ))))
    for frequency in FREQUENCIES_MHZ:
        image(f"exp-{int(frequency)}.fits", exp_amplitude * (frequency / reference_frequency)**-exp_alpha, "Jy/beam")
        image(f"flow-{int(frequency)}.fits", flow_amplitude * (frequency / reference_frequency)**-flow_alpha, "Jy/beam")
    image("exp-mask.fits", exp_mask.astype(np.uint8))
    image("flow-mask.fits", flow_mask.astype(np.uint8))
    image("exp-alpha-truth.fits", np.where(exp_mask, exp_alpha, np.nan), "1")
    image("flow-alpha-truth.fits", np.where(flow_mask, flow_alpha, np.nan), "1")
    image("path-length.fits", path_length, "kpc")

    def config(prefix, *, name, radio_class, geometry, kinematics):
        return {
            "schema_version": "fr2dynage-map-1",
            "source": {"name": name, "radio_class": radio_class, "activity_state": "active"},
            "images": [{"path": f"{prefix}-{int(frequency)}.fits", "frequency_mhz": frequency,
                        "rms_jy_per_beam": RMS_JY_PER_BEAM, "calibration_fraction": 0.02}
                       for frequency in FREQUENCIES_MHZ],
            "selection": {"snr_min": 5, "min_bands": 3, "require_all_bands": True,
                          "mask": f"{prefix}-mask.fits"},
            "geometry": geometry,
            "kinematics": kinematics,
        }

    expansion_index = 3.0 / (5.0 - 1.5)
    expansion = config("exp", name="Synthetic FR II expansion constraint", radio_class="fr2",
        geometry={"distance_kind": "radius", "distance_frame": "projected",
                  "origin_pixel": [24.0, 24.0], "kpc_per_arcsec": 1.0, "distance_error_kpc": 0.2},
        kinematics={"mode": "expansion", "speed_role": "current_expansion", "speed_km_s": 3000.0,
                    "speed_error_km_s": 300.0, "speed_frame": "intrinsic", "inclination_deg": 70.0,
                    "inclination_error_deg": 3.0, "expansion_index": expansion_index,
                    "expansion_index_error": 0.03,
                    "constraint_description": "Synthetic supplied current radial expansion speed; R proportional to t**m"})
    flow = config("flow", name="Synthetic WAT curved-tail flow constraint", radio_class="wat",
        geometry={"distance_kind": "path_length", "distance_frame": "projected",
                  "distance_map": "path-length.fits", "distance_error_kpc": 0.2},
        kinematics={"mode": "flow", "speed_role": "effective_flow", "speed_km_s": 1000.0,
                    "speed_error_km_s": 100.0, "speed_frame": "intrinsic", "inclination_deg": 70.0,
                    "inclination_error_deg": 3.0,
                    "constraint_description": "Synthetic constant effective flow speed along the supplied curved projected spine"})

    exp_pixel = (36, 24)
    flow_pixel = (35, 32)
    origin_world = wcs.pixel_to_world(24.0, 24.0)
    exp_distance = float(origin_world.separation(wcs.pixel_to_world(*exp_pixel)).arcsec)
    exp_age, exp_error = _age_and_error(exp_distance, speed=3000.0, mode="expansion", index=expansion_index)
    flow_distance = float(path_length[flow_pixel[1], flow_pixel[0]])
    flow_age, flow_error = _age_and_error(flow_distance, speed=1000.0, mode="flow")
    truth = {
        "schema_version": "fr2dynage-map-demo-truth-1",
        "description": "Noiseless analytic images with supplied illustrative RMS/calibration uncertainties. No physical radio-source simulation is claimed.",
        "shape_yx": [SIZE, SIZE], "frequencies_mhz": list(FREQUENCIES_MHZ),
        "reference_frequency_mhz": reference_frequency,
        "rms_jy_per_beam": RMS_JY_PER_BEAM, "kpc_per_arcsec": 1.0,
        "expansion": {"configuration": "expansion.json", "pixel_xy": list(exp_pixel),
                      "alpha_truth_fits": "exp-alpha-truth.fits", "expected_alpha": float(exp_alpha[exp_pixel[1], exp_pixel[0]]),
                      "expected_distance_kpc": exp_distance, "expected_age_myr": exp_age,
                      "expected_age_error_myr": exp_error, "expansion_index": expansion_index,
                      "speed_km_s": 3000.0, "inclination_deg": 70.0,
                      "clock": "conditional expansion age, m R/v_current"},
        "flow": {"configuration": "flow.json", "pixel_xy": list(flow_pixel),
                 "alpha_truth_fits": "flow-alpha-truth.fits", "expected_alpha": float(flow_alpha[flow_pixel[1], flow_pixel[0]]),
                 "expected_distance_kpc": flow_distance, "expected_age_myr": flow_age,
                 "expected_age_error_myr": flow_error, "speed_km_s": 1000.0, "inclination_deg": 70.0,
                 "clock": "travel time along a supplied curved path; not total source age"},
    }
    for filename, document in (("expansion.json", expansion), ("flow.json", flow), ("truth.json", truth)):
        (output / filename).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "maps_demo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        generate(args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated expansion and curved-tail examples in {args.output.resolve()}")


if __name__ == "__main__":
    main()
