"""Generate a labelled synthetic active-FR-II analysis with an external field anchor."""
from pathlib import Path
import argparse
import json

from fr2dynage.constants import MYR_S
from fr2dynage.model import Geometry, ModelAssumptions, Observation, SpectrumPoint, KDAForwardModel


def generate(output):
    output = Path(output)
    files = [output/name for name in ("input.json", "truth.json", "sensitivity.json")]
    if any(path.exists() for path in files):
        raise FileExistsError("choose a new synthetic-example directory")
    assumptions = ModelAssumptions(gamma_min=10., integration_order=64, equipartition_field_ug=3.)
    geometry = Geometry(120., 4., 75.)
    frequencies = [100., 300., 600., 1500., 5000., 10000.]
    dummy = Observation.from_points("Synthetic active FR II field-anchor verification", .15, geometry,
                                   [SpectrumPoint(frequency, 1., .03) for frequency in frequencies])
    model = KDAForwardModel(dummy, assumptions)
    truth = {"age_myr": 10., "alpha_inj": .62, "jet_power_w_per_jet": 1e38,
             "equipartition_reference_field_ug": 3., "adopted_field_ug": 1.2,
             "origin": "synthetic", "interpretation": "known external field-reference constraint for software verification, not a measured or independently fitted equipartition field"}
    powers = model.spectrum([value*1e6 for value in frequencies], 1e38, 10*MYR_S, .62)
    data = {"source": {"name": dummy.name, "redshift": .15, "radio_class": "fr2"},
        "geometry": {"projected_size_kpc": 120., "axial_ratio": 4., "inclination_deg": 75., "mode": "two_sided"},
        "spectrum": {"frequency_frame": "rest", "power_convention": "per_sr", "core_subtracted": True, "hotspots_subtracted": True,
                     "points": [{"frequency_mhz": f, "power_w_hz": float(p), "fractional_error": .03} for f, p in zip(frequencies, powers)]},
        "model": {"gamma_min": 10., "integration_order": 64, "equipartition_field_ug": 3.},
        "fit": {"age_myr": [3., 30.], "alpha_inj": [.5, .8], "log10_jet_power_w": [37., 39.5],
                "age_grid_size": 9, "alpha_grid_size": 7, "power_scan_size": 9},
        "runtime": {"bootstrap_samples": 0, "seed": 42}}
    sensitivity = {"cases": [{"label": "Atmosphere beta 1.0", "model": {"beta": 1.0}},
                              {"label": "Higher electron cutoff", "model": {"gamma_min": 100.}}]}
    output.mkdir(parents=True, exist_ok=True)
    for path, document in zip(files, (data, truth, sensitivity)):
        path.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    return truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/dynage_demo"))
    args = parser.parse_args()
    generate(args.output)
    print(args.output.resolve())
