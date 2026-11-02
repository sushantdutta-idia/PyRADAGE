"""Auditable JSON, tables and figures for source-level DYNAGE analyses.

This module renders supplied results; it never fits, changes a magnetic-field
prescription or turns a transport clock into a global source age. New output
directories publish atomically; complete files publish atomically into an
existing staging directory without modifying other products already there.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


class AnalysisProductError(ValueError):
    """An analysis document or output destination cannot be used safely."""


_MANIFEST = "products_manifest.json"


def _encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported analysis value: {type(value).__name__}")


def _finite(value, default=np.nan):
    if isinstance(value, (bool, np.bool_)):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _values(rows, *keys):
    return np.asarray([next((_finite(row[key]) for key in keys if key in row), np.nan)
                       for row in rows], dtype=float)


def _rows(value):
    return value if isinstance(value, list) and all(isinstance(row, dict) for row in value) else []


def _table(rows, default_fields=()):
    fields = list(default_fields)
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else "" if value is None else value)
                         for key, value in row.items()})
    return stream.getvalue()


def _display_value(value):
    """Keep full values in JSON/CSV and readable precision in prose tables."""
    if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return f'{value:.6g}'
    return str(value).replace('\n', ' ').replace('|', '/')


def _source_property_rows(metadata):
    """Report retained inputs/geometry without turning them into fitted data."""
    rows = []
    source = metadata.get('source', {})
    if source.get('redshift') is not None:
        rows.append({'quantity':'redshift', 'value':source['redshift'], 'unit':'1', 'scope':'supplied source input'})
    properties = (metadata.get('pyrats_properties') or {}).get('properties', {})
    geometry = properties.get('geometry', {})
    units = {'volume_m3':'m^3', 'volume_kpc3':'kpc^3', 'largest_linear_size_kpc':'kpc',
        'minor_axis_kpc':'kpc', 'projected_area_kpc2':'kpc^2', 'axis_ratio':'1',
        'beams_per_source':'beam areas', 'kpc_per_arcsec':'kpc/arcsec',
        'volume_fractional_error':'1', 'filling_factor':'1'}
    for key, unit in units.items():
        if key in geometry:
            rows.append({'quantity':key, 'value':geometry[key], 'unit':unit,
                         'scope':'retained conditional PyRATS geometry/input; not refitted by the spectral clock'})
    for key in ['geometry_model', 'volume_estimator', 'assumption_note']:
        if key in geometry:
            rows.append({'quantity':key, 'value':geometry[key], 'unit':'', 'scope':'geometry assumption/provenance'})
    return rows


def _save(figure, target):
    from .plotting import save_publication_figure
    save_publication_figure(figure, target, formats=("png", "pdf", "svg"), clear=True)


def _figure(*, figsize=(12, 10)):
    from .plotting import publication_figure
    return publication_figure(figsize=figsize, layout="constrained")


def _empty(axis, message="Unavailable in this analysis"):
    axis.text(.5, .5, message, transform=axis.transAxes, ha="center", va="center", wrap=True)
    axis.set_xticks([]); axis.set_yticks([])


def _frequency_ticks(axis):
    """Keep enlarged logarithmic frequency labels readable in small panels."""
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
    axis.xaxis.set_major_locator(LogLocator(base=10., subs=(1., 2., 5.)))
    axis.xaxis.set_major_formatter(LogFormatterSciNotation(
        base=10., labelOnlyBase=False, minor_thresholds=(math.inf, math.inf)))
    axis.xaxis.set_minor_formatter(NullFormatter())


def _spectrum(primary, methods, observed, target):
    rows = _rows(primary.get("spectrum")) or observed
    frequency = _values(rows, "frequency_mhz_rest", "frequency_mhz")
    power = _values(rows, "observed_power_w_hz_sr", "power_w_hz_sr")
    good = np.isfinite(frequency) & (frequency > 0) & np.isfinite(power) & (power > 0)
    figure = _figure(figsize=(11, 10))
    upper = figure.add_subplot(2, 1, 1)
    lower = figure.add_subplot(2, 1, 2, sharex=upper)
    if np.any(good):
        error = _values(rows, "fractional_error") * power
        upper.errorbar(frequency[good], power[good], yerr=np.where(np.isfinite(error[good]), error[good], 0.),
                       fmt="o", color="black", label="Observed lobe/source power", capsize=2)
        any_model = False
        for name, result in methods.items():
            if result.get("success") is False:
                continue
            model_rows = _rows(result.get("spectrum"))
            x = _values(model_rows, "frequency_mhz_rest", "frequency_mhz")
            y = _values(model_rows, "predicted_power_w_hz_sr")
            residual = _values(model_rows, "log10_residual_model_over_data")
            valid = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
            if np.any(valid):
                order = np.argsort(x[valid])
                upper.plot(x[valid][order], y[valid][order], ".-", label=name)
                valid_residual = valid & np.isfinite(residual)
                lower.plot(x[valid_residual], residual[valid_residual], "o-", label=name)
                any_model = True
        upper.set_xscale("log"); upper.set_yscale("log")
        _frequency_ticks(upper)
        upper.legend()
        lower.set_xscale("log")
        _frequency_ticks(lower)
        lower.axhline(0., color="black", linewidth=.8)
        if any_model:
            lower.legend()
    else:
        _empty(upper, "No positive spectrum supplied")
        _empty(lower)
    upper.set_ylabel("Power (W Hz$^{-1}$ sr$^{-1}$)")
    lower.set_ylabel("log$_{10}$(model / data)")
    lower.set_xlabel("Rest-frame frequency (MHz)")
    upper.set_title("Observed spectrum and fitted estimators")
    _save(figure, target)


def _intersection_rows(intersections):
    frequencies = intersections.get("frequencies_mhz_rest", [])
    rows = []
    for track in _rows(intersections.get("tracks")):
        q = track.get("log10_q_by_frequency") or []
        rho = track.get("log10_rho0_by_frequency") or []
        for index, frequency in enumerate(frequencies):
            rows.append({"age_myr": track.get("age_myr"), "alpha_inj": intersections.get("alpha_inj"),
                         "frequency_mhz_rest": frequency, "valid": track.get("valid", True),
                         "log10_jet_power_w": q[index] if index < len(q) else None,
                         "log10_rho0_kg_m3": rho[index] if index < len(rho) else None,
                         "delta": track.get("delta"), "status": track.get("status", "ok" if track.get("valid") else track.get("reason", "unavailable"))})
    return rows


def _intersections(intersections, primary, target):
    tracks = _rows(intersections.get("tracks"))
    frequencies = intersections.get("frequencies_mhz_rest", [])
    figure = _figure(figsize=(15, 7))
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    plotted = 0
    for index, frequency in enumerate(frequencies):
        q, rho = [], []
        for row in tracks:
            qq, rr = row.get("log10_q_by_frequency") or [], row.get("log10_rho0_by_frequency") or []
            valid = row.get("valid", True)
            q.append(_finite(qq[index]) if valid and index < len(qq) else np.nan)
            rho.append(_finite(rr[index]) if valid and index < len(rr) else np.nan)
        if np.any(np.isfinite(q) & np.isfinite(rho)):
            left.plot(rho, q, ".-", markersize=6, label=f"{_finite(frequency):g} MHz")
            plotted += 1
    if plotted:
        left.legend(ncol=2 if plotted > 5 else 1)
    else:
        _empty(left, "No valid per-frequency roots")
    age, delta = _values(tracks, "age_myr"), _values(tracks, "delta")
    valid = np.isfinite(age) & (age > 0) & np.isfinite(delta) & (delta >= 0)
    if np.any(valid):
        right.plot(np.where(valid, age, np.nan), np.where(valid, delta, np.nan), ".-")
        right.set_xscale("log")
        _frequency_ticks(right)
        best_age = _finite(primary.get("best_fit", {}).get("age_myr"))
        if np.isfinite(best_age) and best_age > 0:
            right.axvline(best_age, color="black", linestyle="--", label="Reported fit age")
            right.legend()
    else:
        _empty(right, "No valid intersection statistic")
    left.set_xlabel("log$_{10}$ density normalization $\\rho_0$ (kg m$^{-3}$)")
    left.set_ylabel("log$_{10}$ one-jet power (W)")
    right.set_xlabel("Age (Myr)")
    right.set_ylabel("Mean pairwise $\\Delta$ (log Q, log $\\rho_0$)")
    figure.suptitle("Classical frequency intersections\nFixed atmosphere and core-radius assumptions")
    _save(figure, target)


def _alpha_profile(rows, primary, target):
    figure = _figure(figsize=(12, 10))
    x = _values(rows, "alpha_inj")
    panels = [("log10_q_times_t_j", "log$_{10}$(one-jet Q × t / J)"),
              ("age_myr", "Age (Myr)"),
              ("log10_jet_power_w", "log$_{10}$ one-jet power (W)"),
              ("log10_rho0_kg_m3", "log$_{10}$ density $\\rho_0$ (kg m$^{-3}$)")]
    best = _finite(primary.get("best_fit", {}).get("alpha_inj"))
    for index, (key, label) in enumerate(panels, 1):
        axis = figure.add_subplot(2, 2, index)
        y = _values(rows, key, "log10_rho0" if key == "log10_rho0_kg_m3" else key)
        good = np.isfinite(x) & np.isfinite(y)
        if np.any(good):
            order = np.argsort(x)
            axis.plot(np.where(good, x, np.nan)[order], np.where(good, y, np.nan)[order], ".-")
            if np.isfinite(best):
                axis.axvline(best, color="black", linestyle="--", label="Selected classical α")
                if index == 1:
                    axis.legend()
        else:
            _empty(axis)
        axis.set_xlabel("Injection index α")
        axis.set_ylabel(label)
    figure.suptitle("Classical injection-index profile\nMinimum Q × t is a heuristic")
    _save(figure, target)


def _evolution(evolution, target):
    rows = sorted(_rows(evolution.get("tracks")), key=lambda row: _finite(row.get("age_myr"), math.inf))
    age = _values(rows, "age_myr")
    length = _values(rows, "lobe_length_kpc", "lobe_length_kpc_deprojected")
    pressure = _values(rows, "cocoon_pressure_pa", "pressure_pa")
    field = _values(rows, "magnetic_field_t", "newest_cohort_magnetic_field_t") * 1e10
    valid_rows = np.asarray([row.get("valid", True) is not False for row in rows])
    figure = _figure(figsize=(12, 10))
    for index, (values, label) in enumerate([(length, "One-lobe length (kpc)"),
                                            (pressure, "Cocoon pressure (Pa)"),
                                            (field, "Model magnetic field (μG)")], 1):
        axis = figure.add_subplot(2, 2, index)
        good = valid_rows & np.isfinite(age) & (age > 0) & np.isfinite(values) & (values > 0)
        if np.any(good):
            axis.loglog(np.where(good, age, np.nan), np.where(good, values, np.nan), ".-")
        else:
            _empty(axis)
        axis.set_xlabel("Age (Myr)"); axis.set_ylabel(label)
    axis = figure.add_subplot(2, 2, 4)
    total_size = _values(rows, "source_total_length_kpc", "total_size_kpc")
    has_total_size = bool(np.any(np.isfinite(total_size) & (total_size > 0)))
    size = total_size if has_total_size else length
    frequencies = evolution.get("frequencies_mhz_rest", [])
    plotted = 0
    for index, frequency in enumerate(frequencies):
        power = []
        for row in rows:
            values = row.get("radio_power_w_hz_sr_by_frequency", row.get("radio_powers_w_hz_sr", [])) or []
            power.append(_finite(values[index]) if index < len(values) else np.nan)
        power = np.array(power)
        good = valid_rows & np.isfinite(size) & (size > 0) & np.isfinite(power) & (power > 0)
        if np.any(good):
            axis.loglog(np.where(good, size, np.nan), np.where(good, power, np.nan), ".-", label=f"{_finite(frequency):g} MHz")
            plotted += 1
    if plotted:
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=min(3, plotted), frameon=False)
        from matplotlib.ticker import LogLocator, NullLocator
        axis.xaxis.set_major_locator(LogLocator(base=10., subs=(1., 2., 5.)))
        axis.xaxis.set_minor_locator(NullLocator())
    else:
        _empty(axis, "Radio evolution not supplied")
    axis.set_xlabel("Total source size D (kpc)" if has_total_size else "One-lobe length (kpc)")
    axis.set_ylabel("Power (W Hz$^{-1}$ sr$^{-1}$)")
    figure.suptitle("Evolution at fixed fitted Q and density normalization")
    _save(figure, target)


def _comparison_rows(methods):
    result = []
    for name, fit in methods.items():
        best = fit.get("best_fit", {}) if fit.get("success") is not False else {}
        objective = fit.get("objective", {})
        result.append({"method": name, "success": fit.get("success"),
                       "age_myr": best.get("age_myr"), "alpha_inj": best.get("alpha_inj"),
                       "jet_power_w_per_jet": best.get("jet_power_w_per_jet"),
                       "rho0_kg_m3": best.get("rho0_kg_m3_conditional_on_core_radius"),
                       "objective_name": objective.get("name"), "objective_value": objective.get("value"),
                       "warnings": fit.get("warnings", [])})
    return result


def _comparison(rows, target):
    figure = _figure(figsize=(12, 8))
    panels = [("age_myr", "Age (Myr)", False), ("alpha_inj", "Injection index α", False),
              ("jet_power_w_per_jet", "One-jet power (W)", True), ("rho0_kg_m3", "$\\rho_0$ (kg m$^{-3}$)", True)]
    for index, (key, label, logarithmic) in enumerate(panels, 1):
        axis = figure.add_subplot(2, 2, index)
        value = _values(rows, key)
        valid = np.isfinite(value) & ((value > 0) if logarithmic else True)
        positions = np.arange(len(rows))
        if np.any(valid):
            axis.plot(value[valid], positions[valid], "o")
            if logarithmic:
                axis.set_xscale("log")
                _frequency_ticks(axis)
            axis.set_yticks(positions, [row["method"] for row in rows])
            axis.set_ylim(-.5, len(rows)-.5)
        else:
            _empty(axis)
        axis.set_xlabel(label)
    figure.suptitle("Estimator point comparison\nObjective values have different meanings")
    _save(figure, target)


def _sensitivity_rows(records):
    rows = []
    for record in _rows(records):
        fit = record.get("result", record.get("fit", record))
        fit = fit if isinstance(fit, dict) else {}
        best = fit.get("best_fit", fit)
        rows.append({"label": record.get("label", record.get("name", "case")),
                     "status": record.get("status", "ok" if fit.get("success") else "reported"),
                     "age_myr": best.get("age_myr"), "alpha_inj": best.get("alpha_inj"),
                     "jet_power_w_per_jet": best.get("jet_power_w_per_jet", best.get("jet_power_w")),
                     "rho0_kg_m3": best.get("rho0_kg_m3_conditional_on_core_radius", best.get("rho0_kg_m3")),
                     "overrides": record.get("overrides", record.get("parameters", {"model": record.get("model_overrides", {}), "geometry": record.get("geometry_overrides", {})})),
                     "warning": record.get("reason", record.get("warning", record.get("error", "; ".join(record.get("warnings", [])))))})
    return rows


def _transport_rows(metadata):
    rows = []
    for name, value in metadata.get("map_statistics", {}).items():
        if not isinstance(value, dict) or "median" not in value:
            continue
        interval = value.get("credible_interval_68", [None, None])
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise AnalysisProductError("transport-statistic interval must contain two bounds")
        low, median, high = _finite(interval[0]), _finite(value.get("median")), _finite(interval[1])
        if all(np.isfinite(item) for item in (low, median, high)) and not low <= median <= high:
            raise AnalysisProductError("transport-statistic bounds must bracket the posterior median")
        rows.append({"statistic": name, "mean": value.get("mean"), "median": value.get("median"),
                     "low_68": interval[0], "high_68": interval[1],
                     "reporting_status": metadata.get('parameter_quality', {}).get('transport', {}).get('status', metadata.get('map_statistics', {}).get('interpretation', 'conditional_diagnostic')),
                     "clock": "mapped plasma transport time, not global source lifetime"})
    return rows


def _transport_summary(rows, target):
    figure = _figure(figsize=(12, max(3.5, 1.2*len(rows)+1.8)))
    axis = figure.add_subplot(1, 1, 1)
    value = _values(rows, "median")
    low, high = _values(rows, "low_68"), _values(rows, "high_68")
    valid = np.isfinite(value)
    if np.any(valid):
        error = np.vstack((np.maximum(value - low, 0), np.maximum(high - value, 0)))
        axis.errorbar(value[valid], np.arange(len(rows))[valid], xerr=np.where(np.isfinite(error[:, valid]), error[:, valid], 0), fmt="o", capsize=3)
        axis.set_yticks(np.arange(len(rows)), [row["statistic"].replace("_", " ") for row in rows])
        axis.set_ylim(-.5, len(rows)-.5)
    else:
        _empty(axis)
    axis.set_xlabel("Mapped plasma transport time (Myr)")
    title = 'Diagnostic posterior spatial statistics' if any(row.get('reporting_status') == 'diagnostic_only' for row in rows) else 'Posterior spatial statistics'
    axis.set_title(title+"\nCentral 68% intervals")
    _save(figure, target)


def _regional_rows(data):
    frequencies = data.get("frequencies_mhz", [])
    rows = []
    for region in _rows(data.get("regions")):
        arrays = {key: region.get(key, []) for key in ("observed_jy_per_beam", "rms_jy_per_beam", "model_jy_per_beam")}
        if any(not isinstance(value, list) or len(value) != len(frequencies) for value in arrays.values()):
            raise AnalysisProductError("regional spectrum arrays must match frequencies_mhz")
        for index, frequency in enumerate(frequencies):
            rows.append({"region_id": region.get("id"), "branch": region.get("branch"),
                         "pixel_xy": region.get("pixel_xy"), "frequency_mhz_observed": frequency,
                         **{key: value[index] for key, value in arrays.items()}})
    return rows


def _regional_spectra(data, regions, target, page, total):
    columns = min(2, len(regions))
    rows = int(math.ceil(len(regions) / columns))
    figure = _figure(figsize=(6 * columns, 5 * rows))
    frequencies = np.asarray([_finite(value) for value in data.get("frequencies_mhz", [])])
    for index, region in enumerate(regions, 1):
        axis = figure.add_subplot(rows, columns, index)
        observed = np.asarray([_finite(value) for value in region["observed_jy_per_beam"]])
        noise = np.asarray([_finite(value) for value in region["rms_jy_per_beam"]])
        model = np.asarray([_finite(value) for value in region["model_jy_per_beam"]])
        good = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(observed) & (observed > 0)
        if np.any(good):
            axis.errorbar(frequencies[good], observed[good],
                          yerr=np.where(np.isfinite(noise[good]) & (noise[good] > 0), noise[good], 0.),
                          fmt="o", color="black", markersize=6, capsize=2, label="Observed")
            valid_model = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(model) & (model > 0)
            order = np.argsort(frequencies)
            axis.plot(np.where(valid_model, frequencies, np.nan)[order], np.where(valid_model, model, np.nan)[order], "-", label=data.get("model_label", "Flow/JP model"))
            axis.set_xscale("log"); axis.set_yscale("log")
            _frequency_ticks(axis)
            if index == 1:
                axis.legend()
        else:
            _empty(axis, "No positive detected spectrum")
        axis.set_title(f"Region {region.get('id', index)}; branch {region.get('branch', '?')}")
        axis.set_xlabel("Observed frequency (MHz)")
        axis.set_ylabel("Intensity (Jy/beam)")
        axis.tick_params(labelsize=17)
    figure.suptitle(f"Representative radio spectra ({page}/{total})\nThermal error bars; shared errors enter the fit")
    _save(figure, target)


def _core_rows(data):
    """Retain central-beam brightness measurements without assigning an age."""
    frequencies = data.get("frequencies_mhz", [])
    arrays = {key: data.get(key, []) for key in
              ("observed_jy_per_beam", "rms_jy_per_beam", "model_jy_per_beam")}
    if not isinstance(frequencies, list) or any(
            not isinstance(value, list) or len(value) != len(frequencies)
            for value in arrays.values()):
        raise AnalysisProductError("core spectrum arrays must match frequencies_mhz")
    return [{"frequency_mhz_observed": frequency,
             **{key: value[index] for key, value in arrays.items()},
             "pixel_xy": data.get("pixel_xy"),
             "sky_icrs_deg": data.get("sky_icrs_deg"),
             "age_status": "no age inferred by this descriptive power-law diagnostic"}
            for index, frequency in enumerate(frequencies)]


def _core_spectrum(data, target):
    figure = _figure(figsize=(11, 10))
    upper = figure.add_subplot(2, 1, 1)
    lower = figure.add_subplot(2, 1, 2, sharex=upper)
    frequency = np.asarray([_finite(value) for value in data["frequencies_mhz"]])
    observed = np.asarray([_finite(value) for value in data["observed_jy_per_beam"]])
    noise = np.asarray([_finite(value) for value in data["rms_jy_per_beam"]])
    model = np.asarray([_finite(value) for value in data["model_jy_per_beam"]])
    good = np.isfinite(frequency) & (frequency > 0) & np.isfinite(observed) & (observed > 0)
    valid_model = np.isfinite(frequency) & (frequency > 0) & np.isfinite(model) & (model > 0)
    if np.any(good):
        upper.errorbar(frequency[good], observed[good],
                       yerr=np.where(np.isfinite(noise[good]) & (noise[good] > 0), noise[good], 0.),
                       fmt="o", color="black", capsize=3, label="Central-beam brightness; thermal RMS")
        if np.any(valid_model):
            order = np.argsort(frequency)
            upper.plot(np.where(valid_model, frequency, np.nan)[order],
                       np.where(valid_model, model, np.nan)[order], ".-",
                       label="Descriptive power law")
        upper.set_xscale("log"); upper.set_yscale("log")
        _frequency_ticks(upper)
        upper.legend()
    else:
        _empty(upper, "No positive central-beam measurements")
    valid_residual = good & valid_model
    if np.any(valid_residual):
        order = np.argsort(frequency[valid_residual])
        lower.plot(frequency[valid_residual][order],
                   np.log10(model[valid_residual][order] / observed[valid_residual][order]), "o-")
        lower.set_xscale("log")
        _frequency_ticks(lower)
        lower.axhline(0., color="black", linewidth=.8)
    else:
        _empty(lower, "No fitted descriptive spectrum")
    fit = data.get("power_law")
    alpha = _finite(fit.get("alpha_positive_convention")) if isinstance(fit, dict) else np.nan
    suffix = f"; α = {alpha:.3f}" if np.isfinite(alpha) else ""
    upper.set_title("Included central-beam emission" + suffix)
    upper.set_ylabel("Intensity (Jy/beam)")
    lower.set_ylabel("log$_{10}$(model / data)")
    lower.set_xlabel("Observed frequency (MHz)")
    figure.suptitle("Central-beam power-law diagnostic\nAge profiles are fitted separately")
    _save(figure, target)


def _profile_rows(data):
    rows = []
    for branch in _rows(data.get("branches")):
        names = ("distance_kpc", "age_p16_myr", "age_p50_myr", "age_p84_myr")
        arrays = {name: branch.get(name, []) for name in names}
        lengths = [len(value) for value in arrays.values() if isinstance(value, list)]
        if len(lengths) != len(names) or len(set(lengths)) > 1:
            raise AnalysisProductError("branch transport-profile arrays must have equal lengths")
        for index in range(lengths[0] if lengths else 0):
            row = {"branch": branch.get("branch"), **{name: value[index] for name, value in arrays.items()}}
            low, median, high = (_finite(row[name]) for name in names[1:])
            if all(np.isfinite(value) for value in (low, median, high)) and not 0 <= low <= median <= high:
                raise AnalysisProductError("transport-profile intervals must satisfy 0 <= p16 <= p50 <= p84")
            rows.append(row)
    return rows


def _transport_profiles(data, target):
    figure = _figure(figsize=(11, 7))
    axis = figure.add_subplot(1, 1, 1)
    plotted = 0
    for branch in _rows(data.get("branches")):
        x = np.asarray([_finite(value) for value in branch.get("distance_kpc", [])])
        low = np.asarray([_finite(value) for value in branch.get("age_p16_myr", [])])
        median = np.asarray([_finite(value) for value in branch.get("age_p50_myr", [])])
        high = np.asarray([_finite(value) for value in branch.get("age_p84_myr", [])])
        valid = np.isfinite(x) & (x >= 0) & np.isfinite(median) & (median >= 0)
        if np.any(valid):
            order = np.argsort(x)
            xx, yy = np.where(valid, x, np.nan)[order], np.where(valid, median, np.nan)[order]
            line, = axis.plot(xx, yy, label=f"Branch {branch.get('branch', '?')}")
            bounds = valid & np.isfinite(low) & np.isfinite(high)
            axis.fill_between(x[order], np.where(bounds, low, np.nan)[order],
                              np.where(bounds, high, np.nan)[order], color=line.get_color(), alpha=.2)
            plotted += 1
    if plotted:
        axis.legend(title="Median and central 68% interval")
    else:
        _empty(axis, "No valid transport profiles")
    axis.set_xlabel("Adopted projected path length (kpc)")
    axis.set_ylabel("Plasma transport time (Myr)")
    axis.set_title("Shared-model transport profiles\nGlobal source lifetime remains unidentified")
    _save(figure, target)


def _report(document, methods, primary, figures, tables):
    source = document.get("source", document.get("radio_result", {}).get("source", {}))
    name = str(source.get("name", "Unnamed source")).replace("\n", " ")
    kind = document.get("kind")
    lines = [f"# {name}: analysis products", ""]
    if kind == "radio_transport":
        radio = document.get("radio_result", {})
        lines += ["**This is a conditional plasma transport analysis. A global source dynamical age is not inferred by the transport model.**", "",
                  f"Status: `{radio.get('status', document.get('status', 'reported'))}`.", ""]
        quality = document.get('parameter_quality', {}).get('transport', radio.get('parameter_quality', {}).get('transport', {}))
        if quality.get('status') == 'diagnostic_only':
            lines += ['**Supported mean/median transport estimates are withheld. Numerical summaries below are diagnostic model outputs.**', '',
                      'Reasons: '+ '; '.join(quality.get('reasons', []))+'.', '']
        for row in _transport_rows(radio):
            lines.append(f"- {row['statistic']}: posterior median {row['median']} Myr; central 68% interval [{row['low_68']}, {row['high_68']}].")
        parameters = radio.get('posterior_parameters', {})
        if parameters:
            parameter_quality = document.get('parameter_quality', radio.get('parameter_quality', {})).get('parameters', {})
            lines += ['', '## Conditional transport parameters', '',
                '| Parameter | Numerical median | Central 68% interval | Reporting status |', '|---|---:|---|---|']
            for key, values in parameters.items():
                interval = values.get('credible_interval_68', [None, None])
                if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                    raise AnalysisProductError('parameter interval must contain two bounds')
                entry = parameter_quality.get(key, {})
                lines.append(f"| {key.replace('_', ' ')} | {_display_value(values.get('median'))} | [{_display_value(interval[0])}, {_display_value(interval[1])}] | {entry.get('status', 'conditional numerical summary')} |")
            lines += ['', 'These shared-parameter summaries condition on the stated geometry, spectrum, errors and physical inputs. Diagnostic-only values are not supported measurements of plasma flow or source lifetime.', '']
        if document.get("map_products"):
            lines += ["", "[WCS age and uncertainty maps](transport.fits) · [Shared posterior samples](posterior.npz)", "",
                      "![Conditional plasma transport map](transport.png)", ""]
            if document["map_products"].get("preview_pdf"):
                lines += ["Map exports: [PDF](transport.pdf) · [SVG](transport.svg)", ""]
        core = radio.get("core")
        if isinstance(core, dict):
            core_modelled = core.get("policy", "separate") == "include"
            coverage = document.get("source_coverage", document.get("diagnostics", {}).get("source_coverage", radio.get("source_coverage", {})))
            spectrum = document.get("diagnostics", {}).get("core_spectrum", core.get("central_spectrum", {}))
            fit = spectrum.get("power_law") if isinstance(spectrum, dict) else None
            lines += ["", "## Core inclusion and source coverage", "",
                      ("**The core region is included in the radiative/transport likelihood and mapped wherever the supplied geometry is valid. No circular core exclusion is applied.**" if core_modelled else "**The core region is included in the source images, spectral diagnostics and coverage layers. Its transport age is unresolved.**"), "",
                      "| Coverage | Detected pixels |", "|---|---:|",
                      f"| Source spectral coverage | {coverage.get('source_detected_pixels', 'See analysis.json')} |",
                      f"| Pixels with conditional transport times | {coverage.get('age_valid_pixels', radio.get('valid_pixels', 'See analysis.json'))} |",
                      f"| Included core region | {coverage.get('core_detected_pixels', core.get('detected_pixels', 'See analysis.json'))} |", "",
                      ("SOURCE_VALID records detected source emission, CORE labels the central diagnostic footprint, and AGE_VALID/AGE_FLAGS record conditional transport coverage. CORE is not an exclusion mask in this analysis. The supplied core coordinate is marked without a circular overlay." if core_modelled else "SOURCE_VALID records detected source emission, CORE identifies the explicit central region, and AGE_VALID/AGE_FLAGS distinguish transport-time coverage and unresolved ages. The core is shown on the preview even where the age layer is undefined."), "",
                      "The central spectrum measures brightness in one restoring beam. It can blend the compact core, inner jets and plume emission; it is not a deblended compact-core flux measurement. The descriptive power law does not measure a break or age and does not establish absorption.", ""]
            if isinstance(fit, dict) and fit.get("status") == "descriptive_gls_fit":
                lines += ["| Central-beam power-law diagnostic | Value |", "|---|---:|",
                          f"| Spectral index α (I ∝ ν^-α) | {fit.get('alpha_positive_convention')} |",
                          f"| Conditional α error | {fit.get('alpha_error_conditional')} |",
                          f"| Chi-square / degrees of freedom | {fit.get('chi2')} / {fit.get('degrees_of_freedom')} |", "",
                          "The fit uses its stated measurement/calibration covariance. Displayed error bars show thermal RMS; they do not display the full shared covariance.", ""]
            lines += [("Transport mean/median summaries include age-valid core pixels. Central values remain conditional on the supplied path/branch geometry, radiative model and shared priors; beam mixing and path ambiguity are not removed by including the core." if core_modelled else "Transport mean/median summaries use only pixels with an estimable conditional transport time. The unresolved core contributes no invented zero or other age. Those statistics do not determine a whole-source lifetime."), "",
                      "The geometric zero-time reference at the supplied core position is an adopted origin, not a measured zero age throughout a finite beam.", ""]
        warnings = document.get("warnings", []) + radio.get("warnings", [])
        if isinstance(core, dict):
            warnings += core.get("warnings", [])
    else:
        lines += ["**Source-level FR II dynamics under the declared lobe model. These fits must not be applied to WAT plumes by changing a morphology label.**", "",
                  f"Primary estimator: `{document.get('primary_method', '')}`.", ""]
        best = primary.get("best_fit", {})
        lines += ["| Quantity | Reported value |", "|---|---:|",
                  f"| Age | {best.get('age_myr')} Myr |",
                  f"| Injection index | {best.get('alpha_inj')} |",
                  f"| One-jet power | {best.get('jet_power_w_per_jet')} W |",
                  f"| Density normalization rho0 | {best.get('rho0_kg_m3_conditional_on_core_radius')} kg/m^3 |", "",
                  "Density depends on the adopted core radius and atmosphere slope; the identifiable combination is rho0*a0^beta. One-jet Q*t and total injected energy use different lobe-count conventions.", "",
                  "Classical Delta and minimum Q*t are physical selection heuristics, not a likelihood or confidence criterion. Estimator objective values are recorded separately and are not ranked on a common axis.", ""]
        warnings = list(document.get("warnings", []))
        for fit in methods.values():
            warnings.extend(fit.get("warnings", []))
        lines += ["## Derived quantities", "", "| Quantity | Value |", "|---|---:|"]
        for key, value in primary.get("derived", {}).items():
            lines.append(f"| {key.replace('_', ' ')} | {value} |")
        lines += ["", "Evolution products hold fitted jet power and density normalization fixed. They are not a series of independent age refits constrained to the same observed length.", ""]
    if document.get("field_constraint") is not None:
        field = document["field_constraint"]
        lines += ["## Magnetic-field constraint", "", str(field.get("equation", "See full field provenance in analysis.json")), ""]
        if field.get('kind') == 'user_fixed':
            lines += ['**User-supplied fixed field for this run.**', '',
                      f"- Field: {_display_value(field.get('value_t'))} T = {_display_value(field.get('value_ug'))} μG.",
                      '- The field is conditioned exactly as supplied; no field uncertainty was provided or inferred.',
                      '- This input replaces the default equipartition-based field prescription for this run. Retained PyRATS equipartition values are comparison properties, not a fabricated field constraint.', '']
        for key in ("equipartition_field_ug", "adopted_field_ug", "scope", "uncertainty", "description"):
            if field.get(key) is not None:
                lines.append(f"- {key.replace('_', ' ')}: {field[key]}")
        lines += ["", "The full field-input provenance and assumptions are retained in analysis.json.", ""]
    if kind == 'radio_transport':
        properties = _source_property_rows(document.get('radio_result', {}))
        if properties:
            lines += ['## Source inputs and geometry', '', '| Quantity | Value | Unit |', '|---|---:|---|']
            for row in properties:
                lines.append(f"| {row['quantity'].replace('_', ' ')} | {_display_value(row['value'])} | {row['unit']} |")
            lines += ['', 'These are retained source inputs and conditional PyRATS geometry estimates. Changing the adopted field does not measure a new volume, line-of-sight depth, filling factor or projected size.', '']
    if document.get("quality"):
        quality = document["quality"]
        lines += ["## Quality diagnostics", ""]
        if kind == "fr2_dynage":
            lines += ["| Method | Reduced chi-square | Flags |", "|---|---:|---|"]
            for method, item in quality.items():
                if isinstance(item, dict):
                    lines.append(f"| {method} | {item.get('reduced_chi2')} | {', '.join(item.get('flags', [])) or 'No listed flag'} |")
        else:
            lines += [f"- Status: {quality.get('status')}", f"- Sampling: {quality.get('sampling')}"]
        lines += ["", "Inspect the complete numerical and identifiability diagnostics in analysis.json before interpreting the point estimates.", ""]
    if document.get("adopted_source_budget"):
        budget = document["adopted_source_budget"].get("budget", {})
        lines += ["## Nominal source budget at the adopted field", "", "| Quantity | Value |", "|---|---:|"]
        for key in ("B_T", "B_uG", "B_eq_uG", "fraction_of_B_eq", "volume_m3",
                    "electron_energy_density_J_m3", "particle_energy_density_J_m3",
                    "magnetic_energy_density_J_m3", "total_energy_density_J_m3",
                    "electron_energy_J", "particle_energy_J", "magnetic_energy_J",
                    "nonradiating_energy_J", "total_energy_J", "pressure_Pa",
                    "magnetic_energy_fraction", "electron_number_density_m3", "electron_number_total",
                    "B_CMB_uG", "u_B_over_u_particles",
                    "B_over_B_CMB", "inverse_compton_dominated"):
            if key in budget:
                label = 'Actual B / nominal B eq (comparison)' if key == 'fraction_of_B_eq' else key.replace('_', ' ')
                lines.append(f"| {label} | {_display_value(budget[key])} |")
        lines += ["", "This PyRATS budget is conditional on its stated particle/volume/spectrum assumptions. No independent jet power is inferred from it.", ""]
        budget_document = document['adopted_source_budget']
        if budget_document.get('uncertainty'):
            lines += ['Budget uncertainty scope: '+str(budget_document['uncertainty']).replace('\n',' '), '']
        assumptions = budget_document.get('assumptions', {})
        if isinstance(assumptions, dict) and assumptions:
            lines += ['| Budget assumption/input | Value |', '|---|---:|']
            for key in ['electron_spectrum','alpha_inj_positive','kappa_proton_electron_energy_ratio',
                        'gamma_min','gamma_max','filling_factor','normalisation_frequency_Hz','normalisation_flux_Jy']:
                if key in assumptions:
                    lines.append(f"| {key.replace('_', ' ')} | {_display_value(assumptions[key])} |")
            lines += ['']
        checked = document.get('parameter_quality', {}).get('source_budget', {})
        lines += [f"Budget validation: `{checked.get('status', 'not assessed')}`. SI/field/composition/volume/energy/pressure accounting checks do not independently validate flux normalization or physical assumptions.", '']
        comparison = checked.get('injection_spectrum_consistency')
        if comparison:
            lines += [f"Upstream budget index: {comparison['upstream_alpha']}; transport index 68% interval: {comparison['transport_alpha_68']}. Spectrum-assumption comparison: `{comparison['status']}`. The nominal budget is not a refit of the current aged electron population.", '']
        if document.get("resolved_spectral_analysis"):
            lines += ["The retained PyRATS budget uses its upstream homogeneous power-law assumptions. It is not a self-consistent energy/pressure refit of the JP–Tribble or CI spectrum.", ""]
    if document.get("unavailable_classical_parameters"):
        lines += ["## Unavailable classical parameters", "", "| Parameter | Reason |", "|---|---|"]
        for key, reason in document["unavailable_classical_parameters"].items():
            lines.append(f"| {str(key).replace('|', '/')} | {str(reason).replace('|', '/')} |")
        lines += [""]
    if document.get("resolved_spectral_analysis"):
        lines += ["## Full-source JP–Tribble and CI spectral maps", "",
            "Every selected detected source pixel, including the core, is fitted without a circular exclusion or a traced-path requirement. JP–Tribble maps describe radiative time since injection; CI maps describe a continuous-injection duration. Neither is automatically the global dynamical lifetime.", "",
            "Best fits remain visible when the data constrain only a bound or do not establish spectral curvature. Inspect AGE_FLAGS, AGE_CONSTRAINED, reduced chi-square and profile limits. A zero lower-bound best fit is an upper-limit case, not a measured zero age. Missing upper limits indicate an open profile, not excluded source emission.", ""]
        for name, summary in document["resolved_spectral_analysis"].items():
            products = summary["products"]
            lines += [f"### {name.replace('_', ' ')}", "",
                f"Field: {summary.get('field_ug')} μG; fixed injection index: {summary.get('alpha_inj')}. Included central pixels: {summary.get('core_modelled_pixels')}.", "",
                f"Profile-constrained ages: {summary.get('constrained_age_pixels')} of {summary.get('fitted_pixels')} fitted pixels. A numerical best fit by itself is not an age detection.", "",
                f"Pixels passing nominal and both injection-index scenarios: {summary.get('scenario_pass_pixels', 'not assessed')}. Whole detected-source spectral mean/median: `{summary.get('whole_detected_source_summary', {}).get('status', 'not assessed')}`.", '',
                f"[FITS maps]({products['fits']}) · [Profile metadata]({products['metadata']}) · [PDF]({products['pdf']}) · [SVG]({products['svg']})", "",
                f"![{name} full-source spectral maps]({products['preview']})", ""]
        if (document.get('field_constraint') or {}).get('kind') == 'user_fixed':
            lines += ['The supplied field is held exactly fixed in both spectral models. If AGE_B16/AGE_B84 compatibility layers are present, they repeat the nominal age layer and represent no sampled field uncertainty. Profile intervals still condition on the fixed field and injection index.', '']
        else:
            lines += ["The profile limits condition on the supplied field and injection index. AGE_B16 and AGE_B84 rescale the age coordinate and search bounds at alternative fields, retaining boundary and weak-curvature flags; they are not combined confidence bounds.", ""]
        lines += ['Pixels and their errors are beam-correlated; a map-average uncertainty cannot be computed as independent pixel errors.', '']
        lines += ['AGE_ALOW/AGE_AHIGH contain actual refits at the checked injection indices; FLAG_ALOW/FLAG_AHIGH preserve all profile flags. AGE_SCENARIO_OK requires the nominal and both checked profiles to pass. These scenarios are conditional sensitivity checks, not marginalized errors or proof that the model is physically correct.', '']
    lines += ["## Tables and figures", ""]
    lines.extend(f"- [{name}]({name})" for name in tables)
    lines += [""]
    lines += ["Figures are supplied as 600 dpi PNG previews and vector PDF/SVG files. Font sizes are enlarged consistently; regional spectra use at most four panels per page.", ""]
    for name in figures:
        if Path(name).suffix != ".png":
            continue
        stem = str(Path(name).with_suffix(""))
        lines += [f"![{Path(name).stem.replace('_', ' ')}]({name})", "",
                  f"Vector files: [PDF]({stem}.pdf) · [SVG]({stem}.svg)", ""]
    lines += ["## Interpretation and limitations", ""]
    for warning in dict.fromkeys(str(value) for value in warnings):
        lines.append("- " + warning.replace("\n", " "))
    lines += ["", "The full input assumptions, unsuccessful cases and provenance remain in [analysis.json](analysis.json). A converged numerical fit is not independent evidence that the physical model applies.", ""]
    return "\n".join(lines)


def write_analysis_products(document, output_dir, *, overwrite=False, protected_paths=()):
    """Write complete products, preserving other artifacts in a staging dir.

    A new directory publishes with one rename. Existing directories receive
    only the named analysis files, atomically per file; existing FITS/NPZ and
    other unrelated products are never deleted. All collisions and protected
    input paths are checked before any publication.
    """
    if not isinstance(document, dict):
        raise AnalysisProductError("analysis document must be a JSON object")
    try:
        serialized = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=_encode) + "\n"
        normalized = json.loads(serialized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisProductError(f"analysis must contain finite JSON data: {exc}") from exc
    kind = normalized.get("kind")
    if kind not in {"fr2_dynage", "radio_transport"}:
        raise AnalysisProductError("analysis kind must be fr2_dynage or radio_transport")
    target = Path(output_dir).expanduser().resolve()
    protected = [Path(path).expanduser().resolve() for path in protected_paths]
    if any(path == target or path.is_relative_to(target) or (path.is_dir() and target.is_relative_to(path)) for path in protected):
        raise AnalysisProductError("output directory must not replace, contain or modify protected input paths")
    if target.exists():
        if not target.is_dir():
            raise AnalysisProductError("analysis output must be a directory")
    methods = normalized.get("methods", {})
    if not isinstance(methods, dict) or any(not isinstance(value, dict) for value in methods.values()):
        raise AnalysisProductError("methods must map estimator names to result objects")
    primary = methods.get(normalized.get("primary_method"), {})
    if kind == "fr2_dynage" and (not primary or not isinstance(primary.get("best_fit"), dict) or primary.get("success") is False):
        raise AnalysisProductError("FR II analysis requires a primary_method with a best_fit object")
    if kind == "radio_transport" and not isinstance(normalized.get("radio_result"), dict):
        raise AnalysisProductError("radio transport analysis requires radio_result metadata")
    diagnostics = normalized.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise AnalysisProductError("diagnostics must be an object")
    try:
        from matplotlib.figure import Figure  # noqa: F401
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    except ImportError as exc:
        raise AnalysisProductError("analysis figures require pyradage[maps] plotting dependencies") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + target.name + ".", dir=target.parent))
    try:
        (stage / "analysis.json").write_text(serialized, encoding="utf8")
        figures, tables = [], []
        def table(name, rows, fields=()):
            (stage / name).write_text(_table(rows, fields), encoding="utf8")
            tables.append(name)
        if kind == "fr2_dynage":
            observed = _rows(diagnostics.get("observed_spectrum"))
            spectrum = _rows(primary.get("spectrum")) or observed
            table("spectrum.csv", spectrum, ("frequency_mhz_rest", "observed_power_w_hz_sr", "predicted_power_w_hz_sr"))
            table("derived_quantities.csv", [{"quantity": key, "value": value} for key, value in primary.get("derived", {}).items()], ("quantity", "value"))
            _spectrum(primary, methods, observed, stage / "spectrum.png"); figures.append("spectrum.png")
            comparisons = _comparison_rows(methods)
            table("fit_comparison.csv", comparisons, ("method", "age_myr", "alpha_inj"))
            _comparison(comparisons, stage / "fit_comparison.png"); figures.append("fit_comparison.png")
            intersections = diagnostics.get("intersections", {})
            if isinstance(intersections, dict) and _rows(intersections.get("tracks")):
                table("intersection_tracks.csv", _intersection_rows(intersections), ("age_myr", "frequency_mhz_rest", "delta"))
                _intersections(intersections, primary, stage / "intersections.png"); figures.append("intersections.png")
            profile = _rows(diagnostics.get("alpha_profile"))
            if profile:
                table("alpha_profile.csv", profile, ("alpha_inj", "age_myr", "delta", "log10_q_times_t_j"))
                _alpha_profile(profile, methods.get("classic", primary), stage / "alpha_profile.png"); figures.append("alpha_profile.png")
            evolution = diagnostics.get("evolution", {})
            if isinstance(evolution, dict) and _rows(evolution.get("tracks")):
                table("evolution.csv", _rows(evolution["tracks"]), ("age_myr", "lobe_length_kpc", "cocoon_pressure_pa"))
                _evolution(evolution, stage / "evolution.png"); figures.append("evolution.png")
        else:
            radio = normalized["radio_result"]
            source_properties = _source_property_rows(radio)
            if source_properties:
                table('source_properties.csv', source_properties, ('quantity','value','unit','scope'))
            rows = _transport_rows(radio)
            table("transport_statistics.csv", rows, ("statistic", "mean", "median", "low_68", "high_68", "reporting_status", "clock"))
            quality = normalized.get('parameter_quality', {})
            if quality:
                parameters = []
                for name, entry in quality.get('parameters', {}).items():
                    parameters.append({'parameter': name, 'status': entry['status'],
                        'estimate_median': (entry.get('estimate') or {}).get('median'),
                        'diagnostic_median': (entry.get('diagnostic_summary') or {}).get('median'),
                        'reason': '; '.join(entry.get('reasons', []))})
                for name in ['mean_myr', 'median_myr']:
                    transport = quality.get('transport', {})
                    parameters.append({'parameter': 'transport_'+name, 'status': transport.get('status'),
                        'estimate_median': (transport.get(name) or {}).get('median'),
                        'reason': '; '.join(transport.get('reasons', []))})
                table('parameter_reliability.csv', parameters,
                      ('parameter','status','estimate_median','diagnostic_median','reason'))
            _transport_summary(rows, stage / "transport_summary.png"); figures.append("transport_summary.png")
            core = radio.get("core")
            core_spectrum = diagnostics.get("core_spectrum", core.get("central_spectrum", {}) if isinstance(core, dict) else {})
            if isinstance(core_spectrum, dict) and core_spectrum.get("frequencies_mhz"):
                table("core_spectrum.csv", _core_rows(core_spectrum),
                      ("frequency_mhz_observed", "observed_jy_per_beam", "rms_jy_per_beam", "model_jy_per_beam", "age_status"))
                _core_spectrum(core_spectrum, stage / "core_spectrum.png"); figures.append("core_spectrum.png")
            if normalized.get("adopted_source_budget"):
                table("adopted_source_budget.csv", [{"quantity": key, "value": value} for key, value in normalized["adopted_source_budget"].get("budget", {}).items()], ("quantity", "value"))
            regional = diagnostics.get("regional_spectra", {})
            if isinstance(regional, dict) and _rows(regional.get("regions")):
                table("regional_spectra.csv", _regional_rows(regional), ("region_id", "branch", "frequency_mhz_observed"))
                regions = regional["regions"]
                pages = int(math.ceil(len(regions) / 4))
                for page in range(pages):
                    name = f"regional_spectra_{page + 1:02d}.png"
                    _regional_spectra(regional, regions[page*4:(page+1)*4], stage/name, page+1, pages)
                    figures.append(name)
            profiles = diagnostics.get("transport_profiles", {})
            if isinstance(profiles, dict) and _rows(profiles.get("branches")):
                table("transport_profiles.csv", _profile_rows(profiles), ("branch", "distance_kpc", "age_p16_myr", "age_p50_myr", "age_p84_myr"))
                _transport_profiles(profiles, stage/"transport_profiles.png"); figures.append("transport_profiles.png")
        sensitivity = _sensitivity_rows(normalized.get("sensitivity"))
        if sensitivity:
            table("sensitivity.csv", sensitivity, ("label", "status", "age_myr", "alpha_inj"))
        previews = list(figures)
        vectors = [str(Path(name).with_suffix(suffix)) for name in previews for suffix in (".pdf", ".svg")]
        figures = [*previews, *vectors]
        (stage / "REPORT.md").write_text(_report(normalized, methods, primary, figures, tables), encoding="utf8")
        files = ["analysis.json", "REPORT.md", *tables, *figures]
        manifest = {"schema_version": "fr2dynage-analysis-products-1", "kind": kind,
                    "files": files, "figures": figures, "preview_figures": previews,
                    "vector_figures": vectors, "tables": tables,
                    "sha256": {name: hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in files}}
        (stage / _MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        for name in [*files, _MANIFEST]:
            destination = target/name
            resolved = destination.resolve()
            if any(resolved == path or (path.is_dir() and resolved.is_relative_to(path)) for path in protected):
                raise AnalysisProductError("an analysis product would overwrite a protected input")
            if destination.exists() and (not overwrite or not destination.is_file()):
                raise FileExistsError(f"{destination} already exists; request overwrite explicitly")
        if not target.exists():
            os.rename(stage, target)
        else:
            for name in [*files, _MANIFEST]:
                if overwrite:
                    os.replace(stage/name, target/name)
                else:
                    os.link(stage/name, target/name)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
