"""Full-source TJP/CI spectral fits and products, independent of traced paths."""
from pathlib import Path
import json
import math

import numpy as np

from .maps_io import MapInputError, read_fits_plane, science_image, auxiliary_data, atomic_write_fits


def parse_resolved_settings(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec)-{'models', 'age_bounds_myr', 'age_grid_size', 'alpha_inj', 'alpha_sensitivity'}:
        raise MapInputError('resolved_analysis accepts models, age_bounds_myr, age_grid_size, alpha_inj and alpha_sensitivity')
    models = spec.get('models', ['jp_tribble', 'ci'])
    if (not isinstance(models, list) or not 1 <= len(models) <= 2 or
        any(not isinstance(model, str) or model not in {'jp_tribble', 'ci'} for model in models) or len(set(models)) != len(models)):
        raise MapInputError('resolved_analysis.models requires distinct jp_tribble and/or ci')
    bounds = spec.get('age_bounds_myr', [0., 1000.])
    if (not isinstance(bounds, list) or len(bounds) != 2 or
        any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in bounds) or
        not 0 <= bounds[0] < bounds[1] <= 1e5):
        raise MapInputError('resolved_analysis.age_bounds_myr requires 0 <= lower < upper <= 100000')
    grid = spec.get('age_grid_size', 241)
    if isinstance(grid, bool) or not isinstance(grid, int) or not 81 <= grid <= 2001:
        raise MapInputError('resolved_analysis.age_grid_size must be an integer in [81,2001]')
    alpha = spec.get('alpha_inj', 'transport_posterior_median')
    if alpha != 'transport_posterior_median' and (isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not .5 <= alpha <= 1.5):
        raise MapInputError('resolved_analysis.alpha_inj must be 0.5..1.5 or transport_posterior_median')
    sensitivity = spec.get('alpha_sensitivity', 'transport_posterior_68' if isinstance(alpha, str) else None)
    if sensitivity is not None and sensitivity != 'transport_posterior_68':
        if (not isinstance(sensitivity, list) or len(sensitivity) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in sensitivity) or
            not .5 <= sensitivity[0] < sensitivity[1] <= 1.5):
            raise MapInputError('alpha_sensitivity must be transport_posterior_68, null or two increasing values in [0.5,1.5]')
    return {'models': models, 'age_bounds_myr': bounds, 'age_grid_size': grid, 'alpha_inj': alpha, 'alpha_sensitivity': sensitivity}


def compute_resolved_products(radio, settings):
    """Reuse the exact accepted image pixels; apply no central or path mask."""
    from .resolved_spectra import fit_resolved_spectra
    from .radio_spectra import jp_loss_coefficient
    from .advanced_spectra import model_metadata
    from .maps import _file_spec
    config = radio.metadata['input_configuration']
    parent = Path(radio.metadata['configuration_path']).parent
    accepted_hashes = {(item['path'], item.get('hdu', 0)): item['sha256']
                       for item in radio.metadata['input_files'] if item['role'].startswith('images[')}
    fluxes, noises, frequencies, calibrations = [], [], [], []
    for record in config['images']:
        path, hdu = _file_spec({key:record[key] for key in ('path','hdu') if key in record}, parent, 'resolved image')
        plane = read_fits_plane(path, hdu)
        if accepted_hashes.get((str(path), hdu)) != plane.sha256:
            raise MapInputError('image changed between transport and resolved spectral fitting')
        flux, _ = science_image(plane)
        if 'rms_map' in record:
            noise_path, noise_hdu = _file_spec(record['rms_map'], parent, 'resolved rms_map')
            noise_plane = read_fits_plane(noise_path, noise_hdu)
            if accepted_hashes.get((str(noise_path), noise_hdu)) != noise_plane.sha256:
                raise MapInputError('noise map changed between transport and resolved spectral fitting')
            noise = auxiliary_data(noise_plane, plane, 'Jy / beam')
        else:
            noise = np.full(flux.shape, record['rms_jy_per_beam'])
        fluxes.append(flux); noises.append(noise)
        frequencies.append(record['frequency_mhz']); calibrations.append(record.get('calibration_fraction', 0.))
    field = float(np.median(radio.samples['magnetic_field_ug']))
    prescription = radio.metadata.get('field_prescription') or {}
    fixed_field = prescription.get('kind') == 'user_fixed'
    native_beq = ((radio.metadata.get('pyrats_properties') or {}).get('properties', {})
                   .get('native_field_budgets', {}).get('equipartition', {}).get('B_uG'))
    alpha_spec = settings['alpha_inj']
    alpha = float(np.median(radio.samples['alpha_inj'])) if isinstance(alpha_spec, str) else float(alpha_spec)
    sensitivity = settings.get('alpha_sensitivity')
    alpha_cases = (np.quantile(radio.samples['alpha_inj'], [.16, .84]).tolist()
                   if sensitivity == 'transport_posterior_68' else sensitivity)
    if alpha_cases is not None and not alpha_cases[0] <= alpha <= alpha_cases[1]:
        raise MapInputError('alpha_sensitivity must bracket the nominal injection index')
    redshift = radio.metadata['source']['redshift']
    correlation = config.get('calibration_correlation', np.eye(len(frequencies)).tolist())
    source_mask = radio.maps['SOURCE_MASK'].astype(bool)
    core = radio.maps.get('CORE', np.zeros(source_mask.shape)).astype(bool)
    outputs = {}
    for model in settings['models']:
        def fit_at_index(index):
            return fit_resolved_spectra(frequencies, np.stack(fluxes), np.stack(noises), calibrations,
                correlation, source_mask, field, redshift, index, model=model,
                age_bounds_myr=tuple(settings['age_bounds_myr']), age_grid_size=settings['age_grid_size'],
                snr_min=config.get('selection', {}).get('snr_min', 5.),
                min_bands=config.get('selection', {}).get('min_bands', 4))
        fitted = fit_at_index(alpha)
        scenario_ok = fitted['maps']['AGE_CONSTRAINED'].astype(bool).copy()
        alpha_summaries = {}
        if alpha_cases is not None:
            for index, label in zip(alpha_cases, ['ALOW', 'AHIGH']):
                varied = fitted if index == alpha else fit_at_index(index)
                fitted['maps']['AGE_'+label] = varied['maps']['SPECTRAL_AGE'].copy()
                fitted['maps']['FLAG_'+label] = varied['maps']['AGE_FLAGS'].copy()
                scenario_ok &= varied['maps']['AGE_CONSTRAINED'].astype(bool)
                alpha_summaries[label] = {'alpha_inj': index,
                    'constrained_age_pixels': varied['metadata']['constrained_age_pixels'],
                    'flags': varied['metadata']['flags']}
        else:
            scenario_ok[:] = False
        fitted['maps']['AGE_SCENARIO_OK'] = scenario_ok.astype(np.uint8)
        fitted['maps']['CORE'] = core.astype(np.uint8)
        fitted['maps']['SOURCE_VALID'] = radio.maps['SOURCE_VALID']
        valid = fitted['maps']['SPEC_VALID'].astype(bool)
        def spatial_summary(selection):
            values = fitted['maps']['SPECTRAL_AGE'][selection & valid]
            return {'pixels': int(len(values)), 'mean': float(np.mean(values)) if len(values) else None,
                    'median': float(np.median(values)) if len(values) else None,
                    'interpretation': 'descriptive best-fit spectral-time map statistic; not a global dynamical age or independent-pixel uncertainty estimate'}
        if not np.all(valid[radio.maps['SOURCE_VALID'].astype(bool)]):
            # Numerical failures must remain visible, never disguised as a
            # central mask or silently replaced by a model origin.
            fitted['metadata']['numerical_fit_failures_in_detected_source'] = int(np.sum(radio.maps['SOURCE_VALID'].astype(bool) & ~valid))
        field_sensitivity = {}
        for percentile, label in ([] if fixed_field else [(.16, 'AGE_B16'), (.84, 'AGE_B84')]):
            alternative = float(np.quantile(radio.samples['magnetic_field_ug'], percentile))
            factor = math.sqrt(alternative/field)*jp_loss_coefficient(field, redshift)/jp_loss_coefficient(alternative, redshift)
            fitted['maps'][label] = fitted['maps']['SPECTRAL_AGE']*factor
            field_sensitivity[label] = {'field_ug': alternative, 'age_scale': factor,
                                        'rescaled_search_bounds_myr': [factor*x for x in settings['age_bounds_myr']]}
        fitted['metadata'].update(
            model_description=model_metadata(model),
            source=radio.metadata['source'],
            field_prescription=(f"User-fixed B = {prescription['value_t']:.12g} T; RMS field for free-streaming JP–Tribble, uniform field for CI" if fixed_field else
                                'B_rms = 0.4 B_eq for free-streaming JP–Tribble; B = 0.4 B_eq for homogeneous CI'),
            field_input=prescription, field_ug=field,
            equipartition_field_ug=native_beq if fixed_field else field/.4,
            equipartition_field_basis='native deterministic PyRATS reference, not inferred from fixed B' if fixed_field else 'shared posterior field divided by the prescribed fraction',
            field_fraction=None if fixed_field else .4,
            nominal_B_over_Beq=(field/native_beq if native_beq is not None else None), alpha_inj=alpha,
            alpha_input=alpha_spec, source_detected_pixels=int(np.sum(radio.maps['SOURCE_VALID'])),
            core_detected_pixels=int(np.sum(core & radio.maps['SOURCE_VALID'].astype(bool))),
            core_modelled_pixels=int(np.sum(core & valid)), fixed_circular_exclusion=False,
            map_statistics={'all_fitted_best_times_myr': spatial_summary(valid),
                            'profile_constrained_best_times_myr': spatial_summary(fitted['maps']['AGE_CONSTRAINED'].astype(bool)),
                            'central_best_times_myr': spatial_summary(core)},
            field_sensitivity=field_sensitivity,
            sensitivity_interpretation=('Field held exactly at the user value; no AGE_B16/AGE_B84 layers or field-uncertainty interval are manufactured. Injection-index scenarios are separate conditional refits.' if fixed_field else
                'AGE_B16/AGE_B84 rescale the age coordinate and search bounds at alternative fields; boundary/weak-curvature flags persist. They are not fresh fits at the original bounded grid or lower/upper confidence limits. Common field/alpha and same-radio-data covariance are not jointly sampled by these profiles.'),
            model_comparison_scope='Both models condition on the same supplied field and injection index. A TJP-derived injection index is a dependent input; no independent model-selection probability is claimed.',
            input_configuration_sha256=radio.metadata['configuration_sha256'], input_files=radio.metadata['input_files'])
        complete = bool(alpha_cases is not None and np.any(valid) and
                        np.all(scenario_ok[radio.maps['SOURCE_VALID'].astype(bool)]))
        descriptive = spatial_summary(valid)
        fitted['metadata'].update(
            alpha_sensitivity={'input': sensitivity, 'cases': alpha_summaries,
                'scope': 'conditional refits at two index values with fixed field; not a marginalized interval or independent model validation'},
            scenario_pass_pixels=int(np.sum(scenario_ok)),
            whole_detected_source_summary={
                'status': 'conditional_scenario_supported' if complete else 'withheld',
                'mean_myr': descriptive['mean'] if complete else None,
                'median_myr': descriptive['median'] if complete else None,
                'diagnostic_best_fit_mean_myr': descriptive['mean'],
                'diagnostic_best_fit_median_myr': descriptive['median'],
                'reason': 'Requires every detected source pixel to have a closed, adequate, curvature-supported profile at the nominal index and both sensitivity endpoints; passing is conditional and does not determine a global lifetime.'},
            scenario_mask_interpretation='AGE_SCENARIO_OK requires AGE_CONSTRAINED at the nominal index and both checked endpoints. It does not assess intermediate indices, joint field/index uncertainty, coverage, or physical model systematics.')
        outputs[model] = fitted
    return outputs


def write_resolved_products(results, radio, folder):
    """Write FITS, finite metadata and large-font figures into an empty stage."""
    from astropy.io import fits
    from .plotting import publication_figure, save_publication_figure, AXIS_LABEL_SIZE, TICK_SIZE
    folder = Path(folder)
    for model in results:
        if model not in {'jp_tribble', 'ci'}:
            raise MapInputError('resolved products require jp_tribble or ci')
        for suffix in ('.fits', '.json', '.png', '.pdf', '.svg'):
            path = (folder/('resolved_'+model+suffix)).resolve()
            if path in radio.input_paths:
                raise MapInputError('resolved products must not overwrite inputs')
            if path.exists():
                raise FileExistsError(f'{path} exists; choose an empty analysis stage')
    summaries = {}
    for model, fitted in results.items():
        maps, metadata = fitted['maps'], fitted['metadata']
        clock = 'CI injection duration' if model == 'ci' else 'JP–Tribble radiative age'
        stem = 'resolved_'+model
        header = radio.reference.header.copy()
        header['BUNIT'] = 'Myr'; header['AGEKIND'] = 'CI_DURATION' if model == 'ci' else 'RADIATIVE'
        header['SPECMOD'] = 'CI' if model == 'ci' else 'TJP_FREE'
        header['BCONV'] = 'UNIFORM' if model == 'ci' else 'RMS'
        header['B_UG'] = metadata['field_ug']; header['ALPHAINJ'] = metadata['alpha_inj']
        if metadata.get('field_input', {}).get('kind') == 'user_fixed':
            header['BINPUT'] = 'USERFIX'
            header['BFIX_T'] = metadata['field_input']['value_t']
            header['BFIX_UG'] = metadata['field_ug']
        else:
            header['BFRACEQ'] = metadata.get('field_fraction', .4)
        header['COREPOL'] = 'INCLUDE'; header['GLOBAGE'] = 'UNAVAILABLE'
        hdus = [fits.PrimaryHDU(maps['SPECTRAL_AGE'], header=header)]
        for name, values in maps.items():
            if name == 'SPECTRAL_AGE':
                continue
            current = radio.reference.header.copy()
            current['BUNIT'] = 'Jy/beam' if name == 'REF_INTENSITY' else 'Myr' if name in {'AGE_LO68', 'AGE_HI68', 'AGE_UPPER95', 'AGE_B16', 'AGE_B84', 'AGE_ALOW', 'AGE_AHIGH'} else '1'
            hdus.append(fits.ImageHDU(values, header=current, name=name))
        encoded = json.dumps(metadata, allow_nan=False).encode('utf8')
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name='JSON_UTF8', format=f'{len(encoded)}B', array=np.frombuffer(encoded, dtype=np.uint8)[None])], name='PROVENANCE'))
        atomic_write_fits(folder/(stem+'.fits'), fits.HDUList(hdus))
        (folder/(stem+'.json')).write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        figure = publication_figure(figsize=(14, 12))
        panels = [(maps['SPECTRAL_AGE'], clock+' (best fit)', 'Myr', 'viridis'),
                  (maps['AGE_UPPER95'], 'Conditional 95% upper limit', 'Myr', 'magma'),
                  (maps['REDCHI2'], 'Reduced chi-square', '', 'cividis'),
                  (maps['AGE_FLAGS'], 'Age-profile quality flags', 'bit mask', 'tab20')]
        for index, (values, title, unit, cmap) in enumerate(panels, 1):
            axis = figure.add_subplot(2, 2, index, projection=radio.reference.wcs)
            displayed = np.where(maps['SPEC_VALID'], values, np.nan)
            shown = axis.imshow(displayed, origin='lower', cmap=cmap, interpolation='nearest')
            axis.set_title(title)
            for ci, label in enumerate(['Right ascension', 'Declination']):
                axis.coords[ci].set_axislabel(label, fontsize=AXIS_LABEL_SIZE)
                axis.coords[ci].set_ticklabel(size=TICK_SIZE, exclude_overlapping=True)
                axis.coords[ci].set_ticks(number=4)
            if np.any(np.isfinite(displayed)):
                colorbar = figure.colorbar(shown, ax=axis, shrink=.8, pad=.025)
                colorbar.set_label(unit, fontsize=AXIS_LABEL_SIZE)
            else:
                axis.text(.5, .5, 'No finite limit inside\nthe age search range', transform=axis.transAxes,
                          ha='center', va='center')
        figure.suptitle(f"{radio.metadata['source']['name']}\n{clock}; full detected source including core\nFixed field/index profiles; inspect flags and open/boundary limits", fontsize=21)
        save_publication_figure(figure, folder/(stem+'.png'), clear=True)
        summary = dict(metadata)
        summary['products'] = {key: stem+'.'+extension for key, extension in [('fits', 'fits'), ('metadata', 'json'), ('preview', 'png'), ('pdf', 'pdf'), ('svg', 'svg')]}
        summaries[model] = summary
    return summaries
