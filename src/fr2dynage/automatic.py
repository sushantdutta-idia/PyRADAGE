"""Image/redshift-to-publication workflow, with auditable physical limits.

No source class, core coordinate, speed, flow path or AGN lifetime is guessed.
The automatic product is a full-source radiative-clock comparison. A separate
dynamical model/constraint remains necessary for a dynamical-age measurement.
"""
from datetime import datetime, timezone
import csv
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .maps_io import MapInputError
from .reliability import sha256_file, verify_input_records, verify_analysis_manifest


def _rebase(value,old,new):
    if isinstance(value,dict):return {key:_rebase(item,old,new) for key,item in value.items()}
    if isinstance(value,list):return [_rebase(item,old,new) for item in value]
    if isinstance(value,str) and (value==str(old) or value.startswith(str(old)+os.sep)):
        return str(new)+value[len(str(old)):]
    return value


def _write_fits(path,reference,result):
    from astropy.io import fits
    from .maps_io import atomic_write_fits
    maps,metadata=result['maps'],result['metadata']
    header=reference.header.copy()
    header['BUNIT']='Myr';header['AGEKIND']='RADIATIVE' if metadata['model']=='jp_tribble' else 'CI_DURATION'
    header['SPECMOD']='TJP_FREE' if metadata['model']=='jp_tribble' else 'CI'
    header['B_UG']=metadata['field_ug'];header['ALPHAINJ']=metadata['alpha_inj_fixed']
    header['COREPOL']='INCLUDE';header['GLOBAGE']='UNAVAILABLE'
    if metadata['field_input']['prescription']=='user_fixed':
        header['BINPUT']='USERFIX';header['BFIX_T']=metadata['field_ug']*1e-10
    else:
        header['BINPUT']='PYRATS';header['BFRACEQ']=.4
    hdus=[fits.PrimaryHDU(maps['SPECTRAL_AGE'],header=header)]
    for name,data in maps.items():
        if name=='SPECTRAL_AGE':continue
        current=reference.header.copy()
        current['BUNIT']=('Myr' if name.startswith('AGE_') and name not in {'AGE_FLAGS','AGE_CONSTRAINED','AGE_SCENARIO_OK'}
            else 'Jy/beam' if name=='REF_INTENSITY' else '1')
        hdus.append(fits.ImageHDU(data,header=current,name=name))
    encoded=json.dumps(metadata,allow_nan=False).encode('utf8')
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name='JSON_UTF8',
        format=f'{len(encoded)}B',array=np.frombuffer(encoded,dtype=np.uint8)[None])],name='PROVENANCE'))
    atomic_write_fits(path,fits.HDUList(hdus))


def run_automatic(images, redshift, output_dir, *, field_t=None,pyrats_script=None,
                  pyrats_python=None,frequencies_mhz=None,progress=None,cancel_event=None,
                  calibration_fraction=.05,seed=42,max_regions=32,field_mc_samples=128):
    """Prepare cropped FITS images and publish one combined PNG/PDF summary.

    Only images/redshift are normal runtime inputs after dependency setup.
    Defaults (Planck18, kappa=100, filling=1, gamma=10..1e7, assumed 50%
    volume scatter) are explicit model assumptions, never measurements.
    """
    from . import __version__
    from .image_preparation import prepare_images
    from .automatic_field import compute_automatic_field,find_pyrats
    from .automatic_spectra import fit_shared_injection
    from .advanced_spectra import model_metadata
    from .resolved_spectra import fit_resolved_spectra
    from .radio_spectra import jp_loss_coefficient
    from .automatic_summary import render_automatic_summary
    def notify(message,fraction=None):
        if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Analysis cancelled')
        if progress is not None:progress(str(message),fraction)
        if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Analysis cancelled')
    if (isinstance(redshift,bool) or not isinstance(redshift,Real) or
        not math.isfinite(redshift) or redshift<=0):
        raise MapInputError('Redshift must be a finite positive number')
    if field_t is not None and (isinstance(field_t,bool) or not isinstance(field_t,Real) or
        not math.isfinite(field_t) or not 1e-12<=field_t<=1e-7):
        raise MapInputError('Fixed field must be within 1e-12..1e-7 T')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise MapInputError('Seed must be an integer in [0,2**32)')
    if isinstance(max_regions,bool) or not isinstance(max_regions,int) or not 1<=max_regions<=64:
        raise MapInputError('max_regions must lie in 1..64')
    if isinstance(field_mc_samples,bool) or not isinstance(field_mc_samples,int) or not 16<=field_mc_samples<=1000:
        raise MapInputError('field_mc_samples must lie in 16..1000')
    paths=[Path(path).expanduser().resolve() for path in images]
    if not 4<=len(paths)<=64 or len(set(paths))!=len(paths):
        raise MapInputError('Select 4–64 distinct multi-frequency FITS images')
    target=Path(output_dir).expanduser().resolve()
    if target.exists():raise FileExistsError(f'{target} exists; choose a new output directory')
    if any(path.is_relative_to(target) for path in paths):
        raise MapInputError('Output directory must not contain or replace input images')
    if field_t is None:find_pyrats(pyrats_script,pyrats_python)
    originals=[{'path':str(path),'sha256':sha256_file(path)} for path in paths]
    notify('Checking images and preparing a common beam/grid',.02)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'.',dir=target.parent))
    try:
        prepared=prepare_images(paths,float(redshift),stage/'prepared',frequencies_mhz=frequencies_mhz,
            calibration_fraction=calibration_fraction,progress=lambda text:notify(text,.1))
        consumed=[*originals,*prepared.image_provenance]
        verify_input_records(consumed)
        shared={}
        for index,model in enumerate(['jp_tribble','ci']):
            notify(f'Fitting the shared injection spectrum ({model})',.2+.1*index)
            shared[model]=fit_shared_injection(prepared,float(redshift),model=model,
                field_ug=3.,max_regions=max_regions,cancel_event=cancel_event)
        ranked=sorted(shared,key=lambda model:shared[model]['chi2'])
        difference=float(shared[ranked[1]]['chi2']-shared[ranked[0]]['chi2'])
        comparison={'ranked_models':ranked,'delta_chi2':difference,
            'preferred_model':ranked[0] if difference>=9. and shared[ranked[0]]['reduced_chi2']<=4. else None,
            'scope':'Descriptive regional fit ranking at equal parameter count, not a model probability or permission to mix different physical clocks'}
        notify('Estimating the source field and nominal energy budget',.4)
        field=compute_automatic_field(prepared,float(redshift),shared['jp_tribble']['alpha_inj'],stage/'field',
            field_t=field_t,pyrats_script=pyrats_script,pyrats_python=pyrats_python,
            cancel_event=cancel_event,seed=seed,mc_samples=field_mc_samples)
        b=float(field['adopted_field_t'])*1e10
        results={}
        for model_index,model in enumerate(['jp_tribble','ci']):
            info=shared[model];alpha=info['alpha_inj']
            def fit_at(index):
                return fit_resolved_spectra(prepared.frequencies_mhz,prepared.flux_cube,
                    prepared.rms_cube,prepared.calibration_fractions,prepared.calibration_correlation,
                    prepared.source_mask,b,float(redshift),index,model=model,
                    age_bounds_myr=(0.,1000.),age_grid_size=241,cancel_event=cancel_event)
            notify(f'Fitting all detected source pixels ({model})',.5+.17*model_index)
            fitted=fit_at(alpha)
            scenario=fitted['maps']['AGE_CONSTRAINED'].astype(bool).copy()
            sensitivity={}
            for index,label in zip(info['alpha_sensitivity'],['ALOW','AHIGH']):
                varied=fitted if index==alpha else fit_at(index)
                fitted['maps']['AGE_'+label]=varied['maps']['SPECTRAL_AGE'].copy()
                fitted['maps']['FLAG_'+label]=varied['maps']['AGE_FLAGS'].copy()
                scenario &= varied['maps']['AGE_CONSTRAINED'].astype(bool)
                sensitivity[label]={'alpha_inj':index,'constrained_pixels':varied['metadata']['constrained_age_pixels']}
            fitted['maps']['AGE_SCENARIO_OK']=scenario.astype(np.uint8)
            fitted['maps']['SOURCE_MASK']=prepared.source_mask.astype(np.uint8)
            valid=fitted['maps']['SPEC_VALID'].astype(bool)
            values=fitted['maps']['SPECTRAL_AGE'][valid]
            all_ok=bool(np.any(valid) and int(valid.sum())==fitted['metadata']['detected_candidate_pixels']
                        and np.all(scenario[valid]) and not info['flags'])
            stats={'fitted_pixels':int(valid.sum()),'constrained_pixels':int(scenario.sum()),
                'diagnostic_best_fit_mean_myr':float(np.mean(values)) if len(values) else None,
                'diagnostic_best_fit_median_myr':float(np.median(values)) if len(values) else None,
                'supported_detected_source_mean_myr':float(np.mean(values)) if all_ok else None,
                'supported_detected_source_median_myr':float(np.median(values)) if all_ok else None,
                'scope':'Conditional spectral-map statistics, not independent-pixel errors or a global dynamical lifetime'}
            field_sensitivity={}
            if field_t is None and field.get('field_mc'):
                for key,label in [('B_eq_p16_t','AGE_B16'),('B_eq_p84_t','AGE_B84')]:
                    alternative=.4*float(field['field_mc'][key])*1e10
                    factor=math.sqrt(alternative/b)*jp_loss_coefficient(b,redshift)/jp_loss_coefficient(alternative,redshift)
                    fitted['maps'][label]=fitted['maps']['SPECTRAL_AGE']*factor
                    field_sensitivity[label]={'field_ug':alternative,'age_coordinate_scale':factor,
                        'scope':'Common-field sensitivity including rescaled search bounds; not a combined confidence interval'}
            fitted['metadata'].update(software_version=__version__,field_input=field,
                model_description=model_metadata(model),
                shared_injection_fit=info,alpha_sensitivity=sensitivity,field_sensitivity=field_sensitivity,
                map_statistics=stats,core_treatment='All detected source emission is fitted; no core is inferred, removed or assigned zero across an aperture',
                source_mask_pixels=int(prepared.source_mask.sum()),input_files=originals)
            results[model]=fitted
        notify('Writing publication maps and source report',.87)
        metadata={'schema_version':'fr2dynage-automatic-analysis-1','software_name':'PyRADAGE','software_version':__version__,
            'created_utc':datetime.now(timezone.utc).isoformat(),'redshift':float(redshift),
            'input_files':originals,'preparation':_rebase(prepared.metadata,stage,target),
            'field':field,'shared_spectral_fits':shared,
            'spectral_model_ranking':comparison,
            'map_statistics':{model:fit['metadata']['map_statistics'] for model,fit in results.items()},
            'global_dynamical_age':{'status':'not_identifiable_from_images_and_redshift','mean_myr':None,'median_myr':None,
                'reason':'No expansion/transport speed, physical trajectory or dynamical evolution model is inferred by this workflow'},
            'model_comparison':'JP–Tribble and CI clocks are kept separate. Lowest numerical fit statistic does not justify mixing elapsed times and injection durations into one age map.',
            'assumptions':{'field_default':'B=0.4 B_eq from external PyRATS unless fixed field explicitly supplied',
                'particle_content_kappa':100.,'filling_factor':1.,'gamma_bounds':[10.,1e7],
                'volume_fractional_error_assumption':prepared.geometry.get('volume_fractional_error',.5),'cosmology':'Planck18',
                'alpha':'automatic shared-index spectral fit; not an independent field/injection measurement'},
            'warnings':prepared.metadata.get('warnings',[])+[
                'Images must be crops of one physical source; automatic segmentation cannot identify source membership in a crowded field.',
                'Matched WCS/beams do not establish matched uv coverage, missing-flux recovery, calibration accuracy or absence of variability.',
                'Radiative histories, reacceleration, absorption and unresolved population mixing are not reconstructed.',
                'CI assumes continuing injection and JP–Tribble a declared single-injection loss history; a successful fit does not identify the true activity history or radio morphology.',
                'Field/injection/source geometry reuse the same radio data; uncertainty summaries do not represent a full joint physical posterior.']}
        for model,result in results.items():
            _write_fits(stage/f'resolved_{model}.fits',prepared.reference,result)
            (stage/f'resolved_{model}.json').write_text(json.dumps(result['metadata'],indent=2,allow_nan=False)+'\n')
        render_automatic_summary(prepared.reference,prepared.flux_cube[0],results,
            stage/'age_summary.png',source_mask=prepared.source_mask)
        (stage/'analysis.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        from astropy.cosmology import Planck18
        dl=float(Planck18.luminosity_distance(float(redshift)).to_value('m'))
        sed_rows=[]
        for row in prepared.metadata['integrated_sed']:
            sed_rows.append({**row,'emitted_frequency_mhz':(1+redshift)*row['frequency_mhz'],
                'luminosity_at_emitted_frequency_w_hz':4*math.pi*dl**2*row['flux_jy']*1e-26/(1+redshift)})
        with (stage/'integrated_spectrum.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(sed_rows[0]));writer.writeheader();writer.writerows(sed_rows)
        with (stage/'age_statistics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['model',*next(iter(metadata['map_statistics'].values())).keys()])
            writer.writeheader()
            for model,statistics in metadata['map_statistics'].items():writer.writerow({'model':model,**statistics})
        report=['# Automatic radio-source analysis','',
            '[Combined publication PDF](age_summary.pdf) · [PNG](age_summary.png)','',
            '**This is a conditional radiative-age analysis. A global dynamical source age is not identified from images and redshift alone.**','',
            f"Redshift: {redshift:g}. Adopted field: {b:.6g} μG ({field['prescription']}).",'',
            '| Clock | Fitted pixels | Profile/scenario pass | Descriptive mean | Descriptive median |',
            '|---|---:|---:|---:|---:|']
        for model,stats in metadata['map_statistics'].items():
            report.append(f"| {model} | {stats['fitted_pixels']} | {stats['constrained_pixels']} | {stats['diagnostic_best_fit_mean_myr']} Myr | {stats['diagnostic_best_fit_median_myr']} Myr |")
        report+=['','| Shared spectral model | Injection index | Reduced χ² | Flags |',
            '|---|---:|---:|---|']
        for model,info in shared.items():
            report.append(f"| {model} | {info['alpha_inj']:.5g} | {info['reduced_chi2']:.4g} | {', '.join(info['flags']) or 'No listed flag'} |")
        report+=['',f"Descriptive spectral preference: {comparison['preferred_model'] or 'not established / all candidates inadequate'}. The two physical clocks remain separate.",'',
            '## Conditional source geometry and energetics','',
            '| Quantity | Nominal value |','|---|---:|',
            f"| Largest projected extent | {prepared.geometry['largest_linear_size_kpc']:.6g} kpc |",
            f"| Assumed emitting volume | {prepared.geometry['volume_m3']:.6g} m³ |",
            f"| Volume estimator | {prepared.geometry['volume_estimator']} |",
            f"| Assumed/model-spread fractional volume scatter | {prepared.geometry['volume_fractional_error']:.5g} |"]
        if field.get('adopted_budget'):
            budget=field['adopted_budget']
            for key in ['total_energy_J','pressure_Pa','electron_energy_J','particle_energy_J','magnetic_energy_J','electron_number_density_m3','electron_number_total']:
                if key in budget:report.append(f"| {key.replace('_',' ')} | {budget[key]:.6g} |")
            report+=['','These are nominal unaged homogeneous power-law budgets normalized at the adopted field and shared TJP injection index, with κ=100, filling factor1 and γ=10–10⁷. They are not aged TJP/CI energy refits, mechanical jet power or external gas pressure. The geometric depth and scatter are assumptions/model spread, not measured 3-D confidence limits.']
        else:
            report+=['','Energetics and equipartition properties are unavailable because no external PyRATS backend was available; the supplied field alone supports conditional radiative-clock fitting.']
        report+=['','[Integrated flux/error/luminosity table](integrated_spectrum.csv) retains negative fluxes/nondetections. Luminosity is quoted at emitted frequency (1+z)ν_obs using Lν=4πD_L²Sν/(1+z), with Planck18 and no invented same-frequency spectral K correction; see [Hogg (1999), Eq.21](https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html).','']
        report+=['','JP–Tribble values describe single-injection radiative time; CI values describe continuous-injection duration. They are not mixed or renamed as a global dynamical lifetime. Best-fit maps remain visible with flags when ages are unconstrained. Inspect FITS AGE_FLAGS, AGE_CONSTRAINED, AGE_SCENARIO_OK, profile limits and model adequacy.','',
            'Only images and redshift are required after backend setup. Calibration, particle content, electron limits, filling factor, line-of-sight depth and field fraction use documented defaults. They are assumptions, not automatically measured source properties. No source core is deleted or assigned a blanket zero age.','',
            '[Full metadata and assumptions](analysis.json) · [Age statistics](age_statistics.csv) · [Field/backend calculation](field/field.json)','']
        report+=['- '+warning for warning in metadata['warnings']]
        (stage/'REPORT.md').write_text('\n'.join(report)+'\n')
        # Replace only internal staging paths in portable JSON provenance.
        for path in stage.rglob('*.json'):
            data=json.loads(path.read_text());path.write_text(json.dumps(_rebase(data,stage,target),indent=2,allow_nan=False)+'\n')
        verify_input_records(consumed)
        files=sorted(path for path in stage.rglob('*') if path.is_file())
        (stage/'run_manifest.json').write_text(json.dumps({'schema_version':'fr2dynage-analysis-manifest-2',
            'files':[str(path.relative_to(stage)) for path in files],
            'sha256':{str(path.relative_to(stage)):sha256_file(path) for path in files},
            'input_sha256':{row['path']:row['sha256'] for row in originals},
            'scope':'Automatic products; input hashes verified before publication; manifest excluded'},indent=2)+'\n')
        notify('Publishing the completed analysis',.98)
        verify_input_records(consumed)
        if target.exists():raise FileExistsError(f'{target} appeared during analysis; left untouched')
        os.rename(stage,target)
    finally:
        if stage.exists():shutil.rmtree(stage)
    integrity=verify_analysis_manifest(target,check_inputs=True)
    # Do not raise a late cancellation after a successfully published result.
    if progress is not None:progress('Completed: combined PDF/PNG and scientific products are ready',1.)
    return {'output_dir':str(target),'summary_pdf':str(target/'age_summary.pdf'),
        'summary_png':str(target/'age_summary.png'),'report':str(target/'REPORT.md'),
        'analysis':str(target/'analysis.json'),'integrity':integrity}
