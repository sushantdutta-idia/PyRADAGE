"""Portable discovery and isolated execution of the external field backend."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .maps_io import MapInputError
from .reliability import sha256_file


def validate_field_result(result,request):
    """Reject backend inconsistencies before a field enters spectral maps."""
    from .constants import MU_0
    def number(value,label):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
            raise MapInputError(f'Invalid PyRATS {label}; expected a finite positive number')
        return float(value)
    def close(actual,expected,label):
        actual=number(actual,label)
        if not math.isclose(actual,expected,rel_tol=1e-5,abs_tol=0.):
            raise MapInputError(f'Inconsistent PyRATS {label}')
    try:
        if result['schema_version']!='fr2dynage-automatic-pyrats-1':
            raise MapInputError('Unsupported automatic field-backend result schema')
        beq=number(result['B_eq_t'],'B_eq')
        b=number(result['adopted_field_t'],'adopted field')
        if not .01<=b*1e10<=1000.:
            raise MapInputError('Computed field is outside the supported 0.01–1000 microgauss range')
        expected=request['field_t'] if request['field_t'] is not None else .4*beq
        if not math.isclose(b,expected,rel_tol=1e-12,abs_tol=0.):
            raise MapInputError('Backend field does not match the fixed value or B=0.4 B_eq prescription')
        expected_kind='user_fixed' if request['field_t'] is not None else 'B=0.4 B_eq'
        if result['prescription']!=expected_kind:raise MapInputError('Backend field prescription was mislabelled')
        if result['pyrats_sha256']!=request['script_sha256']:
            raise MapInputError('Backend PyRATS code hash differs from the consumed script')
        volume=request['volume_m3']*request['filling_factor']
        for label,budget,field in [('native',result['native_equipartition'],beq),('adopted',result['adopted_budget'],b)]:
            close(budget['B_T'],field,label+' field T')
            close(budget['B_uG'],field*1e10,label+' field microgauss')
            if 'volume_m3' in budget:close(budget['volume_m3'],volume,label+' volume')
            ue=number(budget['electron_energy_density_J_m3'],label+' electron density')
            up=(1+request['kappa'])*ue;ub=field**2/(2*MU_0);ut=up+ub
            for key,value in [('particle_energy_density_J_m3',up),('magnetic_energy_density_J_m3',ub),
                    ('total_energy_density_J_m3',ut),('total_energy_J',ut*volume),('pressure_Pa',ut/3)]:
                close(budget[key],value,label+' '+key)
        close(result['adopted_budget']['B_eq_T'],beq,'adopted native B_eq comparison')
        close(result['adopted_budget']['fraction_of_B_eq'],b/beq,'adopted B/B_eq comparison')
        close(result['native_equipartition']['particle_energy_density_J_m3'],
              result['native_equipartition']['magnetic_energy_density_J_m3'],'native equipartition balance')
        mc=result['field_mc']
        counts=[mc[key] for key in ['requested','success','failed']]
        if any(isinstance(value,bool) or not isinstance(value,int) or value<0 for value in counts):
            raise MapInputError('Invalid backend Monte Carlo counts')
        n,success,failed=counts
        if not 16<=n<=1000 or n!=request['mc_samples'] or success+failed!=n or success<.8*n:
            raise MapInputError('Backend Monte Carlo count/success mismatch')
        lo,median,hi=[number(mc[key],key) for key in ['B_eq_p16_t','B_eq_median_t','B_eq_p84_t']]
        if not lo<=median<=hi:raise MapInputError('Backend field quantiles are unordered')
    except (KeyError,TypeError) as exc:
        raise MapInputError('Incomplete/malformed PyRATS backend result') from exc
    return result


def find_pyrats(script=None,python=None):
    """Return explicit/environment/PATH backend; never guess a user's home."""
    script=script or os.environ.get('PYRADAGE_PYRATS_SCRIPT') or os.environ.get('FR2DYNAGE_PYRATS_SCRIPT')
    python=python or os.environ.get('PYRADAGE_PYRATS_PYTHON') or os.environ.get('FR2DYNAGE_PYRATS_PYTHON') or sys.executable
    if script is None:
        script=shutil.which('PyRATS.py')
    if script is None:
        try:
            found=importlib.util.find_spec('PyRATS')
            if found is not None:script=found.origin
        except (ValueError,ImportError):
            pass
    if script is None:
        raise MapInputError('PyRATS backend not found. Complete the one-time PyRATS/pysynch setup, set PYRADAGE_PYRATS_SCRIPT and PYRADAGE_PYRATS_PYTHON, or supply an independently chosen fixed field in Advanced settings. No field is guessed.')
    script=Path(script).expanduser().resolve()
    if not script.is_file():raise MapInputError(f'PyRATS script not found: {script}')
    executable=shutil.which(str(python))
    if executable is None:raise MapInputError(f'PyRATS Python executable not found: {python}')
    return script,executable


def compute_automatic_field(prepared, redshift, alpha_inj, folder, *, field_t=None,
                            pyrats_script=None,pyrats_python=None,cancel_event=None,
                            seed=42,mc_samples=128):
    """Compute a nominal field and explicitly conditional uncertainty inputs."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    try:
        script,python=find_pyrats(pyrats_script,pyrats_python)
    except MapInputError as exc:
        if field_t is None or pyrats_script is not None or pyrats_python is not None:
            raise
        result={'prescription':'user_fixed','adopted_field_t':float(field_t),
            'B_eq_t':None,'field_mc':None,'adopted_budget':None,
            'scope':'Exact supplied field; no field uncertainty or source energetics inferred',
            'backend_status':'not_available','backend_note':str(exc)}
        (folder/'field.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    sed=prepared.metadata['integrated_sed']
    # Use the lowest-frequency usable flux as the normalization; retain its
    # true observed frequency. Frequency/volume share Planck18 throughout.
    candidates=[i for i,row in enumerate(sed)
                if row['flux_jy'] is not None and math.isfinite(row['flux_jy']) and row['flux_jy']>0]
    if not candidates:raise MapInputError('No positive integrated source flux for the field calculation')
    index=min(candidates,key=lambda i:prepared.frequencies_mhz[i])
    flux=float(sed[index]['flux_jy'])
    uncertainty=float(sed[index]['total_error_jy'])
    request={'script':str(script),'script_sha256':sha256_file(script),
        'alpha_inj':float(alpha_inj),'redshift':float(redshift),
        'volume_m3':float(prepared.geometry['volume_m3']),
        'frequency_hz':float(prepared.frequencies_mhz[index])*1e6,'flux_jy':flux,
        'flux_fractional_error':uncertainty/flux,
        'volume_fractional_error':float(prepared.geometry.get('volume_fractional_error',.5)),
        'kappa':100.,'filling_factor':1.,'gamma_bounds':[10.,1e7],
        'field_t':field_t,'seed':seed,'mc_samples':mc_samples}
    request_path=folder/'field_request.json';request_path.write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
    output=folder/'field.json';log=folder/'backend.log'
    worker=Path(__file__).with_name('_pyrats_worker.py')
    with log.open('wb') as stream:
        environment=dict(os.environ);environment['MPLBACKEND']='Agg'
        process=subprocess.Popen([python,str(worker),str(request_path),str(output)],
            stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment)
        started=time.monotonic()
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError('Analysis cancelled')
                if time.monotonic()-started>600:
                    raise MapInputError('PyRATS field calculation timed out; inspect the backend log')
                time.sleep(.1)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if process.returncode or not output.exists():
        detail=log.read_text(errors='replace')[-2500:]
        raise MapInputError('PyRATS backend failed. Its Python environment needs PyRATS, pysynch/GSL and the documented dependencies.\n'+detail)
    result=json.loads(output.read_text())
    result['backend_python']=str(python)
    validate_field_result(result,request)
    if sha256_file(script)!=request['script_sha256']:
        raise MapInputError('PyRATS script changed during field calculation')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
