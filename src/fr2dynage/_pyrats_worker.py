"""Isolated adapter to an externally installed PyRATS; no vendored backend.

Executed by the Python environment that has PyRATS/pysynch dependencies. This
process uses only numeric request data and the explicitly selected Python
module. Reports and image-header prose are never executable instructions.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def main():
    request_path,output_path=map(Path,sys.argv[1:])
    request=json.loads(request_path.read_text())
    code=Path(request['script']).resolve()
    expected=request['script_sha256']
    if hashlib.sha256(code.read_bytes()).hexdigest()!=expected:
        raise ValueError('PyRATS script changed before backend import')
    spec=importlib.util.spec_from_file_location('fr2dynage_external_pyrats',code)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from astropy.cosmology import Planck18
    import numpy as np
    alpha=-float(request['alpha_inj']) # PyRATS uses S ~ nu**alpha.
    z=float(request['redshift']);volume=float(request['volume_m3'])
    frequency=float(request['frequency_hz']);flux=float(request['flux_jy'])
    kappa=float(request['kappa']);filling=float(request['filling_factor'])
    gmin,gmax=map(float,request['gamma_bounds'])
    effective_volume=volume*filling
    eq=module.pysynch_normalise(effective_volume,alpha,z,frequency,flux,1+kappa,
        gmin,gmax,method='equipartition',spectrum='powerlaw',cosmology=Planck18)
    beq=float(eq['B_T'])
    field=float(request['field_t']) if request.get('field_t') is not None else .4*beq
    adopted=module.sub_equipartition_budget(field/beq,beq,effective_volume,alpha,z,
        frequency,flux,1+kappa,gmin,gmax,spectrum='powerlaw',cosmology=Planck18)
    for component in ['electron','particle','magnetic']:
        adopted[component+'_energy_J']=adopted[component+'_energy_density_J_m3']*effective_volume
    if math.isfinite(float(eq.get('electron_number_density_m3',float('nan')))):
        adopted['electron_number_density_m3']=eq['electron_number_density_m3']*adopted['electron_energy_density_J_m3']/eq['electron_energy_density_J_m3']
        adopted['electron_number_total']=adopted['electron_number_density_m3']*effective_volume
    # Direct solves preserve finite-electron-cutoff effects. These quantiles
    # condition on alpha and the stated independent flux/volume log scatters.
    rng=np.random.default_rng(int(request['seed']))
    fields=[];failures=0
    flux_sigma=math.sqrt(math.log1p(float(request['flux_fractional_error'])**2))
    volume_sigma=math.sqrt(math.log1p(float(request['volume_fractional_error'])**2))
    for _ in range(int(request['mc_samples'])):
        try:
            trial=module.pysynch_normalise(effective_volume*math.exp(rng.normal(0,volume_sigma)),
                alpha,z,frequency,flux*math.exp(rng.normal(0,flux_sigma)),1+kappa,gmin,gmax,
                method='equipartition',spectrum='powerlaw',cosmology=Planck18)
            b=float(trial['B_T'])
            if not math.isfinite(b) or b<=0:raise ValueError('nonphysical field')
            fields.append(b)
        except (ArithmeticError,ValueError,RuntimeError):
            failures+=1
    if len(fields)<.8*int(request['mc_samples']):
        raise ValueError('Fewer than 80% of equipartition uncertainty solves succeeded')
    q=np.quantile(fields,[.16,.5,.84])
    if hashlib.sha256(code.read_bytes()).hexdigest()!=expected:
        raise ValueError('PyRATS script changed during backend calculation')
    result={'schema_version':'fr2dynage-automatic-pyrats-1',
        'pyrats_version':str(module.__version__),'pyrats_script':str(code),
        'pyrats_sha256':expected,
        'native_equipartition':eq,'adopted_budget':adopted,
        'B_eq_t':beq,'adopted_field_t':field,
        'prescription':'user_fixed' if request.get('field_t') is not None else 'B=0.4 B_eq',
        'field_mc':{'requested':request['mc_samples'],'success':len(fields),'failed':failures,
            'B_eq_p16_t':float(q[0]),'B_eq_median_t':float(q[1]),'B_eq_p84_t':float(q[2]),
            'scope':'Conditional independent lognormal flux and assumed volume scatter at fixed injection spectrum; not full joint radio/field inference or model-systematic coverage'},
        'assumptions':{'alpha_inj_positive':-alpha,'electron_spectrum':'unaged homogeneous powerlaw',
            'redshift':z,'volume_m3':volume,'filling_factor':filling,'kappa':kappa,
            'gamma_min':gmin,'gamma_max':gmax,'frequency_hz':frequency,'flux_jy':flux,
            'volume_fractional_error_assumption':request['volume_fractional_error'],
            'cosmology':'Planck18'},
        'scope':'External PyRATS/pysynch field and nominal power-law energetics; not an aged TJP/CI population or dynamical expansion model'}
    output_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
