"""Portable, isolated field-backend orchestration without a real PyRATS install.

The temporary backend below is an explicitly synthetic accounting fixture,
not a validation of PySynch emissivity or observed-source energetics.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("astropy")

from fr2dynage import automatic_field as field
from fr2dynage.maps_io import MapInputError


@pytest.fixture
def prepared():
    frequencies=np.array([150.,350.,1400.,3000.])
    return SimpleNamespace(frequencies_mhz=frequencies,
        geometry={"volume_m3":1e65,"volume_fractional_error":.3},
        metadata={"integrated_sed":[{"flux_jy":float(.1*(nu/150.)**-.65),
            "total_error_jy":float(.005*(nu/150.)**-.65)} for nu in frequencies]})


@pytest.fixture
def synthetic_backend(tmp_path):
    path=tmp_path/"synthetic_pyrats.py"
    path.write_text('''"""Synthetic backend interface fixture; no real synchrotron solver."""
import math
__version__='2.1.0-test'
MU0=1.25663706127e-6
def pysynch_normalise(volume,alpha,z,frequency,flux,zeta,gmin,gmax,**kwargs):
    b=6e-10*((flux/.1)/(volume/1e65))**.25
    ub=b*b/(2*MU0);ue=ub/zeta;up=zeta*ue;ut=up+ub
    return dict(B_T=b,B_uG=b/1e-10,volume_m3=volume,
      electron_energy_density_J_m3=ue,particle_energy_density_J_m3=up,
      magnetic_energy_density_J_m3=ub,total_energy_density_J_m3=ut,
      total_energy_J=volume*ut,pressure_Pa=ut/3.)
def sub_equipartition_budget(fraction,beq,volume,alpha,z,frequency,flux,zeta,gmin,gmax,**kwargs):
    b=fraction*beq;ubeq=beq*beq/(2*MU0)
    ue=ubeq/zeta*fraction**(-(1-alpha));up=zeta*ue;ub=b*b/(2*MU0);ut=up+ub
    return dict(fraction_of_B_eq=fraction,B_T=b,B_uG=b/1e-10,
      B_eq_T=beq,B_eq_uG=beq/1e-10,volume_m3=volume,
      electron_energy_density_J_m3=ue,particle_energy_density_J_m3=up,
      magnetic_energy_density_J_m3=ub,total_energy_density_J_m3=ut,
      total_energy_J=volume*ut,pressure_Pa=ut/3.)
''')
    return path


def no_backend(monkeypatch):
    monkeypatch.delenv("FR2DYNAGE_PYRATS_SCRIPT",raising=False)
    monkeypatch.delenv("FR2DYNAGE_PYRATS_PYTHON",raising=False)
    monkeypatch.setattr(field.shutil,"which",lambda value:None)
    monkeypatch.setattr(field.importlib.util,"find_spec",lambda name:None)


def test_missing_backend_never_guesses_a_field(monkeypatch):
    no_backend(monkeypatch)
    with pytest.raises(MapInputError,match="No field is guessed"):
        field.find_pyrats()


def test_fixed_field_without_backend_has_no_invented_energetics(prepared,tmp_path,monkeypatch):
    no_backend(monkeypatch)
    result=field.compute_automatic_field(prepared,.1469,.65,tmp_path/"field",field_t=2.24e-10)
    assert result["prescription"]=="user_fixed"
    assert result["adopted_field_t"]==2.24e-10
    assert result["B_eq_t"] is None
    assert result["field_mc"] is None
    assert result["adopted_budget"] is None
    assert result["backend_status"]=="not_available"
    assert json.loads((tmp_path/"field/field.json").read_text())==result


def test_explicit_bad_backend_is_not_silently_ignored(prepared,tmp_path):
    with pytest.raises(MapInputError,match="script not found"):
        field.compute_automatic_field(prepared,.1469,.65,tmp_path/"field",
            field_t=2.24e-10,pyrats_script=tmp_path/"missing.py",pyrats_python=sys.executable)


def test_backend_discovery_resolves_a_real_python_and_script(synthetic_backend):
    script,python=field.find_pyrats(synthetic_backend,sys.executable)
    assert script==synthetic_backend
    assert Path(python).is_file()


def test_default_worker_applies_point_four_and_records_uncertainty(prepared,tmp_path,synthetic_backend):
    result=field.compute_automatic_field(prepared,.1469,.65,tmp_path/"field",
        pyrats_script=synthetic_backend,pyrats_python=sys.executable,mc_samples=16)
    assert result["prescription"]=="B=0.4 B_eq"
    assert result["B_eq_t"]==pytest.approx(6e-10)
    assert result["adopted_field_t"]==pytest.approx(.4*result["B_eq_t"])
    assert result["field_mc"]["requested"]==16
    assert result["field_mc"]["success"]==16
    assert result["field_mc"]["B_eq_p16_t"]<result["field_mc"]["B_eq_median_t"]<result["field_mc"]["B_eq_p84_t"]
    assert result["assumptions"]["cosmology"]=="Planck18"
    assert result["adopted_budget"]["total_energy_J"]>0
    json.dumps(result,allow_nan=False)


def test_fixed_worker_preserves_the_requested_field_and_native_beq(prepared,tmp_path,synthetic_backend):
    result=field.compute_automatic_field(prepared,.1469,.65,tmp_path/"field",field_t=2.24e-10,
        pyrats_script=synthetic_backend,pyrats_python=sys.executable,mc_samples=16)
    assert result["prescription"]=="user_fixed"
    assert result["adopted_field_t"]==pytest.approx(2.24e-10,rel=1e-14)
    assert result["B_eq_t"]==pytest.approx(6e-10)
    assert result["B_eq_t"]!=pytest.approx(result["adopted_field_t"]/.4)
    assert result["adopted_budget"]["B_T"]==pytest.approx(2.24e-10)


def test_backend_failure_has_a_log_and_no_fabricated_field(prepared,tmp_path):
    backend=tmp_path/"failing_pyrats.py";backend.write_text("raise RuntimeError('synthetic missing dependency')\n")
    folder=tmp_path/"field"
    with pytest.raises(MapInputError,match="backend failed"):
        field.compute_automatic_field(prepared,.1469,.65,folder,
            pyrats_script=backend,pyrats_python=sys.executable,mc_samples=16)
    assert "synthetic missing dependency" in (folder/"backend.log").read_text()
    assert not (folder/"field.json").exists()


def test_cancellation_terminates_and_reaps_the_backend_process(prepared,tmp_path,monkeypatch):
    backend=tmp_path/"waiting_pyrats.py"
    backend.write_text("import time\ntime.sleep(60)\n")
    cancel=threading.Event();started=threading.Event();processes=[];errors=[]
    real_popen=field.subprocess.Popen
    def capture(*args,**kwargs):
        process=real_popen(*args,**kwargs);processes.append(process);started.set();return process
    monkeypatch.setattr(field.subprocess,"Popen",capture)
    def call():
        try:
            field.compute_automatic_field(prepared,.1469,.65,tmp_path/"field",
                pyrats_script=backend,pyrats_python=sys.executable,cancel_event=cancel,mc_samples=16)
        except Exception as exc:errors.append(exc)
    worker=threading.Thread(target=call,daemon=True);worker.start()
    assert started.wait(5)
    cancel.set();worker.join(8)
    assert not worker.is_alive()
    assert len(errors)==1 and isinstance(errors[0],InterruptedError)
    assert processes[0].poll() is not None
    assert not (tmp_path/"field/field.json").exists()


def test_field_output_folder_is_not_overwritten(prepared,tmp_path,synthetic_backend):
    folder=tmp_path/"field";folder.mkdir();(folder/"keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        field.compute_automatic_field(prepared,.1469,.65,folder,
            pyrats_script=synthetic_backend,pyrats_python=sys.executable,mc_samples=16)
    assert (folder/"keep.txt").read_text()=="preserve"


def wire_result():
    from fr2dynage.constants import MU_0
    volume,beq=1e65,6e-10
    def budget(b,ue):
        up=101*ue;ub=b*b/(2*MU_0);ut=up+ub
        return {'B_T':b,'B_uG':b*1e10,'volume_m3':volume,
            'electron_energy_density_J_m3':ue,'particle_energy_density_J_m3':up,
            'magnetic_energy_density_J_m3':ub,'total_energy_density_J_m3':ut,
            'total_energy_J':ut*volume,'pressure_Pa':ut/3}
    ue=beq*beq/(2*MU_0)/101
    adopted=budget(.4*beq,ue*.4**-1.65)
    adopted.update(B_eq_T=beq,fraction_of_B_eq=.4)
    request={'field_t':None,'script_sha256':'a'*64,'volume_m3':volume,
        'filling_factor':1.,'kappa':100.,'mc_samples':16}
    result={'schema_version':'fr2dynage-automatic-pyrats-1','B_eq_t':beq,
        'adopted_field_t':.4*beq,'prescription':'B=0.4 B_eq','pyrats_sha256':'a'*64,
        'native_equipartition':budget(beq,ue),'adopted_budget':adopted,
        'field_mc':{'requested':16,'success':16,'failed':0,
            'B_eq_p16_t':5e-10,'B_eq_median_t':6e-10,'B_eq_p84_t':7e-10}}
    return result,request


def test_validated_wire_result_preserves_physical_accounting():
    result,request=wire_result()
    assert field.validate_field_result(result,request) is result


@pytest.mark.parametrize('change',[
    lambda d:d.update(adopted_field_t=True),
    lambda d:d.update(adopted_field_t=1.),
    lambda d:d.update(adopted_field_t=2e-10),
    lambda d:d.update(B_eq_t=float('nan')),
    lambda d:d.update(prescription='user_fixed'),
    lambda d:d.update(pyrats_sha256='b'*64),
    lambda d:d['adopted_budget'].update(B_T=2.4e-9),
    lambda d:d['adopted_budget'].update(B_uG=240.),
    lambda d:d['adopted_budget'].update(total_energy_J=1.),
    lambda d:d['adopted_budget'].update(pressure_Pa=1.),
    lambda d:d['field_mc'].update(requested=17),
    lambda d:d['field_mc'].update(success=5,failed=11),
    lambda d:d['field_mc'].update(success=True),
    lambda d:d['field_mc'].update(B_eq_p16_t=8e-10),
    lambda d:d.pop('native_equipartition'),
])
def test_malformed_or_physically_inconsistent_backend_results_rejected(change):
    result,request=wire_result();change(result)
    with pytest.raises(MapInputError):
        field.validate_field_result(result,request)
