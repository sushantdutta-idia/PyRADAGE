"""The GUI's state and worker can be checked without a display or Tk."""
from pathlib import Path
import threading

import pytest

from fr2dynage.gui import RunController, default_output_dir, validate_request, open_path


@pytest.fixture
def images(tmp_path):
    result=[]
    for index in range(4):
        path=tmp_path/f"image{index}.fits";path.write_bytes(b"synthetic picker fixture")
        result.append(path)
    return result


def test_request_has_only_two_required_science_inputs(images,tmp_path):
    request=validate_request(images,"0.1469",tmp_path/"results")
    assert request.images==tuple(images)
    assert request.redshift==.1469
    assert request.field_t is None
    assert request.pyrats_script is None
    assert request.pyrats_python is None
    assert not request.output_dir.exists()


@pytest.mark.parametrize("redshift",["",True,"nan","inf",0,-.1,"0.1 abc"])
def test_bad_redshift_rejected_before_a_worker(images,tmp_path,redshift):
    with pytest.raises(ValueError,match="redshift"):
        validate_request(images,redshift,tmp_path/"results")


def test_duplicate_missing_or_insufficient_images_rejected(images,tmp_path):
    with pytest.raises(ValueError,match="4–64"):
        validate_request(images[:3],.1,tmp_path/"results")
    with pytest.raises(ValueError,match="only once"):
        validate_request(images[:3]+[images[0]],.1,tmp_path/"results")
    with pytest.raises(ValueError,match="file"):
        validate_request(images[:3]+[tmp_path/"missing.fits"],.1,tmp_path/"results")


def test_output_cannot_contain_inputs_or_replace_existing_results(images,tmp_path):
    with pytest.raises(ValueError,match="separate"):
        validate_request(images,.1,tmp_path)
    target=tmp_path/"results";target.mkdir();(target/"keep.txt").write_text("preserve")
    with pytest.raises(ValueError,match="already exists"):
        validate_request(images,.1,target)
    assert (target/"keep.txt").read_text()=="preserve"


def test_even_an_empty_output_directory_must_be_new(images,tmp_path):
    target=tmp_path/"empty";target.mkdir()
    with pytest.raises(ValueError,match="already exists"):
        validate_request(images,.1,target)


@pytest.mark.parametrize("field",[True,0,"nan","a field",1e-13,1e-6])
def test_optional_field_must_be_explicit_valid_tesla(images,tmp_path,field):
    with pytest.raises(ValueError,match="field"):
        validate_request(images,.1,tmp_path/"results",field_t=field)


def test_fixed_field_and_backend_are_passed_without_shell_parsing(images,tmp_path):
    script=tmp_path/"PyRATS.py";script.write_text("# explicit installed backend")
    request=validate_request(images,.1,tmp_path/"results",field_t="2.24e-10",
        pyrats_script=script,pyrats_python="/path with spaces/python")
    assert request.field_t==2.24e-10
    assert request.pyrats_script==script
    assert request.pyrats_python=="/path with spaces/python"


def test_optional_frequencies_can_repair_missing_fits_metadata(images,tmp_path):
    request=validate_request(images,.1,tmp_path/"results",frequencies_mhz="150, 350, 1400, 3000")
    assert request.frequencies_mhz==(150.,350.,1400.,3000.)


@pytest.mark.parametrize("frequencies",["150 350", "150 350 1400 nan", "150 350 1400 150", [150,350,1400,True]])
def test_bad_frequency_overrides_rejected(images,tmp_path,frequencies):
    with pytest.raises(ValueError,match="frequenc|Frequenc"):
        validate_request(images,.1,tmp_path/"results",frequencies_mhz=frequencies)


def test_default_output_is_a_new_child(tmp_path):
    first=default_output_dir(tmp_path);second=default_output_dir(tmp_path)
    assert first.parent==tmp_path and second.parent==tmp_path
    assert first!=second
    assert not first.exists()


def test_worker_is_off_ui_thread_with_queued_progress_and_manifest(images,tmp_path):
    caller=threading.get_ident();calls=[]
    def runner(paths,z,target,**kwargs):
        calls.append((threading.get_ident(),paths,z,target,kwargs))
        kwargs["progress"]("Checking FITS headers",.25)
        return {"summary_pdf":"age_summary.pdf","summary_png":"age_summary.png"}
    request=validate_request(images,.1,tmp_path/"results")
    controller=RunController(runner)
    run_id=controller.start(request);controller.join(2)
    events=controller.drain_events()
    assert not controller.busy
    assert calls[0][0]!=caller
    assert calls[0][1]==images
    assert calls[0][4]["field_t"] is None
    assert [(event.kind,event.run_id) for event in events]==[("progress",run_id),("complete",run_id)]
    assert events[0].payload=={"message":"Checking FITS headers","fraction":.25}
    assert events[1].payload["summary_pdf"]=="age_summary.pdf"


def test_double_start_is_rejected_and_cancel_reaches_runner(images,tmp_path):
    entered=threading.Event();release=threading.Event()
    def runner(paths,z,target,**kwargs):
        entered.set();release.wait(2)
        if kwargs["cancel_event"].is_set():raise RuntimeError("Cancelled between stages")
        return {}
    controller=RunController(runner);request=validate_request(images,.1,tmp_path/"results")
    controller.start(request);assert entered.wait(1)
    with pytest.raises(RuntimeError,match="already running"):
        controller.start(request)
    assert controller.cancel()
    release.set();controller.join(2)
    assert [event.kind for event in controller.drain_events()]==["cancelled"]
    assert not controller.busy
    assert not controller.cancel()


def test_worker_exception_remains_visible_and_future_run_can_start(images,tmp_path):
    def runner(*args,**kwargs):raise ValueError("Missing observing frequency in FITS metadata")
    controller=RunController(runner);request=validate_request(images,.1,tmp_path/"results")
    first=controller.start(request);controller.join(2)
    event=controller.drain_events()[0]
    assert event.kind=="error" and "observing frequency" in event.payload["message"]
    assert "ValueError" in event.payload["traceback"]
    second=controller.start(request);controller.join(2)
    assert second>first


def test_progress_bad_fraction_does_not_break_analysis(images,tmp_path):
    def runner(*args,**kwargs):
        kwargs["progress"]("Preparing",float("nan"))
        return {}
    controller=RunController(runner)
    controller.start(validate_request(images,.1,tmp_path/"results"));controller.join(2)
    assert controller.drain_events()[0].payload["fraction"] is None


def test_os_open_uses_an_argument_vector_not_a_shell(tmp_path,monkeypatch):
    import fr2dynage.gui as gui
    directory=tmp_path/"a folder; not a command";directory.mkdir()
    calls=[]
    monkeypatch.setattr(gui.sys,"platform","darwin")
    monkeypatch.setattr(gui.subprocess,"Popen",lambda args,**kwargs:calls.append((args,kwargs)))
    open_path(directory)
    assert calls[0][0]==["open",str(directory)]
    assert "shell" not in calls[0][1]


def test_missing_tk_gives_an_actionable_cli_error(monkeypatch,capsys):
    import sys
    from fr2dynage.gui import main
    monkeypatch.setitem(sys.modules,"tkinter",None)
    assert main([])==2
    message=capsys.readouterr().err
    assert "Tk-enabled Python" in message and "headless" in message


def test_display_failure_does_not_dump_a_long_tcl_stack(monkeypatch,capsys):
    import sys
    from types import SimpleNamespace
    from fr2dynage.gui import main
    def missing_display():raise RuntimeError("No graphical display\nprivate library details\nlong Tcl trace")
    monkeypatch.setitem(sys.modules,"tkinter",SimpleNamespace(Tk=missing_display))
    assert main([])==2
    message=capsys.readouterr().err
    assert "No graphical display" in message
    assert "private library details" not in message
