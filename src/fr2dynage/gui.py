"""Native image/redshift wizard with a responsive, cancellable worker.

Tk is imported only when opening the interface. Validation and worker control
are intentionally display-independent so they can be tested on CI servers.
Scientific work belongs to :mod:`fr2dynage.automatic`, not the GUI thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
from typing import Callable


@dataclass(frozen=True)
class RunRequest:
    images: tuple[Path, ...]
    redshift: float
    output_dir: Path
    field_t: float | None = None
    pyrats_script: Path | None = None
    pyrats_python: str | None = None
    frequencies_mhz: tuple[float, ...] | None = None


@dataclass(frozen=True)
class RunEvent:
    run_id: int
    kind: str
    payload: object


def default_output_dir(parent: Path | None = None) -> Path:
    """Use a fresh result directory, never the input-image directory itself."""
    root = Path(parent).expanduser() if parent is not None else Path.home()/"PyRADAGE-results"
    name = "source-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root/name


def validate_request(images, redshift, output_dir, *, field_t=None,
                     pyrats_script=None, pyrats_python=None, frequencies_mhz=None) -> RunRequest:
    """Validate user inputs without opening FITS data or importing Tk."""
    if not isinstance(images, (list, tuple)) or not 4 <= len(images) <= 64:
        raise ValueError("Select 4–64 radio FITS images at distinct frequencies.")
    paths = tuple(Path(path).expanduser().resolve() for path in images)
    if len({os.path.normcase(str(path)) for path in paths}) != len(paths):
        raise ValueError("Each input image must be selected only once.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Input image is not a readable file: {path}")
    try:
        if isinstance(redshift, bool):
            raise ValueError
        z = float(redshift)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Enter the source redshift as a positive number, for example 0.1469.") from exc
    if not math.isfinite(z) or z <= 0:
        raise ValueError("The source redshift must be finite and greater than zero.")
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("Choose a folder for the results.")
    target = Path(output_dir).expanduser().resolve()
    if any(path == target or path.is_relative_to(target) for path in paths):
        raise ValueError("The result folder must be separate from the input images.")
    if target.exists():
        raise ValueError("The result folder already exists. Choose a new result folder.")
    field = None
    if field_t is not None and str(field_t).strip():
        try:
            if isinstance(field_t, bool):
                raise ValueError
            field = float(field_t)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("The optional fixed magnetic field must be a number in tesla.") from exc
        if not math.isfinite(field) or not 1e-12 <= field <= 1e-7:
            raise ValueError("The fixed magnetic field must lie between 1e-12 and 1e-7 T.")
    script = None
    if pyrats_script is not None and str(pyrats_script).strip():
        script = Path(pyrats_script).expanduser().resolve()
        if not script.is_file():
            raise ValueError("The selected PyRATS script does not exist.")
    python = str(pyrats_python).strip() if pyrats_python is not None else None
    frequencies = None
    if frequencies_mhz is not None and str(frequencies_mhz).strip():
        values = frequencies_mhz.replace(",", " ").split() if isinstance(frequencies_mhz,str) else frequencies_mhz
        try:
            if any(isinstance(value,bool) for value in values):raise ValueError
            frequencies=tuple(float(value) for value in values)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError("Frequencies must be numbers in MHz, in the displayed image order.") from exc
        if (len(frequencies)!=len(paths) or len(set(frequencies))!=len(frequencies) or
            any(not math.isfinite(value) or value<=0 for value in frequencies)):
            raise ValueError("Provide one distinct positive frequency in MHz per selected image, in the displayed order.")
    return RunRequest(paths, z, target, field, script, python or None, frequencies)


def _automatic_runner(*args, **kwargs):
    from .automatic import run_automatic
    return run_automatic(*args, **kwargs)


class RunController:
    """Own one worker; communicate exclusively through a thread-safe queue."""

    def __init__(self, runner: Callable | None = None):
        self.runner = runner or _automatic_runner
        self.events: queue.Queue[RunEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._running = False
        self._run_id = 0
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running

    def start(self, request: RunRequest) -> int:
        if not isinstance(request, RunRequest):
            raise TypeError("A validated RunRequest is required.")
        with self._lock:
            if self._running:
                raise RuntimeError("An analysis is already running.")
            self._running = True
            self._run_id += 1
            run_id = self._run_id
            cancel_event = threading.Event()
            self._cancel_event = cancel_event

        def progress(message, fraction=None):
            # Numerical/backend code never touches Tk widgets.
            amount = None
            if fraction is not None:
                try:
                    value = float(fraction)
                    amount = min(max(value, 0.), 1.) if math.isfinite(value) else None
                except (TypeError, ValueError, OverflowError):
                    pass
            self.events.put(RunEvent(run_id, "progress", {"message":str(message), "fraction":amount}))

        def worker():
            try:
                result = self.runner(list(request.images), request.redshift, request.output_dir,
                    field_t=request.field_t, pyrats_script=request.pyrats_script,
                    pyrats_python=request.pyrats_python, progress=progress,
                    cancel_event=cancel_event, frequencies_mhz=request.frequencies_mhz)
            except Exception as exc:
                kind = "cancelled" if cancel_event.is_set() else "error"
                self.events.put(RunEvent(run_id, kind,
                    {"message":str(exc) or type(exc).__name__, "traceback":traceback.format_exc()}))
            else:
                self.events.put(RunEvent(run_id, "complete", result))
            finally:
                with self._lock:
                    if run_id == self._run_id:
                        self._running = False

        self._thread = threading.Thread(target=worker, name=f"pyradage-analysis-{run_id}", daemon=True)
        self._thread.start()
        return run_id

    def cancel(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._cancel_event.set()
            return True

    def drain_events(self) -> list[RunEvent]:
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def join(self, timeout=None):
        """For deterministic tests or shutdown; UI polling never blocks here."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


def open_path(path: str | Path):
    """Open a local result through the OS, without a command shell."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Application:
    """Tk view; construct only after Tk is successfully imported."""

    def __init__(self, root, *, controller: RunController | None = None):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root = root
        self.controller = controller or RunController()
        self.images: list[Path] = []
        self.active_run: int | None = None
        self.last_output: Path | None = None
        self.last_result = None
        self.closing = False
        root.title("PyRADAGE — radio-source age maps")
        root.geometry("920x780")
        root.minsize(760, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.option_add("*Font", "{TkDefaultFont} 12")
        frame = ttk.Frame(root, padding=18)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(9, weight=1)
        ttk.Label(frame, text="Radio images → radiative-age maps", font=("TkDefaultFont",16,"bold")).grid(sticky="w")
        ttk.Label(frame, text="Select cropped multi-frequency FITS images, enter the redshift, then run.").grid(row=1,sticky="w",pady=(5,12))

        source = ttk.LabelFrame(frame,text="1. Select the source images",padding=10)
        source.grid(row=2,sticky="nsew")
        source.columnconfigure(0,weight=1); source.rowconfigure(0,weight=1)
        self.file_list = tk.Listbox(source,height=7,selectmode=tk.EXTENDED,exportselection=False)
        self.file_list.grid(row=0,column=0,sticky="nsew")
        scrollbar = ttk.Scrollbar(source,orient="vertical",command=self.file_list.yview)
        scrollbar.grid(row=0,column=1,sticky="ns"); self.file_list.configure(yscrollcommand=scrollbar.set)
        buttons = ttk.Frame(source); buttons.grid(row=1,column=0,columnspan=2,sticky="ew",pady=(8,0))
        self.add_button = ttk.Button(buttons,text="Add FITS images…",command=self.add_images)
        self.add_button.grid(row=0,column=0,padx=(0,8))
        self.remove_button = ttk.Button(buttons,text="Remove selected",command=self.remove_images)
        self.remove_button.grid(row=0,column=1,padx=(0,8))
        self.clear_button = ttk.Button(buttons,text="Clear",command=self.clear_images)
        self.clear_button.grid(row=0,column=2)
        self.file_count = tk.StringVar(value="Select at least four distinct-frequency images.")
        ttk.Label(buttons,textvariable=self.file_count).grid(row=1,column=0,columnspan=3,sticky="w",pady=(6,0))
        self.file_list.bind("<Delete>",lambda event:self.remove_images())
        root.bind("<Control-o>",lambda event:self.add_images())

        details = ttk.Frame(frame); details.grid(row=3,sticky="ew",pady=12)
        details.columnconfigure(1,weight=1)
        ttk.Label(details,text="2. Source redshift").grid(row=0,column=0,sticky="w",padx=(0,10))
        self.redshift = tk.StringVar()
        self.redshift_entry = ttk.Entry(details,textvariable=self.redshift,width=16)
        self.redshift_entry.grid(row=0,column=1,sticky="w")
        ttk.Label(details,text="for example, 0.1469").grid(row=0,column=2,sticky="w",padx=8)
        self.output_dir = tk.StringVar(value=str(default_output_dir()))
        ttk.Label(details,text="Results folder").grid(row=1,column=0,sticky="w",pady=(10,0))
        self.output_entry = ttk.Entry(details,textvariable=self.output_dir)
        self.output_entry.grid(row=1,column=1,sticky="ew",pady=(10,0))
        self.output_button = ttk.Button(details,text="Choose…",command=self.choose_output)
        self.output_button.grid(row=1,column=2,padx=8,pady=(10,0))

        self.advanced_visible = tk.BooleanVar(value=False)
        self.advanced_toggle = ttk.Checkbutton(frame,text="Advanced: supplied magnetic field or PyRATS installation",variable=self.advanced_visible,command=self.toggle_advanced)
        self.advanced_toggle.grid(row=4,sticky="w")
        self.advanced = ttk.Frame(frame,padding=(0,8))
        self.advanced.columnconfigure(1,weight=1)
        self.field_t = tk.StringVar()
        self.pyrats_script = tk.StringVar(value=os.environ.get("PYRADAGE_PYRATS_SCRIPT") or os.environ.get("FR2DYNAGE_PYRATS_SCRIPT", ""))
        self.pyrats_python = tk.StringVar(value=os.environ.get("PYRADAGE_PYRATS_PYTHON") or os.environ.get("FR2DYNAGE_PYRATS_PYTHON", ""))
        self.frequencies_mhz = tk.StringVar()
        self.advanced_entries=[]
        for row,(label,variable) in enumerate([
                ("Fixed magnetic field (T)",self.field_t),
                ("PyRATS.py (optional path)",self.pyrats_script),
                ("PyRATS Python (optional)",self.pyrats_python),
                ("Frequencies in MHz (optional)",self.frequencies_mhz)]):
            ttk.Label(self.advanced,text=label).grid(row=row,column=0,sticky="w",padx=(0,10),pady=3)
            entry=ttk.Entry(self.advanced,textvariable=variable);entry.grid(row=row,column=1,sticky="ew",pady=3)
            self.advanced_entries.append(entry)
        ttk.Label(self.advanced,text="Leave the field blank for B = 0.4 B_eq. A fixed field excludes its uncertainty. Override missing/incorrect frequencies with one MHz value per image, in the displayed order.",wraplength=740).grid(row=4,column=0,columnspan=2,sticky="w",pady=(5,0))

        controls=ttk.Frame(frame);controls.grid(row=6,sticky="ew",pady=(12,8))
        self.run_button=ttk.Button(controls,text="3. Run analysis",command=self.run)
        self.run_button.grid(row=0,column=0,padx=(0,8))
        self.cancel_button=ttk.Button(controls,text="Cancel",command=self.cancel,state="disabled")
        self.cancel_button.grid(row=0,column=1,padx=(0,8))
        self.open_button=ttk.Button(controls,text="Open results",command=self.open_results,state="disabled")
        self.open_button.grid(row=0,column=2,padx=(0,8))
        self.pdf_button=ttk.Button(controls,text="Open summary PDF",command=self.open_summary,state="disabled")
        self.pdf_button.grid(row=0,column=3)
        self.status=tk.StringVar(value="Ready. Only the images and redshift are required after one-time setup.")
        ttk.Label(frame,textvariable=self.status,wraplength=850).grid(row=7,sticky="w",pady=(0,6))
        self.progressbar=ttk.Progressbar(frame,mode="indeterminate")
        self.progressbar.grid(row=8,sticky="ew")
        self.log=ScrolledText(frame,height=9,wrap="word",state="disabled",font=("TkFixedFont",10))
        self.log.grid(row=9,sticky="nsew",pady=(10,0))
        ttk.Label(frame,text="Outputs: one combined PDF/PNG figure, FITS maps, parameter tables and an auditable report. Radiative and dynamical clocks remain distinct.",wraplength=850).grid(row=10,sticky="w",pady=(10,0))
        self.redshift.trace_add("write",lambda *args:self.update_controls())
        self.update_controls()
        root.after(100,self.poll)

    def _append(self,text):
        self.log.configure(state="normal")
        self.log.insert("end",str(text)+"\n")
        self.log.see("end");self.log.configure(state="disabled")

    def update_controls(self):
        busy=self.active_run is not None
        self.run_button.configure(state="normal" if not busy and len(self.images)>=4 and self.redshift.get().strip() else "disabled")
        for widget in (self.add_button,self.remove_button,self.clear_button,self.output_button,self.output_entry,self.advanced_toggle,*self.advanced_entries):
            widget.configure(state="disabled" if busy else "normal")
        self.redshift_entry.configure(state="disabled" if busy or not self.images else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.open_button.configure(state="normal" if self.last_output is not None and not busy else "disabled")
        self.pdf_button.configure(state="normal" if self._summary_path() is not None and not busy else "disabled")

    def add_images(self):
        if self.active_run is not None:return
        selected=self.filedialog.askopenfilenames(parent=self.root,title="Select radio-source FITS images",
            filetypes=[("FITS images","*.fits *.fit *.fts *.fits.gz *.fit.gz *.fts.gz"),("All files","*")])
        known={os.path.normcase(str(path)) for path in self.images}
        for filename in selected:
            path=Path(filename).expanduser().resolve()
            identity=os.path.normcase(str(path))
            if identity not in known:
                self.images.append(path);known.add(identity)
        self.refresh_images()
        if self.images:self.redshift_entry.focus_set()

    def refresh_images(self):
        self.file_list.delete(0,"end")
        for path in self.images:self.file_list.insert("end",str(path))
        self.file_count.set(f"{len(self.images)} images selected. Frequencies and beams will be read from the FITS metadata.")
        self.update_controls()

    def remove_images(self):
        if self.active_run is not None:return
        selected=set(self.file_list.curselection())
        self.images=[path for index,path in enumerate(self.images) if index not in selected]
        self.refresh_images()

    def clear_images(self):
        if self.active_run is not None:return
        self.images=[];self.refresh_images()

    def choose_output(self):
        if self.active_run is not None:return
        parent=self.filedialog.askdirectory(parent=self.root,title="Choose the parent folder for a new result folder")
        if parent:self.output_dir.set(str(default_output_dir(Path(parent))))

    def toggle_advanced(self):
        if self.advanced_visible.get():self.advanced.grid(row=5,sticky="ew")
        else:self.advanced.grid_remove()

    def run(self):
        if self.active_run is not None:return
        try:
            request=validate_request(self.images,self.redshift.get(),self.output_dir.get(),
                field_t=self.field_t.get(),pyrats_script=self.pyrats_script.get(),pyrats_python=self.pyrats_python.get(),
                frequencies_mhz=self.frequencies_mhz.get())
            self.active_run=self.controller.start(request)
        except Exception as exc:
            self.messagebox.showerror("Cannot start analysis",str(exc),parent=self.root)
            return
        self.last_output=request.output_dir;self.last_result=None
        self.status.set("Preparing images and checking the source inputs…")
        self.progressbar.configure(mode="indeterminate");self.progressbar.start(12)
        self._append(f"Started: {len(request.images)} images, z={request.redshift:g}")
        self._append(f"Results: {request.output_dir}")
        self.update_controls()

    def cancel(self):
        if self.controller.cancel():
            self.status.set("Cancellation requested. Waiting for the current numerical step to stop safely…")
            self.cancel_button.configure(state="disabled")
            self._append("Cancellation requested.")

    def _summary_path(self):
        if not isinstance(self.last_result,dict):return None
        value=self.last_result.get("summary_pdf")
        if not value:return None
        path=Path(value)
        if not path.is_absolute() and self.last_output is not None:path=self.last_output/path
        return path if path.is_file() else None

    def open_results(self):
        try:
            if self.last_output is not None:open_path(self.last_output)
        except OSError as exc:self.messagebox.showerror("Cannot open results",str(exc),parent=self.root)

    def open_summary(self):
        try:
            path=self._summary_path()
            if path is not None:open_path(path)
        except OSError as exc:self.messagebox.showerror("Cannot open summary",str(exc),parent=self.root)

    def poll(self):
        for event in self.controller.drain_events():
            if event.run_id != self.active_run:continue
            if event.kind=="progress":
                payload=event.payload
                self.status.set(payload["message"]);self._append(payload["message"])
                if payload["fraction"] is not None:
                    self.progressbar.stop();self.progressbar.configure(mode="determinate",maximum=1.,value=payload["fraction"])
            else:
                self.progressbar.stop()
                self.active_run=None
                if event.kind=="complete":
                    self.last_result=event.payload
                    self.status.set("Complete. Open the combined summary PDF or the result folder.")
                    self.progressbar.configure(mode="determinate",maximum=1.,value=1.)
                    self._append("Analysis completed. Inspect age-quality flags and model assumptions in the report.")
                elif event.kind=="cancelled":
                    self.status.set("Cancelled. Existing source images and earlier results were preserved.")
                    self._append(event.payload["message"])
                else:
                    self.status.set("The analysis could not complete. See the message below.")
                    self._append(event.payload["message"]);self._append(event.payload["traceback"])
                    if not self.closing:self.messagebox.showerror("Analysis stopped",event.payload["message"],parent=self.root)
                self.update_controls()
        if self.closing and self.active_run is None:
            self.root.destroy();return
        self.root.after(100,self.poll)

    def close(self):
        if self.active_run is not None:
            self.closing=True;self.cancel()
        else:self.root.destroy()


def main(argv=None) -> int:
    """Open the wizard, or explain how to install a Tk-enabled Python."""
    import argparse
    parser=argparse.ArgumentParser(prog="pyradage-gui",description="Select radio FITS images and a source redshift to generate age maps.")
    parser.parse_args(argv)
    try:
        import tkinter as tk
        root=tk.Tk()
    except (ImportError, RuntimeError) as exc:
        detail=(str(exc) or type(exc).__name__).splitlines()[0]
        print("Cannot open the graphical interface: this Python needs Tk and a graphical display. "
              "Use a Tk-enabled Python installation (python.org, conda, or your OS python3-tk package). "
              "The headless automatic command remains available. Details: "+detail,file=sys.stderr)
        return 2
    except Exception as exc:
        # TclError also covers an unavailable display; importing it at module
        # scope would make the headless validator depend on Tk.
        detail=(str(exc) or type(exc).__name__).splitlines()[0]
        print("Cannot open the graphical display or Tk runtime: "+detail+
              ". Use a Tk-enabled desktop Python/session, or the headless automatic command. See GUI_GUIDE.md.",file=sys.stderr)
        return 2
    Application(root)
    root.mainloop()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
