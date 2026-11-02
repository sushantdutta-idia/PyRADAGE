@echo off
rem Open PyRADAGE from this source checkout.
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0gui.py"
) else (
    python "%~dp0gui.py"
)
if errorlevel 1 pause
