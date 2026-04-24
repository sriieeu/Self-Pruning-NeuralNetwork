@echo off
REM =============================================================================
REM setup.bat — Create venv, install dependencies, and optionally run training
REM
REM Usage:
REM   setup.bat            -> setup only
REM   setup.bat --run      -> setup + run with default settings
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo =============================================================
echo   Self-Pruning Neural Network  --  Setup
echo =============================================================
echo.

REM ── Check Python ─────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found. Install from https://www.python.org/downloads/
    exit /b 1
)

for /f "delims=" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%v
echo [OK] Python %PYVER% found

REM ── Create venv ──────────────────────────────────────────────
if exist venv\ (
    echo [OK] Existing venv found -- reusing
) else (
    echo [..] Creating virtual environment ...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM ── Activate ─────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] Activated venv

REM ── Upgrade pip ──────────────────────────────────────────────
echo [..] Upgrading pip ...
pip install --upgrade pip --quiet

REM ── Install dependencies ─────────────────────────────────────
echo [..] Installing requirements (this may take a few minutes) ...
pip install -r requirements.txt --quiet
echo [OK] All dependencies installed

REM ── CUDA check ───────────────────────────────────────────────
python -c "import torch; print('[OK] CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '[i]  No CUDA -- will use CPU')"

echo.
echo -------------------------------------------------------------
echo   Setup complete!
echo.
echo   To activate the venv in future sessions:
echo     venv\Scripts\activate.bat
echo.
echo   To train:
echo     python main.py           (defaults)
echo     python main.py --help    (all options)
echo -------------------------------------------------------------
echo.

REM ── Optionally run training ──────────────────────────────────
if "%1"=="--run" (
    echo [..] Starting training ...
    echo.
    python main.py %2 %3 %4 %5 %6 %7 %8 %9
)

endlocal
