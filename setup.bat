@echo off
setlocal enabledelayedexpansion
title GodsEye Universal Setup

echo =================================================================
echo   GodsEye Setup - Hardware Detection ^& Auto-Installation
echo =================================================================
echo.

REM 1. Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM 2. Setup Virtual Environment if not inside one
if not defined VIRTUAL_ENV (
    if not exist ".venv" (
        echo [*] Creating virtual environment (.venv)...
        python -m venv .venv
        if %ERRORLEVEL% NEQ 0 (
            echo [WARN] Could not create .venv. Continuing with current Python...
        )
    )
    if exist ".venv\Scripts\activate.bat" (
        echo [*] Activating virtual environment (.venv)...
        call .venv\Scripts\activate.bat
    )
)

REM 3. Run cross-platform setup script
echo [*] Launching automated setup engine...
python setup_env.py %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo   Setup complete! You can now launch the dashboard by running:
    echo     run_dashboard.bat   (or: python run_dashboard_v2.py)
    echo =================================================================
) else (
    echo.
    echo [ERROR] Setup encountered an issue. See details above.
)

pause
