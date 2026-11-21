@echo off
setlocal
title GodsEye Dashboard v2

REM Check for virtual environment and activate if available
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "..\env\Scripts\activate.bat" (
    call ..\env\Scripts\activate.bat
)

python run_dashboard_v2.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Notice] If this is your first time running GodsEye, please run:
    echo     setup.bat
    pause
)
