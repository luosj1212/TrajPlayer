@echo off
setlocal
cd /d "%~dp0"
set "VENV=.venv-build"

if not exist "%VENV%\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python launcher not found. Install Python 3.10 or 3.11 first.
    pause
    exit /b 1
  )
  py -3 -m venv "%VENV%"
  if errorlevel 1 goto :failed
)

call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -e ".[dev]"
if errorlevel 1 goto :failed
python -m pytest -q
if errorlevel 1 goto :failed
python scripts\check_release.py
if errorlevel 1 goto :failed
python scripts\build_release.py
if errorlevel 1 goto :failed

echo.
echo Release archive created in the project root.
pause
exit /b 0

:failed
echo.
echo Build failed.
pause
exit /b 1
