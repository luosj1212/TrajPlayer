@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py %*
  exit /b %errorlevel%
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3 app.py %*
  exit /b %errorlevel%
)

where python >nul 2>nul
if not errorlevel 1 (
  python app.py %*
  exit /b %errorlevel%
)

echo Python 3.10 or 3.11 was not found.
echo See README.md for source installation instructions.
pause
exit /b 1
