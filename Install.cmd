@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python desktop_chat\setup_local.py --install --shortcut
) else (
  py -3 desktop_chat\setup_local.py --install --shortcut
)
if errorlevel 1 echo Installation failed. Read the message above.
pause
