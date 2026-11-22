@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python desktop_chat\install.py
) else (
  py -3 desktop_chat\install.py
)
if errorlevel 1 echo Installation failed. Read the message above.
pause
