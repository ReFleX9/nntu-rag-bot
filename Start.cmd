@echo off
cd /d "%~dp0"
if not exist "work\desktop_chat.json" (
  echo Run Install.cmd first.
  pause
  exit /b 1
)
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "desktop_chat\launcher.py"
) else (
  echo Use the desktop shortcut created by the installer.
  pause
)
