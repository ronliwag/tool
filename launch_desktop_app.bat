@echo off
REM Enhanced Desktop App Launcher for StreamSpeech Tool with Landing Page
REM This script activates the virtual environment and launches the desktop app with landing page

cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
echo Starting Enhanced Desktop App with Landing Page...
python "Original Streamspeech\Modified Streamspeech\demo\launch_with_landing_page.py"
pause

