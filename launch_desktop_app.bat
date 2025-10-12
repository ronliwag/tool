@echo off
REM Enhanced Desktop App Launcher for StreamSpeech Tool
REM This script activates the virtual environment and launches the desktop app

cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
echo Starting Enhanced Desktop App...
python "Original Streamspeech\Modified Streamspeech\demo\enhanced_desktop_app_streamlit_ui1.py"
pause

