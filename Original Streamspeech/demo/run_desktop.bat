@echo off
echo Starting StreamSpeech Desktop Application...
echo.
cd /d "%~dp0"
set PYTHONPATH=../fairseq
python desktop_app.py
pause

