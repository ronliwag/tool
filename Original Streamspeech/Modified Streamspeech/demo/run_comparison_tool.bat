@echo off
echo Starting StreamSpeech Comparison Tool...
echo.
echo This application allows you to compare:
echo - Original StreamSpeech (untouched)
echo - Modified StreamSpeech (ODConv + GRC + LoRA)
echo.
echo Setting up environment...
set PYTHONPATH=..\..\fairseq
echo.
echo Launching comparison desktop application...
python enhanced_desktop_app.py
echo.
echo Application closed.
pause



