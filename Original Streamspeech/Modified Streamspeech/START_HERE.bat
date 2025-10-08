@echo off
echo ========================================
echo StreamSpeech Thesis Defense Tool
echo ========================================
echo.
echo This tool helps you compare Original vs Modified StreamSpeech
echo for your thesis defense on Wednesday.
echo.
echo Step 1: Integrating your thesis work...
python integrate_thesis_work.py
echo.
echo Step 2: Launching comparison tool...
cd demo
run_comparison_tool.bat
echo.
echo ========================================
echo Ready for thesis defense!
echo ========================================
pause







