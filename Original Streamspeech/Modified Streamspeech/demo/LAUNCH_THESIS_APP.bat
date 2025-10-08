@echo off
echo Starting StreamSpeech Thesis Comparison Tool...
echo.
echo This tool integrates real thesis metrics:
echo - ASR-BLEU for translation quality
echo - Cosine Similarity (SIM) for speaker/emotion preservation
echo - Average Lagging for latency performance
echo.
echo Features:
echo - Original vs Modified HiFi-GAN comparison
echo - Real-time evaluation metrics
echo - Professional thesis reporting
echo.
cd /d "%~dp0"
python enhanced_thesis_app.py
pause







