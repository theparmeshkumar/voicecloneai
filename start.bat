@echo off
title VoiceClone AI Studio
echo ==============================================================
echo       VOICECLONE AI STUDIO - LOCAL NEURAL VOICE CLONING
echo ==============================================================
echo Starting local voice cloning studio on CPU...
echo URL: http://localhost:8000
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py
) else (
    python run.py
)

pause
