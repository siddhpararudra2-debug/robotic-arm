@echo off
title Neurosurgical Robot - Headless Mode
cd /d "%~dp0"

echo ========================================
echo  Neurosurgical 4-Arm Robot Simulator
echo  Stage 1 - Headless Scene Check
echo ========================================
echo.

echo [1/2] Generating URDF from robot_config.yaml...
python generate_urdf.py
if errorlevel 1 (
    echo.
    echo ERROR: URDF generation failed. Check robot_config.yaml for errors.
    pause
    exit /b 1
)

echo.
echo [2/2] Running headless scene check...
python run_simulation.py

pause