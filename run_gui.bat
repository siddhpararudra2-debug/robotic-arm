@echo off
title Neurosurgical Robot - GUI Viewer
cd /d "%~dp0"

echo ========================================
echo  Neurosurgical 4-Arm Robot Simulator
echo  Stage 1 - GUI Viewer (3D + Joint Sliders)
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
echo [2/2] Launching PyBullet 3D viewer...
echo.
echo Controls:
echo   - Drag any joint slider to pose the arms in real time
echo   - Press H to return all arms to home
echo   - Press Q to quit the viewer
echo.
python run_simulation.py --gui

pause