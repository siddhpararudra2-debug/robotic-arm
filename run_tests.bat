@echo off
title Neurosurgical Robot - Test Suite
cd /d "%~dp0"

echo ========================================
echo  Neurosurgical 4-Arm Robot Simulator
echo  Acceptance Test Suite
echo ========================================
echo.

python -m unittest discover -s tests -v

pause