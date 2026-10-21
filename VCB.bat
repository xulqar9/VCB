@echo off
rem VCB - double-click to start. First run installs everything it needs.
rem Options: VCB.bat --settings | --extras | --reinstall | any main.py flag
title VCB
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
if errorlevel 1 (
    echo.
    echo VCB stopped with an error - see the messages above.
    pause
)
