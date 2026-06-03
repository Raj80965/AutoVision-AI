@echo off
echo ========================================
echo   CARLA Advanced Autonomous Driving
echo ========================================
echo.

REM Check if CARLA is running
netstat -ano | findstr ":2000" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] CARLA Server is NOT running!
    echo Please start: D:\carla\WindowsNoEditor\CarlaUE4.exe
    echo.
    pause
    exit /b 1
)

echo [OK] CARLA Server detected on port 2000
echo.

REM Start Dashboard Server in a new window
echo [..] Starting Dashboard Server...
start "CARLA Dashboard" cmd /k "set PYTHONIOENCODING=utf-8 && cd /d D:\carla\MyProject && python dashboard_server.py"
timeout /t 3 /nobreak >nul
echo [OK] Dashboard started at http://localhost:5000

echo.
echo [..] Starting Advanced Driving System...
echo.
echo Controls:
echo   R = Toggle RL Mode
echo   P = Toggle Auto Parking
echo   ESC = Exit
echo.

set PYTHONIOENCODING=utf-8
cd /d D:\carla\MyProject
python advanced_drive.py

pause
