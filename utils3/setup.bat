@echo off
echo ========================================
echo   Ultrasound ADC Capture Tool Setup
echo ========================================
echo.

py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 is not installed.
    pause
    exit /b 1
)

echo Installing packages...
py -3 -m pip install pyserial numpy scipy matplotlib customtkinter

echo.
echo Setup complete!
echo Run: run_gui.bat
echo.
pause
