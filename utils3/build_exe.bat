@echo off
echo ====================================
echo Ultrasound ADC Capture - Build EXE
echo ====================================
echo.

echo [1/2] Installing required packages...
py -3 -m pip install pyinstaller customtkinter scipy pyserial matplotlib numpy --quiet

echo [2/2] Building executable...
py -3 -m PyInstaller --onefile --windowed --name "UltrasoundCapture" ^
    --hidden-import=customtkinter ^
    --hidden-import=scipy.signal ^
    --hidden-import=serial.tools.list_ports ^
    --collect-data customtkinter ^
    adc_capture_gui.py

echo.
echo ====================================
if exist "dist\UltrasoundCapture.exe" (
    echo Build SUCCESS!
    echo Executable: dist\UltrasoundCapture.exe
) else (
    echo Build FAILED! Check errors above.
)
echo ====================================
pause
