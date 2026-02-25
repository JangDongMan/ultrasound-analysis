@echo off
echo ====================================
echo VB5K Capture - Build EXE
echo ====================================
echo.

echo [1/2] Installing required packages...
py -3 -m pip install pyinstaller customtkinter scipy pyserial matplotlib numpy --quiet

echo [2/2] Building executable...
py -3 -m PyInstaller --onefile --windowed --name "vb5k_capture" ^
    --hidden-import=customtkinter ^
    --hidden-import=scipy.signal ^
    --hidden-import=serial.tools.list_ports ^
    --collect-data customtkinter ^
    adc_capture_gui.py

echo.
echo ====================================
if exist "dist\vb5k_capture.exe" (
    echo Build SUCCESS!
    echo Executable: dist\vb5k_capture.exe
) else (
    echo Build FAILED! Check errors above.
)
echo ====================================
pause
