@echo off
echo ====================================
echo Ultrasound Boundary Marker - Build
echo ====================================
echo.

py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 is not installed.
    pause
    exit /b 1
)

echo [1/2] Installing required packages...
py -3 -m pip install pyinstaller customtkinter scipy matplotlib numpy --quiet

echo [2/2] Building executable...
py -3 -m PyInstaller --onefile --windowed --name "BoundaryMarker" ^
    --hidden-import=customtkinter ^
    --hidden-import=scipy.signal ^
    --collect-data customtkinter ^
    boundary_marker_gui.py

echo.
echo ====================================
if exist "dist\BoundaryMarker.exe" (
    echo Build SUCCESS!
    echo Executable: dist\BoundaryMarker.exe
) else (
    echo Build FAILED! Check errors above.
)
echo ====================================
pause
