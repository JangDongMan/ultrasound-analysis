@echo off
echo ===============================================================
echo Ultrasound Marker - Windows Build
echo ===============================================================
echo.

py -3.12 -m PyInstaller --name=UltrasoundMarker --onefile --windowed --icon=NONE --hidden-import=tkinter --hidden-import=matplotlib.backends.backend_tkagg --hidden-import=numpy --hidden-import=pandas --hidden-import=openpyxl --exclude-module=torch --exclude-module=tensorflow --exclude-module=sklearn --exclude-module=cv2 --exclude-module=PIL --exclude-module=IPython --exclude-module=notebook --exclude-module=pytest --exclude-module=scipy.spatial.transform --log-level=WARN manual_position_marker.py

echo.
echo ===============================================================
echo Build Complete!
echo ===============================================================
echo.

pause
