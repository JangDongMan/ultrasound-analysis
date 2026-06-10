@echo off
:: boundary_detector.exe Windows 빌드 스크립트
:: 요구사항: MinGW-w64 (gcc) 가 PATH에 있어야 함
::   설치: https://winlibs.com  또는 choco install mingw

echo ==========================================
echo  Building boundary_detector.exe (Windows)
echo ==========================================

where gcc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gcc 를 찾을 수 없습니다.
    echo         MinGW-w64 를 설치하고 PATH 에 추가하세요.
    echo         https://winlibs.com
    pause
    exit /b 1
)

gcc -O3 -Wall -o boundary_detector.exe boundary_detector.c boundary_detector_cli.c -lm

if errorlevel 1 (
    echo [ERROR] 빌드 실패
    pause
    exit /b 1
)

echo.
echo [OK] boundary_detector.exe 빌드 완료
echo.
echo 실행:
echo   python auto_mark_usdata.py
pause
