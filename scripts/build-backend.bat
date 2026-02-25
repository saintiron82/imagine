@echo off
REM Build backend_cli.exe using PyInstaller (Windows)
REM Run from project root: scripts\build-backend.bat

echo === Imagine Backend Build ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ first.
    exit /b 1
)

REM Install PyInstaller if needed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt
pip install uvicorn fastapi python-jose passlib python-multipart

REM Build
echo.
echo Building backend_cli.exe...
pyinstaller backend_cli.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

echo.
echo === Build complete ===
echo Output: dist\backend_cli\backend_cli.exe
echo.
echo Next: cd frontend ^&^& npm run build ^&^& npx electron-builder --win --x64
