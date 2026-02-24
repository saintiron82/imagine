@echo off
chcp 65001 >nul 2>&1
title Imagine — FastAPI Server

cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8000

rem Activate venv
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

rem Check frontend build
if not exist "frontend\dist" (
    echo [!] frontend\dist\ not found. Building...
    cd frontend
    call npm run build
    if %errorlevel% neq 0 (
        echo [ERROR] Frontend build failed.
        pause
        exit /b 1
    )
    cd ..
)

echo === Imagine — FastAPI Server ===
echo   URL:  http://localhost:%PORT%
echo   Docs: http://localhost:%PORT%/docs
echo.

python -m backend.server.app
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server failed to start.
    pause
)
