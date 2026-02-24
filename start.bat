@echo off
chcp 65001 >nul 2>&1
title Imagine — Electron Dev

cd /d "%~dp0"

rem Activate venv
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

echo === Imagine — Electron Dev ===
echo   Vite:     http://localhost:9274
echo   Electron: auto-launch after Vite ready
echo.

cd frontend
npm run electron:dev
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start. Check Node.js and npm are installed.
    pause
)
