@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Imagine Dev Launcher
cd /d "%~dp0"

:menu
echo.
echo  ======================================
echo   Imagine - Dev Launcher
echo  ======================================
echo   1. Start (clean) - stop all then run
echo   2. Start (as-is) - just run
echo   3. Stop all
echo   4. Status check
echo   0. Exit
echo  ======================================
echo.
set /p choice="Select [0-4]: "

if "%choice%"=="1" goto clean_start
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto status
if "%choice%"=="0" goto :eof
echo  Invalid choice.
goto menu

rem ----------------------------------------
:status
echo.
echo --- Process Status ---

tasklist /fi "imagename eq electron.exe" 2>nul | find /i "electron.exe" >nul
if %errorlevel%==0 (
    echo   [RUNNING] Electron
) else (
    echo   [  --   ] Electron
)

tasklist /fi "imagename eq node.exe" 2>nul | find /i "node.exe" >nul
if %errorlevel%==0 (
    echo   [RUNNING] Node.js (Vite)
) else (
    echo   [  --   ] Node.js (Vite)
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_ipc%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo   [RUNNING] Worker IPC (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%api_search%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo   [RUNNING] Search Daemon (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%ingest_engine%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo   [RUNNING] Pipeline (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%backend.server.app%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo   [RUNNING] FastAPI Server (PID %%a)
)

echo.
echo --- Port Status ---
for %%p in (9274 8000) do (
    netstat -ano 2>nul | findstr :%%p | findstr LISTENING >nul
    if !errorlevel!==0 (
        echo   [IN USE] Port %%p
    ) else (
        echo   [ FREE ] Port %%p
    )
)

echo.
goto menu

rem ----------------------------------------
:stop
echo.
echo --- Stopping all processes ---

tasklist /fi "imagename eq electron.exe" 2>nul | find /i "electron.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im electron.exe >nul 2>&1
    echo   Killed Electron
)

tasklist /fi "imagename eq node.exe" 2>nul | find /i "node.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im node.exe >nul 2>&1
    echo   Killed Node.js
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%backend.server.app%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed FastAPI (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%ingest_engine%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Pipeline (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_ipc%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Worker IPC (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_daemon%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Worker Daemon (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%api_search%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Search Daemon (PID %%a)
)

for %%p in (9274 8000) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        taskkill /f /pid %%a >nul 2>&1
        echo   Freed port %%p (PID %%a)
    )
)

echo   Done.
echo.
goto menu

rem ----------------------------------------
:clean_start
echo.
echo --- Clean start: stopping existing processes ---
call :stop_silent
goto start

rem ----------------------------------------
:start
echo.
echo === Imagine - Electron Dev ===
echo   Vite:     http://localhost:9274
echo   Electron: auto-launch after Vite ready
echo.

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

cd frontend
npm run electron:dev
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start. Check Node.js and npm are installed.
    pause
)
goto :eof

rem ----------------------------------------
:stop_silent
tasklist /fi "imagename eq electron.exe" 2>nul | find /i "electron.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im electron.exe >nul 2>&1
    echo   Killed Electron
)
tasklist /fi "imagename eq node.exe" 2>nul | find /i "node.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im node.exe >nul 2>&1
    echo   Killed Node.js
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%backend.server.app%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed FastAPI (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%ingest_engine%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Pipeline (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_ipc%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Worker IPC (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_daemon%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Worker Daemon (PID %%a)
)
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%api_search%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Search Daemon (PID %%a)
)
for %%p in (9274 8000) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        taskkill /f /pid %%a >nul 2>&1
        echo   Freed port %%p (PID %%a)
    )
)
timeout /t 2 /nobreak >nul
goto :eof
