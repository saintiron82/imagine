@echo off
chcp 65001 >nul 2>&1
title Imagine — Stop All

echo === Imagine — Stop All ===

rem Kill Electron
tasklist /fi "imagename eq electron.exe" 2>nul | find /i "electron.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im electron.exe >nul 2>&1
    echo   Killed Electron
)

rem Kill Node (Vite / npm)
tasklist /fi "imagename eq node.exe" 2>nul | find /i "node.exe" >nul
if %errorlevel%==0 (
    taskkill /f /im node.exe >nul 2>&1
    echo   Killed Node.js
)

rem Kill Python backend processes
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%backend.server.app%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed FastAPI PID %%a
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%ingest_engine%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Ingest PID %%a
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker_daemon%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Worker PID %%a
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%api_search%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo   Killed Search PID %%a
)

rem Force free ports 9274, 8000
for %%p in (9274 8000) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        taskkill /f /pid %%a >nul 2>&1
        echo   Freed port %%p (PID %%a^)
    )
)

echo Done.
