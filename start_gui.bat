@echo off
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install dependencies. Please check if python/pip is installed.
)

echo Starting ImageParser GUI...
cd frontend
npm run electron:dev
pause
