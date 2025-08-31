@echo off
echo 🚀 Starting Academic Assistant API in Development Mode...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Check if .env file exists
if not exist .env (
    echo ⚠️  Warning: .env file not found. Please create one from .env.example
    echo.
)

REM Start the application
python run_dev.py

pause
