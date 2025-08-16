@echo off
REM PhD Outreach Automation - Windows 11 One-Click Setup
title PhD Outreach Setup - 2-Stage System

color 0B
echo.
echo ================================================================
echo   PhD Outreach Automation - 2-Stage System Setup
echo ================================================================
echo   Stage 1: Professor Discovery (gpt-4o-mini - Cost Effective)
echo   Stage 2: Email Generation (gpt-4 - High Quality)
echo ================================================================
echo.

REM Check if Python is installed
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

python --version
echo Python found successfully!
echo.

REM Check Python version
echo [2/5] Verifying Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.8 or higher is required
    echo Please update your Python installation
    echo.
    pause
    exit /b 1
)

echo Python version is compatible!
echo.

REM Create virtual environment
echo [3/5] Creating virtual environment...
if exist "phd_outreach_env" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv phd_outreach_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [4/5] Activating virtual environment...
call phd_outreach_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.

REM Install basic requirements and run setup
echo [5/5] Running comprehensive setup...
echo This will install all dependencies and configure the system...
echo.

REM Install basic packages first
python -m pip install --upgrade pip setuptools wheel
python -m pip install python-dotenv requests

REM Run the comprehensive setup script
python setup.py

if errorlevel 1 (
    echo.
    echo ================================================
    color 0C
    echo   ERROR: Setup failed!
    echo ================================================
    echo Please check the error messages above.
    echo You may need to:
    echo 1. Install Microsoft Visual C++ Redistributable
    echo 2. Install Git (for some packages)
    echo 3. Run as Administrator
    echo 4. Check your internet connection
    echo.
    pause
    exit /b 1
)

echo.
color 0A
echo ================================================================
echo   Setup completed successfully!
echo ================================================================
echo.
echo Your 2-Stage PhD Outreach Automation System is ready!
echo.
echo Quick Start Options:
echo 1. Double-click: start_phd_outreach.bat
echo 2. Or run: streamlit run streamlit_app.py
echo.
echo Cost-Optimized Features:
echo - Stage 1: Smart discovery with gpt-4o-mini (~$0.01-0.05/university)
echo - Stage 2: Quality emails with gpt-4 (~$0.10-0.30/email)
echo - Real-time cost tracking
echo - Windows 11 optimized performance
echo.

choice /C YN /M "Would you like to start the application now"
if errorlevel 2 goto end
if errorlevel 1 goto start

:start
echo.
echo Starting PhD Outreach Automation System...
echo Your browser should open automatically.
echo.
streamlit run streamlit_app.py --server.port 8501
goto end

:end
echo.
echo Thank you for using PhD Outreach Automation!
echo For support, check the logs in phd_outreach.log
pause.py >> start_app.bat

echo Quick start script created: start_app.bat
pause