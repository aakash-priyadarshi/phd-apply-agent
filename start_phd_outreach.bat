@echo off
title PhD Outreach Automation - 2-Stage System

echo.
echo ================================================
echo  PhD Outreach Automation - 2-Stage System  
echo ================================================
echo.

REM Activate virtual environment if it exists
if exist "phd_outreach_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call phd_outreach_env\Scripts\activate.bat
)

echo Starting PhD Outreach System...
echo Your browser will open automatically.
echo.

streamlit run streamlit_app.py --server.port 8501

pause
