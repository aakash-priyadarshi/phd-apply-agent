@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8501
) else (
    py -3 -m streamlit run streamlit_app.py --server.port 8501
)
