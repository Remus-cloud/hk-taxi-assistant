@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow README installation steps first.
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 scripts\doctor.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -X utf8 -m streamlit run streamlit_app.py %*
