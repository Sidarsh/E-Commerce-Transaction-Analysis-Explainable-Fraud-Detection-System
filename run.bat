@echo off
cd /d "%~dp0"
echo Starting E-commerce Fraud Detection Dashboard...
.\.venv\Scripts\streamlit.exe run "Project IDE\project_dashboard.py"
pause

