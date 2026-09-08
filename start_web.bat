@echo off
start "" /b "%~dp0.venv\Scripts\python.exe" "%~dp0project\web_app.py"
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:5000
