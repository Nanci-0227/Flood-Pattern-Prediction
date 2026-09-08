@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0project\app.py"
if errorlevel 1 pause
