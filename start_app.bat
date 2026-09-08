@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo.
  echo 启动失败：未找到项目虚拟环境 .venv。
  echo 请在项目根目录执行：py -3 -m venv .venv
  pause
  exit /b 1
)

cd /d "%ROOT%project"
"%PYTHON%" app.py
if errorlevel 1 pause
