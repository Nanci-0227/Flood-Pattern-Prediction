@echo off
setlocal
cd /d "%~dp0project"
py -3 app.py
if errorlevel 1 (
  echo.
  echo 启动失败：请确认已安装 Python，并在 project 目录安装 requirements.txt 中的依赖。
  pause
)
