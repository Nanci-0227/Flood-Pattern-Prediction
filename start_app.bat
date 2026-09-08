@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "APP=%ROOT%project\app.py"

if not exist "%PYTHON%" (
  echo.
  echo 启动失败：未找到项目虚拟环境 .venv。
  echo 请在项目根目录执行：py -3 -m venv .venv
  pause
  exit /b 1
)

if not exist "%APP%" (
  echo.
  echo 启动失败：未找到界面程序 %APP%
  pause
  exit /b 1
)

"%PYTHON%" "%APP%"
if errorlevel 1 (
  echo.
  echo 界面启动失败，请保留以上错误信息。
  pause
)
