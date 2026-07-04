@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo  太阳活动区自动分析系统 - 环境安装与启动脚本
echo  Solar Active Region Auto-Analysis System - Setup ^& Launch
echo ============================================================
echo.

REM --- 检测 Python ---
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 Python，请先安装 Python 3.9+ 
    echo         下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [INFO] 检测到 Python 版本: %PYVER%

REM --- 确定项目根目录 ---
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM --- 检查/创建虚拟环境 ---
set "VENV_DIR=%PROJECT_DIR%.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] 正在创建虚拟环境...
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo [INFO] 虚拟环境创建成功
)

REM --- 激活虚拟环境 ---
call "%VENV_DIR%\Scripts\activate.bat"
echo [INFO] 已激活虚拟环境: %VENV_DIR%

REM --- 安装依赖 ---
echo [INFO] 正在安装项目依赖...
pip install -r requirements.txt -q
if !errorlevel! neq 0 (
    echo [ERROR] 依赖安装失败，请检查 requirements.txt
    pause
    exit /b 1
)
echo [INFO] 依赖安装完成

REM --- 检查 .env 配置 ---
if not exist ".env" (
    echo [WARN] 未找到 .env 配置文件
    if exist ".env.example" (
        echo [INFO] 正在从 .env.example 创建 .env 文件...
        copy .env.example .env >nul
        echo [INFO] 已创建 .env，请编辑该文件填写您的 DeepSeek API Key
        echo         编辑命令: notepad .env
    ) else (
        echo [ERROR] .env.example 模板文件不存在！
        pause
        exit /b 1
    )
)

REM --- 启动服务 ---
echo.
echo [INFO] 启动服务于 http://localhost:8000
echo [INFO] API 文档: http://localhost:8000/docs
echo [INFO] 按 Ctrl+C 停止服务
echo.

python src\app.py

pause
