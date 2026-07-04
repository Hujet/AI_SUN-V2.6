#!/usr/bin/env bash
# =============================================================================
# 太阳活动区自动分析系统 - 跨平台环境安装与启动脚本 (Linux/macOS)
# Solar Active Region Auto-Analysis System - Setup & Launch Script
# =============================================================================
set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} 太阳活动区自动分析系统 - 环境安装与启动${NC}"
echo -e "${GREEN} Solar Active Region Auto-Analysis System${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Determine project root directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 1. Check Python
# ---------------------------------------------------------------------------
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}[ERROR] Python 未安装，请先安装 Python 3.9+${NC}"
        echo -e "         macOS: brew install python@3.11"
        echo -e "         Linux: sudo apt install python3 python3-venv python3-pip"
        echo -e "         下载: https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON=python
else
    PYTHON=python3
fi

PYVER=$($PYTHON --version 2>&1)
echo -e "${BLUE}[INFO]${NC} 检测到 Python: $PYVER"

# ---------------------------------------------------------------------------
# 2. Create/activate virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/python" ] && [ ! -f "$VENV_DIR/Scripts/python.exe" ]; then
    echo -e "${BLUE}[INFO]${NC} 正在创建虚拟环境..."
    $PYTHON -m venv "$VENV_DIR"
    echo -e "${GREEN}[INFO]${NC} 虚拟环境创建成功: $VENV_DIR"
fi

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
fi
echo -e "${GREEN}[INFO]${NC} 已激活虚拟环境"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo -e "${BLUE}[INFO]${NC} 正在安装项目依赖..."
pip install -r requirements.txt -q
echo -e "${GREEN}[INFO]${NC} 依赖安装完成"

# ---------------------------------------------------------------------------
# 4. Check .env configuration
# ---------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[WARN]${NC} 未找到 .env 配置文件"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}[INFO]${NC} 已从 .env.example 创建 .env 文件"
        echo -e "${YELLOW}[WARN]${NC} 请编辑 .env 文件填写您的 DeepSeek API Key:"
        echo -e "         nano .env  或  vim .env"
        echo ""
        echo -e "         获取 API Key: https://platform.deepseek.com/"
    else
        echo -e "${RED}[ERROR]${NC} .env.example 模板文件不存在！"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 5. Launch server
# ---------------------------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo ""
echo -e "${GREEN}[INFO]${NC} 服务启动于 http://localhost:${PORT}"
echo -e "${GREEN}[INFO]${NC} API 文档: http://localhost:${PORT}/docs"
echo -e "${GREEN}[INFO]${NC} 按 Ctrl+C 停止服务"
echo ""

python3 src/app.py
