"""
Solar Active Region Auto-Analysis System - Main Application

FastAPI-based web application providing solar image upload, AI-powered
analysis, annotation, and data visualization.

Project Structure:
    app.py                    - Main entry point
    api/analyze.py            - Analysis + Reports endpoints (persistent storage)
    api/images.py             - Image upload/management (persistent storage)
    api/statistics.py         - Data visualization & statistics API
    api/data.py               - Helioviewer data acquisition API
    api/cases.py              - Sample case library API
    api/remote_images.py      - Remote solar image API
    persistent_store.py       - Thread-safe JSON file storage
    deepseek_client.py        - DeepSeek API client
    solar_classifier.py       - Solar feature classifier
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import logging
from datetime import datetime

from api.images import router as images_router
from api.analyze import router as analyze_router
from api.statistics import router as statistics_router
from api.cases import router as cases_router
from api.data import router as data_router
from api.remote_images import router as remote_images_router
from api.sunspot import router as sunspot_router
from api.api_keys import router as api_keys_router
from api.batch_comparison import router as batch_comparison_router
from api.cv_files import router as cv_files_router
from api.models import router as models_router
from api.feedback import router as feedback_router
from api.enhanced_detect import router as enhanced_detect_router

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="太阳活动区自动分析系统 API",
    description="基于多模态大语言模型的太阳活动图像分析服务",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
# NOTE: "*" is excluded because allow_credentials=True is set.
# When credentials are allowed, browsers reject wildcard origins.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
ANNOTATED_DIR = DATA_DIR / "annotated"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in [UPLOAD_DIR, REPORTS_DIR, ANNOTATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Error: {exc.status_code} - {exc.detail}")
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "INTERNAL_ERROR")
        message = detail.get("message", str(detail))
    else:
        code = "INTERNAL_ERROR"
        message = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {"code": "INTERNAL_ERROR", "message": "系统内部错误，请稍后重试"},
        },
    )


# ---------------------------------------------------------------------------
# Register API Routers
# ---------------------------------------------------------------------------

app.include_router(images_router, prefix="/api/v1")
app.include_router(remote_images_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")       # Includes /analyze + /reports
app.include_router(statistics_router, prefix="/api/v1")    # Statistics & visualization
app.include_router(cases_router, prefix="/api/v1")
app.include_router(data_router, prefix="/api/v1")
app.include_router(sunspot_router, prefix="/api/v1")
app.include_router(api_keys_router)  # Router has its own prefix /api/v1/api-keys
app.include_router(batch_comparison_router)
app.include_router(cv_files_router)
app.include_router(models_router)  # Router has its own prefix /api/v1/models
app.include_router(feedback_router)  # Router has its own prefix /api/v1/feedback
app.include_router(enhanced_detect_router)  # Router has its own prefix /api/v1/enhanced


@app.get("/", tags=["系统"])
async def root():
    """Serve the frontend single-page application."""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content=content, status_code=200)
    return JSONResponse(
        content={"message": "太阳活动区自动分析系统 API", "status": "running"}
    )


@app.get("/health", tags=["系统"])
async def health_check():
    """System health check endpoint with connection status details."""
    # Check DeepSeek API connectivity
    deepseek_status = "unknown"
    try:
        from deepseek_client import DeepseekAPIClient, DeepSeekConfig
        config = DeepSeekConfig.from_env()
        if config and config.api_key:
            client = DeepseekAPIClient(config=config)
            resp = client.test()
            if resp and resp.success:
                deepseek_status = "connected"
            else:
                deepseek_status = "disconnected"
        else:
            deepseek_status = "not_configured"
    except Exception:
        deepseek_status = "disconnected"

    return {
        "success": True,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "deepseek_api": deepseek_status,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"API documentation: http://localhost:{port}/docs")
    logger.info(f"Frontend: http://localhost:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")
