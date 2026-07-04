"""
AI Model Management API

Provides endpoints for:
- Listing available AI models
- Testing model connections
- Analyzing images with selected models
- Comparing results from multiple models
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_model_adapter import (
    get_model_manager,
    ModelConfig,
    AIAnalysisResult,
)
from solar_preprocessor import preprocess_solar_image, generate_feature_prompt

router = APIRouter(prefix="/api/v1/models", tags=["AI模型管理"])
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Request model for image analysis."""
    image_id: str
    model_key: str = "deepseek"
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


class CompareRequest(BaseModel):
    """Request model for multi-model comparison."""
    image_id: str
    model_keys: Optional[list] = None  # If None, use all available models


@router.get("/available")
async def get_available_models(test_connections: bool = Query(False, description="是否测试所有模型连接")):
    """Get list of all available AI models with optional connection status."""
    manager = get_model_manager()
    models = manager.get_available_models()
    
    # Only test connections if explicitly requested
    if test_connections:
        for model in models:
            if model['available']:
                try:
                    adapter = manager.get_model(model['key'])
                    if adapter:
                        is_connected = adapter.test_connection()
                        model['connected'] = is_connected
                        model['status'] = 'connected' if is_connected else 'error'
                except Exception as e:
                    model['connected'] = False
                    model['status'] = f'error: {str(e)[:50]}'
            else:
                model['connected'] = False
                model['status'] = 'not_configured'
    else:
        # Set default status without testing
        for model in models:
            if model['available']:
                model['status'] = 'configured'  # API key exists, connection not tested
            else:
                model['status'] = 'not_configured'
                model['connected'] = False
    
    return {
        'success': True,
        'data': {
            'models': models,
            'default_model': 'deepseek',
        }
    }


@router.post("/test/{model_key}")
async def test_model_connection(model_key: str):
    """Test connection to a specific model."""
    manager = get_model_manager()
    adapter = manager.get_model(model_key)
    
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    start_time = time.time()
    success = adapter.test_connection()
    elapsed = (time.time() - start_time) * 1000
    
    return {
        'success': success,
        'data': {
            'model_key': model_key,
            'connected': success,
            'response_time_ms': round(elapsed, 2),
        },
        'message': '连接成功' if success else '连接失败',
    }


@router.post("/test-all")
async def test_all_connections():
    """Test connections to all configured models."""
    manager = get_model_manager()
    results = manager.test_all_connections()
    
    return {
        'success': True,
        'data': {
            'results': results,
            'total': len(results),
            'connected': sum(1 for v in results.values() if v),
        }
    }


@router.post("/analyze")
async def analyze_with_model(request: AnalyzeRequest):
    """Analyze an image with a specific AI model."""
    from app import UPLOAD_DIR
    
    manager = get_model_manager()
    adapter = manager.get_model(request.model_key)
    
    if not adapter:
        raise HTTPException(status_code=400, detail=f"模型 '{request.model_key}' 不可用或未配置")
    
    # Check if model supports vision
    model_config = manager.get_available_models()
    model_info = next((m for m in model_config if m['key'] == request.model_key), None)
    
    if model_info and model_info.get('supports_vision'):
        # Vision model - validate original photo
        pass  # Frontend already validates, but we can add additional server-side checks
    
    # Find the image file
    image_path = UPLOAD_DIR / f"{request.image_id}"
    
    # Try to find the actual image file
    if not image_path.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            test_path = UPLOAD_DIR / f"{request.image_id}{ext}"
            if test_path.exists():
                image_path = test_path
                break
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图像 '{request.image_id}' 不存在")
    
    # Validate image file on server side
    try:
        file_size = os.path.getsize(image_path)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图像文件过大（超过10MB限制）")
        
        if file_size < 10000:
            logger.warning(f"Image file is very small ({file_size} bytes), may be compressed")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Could not validate image file: {e}")
    
    # Run CV preprocessing
    try:
        import cv2
        import numpy as np
        from solar_preprocessor import preprocess_solar_image, generate_feature_prompt
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("无法加载图像文件，请确保上传的是有效的图像格式")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_result = preprocess_solar_image(gray)
        cv_report = generate_feature_prompt(cv_result)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"CV preprocessing failed: {e}")
        cv_report = ""
    
    # Use default prompts if not provided
    system_prompt = request.system_prompt or """你是一个专业的太阳活动分析AI助手。请分析提供的太阳图像并输出结构化的JSON结果。

输出格式必须包含以下字段：
- hale_classification: Hale分类 (Alpha/Beta/Beta-Gamma/Gamma/Delta/Beta-Delta/Unknown)
- classification_confidence: 分类置信度 (0-1)
- complexity_score: 复杂度评分 (0-10)
- risk_level: 风险等级 (low/moderate/high)
- features: 特征列表，每个特征包含type, position, confidence等
- summary: 分析摘要
- warnings: 注意事项列表
- recommendations: 建议措施列表"""

    user_prompt = request.user_prompt or """请分析此太阳图像，识别所有太阳活动特征（黑子、耀斑、亮区等），并提供Hale分类和风险评估。

请以JSON格式输出结果，确保包含所有必需字段。"""
    
    # Analyze with selected model
    result = manager.analyze_with_model(
        model_key=request.model_key,
        image_path=str(image_path),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cv_report=cv_report,
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    
    return {
        'success': True,
        'data': result.to_dict(),
        'cv_report': cv_report,
    }


@router.post("/compare")
async def compare_models(request: CompareRequest):
    """Compare analysis results from multiple models."""
    from app import UPLOAD_DIR
    
    manager = get_model_manager()
    
    # Find the image file
    image_path = UPLOAD_DIR / f"{request.image_id}"
    
    if not image_path.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            test_path = UPLOAD_DIR / f"{request.image_id}{ext}"
            if test_path.exists():
                image_path = test_path
                break
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{request.image_id}' not found")
    
    # Run CV preprocessing
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv_result = preprocess_solar_image(gray)
        cv_report = generate_feature_prompt(cv_result)
        
    except Exception as e:
        logger.error(f"CV preprocessing failed: {e}")
        cv_report = ""
    
    # Default prompts
    system_prompt = """你是一个专业的太阳活动分析AI助手。请分析提供的太阳图像并输出结构化的JSON结果。

输出格式必须包含以下字段：
- hale_classification: Hale分类 (Alpha/Beta/Beta-Gamma/Gamma/Delta/Beta-Delta/Unknown)
- classification_confidence: 分类置信度 (0-1)
- complexity_score: 复杂度评分 (0-10)
- risk_level: 风险等级 (low/moderate/high)
- features: 特征列表，每个特征包含type, position, confidence等
- summary: 分析摘要
- warnings: 注意事项列表
- recommendations: 建议措施列表"""

    user_prompt = """请分析此太阳图像，识别所有太阳活动特征（黑子、耀斑、亮区等），并提供Hale分类和风险评估。

请以JSON格式输出结果，确保包含所有必需字段。"""
    
    # Determine which models to use
    if request.model_keys:
        model_keys = request.model_keys
    else:
        # Use all available models
        available = manager.get_available_models()
        model_keys = [m['key'] for m in available if m['available']]
    
    # Analyze with each model
    results = {}
    for model_key in model_keys:
        logger.info(f"Analyzing with model: {model_key}")
        result = manager.analyze_with_model(
            model_key=model_key,
            image_path=str(image_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cv_report=cv_report,
        )
        results[model_key] = result.to_dict()
    
    # Compare results
    result_objects = {}
    for key, result_dict in results.items():
        result_obj = AIAnalysisResult(**result_dict)
        result_objects[key] = result_obj
    
    comparison = manager.compare_results(result_objects)
    
    return {
        'success': True,
        'data': {
            'individual_results': results,
            'comparison': comparison,
        }
    }


@router.get("/models/{model_key}/config")
async def get_model_config(model_key: str):
    """Get configuration details for a specific model."""
    manager = get_model_manager()
    models = manager.get_available_models()
    
    model = next((m for m in models if m['key'] == model_key), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    return {
        'success': True,
        'data': model,
    }
