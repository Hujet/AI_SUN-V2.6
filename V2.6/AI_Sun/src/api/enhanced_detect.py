"""
Enhanced Detection API Router

Provides endpoints for:
1. Multi-scale enhanced detection
2. Prominence detection at solar limb
3. Synchronized image preservation
4. Performance evaluation and comparison
"""

import os
import json
import uuid
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List

import logging
logger = logging.getLogger(__name__)

from solar_preprocessor import detect_solar_disk

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

import sys
sys.path.insert(0, str(BASE_DIR))
from advanced_detector import AdvancedDetectionPipeline
from enhanced_detector import EnhancedDetectionPipeline, generate_detection_report
from image_preservation import ImagePreservationManager
from persistent_store import get_images_store

images_store = get_images_store()
advanced_pipeline = AdvancedDetectionPipeline()
enhanced_pipeline = EnhancedDetectionPipeline()
preservation_manager = ImagePreservationManager()


@router.post("/enhanced/detect/{image_id}", tags=["增强检测"])
async def enhanced_detect(
    image_id: str,
    enable_scales: Optional[str] = Query(None, description="Comma-separated scales: original,medium,fine"),
    enable_prominence: bool = Query(True, description="Enable prominence detection"),
    save_original: bool = Query(True, description="Save original image copy"),
):
    """Run enhanced multi-scale detection on a solar image.
    
    Returns:
    - All detected features with quality metrics
    - Detection statistics
    - Annotated image (base64)
    - Synchronized preservation paths
    """
    img_data = images_store.get(image_id)
    if not img_data:
        raise HTTPException(status_code=404, detail={
            "code": "IMAGE_NOT_FOUND", "message": "图像不存在",
        })
    
    file_path = img_data.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail={
            "code": "FILE_NOT_FOUND", "message": "图像文件不存在",
        })
    
    try:
        # Load image
        image_bytes = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法加载图像")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Parse scales
        scales = None
        if enable_scales:
            scales = [s.strip() for s in enable_scales.split(",")]
        
        # Run actual solar disk detection instead of using hardcoded values
        disk_info = detect_solar_disk(gray)
        if not disk_info.get("detected"):
            logger.warning(f"Solar disk detection failed for {image_id}, using fallback center/radius")
            disk_info = {
                "detected": True,
                "center_x": gray.shape[1] / 2,
                "center_y": gray.shape[0] / 2,
                "radius": min(gray.shape) * 0.45,
                "method": "fallback_center",
                "confidence": 0.3,
            }
        
        result = enhanced_pipeline.detect(gray, disk_info)
        result.image_path = file_path
        
        # Generate annotated image
        annotated_path = str(DATA_DIR / "annotated" / f"enhanced_{image_id}.png")
        enhanced_pipeline.generate_annotated_image(gray, result, annotated_path)
        
        # Load annotated image for base64
        annotated_img = cv2.imread(annotated_path)
        annotated_b64 = None
        if annotated_img is not None:
            _, buffer = cv2.imencode(".png", annotated_img)
            annotated_b64 = base64.b64encode(buffer).decode()
        
        # Generate report
        report = generate_detection_report(result)
        
        # Synchronized preservation
        preservation_paths = {}
        if save_original:
            preservation_paths = preservation_manager.save_all(
                image_id=image_id,
                original_path=file_path,
                annotated_array=annotated_img,
                report_content=report,
                metadata=result.to_dict(),
                debug_images=enhanced_pipeline.debug_images,
            )
        
        return {
            "success": True,
            "data": {
                "result": result.to_dict(),
                "annotated_image_base64": f"data:image/png;base64,{annotated_b64}" if annotated_b64 else None,
                "preservation_paths": preservation_paths,
                "report_preview": report[:500] + "..." if len(report) > 500 else report,
            },
            "message": f"增强检测完成，共发现 {result.statistics.total_features} 个特征",
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "code": "DETECTION_FAILED", "message": str(e),
        })
    except Exception as e:
        logger.error(f"Enhanced detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "code": "INTERNAL_ERROR", "message": f"检测失败: {str(e)}",
        })


@router.get("/enhanced/sessions", tags=["增强检测"])
async def list_enhanced_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent enhanced detection sessions."""
    sessions = preservation_manager.get_session_list(limit=limit)
    
    total = len(sessions)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": sessions[start_idx:end_idx],
        },
    }


@router.get("/enhanced/session/{session_id}", tags=["增强检测"])
async def get_session_details(session_id: str):
    """Get full details of an enhanced detection session."""
    details = preservation_manager.get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail={
            "code": "SESSION_NOT_FOUND", "message": "会话不存在",
        })
    
    return {
        "success": True,
        "data": details,
    }


@router.get("/enhanced/session/{session_id}/image/{image_type}", tags=["增强检测"])
async def get_session_image(session_id: str, image_type: str):
    """Get original or annotated image from a session.
    
    image_type: 'original' or 'annotated'
    """
    image_path = preservation_manager.get_session_image(session_id, image_type)
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail={
            "code": "IMAGE_NOT_FOUND", "message": f"{image_type} 图像不存在",
        })
    
    return FileResponse(image_path, media_type="image/png")


@router.post("/enhanced/compare/{image_id}", tags=["性能评估"])
async def compare_detection_methods(image_id: str):
    """Compare standard vs enhanced detection methods.
    
    Runs both detection pipelines and returns performance comparison.
    """
    img_data = images_store.get(image_id)
    if not img_data:
        raise HTTPException(status_code=404, detail={
            "code": "IMAGE_NOT_FOUND", "message": "图像不存在",
        })
    
    file_path = img_data.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail={
            "code": "FILE_NOT_FOUND", "message": "图像文件不存在",
        })
    
    try:
        # Load image
        image_bytes = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Standard detection (existing pipeline)
        from sunspot_detector import SunspotDetectionPipeline
        standard_pipeline = SunspotDetectionPipeline()
        standard_result = standard_pipeline.process(file_path)
        
        # Enhanced detection (new pipeline)
        disk_info = {
            "center_x": standard_result.solar_disk.center_x,
            "center_y": standard_result.solar_disk.center_y,
            "radius": standard_result.solar_disk.radius_px,
            "method": standard_result.solar_disk.method_used,
            "confidence": standard_result.solar_disk.detection_confidence,
        }
        enhanced_result = enhanced_pipeline.detect(gray, disk_info)
        
        # Comparison metrics
        comparison = {
            "standard": {
                "total_features": standard_result.total_spots,
                "processing_time_ms": standard_result.processing_time_ms,
                "features_by_region": standard_result.spots_by_region,
                "average_confidence": round(
                    sum(s.confidence for s in standard_result.sunspots) / 
                    max(len(standard_result.sunspots), 1), 4
                ),
            },
            "enhanced": {
                "total_features": enhanced_result.statistics.total_features,
                "processing_time_ms": enhanced_result.statistics.processing_time_ms,
                "features_by_type": enhanced_result.statistics.features_by_type,
                "features_by_scale": enhanced_result.statistics.features_by_scale,
                "features_by_region": enhanced_result.statistics.features_by_region,
                "average_confidence": enhanced_result.statistics.average_confidence,
                "average_quality": enhanced_result.statistics.average_quality,
            },
            "improvement": {
                "feature_count_change": enhanced_result.statistics.total_features - standard_result.total_spots,
                "processing_time_change": enhanced_result.statistics.processing_time_ms - standard_result.processing_time_ms,
            },
        }
        
        return {
            "success": True,
            "data": {
                "comparison": comparison,
                "standard_details": standard_result.to_dict(),
                "enhanced_details": enhanced_result.to_dict(),
            },
            "message": "对比分析完成",
        }
    
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "code": "INTERNAL_ERROR", "message": f"对比失败: {str(e)}",
        })
