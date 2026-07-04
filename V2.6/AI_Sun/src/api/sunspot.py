"""
Sunspot Detection API Router

Provides endpoints for solar disk boundary detection and sunspot
identification using computer vision techniques.
"""

import os
import json
import uuid
import base64
import cv2
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List

import logging
logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
SUNSPOT_DIR = DATA_DIR / "sunspot_results"

os.makedirs(SUNSPOT_DIR, exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR))
from sunspot_detector import SunspotDetectionPipeline, detection_to_csv

from persistent_store import get_images_store

images_store = get_images_store()
pipeline = SunspotDetectionPipeline()


@router.post("/sunspot/detect/{image_id}", tags=["黑子检测"])
async def detect_sunspots(
    image_id: str,
    min_spot_area: Optional[float] = None,
    max_spot_area: Optional[float] = None,
    adaptive_block_size: Optional[int] = None,
    adaptive_c: Optional[float] = None,
):
    """Detect sunspots in a previously uploaded solar image.

    Returns solar disk boundary, sunspot count, individual spot positions,
    region classification (center/mid-latitude/limb), and annotated image.

    Query parameters allow tuning detection sensitivity.
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

    # Build params dict from optional query params
    params = {}
    if min_spot_area is not None:
        params["min_spot_area"] = min_spot_area
    if max_spot_area is not None:
        params["max_spot_area"] = max_spot_area
    if adaptive_block_size is not None:
        params["adaptive_block_size"] = adaptive_block_size
    if adaptive_c is not None:
        params["adaptive_c"] = adaptive_c

    try:
        result, annotated_img = pipeline.process_with_annotation(
            file_path, output_dir=str(ANNOTATED_DIR), **params,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "code": "DETECTION_FAILED", "message": str(e),
        })
    except Exception as e:
        logger.error(f"Sunspot detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "code": "INTERNAL_ERROR", "message": f"检测失败: {str(e)}",
        })

    # Save detection result to persistent storage
    result_dict = result.to_dict()
    result_id = f"sunspot-{image_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    sunspot_record = {
        "id": result_id,
        "image_id": image_id,
        "timestamp": result.detection_timestamp,
        "total_spots": result.total_spots,
        "spots_by_region": result.spots_by_region,
        "solar_disk": result_dict["solar_disk"],
        "sunspots": result_dict["sunspots"],
        "processing_time_ms": result.processing_time_ms,
    }

    # Save to file
    record_path = SUNSPOT_DIR / f"{result_id}.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(sunspot_record, f, ensure_ascii=False, indent=2, default=str)

    # Generate annotated image URL
    annotated_url = None
    annotated_path = result.debug_info.get("annotated_path", "")
    if annotated_path and os.path.exists(annotated_path):
        annotated_url = f"/api/v1/sunspot/image/{result_id}"

    # Encode annotated image as base64 for immediate display
    annotated_b64 = None
    if annotated_img is not None:
        _, buffer = cv2.imencode(".png", annotated_img)
        annotated_b64 = base64.b64encode(buffer).decode()

    return {
        "success": True,
        "data": {
            **result_dict,
            "result_id": result_id,
            "annotated_image_url": annotated_url,
            "annotated_image_base64": f"data:image/png;base64,{annotated_b64}" if annotated_b64 else None,
            "csv_summary": detection_to_csv(result),
        },
        "message": f"检测完成，共发现 {result.total_spots} 个黑子",
    }


@router.get("/sunspot/image/{result_id}", tags=["黑子检测"])
async def get_sunspot_annotated_image(result_id: str):
    """Get the annotated sunspot detection image."""
    # Look up result record to find the associated image
    record_path = SUNSPOT_DIR / f"{result_id}.json"
    if not record_path.exists():
        raise HTTPException(status_code=404, detail={
            "code": "NOT_FOUND", "message": "标注结果记录不存在",
        })

    with open(record_path, "r") as f:
        record = json.load(f)

    # Find annotated image: look for the most recent sunspot_*_annotated.png
    # that matches the record's image_id timestamp
    candidates = sorted(ANNOTATED_DIR.glob("sunspot_*_annotated.png"),
                        key=lambda p: p.stat().st_mtime, reverse=True)

    if candidates:
        annotated_path = candidates[0]
        return FileResponse(str(annotated_path), media_type="image/png")

    raise HTTPException(status_code=404, detail={
        "code": "FILE_NOT_FOUND", "message": "标注图像文件不存在",
    })


@router.get("/sunspot/history", tags=["黑子检测"])
async def get_sunspot_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    """List historical sunspot detection results."""
    results = []
    for f in sorted(SUNSPOT_DIR.glob("sunspot-*.json"), reverse=True):
        try:
            with open(f, "r") as fh:
                results.append(json.load(fh))
        except Exception:
            continue

    total = len(results)
    items = results[(page - 1) * limit : page * limit]

    # Add image info
    for item in items:
        img = images_store.get(item.get("image_id", ""))
        if img:
            item["filename"] = img.get("filename", "")

    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": items,
        },
    }


@router.get("/sunspot/statistics", tags=["黑子检测"])
async def get_sunspot_statistics():
    """Get aggregated statistics across all sunspot detections."""
    total_detections = 0
    total_spots = 0
    region_totals = {"center": 0, "mid_latitude": 0, "limb": 0}
    spot_confidences = []

    for f in SUNSPOT_DIR.glob("sunspot-*.json"):
        try:
            with open(f, "r") as fh:
                record = json.load(fh)
            total_detections += 1
            total_spots += record.get("total_spots", 0)
            for region, count in record.get("spots_by_region", {}).items():
                if region in region_totals:
                    region_totals[region] += count
            for spot in record.get("sunspots", []):
                spot_confidences.append(spot.get("confidence", 0))
        except Exception:
            continue

    avg_confidence = (
        sum(spot_confidences) / len(spot_confidences)
        if spot_confidences
        else 0
    )

    return {
        "success": True,
        "data": {
            "total_detections": total_detections,
            "total_spots": total_spots,
            "average_spots_per_image": round(total_spots / max(total_detections, 1), 2),
            "spots_by_region": region_totals,
            "average_confidence": round(avg_confidence, 4),
        },
    }
