import os
import sys
import json
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_acquisition import SolarDataAcquirer
from helioviewer_client import HelioviewerClient, get_source_id_map

acquirer = SolarDataAcquirer()
helioviewer_client = HelioviewerClient()

class DownloadRegionRequest(BaseModel):
    region_id: str
    region_name: str
    date_start: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    classification: Optional[str] = None

@router.get("/data/sources", tags=["数据采集"])
async def get_data_sources():
    try:
        sources = helioviewer_client.getDataSources()
        return {
            "success": True,
            "data": sources
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SOURCES_ERROR", "message": f"获取数据源失败: {str(e)}"}
        )

@router.get("/data/source-map", tags=["数据采集"])
async def get_source_map():
    return {
        "success": True,
        "data": get_source_id_map()
    }

@router.get("/data/active-regions", tags=["数据采集"])
async def get_active_regions(
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    limit: int = Query(20, ge=1, le=100)
):
    try:
        regions = acquirer.collectHistoricalEvents(start_date, end_date, limit)
        return {
            "success": True,
            "data": {
                "total": len(regions),
                "regions": regions
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "REGIONS_ERROR", "message": f"获取活动区失败: {str(e)}"}
        )

@router.post("/data/download-region", tags=["数据采集"])
async def download_region_data(request: DownloadRegionRequest):
    try:
        region = {
            "id": request.region_id,
            "name": request.region_name,
            "date_start": request.date_start,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "classification": request.classification
        }
        result = acquirer.downloadActiveRegionData(region)
        return {
            "success": True,
            "data": result,
            "message": f"成功下载活动区 {request.region_name} 的数据"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DOWNLOAD_ERROR", "message": f"下载失败: {str(e)}"}
        )

@router.post("/data/batch-download", tags=["数据采集"])
async def batch_download_regions(
    regions: List[Dict],
    limit: int = Query(10, ge=1, le=50)
):
    try:
        selected_regions = regions[:limit]
        results = acquirer.batchDownload(selected_regions)
        return {
            "success": True,
            "data": {
                "total_requested": len(selected_regions),
                "success_count": sum(1 for r in results if "error" not in r),
                "results": results
            },
            "message": f"批量下载完成，成功 {sum(1 for r in results if 'error' not in r)} / {len(selected_regions)}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "BATCH_DOWNLOAD_ERROR", "message": f"批量下载失败: {str(e)}"}
        )

@router.get("/data/image-list", tags=["数据采集"])
async def get_image_list(
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    source_id: int = Query(10, description="数据源ID")
):
    try:
        images = helioviewer_client.getImageList(start_date, end_date, source_id)
        return {
            "success": True,
            "data": {
                "total": len(images),
                "images": images
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "IMAGE_LIST_ERROR", "message": f"获取图像列表失败: {str(e)}"}
        )

@router.get("/data/most-recent", tags=["数据采集"])
async def get_most_recent_image(
    source_id: int = Query(10, description="数据源ID")
):
    try:
        image_info = helioviewer_client.getMostRecentImage(source_id)
        return {
            "success": True,
            "data": image_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MOST_RECENT_ERROR", "message": f"获取最新图像失败: {str(e)}"}
        )