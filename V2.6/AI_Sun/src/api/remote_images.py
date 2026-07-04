import os
import sys
import uuid
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from helioviewer_client import HelioviewerClient, get_source_id_map
from persistent_store import get_images_store

images_store = get_images_store()

helioviewer_client = HelioviewerClient()

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def generate_image_id() -> str:
    return f"img-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

@router.get("/images/remote/magnetogram", tags=["远程图像"])
async def download_magnetogram(
    date: str = Query(..., description="日期 (YYYY-MM-DD或YYYY-MM-DDTHH:MM:SS)"),
    save_to_local: bool = Query(False, description="是否保存到本地")
):
    try:
        filepath, data = helioviewer_client.downloadMagnetogram(date)
        
        if save_to_local:
            image_id = generate_image_id()
            ext = "jp2"
            local_path = UPLOAD_DIR / f"{image_id}.{ext}"
            
            with open(local_path, "wb") as f:
                f.write(data)
            
            image_info = {
                "id": image_id,
                "filename": f"magnetogram_{date}.jp2",
                "source": "Helioviewer/HMI",
                "wavelength": "magnetogram",
                "timestamp": date,
                "size": len(data),
                "status": "downloaded",
                "created_at": datetime.now().isoformat(),
                "file_path": str(local_path)
            }
            images_store.set(image_id, image_info)
            
            return JSONResponse(
                status_code=201,
                content={
                    "success": True,
                    "data": {k: v for k, v in image_info.items() if k != "file_path"},
                    "message": "磁图下载并保存成功"
                }
            )
        
        return FileResponse(
            filepath,
            media_type="image/jp2",
            filename=f"magnetogram_{date}.jp2"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MAGNETOGRAM_ERROR", "message": f"下载磁图失败: {str(e)}"}
        )

@router.get("/images/remote/euv", tags=["远程图像"])
async def download_euv_image(
    date: str = Query(..., description="日期 (YYYY-MM-DD或YYYY-MM-DDTHH:MM:SS)"),
    wavelength: int = Query(171, description="波长 (94, 131, 171, 193, 211, 304, 335, 1600)"),
    save_to_local: bool = Query(False, description="是否保存到本地")
):
    try:
        filepath, data = helioviewer_client.downloadEUVImage(date, wavelength=wavelength)
        
        if save_to_local:
            image_id = generate_image_id()
            ext = "jp2"
            local_path = UPLOAD_DIR / f"{image_id}.{ext}"
            
            with open(local_path, "wb") as f:
                f.write(data)
            
            image_info = {
                "id": image_id,
                "filename": f"euv_{wavelength}_{date}.jp2",
                "source": f"Helioviewer/AIA_{wavelength}",
                "wavelength": str(wavelength),
                "timestamp": date,
                "size": len(data),
                "status": "downloaded",
                "created_at": datetime.now().isoformat(),
                "file_path": str(local_path)
            }
            images_store.set(image_id, image_info)
            
            return JSONResponse(
                status_code=201,
                content={
                    "success": True,
                    "data": {k: v for k, v in image_info.items() if k != "file_path"},
                    "message": f"EUV {wavelength}A 图像下载并保存成功"
                }
            )
        
        return FileResponse(
            filepath,
            media_type="image/jp2",
            filename=f"euv_{wavelength}_{date}.jp2"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "EUV_ERROR", "message": f"下载EUV图像失败: {str(e)}"}
        )

@router.get("/images/remote/thumbnail", tags=["远程图像"])
async def download_thumbnail(
    date: str = Query(..., description="日期 (YYYY-MM-DD或YYYY-MM-DDTHH:MM:SS)"),
    image_type: str = Query("magnetogram", description="图像类型 (magnetogram, euv171, euv193)"),
    scale: float = Query(0.5, description="缩放比例"),
    save_to_local: bool = Query(False, description="是否保存到本地")
):
    try:
        filepath, data = helioviewer_client.downloadThumbnail(date, image_type=image_type, scale=scale)
        
        if save_to_local:
            image_id = generate_image_id()
            ext = "jpg"
            local_path = UPLOAD_DIR / f"{image_id}.{ext}"
            
            with open(local_path, "wb") as f:
                f.write(data)
            
            image_info = {
                "id": image_id,
                "filename": f"thumbnail_{image_type}_{date}.jpg",
                "source": f"Helioviewer/{image_type}",
                "wavelength": image_type,
                "timestamp": date,
                "size": len(data),
                "status": "downloaded",
                "created_at": datetime.now().isoformat(),
                "file_path": str(local_path)
            }
            images_store.set(image_id, image_info)
            
            return JSONResponse(
                status_code=201,
                content={
                    "success": True,
                    "data": {k: v for k, v in image_info.items() if k != "file_path"},
                    "message": "缩略图下载并保存成功"
                }
            )
        
        return FileResponse(
            filepath,
            media_type="image/jpeg",
            filename=f"thumbnail_{image_type}_{date}.jpg"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "THUMBNAIL_ERROR", "message": f"下载缩略图失败: {str(e)}"}
        )

@router.get("/images/remote/source-info", tags=["远程图像"])
async def get_source_info():
    return {
        "success": True,
        "data": {
            "magnetogram": {"sourceId": 10, "description": "SDO/HMI Magnetogram"},
            "euv_wavelengths": {
                94: {"sourceId": 15, "description": "AIA 94A - Extreme UV"},
                131: {"sourceId": 16, "description": "AIA 131A - Extreme UV"},
                171: {"sourceId": 17, "description": "AIA 171A - Extreme UV"},
                193: {"sourceId": 18, "description": "AIA 193A - Extreme UV"},
                211: {"sourceId": 19, "description": "AIA 211A - Extreme UV"},
                304: {"sourceId": 20, "description": "AIA 304A - Extreme UV"},
                335: {"sourceId": 21, "description": "AIA 335A - Extreme UV"},
                1600: {"sourceId": 22, "description": "AIA 1600A - UV"}
            }
        }
    }