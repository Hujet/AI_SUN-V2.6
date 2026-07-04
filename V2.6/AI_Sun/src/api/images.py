"""
Image Management API Router

Handles image upload, listing, retrieval and deletion with persistent storage.
"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR))
from persistent_store import get_images_store

images_store = get_images_store()


def generate_image_id() -> str:
    return f"img-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


@router.post("/images", tags=["图像管理"])
async def upload_image(
    image: UploadFile = File(...),
    source: Optional[str] = None,
    wavelength: Optional[str] = None,
    timestamp: Optional[str] = None,
):
    """Upload a solar image for analysis.

    Supported formats: JPG, JPEG, PNG, TIFF, TIF.
    Maximum file size is governed by MAX_IMAGE_SIZE_MB in .env (default 10MB).
    """
    allowed_extensions = {"jpg", "jpeg", "png", "tiff", "tif"}
    ext = image.filename.split(".")[-1].lower() if "." in image.filename else ""

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE",
                "message": f"不支持的文件格式 .{ext}，请上传 JPG、PNG 或 TIFF 格式的图像",
            },
        )

    # Check file size
    max_size_mb = int(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
    contents = await image.read()
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"文件大小超过限制 ({max_size_mb}MB)，请压缩后重新上传",
            },
        )

    image_id = generate_image_id()
    file_path = UPLOAD_DIR / f"{image_id}.{ext}"

    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": f"文件保存失败: {str(e)}"},
        )

    image_info = {
        "id": image_id,
        "filename": image.filename,
        "source": source or "user_upload",
        "wavelength": wavelength or "unknown",
        "timestamp": timestamp or datetime.now().isoformat(),
        "size": len(contents),
        "status": "uploaded",
        "created_at": datetime.now().isoformat(),
        "file_path": str(file_path),
    }

    images_store.set(image_id, image_info)

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "data": {k: v for k, v in image_info.items() if k != "file_path"},
            "message": "图像上传成功",
        },
    )


@router.get("/images", tags=["图像管理"])
async def get_images(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    source: Optional[str] = None,
    status: Optional[str] = None,
):
    """List all uploaded images with pagination and filtering."""
    filtered = images_store.list_all()

    if source:
        filtered = [img for img in filtered if img.get("source") == source]
    if status:
        filtered = [img for img in filtered if img.get("status") == status]

    # Sort by creation time descending
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    items = filtered[start:end]

    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": [
                {k: v for k, v in item.items() if k != "file_path"} for item in items
            ],
        },
    }


@router.get("/images/{image_id}", tags=["图像管理"])
async def get_image(image_id: str):
    """Get information about a specific uploaded image."""
    image_info = images_store.get(image_id)
    if not image_info:
        raise HTTPException(
            status_code=404,
            detail={"code": "IMAGE_NOT_FOUND", "message": "指定的图像不存在"},
        )
    return {
        "success": True,
        "data": {k: v for k, v in image_info.items() if k != "file_path"},
    }


@router.get("/images/{image_id}/file", tags=["图像管理"])
async def get_image_file(image_id: str):
    """Get the actual image file for display."""
    image_info = images_store.get(image_id)
    if not image_info:
        raise HTTPException(
            status_code=404,
            detail={"code": "IMAGE_NOT_FOUND", "message": "指定的图像不存在"},
        )

    file_path = image_info.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail={"code": "FILE_NOT_FOUND", "message": "图像文件不存在"},
        )

    ext = Path(file_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return FileResponse(file_path, media_type=media_type_map.get(ext, "image/jpeg"))


@router.delete("/images/{image_id}", tags=["图像管理"])
async def delete_image(image_id: str):
    """Delete an uploaded image and associated reports."""
    image_info = images_store.get(image_id)
    if not image_info:
        raise HTTPException(
            status_code=404,
            detail={"code": "IMAGE_NOT_FOUND", "message": "指定的图像不存在"},
        )

    # Delete physical file
    file_path = image_info.get("file_path", "")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    images_store.delete(image_id)

    return {"success": True, "message": "图像已删除"}
