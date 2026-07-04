"""
CV File Upload and Management API

Provides endpoints for uploading, listing, viewing, and deleting CV (Computer Vision)
files such as preprocessing reports, intermediate results, and analysis data.
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/cv-files", tags=["CV文件管理"])

# Storage configuration
BASE_DIR = Path(__file__).parent.parent
CV_FILES_DIR = BASE_DIR / "data" / "cv_files"
CV_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Index file to track uploaded files
INDEX_FILE = CV_FILES_DIR / "index.json"


def _load_index() -> List[dict]:
    """Load the file index from disk."""
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_index(index: List[dict]):
    """Save the file index to disk."""
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _detect_file_type(filename: str) -> str:
    """Detect the file type based on extension."""
    ext = Path(filename).suffix.lower()
    type_map = {
        '.json': 'json',
        '.txt': 'txt',
        '.csv': 'csv',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.tif': 'image',
        '.tiff': 'image',
        '.pdf': 'pdf',
    }
    return type_map.get(ext, 'other')


@router.post("/upload")
async def upload_cv_file(file: UploadFile = File(...)):
    """Upload a CV file."""
    # Validate file size (5MB max)
    MAX_SIZE = 5 * 1024 * 1024
    file_id = str(uuid.uuid4())[:12]
    filename = file.filename or f"cv_file_{file_id}"
    file_type = _detect_file_type(filename)

    # Read file content
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过5MB限制")

    # Save file to disk
    file_path = CV_FILES_DIR / f"{file_id}_{filename}"
    with open(file_path, 'wb') as f:
        f.write(content)

    # Extract text content for preview (for text-based files)
    text_content = None
    if file_type in ['json', 'txt', 'csv']:
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('gbk')
            except UnicodeDecodeError:
                text_content = "[无法解码文本内容]"

    # Update index
    index = _load_index()
    file_record = {
        'id': file_id,
        'filename': filename,
        'file_type': file_type,
        'file_size': len(content),
        'uploaded_at': datetime.now().isoformat(),
        'content': text_content,  # Only for text-based files
    }
    index.append(file_record)
    _save_index(index)

    return {
        'success': True,
        'data': {
            'id': file_id,
            'filename': filename,
            'file_type': file_type,
            'file_size': len(content),
        }
    }


@router.get("")
async def list_cv_files(type: Optional[str] = Query(None, description="Filter by file type")):
    """List all uploaded CV files."""
    index = _load_index()

    # Filter by type if specified
    if type:
        index = [f for f in index if f.get('file_type') == type]

    # Sort by upload time (newest first)
    index.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)

    # Remove content from list view for performance
    for f in index:
        f.pop('content', None)

    return {
        'success': True,
        'data': {
            'items': index,
            'total': len(index),
        }
    }


@router.get("/{file_id}")
async def get_cv_file(file_id: str):
    """Get a specific CV file's details."""
    index = _load_index()
    file_record = next((f for f in index if f['id'] == file_id), None)

    if not file_record:
        raise HTTPException(status_code=404, detail="文件不存在")

    return {
        'success': True,
        'data': file_record,
    }


@router.delete("/{file_id}")
async def delete_cv_file(file_id: str):
    """Delete a CV file."""
    index = _load_index()
    file_record = next((f for f in index if f['id'] == file_id), None)

    if not file_record:
        raise HTTPException(status_code=404, detail="文件不存在")

    # Delete file from disk
    filename = file_record['filename']
    file_path = CV_FILES_DIR / f"{file_id}_{filename}"
    if file_path.exists():
        os.remove(file_path)

    # Remove from index
    index = [f for f in index if f['id'] != file_id]
    _save_index(index)

    return {
        'success': True,
        'message': '文件已删除',
    }
