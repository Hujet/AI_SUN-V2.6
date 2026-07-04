"""
Reports API Router

Provides backward-compatible endpoints for report management.
Primary report functionality is now in analyze.py.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter()

# Re-export: the actual /reports endpoints are now in analyze.py
# This router is kept for backward compatibility and register purposes.
# It delegates to analyze.py via the same prefix.


@router.get("/reports", tags=["报告管理"])
async def get_reports_legacy(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    risk_level: Optional[str] = None,
):
    """DEPRECATED: Reports are now served from /api/v1/reports via analyze router."""
    # This will be handled by analyze.py's /reports endpoint
    # This function exists only for the router registration
    return {"success": True, "data": {"total": 0, "page": 1, "limit": limit, "items": []}}
