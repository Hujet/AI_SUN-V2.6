from fastapi import APIRouter, Query, HTTPException
from typing import Optional

router = APIRouter()

mock_cases = [
    {
        "id": "case-001",
        "title": "NOAA 3512 黑子群",
        "type": "sunspot",
        "source": "SDO/HMI",
        "wavelength": "171A",
        "description": "2026年3月20日观测到的大型黑子群，具有beta-gamma磁结构，伴随M级耀斑活动",
        "timestamp": "2026-03-20T12:00:00Z",
        "image_url": "/cases/case-001.jpg",
        "ground_truth": {
            "sunspots": [
                {"id": "NOAA 3512", "magnetic_type": "beta-gamma", "complexity": "high"}
            ],
            "flares": [{"type": "M5.2", "time": "2026-03-20T11:30:00Z"}]
        }
    },
    {
        "id": "case-002",
        "title": "2026年4月15日耀斑事件",
        "type": "flare",
        "source": "SDO/AIA",
        "wavelength": "193A",
        "description": "X级耀斑爆发事件，伴随日冕物质抛射",
        "timestamp": "2026-04-15T08:45:00Z",
        "image_url": "/cases/case-002.jpg",
        "ground_truth": {
            "flares": [{"type": "X2.3", "time": "2026-04-15T08:45:00Z"}]
        }
    },
    {
        "id": "case-003",
        "title": "北极日珥爆发",
        "type": "prominence",
        "source": "SOHO/LASCO",
        "wavelength": "white light",
        "description": "大型日珥结构爆发事件",
        "timestamp": "2026-05-10T14:20:00Z",
        "image_url": "/cases/case-003.jpg",
        "ground_truth": {
            "bright_regions": [{"type": "prominence", "size": "large"}]
        }
    },
    {
        "id": "case-004",
        "title": "NOAA 3520 Delta黑子",
        "type": "sunspot",
        "source": "SDO/HMI",
        "wavelength": "171A",
        "description": "具有delta磁结构的复杂黑子群，高风险等级",
        "timestamp": "2026-05-20T09:30:00Z",
        "image_url": "/cases/case-004.jpg",
        "ground_truth": {
            "sunspots": [{"id": "NOAA 3520", "magnetic_type": "Delta", "complexity": "extreme"}]
        }
    },
    {
        "id": "case-005",
        "title": "CME事件观测",
        "type": "cme",
        "source": "SOHO/LASCO",
        "wavelength": "white light",
        "description": "日冕物质抛射事件，对地影响评估",
        "timestamp": "2026-05-25T16:00:00Z",
        "image_url": "/cases/case-005.jpg",
        "ground_truth": {
            "cme": {"type": "full halo", "speed": "1200 km/s"}
        }
    }
]

@router.get("/cases", tags=["案例库"])
async def get_cases(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    type: Optional[str] = None
):
    filtered = mock_cases
    
    if type:
        filtered = [c for c in filtered if c["type"] == type]
    
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
            "items": items
        }
    }

@router.get("/cases/{case_id}", tags=["案例库"])
async def get_case(case_id: str):
    case = next((c for c in mock_cases if c["id"] == case_id), None)
    
    if not case:
        raise HTTPException(
            status_code=404,
            detail={"code": "CASE_NOT_FOUND", "message": "案例不存在"}
        )
    
    return {
        "success": True,
        "data": case
    }