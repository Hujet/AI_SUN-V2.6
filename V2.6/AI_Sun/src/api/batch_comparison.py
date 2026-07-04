"""
Batch Comparison & Analysis Module

Compares AI recognition results with manual review results, provides
statistical analysis (accuracy, difference rate, trend analysis), and
exportable comparison reports.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/comparison", tags=["批量对比与分析"])

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Storage Helpers
# ---------------------------------------------------------------------------

def _read_reviews() -> List[Dict]:
    path = DATA_DIR / "review_records.json"
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("records", [])
    except Exception as e:
        logger.warning(f"Failed to read reviews: {e}")
    return []


def _read_reports() -> List[Dict]:
    from persistent_store import get_reports_store
    return get_reports_store().list_all()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ComparisonParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status_filter: Optional[str] = None  # confirmed/corrected/disputed
    reviewer: Optional[str] = None


class ComparisonMetric(BaseModel):
    name: str
    value: Any
    unit: str = ""
    description: str = ""


class ComparisonReport(BaseModel):
    total_compared: int = 0
    confirmed_count: int = 0
    corrected_count: int = 0
    disputed_count: int = 0
    accuracy_rate: float = 0
    correction_rate: float = 0
    dispute_rate: float = 0
    avg_complexity_ai: float = 0
    avg_complexity_human: float = 0
    common_corrections: Dict[str, int] = {}
    trend_data: List[Dict] = []
    details: List[Dict] = []


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------

def _compute_comparison(reviews: List[Dict], reports: List[Dict]) -> Dict:
    """Compute detailed comparison between AI and manual review results."""

    report_map = {r["id"]: r for r in reports if "id" in r}

    total = len(reviews)
    confirmed = sum(1 for r in reviews if r.get("verification_status") == "confirmed")
    corrected = sum(1 for r in reviews if r.get("verification_status") == "corrected")
    disputed = sum(1 for r in reviews if r.get("verification_status") == "disputed")
    pending = sum(1 for r in reviews if r.get("verification_status") == "pending")

    accuracy_rate = round(confirmed / max(total, 1) * 100, 2)
    correction_rate = round(corrected / max(total, 1) * 100, 2)
    dispute_rate = round(disputed / max(total, 1) * 100, 2)

    # Analyze corrected fields
    field_correction_counts: Dict[str, int] = {}
    ai_complexities = []
    human_complexities = []
    details = []

    for rev in reviews:
        corr_fields = rev.get("correction_fields", []) or []
        for f in corr_fields:
            field_correction_counts[f] = field_correction_counts.get(f, 0) + 1

        report_id = rev.get("report_id", "")
        rpt = report_map.get(report_id, {})
        analysis = rpt.get("analysis", {})

        ai_complexity = analysis.get("complexity_score")
        if ai_complexity is not None:
            ai_complexities.append(ai_complexity)

        # If corrected, try to extract human's corrected complexity
        corrected_result = rev.get("corrected_result") or {}
        human_complexity = corrected_result.get("complexity_score")
        if human_complexity is not None:
            human_complexities.append(human_complexity)

        details.append({
            "review_id": rev.get("id", ""),
            "report_id": report_id,
            "reviewer": rev.get("reviewer", ""),
            "status": rev.get("verification_status", ""),
            "fields_corrected": corr_fields,
            "ai_complexity": ai_complexity,
            "ai_hale": analysis.get("hale_classification", ""),
            "ai_risk": analysis.get("risk_level", ""),
            "corrected_complexity": human_complexity,
            "comments": rev.get("comments", ""),
            "timestamp": rev.get("timestamp", rev.get("created_at", "")),
        })

    # Trend data (daily)
    daily_trend: Dict[str, Dict] = {}
    for d in details:
        ts = d.get("timestamp", "")
        if ts:
            date_str = ts[:10]
            if date_str not in daily_trend:
                daily_trend[date_str] = {"confirmed": 0, "corrected": 0, "disputed": 0, "pending": 0}
            status = d.get("status", "")
            if status in daily_trend[date_str]:
                daily_trend[date_str][status] += 1

    trend_list = [
        {"date": k, "confirmed": v["confirmed"], "corrected": v["corrected"],
         "disputed": v["disputed"], "pending": v["pending"]}
        for k, v in sorted(daily_trend.items())
    ]

    # Difference analysis
    complexity_diffs = []
    for d in details:
        if d.get("ai_complexity") is not None and d.get("corrected_complexity") is not None:
            complexity_diffs.append(abs(d["ai_complexity"] - d["corrected_complexity"]))
    avg_diff = round(sum(complexity_diffs) / max(len(complexity_diffs), 1), 2)
    max_diff = round(max(complexity_diffs) if complexity_diffs else 0, 2)

    return {
        "total_compared": total,
        "confirmed_count": confirmed,
        "corrected_count": corrected,
        "disputed_count": disputed,
        "pending_count": pending,
        "accuracy_rate": accuracy_rate,
        "correction_rate": correction_rate,
        "dispute_rate": dispute_rate,
        "pending_rate": round(pending / max(total, 1) * 100, 2),
        "avg_complexity_ai": round(sum(ai_complexities) / max(len(ai_complexities), 1), 2) if ai_complexities else 0,
        "avg_complexity_human": round(sum(human_complexities) / max(len(human_complexities), 1), 2) if human_complexities else 0,
        "avg_complexity_diff": avg_diff,
        "max_complexity_diff": max_diff,
        "common_corrections": dict(sorted(field_correction_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "trend_data": trend_list,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary", summary="获取批量对比摘要")
async def get_comparison_summary(
    start_date: Optional[str] = Query(None, description="起始日期 YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="结束日期 YYYY-MM-DD"),
    status_filter: Optional[str] = Query(None, description="状态筛选"),
    reviewer: Optional[str] = Query(None, description="复核人筛选"),
):
    """Get summary statistics of AI vs manual review comparison."""
    reviews = _read_reviews()
    reports = _read_reports()

    # Apply filters
    if start_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] >= start_date]
    if end_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] <= end_date]
    if status_filter:
        reviews = [r for r in reviews if r.get("verification_status") == status_filter]
    if reviewer:
        reviews = [r for r in reviews if r.get("reviewer") == reviewer]

    result = _compute_comparison(reviews, reports)

    return {"success": True, "data": result}


@router.get("/details", summary="获取对比明细")
async def get_comparison_details(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    reviewer: Optional[str] = Query(None),
):
    """Get detailed comparison records with pagination."""
    reviews = _read_reviews()
    reports = _read_reports()

    if start_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] >= start_date]
    if end_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] <= end_date]
    if status_filter:
        reviews = [r for r in reviews if r.get("verification_status") == status_filter]
    if reviewer:
        reviews = [r for r in reviews if r.get("reviewer") == reviewer]

    reviews.sort(key=lambda r: r.get("timestamp") or r.get("created_at", ""), reverse=True)

    total = len(reviews)
    start = (page - 1) * limit
    end = start + limit
    page_reviews = reviews[start:end]

    result = _compute_comparison(page_reviews, reports)

    return {"success": True, "data": {
        "total": total, "page": page, "limit": limit,
        "summary": {k: v for k, v in result.items() if k not in ("details",)},
        "items": result["details"],
    }}


@router.get("/trends", summary="获取对比趋势数据")
async def get_comparison_trends(
    days: int = Query(30, ge=1, le=365),
):
    """Get trend data for the past N days."""
    reviews = _read_reviews()
    reports = _read_reports()

    now = datetime.now()
    cutoff = (now - timedelta(days=days)).isoformat()
    reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "") >= cutoff]

    result = _compute_comparison(reviews, reports)

    return {"success": True, "data": {
        "period_days": days,
        "trend": result["trend_data"],
        "summary": {k: v for k, v in result.items() if k not in ("trend_data", "details")},
    }}


@router.get("/export/csv", summary="导出对比报告CSV")
async def export_comparison_csv(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
):
    """Export comparison report as CSV."""
    import io
    import csv
    from fastapi.responses import Response

    reviews = _read_reviews()
    reports = _read_reports()

    if start_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] >= start_date]
    if end_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] <= end_date]
    if status_filter:
        reviews = [r for r in reviews if r.get("verification_status") == status_filter]

    result = _compute_comparison(reviews, reports)
    details = result["details"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Review ID", "Report ID", "Reviewer", "Status",
        "AI Complexity", "AI Hale Class", "AI Risk Level",
        "Corrected Complexity", "Fields Corrected", "Comments", "Timestamp"
    ])

    for d in details:
        writer.writerow([
            d.get("review_id", ""),
            d.get("report_id", ""),
            d.get("reviewer", ""),
            d.get("status", ""),
            d.get("ai_complexity", ""),
            d.get("ai_hale", ""),
            d.get("ai_risk", ""),
            d.get("corrected_complexity", ""),
            "; ".join(d.get("fields_corrected", [])),
            d.get("comments", ""),
            d.get("timestamp", ""),
        ])

    csv_content = output.getvalue()

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=comparison_report.csv",
        },
    )


@router.get("/export/json", summary="导出对比报告JSON")
async def export_comparison_json(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
):
    """Export comparison report as JSON."""
    from fastapi.responses import JSONResponse

    reviews = _read_reviews()
    reports = _read_reports()

    if start_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] >= start_date]
    if end_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] <= end_date]
    if status_filter:
        reviews = [r for r in reviews if r.get("verification_status") == status_filter]

    result = _compute_comparison(reviews, reports)

    return JSONResponse(
        content={"success": True, "data": result},
        headers={"Content-Disposition": "attachment; filename=comparison_report.json"},
    )


@router.get("/report", summary="生成对比分析报告")
async def generate_comparison_report(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Generate a comprehensive comparison report with chart data."""
    reviews = _read_reviews()
    reports = _read_reports()

    if start_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] >= start_date]
    if end_date:
        reviews = [r for r in reviews if (r.get("timestamp") or r.get("created_at", "") or "")[:10] <= end_date]

    comparison = _compute_comparison(reviews, reports)

    # Generate chart data for frontend
    chart_data = {
        "status_distribution": {
            "labels": ["Confirmed", "Corrected", "Disputed", "Pending"],
            "values": [
                comparison["confirmed_count"],
                comparison["corrected_count"],
                comparison["disputed_count"],
                comparison.get("pending_count", 0),
            ],
        },
        "accuracy_metrics": {
            "labels": ["Accuracy Rate", "Correction Rate", "Dispute Rate"],
            "values": [
                comparison["accuracy_rate"],
                comparison["correction_rate"],
                comparison["dispute_rate"],
            ],
        },
        "common_corrections": {
            "labels": list(comparison["common_corrections"].keys()),
            "values": list(comparison["common_corrections"].values()),
        },
        "trend": comparison["trend_data"],
    }

    return {"success": True, "data": {
        "summary": {k: v for k, v in comparison.items() if k not in ("trend_data", "details",)},
        "charts": chart_data,
        "details_count": len(comparison["details"]),
    }}
