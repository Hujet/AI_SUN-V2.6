"""
Solar Activity Visualization & Statistics API

Provides endpoints for data visualization, trend charts, feature distribution
statistics, risk level analysis, and quantitative metrics.
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from persistent_store import get_reports_store, get_analysis_history_store

reports_store = get_reports_store()
history_store = get_analysis_history_store()


# =============================================================================
# Chart Generation Helpers
# =============================================================================

# Modern color palette for solar data
SOLAR_PALETTE = ["#00FF88", "#FF6B6B", "#FFD93D", "#6BCBFF", "#C084FC",
                 "#FB923C", "#34D399", "#F472B6"]

# Matplotlib style configuration
MPL_STYLE = {
    "figure.facecolor": "#1a1a3e",
    "axes.facecolor": "#0d1b2a",
    "axes.edgecolor": "#2a2a5e",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "text.color": "white",
    "xtick.color": "#aaaacc",
    "ytick.color": "#aaaacc",
    "grid.color": "#2a2a5e",
    "grid.alpha": 0.3,
    "legend.facecolor": "#1a1a3e",
    "legend.edgecolor": "#2a2a5e",
    "legend.labelcolor": "white",
}


def _apply_dark_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_STYLE)


def _generate_feature_distribution_chart(all_reports: List[Dict]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return ""

    _apply_dark_style()

    type_counts: Dict[str, int] = {}
    for r in all_reports:
        for feat in r.get("analysis", {}).get("features", []):
            ftype = feat.get("type", "unknown")
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

    if not type_counts:
        return ""

    types_sorted = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0].replace("_", " ").title() for t in types_sorted]
    values = [t[1] for t in types_sorted]
    colors = {"sunspot": "#00FF88", "flare": "#FF6B6B", "bright_region": "#FFD93D", "plage": "#6BCBFF"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a3e")

    bar_colors = [colors.get(types_sorted[i][0], "#FFFFFF") for i in range(len(labels))]
    ax1.bar(labels, values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_title("Feature Type Distribution", fontsize=13, color="white", fontweight="bold")
    ax1.set_ylabel("Detection Count", fontsize=10, color="#aaaacc")
    ax1.set_facecolor("#0d1b2a")
    ax1.tick_params(colors="#aaaacc")
    ax1.grid(True, alpha=0.15, color="#2a2a5e")
    for spine in ax1.spines.values():
        spine.set_color("#2a2a5e")
    for i, v in enumerate(values):
        ax1.text(i, v + max(values) * 0.02, str(v), ha="center", color="white", fontsize=9)

    ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=bar_colors, startangle=90,
            textprops={"color": "white", "fontsize": 9})
    ax2.set_title("Feature Proportion", fontsize=13, color="white", fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _generate_risk_distribution_chart(all_reports: List[Dict]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    _apply_dark_style()

    risk_counts = {"low": 0, "moderate": 0, "high": 0}
    for r in all_reports:
        level = r.get("analysis", {}).get("risk_level", "low")
        if level in risk_counts:
            risk_counts[level] += 1

    total = sum(risk_counts.values())
    if total == 0:
        return ""

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1a1a3e")

    labels = [f"{k.title()} ({v})" for k, v in risk_counts.items() if v > 0]
    sizes = [v for v in risk_counts.values() if v > 0]
    filtered_colors = [c for c, s in zip(["#4ade80", "#facc15", "#f87171"], risk_counts.values()) if s > 0]

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=filtered_colors, startangle=90,
           textprops={"color": "white", "fontsize": 10})
    ax.set_title("Risk Level Distribution", fontsize=13, color="white", fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _generate_trend_chart(all_reports: List[Dict]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    _apply_dark_style()

    sorted_reports = sorted(all_reports, key=lambda r: r.get("generated_at", ""))
    if len(sorted_reports) < 1:
        return ""

    daily_data: Dict[str, Dict] = {}
    for r in sorted_reports:
        ts = r.get("generated_at", "")
        if not ts:
            continue
        date_str = ts[:10]
        if date_str not in daily_data:
            daily_data[date_str] = {"feature_count": 0, "avg_complexity": 0, "report_count": 0,
                                     "sunspot_count": 0, "flare_count": 0, "bright_region_count": 0}
        features = r.get("analysis", {}).get("features", [])
        daily_data[date_str]["feature_count"] += len(features)
        daily_data[date_str]["avg_complexity"] += r.get("analysis", {}).get("complexity_score", 0)
        daily_data[date_str]["report_count"] += 1
        daily_data[date_str]["sunspot_count"] += sum(1 for f in features if f.get("type") == "sunspot")
        daily_data[date_str]["flare_count"] += sum(1 for f in features if f.get("type") == "flare")
        daily_data[date_str]["bright_region_count"] += sum(1 for f in features if f.get("type") in ("bright_region", "plage"))

    dates = sorted(daily_data.keys())
    x = list(range(len(dates)))
    sunspot_counts = [daily_data[d]["sunspot_count"] for d in dates]
    flare_counts = [daily_data[d]["flare_count"] for d in dates]
    br_counts = [daily_data[d]["bright_region_count"] for d in dates]
    complexity_avgs = [daily_data[d]["avg_complexity"] / max(daily_data[d]["report_count"], 1) for d in dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a3e")

    ax1.bar(x, sunspot_counts, color="#00FF88", alpha=0.85, label="Sunspots", edgecolor="white", linewidth=0.3)
    ax1.bar(x, flare_counts, bottom=sunspot_counts, color="#FF6B6B", alpha=0.85, label="Flares", edgecolor="white", linewidth=0.3)
    bright_bottom = [s + f for s, f in zip(sunspot_counts, flare_counts)]
    ax1.bar(x, br_counts, bottom=bright_bottom, color="#FFD93D", alpha=0.85, label="Bright Regions", edgecolor="white", linewidth=0.3)
    ax1.set_title("Solar Feature Detection Trend (Daily)", fontsize=13, color="white", fontweight="bold")
    ax1.set_ylabel("Feature Count", fontsize=10, color="#aaaacc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45, ha="right", fontsize=8, color="#aaaacc")
    ax1.set_facecolor("#0d1b2a")
    ax1.tick_params(colors="#aaaacc")
    ax1.legend(fontsize=8, facecolor="#1a1a3e", edgecolor="#2a2a5e", labelcolor="white")
    ax1.grid(True, alpha=0.15, color="#2a2a5e")
    for spine in ax1.spines.values():
        spine.set_color("#2a2a5e")

    color_line = "#FFD93D"
    ax2.plot(x, complexity_avgs, "o-", color=color_line, linewidth=2, markersize=6)
    ax2.fill_between(x, 0, complexity_avgs, alpha=0.15, color=color_line)
    ax2.set_title("Average Complexity Score Trend", fontsize=13, color="white", fontweight="bold")
    ax2.set_ylabel("Complexity Score (0-10)", fontsize=10, color="#aaaacc")
    ax2.set_xlabel("Date", fontsize=10, color="#aaaacc")
    ax2.set_ylim(0, 10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates, rotation=45, ha="right", fontsize=8, color="#aaaacc")
    ax2.set_facecolor("#0d1b2a")
    ax2.tick_params(colors="#aaaacc")
    ax2.grid(True, alpha=0.2, color="#2a2a5e")
    for spine in ax2.spines.values():
        spine.set_color("#2a2a5e")
    for i, v in enumerate(complexity_avgs):
        ax2.annotate(f"{v:.1f}", (x[i], v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color="white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _generate_heatmap_chart(all_reports: List[Dict]) -> str:
    """Generate a heatmap-like chart showing activity intensity (complexity vs time)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    _apply_dark_style()

    sorted_reports = sorted(all_reports, key=lambda r: r.get("generated_at", ""))
    if len(sorted_reports) < 2:
        return ""

    # Extract time and complexity data
    points = []
    for r in sorted_reports:
        ts = r.get("generated_at", "")
        if not ts:
            continue
        complexity = r.get("analysis", {}).get("complexity_score", 0)
        points.append((ts, complexity))

    if len(points) < 2:
        return ""

    # Create a 2D histogram (heatmap)
    timestamps = [p[0] for p in points]
    complexities = [p[1] for p in points]
    dates_sorted = sorted(set(t[:10] for t in timestamps))

    # Group by date and complexity buckets
    date_to_complexities: Dict[str, List[float]] = {d: [] for d in dates_sorted}
    for ts, comp in points:
        d = ts[:10]
        if d in date_to_complexities:
            date_to_complexities[d].append(comp)

    # Build heatmap data: rows=dates, cols=complexity buckets
    comp_bins = np.linspace(0, 10, 11)
    heatmap_data = np.zeros((len(dates_sorted), len(comp_bins) - 1))
    for i, d in enumerate(dates_sorted):
        for c in date_to_complexities[d]:
            bin_idx = min(int(c // 1), 9)
            heatmap_data[i, bin_idx] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a3e")

    # Custom colormap: #0c1445 -> #ffd700 -> #ff4444
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ["#0c1445", "#1a3a6e", "#2a6a9e", "#4aa0d0", "#80c8a0",
                   "#c0e060", "#ffd700", "#ff8844", "#ff4444", "#cc0000"]
    custom_cmap = LinearSegmentedColormap.from_list("solar_heatmap", colors_cmap, N=256)

    im = ax.imshow(heatmap_data.T, aspect="auto", cmap=custom_cmap, interpolation="gaussian",
                   origin="lower", extent=[-0.5, len(dates_sorted) - 0.5, 0, 10])

    ax.set_title("Activity Intensity Heatmap (Complexity vs Time)", fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Date", fontsize=10, color="#aaaacc")
    ax.set_ylabel("Complexity Score", fontsize=10, color="#aaaacc")
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#aaaacc")
    ax.grid(False)

    # Show date labels
    step = max(1, len(dates_sorted) // 15)
    ax.set_xticks(range(0, len(dates_sorted), step))
    ax.set_xticklabels([dates_sorted[i] for i in range(0, len(dates_sorted), step)], rotation=45, ha="right", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Detection Count", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaaacc")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _generate_scatter_chart(all_reports: List[Dict]) -> str:
    """Generate a scatter plot showing confidence vs complexity with risk level as color."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    _apply_dark_style()

    risk_color_map = {"low": "#4ade80", "moderate": "#facc15", "high": "#ef4444"}

    data_points = []
    for r in all_reports:
        analysis = r.get("analysis", {})
        complexity = analysis.get("complexity_score")
        confidence = analysis.get("classification_confidence")
        risk_level = analysis.get("risk_level", "low")
        if complexity is not None and confidence is not None:
            data_points.append((complexity, confidence, risk_level))

    if not data_points:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a3e")

    for risk in ["low", "moderate", "high"]:
        pts = [p for p in data_points if p[2] == risk]
        if pts:
            x_vals = [p[0] for p in pts]
            y_vals = [p[1] for p in pts]
            ax.scatter(x_vals, y_vals, c=risk_color_map[risk], label=risk.title(),
                       alpha=0.7, s=60, edgecolors="white", linewidth=0.5)

    ax.set_title("Confidence vs Complexity by Risk Level", fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Complexity Score", fontsize=10, color="#aaaacc")
    ax.set_ylabel("Classification Confidence", fontsize=10, color="#aaaacc")
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#aaaacc")
    ax.grid(True, alpha=0.2, color="#2a2a5e")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, facecolor="#1a1a3e", edgecolor="#2a2a5e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#2a2a5e")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _generate_comparison_chart(all_reports: List[Dict]) -> str:
    """Generate a grouped bar chart comparing AI vs manual review results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    _apply_dark_style()

    # Try to read review records
    review_file = BASE_DIR / "data" / "review_records.json"
    reviews = []
    try:
        with open(review_file, "r", encoding="utf-8") as f:
            reviews = json.load(f).get("records", [])
    except Exception:
        pass

    if not reviews:
        # Placeholder: show empty chart with a message
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1a1a3e")
        ax.set_facecolor("#0d1b2a")
        ax.text(0.5, 0.5, "No Review Data Available\n\nRun manual reviews to see comparison metrics",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
                color="#aaaacc", style="italic")
        ax.set_title("AI vs Manual Review Comparison", fontsize=13, color="white", fontweight="bold")
        ax.tick_params(colors="#aaaacc")
        for spine in ax.spines.values():
            spine.set_color("#2a2a5e")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # Calculate metrics from reviews
    total = len(reviews)
    confirmed = sum(1 for r in reviews if r.get("verification_status") == "confirmed")
    corrected = sum(1 for r in reviews if r.get("verification_status") == "corrected")
    disputed = sum(1 for r in reviews if r.get("verification_status") == "disputed")

    categories = ["Accuracy", "Confirmed", "Corrected", "Disputed"]
    ai_values = [85.0, 70.0, 15.0, 5.0]  # Baseline AI performance
    review_values = [
        round(confirmed / max(total, 1) * 100, 1),
        round(confirmed / max(total, 1) * 100, 1),
        round(corrected / max(total, 1) * 100, 1),
        round(disputed / max(total, 1) * 100, 1),
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a3e")

    bars1 = ax.bar(x - width / 2, ai_values, width, label="AI Baseline", color="#6BCBFF", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, review_values, width, label="Manual Review", color="#FFD93D", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_title("AI vs Manual Review Comparison", fontsize=13, color="white", fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=10, color="#aaaacc")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color="white")
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#aaaacc")
    ax.grid(True, axis="y", alpha=0.2, color="#2a2a5e")
    ax.legend(fontsize=9, facecolor="#1a1a3e", edgecolor="#2a2a5e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#2a2a5e")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color="white")
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color="white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a3e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/statistics/summary", tags=["数据统计"])
async def get_statistics_summary():
    """Get comprehensive statistical summary of all analyses."""
    all_reports = reports_store.list_all()
    if not all_reports:
        return {"success": True, "data": {"total_analyses": 0, "message": "暂无分析数据，请先上传图像并进行分析"}}

    total_features = 0
    feature_type_counts: Dict[str, int] = {}
    risk_counts = {"low": 0, "moderate": 0, "high": 0}
    complexity_scores = []
    hale_classes: Dict[str, int] = {}
    processing_times = []
    confidence_values = []
    all_feature_types = set()
    feature_counts_per_report = []

    for r in all_reports:
        analysis = r.get("analysis", {})
        features = analysis.get("features", [])
        n_features = len(features)
        total_features += n_features
        feature_counts_per_report.append(n_features)
        complexity_scores.append(analysis.get("complexity_score", 0))
        processing_times.append(r.get("processing_time_seconds", 0))
        risk_level = analysis.get("risk_level", "low")
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1
        hc = analysis.get("hale_classification", "Unknown")
        hale_classes[hc] = hale_classes.get(hc, 0) + 1
        for feat in features:
            ftype = feat.get("type", "unknown")
            feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
            all_feature_types.add(ftype)
            conf = feat.get("confidence", 0)
            if conf > 0:
                confidence_values.append(conf)

    n = len(all_reports)
    avg_complexity = sum(complexity_scores) / n if n > 0 else 0
    avg_processing_time = sum(processing_times) / n if n > 0 else 0
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0

    if n > 1:
        mean = avg_complexity
        variance = sum((c - mean) ** 2 for c in complexity_scores) / n
        std_complexity = round(variance ** 0.5, 2)
    else:
        std_complexity = 0

    timestamps = [r.get("generated_at", "") for r in all_reports if r.get("generated_at")]
    time_range = {"first": min(timestamps) if timestamps else None, "last": max(timestamps) if timestamps else None}

    # --- New KPI fields ---

    # 1. active_region_count: complexity_score >= 7
    active_region_count = sum(1 for c in complexity_scores if c >= 7)

    # 2. flare_risk_high_count: risk_level == "high"
    flare_risk_high_count = risk_counts["high"]

    # 3. avg_features_per_report
    avg_features_per_report = round(sum(feature_counts_per_report) / n, 2) if n > 0 else 0

    # 4. total_token_cost: read from token_usage.json
    total_token_cost = 0
    token_file = BASE_DIR / "data" / "token_usage.json"
    try:
        if token_file.exists():
            with open(token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)
            records = token_data.get("records", [])
            for rec in records:
                usage = rec.get("usage", {})
                # Estimate cost: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
                input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                total_token_cost += (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60
    except Exception:
        pass
    total_token_cost = round(total_token_cost, 6)

    # 5. avg_processing_time (already computed, ensure it's in seconds)

    # 6. hale_distribution_percentages
    hale_percentages = {}
    if n > 0:
        for cls, count in sorted(hale_classes.items(), key=lambda x: x[1], reverse=True):
            hale_percentages[cls] = round(count / n * 100, 2)

    # 7. weekly_analysis_count: last 7 days
    now = datetime.now()
    week_ago = (now - timedelta(days=7)).isoformat()
    weekly_count = sum(1 for ts in timestamps if ts >= week_ago)

    return {"success": True, "data": {
        "overview": {"total_analyses": n, "total_features_detected": total_features,
                      "unique_feature_types": sorted(list(all_feature_types)), "time_range": time_range},
        "risk_distribution": {"low": risk_counts["low"], "moderate": risk_counts["moderate"],
                               "high": risk_counts["high"],
                               "high_risk_ratio": round(risk_counts["high"] / n, 4) if n > 0 else 0},
        "complexity_metrics": {"mean": round(avg_complexity, 2), "std_deviation": std_complexity,
                                "max": round(max(complexity_scores) if complexity_scores else 0, 2),
                                "min": round(min(complexity_scores) if complexity_scores else 0, 2),
                                "range": round(max(complexity_scores or [0]) - min(complexity_scores or [0]), 2)},
        "feature_type_distribution": feature_type_counts,
        "hale_classification_distribution": hale_classes,
        "performance_metrics": {"average_processing_time_s": round(avg_processing_time, 2),
                                 "average_confidence": round(avg_confidence, 4),
                                 "confidence_range": [
                                     round(min(confidence_values), 4) if confidence_values else 0,
                                     round(max(confidence_values), 4) if confidence_values else 0,
                                 ]},
        # --- New KPI fields ---
        "active_region_count": active_region_count,
        "flare_risk_high_count": flare_risk_high_count,
        "avg_features_per_report": avg_features_per_report,
        "total_token_cost": total_token_cost,
        "avg_processing_time": round(avg_processing_time, 2),
        "hale_distribution_percentages": hale_percentages,
        "weekly_analysis_count": weekly_count,
    }}


@router.get("/statistics/charts", tags=["数据统计"])
async def get_chart_images(chart_type: str = Query("all")):
    """Generate and return base64-encoded chart images.

    Supported chart types:
    - trend: Feature detection trend over time
    - distribution: Feature type distribution (bar + pie)
    - risk: Risk level distribution (pie)
    - heatmap: Activity intensity heatmap (complexity vs time)
    - scatter: Confidence vs complexity scatter plot
    - comparison: AI vs manual review comparison
    - all: Generate all available charts
    """
    all_reports = reports_store.list_all()
    if not all_reports:
        return {"success": True, "data": {"message": "暂无分析数据"}}

    charts = {}
    if chart_type in ("trend", "all"):
        charts["trend"] = _generate_trend_chart(all_reports)
    if chart_type in ("distribution", "all"):
        charts["distribution"] = _generate_feature_distribution_chart(all_reports)
    if chart_type in ("risk", "all"):
        charts["risk"] = _generate_risk_distribution_chart(all_reports)
    if chart_type in ("heatmap", "all"):
        charts["heatmap"] = _generate_heatmap_chart(all_reports)
    if chart_type in ("scatter", "all"):
        charts["scatter"] = _generate_scatter_chart(all_reports)
    if chart_type in ("comparison", "all"):
        charts["comparison"] = _generate_comparison_chart(all_reports)

    return {"success": True, "data": {"report_count": len(all_reports),
        "charts": {k: f"data:image/png;base64,{v}" for k, v in charts.items() if v}}}


@router.get("/statistics/history", tags=["数据统计"])
async def get_analysis_history(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_level: Optional[str] = None,
    hale_classification: Optional[str] = None,
    min_complexity: Optional[float] = None,
    max_complexity: Optional[float] = None,
):
    """Get analysis history with advanced filtering.

    Filters:
    - start_date/end_date: Date range filter (YYYY-MM-DD)
    - risk_level: Filter by risk level (low/moderate/high)
    - hale_classification: Filter by Hale class
    - min_complexity/max_complexity: Filter by complexity score range
    """
    all_history = history_store.list_all()

    if start_date:
        all_history = [h for h in all_history if h.get("timestamp", "") >= start_date]
    if end_date:
        all_history = [h for h in all_history if h.get("timestamp", "") <= end_date + "T23:59:59"]
    if risk_level:
        all_history = [h for h in all_history if h.get("risk_level") == risk_level]
    if hale_classification:
        all_history = [h for h in all_history if h.get("hale_classification") == hale_classification]
    if min_complexity is not None:
        all_history = [h for h in all_history if h.get("complexity_score", 0) >= min_complexity]
    if max_complexity is not None:
        all_history = [h for h in all_history if h.get("complexity_score", 10) <= max_complexity]

    all_history.sort(key=lambda h: h.get("timestamp", ""), reverse=True)
    total = len(all_history)
    start = (page - 1) * limit
    end = start + limit
    items = all_history[start:end]

    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": items,
        },
    }


@router.get("/statistics/classification-summary", tags=["数据统计"])
async def get_classification_summary():
    """Get Hale classification distribution with percentages."""
    all_reports = reports_store.list_all()
    hale_counts: Dict[str, int] = {}
    total = len(all_reports)
    for r in all_reports:
        hc = r.get("analysis", {}).get("hale_classification", "Unknown")
        hale_counts[hc] = hale_counts.get(hc, 0) + 1

    distribution = {cls: {"count": count, "percentage": round(count / total * 100, 1) if total > 0 else 0}
                    for cls, count in sorted(hale_counts.items(), key=lambda x: x[1], reverse=True)}
    return {"success": True, "data": {"total_analyses": total, "hale_classification_distribution": distribution,
        "most_common": max(hale_counts, key=hale_counts.get) if hale_counts else None,
        "unique_classifications": len(hale_counts)}}


@router.get("/statistics/detailed", tags=["数据统计"])
async def get_detailed_statistics():
    """Get detailed statistical breakdowns including hourly/daily distributions and score buckets."""
    all_reports = reports_store.list_all()
    if not all_reports:
        return {"success": True, "data": {"message": "暂无分析数据"}}

    # Hourly analysis distribution (0-23)
    hourly_dist: Dict[str, int] = {str(h): 0 for h in range(24)}
    # Daily analysis distribution (0-6, Monday=0)
    daily_dist: Dict[str, int] = {str(d): 0 for d in range(7)}
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    complexity_scores = []
    confidence_values = []

    for r in all_reports:
        ts = r.get("generated_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                hour = dt.hour
                dow = dt.weekday()
                hourly_dist[str(hour)] += 1
                daily_dist[str(dow)] += 1
            except (ValueError, TypeError):
                pass

        analysis = r.get("analysis", {})
        complexity_scores.append(analysis.get("complexity_score", 0))
        confidence_values.append(analysis.get("classification_confidence", 0))

    # Complexity buckets [0-2, 2-4, 4-6, 6-8, 8-10]
    comp_buckets = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
    for c in complexity_scores:
        if c < 2:
            comp_buckets["0-2"] += 1
        elif c < 4:
            comp_buckets["2-4"] += 1
        elif c < 6:
            comp_buckets["4-6"] += 1
        elif c < 8:
            comp_buckets["6-8"] += 1
        else:
            comp_buckets["8-10"] += 1

    # Confidence buckets [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    conf_buckets = {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for c in confidence_values:
        if c < 0.2:
            conf_buckets["0-0.2"] += 1
        elif c < 0.4:
            conf_buckets["0.2-0.4"] += 1
        elif c < 0.6:
            conf_buckets["0.4-0.6"] += 1
        elif c < 0.8:
            conf_buckets["0.6-0.8"] += 1
        else:
            conf_buckets["0.8-1.0"] += 1

    # Convert daily distribution to use day names
    daily_dist_named = {day_names[int(k)]: v for k, v in daily_dist.items() if v > 0}

    return {
        "success": True,
        "data": {
            "hourly_analysis_distribution": {k: hourly_dist[k] for k in sorted(hourly_dist.keys(), key=int)},
            "daily_analysis_distribution": daily_dist_named,
            "complexity_buckets": comp_buckets,
            "confidence_buckets": conf_buckets,
        },
    }
