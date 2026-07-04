"""
Solar Image Analysis API Router

Handles analysis task creation, progress tracking, image annotation,
report generation, token usage tracking, traceability, and review system.
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
ANNOTATED_DIR = DATA_DIR / "annotated"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

from persistent_store import (
    get_images_store, get_reports_store, get_tasks_store, get_analysis_history_store,
)
from solar_classifier import SolarClassifier, SolarRegionAnalysis
from deepseek_client import DeepseekAPIClient, DeepSeekConfig
from token_usage_tracker import TokenUsageTracker
from traceability import TraceabilityStore, TraceabilityRecord, compute_image_hash
from annotate_image import (
    DetectionReport, build_detection_report,
    generate_annotated_image as generate_annotated_image_v2,
    generate_detection_report_image,
    generate_combined_report_image,
)

# ---------------------------------------------------------------------------
# Persistent Stores
# ---------------------------------------------------------------------------

images_store = get_images_store()
reports_store = get_reports_store()
tasks_store = get_tasks_store()
history_store = get_analysis_history_store()
token_tracker = TokenUsageTracker(DATA_DIR / "token_usage.json")
traceability_store = TraceabilityStore(DATA_DIR / "traceability.json")

# Review store (manual verification records)
review_store_path = DATA_DIR / "review_records.json"


def _read_reviews() -> List[Dict]:
    try:
        with open(review_store_path, "r", encoding="utf-8") as f:
            return json.load(f).get("records", [])
    except Exception:
        return []


def _write_reviews(records: List[Dict]) -> None:
    tmp = review_store_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, review_store_path)


def _save_review(record: Dict) -> None:
    records = _read_reviews()
    records.append(record)
    _write_reviews(records)


# ---------------------------------------------------------------------------
# Initialize Classifier
# ---------------------------------------------------------------------------

deepseek_client = None
classifier = None

try:
    config = DeepSeekConfig.from_env()
    deepseek_client = DeepseekAPIClient(config=config)
    classifier = SolarClassifier(deepseek_client=deepseek_client)
    logger.info("DeepSeek client initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize DeepSeek client: {e}")
    try:
        classifier = SolarClassifier()
    except Exception as e2:
        logger.error(f"Failed to create fallback classifier: {e2}")
        classifier = None


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    image_id: str
    analysis_type: Optional[str] = "full"
    options: Optional[Dict[str, bool]] = None
    model_key: Optional[str] = "glm"  # Selected AI model
    prompt_config: Optional[Dict] = None  # Prompt template configuration
    confidence_threshold: Optional[float] = None  # 分析阶段置信度过滤阈值 (None=auto: 0.3 general, 0.15 prominence/flare)


class BatchAnalyzeRequest(BaseModel):
    image_ids: List[str]
    analysis_type: Optional[str] = "full"
    options: Optional[Dict[str, bool]] = None
    model_key: Optional[str] = "glm"  # Selected AI model
    prompt_config: Optional[Dict] = None  # Prompt template configuration


class ReviewRecord(BaseModel):
    report_id: str
    task_id: str
    reviewer: str = "operator"
    original_result: Optional[Dict] = None
    corrected_result: Optional[Dict] = None
    verification_status: str = "pending"  # pending | confirmed | corrected | disputed
    comments: str = ""
    correction_fields: Optional[List[str]] = None
    # Enhanced review: parameter modification
    modified_params: Optional[Dict[str, Any]] = None  # key=value pairs changed by reviewer
    param_change_reason: str = ""


class ReviewParamChange(BaseModel):
    """Record a parameter change made during manual review."""
    report_id: str
    task_id: str
    reviewer: str = "operator"
    param_name: str
    old_value: Any = None
    new_value: Any = None
    reason: str = ""


class AdminConfig(BaseModel):
    max_tokens_per_request: Optional[int] = None
    confidence_threshold: Optional[float] = None
    enable_heuristic_fallback: Optional[bool] = None
    # Enhanced config options
    recognition_threshold: Optional[float] = None  # min confidence for feature detection
    hale_classification_weights: Optional[Dict[str, float]] = None  # weights for classification
    feature_type_weights: Optional[Dict[str, float]] = None  # weights per feature type
    complexity_score_formula: Optional[str] = None  # formula override


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_task_id() -> str:
    return f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def generate_report_id() -> str:
    return f"rpt-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def generate_annotated_image(image_path: str, analysis: SolarRegionAnalysis, features: List[Dict], output_id: str) -> str:
    """Generate annotated solar image with bounding boxes and solar disk overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
        import numpy as np
    except ImportError as e:
        logger.warning(f"Matplotlib/Pillow not available: {e}")
        return ""

    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img_array)

        # Draw solar disk boundary if available
        disk_info = analysis.intermediate_steps.get("solar_disk")
        if disk_info:
            disk_cx = disk_info["center_x"]
            disk_cy = disk_info["center_y"]
            disk_r = disk_info["radius"]
            disk_circle = mpatches.Circle((disk_cx, disk_cy), radius=disk_r, fill=False,
                                         edgecolor="#FF8C00", linewidth=2, linestyle="-", alpha=0.8)
            ax.add_patch(disk_circle)

        # Color map for feature types
        color_map = {
            "sunspot": "#00FF00", "flare": "#FF4444", "bright_region": "#FFD700",
            "plage": "#00BFFF", "filament": "#9370DB", "coronal_hole": "#20B2AA",
            "prominence": "#FFD700", "facula": "#A855F7",
        }

        for idx, feat in enumerate(features):
            ftype = feat.get("type", "unknown")
            color = color_map.get(ftype, "#FFFFFF")
            pos = feat.get("position", {})
            size_rel = feat.get("size_relative", 0.05)
            px = pos.get("x", 0.5)
            py = pos.get("y", 0.5)

            # All coordinates are normalized (0-1)
            x = int(px * w)
            y = int(py * h)

            # Bounding box size: size_relative = feature_diameter / disk_diameter
            # Use solar disk diameter if available, otherwise estimate from image
            if disk_info:
                disk_diameter_px = disk_info["radius"] * 2
            else:
                disk_diameter_px = min(w, h) * 0.7  # fallback estimate
            
            box_size = int(disk_diameter_px * size_rel)
            box_size = max(6, min(box_size, 200))  # clamp to 6-200px

            # Draw bounding box (rectangle)
            rect = mpatches.Rectangle(
                (x - box_size // 2, y - box_size // 2),
                box_size, box_size,
                fill=False, edgecolor=color, linewidth=2, alpha=0.9
            )
            ax.add_patch(rect)

            # Draw crosshair at center
            ax.plot(x, y, "+", color=color, markersize=10, markeredgewidth=2)

            # Numbered label
            label_text = f"#{idx+1} {feat.get('label', ftype)}"
            conf = feat.get("confidence", 0)

            ax.annotate(label_text, xy=(x + box_size // 2 + 3, y - box_size // 2),
                        fontsize=9, color=color, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor=color, alpha=0.85))

        # Legend
        legend_patches = []
        legend_patches.extend([mpatches.Patch(color=c, label=f"{t.replace('_', ' ').title()}")
                          for t, c in color_map.items() if any(f.get("type") == t for f in features)])

        if disk_info:
            legend_patches.insert(0, mpatches.Patch(color="#FF8C00", label="Solar Disk"))

        if legend_patches:
            ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
                      framealpha=0.8, facecolor="black", edgecolor="white", labelcolor="white")

        ax.set_title(f"Solar Active Region Analysis\n"
                     f"Hale: {analysis.hale_classification} | "
                     f"Complexity: {analysis.complexity_score:.1f}/10 | "
                     f"Features: {len(features)}",
                     fontsize=12, color="white", pad=15)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="gray")
        plt.tight_layout()

        output_path = ANNOTATED_DIR / f"{output_id}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to generate annotated image: {e}", exc_info=True)
        return ""


def normalize_feature_coordinates(features: List[Dict], disk_info: Dict) -> List[Dict]:
    """Convert AI feature coordinates from normalized disk coords (-1~+1) to pixel coords.
    
    AI returns coords in normalized solar disk format:
    - x: -1 (east limb) to +1 (west limb), 0 = disk center
    - y: -1 (south limb) to +1 (north limb), 0 = disk center
    
    This function converts to display-ready pixel coordinates.
    """
    if not disk_info or not disk_info.get("detected"):
        # Fallback: assume coords are already 0~1 relative
        return features
    
    cx = disk_info["center_x"]
    cy = disk_info["center_y"]
    r = disk_info["radius"]
    
    for feat in features:
        pos = feat.get("position", {})
        nx = pos.get("x", 0)
        ny = pos.get("y", 0)
        
        # Detect coordinate system: if values are in -1~+1 range, convert to pixels
        if -1.5 <= nx <= 1.5 and -1.5 <= ny <= 1.5:
            # Normalized disk coords -> pixel coords
            pixel_x = cx + nx * r
            pixel_y = cy - ny * r  # Y is flipped (AI: +y=north, image: +y=down)
            
            # Store as 0~1 relative to image dimensions (need image size from disk_info)
            # Estimate image size from disk (disk fills ~80% of image)
            img_w = r * 2 / 0.8
            img_h = r * 2 / 0.8
            feat["position_relative"] = {"x": pixel_x / max(img_w, 1), "y": pixel_y / max(img_h, 1)}
            feat["position"] = {"x": round(pixel_x, 2), "y": round(pixel_y, 2)}
            feat["coord_system"] = "pixel_from_normalized_disk"
        # If coords are already > 1.5, assume they're already in pixels
    
    return features


def extract_features_for_display(analysis: SolarRegionAnalysis) -> List[Dict]:
    """Convert SolarFeature objects to display-ready dicts with proper coordinates."""
    if analysis.features:
        return [f.to_dict() for f in analysis.features]
    return []


def analyze_risk_level(complexity_score: float) -> str:
    if complexity_score >= 8:
        return "high"
    if complexity_score >= 5:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Background Task
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Confidence-based feature filtering (analysis-stage, not upload-stage)
# ---------------------------------------------------------------------------

# Prominence/flare features get a lower threshold since AI models tend to
# assign them low confidence even when correctly identified
CONFIDENCE_THRESHOLDS = {
    "prominence": 0.15,   # 日珥: lower threshold, edge features are harder
    "flare": 0.20,         # 耀斑: slightly raised threshold
    "default": 0.30,       # General features
}


def _filter_features_by_confidence(
    features: List[Dict],
    threshold: Optional[float] = None,
) -> tuple:
    """Filter features by confidence at analysis stage.

    Uses per-type thresholds: prominence (0.15), flare (0.20), default (0.30).
    A global threshold override can be supplied via the request.

    Returns (filtered_features, removed_features, filter_stats)
    """
    filtered = []
    removed = []
    stats = {"total": len(features), "kept": 0, "removed": 0, "by_type": {}}

    for feat in features:
        ftype = feat.get("type", "other")
        conf = feat.get("confidence", 0)

        # Determine threshold: global override > per-type > default
        if threshold is not None:
            min_conf = threshold
        else:
            min_conf = CONFIDENCE_THRESHOLDS.get(ftype, CONFIDENCE_THRESHOLDS["default"])

        # Track per-type stats
        if ftype not in stats["by_type"]:
            stats["by_type"][ftype] = {"total": 0, "kept": 0, "removed": 0, "threshold": min_conf}
        stats["by_type"][ftype]["total"] += 1

        if conf >= min_conf:
            filtered.append(feat)
            stats["by_type"][ftype]["kept"] += 1
            stats["kept"] += 1
        else:
            # Mark as removed but preserve original data
            feat_copy = dict(feat)
            feat_copy["removed_reason"] = f"confidence {conf:.2f} < threshold {min_conf:.2f} ({ftype})"
            removed.append(feat_copy)
            stats["by_type"][ftype]["removed"] += 1
            stats["removed"] += 1

    return filtered, removed, stats


async def process_analysis(task_id: str, image_id: str, analysis_type: str, options: Dict, model_key: str = "glm", prompt_config: Optional[Dict] = None):
    """Background analysis task with full traceability, token tracking, and multi-model support.
    
    Includes analysis-stage confidence filtering for high-quality reports.
    """
    start_time = datetime.now()
    
    # Verify image
    img_data = images_store.get(image_id)
    if not img_data:
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "failed", "progress": 0, "error": "IMAGE_NOT_FOUND", "created_at": datetime.now().isoformat()})
        return
    
    image_path = img_data.get("file_path", "")
    if not image_path or not os.path.exists(image_path):
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "failed", "progress": 0, "error": "FILE_MISSING", "created_at": datetime.now().isoformat()})
        return
    
    # Status: processing
    tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
        "status": "processing", "progress": 15, "created_at": datetime.now().isoformat()})
    
    try:
        # Step 1: AI analysis with GLM vision model
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "processing", "progress": 25, "created_at": datetime.now().isoformat()})
        
        analysis = None
        
        try:
            from ai_model_adapter import get_model_manager
            from solar_disk_locator import detect_solar_disk
            
            manager = get_model_manager()
            adapter = manager.get_model(model_key)
            
            if adapter:
                logger.info(f"Analyzing with model: {model_key}")

                # Step 1: Detect solar disk FIRST (before AI analysis)
                disk_info = detect_solar_disk(str(image_path))
                disk_validated = disk_info is not None and disk_info.confidence > 0.5
                disk_dict = disk_info.to_dict() if disk_validated else None
                
                if disk_validated:
                    logger.info(f"Solar disk detected: center=({disk_info.center_x:.0f},{disk_info.center_y:.0f}), r={disk_info.radius:.0f}, conf={disk_info.confidence:.2f}")
                else:
                    logger.warning("Solar disk detection failed or low confidence, AI will work without disk context")

                # Step 1.5: Limb enhancement for AI prominence detection
                ai_image_path = image_path
                enhanced_path = None
                if disk_validated:
                    try:
                        import cv2
                        from pathlib import Path
                        from solar_preprocessor import enhance_limb_for_ai
                        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if original_img is not None:
                            enhanced_img = enhance_limb_for_ai(original_img, disk_dict)
                            if enhanced_img is not original_img:
                                p = Path(image_path)
                                enhanced_path = str(p.parent / f"enhanced_limb_{p.stem}.jpg")
                                cv2.imwrite(enhanced_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                ai_image_path = enhanced_path
                                logger.info(f"Limb-enhanced image saved for AI: {enhanced_path}")
                    except Exception as e:
                        logger.warning(f"Limb enhancement skipped: {e}")

                # Step 2: Build prompts WITH disk context
                from prompt_templates import PromptConfig, build_system_prompt, build_user_prompt
                if prompt_config:
                    pconfig = PromptConfig.from_dict(prompt_config)
                else:
                    pconfig = PromptConfig()
                system_prompt = build_system_prompt(pconfig, disk_info=disk_dict)
                user_prompt = build_user_prompt(pconfig, disk_info=disk_dict)
                
                # Run AI analysis in a thread to allow progress updates
                import concurrent.futures
                analysis_done = {"result": None}
                
                def run_analysis():
                    analysis_done["result"] = adapter.analyze_image(
                        image_path=ai_image_path,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_analysis)
                    for pct in range(30, 60, 3):
                        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
                            "status": "processing", "progress": pct,
                            "message": f"AI视觉分析中... ({pct-25}%)",
                            "created_at": datetime.now().isoformat()})
                        await asyncio.sleep(1)
                    future.result(timeout=180)
                
                # Clean up enhanced temp file
                if enhanced_path and os.path.exists(enhanced_path):
                    try:
                        os.remove(enhanced_path)
                        logger.debug(f"Cleaned up enhanced image: {enhanced_path}")
                    except Exception:
                        pass
                
                result = analysis_done["result"]
                
                if result and result.success:
                    from solar_classifier import SolarRegionAnalysis, SolarFeature
                    from solar_dark_region_detector import detect_dark_regions, match_features_to_dark_regions
                    
                    # Step 3: Detect real dark regions using CV
                    raw_features = result.features
                    original_count = len(raw_features)
                    
                    disk_dict = disk_info.to_dict() if disk_validated else None
                    dark_regions = detect_dark_regions(
                        str(image_path),
                        disk_info=disk_dict,
                        min_size_ratio=0.01,
                        max_regions=20,
                    )
                    logger.info(f"Detected {len(dark_regions)} real dark regions via CV")
                    
                    # Step 4: Match AI features to real dark regions
                    if dark_regions:
                        try:
                            # Get image dimensions for coordinate conversion
                            img_w, img_h = 0, 0
                            try:
                                from PIL import Image as PILImage
                                with PILImage.open(str(image_path)) as _img:
                                    img_w, img_h = _img.size
                            except Exception:
                                pass
                            
                            # Keep AI features in 0~1 normalized coords
                            # match_features_to_dark_regions handles conversion internally
                            
                            matched_features = match_features_to_dark_regions(
                                raw_features,
                                dark_regions,
                                disk_info=disk_dict,
                                max_distance_ratio=0.30,  # GLM-4V spatial offset can be large
                                image_width=img_w,
                                image_height=img_h,
                            )
                            logger.info(f"CV matching: {len(raw_features)} AI features -> {len(matched_features)} matched features")
                            raw_features = matched_features
                        except Exception as e:
                            logger.error(f"CV feature matching failed: {e}", exc_info=True)
                    else:
                        logger.warning("No dark regions detected, keeping AI original coordinates")
                    
                    features = []
                    for feat in raw_features:
                        sf = SolarFeature.from_dict(feat)
                        features.append(sf)
                    
                    # Build intermediate steps
                    processing_steps = [
                        {"step": "ai_visual_analysis", "status": "completed", "model": result.model_name},
                    ]
                    if disk_validated:
                        processing_steps.append({
                            "step": "solar_disk_detection",
                            "status": "completed",
                            "method": "auto",
                            "confidence": disk_info.confidence,
                            "disk_info": disk_info.to_dict(),
                        })
                        processing_steps.append({
                            "step": "coordinate_validation",
                            "status": "completed",
                            "original_count": original_count,
                            "validated_count": len(raw_features),
                            "filtered_count": original_count - len(raw_features),
                        })
                    
                    analysis = SolarRegionAnalysis(
                        image_id=image_id,
                        image_path=image_path,
                        analysis_time=result.analysis_time,
                        is_solar_image=True,
                        region_count=len(features),
                        features=features,
                        hale_classification=result.hale_classification,
                        classification_confidence=result.classification_confidence,
                        confidence_level="high" if result.classification_confidence > 0.7 else "medium" if result.classification_confidence > 0.4 else "low",
                        complexity_score=result.complexity_score,
                        risk_level=result.risk_level,
                        reasoning=result.summary,
                        warnings=result.warnings,
                        recommended_actions=result.recommendations,
                        raw_model_output=result.raw_output,
                        token_usage={"model": result.model_name, "processing_time_ms": result.processing_time_ms},
                        intermediate_steps={
                            "processing_steps": processing_steps,
                            "solar_disk": disk_info.to_dict() if disk_validated else None,
                        },
                    )
                    logger.info(f"Model {model_key} analysis successful: {len(features)} features (after validation), Hale={result.hale_classification}")
                elif result and not result.success:
                    logger.warning(f"Model {model_key} analysis failed: {result.error}")
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        # If analysis failed, return error
        if analysis is None:
            tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
                "status": "failed", "progress": 0,
                "error": "AI分析失败，请检查API密钥配置和网络连接",
                "created_at": datetime.now().isoformat()})
            return
        
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "processing", "progress": 65, "created_at": datetime.now().isoformat()})

        # Get ALL features for display (before filtering)
        all_features = extract_features_for_display(analysis)

        # ======== ANALYSIS-STAGE CONFIDENCE FILTERING ========
        # Extract confidence threshold from prompt_config or options
        conf_threshold = None
        if prompt_config:
            conf_threshold = prompt_config.get("min_confidence")
        if conf_threshold is None:
            conf_threshold = options.get("confidence_threshold")

        filtered_features, removed_features, filter_stats = _filter_features_by_confidence(
            all_features, threshold=conf_threshold
        )
        per_type_info = []
        for k, v in filter_stats.get("by_type", {}).items():
            per_type_info.append(f"{k}: {v.get('kept',0)}/{v.get('total',0)}")
        logger.info(
            f"Confidence filter: {filter_stats['kept']}/{filter_stats['total']} features kept "
            f"(removed {filter_stats['removed']}). "
            f"Per-type: {{{', '.join(per_type_info)}}}"
        )
        # Use filtered features for reports/images/PDF/history
        features = filtered_features

        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "processing", "progress": 80, "created_at": datetime.now().isoformat()})

        # Record token usage
        if analysis.token_usage:
            token_tracker.record(
                task_id=task_id, image_id=image_id,
                model=analysis.token_usage.get("model", "deepseek-chat"),
                usage=analysis.token_usage,
            )

        # Record traceability (includes ALL features for full audit trail)
        trace_record = TraceabilityRecord(
            task_id=task_id,
            image_id=image_id,
            image_hash=compute_image_hash(image_path) if os.path.exists(image_path) else "",
            input_metadata={
                "filename": img_data.get("filename", ""),
                "source": img_data.get("source", ""),
                "wavelength": img_data.get("wavelength", ""),
            },
            image_features=analysis.intermediate_steps.get("processing_steps", [{}])[0].get("features", {}),
            algorithm_params={"analysis_type": analysis_type, "options": options},
            model_config={"model": analysis.token_usage.get("model", "deepseek-chat")} if analysis.token_usage else {},
            raw_model_output=analysis.raw_model_output,
            parsing_intermediates={"structured_json": analysis.intermediate_steps.get("structured_json")},
            final_result=analysis.to_dict(),
            warnings=analysis.warnings,
            processing_steps=analysis.intermediate_steps.get("processing_steps", []),
            scientific_conclusion=analysis.scientific_conclusion,
            flare_risk_assessment=analysis.flare_risk_assessment,
        )
        traceability_store.save(trace_record)

        # Build detection report from FILTERED features only
        detection_report = build_detection_report(
            image_id=image_id,
            image_path=image_path,
            features=features,
            hale_classification=analysis.hale_classification,
            complexity_score=analysis.complexity_score,
            disk_info=disk_dict,
        )
        logger.info(f"Built detection report from filtered features: {len(detection_report.sunspots)} sunspots")

        # Generate annotated image with filtered features
        annotated_path = generate_annotated_image_v2(image_path, detection_report, task_id)

        # Generate detection report image (statistics table)
        report_image_path = generate_detection_report_image(detection_report, task_id)

        # Generate combined report image (original + annotated side by side)
        combined_path = generate_combined_report_image(
            image_path, annotated_path, detection_report, task_id
        )

        # Build report with filtered content
        processing_time = (datetime.now() - start_time).total_seconds()
        risk = analyze_risk_level(analysis.complexity_score)

        report_id = generate_report_id()
        # Get image dimensions
        img_w, img_h = 0, 0
        try:
            from PIL import Image as PILImage
            with PILImage.open(str(image_path)) as _i:
                img_w, img_h = _i.size
        except Exception:
            pass

        report = {
            "id": report_id, "task_id": task_id, "image_id": image_id,
            "image_info": {
                "filename": img_data.get("filename", ""), "source": img_data.get("source", ""),
                "wavelength": img_data.get("wavelength", ""),
                "timestamp": img_data.get("timestamp", datetime.now().isoformat()),
                "image_size_bytes": img_data.get("size", 0),
                "width": img_w, "height": img_h,
            },
            "original_image_path": str(image_path) if os.path.exists(image_path) else "",
            "disk_info": disk_dict,
            "model_used": model_key,
            "analysis": {
                "hale_classification": analysis.hale_classification,
                "hale_distribution": analysis.hale_distribution,
                "classification_confidence": round(analysis.classification_confidence, 4),
                "confidence_level": analysis.confidence_level,
                "complexity_score": round(analysis.complexity_score, 2),
                "region_count": analysis.region_count,
                "polarity_distribution": analysis.polarity_distribution,
                "separation_pattern": analysis.separation_pattern,
                "risk_level": risk,
                "risk_score": round(min(analysis.complexity_score / 10, 1.0), 4),
                "features": features,  # filtered features for display
                "quantitative_metrics": {
                    "total_features_detected": len(features),
                    "sunspot_count": detection_report.total_sunspots,
                    "flare_count": detection_report.total_flares,
                    "sunspot_group_count": len(detection_report.sunspot_groups),
                    "bright_region_count": sum(1 for f in features if f.get("type") in ("bright_region", "plage")),
                    "mean_confidence": round(sum(f.get("confidence", 0) for f in features) / max(len(features), 1), 4),
                    "average_feature_size_relative": round(sum(f.get("size_relative", 0) for f in features) / max(len(features), 1), 4),
                },
                "feature_types_present": list(set(f.get("type") for f in features)),
                "recommendations": analysis.recommended_actions or [],
                "warnings": analysis.warnings or [],
            },
            # Save confidence filter info for traceability
            "confidence_filter": {
                "applied": filter_stats["removed"] > 0,
                "threshold_used": conf_threshold,
                "per_type_thresholds": CONFIDENCE_THRESHOLDS,
                "stats": filter_stats,
                "removed_features_count": len(removed_features),
            },
            # Keep original unfiltered features for audit trail
            "all_features_raw": all_features,
            "removed_features": removed_features,
            "detection_report": detection_report.to_dict(),
            "summary": analysis.reasoning or "",
            "original_image_path": str(image_path) if os.path.exists(image_path) else "",
            "annotated_image_path": annotated_path,
            "annotated_image_url": f"/api/v1/analyze/{task_id}/image" if annotated_path else None,
            "report_image_path": report_image_path,
            "report_image_url": f"/api/v1/analyze/{task_id}/report-image" if report_image_path else None,
            "combined_image_path": combined_path,
            "combined_image_url": f"/api/v1/analyze/{task_id}/combined-image" if combined_path else None,
            "token_usage": analysis.token_usage,
            "generated_at": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
        }

        # Save
        reports_store.set(report_id, report)

        history_store.set(report_id, {
            "id": report_id, "task_id": task_id, "image_id": image_id,
            "timestamp": datetime.now().isoformat(),
            "risk_level": risk,
            "hale_classification": analysis.hale_classification,
            "complexity_score": round(analysis.complexity_score, 2),
            "feature_count": len(features),
            "feature_count_raw": len(all_features),
            "feature_types": list(set(f.get("type") for f in features)),
            "processing_time_seconds": round(processing_time, 2),
            "confidence_filter_applied": filter_stats["removed"] > 0,
            "confidence_filter_threshold": conf_threshold,
            "annotated_image_path": annotated_path,
            "combined_image_path": combined_path,
            "has_images": bool(annotated_path and combined_path),
        })

        img_data["status"] = "analyzed"
        img_data["last_report_id"] = report_id
        images_store.set(image_id, img_data)

        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "completed", "progress": 100, "report_id": report_id,
            "created_at": datetime.now().isoformat(), "completed_at": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2)})

        logger.info(f"Analysis completed: task={task_id}, report={report_id}, "
                     f"features={len(features)}, time={processing_time:.1f}s")

    except Exception as e:
        logger.error(f"Analysis failed for task {task_id}: {e}", exc_info=True)
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "failed", "progress": 0, "error": str(e), "created_at": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# API Endpoints - Analysis
# ---------------------------------------------------------------------------

@router.post("/analyze", tags=["分析服务"])
async def create_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Create an analysis task for a solar image."""
    task_id = generate_task_id()
    model_key = request.model_key or "glm"
    tasks_store.set(task_id, {"task_id": task_id, "image_id": request.image_id,
        "status": "pending", "progress": 0, "created_at": datetime.now().isoformat(), "model_key": model_key})
    background_tasks.add_task(process_analysis, task_id, request.image_id, request.analysis_type, request.options or {}, model_key, request.prompt_config)
    return {"success": True, "data": {"task_id": task_id, "image_id": request.image_id,
        "status": "pending", "created_at": datetime.now().isoformat(), "model_key": model_key}, "message": "分析任务已创建，正在处理中"}


@router.get("/analyze/{task_id}", tags=["分析服务"])
async def get_analysis_status(task_id: str):
    """Query the status of an analysis task."""
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "任务不存在"})
    return {"success": True, "data": task}


@router.get("/analyze/{task_id}/report", tags=["分析服务"])
async def get_analysis_report(task_id: str):
    """Get the full analysis report for a completed task."""
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "任务不存在"})
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail={"code": "TASK_NOT_COMPLETED", "message": f"任务状态: {task.get('status')}"})
    report_id = task.get("report_id")
    if not report_id:
        raise HTTPException(status_code=404, detail={"code": "REPORT_NOT_FOUND", "message": "报告不存在"})
    report = reports_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail={"code": "REPORT_NOT_FOUND", "message": "报告不存在"})
    return {"success": True, "data": report}


@router.get("/analyze/{task_id}/image", tags=["分析服务"])
async def get_annotated_image(task_id: str):
    """Get the annotated solar image for a completed analysis."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "标注图像不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})
    annotated_path = report.get("annotated_image_path", "")
    if not annotated_path or not os.path.exists(annotated_path):
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "标注图像文件不存在"})
    return FileResponse(annotated_path, media_type="image/png")


@router.get("/analyze/{task_id}/report-image", tags=["分析服务"])
async def get_detection_report_image(task_id: str):
    """Get the detection report visualization image."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告图像不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})
    report_image_path = report.get("report_image_path", "")
    if not report_image_path or not os.path.exists(report_image_path):
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告图像文件不存在"})
    return FileResponse(report_image_path, media_type="image/png")


@router.get("/analyze/{task_id}/combined-image", tags=["分析服务"])
async def get_combined_report_image(task_id: str):
    """Get the combined report image (original + annotated side by side)."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "组合报告图像不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})
    combined_path = report.get("combined_image_path", "")
    if not combined_path or not os.path.exists(combined_path):
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "组合报告图像文件不存在"})
    return FileResponse(combined_path, media_type="image/png")


@router.get("/analyze/{task_id}/detection-report", tags=["分析服务"])
async def get_detection_report(task_id: str, format: str = "json"):
    """Get detection report in JSON or CSV format."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})

    detection_report = report.get("detection_report", {})

    if format == "csv":
        from fastapi.responses import StreamingResponse
        import io

        # Rebuild CSV from detection report data
        lines = ["# Solar Feature Detection Report"]
        lines.append(f"# Image: {report.get('image_info', {}).get('filename', '')}")
        lines.append(f"# Generated: {report.get('generated_at', '')}")
        lines.append(f"# Hale Classification: {detection_report.get('hale_classification', '')}")
        lines.append(f"# Complexity Score: {detection_report.get('complexity_score', 0)}")
        summary = detection_report.get("summary", {})
        lines.append(f"# Total Sunspots: {summary.get('total_sunspots', 0)}")
        lines.append(f"# Total Flares: {summary.get('total_flares', 0)}")
        lines.append("")
        lines.append("Type,Index,Label,X,Y,Size,Confidence,Group_ID,Checked")

        for s in detection_report.get("sunspots", []):
            pos = s.get("position", {})
            lines.append(f"sunspot,{s['index']},{s['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{s['size_relative']:.4f},{s['confidence']:.4f},{s.get('group_id','')},{s.get('checked',True)}")

        for g in detection_report.get("sunspot_groups", []):
            pos = g.get("position", {})
            lines.append(f"sunspot_group,{g['index']},{g['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{g['size_relative']:.4f},{g['confidence']:.4f},,{g.get('checked',True)}")

        for f in detection_report.get("flares", []):
            pos = f.get("position", {})
            lines.append(f"flare,{f['index']},{f['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{f['size_relative']:.4f},{f['confidence']:.4f},,{f.get('checked',True)}")

        for o in detection_report.get("other_features", []):
            pos = o.get("position", {})
            lines.append(f"{o['type']},{o['index']},{o['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{o['size_relative']:.4f},{o['confidence']:.4f},,{o.get('checked',True)}")

        csv_content = "\n".join(lines)
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=detection_report_{task_id}.csv"},
        )
    else:
        return {"success": True, "data": detection_report}


@router.get("/analyze/{task_id}/report-pdf", tags=["分析服务"])
async def get_report_pdf(task_id: str):
    """Download single-task PDF report with images and analysis."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})
    
    from pdf_report_generator import generate_single_report_pdf
    
    pdf_filename = f"solar_report_{task_id}.pdf"
    pdf_path = str(REPORTS_DIR / pdf_filename)
    
    try:
        generate_single_report_pdf(report, pdf_path, include_images=True)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_filename,
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"code": "PDF_ERROR", "message": str(e)})


@router.post("/analyze/batch-export-pdf", tags=["分析服务"])
async def batch_export_pdf(request: Dict[str, Any]):
    """Batch export multiple reports as a single multi-page PDF.
    
    Body: {"task_ids": ["task-1", "task-2", ...]}
    """
    task_ids = request.get("task_ids", [])
    if not task_ids:
        raise HTTPException(status_code=400, detail={"code": "NO_TASKS", "message": "请提供任务ID列表"})
    
    from pdf_report_generator import generate_batch_pdf
    
    reports = []
    for tid in task_ids:
        task = tasks_store.get(tid)
        if not task or task.get("status") != "completed":
            continue
        report = reports_store.get(task.get("report_id", ""))
        if report:
            reports.append(report)
    
    if not reports:
        raise HTTPException(status_code=404, detail={"code": "NO_REPORTS", "message": "未找到有效的报告"})
    
    pdf_filename = f"solar_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = str(REPORTS_DIR / pdf_filename)
    
    try:
        generate_batch_pdf(reports, pdf_path, include_images=True)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_filename,
        )
    except Exception as e:
        logger.error(f"Batch PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"code": "PDF_ERROR", "message": str(e)})


@router.put("/analyze/{task_id}/detection-report/features/{feature_type}/{feature_index}", tags=["分析服务"])
async def update_feature_check_status(
    task_id: str,
    feature_type: str,
    feature_index: int,
    request: Dict[str, bool],
):
    """Update the checked status of a detected feature (for interactive correction).

    Body: {"checked": true/false}
    
    Frontend sends 1-based position index from analysis.features array.
    Updates both analysis.features and detection_report sections.
    """
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})

    analysis = report.get("analysis", {})
    features = analysis.get("features", [])
    
    detection_report = report.get("detection_report", {})
    checked = request.get("checked", True)

    # Map feature type to detection report section name
    feature_type_to_section = {
        "sunspot": "sunspots",
        "sunspot_group": "sunspot_groups",
        "flare": "flares",
        "bright_region": "bright_regions",
        "plage": "bright_regions",
        "facula": "bright_regions",
        "prominence": "other_features",
        "filament": "other_features",
        "other": "other_features",
    }
    feature_list_name = feature_type_to_section.get(feature_type, f"{feature_type}s")
    section_features = detection_report.get(feature_list_name, [])

    updated = False
    
    # Primary: match by position in the flat features array (1-based from frontend)
    if 1 <= feature_index <= len(features):
        feat = features[feature_index - 1]
        # Verify type matches
        if feat.get("type") == feature_type or feat.get("feature_type") == feature_type:
            feat["checked"] = checked
            updated = True
            # Also update corresponding detection_report feature
            for d_feat in section_features:
                if d_feat.get("label") == feat.get("label"):
                    d_feat["checked"] = checked
    
    # Fallback: try to find by type count in section features
    if not updated and section_features:
        type_count = 0
        for d_feat in section_features:
            feat_type = d_feat.get("type", d_feat.get("feature_type", ""))
            if feat_type == feature_type:
                type_count += 1
                if type_count == feature_index:
                    d_feat["checked"] = checked
                    updated = True
                    break

    if not updated:
        raise HTTPException(status_code=404, detail={"code": "FEATURE_NOT_FOUND", "message": f"特征 {feature_type}#{feature_index} 不存在"})

    # Update report
    report["analysis"] = analysis
    report["detection_report"] = detection_report
    reports_store.set(report["id"], report)

    return {"success": True, "message": f"特征 {feature_type}#{feature_index} 状态已更新为 {'已勾选' if checked else '已取消'}"}


@router.post("/analyze/batch", tags=["分析服务"])
async def batch_analyze(request: BatchAnalyzeRequest, background_tasks: BackgroundTasks):
    """Create batch analysis for multiple images."""
    batch_task_id = f"batch-{generate_task_id()}"
    tasks_store.set(batch_task_id, {"task_id": batch_task_id, "image_ids": request.image_ids,
        "status": "pending", "progress": 0, "total_images": len(request.image_ids),
        "completed_images": 0, "created_at": datetime.now().isoformat()})
    for image_id in request.image_ids:
        task_id = generate_task_id()
        tasks_store.set(task_id, {"task_id": task_id, "image_id": image_id,
            "status": "pending", "progress": 0, "batch_id": batch_task_id, "created_at": datetime.now().isoformat()})
        background_tasks.add_task(process_analysis, task_id, image_id, request.analysis_type, request.options or {}, getattr(request, 'model_key', 'glm'), getattr(request, 'prompt_config', None))
    return {"success": True, "data": {"batch_task_id": batch_task_id, "image_ids": request.image_ids, "status": "processing"},
            "message": f"批量分析已创建，共 {len(request.image_ids)} 张图像"}


# ---------------------------------------------------------------------------
# API Endpoints - Prompt Templates
# ---------------------------------------------------------------------------

@router.get("/prompt-templates", tags=["提示词模板"])
async def get_prompt_templates():
    """获取所有预设提示词模板"""
    from prompt_templates import list_presets
    return {"success": True, "data": list_presets()}


@router.get("/prompt-templates/{name}", tags=["提示词模板"])
async def get_prompt_template(name: str):
    """获取指定预设模板"""
    from prompt_templates import get_preset
    config = get_preset(name)
    if config:
        return {"success": True, "data": {"name": name, "config": config.to_dict()}}
    return {"success": False, "error": "TEMPLATE_NOT_FOUND", "message": f"模板 '{name}' 不存在"}


@router.post("/prompt-templates/build", tags=["提示词模板"])
async def build_prompt(config: Dict):
    """根据配置生成提示词（用于预览）"""
    from prompt_templates import PromptConfig, build_system_prompt, build_user_prompt
    pconfig = PromptConfig.from_dict(config)
    return {
        "success": True,
        "data": {
            "system_prompt": build_system_prompt(pconfig),
            "user_prompt": build_user_prompt(pconfig),
        }
    }


# ---------------------------------------------------------------------------
# API Endpoints - Reports
# ---------------------------------------------------------------------------

@router.get("/reports", tags=["报告管理"])
async def get_reports(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = None,
    hale_classification: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    feature_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """List all analysis reports with pagination and advanced filtering.

    Filters:
    - risk_level: Filter by risk level (low/moderate/high)
    - hale_classification: Filter by Hale class (Alpha/Beta/Beta-Gamma/Gamma/Delta/Beta-Delta)
    - min_confidence/max_confidence: Filter by classification confidence range
    - feature_type: Filter by detected feature type (sunspot/flare/plage/etc)
    - start_date/end_date: Filter by analysis date range (YYYY-MM-DD)
    """
    all_reports = reports_store.list_all()

    # Apply filters
    if risk_level:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("risk_level") == risk_level]

    if hale_classification:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("hale_classification") == hale_classification]

    if min_confidence is not None:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("classification_confidence", 0) >= min_confidence]

    if max_confidence is not None:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("classification_confidence", 1) <= max_confidence]

    if feature_type:
        all_reports = [
            r for r in all_reports
            if any(f.get("type") == feature_type for f in r.get("analysis", {}).get("features", []))
        ]

    if start_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") >= start_date]

    if end_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") <= end_date + "T23:59:59"]

    all_reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    total = len(all_reports)
    start = (page - 1) * limit
    end = start + limit
    items = all_reports[start:end]

    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": [
                {
                    "id": r.get("id"),
                    "image_id": r.get("image_id"),
                    "task_id": r.get("task_id"),
                    "risk_level": r.get("analysis", {}).get("risk_level", "unknown"),
                    "hale_classification": r.get("analysis", {}).get("hale_classification", "Unknown"),
                    "complexity_score": r.get("analysis", {}).get("complexity_score", 0),
                    "classification_confidence": r.get("analysis", {}).get("classification_confidence", 0),
                    "feature_count": len(r.get("analysis", {}).get("features", [])),
                    "feature_types": list(set(f.get("type") for f in r.get("analysis", {}).get("features", []))),
                    "generated_at": r.get("generated_at"),
                    "processing_time_seconds": r.get("processing_time_seconds", 0),
                    "has_annotated_image": bool(r.get("annotated_image_path")),
                }
                for r in items
            ],
        },
    }


@router.get("/reports/{report_id}", tags=["报告管理"])
async def get_report(report_id: str):
    """Get a single report by ID."""
    report = reports_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail={"code": "REPORT_NOT_FOUND", "message": "报告不存在"})
    return {"success": True, "data": report}


@router.delete("/reports/{report_id}", tags=["报告管理"])
async def delete_report(report_id: str):
    """Delete a report and its associated data."""
    report = reports_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail={"code": "REPORT_NOT_FOUND", "message": "报告不存在"})
    annotated_path = report.get("annotated_image_path", "")
    if annotated_path and os.path.exists(annotated_path):
        try:
            os.remove(annotated_path)
        except OSError:
            pass
    reports_store.delete(report_id)
    history_store.delete(report_id)
    return {"success": True, "message": "报告已删除"}


@router.get("/reports/export/csv", tags=["报告管理"])
async def export_reports_csv(
    risk_level: Optional[str] = None,
    hale_classification: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Export filtered reports as CSV file.

    Uses same filters as GET /reports endpoint.
    Returns CSV with columns: ID, Image_ID, Task_ID, Risk_Level, Hale_Classification,
    Complexity_Score, Confidence, Feature_Count, Generated_At, Processing_Time
    """
    import csv
    import io
    from fastapi.responses import StreamingResponse

    all_reports = reports_store.list_all()

    # Apply filters
    if risk_level:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("risk_level") == risk_level]
    if hale_classification:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("hale_classification") == hale_classification]
    if start_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") >= start_date]
    if end_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") <= end_date + "T23:59:59"]

    all_reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Image_ID", "Task_ID", "Risk_Level", "Hale_Classification",
        "Complexity_Score", "Confidence", "Feature_Count", "Generated_At", "Processing_Time"
    ])

    for r in all_reports:
        writer.writerow([
            r.get("id", ""),
            r.get("image_id", ""),
            r.get("task_id", ""),
            r.get("analysis", {}).get("risk_level", ""),
            r.get("analysis", {}).get("hale_classification", ""),
            r.get("analysis", {}).get("complexity_score", 0),
            r.get("analysis", {}).get("classification_confidence", 0),
            len(r.get("analysis", {}).get("features", [])),
            r.get("generated_at", ""),
            r.get("processing_time_seconds", 0),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=solar_analysis_reports.csv"},
    )


@router.get("/reports/export/json", tags=["报告管理"])
async def export_reports_json(
    risk_level: Optional[str] = None,
    hale_classification: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Export filtered reports as JSON file.

    Uses same filters as GET /reports endpoint.
    Returns complete JSON data for all matching reports.
    """
    from fastapi.responses import JSONResponse

    all_reports = reports_store.list_all()

    # Apply filters
    if risk_level:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("risk_level") == risk_level]
    if hale_classification:
        all_reports = [r for r in all_reports if r.get("analysis", {}).get("hale_classification") == hale_classification]
    if start_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") >= start_date]
    if end_date:
        all_reports = [r for r in all_reports if r.get("generated_at", "") <= end_date + "T23:59:59"]

    all_reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)

    return JSONResponse(
        content={"success": True, "data": {"total": len(all_reports), "reports": all_reports}},
        headers={"Content-Disposition": "attachment; filename=solar_analysis_reports.json"},
    )


# ---------------------------------------------------------------------------
# API Endpoints - Traceability
# ---------------------------------------------------------------------------

@router.get("/traceability/{task_id}", tags=["可追溯性"])
async def get_traceability(task_id: str):
    """Get complete traceability record for an analysis task."""
    record = traceability_store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "追溯记录不存在"})
    return {"success": True, "data": record}


@router.get("/traceability", tags=["可追溯性"])
async def list_traceability(start: Optional[str] = None, end: Optional[str] = None, page: int = Query(1, ge=1), limit: int = Query(50, ge=1)):
    """List traceability records with pagination."""
    records = traceability_store.list_all(start=start, end=end)
    total = len(records)
    items = [{"task_id": r.get("task_id"), "image_id": r.get("image_id"), "created_at": r.get("created_at"),
              "hale_classification": r.get("final_result", {}).get("hale_classification", ""),
              "is_solar_image": r.get("final_result", {}).get("is_solar_image", True),
              "feature_count": len(r.get("final_result", {}).get("features", []))} for r in records[(page-1)*limit:page*limit]]
    return {"success": True, "data": {"total": total, "page": page, "limit": limit, "items": items}}


# ---------------------------------------------------------------------------
# API Endpoints - Token Usage
# ---------------------------------------------------------------------------

@router.get("/token-usage/summary", tags=["Token使用量"])
async def get_token_usage_summary(start: Optional[str] = None, end: Optional[str] = None):
    """Get aggregated token usage statistics."""
    return {"success": True, "data": token_tracker.get_summary(start=start, end=end)}


@router.get("/token-usage/daily", tags=["Token使用量"])
async def get_token_usage_daily(days: int = Query(7, ge=1, le=90)):
    """Get daily token usage for the last N days."""
    return {"success": True, "data": {"days": token_tracker.get_daily_report(days)}}


@router.get("/token-usage/report/{period}", tags=["Token使用量"])
async def get_token_usage_periodic_report(period: str):
    """Generate periodic token usage report.

    Args:
        period: "daily", "weekly", or "monthly"

    Returns comprehensive report with usage statistics, trends, and cost analysis.
    """
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PERIOD", "message": "Period must be 'daily', 'weekly', or 'monthly'"},
        )
    report = token_tracker.get_periodic_report(period)
    return {"success": True, "data": report}


@router.get("/token-usage/records", tags=["Token使用量"])
async def get_token_usage_records(task_id: Optional[str] = None, page: int = Query(1, ge=1), limit: int = Query(50, ge=1)):
    """Get individual token usage records."""
    records = token_tracker.get_records(task_id=task_id)
    total = len(records)
    items = records[(page-1)*limit:page*limit]
    return {"success": True, "data": {"total": total, "page": page, "limit": limit, "items": items}}


# ---------------------------------------------------------------------------
# API Endpoints - Review / Manual Verification
# ---------------------------------------------------------------------------

@router.post("/reviews", tags=["人工复核"])
async def create_review(request: ReviewRecord):
    """Create a manual review record with enhanced parameter modification tracking."""
    review = {
        "id": f"rev-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "report_id": request.report_id,
        "task_id": request.task_id,
        "reviewer": request.reviewer,
        "original_result": request.original_result,
        "corrected_result": request.corrected_result,
        "verification_status": request.verification_status,
        "comments": request.comments,
        "correction_fields": request.correction_fields or [],
        "modified_params": request.modified_params,
        "param_change_reason": request.param_change_reason,
        "param_change_history": [],  # track individual param changes
        "created_at": datetime.now().isoformat(),
    }

    # Build param change history if modified_params provided
    if request.modified_params and request.original_result:
        for key, new_val in request.modified_params.items():
            old_val = request.original_result.get(key)
            if old_val != new_val:
                review["param_change_history"].append({
                    "param": key,
                    "old_value": old_val,
                    "new_value": new_val,
                    "changed_at": datetime.now().isoformat(),
                })

    _save_review(review)
    return {"success": True, "data": review, "message": "复核记录已保存"}


@router.get("/reviews", tags=["人工复核"])
async def list_reviews(status: Optional[str] = None, reviewer: Optional[str] = None, page: int = Query(1, ge=1), limit: int = Query(50, ge=1)):
    """List all review records with filtering."""
    records = _read_reviews()
    if status:
        records = [r for r in records if r.get("verification_status") == status]
    if reviewer:
        records = [r for r in records if r.get("reviewer") == reviewer]
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    total = len(records)
    items = records[(page-1)*limit:page*limit]
    return {"success": True, "data": {"total": total, "page": page, "limit": limit, "items": items}}


@router.patch("/reviews/{review_id}", tags=["人工复核"])
async def update_review(review_id: str, status: str = "", comments: str = ""):
    """Update a review record."""
    records = _read_reviews()
    for r in records:
        if r.get("id") == review_id:
            if status:
                r["verification_status"] = status
            if comments:
                r["comments"] = comments
            r["updated_at"] = datetime.now().isoformat()
            _write_reviews(records)
            return {"success": True, "data": r}
    raise HTTPException(status_code=404, detail={"code": "REVIEW_NOT_FOUND", "message": "复核记录不存在"})


# ---------------------------------------------------------------------------
# API Endpoints - Administrator
# ---------------------------------------------------------------------------

@router.get("/admin/dashboard", tags=["管理员工具"])
async def get_admin_dashboard():
    """Get comprehensive admin dashboard with system performance metrics."""
    all_reports = reports_store.list_all()
    reviews = _read_reviews()
    token_summary = token_tracker.get_summary()

    n = len(all_reports)
    if n == 0:
        return {"success": True, "data": {"message": "暂无数据，请先进行图像分析"}}

    # Accuracy metrics from reviews
    confirmed = sum(1 for r in reviews if r.get("verification_status") == "confirmed")
    corrected = sum(1 for r in reviews if r.get("verification_status") == "corrected")
    disputed = sum(1 for r in reviews if r.get("verification_status") == "disputed")
    review_total = len(reviews)

    # Risk distribution
    risk_dist = {"low": 0, "moderate": 0, "high": 0}
    hale_dist: Dict[str, int] = {}
    complexity_scores = []

    for r in all_reports:
        a = r.get("analysis", {})
        rl = a.get("risk_level", "low")
        if rl in risk_dist: risk_dist[rl] += 1
        hc = a.get("hale_classification", "Unknown")
        hale_dist[hc] = hale_dist.get(hc, 0) + 1
        complexity_scores.append(a.get("complexity_score", 0))

    avg_complexity = sum(complexity_scores) / n if n > 0 else 0

    return {
        "success": True,
        "data": {
            "system_metrics": {
                "total_analyses": n,
                "total_images": images_store.count(),
                "total_tasks": tasks_store.count(),
                "total_reviews": review_total,
                "review_coverage_rate": round(review_total / max(n, 1) * 100, 1),
            },
            "accuracy_metrics": {
                "confirmed_rate": round(confirmed / max(review_total, 1) * 100, 1),
                "corrected_rate": round(corrected / max(review_total, 1) * 100, 1),
                "disputed_rate": round(disputed / max(review_total, 1) * 100, 1),
                "total_reviews": review_total,
            },
            "risk_distribution": risk_dist,
            "hale_classification_distribution": hale_dist,
            "complexity_metrics": {
                "mean": round(avg_complexity, 2),
                "max": round(max(complexity_scores) if complexity_scores else 0, 2),
                "min": round(min(complexity_scores) if complexity_scores else 0, 2),
            },
            "token_usage_summary": token_summary,
        },
    }


@router.get("/admin/performance-report", tags=["管理员工具"])
async def get_performance_report(days: int = Query(30, ge=1, le=365)):
    """Generate a comprehensive performance report."""
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    token_summary = token_tracker.get_summary(start=cutoff)
    daily_usage = token_tracker.get_daily_report(days)

    return {
        "success": True,
        "data": {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "token_usage": {
                "summary": token_summary,
                "daily_breakdown": daily_usage,
            },
        },
    }


@router.post("/admin/config", tags=["管理员工具"])
async def update_admin_config(config: AdminConfig):
    """Update system-wide analysis configuration (admin only)."""
    current_config = {}
    config_path = DATA_DIR / "admin_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                current_config = json.load(f)
        except Exception:
            pass

    if config.max_tokens_per_request is not None:
        current_config["max_tokens_per_request"] = config.max_tokens_per_request
    if config.confidence_threshold is not None:
        current_config["confidence_threshold"] = config.confidence_threshold
    if config.enable_heuristic_fallback is not None:
        current_config["enable_heuristic_fallback"] = config.enable_heuristic_fallback
    if config.recognition_threshold is not None:
        current_config["recognition_threshold"] = config.recognition_threshold
    if config.hale_classification_weights is not None:
        current_config["hale_classification_weights"] = config.hale_classification_weights
    if config.feature_type_weights is not None:
        current_config["feature_type_weights"] = config.feature_type_weights
    if config.complexity_score_formula is not None:
        current_config["complexity_score_formula"] = config.complexity_score_formula

    with open(config_path, "w") as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

    return {"success": True, "data": current_config, "message": "配置已更新"}


class FeatureFlagRequest(BaseModel):
    """Request to flag a detected feature with a comment."""
    flag: str  # "correct" | "false_positive" | "suspicious" | "missed"
    comment: str = ""


class ArchiveRequest(BaseModel):
    """Request to archive an analysis result with screenshot."""
    task_id: str
    screenshot_data: Optional[str] = None  # base64 encoded screenshot
    notes: str = ""


@router.get("/admin/config", tags=["管理员工具"])
async def get_admin_config():
    """Get current system configuration."""
    config_path = DATA_DIR / "admin_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return {"success": True, "data": json.load(f)}
        except Exception:
            pass
    return {"success": True, "data": {}}


# ---------------------------------------------------------------------------
# API Endpoints - Feature Flagging
# ---------------------------------------------------------------------------

ARCHIVE_DIR = DATA_DIR / "archives"
os.makedirs(ARCHIVE_DIR, exist_ok=True)


@router.put("/analyze/{task_id}/features/{feature_type}/{feature_index}/flag", tags=["分析服务"])
async def flag_feature(
    task_id: str,
    feature_type: str,
    feature_index: int,
    request: FeatureFlagRequest,
):
    """Flag a detected feature with status and comment.
    
    Flags: correct (正确) | false_positive (误判) | suspicious (可疑) | missed (漏检)
    Works directly with analysis.features list (1-based index).
    """
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})

    # Work directly with analysis.features (flat list, matches frontend display)
    analysis = report.get("analysis", {})
    features = analysis.get("features", [])
    
    # Build section-level features list from detection_report too
    detection_report = report.get("detection_report", {})
    feature_type_to_section = {
        "sunspot": "sunspots",
        "sunspot_group": "sunspot_groups",
        "flare": "flares",
        "bright_region": "bright_regions",
        "plage": "bright_regions",
        "facula": "bright_regions",
        "prominence": "other_features",
        "filament": "other_features",
        "other": "other_features",
    }
    section_name = feature_type_to_section.get(feature_type, f"{feature_type}s")
    section_features = detection_report.get(section_name, [])
    
    updated = False
    
    # Frontend sends 1-based position index from the features array
    # Match by position in the flat features list (frontend: features.map((f, idx) => idx + 1))
    if 1 <= feature_index <= len(features):
        feat = features[feature_index - 1]
        # Verify type matches
        if feat.get("type") == feature_type or feat.get("feature_type") == feature_type:
            feat["flag"] = request.flag
            feat["flag_comment"] = request.comment
            feat["flagged_at"] = datetime.now().isoformat()
            updated = True
        # Also update corresponding detection_report feature
        for d_feat in section_features:
            # Match by position in section or by label/type proximity
            if d_feat.get("index") == feature_index or d_feat.get("label") == feat.get("label"):
                d_feat["flag"] = request.flag
                d_feat["flag_comment"] = request.comment
                d_feat["flagged_at"] = datetime.now().isoformat()
    
    # Fallback: try to find by type matching at the same index in section features
    if not updated and section_features:
        # Count features of the same type up to the index
        type_count = 0
        for d_feat in section_features:
            feat_type = d_feat.get("type", d_feat.get("feature_type", ""))
            if feat_type == feature_type:
                type_count += 1
                if type_count == feature_index:
                    d_feat["flag"] = request.flag
                    d_feat["flag_comment"] = request.comment
                    d_feat["flagged_at"] = datetime.now().isoformat()
                    updated = True
                    break
    
    if not updated:
        raise HTTPException(status_code=404, detail={"code": "FEATURE_NOT_FOUND", "message": f"特征 {feature_type}#{feature_index} 不存在"})

    report["analysis"] = analysis
    report["detection_report"] = detection_report
    reports_store.set(report["id"], report)

    flag_labels = {
        "correct": "正确",
        "false_positive": "误判",
        "suspicious": "可疑",
        "missed": "漏检",
    }
    flag_label = flag_labels.get(request.flag, request.flag)

    return {"success": True, "message": f"特征 {feature_type}#{feature_index} 已标记为 {flag_label}"}


@router.get("/analyze/{task_id}/flags", tags=["分析服务"])
async def get_feature_flags(task_id: str):
    """Get all feature flags for an analysis task."""
    task = tasks_store.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})

    detection_report = report.get("detection_report", {})
    flags = []
    
    for section in ["sunspots", "sunspot_groups", "flares", "bright_regions", "other_features"]:
        for feat in detection_report.get(section, []):
            if "flag" in feat:
                flags.append({
                    "feature_type": section[:-1] if section != "other_features" else "other",
                    "index": feat.get("index"),
                    "label": feat.get("label", ""),
                    "flag": feat.get("flag"),
                    "flag_comment": feat.get("flag_comment", ""),
                    "flagged_at": feat.get("flagged_at"),
                })

    return {"success": True, "data": {"flags": flags, "total": len(flags)}}


# ---------------------------------------------------------------------------
# API Endpoints - Archive
# ---------------------------------------------------------------------------

@router.post("/archive", tags=["归档管理"])
async def archive_analysis(request: ArchiveRequest):
    """Archive an analysis result with optional screenshot.
    
    Saves the full report, flags, detection results, and optional page screenshot
    to the archive directory for long-term storage.
    """
    task = tasks_store.get(request.task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "任务不存在"})
    report = reports_store.get(task.get("report_id", ""))
    if not report:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "报告不存在"})

    # Build archive record
    archive_id = f"arch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    archive_record = {
        "archive_id": archive_id,
        "task_id": request.task_id,
        "report_id": report.get("id"),
        "image_id": report.get("image_id"),
        "image_info": report.get("image_info", {}),
        "analysis": report.get("analysis", {}),
        "detection_report": report.get("detection_report", {}),
        "summary": report.get("summary", ""),
        "flags": [],
        "notes": request.notes,
        "archived_at": datetime.now().isoformat(),
        "processing_time_seconds": report.get("processing_time_seconds", 0),
    }

    # Collect all flags
    detection_report = report.get("detection_report", {})
    for section in ["sunspots", "sunspot_groups", "flares", "bright_regions", "other_features"]:
        for feat in detection_report.get(section, []):
            if "flag" in feat:
                archive_record["flags"].append({
                    "feature_type": section[:-1] if section != "other_features" else "other",
                    "index": feat.get("index"),
                    "label": feat.get("label", ""),
                    "flag": feat.get("flag"),
                    "flag_comment": feat.get("flag_comment", ""),
                })

    # Save screenshot if provided
    screenshot_path = None
    if request.screenshot_data:
        try:
            import base64
            img_data = base64.b64decode(request.screenshot_data.split(",")[-1] if "," in request.screenshot_data else request.screenshot_data)
            screenshot_path = str(ARCHIVE_DIR / f"{archive_id}_screenshot.png")
            with open(screenshot_path, "wb") as f:
                f.write(img_data)
            archive_record["screenshot_path"] = screenshot_path
        except Exception as e:
            logger.warning(f"Failed to save screenshot: {e}")

    # Save annotated image copy if available
    annotated_path = report.get("annotated_image_path", "")
    if annotated_path and os.path.exists(annotated_path):
        import shutil
        dest_path = str(ARCHIVE_DIR / f"{archive_id}_annotated.png")
        shutil.copy2(annotated_path, dest_path)
        archive_record["annotated_image_path"] = dest_path

    # Save report image copy if available
    report_image_path = report.get("report_image_path", "")
    if report_image_path and os.path.exists(report_image_path):
        import shutil
        dest_path = str(ARCHIVE_DIR / f"{archive_id}_report.png")
        shutil.copy2(report_image_path, dest_path)
        archive_record["report_image_path"] = dest_path

    # Save archive JSON
    archive_json_path = ARCHIVE_DIR / f"{archive_id}.json"
    with open(archive_json_path, "w", encoding="utf-8") as f:
        json.dump(archive_record, f, ensure_ascii=False, indent=2, default=str)

    return {
        "success": True,
        "data": {
            "archive_id": archive_id,
            "task_id": request.task_id,
            "flag_count": len(archive_record["flags"]),
            "has_screenshot": bool(screenshot_path),
            "has_annotated_image": bool(archive_record.get("annotated_image_path")),
            "archived_at": archive_record["archived_at"],
        },
        "message": "归档已保存"
    }


@router.get("/archive", tags=["归档管理"])
async def list_archives(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """List all archived analysis results."""
    archives = []
    if ARCHIVE_DIR.exists():
        for f in ARCHIVE_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    archives.append(json.load(fh))
            except Exception:
                pass

    # Sort by archived_at
    archives.sort(key=lambda a: a.get("archived_at", ""), reverse=True)

    # Apply date filters
    if start_date:
        archives = [a for a in archives if a.get("archived_at", "") >= start_date]
    if end_date:
        archives = [a for a in archives if a.get("archived_at", "") <= end_date + "T23:59:59"]

    total = len(archives)
    start = (page - 1) * limit
    end = start + limit
    items = archives[start:end]

    return {
        "success": True,
        "data": {
            "total": total,
            "page": page,
            "limit": limit,
            "items": [
                {
                    "archive_id": a.get("archive_id"),
                    "task_id": a.get("task_id"),
                    "image_id": a.get("image_id"),
                    "filename": a.get("image_info", {}).get("filename", ""),
                    "hale_classification": a.get("analysis", {}).get("hale_classification", ""),
                    "complexity_score": a.get("analysis", {}).get("complexity_score", 0),
                    "feature_count": len(a.get("analysis", {}).get("features", [])),
                    "flag_count": len(a.get("flags", [])),
                    "has_screenshot": bool(a.get("screenshot_path")),
                    "archived_at": a.get("archived_at"),
                    "notes": a.get("notes", ""),
                }
                for a in items
            ],
        },
    }


@router.get("/archive/{archive_id}", tags=["归档管理"])
async def get_archive(archive_id: str):
    """Get a single archived analysis result."""
    archive_path = ARCHIVE_DIR / f"{archive_id}.json"
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "归档不存在"})
    with open(archive_path, "r", encoding="utf-8") as f:
        archive = json.load(f)
    return {"success": True, "data": archive}


@router.delete("/archive/{archive_id}", tags=["归档管理"])
async def delete_archive(archive_id: str):
    """Delete an archived analysis result and its associated files."""
    archive_path = ARCHIVE_DIR / f"{archive_id}.json"
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "归档不存在"})
    
    with open(archive_path, "r", encoding="utf-8") as f:
        archive = json.load(f)
    
    # Remove associated files
    for path_key in ["screenshot_path", "annotated_image_path", "report_image_path"]:
        p = archive.get(path_key, "")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    
    os.remove(archive_path)
    return {"success": True, "message": "归档已删除"}
