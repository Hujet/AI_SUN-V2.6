"""
CV Preprocessing Preview API - Step-by-step visualization of solar image analysis.

Provides detailed intermediate results of CV detection for debugging and verification.
"""

import base64
import io
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

# Data directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CV_PREVIEW_DIR = DATA_DIR / "cv_preview"
CV_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# In-memory image store (shared with images.py)
try:
    from persistent_store import get_images_store
    images_store = get_images_store()
except ImportError:
    images_store: Dict[str, Dict[str, Any]] = {}


def _resolve_image_path(image_id: str) -> Optional[str]:
    """Resolve image file path through multiple methods."""
    # Method 1: Import the images module to access its store directly
    try:
        from images import images_store as images_module_store
        img_data = images_module_store.get(image_id)
        if img_data and img_data.get("file_path"):
            return img_data["file_path"]
    except (ImportError, Exception):
        pass
    
    # Method 2: Check our local store
    if images_store:
        img_data = images_store.get(image_id)
        if img_data and img_data.get("file_path"):
            return img_data["file_path"]
    
    # Method 3: Search upload directory for matching files
    if UPLOAD_DIR.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            path = UPLOAD_DIR / f"{image_id}{ext}"
            if path.exists():
                return str(path)
        
        # Try partial match
        for f in UPLOAD_DIR.iterdir():
            if image_id in f.stem and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                return str(f)
    
    return None

router = APIRouter(prefix="/api/v1", tags=["CV检测预览"])


def _encode_image_to_base64(img_array: np.ndarray) -> str:
    """Encode a numpy image array to base64 string."""
    try:
        from PIL import Image
        if img_array.ndim == 2:
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        elif img_array.ndim == 3:
            img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        else:
            return ""
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Image encoding failed: {e}")
        return ""


def _matplotlib_figure_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 PNG string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='black')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Matplotlib figure encoding failed: {e}")
        return ""
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def _generate_step_visualizations(
    image: np.ndarray,
    disk_info: Dict,
    sunspots: List,
    bright_regions: List,
    groups: List,
    stats: Dict,
) -> Dict[str, str]:
    """Generate visualization images for each CV step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    result_images = {}
    h, w = image.shape
    
    # Step 1: Original + Disk Boundary
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(image, cmap='gray')
    if disk_info.get("detected"):
        cx, cy, r = disk_info["center_x"], disk_info["center_y"], disk_info["radius"]
        circle = mpatches.Circle((cx, cy), r, fill=False, edgecolor='orange', linewidth=3, linestyle='-')
        ax1.add_patch(circle)
        ax1.plot(cx, cy, 'o', color='orange', markersize=10)
        ax1.annotate(f'日面\nr={r:.0f}', xy=(cx+r*0.05, cy-r*0.05), color='orange', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="orange", alpha=0.8))
        ax1.set_title(f'Step 1: 日面边界检测 (方法: {disk_info["method"]}, 置信度: {disk_info["confidence"]:.2f})', fontsize=11, color='white')
    else:
        ax1.set_title('Step 1: 日面边界检测 (未检测到)', fontsize=11, color='white')
    ax1.axis('off')
    fig1.patch.set_facecolor('black')
    result_images["step1_disk"] = _matplotlib_figure_to_base64(fig1)
    
    # Step 2: Disk Mask
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    disk_mask = _create_mask(h, w, disk_info)
    # Color overlay: inside disk = green, outside = red
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[disk_mask > 0] = [0, 255, 0]  # Green for disk area
    overlay[disk_mask == 0] = [255, 0, 0]  # Red for background
    
    # Blend with original
    img_rgb = np.stack([image] * 3, axis=-1)
    blended = (img_rgb * 0.6 + overlay * 0.4).astype(np.uint8)
    ax2.imshow(blended)
    
    disk_pixels = image[disk_mask > 0]
    bg_pixels = image[disk_mask == 0]
    info_text = f"日面区域: {disk_mask.sum()}px ({disk_mask.sum()/(h*w)*100:.1f}%)\n背景区域: {(h*w)-disk_mask.sum()}px ({(h*w-disk_mask.sum())/(h*w)*100:.1f}%)"
    if len(disk_pixels) > 0:
        info_text += f"\n日面平均亮度: {disk_pixels.mean():.1f}"
    if len(bg_pixels) > 0:
        info_text += f"\n背景平均亮度: {bg_pixels.mean():.1f}"
    ax2.set_title('Step 2: 日面掩模 (绿=日面, 红=背景)', fontsize=11, color='white')
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=8, color='white',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7))
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    result_images["step2_mask"] = _matplotlib_figure_to_base64(fig2)
    
    # Step 3: Threshold Detection
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    try:
        import cv2
        # Re-run thresholding for visualization
        disk_mask_cv = disk_mask.astype(np.uint8) * 255
        mean_brightness = np.mean(image[disk_mask > 0]) if disk_mask.sum() > 0 else np.mean(image)
        std_brightness = np.std(image[disk_mask > 0]) if disk_mask.sum() > 0 else np.std(image)
        thresh_val = max(1, int(mean_brightness - 1.5 * std_brightness))
        _, thresh_img = cv2.threshold(image.astype(np.uint8), thresh_val, 255, cv2.THRESH_BINARY_INV)
        # Apply disk mask
        thresh_masked = cv2.bitwise_and(thresh_img, thresh_img, mask=disk_mask_cv)
        # Draw detected spot outlines
        if sunspots:
            for s in sunspots:
                x, y, radius = int(s['x']), int(s['y']), int(s.get('radius', 10))
                cv2.circle(thresh_masked, (x, y), max(radius, 3), 128, 2)
                cv2.drawMarker(thresh_masked, (x, y), 128, cv2.MARKER_CROSS, 10, 2)
        
        show_img = cv2.cvtColor(thresh_masked, cv2.COLOR_GRAY2RGB)
        show_img[thresh_masked == 128] = [0, 255, 0]  # Green outlines for detected spots
        ax3.imshow(show_img)
        spot_text = f"阈值: <{thresh_val}\n检测到 {len(sunspots)} 个黑子候选" if sunspots else f"阈值: <{thresh_val}\n无黑子候选"
    except Exception:
        ax3.imshow(image, cmap='gray')
        spot_text = f"阈值分割失败\n已检测 {len(sunspots)} 个区域"
    
    ax3.set_title('Step 3: 反色阈值分割 (黑子检测)', fontsize=11, color='white')
    ax3.text(0.02, 0.98, spot_text, transform=ax3.transAxes, fontsize=9, color='white',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7))
    ax3.axis('off')
    fig3.patch.set_facecolor('black')
    result_images["step3_threshold"] = _matplotlib_figure_to_base64(fig3)
    
    # Step 4: Bright Region Detection
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.imshow(image, cmap='gray')
    if bright_regions:
        for br in bright_regions:
            x, y = br['x'], br['y']
            w_br, h_br = br.get('width', 20), br.get('height', 20)
            rect = mpatches.Rectangle((x - w_br/2, y - h_br/2), w_br, h_br, 
                                      fill=False, edgecolor='yellow', linewidth=2, linestyle='-')
            ax4.add_patch(rect)
            ax4.plot(x, y, 'x', color='yellow', markersize=8)
            ax4.annotate(br.get('type', ''), xy=(x, y - h_br/2 - 5), fontsize=7, color='yellow',
                        ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        br_text = f"检测到 {len(bright_regions)} 个亮区"
    else:
        br_text = "无亮区候选"
    ax4.set_title('Step 4: 亮区检测 (耀斑/谱斑)', fontsize=11, color='white')
    ax4.text(0.02, 0.98, br_text, transform=ax4.transAxes, fontsize=9, color='white',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7))
    ax4.axis('off')
    fig4.patch.set_facecolor('black')
    result_images["step4_bright"] = _matplotlib_figure_to_base64(fig4)
    
    # Step 5: Final Detection Summary
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    ax5.imshow(image, cmap='gray')
    # Draw disk boundary
    if disk_info.get("detected"):
        cx, cy, r = disk_info["center_x"], disk_info["center_y"], disk_info["radius"]
        disk_c = mpatches.Circle((cx, cy), r, fill=False, edgecolor='orange', linewidth=2, linestyle='-')
        ax5.add_patch(disk_c)
    # Draw sunspots
    for s in sunspots:
        c = mpatches.Circle((s['x'], s['y']), max(s.get('radius', 5), 5), fill=False, 
                           edgecolor='lime', linewidth=2, linestyle='--')
        ax5.add_patch(c)
        ax5.plot(s['x'], s['y'], '+', color='lime', markersize=8)
        conf = s.get('confidence', 0)
        ax5.annotate(f"黑子\n{conf:.0%}", xy=(s['x'], s['y'] + s.get('radius', 10) + 3),
                    fontsize=7, color='lime', ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    # Draw bright regions
    for br in bright_regions:
        ax5.plot(br['x'], br['y'], 'x', color='yellow', markersize=10)
        ax5.annotate(br.get('type', ''), xy=(br['x'], br['y']), fontsize=7, color='yellow',
                    ha='left', bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    
    summary = f"日面: {'✅' if disk_info.get('detected') else '❌'} | 黑子: {len(sunspots)} | 亮区: {len(bright_regions)} | 群组: {len(groups)}"
    ax5.set_title(f'Step 5: 检测结果汇总\n{summary}', fontsize=11, color='white')
    ax5.axis('off')
    fig5.patch.set_facecolor('black')
    result_images["step5_summary"] = _matplotlib_figure_to_base64(fig5)
    
    return result_images


def _create_mask(h: int, w: int, disk_info: Dict) -> np.ndarray:
    """Create a binary disk mask."""
    if disk_info.get("detected"):
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        y_grid, x_grid = np.ogrid[:h, :w]
        return ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
    else:
        cy, cx = h // 2, w // 2
        r = min(h, w) * 0.4
        y_grid, x_grid = np.ogrid[:h, :w]
        return ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2


@router.get("/cv-preview/{image_id}", tags=["CV检测预览"])
async def get_cv_preview(image_id: str):
    """Run CV preprocessing step-by-step and return all intermediate results."""
    from solar_preprocessor import (
        detect_solar_disk, segment_sunspots, detect_bright_regions,
        cluster_sunspot_groups, generate_feature_prompt
    )
    import cv2
    
    # Resolve image path using multiple methods
    image_path = _resolve_image_path(image_id)
    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    
    try:
        # Read image - handle Chinese path by reading as bytes first
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Run all CV steps
        disk_info = detect_solar_disk(gray)
        sunspots = segment_sunspots(gray, disk_info) if disk_info.get("detected") else segment_sunspots(gray, None)
        bright_regions = detect_bright_regions(gray, disk_info, sunspots)
        groups = cluster_sunspot_groups(sunspots, disk_info) if sunspots else []
        
        # Image stats
        stats = {
            "width": w, "height": h,
            "mean_brightness": float(np.mean(gray)),
            "std_brightness": float(np.std(gray)),
            "min_brightness": float(np.min(gray)),
            "max_brightness": float(np.max(gray)),
        }
        
        # Generate CV report text
        from solar_preprocessor import preprocess_solar_image
        full_result = {
            "solar_disk": disk_info,
            "sunspots": sunspots,
            "bright_regions": bright_regions,
            "sunspot_groups": groups,
            "image_stats": stats,
            "processing_info": {"image_path": image_path, "pipeline_version": "1.0"},
        }
        cv_report = generate_feature_prompt(full_result)
        
        # Generate step-by-step visualizations
        step_images = _generate_step_visualizations(gray, disk_info, sunspots, bright_regions, groups, stats)
        
        # Original image base64
        orig_b64 = _encode_image_to_base64(gray)
        
        # Build response
        return {
            "success": True,
            "data": {
                "image_id": image_id,
                "image_size": f"{w}x{h}",
                "steps": {
                    "disk_detection": {
                        "title": "日面边界检测",
                        "detected": disk_info.get("detected", False),
                        "method": disk_info.get("method", "unknown"),
                        "confidence": disk_info.get("confidence", 0),
                        "center_x": disk_info.get("center_x", 0),
                        "center_y": disk_info.get("center_y", 0),
                        "radius": disk_info.get("radius", 0),
                    },
                    "sunspot_detection": {
                        "title": "黑子检测",
                        "count": len(sunspots),
                        "details": [{
                            "id": i+1, "x": s["x"], "y": s["y"],
                            "radius": s.get("radius", 0), "area": s.get("area", 0),
                            "contrast": s.get("contrast", 0),
                            "region": s.get("region", "unknown"),
                            "confidence": s.get("confidence", 0),
                        } for i, s in enumerate(sunspots)],
                    },
                    "bright_region_detection": {
                        "title": "亮区检测",
                        "count": len(bright_regions),
                        "details": [{
                            "id": i+1, "x": br["x"], "y": br["y"],
                            "type": br.get("type", "unknown"),
                            "brightness_ratio": br.get("brightness_ratio", 0),
                            "confidence": br.get("confidence", 0),
                        } for i, br in enumerate(bright_regions)],
                    },
                    "sunspot_groups": {
                        "title": "黑子群组",
                        "count": len(groups),
                        "details": [{
                            "id": g["id"], "members": g["member_count"],
                            "complexity": g["complexity"],
                            "confidence": g.get("confidence", 0),
                        } for g in groups],
                    },
                    "image_stats": stats,
                },
                "report_text": cv_report,
                "images": {
                    "original": orig_b64,
                    **step_images,
                },
                "image_path": str(image_path),
            }
        }
    
    except Exception as e:
        logger.error(f"CV preview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CV preview failed: {str(e)}")


import os
