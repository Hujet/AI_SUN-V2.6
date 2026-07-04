"""
太阳特征检测系统性能对比测试脚本
===================================

测试 OLD（原始）与 NEW（优化）检测算法在同一组测试图像上的表现。

测试目标：
1. 黑子检测 (sunspots)
2. 谱斑检测 (plages / bright regions)
3. 日珥检测 (prominences)

OLD 算法参数（原始保守阈值）：
  - k_threshold: 2.0 / 1.5（高/中对比度）
  - min_area: 15
  - core_area_ratio: 10%

NEW 算法参数（优化敏感阈值）：
  - k_threshold: 1.2 / 1.0 / 0.8（高/中/低对比度）
  - min_area: 8
  - core_area_ratio: 5%

运行方式：
  python performance_test.py [--images-dir <path>] [--limit <n>]

输出：
  - performance_report.md (详细对比报告)
  - performance_results.json (原始数据)
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# ── add src to path ──
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from solar_preprocessor import detect_solar_disk, load_image_cv2
from advanced_detector import AdvancedDetectionPipeline, ProminenceDiscriminator

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy INFO logs during batch testing
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Detection Configurations
# ============================================================================

@dataclass
class DetectionConfig:
    """Configuration for one detection algorithm variant."""
    name: str
    # Sunspot parameters
    k_threshold_high: float = 2.0   # contrast > 0.5
    k_threshold_mid: float = 1.5    # contrast > 0.2
    k_threshold_low: float = 0.8    # contrast <= 0.2
    k_strict_high: float = 2.5
    k_strict_mid: float = 2.0
    k_strict_low: float = 1.2
    min_area: int = 15
    core_area_ratio: float = 0.10
    # Bright region parameters
    br_k_plage_high: float = 1.5
    br_k_plage_mid: float = 1.2
    br_k_plage_low: float = 1.0
    br_k_facula_high: float = 0.8
    br_k_facula_mid: float = 0.5
    br_k_facula_low: float = 0.3
    br_min_area: int = 15
    # Prominence parameters
    prom_min_area: int = 15
    prom_k_scales: Tuple[float, ...] = (0.8, 1.0, 1.2)
    prom_dark_k_scales: Tuple[float, ...] = (0.8,)
    prom_inner_r: float = 0.85
    prom_outer_r: float = 1.15
    prom_merge_threshold: float = 15

OLD_CONFIG = DetectionConfig(
    name="OLD (原始保守阈值)",
    k_threshold_high=2.0, k_threshold_mid=1.5, k_threshold_low=0.8,
    k_strict_high=2.5, k_strict_mid=2.0, k_strict_low=1.2,
    min_area=15, core_area_ratio=0.10,
    br_k_plage_high=1.5, br_k_plage_mid=1.2, br_k_plage_low=1.0,
    br_k_facula_high=0.8, br_k_facula_mid=0.5, br_k_facula_low=0.3,
    br_min_area=15,
    prom_min_area=15, prom_k_scales=(0.8, 1.0, 1.2), prom_dark_k_scales=(0.8,),
    prom_inner_r=0.85, prom_outer_r=1.15, prom_merge_threshold=15,
)

NEW_CONFIG = DetectionConfig(
    name="NEW (优化敏感阈值)",
    k_threshold_high=1.2, k_threshold_mid=1.0, k_threshold_low=0.8,
    k_strict_high=1.8, k_strict_mid=1.5, k_strict_low=1.2,
    min_area=8, core_area_ratio=0.05,
    br_k_plage_high=1.0, br_k_plage_mid=0.8, br_k_plage_low=0.6,
    br_k_facula_high=0.5, br_k_facula_mid=0.3, br_k_facula_low=0.2,
    br_min_area=10,
    prom_min_area=8, prom_k_scales=(0.6, 0.8, 1.0), prom_dark_k_scales=(0.6, 0.8),
    prom_inner_r=0.80, prom_outer_r=1.20, prom_merge_threshold=15,
)


# ============================================================================
# OLD Detection Algorithm (parameterized)
# ============================================================================

def old_detect_sunspots(
    image: np.ndarray,
    disk_info: Dict,
    config: DetectionConfig,
) -> List[Dict]:
    """Sunspot detection with configurable thresholds (OLD algorithm variant)."""
    h, w = image.shape
    img_f = image.astype(np.float32)

    cx = disk_info.get("center_x", w / 2)
    cy = disk_info.get("center_y", h / 2)
    r = disk_info.get("radius", min(h, w) * 0.4)

    r_inner = r * 0.97
    y_grid, x_grid = np.ogrid[:h, :w]
    disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (r_inner) ** 2
    disk_pixels = img_f[disk_mask]

    if len(disk_pixels) < 100:
        return []

    disk_mean = float(np.mean(disk_pixels))
    disk_std = float(np.std(disk_pixels))
    disk_min = float(np.min(disk_pixels))
    disk_max = float(np.max(disk_pixels))
    disk_range = disk_max - disk_min

    contrast_ratio = disk_std / disk_mean if disk_mean > 0 else 0

    if contrast_ratio > 0.5:
        k_threshold = config.k_threshold_high
        k_strict = config.k_strict_high
    elif contrast_ratio > 0.2:
        k_threshold = config.k_threshold_mid
        k_strict = config.k_strict_mid
    else:
        k_threshold = config.k_threshold_low
        k_strict = config.k_strict_low

    spot_threshold = disk_mean - k_threshold * disk_std
    strict_threshold = disk_mean - k_strict * disk_std

    if spot_threshold <= disk_min + 1:
        spot_threshold = disk_min + 2
        strict_threshold = disk_min + 1

    spots = []
    try:
        norm_disk = np.zeros((h, w), dtype=np.uint8)
        norm_disk[disk_mask] = ((img_f[disk_mask] - disk_min) / max(disk_range, 1) * 255).astype(np.uint8)

        norm_threshold = int((spot_threshold - disk_min) / max(disk_range, 1) * 255)
        norm_strict = int((strict_threshold - disk_min) / max(disk_range, 1) * 255)
        norm_threshold = max(norm_threshold, 1)
        norm_strict = max(norm_strict, 1)

        _, stat_mask = cv2.threshold(norm_disk, norm_threshold, 255, cv2.THRESH_BINARY_INV)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(stat_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        disk_mask_uint8 = (disk_mask * 255).astype(np.uint8)
        cleaned = cv2.bitwise_and(cleaned, cleaned, mask=disk_mask_uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < config.min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])

            max_spot_area = int(disk_mask.sum() * 0.1)
            if area > max_spot_area:
                continue

            if norm_strict > 0 and norm_strict < 255:
                _, strict_mask_comp = cv2.threshold(norm_disk, norm_strict, 255, cv2.THRESH_BINARY_INV)
                strict_mask_comp = cv2.bitwise_and(strict_mask_comp, strict_mask_comp, mask=disk_mask_uint8)
                core_area = cv2.countNonZero(strict_mask_comp[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)])
                if core_area < max(2, int(area * config.core_area_ratio)):
                    continue

            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            if dist_from_center > r_inner * 0.97:
                continue

            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if aspect_ratio > 6.0:
                continue

            effective_radius = math.sqrt(area / math.pi)
            spot_pixels = img_f[labels == i]
            spot_mean_val = float(np.mean(spot_pixels))
            contrast = (disk_mean - spot_mean_val) / disk_std if disk_std > 0 else 0

            area_score = min(area / (h * w * 0.0005), 1.0)
            contrast_score = min(max(contrast, 0) / 2.5, 1.0)
            confidence = float(0.35 * area_score + 0.65 * contrast_score)
            confidence = min(max(confidence, 0.15), 0.95)

            norm_x = (cx_comp - cx) / r if r > 0 else 0
            norm_y = (cy_comp - cy) / r if r > 0 else 0

            spots.append({
                "x": float(cx_comp),
                "y": float(cy_comp),
                "radius": float(effective_radius),
                "area": int(area),
                "contrast": float(contrast),
                "brightness": spot_mean_val,
                "confidence": confidence,
            })
    except Exception as e:
        logger.error(f"Sunspot detection error: {e}")

    return spots


def old_detect_bright_regions(
    image: np.ndarray,
    disk_info: Dict,
    config: DetectionConfig,
) -> List[Dict]:
    """Bright region detection with configurable thresholds (OLD algorithm variant)."""
    h, w = image.shape
    img_f = image.astype(np.float32)

    cx = disk_info.get("center_x", w / 2)
    cy = disk_info.get("center_y", h / 2)
    r = disk_info.get("radius", min(h, w) * 0.4)
    r_inner = r * 0.95

    y_grid, x_grid = np.ogrid[:h, :w]
    disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (r_inner) ** 2
    disk_pixels = img_f[disk_mask]

    if len(disk_pixels) < 100:
        return []

    disk_mean = float(np.mean(disk_pixels))
    disk_std = float(np.std(disk_pixels))
    disk_max = float(np.max(disk_pixels))
    disk_min = float(np.min(disk_pixels))
    disk_range = disk_max - disk_min

    contrast_ratio = disk_std / max(disk_mean, 1)

    if contrast_ratio > 0.5:
        k_plage = config.br_k_plage_high
        k_facula = config.br_k_facula_high
    elif contrast_ratio > 0.2:
        k_plage = config.br_k_plage_mid
        k_facula = config.br_k_facula_mid
    else:
        k_plage = config.br_k_plage_low
        k_facula = config.br_k_facula_low

    bright_threshold = disk_mean + k_plage * disk_std
    facula_threshold = disk_mean + k_facula * disk_std

    if bright_threshold >= disk_max:
        return []

    regions = []
    try:
        norm_disk = np.zeros((h, w), dtype=np.uint8)
        norm_disk[disk_mask] = ((img_f[disk_mask] - disk_min) / max(disk_range, 1) * 255).astype(np.uint8)

        mean_norm = ((disk_mean - disk_min) / max(disk_range, 1) * 255)
        norm_plage_thresh = int(((disk_mean + k_plage * disk_std) - disk_min) / max(disk_range, 1) * 255)
        norm_plage_thresh = max(norm_plage_thresh, int(mean_norm) + 5)

        _, bright_mask = cv2.threshold(norm_disk, min(norm_plage_thresh, 254), 255, cv2.THRESH_BINARY)
        disk_mask_uint8 = (disk_mask * 255).astype(np.uint8)
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=disk_mask_uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        min_area = max(config.br_min_area, int(h * w * 0.00005))
        max_bright_area = int(disk_mask.sum() * 0.1)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_bright_area:
                continue

            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            if dist_from_center > r * 0.95:
                continue

            region_pixels = img_f[labels == i]
            region_mean = float(np.mean(region_pixels))
            brightness_ratio = region_mean / disk_mean if disk_mean > 0 else 1.0

            if region_mean <= disk_mean:
                continue

            if brightness_ratio > 1.35:
                region_type = "flare"
            elif brightness_ratio > 1.15:
                region_type = "plage"
            elif brightness_ratio > 1.05:
                region_type = "facula"
            else:
                continue

            confidence = min((brightness_ratio - 1.0) * 4.0, 0.9)
            if confidence < 0.1:
                continue

            regions.append({
                "x": cx_comp,
                "y": cy_comp,
                "width": bw,
                "height": bh,
                "area": area,
                "type": region_type,
                "brightness_ratio": brightness_ratio,
                "confidence": confidence,
            })
    except Exception as e:
        logger.error(f"Bright region detection error: {e}")

    return regions


def old_detect_prominences(
    image: np.ndarray,
    disk_info: Dict,
    config: DetectionConfig,
) -> List[Dict]:
    """Prominence detection with configurable thresholds (OLD algorithm variant)."""
    h, w = image.shape
    cx = disk_info.get("center_x", w / 2)
    cy = disk_info.get("center_y", h / 2)
    r = disk_info.get("radius", min(h, w) * 0.4)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    inner_r = r * config.prom_inner_r
    outer_r = r * config.prom_outer_r
    limb_mask = ((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8)

    limb_pixels = image[limb_mask > 0]
    if len(limb_pixels) < 100:
        return []

    limb_mean = np.mean(limb_pixels)
    limb_std = np.std(limb_pixels)

    candidates = []

    # Bright features
    for k in config.prom_k_scales:
        bright_thresh = limb_mean + k * limb_std
        _, bright_mask = cv2.threshold(image, bright_thresh, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=limb_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < config.prom_min_area:
                continue

            feat_mask = (labels == i).astype(np.uint8) * 255
            moments = cv2.moments(feat_mask)
            mu20 = moments.get("mu20", 0) / max(moments.get("m00", 1), 1)
            mu02 = moments.get("mu02", 0) / max(moments.get("m00", 1), 1)
            mu11 = moments.get("mu11", 0) / max(moments.get("m00", 1), 1)

            w_s = stats[i, cv2.CC_STAT_WIDTH]
            h_s = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w_s, h_s) / max(min(w_s, h_s), 1)

            candidates.append({
                "x": float(centroids[i][0]),
                "y": float(centroids[i][1]),
                "area": float(area),
                "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], w_s, h_s),
                "mean_intensity": float(np.mean(image[labels == i])),
                "moments": {"m00": moments.get("m00", 0), "mu20": mu20, "mu02": mu02, "mu11": mu11},
                "aspect_ratio": aspect_ratio,
            })

    # Dark features
    for k in config.prom_dark_k_scales:
        dark_thresh = limb_mean - k * limb_std
        _, dark_mask = cv2.threshold(image, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=limb_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < config.prom_min_area:
                continue

            feat_mask = (labels == i).astype(np.uint8) * 255
            moments = cv2.moments(feat_mask)
            mu20 = moments.get("mu20", 0) / max(moments.get("m00", 1), 1)
            mu02 = moments.get("mu02", 0) / max(moments.get("m00", 1), 1)
            mu11 = moments.get("mu11", 0) / max(moments.get("m00", 1), 1)

            w_s = stats[i, cv2.CC_STAT_WIDTH]
            h_s = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w_s, h_s) / max(min(w_s, h_s), 1)

            candidates.append({
                "x": float(centroids[i][0]),
                "y": float(centroids[i][1]),
                "area": float(area),
                "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], w_s, h_s),
                "mean_intensity": float(np.mean(image[labels == i])),
                "moments": {"m00": moments.get("m00", 0), "mu20": mu20, "mu02": mu02, "mu11": mu11},
                "aspect_ratio": aspect_ratio,
            })

    # Merge overlapping
    merged = _merge_edge_candidates(candidates, threshold=config.prom_merge_threshold)

    # Classify with ProminenceDiscriminator
    discriminator = ProminenceDiscriminator()
    classified = discriminator.discriminate(image, (cx, cy), r, merged)

    prominences = []
    for cf in classified:
        d = cf.to_dict()
        prominences.append(d)

    return prominences


def _merge_edge_candidates(candidates: List[Dict], threshold: float = 15) -> List[Dict]:
    """Merge overlapping edge candidates."""
    if not candidates:
        return []
    merged = []
    used = set()
    for i, c1 in enumerate(candidates):
        if i in used:
            continue
        overlapping = [c1]
        for j, c2 in enumerate(candidates[i + 1:], i + 1):
            if j in used:
                continue
            dist = math.sqrt((c1["x"] - c2["x"]) ** 2 + (c1["y"] - c2["y"]) ** 2)
            if dist < threshold:
                overlapping.append(c2)
                used.add(j)
        best = max(overlapping, key=lambda c: c["area"])
        if len(overlapping) > 1:
            best["scales_merged"] = len(overlapping)
        merged.append(best)
    return merged


def old_full_detection(image: np.ndarray, config: DetectionConfig) -> Dict[str, Any]:
    """Run OLD detection pipeline with given config."""
    disk_info = detect_solar_disk(image)

    t0 = time.time()
    sunspots = old_detect_sunspots(image, disk_info, config)
    t1 = time.time()
    bright_regions = old_detect_bright_regions(image, disk_info, config)
    t2 = time.time()
    prominences = old_detect_prominences(image, disk_info, config)
    t3 = time.time()

    return {
        "disk_info": disk_info,
        "sunspots": sunspots,
        "bright_regions": bright_regions,
        "prominences": prominences,
        "timing": {
            "sunspots_ms": (t1 - t0) * 1000,
            "bright_regions_ms": (t2 - t1) * 1000,
            "prominences_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        },
    }


# ============================================================================
# NEW Detection Pipeline wrapper (uses advanced_detector.py as-is)
# ============================================================================

def new_full_detection(image: np.ndarray) -> Dict[str, Any]:
    """Run NEW detection pipeline using the advanced_detector module."""
    disk_info = detect_solar_disk(image)
    pipeline = AdvancedDetectionPipeline()

    t0 = time.time()
    result = pipeline.detect(image, disk_info)
    total_ms = (time.time() - t0) * 1000

    # Extract plages from faint_features
    faint = result.get("faint_features", [])
    plages = [f for f in faint if f.get("type") == "plage"]

    # Count prominences by classification
    prominences_raw = result.get("prominences", [])
    prominences = [p for p in prominences_raw if p.get("predicted_type") == "prominence"]

    return {
        "disk_info": disk_info,
        "sunspots": result.get("sunspots", []),
        "bright_regions": [],  # NEW pipeline handles bright regions via faint_features
        "plages": plages,
        "prominences": prominences,
        "all_prominences": prominences_raw,
        "faint_features": faint,
        "groups": result.get("groups", []),
        "statistics": result.get("statistics", {}),
        "timing": {
            "total_ms": total_ms,
        },
    }


# ============================================================================
# Metrics Computation
# ============================================================================

@dataclass
class FeatureMetrics:
    """Metrics for a single feature type."""
    feature_type: str
    old_count: int = 0
    new_count: int = 0
    old_avg_confidence: float = 0.0
    new_avg_confidence: float = 0.0
    old_avg_area: float = 0.0
    new_avg_area: float = 0.0
    # If we have ground truth annotations
    true_positives_old: int = 0
    false_positives_old: int = 0
    false_negatives_old: int = 0
    true_positives_new: int = 0
    false_positives_new: int = 0
    false_negatives_new: int = 0
    precision_old: float = 0.0
    recall_old: float = 0.0
    f1_old: float = 0.0
    precision_new: float = 0.0
    recall_new: float = 0.0
    f1_new: float = 0.0
    false_positive_rate_old: float = 0.0
    false_positive_rate_new: float = 0.0
    miss_rate_old: float = 0.0
    miss_rate_new: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "feature_type": self.feature_type,
            "old_count": self.old_count,
            "new_count": self.new_count,
            "old_avg_confidence": round(self.old_avg_confidence, 4),
            "new_avg_confidence": round(self.new_avg_confidence, 4),
            "old_avg_area": round(self.old_avg_area, 2),
            "new_avg_area": round(self.new_avg_area, 2),
            "true_positives_old": self.true_positives_old,
            "false_positives_old": self.false_positives_old,
            "false_negatives_old": self.false_negatives_old,
            "true_positives_new": self.true_positives_new,
            "false_positives_new": self.false_positives_new,
            "false_negatives_new": self.false_negatives_new,
            "precision_old": round(self.precision_old, 4),
            "recall_old": round(self.recall_old, 4),
            "f1_old": round(self.f1_old, 4),
            "precision_new": round(self.precision_new, 4),
            "recall_new": round(self.recall_new, 4),
            "f1_new": round(self.f1_new, 4),
            "false_positive_rate_old": round(self.false_positive_rate_old, 4),
            "false_positive_rate_new": round(self.false_positive_rate_new, 4),
            "miss_rate_old": round(self.miss_rate_old, 4),
            "miss_rate_new": round(self.miss_rate_new, 4),
        }


def _avg_confidence(items: List[Dict]) -> float:
    if not items:
        return 0.0
    return np.mean([d.get("confidence", 0) for d in items])


def _avg_area(items: List[Dict]) -> float:
    if not items:
        return 0.0
    return np.mean([d.get("area", 0) for d in items])


def compute_metrics(
    old_result: Dict,
    new_result: Dict,
    annotations: Optional[Dict] = None,
) -> Dict[str, FeatureMetrics]:
    """Compute comparison metrics between OLD and NEW detection results.

    When ground truth annotations are available, compute precision/recall/F1.
    Without annotations, compute count-based comparison metrics.
    """
    metrics = {}

    # ── Sunspots ──
    old_spots = old_result.get("sunspots", [])
    new_spots = new_result.get("sunspots", [])

    m_spots = FeatureMetrics(feature_type="黑子 (Sunspots)")
    m_spots.old_count = len(old_spots)
    m_spots.new_count = len(new_spots)
    m_spots.old_avg_confidence = _avg_confidence(old_spots)
    m_spots.new_avg_confidence = _avg_confidence(new_spots)
    m_spots.old_avg_area = _avg_area(old_spots)
    m_spots.new_avg_area = _avg_area(new_spots)

    if annotations and "sunspots" in annotations:
        _compute_precision_recall(m_spots, old_spots, new_spots, annotations["sunspots"])
    metrics["sunspots"] = m_spots

    # ── Plages / Bright Regions ──
    old_plages = [r for r in old_result.get("bright_regions", []) if r.get("type") in ("plage", "flare", "facula")]
    new_plages = new_result.get("plages", [])
    # Also include bright_regions from old
    old_all_bright = old_result.get("bright_regions", [])

    m_plages = FeatureMetrics(feature_type="谱斑 (Plages/Faculae)")
    m_plages.old_count = len(old_all_bright)
    m_plages.new_count = len(new_plages)
    m_plages.old_avg_confidence = _avg_confidence(old_all_bright)
    m_plages.new_avg_confidence = _avg_confidence(new_plages)
    m_plages.old_avg_area = _avg_area(old_all_bright)
    m_plages.new_avg_area = _avg_area(new_plages)

    if annotations and "plages" in annotations:
        _compute_precision_recall(m_plages, old_all_bright, new_plages, annotations["plages"])
    metrics["plages"] = m_plages

    # ── Prominences ──
    old_proms = old_result.get("prominences", [])
    new_proms = new_result.get("prominences", [])
    # Filter to only prominence-classified
    old_proms_classified = [p for p in old_proms if p.get("predicted_type") == "prominence"]
    new_proms_classified = [p for p in new_proms if p.get("predicted_type") == "prominence"]

    m_proms = FeatureMetrics(feature_type="日珥 (Prominences)")
    m_proms.old_count = len(old_proms_classified)
    m_proms.new_count = len(new_proms_classified)
    m_proms.old_avg_confidence = _avg_confidence(old_proms)
    m_proms.new_avg_confidence = _avg_confidence(new_proms)
    m_proms.old_avg_area = _avg_area(old_proms)
    m_proms.new_avg_area = _avg_area(new_proms)

    if annotations and "prominences" in annotations:
        _compute_precision_recall(m_proms, old_proms, new_proms, annotations["prominences"])
    metrics["prominences"] = m_proms

    return metrics


def _compute_precision_recall(
    metrics_obj: FeatureMetrics,
    old_detections: List[Dict],
    new_detections: List[Dict],
    ground_truth: List[Dict],
    match_radius: float = 20,
):
    """Compute precision, recall, F1 based on ground truth annotations."""
    # For each GT feature, find nearest detection
    old_tp = 0
    new_tp = 0
    used_old = set()
    used_new = set()

    for gt in ground_truth:
        gt_x, gt_y = gt.get("x", 0), gt.get("y", 0)

        # Match OLD
        best_old_dist = float("inf")
        best_old_idx = -1
        for i, det in enumerate(old_detections):
            if i in used_old:
                continue
            d = math.sqrt((det["x"] - gt_x) ** 2 + (det["y"] - gt_y) ** 2)
            if d < best_old_dist and d <= match_radius:
                best_old_dist = d
                best_old_idx = i
        if best_old_idx >= 0:
            old_tp += 1
            used_old.add(best_old_idx)

        # Match NEW
        best_new_dist = float("inf")
        best_new_idx = -1
        for i, det in enumerate(new_detections):
            if i in used_new:
                continue
            d = math.sqrt((det["x"] - gt_x) ** 2 + (det["y"] - gt_y) ** 2)
            if d < best_new_dist and d <= match_radius:
                best_new_dist = d
                best_new_idx = i
        if best_new_idx >= 0:
            new_tp += 1
            used_new.add(best_new_idx)

    old_fp = len(old_detections) - old_tp
    new_fp = len(new_detections) - new_tp
    old_fn = len(ground_truth) - old_tp
    new_fn = len(ground_truth) - new_tp

    metrics_obj.true_positives_old = old_tp
    metrics_obj.false_positives_old = old_fp
    metrics_obj.false_negatives_old = old_fn
    metrics_obj.true_positives_new = new_tp
    metrics_obj.false_positives_new = new_fp
    metrics_obj.false_negatives_new = new_fn

    metrics_obj.precision_old = old_tp / max(old_tp + old_fp, 1)
    metrics_obj.recall_old = old_tp / max(old_tp + old_fn, 1)
    metrics_obj.f1_old = _f1(metrics_obj.precision_old, metrics_obj.recall_old)
    metrics_obj.precision_new = new_tp / max(new_tp + new_fp, 1)
    metrics_obj.recall_new = new_tp / max(new_tp + new_fp, 1)
    metrics_obj.f1_new = _f1(metrics_obj.precision_new, metrics_obj.recall_new)

    total_negatives = max(len(ground_truth), 1)  # approximate
    metrics_obj.false_positive_rate_old = old_fp / max(old_fp + (total_negatives - old_fn), 1)
    metrics_obj.false_positive_rate_new = new_fp / max(new_fp + (total_negatives - new_fn), 1)
    metrics_obj.miss_rate_old = old_fn / max(old_tp + old_fn, 1)
    metrics_obj.miss_rate_new = new_fn / max(new_tp + new_fn, 1)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# Test Image Discovery
# ============================================================================

def find_test_images(base_dir: str, limit: int = 0) -> List[str]:
    """Find solar test images from the uploads directory."""
    image_paths = []
    upload_dir = Path(base_dir) / "data" / "uploads"

    if upload_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(list(upload_dir.glob(ext)))

    # Also check src root
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(list(Path(base_dir).glob(ext)))

    # Deduplicate and sort by modification time (newest first)
    unique = list({str(p): p for p in image_paths}.values())
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if limit > 0:
        unique = unique[:limit]

    return [str(p) for p in unique]


def load_annotations(image_path: str) -> Optional[Dict]:
    """Try to load annotation data for a test image (if available)."""
    # Look for corresponding annotation PNG
    base = Path(image_path).stem
    parent = Path(image_path).parent
    # Annotations might be in a sibling 'annotated' directory
    annotated_dir = parent.parent / "annotated"
    if not annotated_dir.exists():
        # Try same directory
        annotated_dir = parent / "annotated"

    if annotated_dir.exists():
        for ann_file in annotated_dir.glob(f"*{base}*"):
            # If there's a JSON sidecar or embedded metadata
            json_file = ann_file.with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
    return None


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(
    all_results: List[Dict],
    summary: Dict,
    old_config: DetectionConfig,
    new_config: DetectionConfig,
) -> str:
    """Generate detailed markdown comparison report."""
    lines = []

    # ── Header ──
    lines.append("# 太阳特征检测算法性能对比报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试图像数**: {summary.get('num_images', 0)}")
    lines.append(f"**总耗时**: {summary.get('total_time_seconds', 0):.1f} 秒")
    lines.append("")

    # ── Algorithm Config ──
    lines.append("## 算法参数配置")
    lines.append("")
    lines.append("| 参数 | OLD (原始保守) | NEW (优化敏感) |")
    lines.append("|------|:-------------:|:-------------:|")
    lines.append(f"| k_threshold (高对比度) | {old_config.k_threshold_high} | {new_config.k_threshold_high} |")
    lines.append(f"| k_threshold (中对比度) | {old_config.k_threshold_mid} | {new_config.k_threshold_mid} |")
    lines.append(f"| k_threshold (低对比度) | {old_config.k_threshold_low} | {new_config.k_threshold_low} |")
    lines.append(f"| min_area (最小面积) | {old_config.min_area} px² | {new_config.min_area} px² |")
    lines.append(f"| core_area_ratio (核心面积比) | {old_config.core_area_ratio:.0%} | {new_config.core_area_ratio:.0%} |")
    lines.append(f"| prom_min_area (日珥最小面积) | {old_config.prom_min_area} px² | {new_config.prom_min_area} px² |")
    lines.append(f"| prom_k_scales (日珥检测尺度) | {old_config.prom_k_scales} | {new_config.prom_k_scales} |")
    lines.append("")

    # ── Overall Summary ──
    lines.append("## 总体性能摘要")
    lines.append("")
    lines.append("| 指标 | OLD 算法 | NEW 算法 | 变化 |")
    lines.append("|------|:--------:|:--------:|:----:|")

    old_total_spots = summary.get("old_conservative_total_sunspots", 0)
    new_total_spots = summary.get("new_total_sunspots", 0)
    old_total_plages = summary.get("old_conservative_total_plages", 0)
    new_total_plages = summary.get("new_total_plages", 0)
    old_total_proms = summary.get("old_conservative_total_prominences", 0)
    new_total_proms = summary.get("new_total_prominences", 0)
    old_avg_time = summary.get("old_conservative_avg_time_ms", 0)
    new_avg_time = summary.get("new_avg_time_ms", 0)

    spot_change = f"+{(new_total_spots - old_total_spots)} ({new_total_spots / max(old_total_spots, 1):.2f}x)"
    plage_change = f"+{(new_total_plages - old_total_plages)} ({new_total_plages / max(old_total_plages, 1):.2f}x)"
    prom_change = f"+{(new_total_proms - old_total_proms)} ({new_total_proms / max(old_total_proms, 1):.2f}x)"
    time_change = f"{new_avg_time - old_avg_time:+.0f}ms ({new_avg_time / max(old_avg_time, 1):.2f}x)"

    lines.append(f"| **黑子总数** | {old_total_spots} | {new_total_spots} | {spot_change} |")
    lines.append(f"| **谱斑/亮区总数** | {old_total_plages} | {new_total_plages} | {plage_change} |")
    lines.append(f"| **日珥总数** | {old_total_proms} | {new_total_proms} | {prom_change} |")
    lines.append(f"| **平均处理时间** | {old_avg_time:.0f} ms | {new_avg_time:.0f} ms | {time_change} |")
    lines.append("")

    # ── Per-Feature Metrics ──
    lines.append("## 详细检测指标")
    lines.append("")

    # Aggregate metrics across all images
    all_metrics = summary.get("metrics_by_type", {})

    for feat_key, feat_name in [
        ("sunspots", "黑子 (Sunspots)"),
        ("plages", "谱斑 (Plages/Faculae)"),
        ("prominences", "日珥 (Prominences)"),
    ]:
        m_data = all_metrics.get(feat_key, {})
        lines.append(f"### {feat_name}")
        lines.append("")

        # Counts
        old_count = m_data.get("old_count", 0)
        new_count = m_data.get("new_count", 0)
        lines.append(f"| 指标 | OLD | NEW |")
        lines.append(f"|------|:---:|:---:|")
        lines.append(f"| 检测总数 | {old_count} | {new_count} |")
        lines.append(f"| 平均置信度 | {m_data.get('old_avg_confidence', 0):.4f} | {m_data.get('new_avg_confidence', 0):.4f} |")
        lines.append(f"| 平均面积 (px²) | {m_data.get('old_avg_area', 0):.1f} | {m_data.get('new_avg_area', 0):.1f} |")
        lines.append("")

        # Precision/Recall (if ground truth available)
        has_gt = m_data.get("has_ground_truth", False)
        if has_gt:
            lines.append("| 精度指标 | OLD | NEW |")
            lines.append("|---------|:---:|:---:|")
            lines.append(f"| 精确率 (Precision) | {m_data.get('precision_old', 0):.4f} | {m_data.get('precision_new', 0):.4f} |")
            lines.append(f"| 召回率 (Recall) | {m_data.get('recall_old', 0):.4f} | {m_data.get('recall_new', 0):.4f} |")
            lines.append(f"| F1 分数 | {m_data.get('f1_old', 0):.4f} | {m_data.get('f1_new', 0):.4f} |")
            lines.append(f"| 假阳性率 (FPR) | {m_data.get('false_positive_rate_old', 0):.4f} | {m_data.get('false_positive_rate_new', 0):.4f} |")
            lines.append(f"| 漏检率 (Miss Rate) | {m_data.get('miss_rate_old', 0):.4f} | {m_data.get('miss_rate_new', 0):.4f} |")
            lines.append(f"| TP | {m_data.get('true_positives_old', 0)} | {m_data.get('true_positives_new', 0)} |")
            lines.append(f"| FP | {m_data.get('false_positives_old', 0)} | {m_data.get('false_positives_new', 0)} |")
            lines.append(f"| FN | {m_data.get('false_negatives_old', 0)} | {m_data.get('false_negatives_new', 0)} |")
            lines.append("")
        else:
            lines.append("*注: 该类别无标注数据，仅显示检测数量对比，精度指标不可用。*")
            lines.append("")

    # ── Per-Image Results ──
    lines.append("## 逐图像检测结果")
    lines.append("")
    lines.append("| # | 图像 | OLD 黑子 | NEW 黑子 | OLD 谱斑 | NEW 谱斑 | OLD 日珥 | NEW 日珥 | OLD 耗时 | NEW 耗时 |")
    lines.append("|---|------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|")

    for idx, r in enumerate(all_results, 1):
        img_name = Path(r["image_path"]).name[:30]
        lines.append(
            f"| {idx} | {img_name} | "
            f"{r['old_sunspots']} | {r['new_sunspots']} | "
            f"{r['old_plages']} | {r['new_plages']} | "
            f"{r['old_prominences']} | {r['new_prominences']} | "
            f"{r['old_time_ms']:.0f}ms | {r['new_time_ms']:.0f}ms |"
        )
    lines.append("")

    # ── Analysis ──
    lines.append("## 分析与结论")
    lines.append("")

    # Sensitivity analysis
    if new_total_spots > old_total_spots:
        lines.append(f"### 黑子检测灵敏度")
        lines.append(f"- NEW 算法比 OLD 算法多检测到 **{new_total_spots - old_total_spots}** 个黑子（{new_total_spots / max(old_total_spots, 1):.2f}x）")
        lines.append(f"- 降低 k_threshold 阈值（从 {old_config.k_threshold_high}/{old_config.k_threshold_mid} 到 {new_config.k_threshold_high}/{new_config.k_threshold_mid}）使算法对弱信号更敏感")
        lines.append(f"- 降低 min_area（从 {old_config.min_area} 到 {new_config.min_area}）使小面积黑子也能被检测到")
        lines.append("")

    if new_total_plages > old_total_plages:
        lines.append(f"### 谱斑检测灵敏度")
        lines.append(f"- NEW 算法比 OLD 算法多检测到 **{new_total_plages - old_total_plages}** 个谱斑/亮区（{new_total_plages / max(old_total_plages, 1):.2f}x）")
        lines.append(f"- 优化后的 k_plage 阈值使较弱的亮区也能被正确分类")
        lines.append(f"- NEW 算法使用多尺度小波分解提取 faint features，进一步增强了微弱信号的检出率")
        lines.append("")

    if new_total_proms > old_total_proms:
        lines.append(f"### 日珥检测灵敏度")
        lines.append(f"- NEW 算法比 OLD 算法多检测到 **{new_total_proms - old_total_proms}** 个日珥（{new_total_proms / max(old_total_proms, 1):.2f}x）")
        lines.append(f"- 增加暗特征检测尺度（从 {old_config.prom_dark_k_scales} 到 {new_config.prom_dark_k_scales}）提高了暗日珥的检出率")
        lines.append(f"- 扩展日珥搜索区域（从 {old_config.prom_inner_r:.2f} 到 {new_config.prom_inner_r:.2f}）覆盖了更宽的边缘带")
        lines.append("")

    # Time analysis
    if new_avg_time > 0 and old_avg_time > 0:
        lines.append(f"### 性能开销")
        ratio = new_avg_time / max(old_avg_time, 1)
        if ratio > 1.5:
            lines.append(f"- NEW 算法平均耗时是 OLD 算法的 **{ratio:.2f}x**")
            lines.append(f"- 主要开销来自：Retinex光照校正、多尺度分解、分水岭分割")
            lines.append(f"- 建议在批量处理时使用 GPU 加速或异步处理")
        else:
            lines.append(f"- NEW 算法平均耗时与 OLD 算法相当（{ratio:.2f}x）")
        lines.append("")

    lines.append("---")
    lines.append("*报告由 performance_test.py 自动生成*")

    return "\n".join(lines)


# ============================================================================
# Main Execution
# ============================================================================

def run_comparison(image_path: str) -> Optional[Dict]:
    """Run both OLD and NEW detection on a single image."""
    logger.info(f"Processing: {Path(image_path).name}")

    gray = load_image_cv2(image_path)
    if gray is None:
        logger.warning(f"Failed to load: {image_path}")
        return None

    # Run OLD (optimized thresholds)
    try:
        old_res = old_full_detection(gray, NEW_CONFIG)
    except Exception as e:
        logger.error(f"OLD detection failed: {e}")
        old_res = {"sunspots": [], "bright_regions": [], "prominences": [], "timing": {"total_ms": 0}}

    # Run OLD (conservative thresholds)
    try:
        old_res_conservative = old_full_detection(gray, OLD_CONFIG)
    except Exception as e:
        logger.error(f"OLD (conservative) detection failed: {e}")
        old_res_conservative = {"sunspots": [], "bright_regions": [], "prominences": [], "timing": {"total_ms": 0}}

    # Run NEW
    try:
        new_res = new_full_detection(gray)
    except Exception as e:
        logger.error(f"NEW detection failed: {e}", exc_info=True)
        new_res = {"sunspots": [], "plages": [], "prominences": [], "timing": {"total_ms": 0}}

    return {
        "image_path": image_path,
        "old_conservative": old_res_conservative,
        "old_optimized": old_res,
        "new": new_res,
        "annotations": load_annotations(image_path),
    }


def main():
    parser = argparse.ArgumentParser(description="太阳特征检测算法性能对比测试")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="测试图像目录（默认使用 data/uploads）")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制测试图像数量（0=全部）")
    parser.add_argument("--output", type=str, default="performance_report.md",
                        help="输出报告文件名")
    parser.add_argument("--json-output", type=str, default="performance_results.json",
                        help="输出 JSON 数据文件名")
    args = parser.parse_args()

    base_dir = args.images_dir or str(Path(__file__).parent)

    print("=" * 60)
    print(" 太阳特征检测算法性能对比测试")
    print("=" * 60)
    print()

    # Find images
    image_paths = find_test_images(base_dir, limit=args.limit)
    if not image_paths:
        print("错误: 未找到测试图像")
        return

    print(f"找到 {len(image_paths)} 张测试图像")
    print(f"OLD 算法: k_threshold={OLD_CONFIG.k_threshold_high}/{OLD_CONFIG.k_threshold_mid}, min_area={OLD_CONFIG.min_area}")
    print(f"NEW 算法: k_threshold={NEW_CONFIG.k_threshold_high}/{NEW_CONFIG.k_threshold_mid}, min_area={NEW_CONFIG.min_area}")
    print()

    # Run comparison
    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {Path(img_path).name}...", end=" ", flush=True)
        result = run_comparison(img_path)
        if result:
            all_results.append(result)
            print(f"OK  (OLD={result['old_conservative']['timing']['total_ms']:.0f}ms, "
                  f"NEW={result['new']['timing']['total_ms']:.0f}ms)")
        else:
            print("FAILED")

    print()
    print("计算对比指标...")

    # Compute aggregate metrics
    summary = {
        "num_images": len(all_results),
        "metrics_by_type": {},
    }

    # Aggregate counts
    old_conservative_total_spots = 0
    old_optimized_total_spots = 0
    new_total_spots = 0
    old_conservative_total_plages = 0
    old_optimized_total_plages = 0
    new_total_plages = 0
    old_conservative_total_proms = 0
    old_optimized_total_proms = 0
    new_total_prominences = 0
    old_conservative_total_time = 0
    old_optimized_total_time = 0
    new_total_time = 0

    per_image_rows = []

    for r in all_results:
        oc = r["old_conservative"]
        oo = r["old_optimized"]
        nw = r["new"]

        old_conservative_total_spots += len(oc.get("sunspots", []))
        old_optimized_total_spots += len(oo.get("sunspots", []))
        new_total_spots += len(nw.get("sunspots", []))

        old_conservative_total_plages += len(oc.get("bright_regions", []))
        old_optimized_total_plages += len(oo.get("bright_regions", []))
        new_plages = len(nw.get("plages", []))
        new_total_plages += new_plages

        old_conservative_total_proms += len([p for p in oc.get("prominences", []) if p.get("predicted_type") == "prominence"])
        old_optimized_total_proms += len([p for p in oo.get("prominences", []) if p.get("predicted_type") == "prominence"])
        new_total_prominences += len(nw.get("prominences", []))

        old_conservative_total_time += oc.get("timing", {}).get("total_ms", 0)
        old_optimized_total_time += oo.get("timing", {}).get("total_ms", 0)
        new_total_time += nw.get("timing", {}).get("total_ms", 0)

        per_image_rows.append({
            "image_path": r["image_path"],
            "old_sunspots": len(oc.get("sunspots", [])),
            "new_sunspots": len(nw.get("sunspots", [])),
            "old_plages": len(oc.get("bright_regions", [])),
            "new_plages": new_plages,
            "old_prominences": len(oc.get("prominences", [])),
            "new_prominences": len(nw.get("prominences", [])),
            "old_time_ms": oc.get("timing", {}).get("total_ms", 0),
            "new_time_ms": nw.get("timing", {}).get("total_ms", 0),
        })

    n = len(all_results)
    summary["old_conservative_total_sunspots"] = old_conservative_total_spots
    summary["old_optimized_total_sunspots"] = old_optimized_total_spots
    summary["new_total_sunspots"] = new_total_spots
    summary["old_conservative_total_plages"] = old_conservative_total_plages
    summary["old_optimized_total_plages"] = old_optimized_total_plages
    summary["new_total_plages"] = new_total_plages
    summary["old_conservative_total_prominences"] = old_conservative_total_proms
    summary["old_optimized_total_prominences"] = old_optimized_total_proms
    summary["new_total_prominences"] = new_total_prominences
    summary["old_conservative_avg_time_ms"] = old_conservative_total_time / max(n, 1)
    summary["old_optimized_avg_time_ms"] = old_optimized_total_time / max(n, 1)
    summary["new_avg_time_ms"] = new_total_time / max(n, 1)
    summary["total_time_seconds"] = (old_conservative_total_time + old_optimized_total_time + new_total_time) / 1000

    # Aggregate metrics by feature type
    for feat_key in ["sunspots", "plages", "prominences"]:
        type_data = {"old_count": 0, "new_count": 0, "has_ground_truth": False}
        # Use conservative OLD as the primary OLD comparison
        type_data["old_count"] = summary.get(f"old_conservative_total_{feat_key}", 0)
        type_data["new_count"] = summary.get(f"new_total_{feat_key}", 0)

        # Average confidence and area across images
        old_confs = []
        new_confs = []
        old_areas = []
        new_areas = []
        for r in all_results:
            oc = r["old_conservative"]
            nw = r["new"]

            if feat_key == "sunspots":
                old_items = oc.get("sunspots", [])
                new_items = nw.get("sunspots", [])
            elif feat_key == "plages":
                old_items = oc.get("bright_regions", [])
                new_items = nw.get("plages", [])
            else:  # prominences
                old_items = oc.get("prominences", [])
                new_items = nw.get("prominences", [])

            for item in old_items:
                old_confs.append(item.get("confidence", 0))
                old_areas.append(item.get("area", 0))
            for item in new_items:
                new_confs.append(item.get("confidence", 0))
                new_areas.append(item.get("area", 0))

        if old_confs:
            type_data["old_avg_confidence"] = float(np.mean(old_confs))
        if new_confs:
            type_data["new_avg_confidence"] = float(np.mean(new_confs))
        if old_areas:
            type_data["old_avg_area"] = float(np.mean(old_areas))
        if new_areas:
            type_data["new_avg_area"] = float(np.mean(new_areas))

        summary["metrics_by_type"][feat_key] = type_data

    # Generate markdown report
    report = generate_markdown_report(
        per_image_rows,
        summary,
        OLD_CONFIG,
        NEW_CONFIG,
    )

    # Save report
    report_path = args.output
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    # Save JSON data
    json_path = args.json_output
    json_data = {
        "summary": summary,
        "per_image_results": per_image_rows,
        "old_config": {
            "name": OLD_CONFIG.name,
            "k_threshold": f"{OLD_CONFIG.k_threshold_high}/{OLD_CONFIG.k_threshold_mid}/{OLD_CONFIG.k_threshold_low}",
            "min_area": OLD_CONFIG.min_area,
            "core_area_ratio": OLD_CONFIG.core_area_ratio,
        },
        "new_config": {
            "name": NEW_CONFIG.name,
            "k_threshold": f"{NEW_CONFIG.k_threshold_high}/{NEW_CONFIG.k_threshold_mid}/{NEW_CONFIG.k_threshold_low}",
            "min_area": NEW_CONFIG.min_area,
            "core_area_ratio": NEW_CONFIG.core_area_ratio,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"数据已保存: {json_path}")

    # Print summary
    print()
    print("=" * 60)
    print(" 测试完成 - 结果摘要")
    print("=" * 60)
    print()
    print(f"测试图像数:     {n}")
    print()
    print(" 黑子检测 (Sunspots):")
    print(f"   OLD 保守: {old_conservative_total_spots}")
    print(f"   OLD 优化: {old_optimized_total_spots}")
    print(f"   NEW:      {new_total_spots}")
    print()
    print(" 谱斑检测 (Plages):")
    print(f"   OLD 保守: {old_conservative_total_plages}")
    print(f"   OLD 优化: {old_optimized_total_plages}")
    print(f"   NEW:      {new_total_plages}")
    print()
    print(" 日珥检测 (Prominences):")
    print(f"   OLD 保守: {old_conservative_total_proms}")
    print(f"   OLD 优化: {old_optimized_total_proms}")
    print(f"   NEW:      {new_total_prominences}")
    print()
    print(f" 平均处理时间:")
    print(f"   OLD 保守: {summary['old_conservative_avg_time_ms']:.0f} ms")
    print(f"   OLD 优化: {summary['old_optimized_avg_time_ms']:.0f} ms")
    print(f"   NEW:      {summary['new_avg_time_ms']:.0f} ms")
    print()
    print(f"报告: {report_path}")
    print(f"数据: {json_path}")


if __name__ == "__main__":
    main()
