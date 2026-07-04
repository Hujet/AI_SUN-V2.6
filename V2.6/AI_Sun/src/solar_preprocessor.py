"""
Solar Image Preprocessing Pipeline - v2.5 (Release)

Provides local computer vision based preprocessing for solar images:
1. Solar disk detection (multi-method ensemble with confidence scoring)
2. Disk mask generation (binary mask for feature extraction)
3. Sunspot segmentation (robust statistical thresholding + morphological ops)
4. Flare/plage detection (multi-scale brightness anomaly analysis)
5. Filament detection (dark elongated structures on disk surface)
6. Prominence detection (limb features, both bright and dark)
7. Feature clustering (DBSCAN for sunspot groups)
8. Coordinate normalization relative to solar disk

Key improvements:
- On-disk filament detection (separate from limb prominences)
- Multi-scale plage detection for large-scale features
- Parallelized independent detection modules
- Optimized NumPy operations and reduced redundant computations
"""

import logging
import math
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Solar Disk Detection (Enhanced Multi-Method)
# ============================================================

def detect_solar_disk(image: np.ndarray) -> Dict[str, Any]:
    """Detect the solar disk boundary using multi-method ensemble.

    Uses multiple detection strategies and selects the best result based on confidence:
    1. Hough Circle transform (edge-based)
    2. Radial brightness profile (threshold-based)
    3. Contour detection (shape-based)
    4. Bright region analysis (fallback)

    Args:
        image: Grayscale solar image (H, W)

    Returns:
        Dict with keys:
            - detected: bool
            - center_x: float (pixel)
            - center_y: float (pixel)
            - radius: float (pixel)
            - method: str ("hough", "radial", "contour", "brightness")
            - confidence: float (0-1)
            - methods_tried: list of methods attempted
    """
    h, w = image.shape
    methods_tried = []

    # Method 1: Hough Circle transform
    result_hough = _detect_hough_circle(image)
    methods_tried.append("hough")
    if result_hough["detected"]:
        logger.info(f"Hough Circle: center=({result_hough['center_x']:.1f},{result_hough['center_y']:.1f}), r={result_hough['radius']:.1f}, conf={result_hough['confidence']:.2f}")

    # Method 2: Radial brightness profile
    result_radial = _detect_radial_profile(image)
    methods_tried.append("radial")
    if result_radial["detected"]:
        logger.info(f"Radial Profile: center=({result_radial['center_x']:.1f},{result_radial['center_y']:.1f}), r={result_radial['radius']:.1f}, conf={result_radial['confidence']:.2f}")

    # Method 3: Contour detection
    result_contour = _detect_contour(image)
    methods_tried.append("contour")
    if result_contour["detected"]:
        logger.info(f"Contour: center=({result_contour['center_x']:.1f},{result_contour['center_y']:.1f}), r={result_contour['radius']:.1f}, conf={result_contour['confidence']:.2f}")

    # Method 4: Bright region analysis (fallback)
    result_brightness = _detect_brightness_region(image)
    methods_tried.append("brightness")
    if result_brightness["detected"]:
        logger.info(f"Brightness: center=({result_brightness['center_x']:.1f},{result_brightness['center_y']:.1f}), r={result_brightness['radius']:.1f}, conf={result_brightness['confidence']:.2f}")

    # Select best result based on confidence
    candidates = [result_hough, result_radial, result_contour, result_brightness]
    detected = [c for c in candidates if c["detected"]]
    if detected:
        best = max(detected, key=lambda x: x["confidence"])
        best["methods_tried"] = methods_tried
        logger.info(f"Selected method: {best['method']} (confidence: {best['confidence']:.2f})")
        return best

    # Fallback
    fallback = {
        "detected": True,
        "center_x": float(w / 2),
        "center_y": float(h / 2),
        "radius": float(min(h, w) * 0.4),
        "method": "fallback",
        "confidence": 0.3,
        "methods_tried": methods_tried,
    }
    logger.info(f"Using fallback: center=({fallback['center_x']:.1f},{fallback['center_y']:.1f}), r={fallback['radius']:.1f}")
    return fallback


def _detect_hough_circle(image: np.ndarray) -> Dict[str, Any]:
    """Detect solar disk using Hough Circle transform."""
    h, w = image.shape
    result = {"detected": False, "center_x": w / 2, "center_y": h / 2, "radius": min(h, w) * 0.4, "method": "hough", "confidence": 0.0}
    try:
        import cv2
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        param_sets = [
            {"dp": 1, "minDist": min(h, w) * 0.3, "param1": 50, "param2": min(h, w) * 0.1, "minR": 0.2, "maxR": 0.5},
            {"dp": 1, "minDist": min(h, w) * 0.2, "param1": 80, "param2": min(h, w) * 0.05, "minR": 0.15, "maxR": 0.6},
            {"dp": 1.2, "minDist": min(h, w) * 0.3, "param1": 100, "param2": min(h, w) * 0.08, "minR": 0.2, "maxR": 0.7},
        ]
        best_circle = None
        best_confidence = 0
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=params["dp"], minDist=params["minDist"],
                param1=params["param1"], param2=params["param2"],
                minRadius=int(min(h, w) * params["minR"]),
                maxRadius=int(min(h, w) * params["maxR"])
            )
            if circles is not None and len(circles) > 0:
                circle = circles[0][0]
                cx, cy, r = circle
                coverage = (math.pi * r * r) / (h * w)
                if 0.15 < coverage < 0.85:
                    confidence = min(coverage * 2, 1.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_circle = (float(cx), float(cy), float(r), confidence)
        if best_circle:
            cx, cy, r, conf = best_circle
            result.update({"detected": True, "center_x": cx, "center_y": cy, "radius": r, "confidence": conf})
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Hough Circle detection failed: {e}")
    return result


def _detect_radial_profile(image: np.ndarray) -> Dict[str, Any]:
    """Detect solar disk using radial brightness profile analysis."""
    h, w = image.shape
    result = {"detected": False, "center_x": w / 2, "center_y": h / 2, "radius": min(h, w) * 0.4, "method": "radial", "confidence": 0.0}
    try:
        cy, cx = h // 2, w // 2
        max_r = min(h, w) // 2 - 10
        angles = np.linspace(0, 2 * math.pi, 72)
        radii = []
        valid_angles = 0
        center_brightness = image[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)].mean()
        threshold = center_brightness * 0.3
        for angle in angles:
            dx, dy = math.cos(angle), math.sin(angle)
            for r in range(10, max_r):
                x = int(cx + r * dx)
                y = int(cy + r * dy)
                if 0 <= x < w and 0 <= y < h:
                    if image[y, x] < threshold:
                        radii.append(r)
                        valid_angles += 1
                        break
                else:
                    break
        if radii and valid_angles > 18:
            avg_r = np.mean(radii)
            std_r = np.std(radii)
            circularity = 1.0 - min(std_r / avg_r, 1.0) if avg_r > 0 else 0
            angle_coverage = valid_angles / len(angles)
            confidence = circularity * 0.7 + angle_coverage * 0.3
            result.update({"detected": True, "center_x": float(cx), "center_y": float(cy), "radius": float(avg_r), "confidence": float(confidence)})
    except Exception as e:
        logger.debug(f"Radial profile detection failed: {e}")
    return result


def _detect_contour(image: np.ndarray) -> Dict[str, Any]:
    """Detect solar disk using contour analysis."""
    h, w = image.shape
    result = {"detected": False, "center_x": w / 2, "center_y": h / 2, "radius": min(h, w) * 0.4, "method": "contour", "confidence": 0.0}
    try:
        import cv2
        _, thresh = cv2.threshold(image, image.mean() * 0.5, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            coverage = area / (h * w)
            if 0.1 < coverage < 0.9:
                (cx, cy), radius = cv2.minEnclosingCircle(largest)
                contour_perimeter = cv2.arcLength(largest, True)
                circularity = (4 * math.pi * area) / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
                confidence = min(coverage * 1.5, 1.0) * (0.5 + 0.5 * circularity)
                result.update({"detected": True, "center_x": float(cx), "center_y": float(cy), "radius": float(radius), "confidence": float(confidence)})
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Contour detection failed: {e}")
    return result


def _detect_brightness_region(image: np.ndarray) -> Dict[str, Any]:
    """Detect solar disk using brightness region analysis."""
    h, w = image.shape
    result = {"detected": False, "center_x": w / 2, "center_y": h / 2, "radius": min(h, w) * 0.4, "method": "brightness", "confidence": 0.0}
    try:
        brightness_mask = image > image.mean() * 0.5
        y_coords, x_coords = np.where(brightness_mask)
        if len(y_coords) > 100:
            estimated_cx = float(np.mean(x_coords))
            estimated_cy = float(np.mean(y_coords))
            max_extent = max(np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)) / 2
            estimated_r = float(max_extent * 0.9)
            distances = np.sqrt((x_coords - estimated_cx) ** 2 + (y_coords - estimated_cy) ** 2)
            r_std = np.std(distances) / estimated_r if estimated_r > 0 else 1.0
            confidence = max(0.3, 1.0 - r_std)
            result.update({"detected": True, "center_x": estimated_cx, "center_y": estimated_cy, "radius": estimated_r, "confidence": float(confidence)})
    except Exception as e:
        logger.debug(f"Brightness region detection failed: {e}")
    return result


def create_disk_mask(image_shape: Tuple[int, int], disk_info: Dict[str, Any], margin: float = 0.0) -> np.ndarray:
    """Create a binary mask for the solar disk."""
    h, w = image_shape
    cx = disk_info["center_x"]
    cy = disk_info["center_y"]
    r = disk_info["radius"] * (1.0 + margin)
    y_grid, x_grid = np.ogrid[:h, :w]
    mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
    return mask.astype(np.uint8)


# ============================================================
# Sunspot Segmentation (Robust Statistical Method)
# ============================================================

def segment_sunspots(image: np.ndarray, disk_info: Optional[Dict] = None) -> List[Dict]:
    """Detect sunspot candidates using a robust statistical approach.

    Algorithm:
    1. Create disk mask (only analyze pixels within the solar disk)
    2. Compute disk statistics (mean, std, median, min, max)
    3. Set adaptive threshold based on image contrast
    4. Apply statistical thresholding (pixels below mean - k*std)
    5. Clean noise with morphological opening
    6. Connected component analysis
    7. Validation: dark core requirement, aspect ratio, edge exclusion
    """
    h, w = image.shape
    img_f = image.astype(np.float32)
    spots = []

    # Create disk mask
    if disk_info and disk_info.get("detected"):
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        r_inner = r * 0.97  # Slightly larger to catch edge spots
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (r_inner) ** 2
        disk_pixels = img_f[disk_mask]
    else:
        cy, cx = h // 2, w // 2
        r = min(h, w) * 0.45
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
        disk_pixels = img_f[disk_mask]

    if len(disk_pixels) < 100:
        logger.warning("Too few disk pixels for reliable detection")
        return []

    # Disk statistics
    disk_mean = float(np.mean(disk_pixels))
    disk_std = float(np.std(disk_pixels))
    disk_median = float(np.median(disk_pixels))
    disk_min = float(np.min(disk_pixels))
    disk_max = float(np.max(disk_pixels))
    disk_range = disk_max - disk_min

    logger.info(f"Disk stats: mean={disk_mean:.1f}, std={disk_std:.1f}, median={disk_median:.1f}, range=[{disk_min:.1f}, {disk_max:.1f}]")

    # Adaptive threshold based on contrast - USE PERCENTILE METHOD for robustness
    contrast_ratio = disk_std / disk_mean if disk_mean > 0 else 0
    
    # For white-light solar photos: black spots are very obvious, use percentile-based threshold
    # This is more robust than mean-std method for low-contrast images
    
    # Use a combination of percentile and mean-std methods
    if contrast_ratio > 0.5:
        # High contrast: use moderate threshold
        k_threshold = 0.4
        k_strict = 0.8
        spot_percentile = np.percentile(disk_pixels, 10)
        spot_threshold = min(spot_percentile, disk_mean - k_threshold * disk_std)
        strict_threshold = disk_mean - k_strict * disk_std
    elif contrast_ratio > 0.2:
        # Medium contrast: lower threshold
        k_threshold = 0.2
        k_strict = 0.5
        spot_percentile = np.percentile(disk_pixels, 8)
        spot_threshold = min(spot_percentile, disk_mean - k_threshold * disk_std)
        strict_threshold = disk_mean - k_strict * disk_std
    else:
        # Low contrast: use percentile as primary threshold (more robust)
        k_threshold = 0.1
        k_strict = 0.25
        # For low contrast images, use 25th percentile to capture more dark regions
        spot_percentile = np.percentile(disk_pixels, 25)
        spot_threshold = spot_percentile  # Use 25th percentile directly
        strict_threshold = np.percentile(disk_pixels, 5)  # Very dark core (5th percentile)
    
    # 低对比度图像直接使用百分位数作为阈值（更稳健）
    # 高/中对比度图像取百分位数和mean-std方法中较低的值
    
    logger.info(f"Contrast={contrast_ratio:.3f}, k={k_threshold}, spot_threshold={spot_threshold:.1f}, strict={strict_threshold:.1f}")

    # If threshold is below the minimum disk brightness, clamp it
    if spot_threshold <= disk_min + 1:
        spot_threshold = disk_min + 2
        strict_threshold = disk_min + 1
        logger.info(f"Clamped thresholds: spot={spot_threshold:.1f}, strict={strict_threshold:.1f}")

    try:
        import cv2

        # CRITICAL FIX: Use ORIGINAL pixel values directly, NOT normalized values
        # Normalization distorts thresholds for images with limb darkening
        # Apply threshold directly on original float image
        
        logger.info(f"Applying threshold: spot_threshold={spot_threshold:.1f}, strict={strict_threshold:.1f}")

        # Step 1: Threshold on ORIGINAL image - pixels below spot_threshold
        # THRESH_BINARY_INV: pixels < threshold become 255 (white = candidate), others become 0
        # SAFETY: clamp to 0-255 range before uint8 conversion to avoid truncation
        img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
        _, stat_mask = cv2.threshold(img_u8, int(spot_threshold), 255, cv2.THRESH_BINARY_INV)

        # Step 2: NO morphological opening - preserve ALL small spots
        # Only dilate slightly to connect very tiny gaps
        cleaned = stat_mask  # Keep all thresholded pixels

        # Step 3: Apply disk mask to exclude background
        disk_mask_uint8 = (disk_mask * 255).astype(np.uint8)
        cleaned = cv2.bitwise_and(cleaned, cleaned, mask=disk_mask_uint8)

        # Step 4: Pre-compute strict mask ONCE for dark-core validation (performance optimization)
        strict_mask_global = None
        if strict_threshold > disk_min:
            _, strict_mask_global = cv2.threshold(img_u8, int(strict_threshold), 255, cv2.THRESH_BINARY_INV)
            strict_mask_global = cv2.bitwise_and(strict_mask_global, strict_mask_global, mask=disk_mask_uint8)

        # Step 5: Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        logger.info(f"Connected components: {num_labels - 1} candidates (excluding background)")

        # Validate candidates with relaxed criteria
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])

            # Filter 1: Minimum area - LOWERED for better detection
            min_area = max(5, int(h * w * 0.00001))  # Reduced from 8 and 0.00002
            if area < min_area:
                continue

            # Filter 2: Maximum area (reject large artifacts)
            max_spot_area = int(disk_mask.sum() * 0.1)
            if area > max_spot_area:
                continue

            # Filter 3: Must have a dark core (MINIMAL requirement - just 1%)
            # Use pre-computed global strict mask (moved outside loop for performance)
            if strict_mask_global is not None:
                roi_mask_y1, roi_mask_y2 = max(0, y), min(h, y + bh)
                roi_mask_x1, roi_mask_x2 = max(0, x), min(w, x + bw)
                roi_labels = labels[roi_mask_y1:roi_mask_y2, roi_mask_x1:roi_mask_x2]
                roi_strict = strict_mask_global[roi_mask_y1:roi_mask_y2, roi_mask_x1:roi_mask_x2]
                core_mask = (roi_labels == i) & (roi_strict > 0)
                core_area = int(np.count_nonzero(core_mask))
                if core_area < max(1, area * 0.01):  # RELAXED: only 1% needs to be very dark
                    continue

            # Filter 4: Must be within the disk (relaxed edge exclusion)
            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            if dist_from_center > r_inner * 0.97:
                continue

            # Filter 5: Aspect ratio check (relaxed for irregular sunspots)
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if aspect_ratio > 6.0:
                continue

            # Calculate effective radius
            effective_radius = math.sqrt(area / math.pi)

            # Calculate contrast
            spot_pixels = img_f[labels == i]
            spot_mean_val = float(np.mean(spot_pixels))
            contrast = (disk_mean - spot_mean_val) / disk_std if disk_std > 0 else 0

            # Confidence scoring - adjusted for lower thresholds
            area_score = min(area / (h * w * 0.0005), 1.0)
            contrast_score = min(max(contrast, 0) / 2.5, 1.0)
            confidence = float(0.35 * area_score + 0.65 * contrast_score)
            confidence = min(max(confidence, 0.15), 0.95)

            # Normalized coordinates
            norm_x = (cx_comp - cx) / r if r > 0 else 0
            norm_y = (cy_comp - cy) / r if r > 0 else 0

            spots.append({
                "x": float(cx_comp),
                "y": float(cy_comp),
                "radius": float(effective_radius),
                "area": int(area),
                "bbox_width": int(bw),
                "bbox_height": int(bh),
                "contrast": float(contrast),
                "brightness": spot_mean_val,
                "region": _classify_solar_region(norm_x, norm_y),
                "confidence": confidence,
            })
            logger.info(f"  Spot #{len(spots)}: center=({cx_comp:.1f},{cy_comp:.1f}), area={area}, contrast={contrast:.2f}, conf={confidence:.2f}")

    except ImportError:
        logger.warning("OpenCV not available for sunspot detection")
    except Exception as e:
        logger.error(f"Sunspot detection failed: {e}", exc_info=True)

    logger.info(f"Final sunspot count: {len(spots)}")
    return spots


def _classify_solar_region(norm_x: float, norm_y: float) -> str:
    """Classify the position of a feature on the solar disk."""
    r = math.sqrt(norm_x ** 2 + norm_y ** 2)
    if r > 0.8:
        return "edge"
    elif r > 0.5:
        return "mid"
    else:
        return "center"


# ============================================================
# Bright Region Detection (Flare/Plage)
# ============================================================

def detect_bright_regions(image: np.ndarray, disk_info: Optional[Dict] = None) -> List[Dict]:
    """Detect bright regions (flares, plages, faculae) on the solar disk.

    REVISED APPROACH (v2.4) - Multi-Scale Analysis for Plage:
    - Large-scale analysis: detect broad plage regions using large Gaussian blur
    - Medium-scale analysis: detect faculae and bright complexes
    - Small-scale analysis: detect flares and small bright points
    - Merge overlapping detections, prefer larger regions over small noise
    - Suppress small uncertain features that are likely noise
    """
    h, w = image.shape
    img_f = image.astype(np.float32)
    regions = []

    # Create disk mask
    if disk_info and disk_info.get("detected"):
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        r_inner = r * 0.97
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (r_inner) ** 2
        disk_pixels = img_f[disk_mask]
    else:
        cy, cx = h // 2, w // 2
        r = min(h, w) * 0.45
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
        disk_pixels = img_f[disk_mask]

    if len(disk_pixels) < 100:
        return []

    disk_mean = float(np.mean(disk_pixels))
    disk_std = float(np.std(disk_pixels))
    disk_max = float(np.max(disk_pixels))
    disk_min = float(np.min(disk_pixels))

    logger.info(f"Bright region detection: mean={disk_mean:.1f}, std={disk_std:.1f}, "
                f"range=[{disk_min:.1f}, {disk_max:.1f}]")

    try:
        import cv2

        # === Multi-scale local contrast analysis (4 scales) ===
        # Scale 1: Very large window (8% of disk radius) - for large plage regions
        sigma_vl = max(r * 0.08, 15)
        blurred_vl = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_vl)
        contrast_vl = img_f - blurred_vl
        
        # Scale 2: Large window (4% of disk radius) - for plage
        sigma_large = max(r * 0.04, 8)
        blurred_large = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_large)
        contrast_large = img_f - blurred_large
        
        # Scale 3: Intermediate window (2.5% of disk radius) - for mid-scale bright features
        sigma_mid = max(r * 0.025, 5)
        blurred_mid = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_mid)
        contrast_mid = img_f - blurred_mid
        
        # Scale 4: Medium window (1.2% of disk radius) - for faculae and small bright points
        sigma_med = max(r * 0.012, 3)
        blurred_med = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_med)
        contrast_med = img_f - blurred_med
        
        # Compute local statistics within disk
        contrast_pixels_vl = contrast_vl[disk_mask]
        contrast_pixels_large = contrast_large[disk_mask]
        contrast_pixels_mid = contrast_mid[disk_mask]
        contrast_pixels_med = contrast_med[disk_mask]
        
        contrast_std_vl = float(np.std(contrast_pixels_vl))
        contrast_std_large = float(np.std(contrast_pixels_large))
        contrast_std_mid = float(np.std(contrast_pixels_mid))
        contrast_std_med = float(np.std(contrast_pixels_med))
        
        logger.info(f"Local contrast std: vl={contrast_std_vl:.1f}, large={contrast_std_large:.1f}, "
                   f"mid={contrast_std_mid:.1f}, med={contrast_std_med:.1f}")
        
        # Threshold: use moderate thresholds for all scales
        # Lower thresholds mean higher sensitivity
        thresh_vl = 1.0 * contrast_std_vl
        thresh_large = 1.2 * contrast_std_large
        thresh_mid = 1.0 * contrast_std_mid
        thresh_med = 1.0 * contrast_std_med
        
        # Global minimum contrast floor to suppress noise
        global_min_contrast = disk_std * 0.15
        
        thresh_vl = max(thresh_vl, global_min_contrast)
        thresh_large = max(thresh_large, global_min_contrast)
        thresh_mid = max(thresh_mid, global_min_contrast)
        thresh_med = max(thresh_med, global_min_contrast)
        
        # Create binary masks for bright regions at each scale
        _, mask_vl = cv2.threshold(
            contrast_vl.astype(np.float32), 
            float(thresh_vl), 1.0, cv2.THRESH_BINARY
        )
        _, mask_large = cv2.threshold(
            contrast_large.astype(np.float32), 
            float(thresh_large), 1.0, cv2.THRESH_BINARY
        )
        _, mask_mid = cv2.threshold(
            contrast_mid.astype(np.float32),
            float(thresh_mid), 1.0, cv2.THRESH_BINARY
        )
        _, mask_med = cv2.threshold(
            contrast_med.astype(np.float32),
            float(thresh_med), 1.0, cv2.THRESH_BINARY
        )
        
        # Merge all scale masks
        combined_mask = cv2.bitwise_or(
            (mask_vl * 255).astype(np.uint8),
            (mask_large * 255).astype(np.uint8),
        )
        combined_mask = cv2.bitwise_or(combined_mask, (mask_mid * 255).astype(np.uint8))
        combined_mask = cv2.bitwise_or(combined_mask, (mask_med * 255).astype(np.uint8))
        
        # Apply disk mask
        disk_mask_uint8 = (disk_mask * 255).astype(np.uint8)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=disk_mask_uint8)
        
        # Morphological cleanup - gentle, preserve large features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        
        # Minimum area - lowered to catch smaller features
        min_area = max(12, int(h * w * 0.00005))
        max_bright_area = int(disk_mask.sum() * 0.25)  # Allow larger plage regions
        
        # First pass: collect all candidate regions with their properties
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_bright_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            
            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])
            
            # Must be within disk
            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            if dist_from_center > r * 0.97:
                continue
            
            # Calculate actual brightness on ORIGINAL image
            region_pixels = img_f[labels == i]
            region_mean = float(np.mean(region_pixels))
            brightness_ratio = region_mean / disk_mean if disk_mean > 0 else 1.0
            
            # Safety: reject if darker than mean
            if region_mean <= disk_mean:
                continue
            
            # Calculate local contrast at all 4 scales
            local_contrast_vl = float(np.mean(contrast_vl[labels == i]))
            local_contrast_l = float(np.mean(contrast_large[labels == i]))
            local_contrast_mid = float(np.mean(contrast_mid[labels == i]))
            local_contrast_m = float(np.mean(contrast_med[labels == i]))
            max_contrast = max(local_contrast_vl, local_contrast_l, local_contrast_mid, local_contrast_m)
            
            candidates.append({
                "x": cx_comp,
                "y": cy_comp,
                "left": x,
                "top": y,
                "width": bw,
                "height": bh,
                "area": area,
                "brightness_ratio": brightness_ratio,
                "local_contrast_vl": local_contrast_vl,
                "local_contrast_l": local_contrast_l,
                "local_contrast_mid": local_contrast_mid,
                "local_contrast_m": local_contrast_m,
                "max_contrast": max_contrast,
                "label": i,
            })
        
        # Second pass: prioritize large features, suppress small noise near large regions
        # Sort by area descending
        candidates.sort(key=lambda c: c["area"], reverse=True)
        
        # Remove small candidates that overlap significantly with large ones
        filtered_candidates = []
        for cand in candidates:
            is_redundant = False
            for kept in filtered_candidates:
                # If kept feature is much larger, skip small overlapping ones
                if kept["area"] > cand["area"] * 3:
                    dx = abs(cand["x"] - kept["x"])
                    dy = abs(cand["y"] - kept["y"])
                    dist = math.sqrt(dx * dx + dy * dy)
                    overlap_radius = (math.sqrt(kept["area"]) + math.sqrt(cand["area"])) / 2
                    if dist < overlap_radius * 0.8:
                        is_redundant = True
                        break
            if not is_redundant:
                filtered_candidates.append(cand)
        
        # Third pass: classify and build final regions
        for cand in filtered_candidates:
            area = cand["area"]
            brightness_ratio = cand["brightness_ratio"]
            max_contrast = cand["max_contrast"]
            
            # Determine type based on brightness ratio and size
            if brightness_ratio > 1.20 and area < int(h * w * 0.01):
                region_type = "flare"      # Very bright, compact
            elif brightness_ratio > 1.08:
                region_type = "plage"      # Moderately bright (plage)
            else:
                region_type = "facula"     # Slightly bright
            
            # Confidence based on contrast strength and area bonus
            confidence = min(max_contrast / max(contrast_std_large * 2, 1e-8), 0.95)
            # Bonus for larger features (more reliable)
            area_bonus = min(math.log2(max(area / min_area, 1)) / 5.0, 0.2)
            confidence = min(confidence + area_bonus, 0.95)
            
            if confidence < 0.1:
                continue
            
            regions.append({
                "x": cand["x"],
                "y": cand["y"],
                "width": cand["width"],
                "height": cand["height"],
                "area": area,
                "type": region_type,
                "brightness_ratio": brightness_ratio,
                "confidence": confidence,
                "local_contrast": max_contrast,
            })
            logger.info(f"  Bright region #{len(regions)}: type={region_type}, center=({cand['x']:.1f},{cand['y']:.1f}), "
                       f"area={area}, brightness_ratio={brightness_ratio:.2f}, contrast={max_contrast:.1f}, conf={confidence:.2f}")
    
    except Exception as e:
        logger.error(f"Bright region detection failed: {e}", exc_info=True)

    logger.info(f"Final bright region count: {len(regions)}")
    return regions


# ============================================================
# Sunspot Group Clustering
# ============================================================

def cluster_sunspot_groups(sunspots: List[Dict], disk_info: Optional[Dict] = None) -> List[Dict]:
    """Cluster sunspots into groups using DBSCAN or distance-based clustering."""
    groups = []
    if len(sunspots) < 2:
        return groups

    try:
        from sklearn.cluster import DBSCAN
        coords = np.array([[s["x"], s["y"]] for s in sunspots])
        # Use disk radius as scale reference
        # BUG FIX: coords.shape[1] is always 2 (x,y), not image dimension. Use image shape from first sunspot if available.
        if disk_info and disk_info.get("detected"):
            r = disk_info["radius"]
        else:
            # Estimate from sunspot spread: find max distance between any two spots
            if len(coords) >= 2:
                max_spot_dist = max(
                    math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
                    for i in range(len(coords)) for j in range(i + 1, len(coords))
                )
                r = max_spot_dist * 2.5  # Assume spots span ~40% of disk diameter
            else:
                r = 200  # Hard fallback
        # DBSCAN eps: spots within 10% of disk radius are grouped
        clustering = DBSCAN(eps=r * 0.1, min_samples=2).fit(coords)
        labels = clustering.labels_

        groups = []
        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            member_indices = np.where(labels == label)[0]
            if len(member_indices) < 2:
                continue
            members = [sunspots[i] for i in member_indices]
            group_cx = float(np.mean([m["x"] for m in members]))
            group_cy = float(np.mean([m["y"] for m in members]))
            max_dist = max(math.sqrt((m["x"] - group_cx) ** 2 + (m["y"] - group_cy) ** 2) for m in members)
            complexity = min(len(members) * 1.5 + max_dist / r * 2, 10.0)
            groups.append({
                "id": len(groups) + 1,
                "member_count": len(members),
                "member_indices": member_indices.tolist(),
                "center_x": group_cx,
                "center_y": group_cy,
                "max_spread": float(max_dist),
                "complexity": float(complexity),
                "confidence": min(0.5 + len(members) * 0.1, 0.95),
            })
            logger.info(f"  Group #{groups[-1]['id']}: {len(members)} members, complexity={complexity:.1f}")

    except ImportError:
        # Simple distance-based clustering fallback
        r = disk_info["radius"] if disk_info and disk_info.get("detected") else 200
        threshold = r * 0.15
        used = set()
        groups = []
        for i, s1 in enumerate(sunspots):
            if i in used:
                continue
            members = [i]
            for j, s2 in enumerate(sunspots):
                if j in used or j == i:
                    continue
                dist = math.sqrt((s1["x"] - s2["x"]) ** 2 + (s1["y"] - s2["y"]) ** 2)
                if dist < threshold:
                    members.append(j)
            if len(members) >= 2:
                for m in members:
                    used.add(m)
                member_spots = [sunspots[m] for m in members]
                group_cx = float(np.mean([m["x"] for m in member_spots]))
                group_cy = float(np.mean([m["y"] for m in member_spots]))
                groups.append({
                    "id": len(groups) + 1,
                    "member_count": len(members),
                    "member_indices": members,
                    "center_x": group_cx,
                    "center_y": group_cy,
                    "complexity": min(len(members) * 2.0, 10.0),
                    "confidence": min(0.5 + len(members) * 0.1, 0.95),
                })

    logger.info(f"Final sunspot group count: {len(groups)}")
    return groups


# ============================================================
# Image Statistics
# ============================================================

def compute_image_stats(image: np.ndarray, disk_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Compute basic image statistics."""
    h, w = image.shape
    img_f = image.astype(np.float32)

    if disk_info and disk_info.get("detected"):
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
        disk_pixels = img_f[disk_mask]
    else:
        disk_pixels = img_f.flatten()

    # SAFETY: guard against empty disk_pixels (division by zero)
    if len(disk_pixels) == 0:
        return {
            "width": w, "height": h,
            "mean_brightness": 0.0, "std_brightness": 0.0,
            "min_brightness": 0.0, "max_brightness": 0.0,
            "median_brightness": 0.0, "contrast": 0.0,
            "dark_pixel_ratio": 0.0, "bright_pixel_ratio": 0.0,
            "edge_density": 0.0,
        }

    dark_ratio = np.sum(disk_pixels < np.median(disk_pixels) * 0.7) / len(disk_pixels)
    bright_ratio = np.sum(disk_pixels > np.median(disk_pixels) * 1.2) / len(disk_pixels)

    # Edge density
    try:
        import cv2
        blurred = cv2.GaussianBlur(image, (5, 5), 1)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
    except Exception:
        edge_density = 0.0

    return {
        "width": w,
        "height": h,
        "mean_brightness": float(np.mean(disk_pixels)),
        "std_brightness": float(np.std(disk_pixels)),
        "min_brightness": float(np.min(disk_pixels)),
        "max_brightness": float(np.max(disk_pixels)),
        "median_brightness": float(np.median(disk_pixels)),
        "contrast": float(np.std(disk_pixels) / (np.mean(disk_pixels) + 1e-8)),
        "dark_pixel_ratio": float(dark_ratio),
        "bright_pixel_ratio": float(bright_ratio),
        "edge_density": float(edge_density),
    }


# ============================================================
# On-Disk Filament Detection
# ============================================================

def detect_filaments(image: np.ndarray, disk_info: Optional[Dict] = None, sunspots: Optional[List] = None) -> List[Dict]:
    """Detect dark elongated structures (filaments) on the solar disk surface.
    
    Filaments are dark, thread-like structures that appear ON the solar disk,
    distinctly different from prominences which appear at the limb.
    
    Key characteristics:
    - Located ON the disk surface (not at the limb)
    - Dark, elongated, thread-like morphology
    - Often span large areas along magnetic neutral lines
    - Lower contrast than sunspots, harder to detect
    """
    h, w = image.shape
    img_f = image.astype(np.float32)
    filaments = []
    
    if not disk_info or not disk_info.get("detected"):
        return []
    
    try:
        import cv2
        
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        
        # Create inner disk mask (exclude limb region)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        
        # Search region: inner disk, exclude very center and far limb
        # Expanded: 0.2r ~ 0.95r to catch filaments near center and edge
        disk_inner_mask = (dist_sq >= (r * 0.2) ** 2) & (dist_sq <= (r * 0.95) ** 2)
        
        if disk_inner_mask.sum() < 100:
            return []
        
        # Get statistics on the inner disk region
        disk_pixels = img_f[disk_inner_mask]
        if len(disk_pixels) < 100:
            return []
        
        disk_mean = float(np.mean(disk_pixels))
        disk_std = float(np.std(disk_pixels))
        
        logger.info(f"Filament detection: disk_mean={disk_mean:.1f}, disk_std={disk_std:.1f}")
        
        # Create sunspot exclusion mask to avoid false positives
        sunspot_mask = np.zeros((h, w), dtype=np.uint8)
        if sunspots:
            for spot in sunspots:
                spot_x, spot_y = int(spot["x"]), int(spot["y"])
                spot_r = int(spot.get("radius", 10) * 1.5)
                if spot_r > 0:
                    cv2.circle(sunspot_mask, (spot_x, spot_y), spot_r, 255, -1)
        
        # Method 1: Multi-scale dark structure detection
        # Use directional filters to detect elongated dark features
        
        # Compute local background using large Gaussian blur
        sigma_bg = max(r * 0.1, 20)
        blurred_bg = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_bg)
        local_contrast = img_f - blurred_bg
        
        # Threshold for dark features
        dark_thresh = -disk_std * 0.8
        _, dark_mask = cv2.threshold(
            local_contrast.astype(np.float32),
            float(dark_thresh), 1.0, cv2.THRESH_BINARY_INV
        )
        
        # Method 2: Directional derivative analysis
        # Filaments are elongated - use directional filters
        kernel_size = max(int(r * 0.02), 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Sobel filters for edge detection
        sobel_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=5)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # Detect dark regions with high gradient (structured dark features)
        _, grad_mask = cv2.threshold(
            gradient_mag.astype(np.float32),
            float(disk_std * 0.3), 1.0, cv2.THRESH_BINARY
        )
        
        # Combine: dark AND structured (high gradient)
        combined_mask = cv2.bitwise_and(
            (dark_mask * 255).astype(np.uint8),
            (grad_mask * 255).astype(np.uint8)
        )
        
        # Apply disk mask and exclude sunspots
        disk_mask_uint8 = (disk_inner_mask * 255).astype(np.uint8)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=disk_mask_uint8)
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(sunspot_mask))
        
        # Morphological operations - use elongated kernel to preserve filament shapes
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, kernel_size), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, kernel_size)))
        
        # Apply both horizontal and vertical morphological operations
        h_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, v_kernel, iterations=1)
        combined_mask = cv2.bitwise_or(h_opened, v_opened)
        
        # Gentle dilation to connect broken segments
        combined_mask = cv2.dilate(combined_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        
        # Minimum area - filaments are typically elongated but not tiny
        min_filament_area = max(30, int(h * w * 0.00005))
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_filament_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Elongation check - filaments are elongated
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if aspect_ratio < 1.5:
                continue  # Too round, likely not a filament
            
            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])
            
            # Must be within inner disk (consistent with mask: 0.2r ~ 0.95r)
            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            norm_dist = dist_from_center / r if r > 0 else 0
            if norm_dist < 0.18 or norm_dist > 0.95:
                continue
            
            # Verify it's actually dark
            feature_pixels = img_f[labels == i]
            feature_mean = float(np.mean(feature_pixels))
            brightness_contrast = (disk_mean - feature_mean) / max(disk_std, 1)
            
            if brightness_contrast < 0.3:
                continue  # Not dark enough
            
            # Confidence based on elongation and contrast
            elongation_score = min((aspect_ratio - 1.0) / 3.0, 1.0)
            confidence = min(0.4 * elongation_score + 0.4 * min(brightness_contrast / 1.5, 1.0) + 0.2 * min(math.log2(max(area / min_filament_area, 1)) / 4.0, 1.0), 0.95)
            
            if confidence < 0.15:
                continue
            
            # Normalized coordinates
            norm_x = (cx_comp - cx) / r
            norm_y = (cy_comp - cy) / r
            
            filaments.append({
                "x": float(cx_comp),
                "y": float(cy_comp),
                "width": int(bw),
                "height": int(bh),
                "area": int(area),
                "aspect_ratio": float(aspect_ratio),
                "type": "filament",
                "brightness_contrast": float(brightness_contrast),
                "elongation_score": float(elongation_score),
                "confidence": confidence,
                "norm_x": float(norm_x),
                "norm_y": float(norm_y),
            })
            logger.info(f"  Filament #{len(filaments)}: center=({cx_comp:.1f},{cy_comp:.1f}), "
                       f"area={area}, aspect_ratio={aspect_ratio:.1f}, contrast={brightness_contrast:.2f}, conf={confidence:.2f}")
    
    except ImportError:
        logger.warning("OpenCV not available for filament detection")
    except Exception as e:
        logger.error(f"Filament detection failed: {e}", exc_info=True)
    
    logger.info(f"Final filament count: {len(filaments)}")
    return filaments


# ============================================================
# Prominence Detection (Limb Features)
# ============================================================

def detect_prominences(image: np.ndarray, disk_info: Optional[Dict] = None) -> List[Dict]:
    """Detect prominence candidates at solar limb (edge features) - v3.0.

    MULTI-SCALE APPROACH (v3.0):
    1. TRIPLE-SCALE local contrast analysis: small(0.03r), medium(0.06r), large(0.15r)
       - Small scale: catches narrow spike-like prominences
       - Medium scale: catches typical flame/shrub prominences  
       - Large scale: catches broad arc/ring prominences (the ones most often missed)
    2. Radial brightness normalization to compensate limb darkening/brightening
    3. Lowered absolute thresholds for large-scale detection
    4. Minimum confidence 0.08 (was 0.10) to admit borderline features
    """
    h, w = image.shape
    img_f = image.astype(np.float32)
    prominences = []
    
    if not disk_info or not disk_info.get("detected"):
        return []
    
    try:
        import cv2
        
        cx, cy = disk_info["center_x"], disk_info["center_y"]
        r = disk_info["radius"]
        
        # Step 1: Create annular mask - extended range for large prominences
        # Search: 0.90r (slightly inside) to 2.0r (extended region for giant prominences)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        dist_map = np.sqrt(dist_sq)
        
        # Triple annular masks for different scales
        inner_mask = (dist_sq >= (r * 0.90) ** 2) & (dist_sq <= (r * 1.15) ** 2)
        mid_mask = (dist_sq >= (r * 0.90) ** 2) & (dist_sq <= (r * 1.45) ** 2)
        outer_mask = (dist_sq >= (r * 0.90) ** 2) & (dist_sq <= (r * 2.0) ** 2)
        
        if outer_mask.sum() < 50:
            return []
        
        # Step 2: Radial brightness normalization
        # The limb region has a brightness gradient; compensate it
        limb_pixels_all = img_f[outer_mask]
        if len(limb_pixels_all) < 50:
            return []
        
        limb_mean = float(np.mean(limb_pixels_all))
        limb_std = float(np.std(limb_pixels_all))
        
        # Create normalized image where limb background is ~0
        img_normalized = img_f.copy()
        # Smooth radial profile estimation
        dist_binned = np.clip((dist_map / max(r, 1) * 20).astype(np.int32), 0, 40)
        radial_mean = np.zeros(41, dtype=np.float32)
        radial_count = np.zeros(41, dtype=np.int32)
        for b in range(41):
            mask_bin = (dist_binned == b)
            radial_count[b] = mask_bin.sum()
            if radial_count[b] > 0:
                radial_mean[b] = img_f[mask_bin].mean()
        # Interpolate gaps
        for b in range(1, 40):
            if radial_count[b] == 0:
                radial_mean[b] = (radial_mean[b-1] + radial_mean[b+1]) / 2 if radial_count[b+1] > 0 else radial_mean[b-1]
        # Apply normalization
        for b in range(41):
            mask_bin = (dist_binned == b)
            if mask_bin.sum() > 0 and radial_mean[b] > 0:
                img_normalized[mask_bin] = img_f[mask_bin] - radial_mean[b]
        
        # Step 3: TRIPLE-SCALE local contrast analysis
        # Scale 1: small (catches spike/sharp prominences)
        sigma_small = max(r * 0.025, 4)
        blurred_small = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_small)
        contrast_small = img_normalized - cv2.GaussianBlur(img_normalized, (0, 0), sigmaX=sigma_small)
        
        # Scale 2: medium (catches flame/bush prominences)
        sigma_med = max(r * 0.06, 8)
        contrast_med = img_normalized - cv2.GaussianBlur(img_normalized, (0, 0), sigmaX=sigma_med)
        
        # Scale 3: LARGE (catches broad arc/ring prominences - the main missing ones)
        sigma_large = max(r * 0.15, 15)
        blurred_large = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_large)
        contrast_large = img_normalized - cv2.GaussianBlur(img_normalized, (0, 0), sigmaX=sigma_large)
        
        # Compute contrast STD per scale on outer mask
        cs_std = float(np.std(contrast_small[outer_mask]))
        cm_std = float(np.std(contrast_med[outer_mask]))
        cl_std = float(np.std(contrast_large[outer_mask]))
        
        logger.info(f"Prominence v3: limb_mean={limb_mean:.1f}, limb_std={limb_std:.1f}, "
                   f"contrast_std[s={cs_std:.1f}, m={cm_std:.1f}, l={cl_std:.1f}]")
        
        # Step 4: Per-scale thresholding with LOWER thresholds for large scale
        # Small: 0.7x std, Medium: 0.55x std, Large: 0.35x std (much lower!)
        thr_bright_s = cs_std * 0.7
        thr_dark_s = -cs_std * 0.7
        thr_bright_m = cm_std * 0.55
        thr_dark_m = -cm_std * 0.55
        thr_bright_l = cl_std * 0.35  # MUCH lower - large prominences are often faint
        thr_dark_l = -cl_std * 0.35
        
        # Generate binary masks per scale
        def _thresh_binary(img, thresh):
            result = np.zeros_like(img, dtype=np.uint8)
            result[img > thresh] = 255
            return result
        
        bin_bright_s = _thresh_binary(contrast_small, thr_bright_s) if thr_bright_s > 0 else np.zeros_like(contrast_small, dtype=np.uint8)
        bin_bright_m = _thresh_binary(contrast_med, thr_bright_m) if thr_bright_m > 0 else np.zeros_like(contrast_med, dtype=np.uint8)
        bin_bright_l = _thresh_binary(contrast_large, thr_bright_l) if thr_bright_l > 0 else np.zeros_like(contrast_large, dtype=np.uint8)
        
        # Combine all scales: OR operation
        combined_bright = cv2.bitwise_or(bin_bright_s, bin_bright_m)
        combined_bright = cv2.bitwise_or(combined_bright, bin_bright_l)
        
        # Apply outer annular mask
        outer_mask_uint8 = (outer_mask * 255).astype(np.uint8)
        combined_masked = cv2.bitwise_and(combined_bright, combined_bright, mask=outer_mask_uint8)
        
        # Very gentle morphological cleanup for large scale
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_masked = cv2.morphologyEx(combined_masked, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        # Don't open - would kill large faint regions! Only dilate slightly
        combined_masked = cv2.dilate(combined_masked, kernel_small, iterations=2)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_masked, connectivity=4
        )
        
        # VERY low min area for large scale (but filter by confidence later)
        min_prominence_area = max(6, int(h * w * 0.00001))
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_prominence_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            
            cx_comp = float(centroids[i][0])
            cy_comp = float(centroids[i][1])
            
            # Distance from disk center (normalized)
            dist_from_center = math.sqrt((cx_comp - cx) ** 2 + (cy_comp - cy) ** 2)
            norm_dist = dist_from_center / r if r > 0 else 0
            
            # Must be OUTSIDE the disk (>=0.92r, allow wider tolerance)
            if norm_dist < 0.92 or norm_dist > 2.05:
                continue
            
            # Extract feature pixels to determine if bright
            feature_pixels = img_f[labels == i]
            feature_mean = float(np.mean(feature_pixels))
            
            # Multi-scale contrast for this feature
            fc_small = float(np.mean(contrast_small[labels == i])) if labels.sum() > 0 else 0
            fc_med = float(np.mean(contrast_med[labels == i])) if labels.sum() > 0 else 0
            fc_large = float(np.mean(contrast_large[labels == i])) if labels.sum() > 0 else 0
            feature_contrast = max(fc_small, fc_med, fc_large)
            
            # Classify: bright vs dark
            if feature_mean > limb_mean:
                prom_type = "prominence"
                brightness_contrast = (feature_mean - limb_mean) / max(limb_std, 1)
            else:
                prom_type = "filament"
                brightness_contrast = (limb_mean - feature_mean) / max(limb_std, 1)
            
            # LOWERED minimum brightness contrast: 0.05 for large scale (was 0.08)
            # Large prominences are often faint but extended - they may have negative
            # local contrast (smooth structure) but still be clearly brighter than
            # the dark space background. Don't filter them out.
            if brightness_contrast < 0.05:
                continue
            
            # Size bonus: larger features get confidence boost
            size_bonus = min(area / max(r * r * 0.01, 1), 0.3)
            
            # Elongation score
            elongation_score = min(aspect_ratio / 2.0, 1.0)
            
            # Position score (distance from disk edge, normalized)
            position_score = min(max(norm_dist - 0.92, 0) / 0.8, 1.0)
            
            # Combined confidence - heavily weight contrast for large features
            confidence = float(
                0.20 * position_score + 
                0.15 * elongation_score + 
                0.45 * min(max(brightness_contrast, 0) / 1.5, 1.0) +
                0.20 * min(size_bonus / 0.3, 1.0)
            )
            confidence = min(max(confidence, 0.08), 0.95)
            
            if confidence < 0.08:
                continue
            
            # Normalized coordinates
            norm_x = (cx_comp - cx) / r
            norm_y = (cy_comp - cy) / r
            
            prominences.append({
                "x": float(cx_comp),
                "y": float(cy_comp),
                "radius": float(math.sqrt(area / math.pi)),
                "area": int(area),
                "bbox_width": int(bw),
                "bbox_height": int(bh),
                "aspect_ratio": float(aspect_ratio),
                "type": prom_type,
                "norm_distance": float(norm_dist),
                "brightness_contrast": float(brightness_contrast),
                "feature_contrast": float(feature_contrast),
                "position_score": float(position_score),
                "elongation_score": float(elongation_score),
                "size_bonus": float(size_bonus),
                "confidence": confidence,
                "norm_x": float(norm_x),
                "norm_y": float(norm_y),
            })
            logger.info(f"  Prominence #{len(prominences)}: type={prom_type}, center=({cx_comp:.1f},{cy_comp:.1f}), "
                       f"area={area}, norm_dist={norm_dist:.2f}, contrast[b={brightness_contrast:.2f},f={feature_contrast:.1f}], "
                       f"size_bonus={size_bonus:.2f}, conf={confidence:.2f}")
    
    except ImportError:
        logger.warning("OpenCV not available for prominence detection")
    except Exception as e:
        logger.error(f"Prominence detection failed: {e}", exc_info=True)
    
    logger.info(f"Final prominence count: {len(prominences)}")
    return prominences


# ============================================================
# Limb Enhancement for AI Analysis
# ============================================================

def enhance_limb_for_ai(image: np.ndarray, disk_info: Optional[Dict] = None) -> np.ndarray:
    """Enhance the limb region of a solar image to improve AI prominence detection.

    Creates a copy of the input image where the annular region outside the solar 
    disk (0.90r to ~2r) has stretched contrast using CLAHE (Contrast Limited 
    Adaptive Histogram Equalization). This makes faint prominences much more 
    visible to AI vision models without distorting the solar disk itself.

    Args:
        image: Grayscale solar image (H, W) or (H, W, 1)
        disk_info: Solar disk detection result with center_x, center_y, radius

    Returns:
        Enhanced image (same shape and dtype as input), or original if enhancement fails
    """
    if disk_info is None or not disk_info.get("detected"):
        return image

    try:
        import cv2

        h, w = image.shape[:2]
        cx = disk_info["center_x"]
        cy = disk_info["center_y"]
        r = disk_info["radius"]

        # Create mask for limb region: 0.90r to 2.0r
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        limb_mask = (dist_sq >= (r * 0.90) ** 2) & (dist_sq <= (r * 2.0) ** 2)

        if limb_mask.sum() < 50:
            return image

        # Create enhanced copy
        enhanced = image.astype(np.float32)

        # Extract limb pixels
        limb_pixels = enhanced[limb_mask]
        if len(limb_pixels) < 100:
            return image

        # Apply CLAHE to the limb region via local processing
        # Rescale limb pixels to 0-255 for CLAHE
        limb_min = float(limb_pixels.min())
        limb_max = float(limb_pixels.max())
        limb_range = limb_max - limb_min
        if limb_range < 1:
            return image

        limb_norm = ((limb_pixels - limb_min) / limb_range * 255).astype(np.uint8)

        # Apply CLAHE (requires 2D, so reshape to a thin strip)
        n_pixels = len(limb_norm)
        strip_w = min(n_pixels, 512)
        strip_h = (n_pixels + strip_w - 1) // strip_w
        padded = np.zeros(strip_h * strip_w, dtype=np.uint8)
        padded[:n_pixels] = limb_norm
        strip = padded.reshape(strip_h, strip_w)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        strip_enhanced = clahe.apply(strip)
        limb_enhanced = strip_enhanced.ravel()[:n_pixels]

        # Convert back to original range
        limb_enhanced_f = limb_enhanced.astype(np.float32) / 255.0 * limb_range + limb_min

        # Apply enhanced values to limb region
        enhanced[limb_mask] = limb_enhanced_f

        # Blend: 70% enhanced + 30% original for natural look
        blended = enhanced * 0.7 + image.astype(np.float32) * 0.3

        result = np.clip(blended, 0, 255).astype(image.dtype)

        logger.info(
            f"Limb enhancement applied: {limb_mask.sum()} pixels, "
            f"range {limb_min:.0f}-{limb_max:.0f} -> CLAHE stretched"
        )
        return result

    except ImportError:
        logger.warning("OpenCV not available for limb enhancement")
        return image
    except Exception as e:
        logger.error(f"Limb enhancement failed: {e}", exc_info=True)
        return image


# ============================================================
# Preprocessing Pipeline
# ============================================================

def _add_index(features: List[Dict], start: int = 1) -> List[Dict]:
    """Add sequential index to each feature dict."""
    for i, feat in enumerate(features):
        feat["index"] = start + i
    return features


def preprocess_solar_image(image: np.ndarray) -> Dict[str, Any]:
    """Run the complete solar image preprocessing pipeline (OPTIMIZED v2.5).

    Pipeline execution order:
    1. Detect solar disk (blocking, required for all other steps)
    2. Segment sunspots (blocking, required for bright region detection)
    3. Parallel execution of independent detectors:
       - Bright regions (flares/plages/faculae)
       - Filaments (dark elongated structures on disk)
       - Prominences (limb features)
    4. Cluster sunspot groups
    5. Compute image statistics
    6. Assign sequential index to ALL features across the full pipeline

    Returns:
        Dict with keys:
            - solar_disk: disk detection result
            - sunspots: list of sunspot candidates (with index)
            - bright_regions: list of bright region candidates (with index)
            - filaments: list of filament candidates (with index)
            - prominences: list of prominence candidates (with index)
            - sunspot_groups: list of sunspot groups
            - image_stats: image statistics
            - processing_info: pipeline metadata
    """
    import concurrent.futures
    
    logger.info("=" * 60)
    logger.info("Starting solar image preprocessing pipeline (v2.5 optimized)")
    logger.info(f"Image size: {image.shape}")

    # Step 1: Detect solar disk (blocking - required for all other steps)
    logger.info("Step 1: Detecting solar disk...")
    disk_info = detect_solar_disk(image)

    # Step 2: Segment sunspots (blocking - required for bright region detection)
    logger.info("Step 2: Segmenting sunspots...")
    sunspots = segment_sunspots(image, disk_info)

    # Step 3: Parallel execution of independent detectors
    logger.info("Step 3: Running parallel detection (bright regions, filaments, prominences)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_bright = executor.submit(detect_bright_regions, image, disk_info)
        f_filaments = executor.submit(detect_filaments, image, disk_info, sunspots)
        f_prominences = executor.submit(detect_prominences, image, disk_info)
        
        bright_regions = f_bright.result()
        filaments = f_filaments.result()
        prominences = f_prominences.result()

    # Step 4: Cluster sunspot groups
    logger.info("Step 4: Clustering sunspot groups...")
    groups = cluster_sunspot_groups(sunspots, disk_info)

    # Step 5: Compute image stats
    logger.info("Step 5: Computing image statistics...")
    stats = compute_image_stats(image, disk_info)

    # Step 6: Assign sequential index to ALL features across the full pipeline
    idx = 1
    _add_index(sunspots, idx); idx += len(sunspots)
    _add_index(bright_regions, idx); idx += len(bright_regions)
    _add_index(filaments, idx); idx += len(filaments)
    _add_index(prominences, idx); idx += len(prominences)

    result = {
        "solar_disk": disk_info,
        "sunspots": sunspots,
        "bright_regions": bright_regions,
        "filaments": filaments,
        "prominences": prominences,
        "sunspot_groups": groups,
        "image_stats": stats,
        "processing_info": {"pipeline_version": "2.5"},
    }

    logger.info(f"Pipeline complete: {len(sunspots)} sunspots, {len(bright_regions)} bright regions, "
                f"{len(filaments)} filaments, {len(prominences)} prominences, {len(groups)} groups")
    logger.info("=" * 60)
    return result


# ============================================================
# Feature Prompt Generation
# ============================================================

def generate_feature_prompt(preprocess_result: Dict[str, Any]) -> str:
    """Generate a detailed feature description prompt from preprocessing results."""
    disk = preprocess_result.get("solar_disk", {})
    sunspots = preprocess_result.get("sunspots", [])
    bright_regions = preprocess_result.get("bright_regions", [])
    filaments = preprocess_result.get("filaments", [])
    prominences = preprocess_result.get("prominences", [])
    groups = preprocess_result.get("sunspot_groups", [])
    stats = preprocess_result.get("image_stats", {})

    lines = []
    lines.append("=== 太阳图像特征预处理报告 ===")
    lines.append("")

    # Image stats
    lines.append(f"图像尺寸: {stats.get('width', '?')}x{stats.get('height', '?')} 像素")
    lines.append(f"平均亮度: {stats.get('mean_brightness', 0):.1f} (0-255)")
    lines.append(f"亮度范围: {stats.get('min_brightness', 0):.0f} - {stats.get('max_brightness', 0):.0f}")
    lines.append(f"对比度: {stats.get('contrast', 0):.3f}")
    lines.append(f"暗像素比例: {stats.get('dark_pixel_ratio', 0)*100:.1f}%")
    lines.append(f"亮像素比例: {stats.get('bright_pixel_ratio', 0)*100:.1f}%")
    lines.append(f"边缘密度: {stats.get('edge_density', 0):.4f}")
    lines.append("")

    # Solar disk (LOCKED BOUNDARY)
    lines.append("【日面边界锁定】")
    if disk.get("detected"):
        lines.append(f"  ✅ 日面已成功锁定")
        lines.append(f"  检测方法: {disk['method']}")
        lines.append(f"  置信度: {disk['confidence']:.2f}")
        lines.append(f"  中心坐标: ({disk['center_x']:.1f}, {disk['center_y']:.1f}) 像素")
        lines.append(f"  半径: {disk['radius']:.1f} 像素")
        h, w = stats.get('height', 1), stats.get('width', 1)
        disk_area = math.pi * disk['radius'] ** 2
        coverage = disk_area / (h * w) * 100
        lines.append(f"  日面覆盖率: {coverage:.1f}%")
        methods_tried = disk.get('methods_tried', [])
        if methods_tried:
            lines.append(f"  尝试方法: {', '.join(methods_tried)}")
    else:
        lines.append("  ❌ 日面未检测到，使用图像中心作为近似")
    lines.append("")

    # Sunspots
    lines.append(f"【黑子检测】共 {len(sunspots)} 个候选区域")
    if sunspots:
        for i, spot in enumerate(sunspots):
            lines.append(
                f"  黑子#{i+1}: 中心({spot['x']:.1f}, {spot['y']:.1f}), "
                f"半径{spot['radius']:.1f}px, "
                f"面积{spot['area']}px², "
                f"对比度{spot['contrast']:.2f}σ, "
                f"区域{spot['region']}, "
                f"置信度{spot['confidence']:.2f}"
            )
    else:
        lines.append("  (未检测到明显黑子)")
    lines.append("")

    # Sunspot groups
    if groups:
        lines.append(f"【黑子群组】共 {len(groups)} 个群组")
        for group in groups:
            members_str = ", ".join(f"#{i+1}" for i in group["member_indices"])
            lines.append(
                f"  群组{group['id']}: {group['member_count']}个黑子({members_str}), "
                f"中心({group['center_x']:.1f}, {group['center_y']:.1f}), "
                f"复杂度{group['complexity']:.1f}"
            )
        lines.append("")

    # Bright regions
    if bright_regions:
        lines.append(f"【亮区检测】共 {len(bright_regions)} 个候选区域")
        for i, region in enumerate(bright_regions):
            lines.append(
                f"  亮区#{i+1}: 类型{region['type']}, "
                f"中心({region['x']:.1f}, {region['y']:.1f}), "
                f"亮度比{region['brightness_ratio']:.2f}, "
                f"置信度{region['confidence']:.2f}"
            )
        lines.append("")

    # Filaments
    if filaments:
        lines.append(f"【暗条检测】共 {len(filaments)} 个候选区域")
        for i, fil in enumerate(filaments):
            lines.append(
                f"  暗条#{i+1}: 中心({fil['x']:.1f}, {fil['y']:.1f}), "
                f"面积{fil['area']}px², "
                f"纵横比{fil['aspect_ratio']:.1f}, "
                f"亮度对比{fil['brightness_contrast']:.2f}σ, "
                f"置信度{fil['confidence']:.2f}"
            )
        lines.append("")

    # Prominences
    if prominences:
        lines.append(f"【日珥检测】共 {len(prominences)} 个候选区域")
        for i, prom in enumerate(prominences):
            lines.append(
                f"  日珥#{i+1}: 类型{prom['type']}, "
                f"中心({prom['x']:.1f}, {prom['y']:.1f}), "
                f"面积{prom['area']}px², "
                f"距中心{prom['norm_distance']:.2f}R, "
                f"亮度对比{prom['brightness_contrast']:.2f}σ, "
                f"置信度{prom['confidence']:.2f}"
            )
        lines.append("")

    # Summary
    lines.append("=== 分析建议 ===")
    if not disk.get("detected"):
        lines.append("⚠️ 日面检测失败，建议检查图像是否为完整的太阳日面")
    elif not sunspots and not bright_regions and not prominences and not filaments:
        lines.append("✅ 日面已锁定，未检测到明显活动区特征")
        lines.append("  - 可能为宁静太阳图像，建议 Hale 分类: Alpha")
    else:
        lines.append(f"✅ 日面已锁定，检测到 {len(sunspots)} 个黑子，{len(bright_regions)} 个亮区，{len(filaments)} 个暗条，{len(prominences)} 个日珥")
        if groups:
            lines.append(f"  - 发现 {len(groups)} 个黑子群组，可能存在复杂活动区")

    return "\n".join(lines)


# ============================================================
# Image Loading Helper
# ============================================================

def load_image_cv2(image_path: str) -> Optional[np.ndarray]:
    """Load an image with OpenCV, handling Chinese paths."""
    import cv2
    # Use np.fromfile to handle Chinese characters in path
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray
