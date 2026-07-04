"""
Solar Sunspot Detection Module

Provides robust solar disk boundary detection and sunspot identification
using computer vision techniques. Handles varying illumination conditions,
different observation angles, and detects sunspots from center to limb.

Detection Pipeline:
1. Solar Disk Detection (Hough Circle Transform + Canny Edge)
2. Adaptive Thresholding for Sunspot Candidate Extraction
3. Morphological Filtering & Contour Analysis
4. Region Classification (Center / Mid-Latitude / Limb)
5. Coordinate Normalization & Output Formatting
"""

import math
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SolarDisk:
    """Detected solar disk boundary."""
    center_x: float          # Pixel x coordinate of disk center
    center_y: float          # Pixel y coordinate of disk center
    radius_px: float         # Disk radius in pixels
    image_width: int         # Original image width
    image_height: int        # Original image height
    detection_confidence: float = 0.0  # Confidence of disk detection (0-1)
    method_used: str = ""    # Detection method name

    @property
    def normalized_center(self) -> Tuple[float, float]:
        """Normalized center coordinates (0-1 relative to image dimensions)."""
        return (self.center_x / self.image_width, self.center_y / self.image_height)

    @property
    def normalized_radius(self) -> float:
        """Normalized radius relative to image width."""
        return self.radius_px / self.image_width

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": {"x": round(float(self.center_x), 2), "y": round(float(self.center_y), 2)},
            "center_normalized": {
                "x": round(float(self.normalized_center[0]), 4),
                "y": round(float(self.normalized_center[1]), 4),
            },
            "radius_px": round(float(self.radius_px), 2),
            "radius_normalized": round(float(self.normalized_radius), 4),
            "image_dimensions": {"width": int(self.image_width), "height": int(self.image_height)},
            "detection_confidence": round(float(self.detection_confidence), 4),
            "method": self.method_used,
        }


@dataclass
class Sunspot:
    """One detected sunspot with position, size, and classification."""
    spot_id: int = 0
    position_x: float = 0.0       # Pixel x of centroid
    position_y: float = 0.0       # Pixel y of centroid
    area_px: float = 0.0          # Area in pixels
    equivalent_diameter_px: float = 0.0  # Diameter of equivalent circle
    perimeter_px: float = 0.0     # Perimeter in pixels
    circularity: float = 0.0      # Shape circularity (0-1)
    mean_intensity: float = 0.0   # Mean pixel intensity inside spot
    confidence: float = 0.0       # Detection confidence (0-1)
    region: str = ""               # "center" | "mid_latitude" | "limb"
    distance_from_center: float = 0.0  # Normalized distance from disk center (0-1)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)

    @property
    def position_normalized(self) -> Tuple[float, float]:
        return (self.position_x, self.position_y)  # Already in normalized coords

    def to_dict(self, disk: SolarDisk) -> Dict[str, Any]:
        """Serialize with both pixel and normalized coordinates."""
        # Normalize to solar disk coordinates
        nx = (self.position_x - disk.center_x) / disk.radius_px
        ny = (self.position_y - disk.center_y) / disk.radius_px

        return {
            "spot_id": int(self.spot_id),
            "position_px": {"x": round(float(self.position_x), 2), "y": round(float(self.position_y), 2)},
            "position_solar_normalized": {"x": round(float(nx), 4), "y": round(float(ny), 4)},
            "area_px": round(float(self.area_px), 2),
            "equivalent_diameter_px": round(float(self.equivalent_diameter_px), 2),
            "circularity": round(float(self.circularity), 4),
            "mean_intensity": round(float(self.mean_intensity), 2),
            "confidence": round(float(self.confidence), 4),
            "region": self.region,
            "distance_from_center": round(float(self.distance_from_center), 4),
            "bounding_box": {"x": int(self.bbox[0]), "y": int(self.bbox[1]),
                            "width": int(self.bbox[2]), "height": int(self.bbox[3])},
        }


@dataclass
class SunspotDetectionResult:
    """Complete sunspot detection result."""
    solar_disk: Optional[SolarDisk] = None
    sunspots: List[Sunspot] = field(default_factory=list)
    total_spots: int = 0
    spots_by_region: Dict[str, int] = field(default_factory=lambda: {
        "center": 0, "mid_latitude": 0, "limb": 0,
    })
    processing_time_ms: float = 0.0
    image_path: str = ""
    detection_timestamp: str = ""
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        disk_dict = self.solar_disk.to_dict() if self.solar_disk else {}
        return {
            "solar_disk": disk_dict,
            "sunspots": [s.to_dict(self.solar_disk) for s in self.sunspots] if self.solar_disk else [],
            "total_spots": self.total_spots,
            "spots_by_region": self.spots_by_region,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "image_path": self.image_path,
            "detection_timestamp": self.detection_timestamp,
            "algorithm_params": self.algorithm_params,
        }


# ---------------------------------------------------------------------------
# Solar Disk Detection
# ---------------------------------------------------------------------------

class SolarDiskDetector:
    """Detect the solar disk boundary in a solar image.

    Uses a multi-strategy approach:
    1. Hough Circle Transform (primary)
    2. Contour-based ellipse/circle fitting (fallback)
    3. Intensity profile analysis (final fallback)
    """

    def __init__(self):
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect(self, image: np.ndarray) -> Optional[SolarDisk]:
        """Detect solar disk boundary. Tries multiple strategies."""
        h, w = image.shape[:2]

        # Strategy 1: Hough Circle Transform (best for well-defined disks)
        disk = self._detect_hough(image)
        if disk and disk.radius_px > w * 0.1:
            disk.method_used = "hough_circle"
            return disk

        # Strategy 2: Canny edge + contour-based circle fitting
        disk = self._detect_contour(image)
        if disk and disk.radius_px > w * 0.1:
            disk.method_used = "contour_fit"
            return disk

        # Strategy 3: Intensity profile (assume disk fills most of image)
        disk = self._detect_intensity_profile(image)
        if disk:
            disk.method_used = "intensity_profile"
            disk.detection_confidence = 0.5
            return disk

        return None

    def _detect_hough(self, image: np.ndarray) -> Optional[SolarDisk]:
        """Detect solar disk using Hough Circle Transform."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Preprocess
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        self.debug_images["hough_input"] = blurred

        # HoughCircles with parameter search
        min_radius = int(min(w, h) * 0.25)
        max_radius = int(min(w, h) * 0.52)

        for dp in [1.2, 1.5, 2.0]:
            for param1 in [50, 80, 100]:
                for param2 in [30, 40, 50, 60]:
                    circles = cv2.HoughCircles(
                        blurred, cv2.HOUGH_GRADIENT, dp=dp,
                        minDist=min_radius, param1=param1, param2=param2,
                        minRadius=min_radius, maxRadius=max_radius,
                    )
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        if len(circles) > 0:
                            best = circles[0]  # Take the first (strongest)
                            cx, cy, r = best[0], best[1], best[2]
                            confidence = min(1.0, param2 / 60.0)
                            return SolarDisk(
                                center_x=cx, center_y=cy, radius_px=r,
                                image_width=w, image_height=h,
                                detection_confidence=confidence,
                            )
        return None

    def _detect_contour(self, image: np.ndarray) -> Optional[SolarDisk]:
        """Detect solar disk by finding the largest contour and fitting a circle."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Threshold to separate bright disk from dark background
        blurred = cv2.GaussianBlur(gray, (15, 15), 3)

        # Try different threshold methods
        methods = [
            ("otsu", lambda: cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("triangle", lambda: cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[1]),
        ]

        best_disk = None
        best_confidence = 0.0

        for method_name, method_fn in methods:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.debug_images[f"binary_{method_name}"] = binary

            # Morphological closing to fill holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.debug_images[f"closed_{method_name}"] = closed

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Find the largest contour (presumably the solar disk)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area < w * h * 0.1:  # Too small to be the solar disk
                continue

            # Fit a minimum enclosing circle
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            cx, cy, r = float(cx), float(cy), float(r)

            # Check how well the contour matches a circle
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Record contour match quality for debug
            confidence = min(1.0, circularity * 0.8 + 0.2)

            if confidence > best_confidence:
                best_confidence = confidence
                best_disk = SolarDisk(
                    center_x=cx, center_y=cy, radius_px=r,
                    image_width=w, image_height=h,
                    detection_confidence=confidence,
                )

        return best_disk

    def _detect_intensity_profile(self, image: np.ndarray) -> Optional[SolarDisk]:
        """Fallback: estimate disk from image intensity profile."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Assume disk center is at image center
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.45  # Assume disk fills ~90% of the smaller dimension

        return SolarDisk(
            center_x=cx, center_y=cy, radius_px=r,
            image_width=w, image_height=h,
            detection_confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Sunspot Detection
# ---------------------------------------------------------------------------

class SunspotDetector:
    """Detect sunspots within the solar disk region.

    Uses adaptive thresholding to handle varying contrast conditions,
    with morphological filtering to isolate genuine sunspot features
    from noise and limb darkening artifacts.
    """

    # Region thresholds (as fraction of solar disk radius)
    CENTER_THRESHOLD = 0.35    # Central region: |distance| < 0.35 * R
    MID_THRESHOLD = 0.70       # Mid-latitude: 0.35 <= |distance| < 0.70 * R
    # Limb: |distance| >= 0.70 * R

    # Limb edge buffer (fraction of disk radius to search beyond detected edge)
    LIMB_BUFFER_MIN = 0.05     # 5% beyond disk edge
    LIMB_BUFFER_MAX = 0.15     # 15% beyond disk edge

    def __init__(
        self,
        min_spot_area: float = 15.0,       # Minimum spot area in pixels
        max_spot_area: float = 50000.0,    # Maximum spot area in pixels
        adaptive_block_size: int = 51,      # Block size for adaptive thresholding
        adaptive_c: float = -8.0,          # C parameter for adaptive threshold
        use_morphology: bool = True,
    ):
        self.min_spot_area = min_spot_area
        self.max_spot_area = max_spot_area
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.use_morphology = use_morphology
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect(
        self,
        image: np.ndarray,
        disk: SolarDisk,
    ) -> List[Sunspot]:
        """Detect sunspots within the solar disk region.

        Args:
            image: Input image (BGR or grayscale)
            disk: Detected solar disk boundary

        Returns:
            List of detected Sunspot objects
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        cx, cy, r = disk.center_x, disk.center_y, disk.radius_px

        # Create a mask for the solar disk (+ limb buffer zone)
        # The buffer extends beyond the detected disk edge to catch limb sunspots
        buffer_px = int(r * self.LIMB_BUFFER_MAX)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r) + buffer_px, 255, -1)
        self.debug_images["disk_mask"] = mask

        # --- Multiple threshold strategies ---

        candidates = []

        # Strategy 1: Adaptive Gaussian thresholding (best for varying contrast)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_c,
        )
        adaptive_masked = cv2.bitwise_and(adaptive, mask)
        self.debug_images["adaptive_thresh"] = adaptive_masked

        candidates.extend(self._extract_contours(adaptive_masked, "adaptive", gray))

        # Strategy 2: Otsu thresholding (good for high-contrast images)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_masked = cv2.bitwise_and(otsu, mask)
        self.debug_images["otsu_thresh"] = otsu_masked

        candidates.extend(self._extract_contours(otsu_masked, "otsu", gray))

        # Strategy 3: Local contrast enhancement for faint spots
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(15, 15))
        enhanced = clahe.apply(gray)
        _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        enhanced_masked = cv2.bitwise_and(enhanced_thresh, mask)
        self.debug_images["enhanced_thresh"] = enhanced_masked

        candidates.extend(self._extract_contours(enhanced_masked, "clahe", gray))

        # Merge overlapping candidates (same spot detected by multiple strategies)
        merged = self._merge_candidates(candidates)

        # Classify by region
        for spot in merged:
            spot.region = self.classify_spot_region(spot, disk)
            # Convert distance to normalized (fraction of disk radius)
            dx = spot.position_x - cx
            dy = spot.position_y - cy
            spot.distance_from_center = math.sqrt(dx * dx + dy * dy) / r

        # Sort by confidence (highest first)
        merged.sort(key=lambda s: s.confidence, reverse=True)

        # Re-assign IDs
        for i, spot in enumerate(merged):
            spot.spot_id = i + 1

        return merged

    def _extract_contours(
        self,
        binary: np.ndarray,
        source: str,
        gray: np.ndarray,
    ) -> List[Sunspot]:
        """Extract sunspot candidates from a binary mask using contour analysis."""
        disk_center = None  # Will be resolved later

        if self.use_morphology:
            # Clean up noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Remove small noise dots
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

            # Close small gaps within spots
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_med)
        else:
            cleaned = binary

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_spot_area or area > self.max_spot_area:
                continue

            # Compute moments for centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Equivalent diameter
            eq_diam = math.sqrt(4 * area / math.pi)

            # Perimeter
            perimeter = cv2.arcLength(contour, True)

            # Circularity
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Mean intensity inside the contour
            mask_spot = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_spot, [contour], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask_spot)[0]

            # Confidence based on contour quality and intensity
            size_factor = min(1.0, area / 500)
            conf = min(1.0, circularity * 0.3 + size_factor * 0.4 +
                       (1.0 - mean_intensity / 255) * 0.3)

            spots.append(Sunspot(
                position_x=cx, position_y=cy,
                area_px=area, equivalent_diameter_px=eq_diam,
                perimeter_px=perimeter, circularity=circularity,
                mean_intensity=mean_intensity, confidence=conf,
                bbox=(x, y, bw, bh),
            ))

        return spots

    def _merge_candidates(self, candidates: List[Sunspot]) -> List[Sunspot]:
        """Merge overlapping candidates detected by multiple strategies.

        Uses a simple distance-based clustering: if two spots are within
        a threshold distance, keep the one with higher confidence.
        """
        if not candidates:
            return []

        # Sort by confidence descending
        candidates.sort(key=lambda s: s.confidence, reverse=True)

        merged = []
        kept = [True] * len(candidates)

        for i in range(len(candidates)):
            if not kept[i]:
                continue
            ref = candidates[i]
            merged.append(ref)
            for j in range(i + 1, len(candidates)):
                if not kept[j]:
                    continue
                other = candidates[j]
                dx = ref.position_x - other.position_x
                dy = ref.position_y - other.position_y
                dist = math.sqrt(dx * dx + dy * dy)
                # Merge if centroids are within 10 pixels
                if dist < 10:
                    kept[j] = False
                    # Merge: refine position
                    total_conf = ref.confidence + other.confidence
                    if total_conf > 0:
                        ref.position_x = (
                            ref.position_x * ref.confidence +
                            other.position_x * other.confidence
                        ) / total_conf
                        ref.position_y = (
                            ref.position_y * ref.confidence +
                            other.position_y * other.confidence
                        ) / total_conf
                        ref.area_px += other.area_px
                    ref.confidence = max(ref.confidence, other.confidence)

        return merged

    def classify_spot_region(self, spot: Sunspot, disk: SolarDisk) -> str:
        """Classify a sunspot by region relative to the solar disk."""
        dx = spot.position_x - disk.center_x
        dy = spot.position_y - disk.center_y
        dist = math.sqrt(dx * dx + dy * dy) / disk.radius_px

        if dist < self.CENTER_THRESHOLD:
            return "center"
        elif dist < self.MID_THRESHOLD:
            return "mid_latitude"
        else:
            return "limb"

    def debug_annotate(self, image: np.ndarray, disk: SolarDisk,
                       spots: List[Sunspot]) -> np.ndarray:
        """Generate a debug annotation image showing detection results."""
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        h, w = vis.shape[:2]
        cx, cy, r = int(disk.center_x), int(disk.center_y), int(disk.radius_px)

        # Draw solar disk boundary
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 2)  # Yellow = disk edge
        cv2.circle(vis, (cx, cy), int(r * 0.15) + r, (255, 165, 0), 1, cv2.LINE_AA)  # Orange = limb search zone

        # Region boundaries (dashed)
        for frac, color in [(0.35, (100, 200, 100)), (0.70, (100, 100, 200))]:
            rr = int(r * frac)
            cv2.circle(vis, (cx, cy), rr, color, 1, cv2.LINE_AA)

        # Disk center
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

        # Draw each sunspot
        for spot in spots:
            sx, sy = int(spot.position_x), int(spot.position_y)

            # Color by region
            region_colors = {
                "center": (0, 255, 0),       # Green
                "mid_latitude": (255, 255, 0),  # Cyan
                "limb": (0, 165, 255),        # Orange
            }
            color = region_colors.get(spot.region, (0, 0, 255))

            # Draw spot boundary
            er = int(spot.equivalent_diameter_px / 2)
            cv2.circle(vis, (sx, sy), max(er, 2), color, 1)
            cv2.drawMarker(vis, (sx, sy), color, cv2.MARKER_CROSS, 6, 1)

            # Spot ID label
            label = f"S{spot.spot_id}"
            cv2.putText(vis, label, (sx + er + 3, sy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Info overlay
        info_lines = [
            f"Solar Disk: C({cx},{cy}) R={r}px",
            f"Sunspots: {len(spots)}",
            f"Center: {sum(1 for s in spots if s.region == 'center')}",
            f"Mid-Lat: {sum(1 for s in spots if s.region == 'mid_latitude')}",
            f"Limb: {sum(1 for s in spots if s.region == 'limb')}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return vis


# ---------------------------------------------------------------------------
# Main Detection Pipeline
# ---------------------------------------------------------------------------

class SunspotDetectionPipeline:
    """Complete sunspot detection pipeline combining disk detection and spot extraction."""

    def __init__(self):
        self.disk_detector = SolarDiskDetector()
        self.spot_detector = SunspotDetector()

    def process(self, image_path: str, **params: Any) -> SunspotDetectionResult:
        """Run the full sunspot detection pipeline on an image.

        Args:
            image_path: Path to the solar image file
            **params: Override default detector parameters

        Returns:
            SunspotDetectionResult with complete detection data
        """
        start_time = datetime.now()

        # Load image (use imdecode for Unicode path support)
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        h, w = image.shape[:2]

        # Update detector parameters if provided
        if "min_spot_area" in params:
            self.spot_detector.min_spot_area = params["min_spot_area"]
        if "max_spot_area" in params:
            self.spot_detector.max_spot_area = params["max_spot_area"]
        if "adaptive_block_size" in params:
            self.spot_detector.adaptive_block_size = params["adaptive_block_size"]
        if "adaptive_c" in params:
            self.spot_detector.adaptive_c = params["adaptive_c"]

        # Step 1: Detect solar disk
        disk = self.disk_detector.detect(image)
        if disk is None:
            raise ValueError("无法检测到日面边界，请确认图像包含完整的太阳盘面")

        # Step 2: Detect sunspots
        spots = self.spot_detector.detect(image, disk)

        # Step 3: Classify each spot by region
        for spot in spots:
            spot.region = self.spot_detector.classify_spot_region(spot, disk)

        # Step 4: Aggregate results
        processing_ms = (datetime.now() - start_time).total_seconds() * 1000

        spots_by_region = {
            "center": sum(1 for s in spots if s.region == "center"),
            "mid_latitude": sum(1 for s in spots if s.region == "mid_latitude"),
            "limb": sum(1 for s in spots if s.region == "limb"),
        }

        result = SunspotDetectionResult(
            solar_disk=disk,
            sunspots=spots,
            total_spots=len(spots),
            spots_by_region=spots_by_region,
            processing_time_ms=processing_ms,
            image_path=image_path,
            detection_timestamp=datetime.now().isoformat(),
            algorithm_params={
                "min_spot_area": self.spot_detector.min_spot_area,
                "max_spot_area": self.spot_detector.max_spot_area,
                "adaptive_block_size": self.spot_detector.adaptive_block_size,
                "adaptive_c": self.spot_detector.adaptive_c,
                "limb_buffer": [self.spot_detector.LIMB_BUFFER_MIN, self.spot_detector.LIMB_BUFFER_MAX],
            },
        )

        return result

    def process_with_annotation(self, image_path: str,
                                output_dir: Optional[str] = None,
                                **params: Any) -> Tuple[SunspotDetectionResult, Optional[np.ndarray]]:
        """Run detection and return both results and annotated visualization."""
        result = self.process(image_path, **params)

        # Load image with Unicode path support
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return result, None

        disk = self.disk_detector.detect(image)
        if disk:
            # Use the re-detected disk for annotation
            spots = self.spot_detector.detect(image, disk)
            for spot in spots:
                spot.region = self.spot_detector.classify_spot_region(spot, disk)
                # Match with result spots for IDs
                for rs in result.sunspots:
                    if abs(spot.position_x - rs.position_x) < 5 and \
                       abs(spot.position_y - rs.position_y) < 5:
                        spot.spot_id = rs.spot_id
                        break

            annotated = self.spot_detector.debug_annotate(image, disk, spots)
        else:
            annotated = image.copy()

        # Save if output_dir provided
        if output_dir:
            out_path = Path(output_dir) / f"sunspot_{Path(image_path).stem}_annotated.png"
            cv2.imwrite(str(out_path), annotated)
            result.debug_info["annotated_path"] = str(out_path)

        return result, annotated


# ---------------------------------------------------------------------------
# Utility: generate CSV summary
# ---------------------------------------------------------------------------

def detection_to_csv(result: SunspotDetectionResult) -> str:
    """Format detection result as CSV string."""
    lines = ["# 太阳黑子检测报告"]
    lines.append(f"# 检测时间: {result.detection_timestamp}")
    lines.append(f"# 图像路径: {result.image_path}")
    lines.append(f"# 处理耗时: {result.processing_time_ms:.1f}ms")
    lines.append("")

    if result.solar_disk:
        d = result.solar_disk
        lines.append("# 日面边界信息")
        lines.append(f"# 中心坐标(像素): ({d.center_x:.1f}, {d.center_y:.1f})")
        lines.append(f"# 半径(像素): {d.radius_px:.1f}")
        lines.append("")

    lines.append("ID,Position_X_px,Position_Y_px,PosX_Normalized,PosY_Normalized,"
                 "Area_px,Diameter_px,Circularity,Mean_Intensity,"
                 "Confidence,Region,Distance_From_Center")
    for spot in result.sunspots:
        d = spot.to_dict(result.solar_disk) if result.solar_disk else {}
        pos_n = d.get("position_solar_normalized", {})
        lines.append(
            f"{spot.spot_id},{spot.position_x:.2f},{spot.position_y:.2f},"
            f"{pos_n.get('x', 0):.4f},{pos_n.get('y', 0):.4f},"
            f"{spot.area_px:.2f},{spot.equivalent_diameter_px:.2f},"
            f"{spot.circularity:.4f},{spot.mean_intensity:.2f},"
            f"{spot.confidence:.4f},{spot.region},{spot.distance_from_center:.4f}"
        )

    return "\n".join(lines)
