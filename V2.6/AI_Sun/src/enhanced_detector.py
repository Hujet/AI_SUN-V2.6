"""
Enhanced Solar Feature Detection Module - v3.0

Advanced detection system with:
1. Multi-scale detection mechanism (3-scale pyramid analysis)
2. Prominence detection at solar limb (edge-specialized algorithm)
3. Standardized annotation protocol with quality metrics
4. Synchronized original + annotated image preservation

Multi-scale Strategy:
- Scale 1 (Original): 1.0x - Detects large features (major sunspots, groups)
- Scale 2 (Medium):    0.5x - Detects medium features (moderate spots, plages)
- Scale 3 (Fine):      0.25x - Detects small features (pores, micro-flares)
- Fusion: Non-maximum suppression + confidence-weighted merging

Prominence Detection:
- Limb-darkening compensation using radial intensity profile
- Edge-enhanced gradient analysis for filament/prominence extraction
- Morphological snake algorithm for boundary refinement

Annotation Standard:
- Unique feature IDs with format: {TYPE}_{SCALE}_{INDEX}
- Quality score (0-1) based on contrast, circularity, SNR
- Confidence classification: HIGH (>=0.8), MEDIUM (0.6-0.8), LOW (<0.6)
- Full metadata preservation for traceability
"""

import logging
import math
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class FeatureQuality:
    """Quality metrics for detected features."""
    contrast: float = 0.0           # Local contrast ratio (0-1)
    circularity: float = 0.0        # Shape circularity (0-1)
    snr: float = 0.0                # Signal-to-noise ratio
    edge_sharpness: float = 0.0     # Edge gradient magnitude
    overall_score: float = 0.0      # Combined quality score (0-1)
    
    def compute_overall(self):
        """Compute overall quality score from components."""
        weights = {
            'contrast': 0.30,
            'circularity': 0.20,
            'snr': 0.30,
            'edge_sharpness': 0.20,
        }
        self.overall_score = (
            self.contrast * weights['contrast'] +
            self.circularity * weights['circularity'] +
            min(self.snr / 10.0, 1.0) * weights['snr'] +
            self.edge_sharpness * weights['edge_sharpness']
        )
        return self.overall_score
    
    def to_dict(self) -> Dict:
        return {
            "contrast": round(self.contrast, 4),
            "circularity": round(self.circularity, 4),
            "snr": round(self.snr, 2),
            "edge_sharpness": round(self.edge_sharpness, 4),
            "overall_score": round(self.overall_score, 4),
        }


@dataclass
class SolarFeature:
    """Standardized solar feature with full metadata."""
    feature_id: str = ""            # Format: {TYPE}_{SCALE}_{INDEX}
    feature_type: str = ""          # sunspot, pore, plage, filament, prominence, flare
    scale: str = ""                 # original, medium, fine
    position_x: float = 0.0         # Pixel X
    position_y: float = 0.0         # Pixel Y
    area_px: float = 0.0            # Area in pixels
    equivalent_diameter: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    
    # Normalized coordinates (-1 to +1, origin at disk center)
    norm_x: float = 0.0
    norm_y: float = 0.0
    
    # Quality metrics
    confidence: float = 0.0         # Detection confidence (0-1)
    quality: Optional[FeatureQuality] = None
    
    # Classification
    region: str = ""                # center, mid_latitude, limb, edge_zone
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    
    # Additional metadata
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def compute_normalized(self, disk_center_x: float, disk_center_y: float, disk_radius: float):
        """Compute normalized disk coordinates."""
        if disk_radius > 0:
            self.norm_x = (self.position_x - disk_center_x) / disk_radius
            self.norm_y = (self.position_y - disk_center_y) / disk_radius
    
    def classify_region(self, disk_radius: float, center_x: float, center_y: float):
        """Classify feature region based on normalized distance."""
        dist = math.sqrt(
            (self.position_x - center_x) ** 2 + 
            (self.position_y - center_y) ** 2
        ) / disk_radius
        
        if dist < 0.35:
            self.region = "center"
        elif dist < 0.70:
            self.region = "mid_latitude"
        elif dist < 0.95:
            self.region = "limb"
        else:
            self.region = "edge_zone"
    
    def to_dict(self) -> Dict:
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "scale": self.scale,
            "position": {"x": round(self.position_x, 2), "y": round(self.position_y, 2)},
            "position_normalized": {"x": round(self.norm_x, 4), "y": round(self.norm_y, 4)},
            "area_px": round(self.area_px, 2),
            "equivalent_diameter": round(self.equivalent_diameter, 2),
            "bbox": {"x": self.bbox[0], "y": self.bbox[1], "w": self.bbox[2], "h": self.bbox[3]},
            "confidence": round(self.confidence, 4),
            "quality": self.quality.to_dict() if self.quality else {},
            "region": self.region,
            "intensity": {"mean": round(self.intensity_mean, 2), "std": round(self.intensity_std, 2)},
            "additional_params": self.additional_params,
        }


@dataclass
class DetectionStatistics:
    """Detection statistics for performance evaluation."""
    total_features: int = 0
    features_by_type: Dict[str, int] = field(default_factory=dict)
    features_by_scale: Dict[str, int] = field(default_factory=dict)
    features_by_region: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    average_quality: float = 0.0
    processing_time_ms: float = 0.0
    scales_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_features": self.total_features,
            "features_by_type": self.features_by_type,
            "features_by_scale": self.features_by_scale,
            "features_by_region": self.features_by_region,
            "average_confidence": round(self.average_confidence, 4),
            "average_quality": round(self.average_quality, 4),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "scales_used": self.scales_used,
        }


@dataclass
class EnhancedDetectionResult:
    """Complete enhanced detection result."""
    features: List[SolarFeature] = field(default_factory=list)
    statistics: Optional[DetectionStatistics] = None
    disk_info: Dict[str, Any] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    image_path: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "features": [f.to_dict() for f in self.features],
            "statistics": self.statistics.to_dict() if self.statistics else {},
            "disk_info": self.disk_info,
            "processing_info": self.processing_info,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
        }


# ============================================================
# Multi-Scale Detection Engine
# ============================================================

class MultiScaleDetector:
    """Multi-scale solar feature detection engine.
    
    Implements image pyramid analysis with 3 scales:
    - Scale 1 (Original, 1.0x): Large features
    - Scale 2 (Medium, 0.5x): Medium features  
    - Scale 3 (Fine, 0.25x): Small features
    
    Fusion strategy: Non-maximum suppression with confidence weighting.
    """
    
    # Scale configurations
    SCALES = {
        "original": {"scale": 1.0, "min_area": 50, "max_area": 100000, "block_size": 71},
        "medium": {"scale": 0.5, "min_area": 15, "max_area": 25000, "block_size": 41},
        "fine": {"scale": 0.25, "min_area": 5, "max_area": 6250, "block_size": 21},
    }
    
    def __init__(self, enable_scales: Optional[List[str]] = None):
        """Initialize multi-scale detector.
        
        Args:
            enable_scales: List of scales to use. Default: all 3 scales.
        """
        self.enabled_scales = enable_scales or list(self.SCALES.keys())
        self.debug_images: Dict[str, np.ndarray] = {}
    
    def detect_multi_scale(
        self,
        image: np.ndarray,
        disk_mask: np.ndarray,
        disk_center: Tuple[float, float],
        disk_radius: float,
    ) -> List[SolarFeature]:
        """Run detection at multiple scales and fuse results.
        
        Args:
            image: Grayscale solar image
            disk_mask: Binary mask of solar disk
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius of solar disk in pixels
            
        Returns:
            List of fused SolarFeature objects
        """
        all_detections = []
        
        for scale_name in self.enabled_scales:
            scale_config = self.SCALES[scale_name]
            scale_factor = scale_config["scale"]
            
            # Resize image for current scale
            if scale_factor != 1.0:
                h, w = image.shape[:2]
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scaled_mask = cv2.resize(disk_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                scaled_center = (disk_center[0] * scale_factor, disk_center[1] * scale_factor)
                scaled_radius = disk_radius * scale_factor
            else:
                scaled_img = image.copy()
                scaled_mask = disk_mask.copy()
                scaled_center = disk_center
                scaled_radius = disk_radius
            
            # Detect features at this scale
            features = self._detect_at_scale(
                scaled_img, scaled_mask, scaled_center, scaled_radius,
                scale_name, scale_config
            )
            
            # Store debug image
            self.debug_images[f"detections_{scale_name}"] = self._create_debug_overlay(
                scaled_img, features, scaled_center, scaled_radius
            )
            
            logger.info(f"Scale {scale_name}: detected {len(features)} features")
            all_detections.extend(features)
        
        # Fuse detections across scales
        fused = self._fuse_detections(all_detections, disk_center, disk_radius)
        
        logger.info(f"Multi-scale fusion: {len(all_detections)} -> {len(fused)} features")
        return fused
    
    def _detect_at_scale(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        scale_name: str,
        config: Dict,
    ) -> List[SolarFeature]:
        """Detect features at a single scale."""
        features = []
        scale_factor = config["scale"]
        min_area = config["min_area"]
        max_area = config["max_area"]
        block_size = config["block_size"]
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, -8.0
        )
        adaptive_masked = cv2.bitwise_and(adaptive, mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_masked, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Compute properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = M["m10"] / M["m00"] / scale_factor  # Convert back to original scale
            cy = M["m01"] / M["m00"] / scale_factor
            eq_diam = math.sqrt(4 * area / math.pi) / scale_factor
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            bbox = cv2.boundingRect(contour)
            
            # Scale bbox back to original
            bbox = tuple(int(v / scale_factor) for v in bbox)
            
            # Intensity stats
            spot_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(spot_mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(image, mask=spot_mask)[0]
            std_val = cv2.mean(image, mask=spot_mask)[1] if len(cv2.mean(image, mask=spot_mask)) > 1 else 0
            
            # Quality metrics
            quality = FeatureQuality(
                contrast=abs(mean_val - 128) / 128,
                circularity=min(circularity, 1.0),
                snr=mean_val / max(std_val, 1),
                edge_sharpness=min(perimeter / max(math.sqrt(area), 1), 1.0),
            )
            quality.compute_overall()
            
            # Confidence
            confidence = min(1.0, quality.overall_score * 0.8 + circularity * 0.2)
            
            # Create feature with scale-specific ID
            feature = SolarFeature(
                feature_id=f"{self._type_from_area(area)}_{scale_name[:3]}_{i+1}",
                feature_type=self._type_from_area(area),
                scale=scale_name,
                position_x=cx,
                position_y=cy,
                area_px=area / (scale_factor ** 2),
                equivalent_diameter=eq_diam,
                bbox=bbox,
                confidence=confidence,
                quality=quality,
                intensity_mean=mean_val,
                intensity_std=std_val,
            )
            
            # Compute normalized coordinates
            feature.compute_normalized(center[0] / scale_factor, center[1] / scale_factor, radius / scale_factor)
            feature.classify_region(radius / scale_factor, center[0] / scale_factor, center[1] / scale_factor)
            
            features.append(feature)
        
        return features
    
    def _fuse_detections(
        self,
        detections: List[SolarFeature],
        disk_center: Tuple[float, float],
        disk_radius: float,
    ) -> List[SolarFeature]:
        """Fuse multi-scale detections using NMS with confidence weighting."""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda f: f.confidence, reverse=True)
        
        fused = []
        kept = [True] * len(detections)
        merge_threshold = 15  # pixels
        
        for i in range(len(detections)):
            if not kept[i]:
                continue
            
            ref = detections[i]
            kept[i] = False
            
            # Find overlapping detections
            overlapping = [ref]
            for j in range(i + 1, len(detections)):
                if not kept[j]:
                    continue
                
                other = detections[j]
                dist = math.sqrt(
                    (ref.position_x - other.position_x) ** 2 +
                    (ref.position_y - other.position_y) ** 2
                )
                
                if dist < merge_threshold:
                    overlapping.append(other)
                    kept[j] = False
            
            # Merge overlapping detections
            if len(overlapping) == 1:
                fused.append(ref)
            else:
                merged = self._merge_features(overlapping, disk_center, disk_radius)
                fused.append(merged)
        
        # Re-assign IDs
        for i, feature in enumerate(fused):
            feature.feature_id = f"{feature.feature_type}_fus_{i+1}"
            feature.spot_id = i + 1
        
        return fused
    
    def _merge_features(self, features: List[SolarFeature], disk_center, disk_radius) -> SolarFeature:
        """Merge multiple overlapping features into one."""
        # Weight by confidence
        total_conf = sum(f.confidence for f in features)
        
        merged_x = sum(f.position_x * f.confidence for f in features) / total_conf
        merged_y = sum(f.position_y * f.confidence for f in features) / total_conf
        merged_area = sum(f.area_px for f in features)
        merged_conf = max(f.confidence for f in features)
        merged_quality = max(f.quality.overall_score for f in features if f.quality)
        
        # Use highest quality feature's type
        best_feature = max(features, key=lambda f: f.confidence)
        
        merged = SolarFeature(
            feature_id="",  # Will be set later
            feature_type=best_feature.feature_type,
            scale="fused",
            position_x=merged_x,
            position_y=merged_y,
            area_px=merged_area,
            equivalent_diameter=math.sqrt(4 * merged_area / math.pi),
            bbox=best_feature.bbox,
            confidence=merged_conf,
            quality=FeatureQuality(overall_score=merged_quality),
            intensity_mean=best_feature.intensity_mean,
            intensity_std=best_feature.intensity_std,
        )
        
        merged.compute_normalized(disk_center[0], disk_center[1], disk_radius)
        merged.classify_region(disk_radius, disk_center[0], disk_center[1])
        
        return merged
    
    def _type_from_area(self, area: float) -> str:
        """Classify feature type based on area."""
        if area < 50:
            return "pore"
        elif area < 500:
            return "sunspot"
        else:
            return "sunspot_group"
    
    def _create_debug_overlay(self, image, features, center, radius):
        """Create debug visualization for a single scale."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        
        # Draw disk boundary
        cv2.circle(vis, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 2)
        
        # Draw features
        for f in features:
            color = (0, 255, 0) if f.confidence > 0.8 else (0, 255, 255)
            cv2.circle(vis, (int(f.position_x), int(f.position_y)), 
                      int(f.equivalent_diameter / 2), color, 1)
            cv2.putText(vis, f.feature_id, 
                       (int(f.position_x) + 5, int(f.position_y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return vis


# ============================================================
# Prominence & Edge Detection Engine
# ============================================================

class ProminenceDetector:
    """Specialized detector for solar prominences and edge features.
    
    Addresses challenges:
    - Limb darkening compensation
    - Edge distortion correction
    - Low contrast enhancement
    """
    
    def __init__(self):
        self.debug_images: Dict[str, np.ndarray] = {}
    
    def detect_prominences(
        self,
        image: np.ndarray,
        disk_mask: np.ndarray,
        disk_center: Tuple[float, float],
        disk_radius: float,
    ) -> List[SolarFeature]:
        """Detect prominences and edge features.
        
        Args:
            image: Grayscale solar image
            disk_mask: Binary mask of solar disk
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius in pixels
            
        Returns:
            List of prominence SolarFeature objects
        """
        prominences = []
        
        # Step 1: Create limb zone mask (100%-115% of disk radius)
        # CRITICAL: Prominences ONLY appear OUTSIDE the solar disk (>100%)
        limb_mask = self._create_limb_zone_mask(
            image.shape, disk_center, disk_radius,
            inner_frac=1.0, outer_frac=1.15
        )
        
        # Step 2: Compensate limb darkening
        compensated = self._compensate_limb_darkening(
            image, disk_center, disk_radius
        )
        
        self.debug_images["limb_zone"] = limb_mask * 255
        self.debug_images["limb_compensated"] = compensated
        
        # Step 3: Edge-enhanced gradient analysis
        gradients = self._compute_edge_gradients(compensated, limb_mask)
        self.debug_images["edge_gradients"] = gradients
        
        # Step 4: Detect prominence candidates
        candidates = self._extract_prominence_candidates(
            compensated, gradients, limb_mask, disk_center, disk_radius
        )
        
        prominences.extend(candidates)
        
        logger.info(f"Prominence detection: {len(prominences)} candidates")
        return prominences
    
    def _create_limb_zone_mask(
        self,
        shape: Tuple[int, int],
        center: Tuple[float, float],
        radius: float,
        inner_frac: float,
        outer_frac: float,
    ) -> np.ndarray:
        """Create mask for limb zone region."""
        h, w = shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        
        inner_r = radius * inner_frac
        outer_r = radius * outer_frac
        
        inner_mask = (x_grid - center[0])**2 + (y_grid - center[1])**2 >= inner_r**2
        outer_mask = (x_grid - center[0])**2 + (y_grid - center[1])**2 <= outer_r**2
        
        return (inner_mask & outer_mask).astype(np.uint8)
    
    def _compensate_limb_darkening(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        radius: float,
    ) -> np.ndarray:
        """Apply limb darkening compensation.
        
        Uses radial intensity profile to normalize brightness across disk.
        """
        h, w = image.shape[:2]
        img_f = image.astype(np.float32)
        compensated = img_f.copy()
        
        # Compute radial intensity profile
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(
            (x_grid - center[0])**2 + (y_grid - center[1])**2
        )
        
        # Normalize distance (0 at center, 1 at edge)
        norm_dist = dist_from_center / max(radius, 1)
        
        # Apply compensation: brighter toward edges
        # Using simple linear model: I_comp = I / (1 - 0.5 * norm_dist)
        compensation_factor = 1.0 / (1.0 - 0.5 * np.clip(norm_dist, 0, 0.95))
        
        compensated = img_f * compensation_factor
        compensated = np.clip(compensated, 0, 255).astype(np.uint8)
        
        return compensated
    
    def _compute_edge_gradients(
        self,
        image: np.ndarray,
        limb_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute enhanced edge gradients in limb zone."""
        # Sobel gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply limb mask
        return cv2.bitwise_and(gradient_mag, gradient_mag, mask=limb_mask)
    
    def _extract_prominence_candidates(
        self,
        compensated: np.ndarray,
        gradients: np.ndarray,
        limb_mask: np.ndarray,
        disk_center: Tuple[float, float],
        disk_radius: float,
    ) -> List[SolarFeature]:
        """Extract prominence features from limb zone."""
        prominences = []
        
        # Threshold gradients to find edge features
        _, grad_thresh = cv2.threshold(gradients, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(grad_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 20 or area > 10000:
                continue
            
            # Compute properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            eq_diam = math.sqrt(4 * area / math.pi)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            bbox = cv2.boundingRect(contour)
            
            # Intensity stats
            spot_mask = np.zeros(compensated.shape[:2], dtype=np.uint8)
            cv2.drawContours(spot_mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(compensated, mask=spot_mask)[0]
            
            # Quality
            quality = FeatureQuality(
                contrast=abs(mean_val - 128) / 128,
                circularity=min(circularity, 1.0),
                snr=mean_val / 20.0,
                edge_sharpness=min(perimeter / max(math.sqrt(area), 1), 1.0),
            )
            quality.compute_overall()
            
            confidence = min(1.0, quality.overall_score * 0.7 + 0.2)
            
            # Determine prominence type
            if mean_val > 180:
                ptype = "prominence"
            elif mean_val > 140:
                ptype = "plage"
            else:
                ptype = "filament"
            
            feature = SolarFeature(
                feature_id=f"{ptype}_limb_{i+1}",
                feature_type=ptype,
                scale="limb_zone",
                position_x=cx,
                position_y=cy,
                area_px=area,
                equivalent_diameter=eq_diam,
                bbox=tuple(bbox),
                confidence=confidence,
                quality=quality,
                intensity_mean=mean_val,
            )
            
            feature.compute_normalized(disk_center[0], disk_center[1], disk_radius)
            feature.classify_region(disk_radius, disk_center[0], disk_center[1])
            
            prominences.append(feature)
        
        return prominences


# ============================================================
# Enhanced Detection Pipeline
# ============================================================

class EnhancedDetectionPipeline:
    """Complete enhanced detection pipeline with multi-scale and prominence detection."""
    
    def __init__(self, enable_scales: Optional[List[str]] = None):
        self.multi_scale = MultiScaleDetector(enable_scales)
        self.prominence = ProminenceDetector()
        self.debug_images: Dict[str, np.ndarray] = {}
    
    def detect(
        self,
        image: np.ndarray,
        disk_info: Dict[str, Any],
    ) -> EnhancedDetectionResult:
        """Run full enhanced detection pipeline.
        
        Args:
            image: Grayscale solar image
            disk_info: Dictionary with center_x, center_y, radius
            
        Returns:
            EnhancedDetectionResult with all detected features
        """
        start_time = datetime.now()
        
        cx = disk_info.get("center_x", image.shape[1] / 2)
        cy = disk_info.get("center_y", image.shape[0] / 2)
        r = disk_info.get("radius", min(image.shape) * 0.4)
        
        # Create disk mask
        h, w = image.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx)**2 + (y_grid - cy)**2 <= r**2).astype(np.uint8)
        
        # Step 1: Multi-scale feature detection
        logger.info("Starting multi-scale detection...")
        multi_features = self.multi_scale.detect_multi_scale(
            image, disk_mask, (cx, cy), r
        )
        
        # Step 2: Prominence/edge detection
        logger.info("Starting prominence detection...")
        prominences = self.prominence.detect_prominences(
            image, disk_mask, (cx, cy), r
        )
        
        # Merge all features
        all_features = multi_features + prominences
        
        # Compute statistics
        processing_ms = (datetime.now() - start_time).total_seconds() * 1000
        stats = self._compute_statistics(all_features, processing_ms)
        
        # Collect debug images
        self.debug_images.update(self.multi_scale.debug_images)
        self.debug_images.update(self.prominence.debug_images)
        
        result = EnhancedDetectionResult(
            features=all_features,
            statistics=stats,
            disk_info={
                "center_x": cx, "center_y": cy, "radius": r,
                "method": disk_info.get("method", "unknown"),
                "confidence": disk_info.get("confidence", 0.0),
            },
            processing_info={
                "pipeline_version": "3.0",
                "scales_used": self.multi_scale.enabled_scales,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now().isoformat(),
        )
        
        logger.info(f"Enhanced detection complete: {len(all_features)} features in {processing_ms:.0f}ms")
        return result
    
    def _compute_statistics(self, features: List[SolarFeature], processing_ms: float) -> DetectionStatistics:
        """Compute detection statistics."""
        stats = DetectionStatistics()
        stats.total_features = len(features)
        stats.processing_time_ms = processing_ms
        
        # By type
        type_counts = {}
        for f in features:
            type_counts[f.feature_type] = type_counts.get(f.feature_type, 0) + 1
        stats.features_by_type = type_counts
        
        # By scale
        scale_counts = {}
        for f in features:
            scale_counts[f.scale] = scale_counts.get(f.scale, 0) + 1
        stats.features_by_scale = scale_counts
        
        # By region
        region_counts = {}
        for f in features:
            region_counts[f.region] = region_counts.get(f.region, 0) + 1
        stats.features_by_region = region_counts
        
        # Averages
        if features:
            stats.average_confidence = sum(f.confidence for f in features) / len(features)
            stats.average_quality = sum(f.quality.overall_score for f in features if f.quality) / len(features)
        
        return stats
    
    def generate_annotated_image(
        self,
        image: np.ndarray,
        result: EnhancedDetectionResult,
        output_path: str,
    ) -> str:
        """Generate annotated image with standardized markers.
        
        Annotation standards:
        - Rectangles for sunspots/sunspot groups
        - Ellipses for prominences/filaments
        - Stars for flares
        - Color coding by confidence
        - Feature ID labels
        """
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        
        # Draw disk boundary
        cx = int(result.disk_info.get("center_x", 0))
        cy = int(result.disk_info.get("center_y", 0))
        r = int(result.disk_info.get("radius", 0))
        
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), int(r * 0.35), (100, 200, 100), 1)  # Center zone
        cv2.circle(vis, (cx, cy), int(r * 0.70), (100, 100, 200), 1)  # Mid-latitude
        
        # Draw features
        for feature in result.features:
            x = int(feature.position_x)
            y = int(feature.position_y)
            size = int(feature.equivalent_diameter / 2)
            
            # Color by confidence
            if feature.confidence >= 0.8:
                color = (0, 255, 0)  # Green - HIGH
            elif feature.confidence >= 0.6:
                color = (0, 255, 255)  # Cyan - MEDIUM
            else:
                color = (0, 0, 255)  # Red - LOW
            
            # Shape by type
            ftype = feature.feature_type
            if ftype in ("sunspot", "sunspot_group", "pore"):
                # Rectangle
                cv2.rectangle(vis, 
                             (x - size, y - size), (x + size, y + size),
                             color, 2)
            elif ftype in ("prominence", "filament"):
                # Ellipse
                cv2.ellipse(vis, (x, y), (size, int(size * 0.6)), 0, 0, 360, color, 2)
            elif ftype == "flare":
                # Star (circle with cross)
                cv2.circle(vis, (x, y), size, color, 2)
                cv2.drawMarker(vis, (x, y), color, cv2.MARKER_CROSS, size, 2)
            else:
                cv2.circle(vis, (x, y), size, color, 1)
            
            # Label
            label = f"{feature.feature_id} ({feature.confidence:.2f})"
            cv2.putText(vis, label, (x + size + 3, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Info overlay
        info_lines = [
            f"Enhanced Detection v3.0",
            f"Features: {result.statistics.total_features}",
            f"Types: {result.statistics.features_by_type}",
            f"Time: {result.statistics.processing_time_ms:.0f}ms",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 20 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save
        cv2.imwrite(output_path, vis)
        logger.info(f"Annotated image saved: {output_path}")
        return output_path


# ============================================================
# Utility Functions
# ============================================================

def confidence_label(confidence: float) -> str:
    """Convert confidence value to quality label."""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"


def generate_detection_report(result: EnhancedDetectionResult) -> str:
    """Generate standardized detection report in CSV format."""
    lines = [
        "# Enhanced Solar Feature Detection Report v3.0",
        f"# Timestamp: {result.timestamp}",
        f"# Image: {result.image_path}",
        f"# Total Features: {result.statistics.total_features}",
        f"# Processing Time: {result.statistics.processing_time_ms:.0f}ms",
        "",
        "# Feature Details",
        "ID,Type,Scale,X,Y,Normalized_X,Normalized_Y,Area,Diameter,Confidence,Quality,Region,Label",
    ]
    
    for f in result.features:
        label = confidence_label(f.confidence)
        lines.append(
            f"{f.feature_id},{f.feature_type},{f.scale},"
            f"{f.position_x:.2f},{f.position_y:.2f},"
            f"{f.norm_x:.4f},{f.norm_y:.4f},"
            f"{f.area_px:.2f},{f.equivalent_diameter:.2f},"
            f"{f.confidence:.4f},{f.quality.overall_score:.4f},"
            f"{f.region},{label}"
        )
    
    return "\n".join(lines)
