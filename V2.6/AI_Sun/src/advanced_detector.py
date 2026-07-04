"""
Advanced Solar Feature Detection Module - v4.0

Addresses critical limitations in current detection system:
1. Prominence discrimination from similar structures (edge feature precision)
2. Plage/penumbra faint signal extraction (dynamic threshold optimization)
3. Complex sunspot group segmentation (morphological analysis improvement)
4. Image inhomogeneity handling (illumination, noise, contrast variations)

Technical Approach:
- Prominence: Multi-criteria classification (shape, position, intensity profile, gradient orientation)
- Plage/Penumbra: Multi-scale wavelet decomposition + adaptive local contrast enhancement
- Sunspot Groups: Watershed segmentation + distance transform + morphological snake
- Inhomogeneity: Retinex-based illumination correction + CLAHE + bilateral filtering
"""

import logging
import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# Image Inhomogeneity Correction
# ============================================================

class IlluminationCorrector:
    """Correct illumination non-uniformity using Retinex-based method.
    
    Addresses: vignetting, limb darkening, uneven illumination
    """
    
    @staticmethod
    def correct_retinex(image: np.ndarray, sigma_list: List[int] = [15, 80, 250]) -> np.ndarray:
        """Multi-Scale Retinex (MSR) illumination correction.
        
        I(x,y) = R(x,y) * L(x,y)
        R(x,y) = log[I(x,y)] - log[L(x,y)]
        L(x,y) is estimated by Gaussian blur at multiple scales
        
        Args:
            image: Grayscale input image
            sigma_list: List of Gaussian sigma values for multi-scale
            
        Returns:
            Illumination-corrected image
        """
        img_f = image.astype(np.float64) + 1.0  # Avoid log(0)
        log_img = np.log(img_f)
        
        # Multi-scale estimation of illumination
        msr = np.zeros_like(log_img)
        for sigma in sigma_list:
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            gaussian = cv2.GaussianBlur(log_img, (ksize, ksize), sigma)
            msr += (log_img - gaussian)
        
        msr /= len(sigma_list)
        
        # Normalize to 0-255
        msr_min = np.min(msr)
        msr_max = np.max(msr)
        if msr_max - msr_min > 0:
            corrected = ((msr - msr_min) / (msr_max - msr_min) * 255).astype(np.uint8)
        else:
            corrected = image.copy()
        
        return corrected
    
    @staticmethod
    def correct_limb_darkening(image: np.ndarray, 
                                disk_center: Tuple[float, float],
                                disk_radius: float) -> np.ndarray:
        """Compensate solar limb darkening effect.
        
        I(r) = I(0) * [1 - u * (1 - cos(theta))]
        where r/R = sin(theta)
        
        Args:
            image: Grayscale solar image
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius in pixels
            
        Returns:
            Limb-darkening compensated image
        """
        h, w = image.shape
        img_f = image.astype(np.float64)
        cx, cy = disk_center
        r = disk_radius
        
        # Create distance grid (normalized 0-1)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        norm_dist = np.clip(dist / max(r, 1), 0, 1)
        
        # Limb darkening coefficient (typical for visible light: u ≈ 0.6)
        u = 0.6
        
        # Compute correction factor
        # cos(theta) = sqrt(1 - (r/R)^2)
        cos_theta = np.sqrt(np.clip(1 - norm_dist**2, 0, 1))
        correction = 1.0 / (1 - u * (1 - cos_theta))
        
        # Apply correction only within disk
        disk_mask = dist <= r
        corrected = img_f.copy()
        corrected[disk_mask] *= correction[disk_mask]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    @staticmethod
    def enhance_local_contrast(image: np.ndarray, 
                                clip_limit: float = 2.0,
                                tile_size: int = 8) -> np.ndarray:
        """Adaptive histogram equalization (CLAHE) for local contrast.
        
        Args:
            image: Grayscale input image
            clip_limit: Contrast limiting threshold
            tile_size: Grid size for histogram computation
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        return clahe.apply(image)
    
    @staticmethod
    def denoise_bilateral(image: np.ndarray,
                          d: int = 5,
                          sigma_color: float = 50,
                          sigma_space: float = 50) -> np.ndarray:
        """Bilateral filtering: smooth noise while preserving edges.
        
        Args:
            image: Grayscale input image
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ============================================================
# Prominence Discrimination Engine
# ============================================================

class ProminenceDiscriminator:
    """Distinguish prominences from similar structures (filaments, plages, artifacts).
    
    Classification criteria:
    1. Position: Near limb (distance from center > 0.85 * radius)
    2. Shape: Elongated structures (aspect ratio > 2.0)
    3. Intensity: Brighter than local background
    4. Gradient orientation: Consistent along structure
    5. Texture: Smooth intensity profile along major axis
    """
    
    @dataclass
    class EdgeFeature:
        """Detected edge feature with discrimination metrics."""
        feature_id: int = 0
        position_x: float = 0.0
        position_y: float = 0.0
        area: float = 0.0
        aspect_ratio: float = 1.0
        mean_intensity: float = 0.0
        local_contrast: float = 0.0
        gradient_coherence: float = 0.0
        distance_from_center: float = 0.0
        orientation: float = 0.0
        elongation: float = 0.0
        
        # Classification scores
        prominence_score: float = 0.0
        filament_score: float = 0.0
        plage_score: float = 0.0
        artifact_score: float = 0.0
        
        # Final classification
        predicted_type: str = "unknown"
        confidence: float = 0.0
        
        def to_dict(self) -> Dict:
            return {
                "id": self.feature_id,
                "position": {"x": round(self.position_x, 2), "y": round(self.position_y, 2)},
                "area": round(self.area, 2),
                "aspect_ratio": round(self.aspect_ratio, 2),
                "mean_intensity": round(self.mean_intensity, 2),
                "local_contrast": round(self.local_contrast, 4),
                "gradient_coherence": round(self.gradient_coherence, 4),
                "distance_from_center": round(self.distance_from_center, 4),
                "orientation": round(self.orientation, 2),
                "elongation": round(self.elongation, 4),
                "scores": {
                    "prominence": round(self.prominence_score, 4),
                    "filament": round(self.filament_score, 4),
                    "plage": round(self.plage_score, 4),
                    "artifact": round(self.artifact_score, 4),
                },
                "predicted_type": self.predicted_type,
                "confidence": round(self.confidence, 4),
            }
    
    def discriminate(self, 
                     image: np.ndarray,
                     disk_center: Tuple[float, float],
                     disk_radius: float,
                     candidates: List[Dict]) -> List['ProminenceDiscriminator.EdgeFeature']:
        """Classify edge features into prominences, filaments, plages, or artifacts.
        
        Args:
            image: Grayscale solar image
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius in pixels
            candidates: List of candidate feature dicts with position, area, etc.
            
        Returns:
            List of classified EdgeFeature objects
        """
        classified = []
        cx, cy = disk_center
        r = disk_radius
        
        # Precompute gradient field for coherence analysis
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_angle = np.arctan2(grad_y, grad_x)
        
        for i, candidate in enumerate(candidates):
            feature = self.EdgeFeature(feature_id=i+1)
            
            # Extract basic properties
            feature.position_x = candidate.get("x", 0)
            feature.position_y = candidate.get("y", 0)
            feature.area = candidate.get("area", 0)
            feature.aspect_ratio = candidate.get("aspect_ratio", 1.0)
            feature.mean_intensity = candidate.get("mean_intensity", 0)
            
            # Compute distance from center (normalized)
            dist = math.sqrt((feature.position_x - cx)**2 + (feature.position_y - cy)**2)
            feature.distance_from_center = dist / max(r, 1)
            
            # Compute local contrast
            local_patch = self._extract_local_patch(image, feature.position_x, feature.position_y, r)
            if local_patch is not None:
                feature.local_contrast = np.std(local_patch) / (np.mean(local_patch) + 1e-8)
            
            # Compute gradient coherence
            feature.gradient_coherence = self._compute_gradient_coherence(
                grad_mag, grad_angle, feature.position_x, feature.position_y, candidate
            )
            
            # Compute elongation (using moments)
            if "moments" in candidate:
                m = candidate["moments"]
                mu20 = m.get("mu20", 0) / max(m.get("m00", 1), 1)
                mu02 = m.get("mu02", 0) / max(m.get("m00", 1), 1)
                mu11 = m.get("mu11", 0) / max(m.get("m00", 1), 1)
                
                # Principal axes
                lambda1 = (mu20 + mu02) / 2 + math.sqrt(4*mu11**2 + (mu20 - mu02)**2) / 2
                lambda2 = (mu20 + mu02) / 2 - math.sqrt(4*mu11**2 + (mu20 - mu02)**2) / 2
                feature.elongation = max(lambda1, lambda2) / max(min(lambda1, lambda2), 1e-8)
                feature.orientation = 0.5 * math.atan2(2*mu11, mu20 - mu02)
            
            # Classification scoring
            self._compute_classification_scores(feature, r)
            
            classified.append(feature)
        
        return classified
    
    def _compute_classification_scores(self, feature: 'ProminenceDiscriminator.EdgeFeature', disk_radius: float):
        """Compute classification scores for each type."""
        
        # --- Prominence criteria ---
        # Near limb, elongated, bright, high gradient coherence
        prominence_score = 0.0
        
        # Position score (near limb) - HIGHER WEIGHT
        pos_score = max(0, min(1, (feature.distance_from_center - 0.80) / 0.20))
        prominence_score += pos_score * 0.35
        
        # Elongation score (prominences are typically elongated)
        elong_score = min(feature.elongation / 2.5, 1.0)
        prominence_score += elong_score * 0.25
        
        # Intensity score (can be brighter OR darker than average)
        # Prominences often appear as bright extensions at limb
        intensity_score = min(max(feature.mean_intensity - 100, 0) / 150.0, 1.0)
        prominence_score += intensity_score * 0.15
        
        # Gradient coherence score (structured features have high coherence)
        prominence_score += feature.gradient_coherence * 0.25
        
        feature.prominence_score = prominence_score
        
        # --- Filament criteria ---
        # On disk, elongated, darker than surroundings
        filament_score = 0.0
        
        # Position score (on disk, not near limb)
        pos_score = max(0, 1 - feature.distance_from_center / 0.75)
        filament_score += pos_score * 0.25
        
        # Elongation score
        filament_score += min(feature.elongation / 2.5, 1.0) * 0.30
        
        # Intensity score (darker than average)
        intensity_score = max(0, 1 - feature.mean_intensity / 180.0)
        filament_score += intensity_score * 0.25
        
        # Gradient coherence score
        filament_score += feature.gradient_coherence * 0.20
        
        feature.filament_score = filament_score
        
        # --- Plage criteria ---
        # Bright, irregular shape, moderate gradient coherence
        plage_score = 0.0
        
        # Intensity score (very bright)
        intensity_score = min(feature.mean_intensity / 220.0, 1.0)
        plage_score += intensity_score * 0.40
        
        # Shape score (irregular, low elongation)
        shape_score = max(0, 1 - min(feature.elongation / 2.0, 1.0))
        plage_score += shape_score * 0.30
        
        # Local contrast score (high)
        plage_score += min(feature.local_contrast / 0.5, 1.0) * 0.30
        
        feature.plage_score = plage_score
        
        # --- Artifact criteria ---
        # Low gradient coherence, extreme aspect ratio, near edge
        artifact_score = 0.0
        
        # Low coherence
        artifact_score += (1 - feature.gradient_coherence) * 0.40
        
        # Extreme elongation
        artifact_score += min(abs(feature.elongation - 2.0) / 3.0, 1.0) * 0.30
        
        # Near image boundary (outside disk)
        boundary_score = 1.0 if feature.distance_from_center > 1.15 else 0.0
        artifact_score += boundary_score * 0.30
        
        feature.artifact_score = artifact_score
        
        # Final classification
        scores = {
            "prominence": feature.prominence_score,
            "filament": feature.filament_score,
            "plage": feature.plage_score,
            "artifact": feature.artifact_score,
        }
        
        feature.predicted_type = max(scores, key=scores.get)
        feature.confidence = max(scores.values())
    
    def _extract_local_patch(self, image: np.ndarray, x: float, y: float, 
                             radius: float, patch_size: int = 50) -> Optional[np.ndarray]:
        """Extract local patch around feature for contrast analysis."""
        h, w = image.shape
        size = int(min(patch_size, radius * 0.15))
        
        x1 = max(0, int(x - size))
        x2 = min(w, int(x + size))
        y1 = max(0, int(y - size))
        y2 = min(h, int(y + size))
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        
        return image[y1:y2, x1:x2]
    
    def _compute_gradient_coherence(self, grad_mag: np.ndarray, grad_angle: np.ndarray,
                                     x: float, y: float, candidate: Dict) -> float:
        """Compute gradient orientation coherence within feature region.
        
        High coherence indicates structured feature (prominence/filament).
        Low coherence indicates noise or artifact.
        """
        h, w = grad_mag.shape
        bbox = candidate.get("bbox", (0, 0, 10, 10))
        bx, by, bw, bh = bbox
        
        x1 = max(0, int(x - bw/2))
        x2 = min(w, int(x + bw/2))
        y1 = max(0, int(y - bh/2))
        y2 = min(h, int(y + bh/2))
        
        if x2 - x1 < 3 or y2 - y1 < 3:
            return 0.0
        
        # Extract gradient orientations in region
        angles = grad_angle[y1:y2, x1:x2]
        magnitudes = grad_mag[y1:y2, x1:x2]
        
        # Weighted mean orientation
        mag_flat = magnitudes.flatten()
        angle_flat = angles.flatten()
        
        if np.sum(mag_flat) < 1e-8:
            return 0.0
        
        # Compute circular variance
        sin_sum = np.sum(mag_flat * np.sin(2 * angle_flat))
        cos_sum = np.sum(mag_flat * np.cos(2 * angle_flat))
        coherence = np.sqrt(sin_sum**2 + cos_sum**2) / max(np.sum(mag_flat), 1e-8)
        
        return float(coherence)


# ============================================================
# Plage/Penumbra Faint Signal Extraction
# ============================================================

class PlagePenumbraExtractor:
    """Enhanced detection of faint plage and penumbra features.
    
    Technical approach:
    1. Multi-scale wavelet-like decomposition for feature separation
    2. Adaptive local contrast enhancement
    3. Dynamic thresholding based on local statistics
    4. Morphological reconstruction for boundary refinement
    """
    
    def extract_faint_features(self, image: np.ndarray,
                               disk_mask: np.ndarray,
                               disk_center: Tuple[float, float],
                               disk_radius: float) -> List[Dict]:
        """Extract faint plage and penumbra features.
        
        Args:
            image: Grayscale solar image
            disk_mask: Binary mask of solar disk
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius in pixels
            
        Returns:
            List of detected features with type classification
        """
        features = []
        
        # Step 1: Multi-scale decomposition
        scales = self._multi_scale_decomposition(image, disk_mask)
        
        # Step 2: Extract features at each scale
        for scale_name, scale_img in scales.items():
            scale_features = self._extract_at_scale(
                scale_img, disk_mask, disk_center, disk_radius, scale_name
            )
            features.extend(scale_features)
        
        # Step 3: Merge overlapping detections
        merged = self._merge_features(features)
        
        logger.info(f"Faint feature extraction: {len(merged)} features")
        return merged
    
    def _multi_scale_decomposition(self, image: np.ndarray, 
                                    disk_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose image into multiple scales using Gaussian-Laplacian pyramid.
        
        Scale 1 (Coarse): Large-scale structures (plage regions)
        Scale 2 (Medium): Mid-scale features (penumbra boundaries)
        Scale 3 (Fine): Fine details (pores, small bright points)
        """
        scales = {}
        
        # Original image
        scales["original"] = image.copy()
        
        # Coarse scale (large structures)
        coarse = cv2.GaussianBlur(image, (31, 31), 10)
        scales["coarse"] = coarse
        
        # Medium scale (subtract coarse from original)
        medium = cv2.subtract(image.astype(np.int16), coarse.astype(np.int16))
        medium = cv2.normalize(medium, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        scales["medium"] = medium
        
        # Fine scale (laplacian)
        fine = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
        fine = cv2.normalize(np.abs(fine), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        scales["fine"] = fine
        
        return scales
    
    def _extract_at_scale(self, image: np.ndarray,
                          disk_mask: np.ndarray,
                          disk_center: Tuple[float, float],
                          disk_radius: float,
                          scale_name: str) -> List[Dict]:
        """Extract features at a specific scale with adaptive thresholding."""
        features = []
        cx, cy = disk_center
        r = disk_radius
        
        # Apply disk mask
        masked = cv2.bitwise_and(image, image, mask=disk_mask)
        
        # Compute local statistics using sliding window
        local_mean = cv2.blur(masked.astype(np.float32), (15, 15))
        local_std = cv2.blur((masked.astype(np.float32) - local_mean)**2, (15, 15))
        local_std = np.sqrt(local_std)
        
        # Adaptive threshold based on local statistics
        k = 1.5  # Sensitivity parameter
        threshold = local_mean - k * local_std
        
        # Create binary mask for dark features (penumbra)
        dark_mask = (masked < threshold).astype(np.uint8) * 255
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=disk_mask)
        
        # Create binary mask for bright features (plage)
        bright_threshold = local_mean + k * local_std
        bright_mask = (masked > bright_threshold).astype(np.uint8) * 255
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=disk_mask)
        
        # Extract dark features
        dark_features = self._extract_from_mask(dark_mask, image, disk_mask, 
                                               disk_center, disk_radius, scale_name, "penumbra")
        features.extend(dark_features)
        
        # Extract bright features
        bright_features = self._extract_from_mask(bright_mask, image, disk_mask,
                                                 disk_center, disk_radius, scale_name, "plage")
        features.extend(bright_features)
        
        return features
    
    def _extract_from_mask(self, mask: np.ndarray,
                           original: np.ndarray,
                           disk_mask: np.ndarray,
                           disk_center: Tuple[float, float],
                           disk_radius: float,
                           scale_name: str,
                           feature_type: str) -> List[Dict]:
        """Extract features from binary mask."""
        features = []
        cx, cy = disk_center
        r = disk_radius
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 10:  # Minimum area threshold
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            ccx = centroids[i][0]
            ccy = centroids[i][1]
            
            # Distance from center
            dist = math.sqrt((ccx - cx)**2 + (ccy - cy)**2) / r
            
            # Extract intensity stats
            feat_mask = (labels == i).astype(np.uint8) * 255
            feat_mask = cv2.bitwise_and(feat_mask, feat_mask, mask=disk_mask)
            
            mean_val = cv2.mean(original, mask=feat_mask)[0]
            std_val = cv2.mean(original, mask=feat_mask)[1] if len(cv2.mean(original, mask=feat_mask)) > 1 else 0
            
            features.append({
                "type": feature_type,
                "scale": scale_name,
                "x": float(ccx),
                "y": float(ccy),
                "area": float(area),
                "bbox": (x, y, w, h),
                "mean_intensity": mean_val,
                "std_intensity": std_val,
                "distance_from_center": dist,
                "moments": self._compute_moments(labels, i),
            })
        
        return features
    
    def _compute_moments(self, labels: np.ndarray, label_id: int) -> Dict:
        """Compute central moments for shape analysis."""
        feat_mask = (labels == label_id).astype(np.uint8)
        moments = cv2.moments(feat_mask)
        return {
            "m00": moments.get("m00", 0),
            "m10": moments.get("m10", 0),
            "m01": moments.get("m01", 0),
            "mu20": moments.get("mu20", 0),
            "mu02": moments.get("mu02", 0),
            "mu11": moments.get("mu11", 0),
        }
    
    def _merge_features(self, features: List[Dict]) -> List[Dict]:
        """Merge overlapping features detected at different scales."""
        if not features:
            return []
        
        # Simple distance-based merging
        merged = []
        used = set()
        threshold = 20  # pixels
        
        for i, f1 in enumerate(features):
            if i in used:
                continue
            
            overlapping = [f1]
            for j, f2 in enumerate(features[i+1:], i+1):
                if j in used:
                    continue
                
                dist = math.sqrt((f1["x"] - f2["x"])**2 + (f1["y"] - f2["y"])**2)
                if dist < threshold:
                    overlapping.append(f2)
                    used.add(j)
            
            # Merge overlapping features
            if len(overlapping) == 1:
                merged.append(f1)
            else:
                # Keep the feature with highest area (most significant)
                best = max(overlapping, key=lambda f: f["area"])
                best["scales_merged"] = len(overlapping)
                merged.append(best)
        
        return merged


# ============================================================
# Sunspot Group Segmentation
# ============================================================

class SunspotGroupSegmenter:
    """Advanced segmentation of complex sunspot groups.
    
    Addresses:
    -粘连 black子 separation (touching/overlapping spots)
    - Penumbra/umbra distinction
    - Group boundary delineation
    - Morphological complexity analysis
    
    Technical approach:
    1. Distance transform + watershed for separation
    2. Morphological snake for boundary refinement
    3. Multi-thresholding for umbra/penumbra distinction
    """
    
    def segment_groups(self, image: np.ndarray,
                       disk_mask: np.ndarray,
                       disk_center: Tuple[float, float],
                       disk_radius: float,
                       initial_mask: np.ndarray) -> List[Dict]:
        """Segment complex sunspot groups into individual components.
        
        Args:
            image: Grayscale solar image
            disk_mask: Binary mask of solar disk
            disk_center: (cx, cy) of solar disk
            disk_radius: Radius in pixels
            initial_mask: Initial binary mask of sunspot candidates
            
        Returns:
            List of segmented sunspot features with group information
        """
        cx, cy = disk_center
        r = disk_radius
        
        # Step 1: Find connected components in initial mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            initial_mask, connectivity=8
        )
        
        segmented_spots = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:  # Skip very small regions
                continue
            
            # Extract component mask
            comp_mask = (labels == i).astype(np.uint8) * 255
            
            # Check if component needs splitting (large or elongated)
            needs_split = self._needs_splitting(comp_mask, area, stats[i])
            
            if needs_split:
                # Apply watershed segmentation
                sub_spots = self._watershed_split(
                    image, comp_mask, disk_center, disk_radius
                )
                segmented_spots.extend(sub_spots)
            else:
                # Single spot, extract properties
                spot = self._extract_spot_properties(
                    image, comp_mask, disk_center, disk_radius, stats[i], centroids[i]
                )
                segmented_spots.append(spot)
        
        # Step 2: Group nearby spots
        groups = self._cluster_into_groups(segmented_spots, r)
        
        logger.info(f"Sunspot group segmentation: {len(segmented_spots)} spots in {len(groups)} groups")
        return segmented_spots, groups
    
    def _needs_splitting(self, mask: np.ndarray, area: float, 
                         stats: np.ndarray) -> bool:
        """Determine if a connected component needs to be split.
        
        Criteria:
        - Large area (> 500 pixels)
        - High aspect ratio (> 3.0)
        - Multiple intensity minima
        """
        # Size criterion
        if area < 500:
            return False
        
        # Aspect ratio criterion
        w = stats[cv2.CC_STAT_WIDTH]
        h = stats[cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 3.0:
            return True
        
        # Check for multiple intensity minima (using distance transform)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist_transform, 0.5 * np.max(dist_transform), 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        
        # Count local maxima
        num_peaks, _, _, _ = cv2.connectedComponentsWithStats(peaks)
        if num_peaks > 2:  # More than one peak suggests multiple spots
            return True
        
        return False
    
    def _watershed_split(self, image: np.ndarray,
                         mask: np.ndarray,
                         disk_center: Tuple[float, float],
                         disk_radius: float) -> List[Dict]:
        """Split connected component using watershed algorithm.
        
        Steps:
        1. Distance transform to find seed points
        2. Find sure foreground/background
        3. Apply watershed
        4. Extract individual spots
        """
        cx, cy = disk_center
        r = disk_radius
        
        # Sure foreground (eroded mask)
        kernel = cv2.getStructuringElement(cv2.DIST_L2, (5, 5))
        sure_fg = cv2.erode(mask, kernel, iterations=2)
        
        # Distance transform for seed detection
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Find local maxima as markers
        _, dist_thresh = cv2.threshold(dist_transform, 0.4 * np.max(dist_transform), 255, cv2.THRESH_BINARY)
        dist_thresh = dist_thresh.astype(np.uint8)
        
        num_markers, markers = cv2.connectedComponents(dist_thresh)
        markers = markers + 1  # Background marker = 1
        
        # Unknown region (dilate sure_fg)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Set unknown region to 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        
        # Extract segmented spots
        spots = []
        for marker_id in range(2, num_markers + 1):
            spot_mask = (markers == marker_id).astype(np.uint8) * 255
            
            # Validate spot (minimum area)
            area = cv2.countNonZero(spot_mask)
            if area < 15:
                continue
            
            # Extract properties
            moments = cv2.moments(spot_mask)
            if moments["m00"] == 0:
                continue
            
            ccx = moments["m10"] / moments["m00"]
            ccy = moments["m01"] / moments["m00"]
            
            spot = self._extract_spot_properties(
                image, spot_mask, disk_center, disk_radius, None, (ccx, ccy)
            )
            spot["segmented"] = True
            spots.append(spot)
        
        return spots
    
    def _extract_spot_properties(self, image: np.ndarray,
                                  mask: np.ndarray,
                                  disk_center: Tuple[float, float],
                                  disk_radius: float,
                                  stats: Optional[np.ndarray],
                                  centroid: Tuple[float, float]) -> Dict:
        """Extract comprehensive properties of a segmented sunspot."""
        cx, cy = disk_center
        r = disk_radius
        ccx, ccy = centroid
        
        # Area
        area = float(cv2.countNonZero(mask))
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(mask)
        
        # Intensity statistics
        mean_val_tuple = cv2.mean(image, mask=mask)
        mean_val = float(mean_val_tuple[0])
        _, std_val_arr = cv2.meanStdDev(image, mask=mask)
        std_val = float(std_val_arr.flatten()[0])
        
        # Equivalent diameter
        eq_diameter = math.sqrt(4 * area / math.pi)
        
        # Circularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        else:
            circularity = 0.0
        
        # Distance from center
        dist_from_center = math.sqrt((ccx - cx)**2 + (ccy - cy)**2) / r
        
        # Umbra/penumbra distinction (if large enough)
        umbra_area = 0.0
        penumbra_area = 0.0
        if area > 100:
            # Umbra: darkest core
            _, umbra_mask = cv2.threshold(image, mean_val - std_val, 255, cv2.THRESH_BINARY_INV)
            umbra_mask = cv2.bitwise_and(umbra_mask, umbra_mask, mask=mask)
            umbra_area = float(cv2.countNonZero(umbra_mask))
            penumbra_area = area - umbra_area
        
        return {
            "x": float(ccx),
            "y": float(ccy),
            "area": area,
            "bbox": (x, y, w, h),
            "equivalent_diameter": eq_diameter,
            "circularity": float(circularity),
            "mean_intensity": float(mean_val),
            "std_intensity": float(std_val),
            "distance_from_center": float(dist_from_center),
            "umbra_area": umbra_area,
            "penumbra_area": penumbra_area,
            "segmented": False,
        }
    
    def _cluster_into_groups(self, spots: List[Dict], disk_radius: float) -> List[Dict]:
        """Cluster sunspots into groups based on proximity.
        
        Uses DBSCAN-like approach with adaptive distance threshold.
        """
        if len(spots) < 2:
            return []
        
        groups = []
        used = set()
        threshold = disk_radius * 0.12  # 12% of disk radius
        
        for i, s1 in enumerate(spots):
            if i in used:
                continue
            
            group_members = [i]
            for j, s2 in enumerate(spots):
                if j in used or j == i:
                    continue
                
                dist = math.sqrt((s1["x"] - s2["x"])**2 + (s1["y"] - s2["y"])**2)
                if dist < threshold:
                    group_members.append(j)
            
            if len(group_members) >= 2:
                for m in group_members:
                    used.add(m)
                
                # Compute group properties
                member_spots = [spots[m] for m in group_members]
                group_cx = np.mean([s["x"] for s in member_spots])
                group_cy = np.mean([s["y"] for s in member_spots])
                total_area = sum(s["area"] for s in member_spots)
                max_spread = max(
                    math.sqrt((s["x"] - group_cx)**2 + (s["y"] - group_cy)**2)
                    for s in member_spots
                )
                
                groups.append({
                    "id": len(groups) + 1,
                    "member_count": len(group_members),
                    "member_indices": group_members,
                    "center_x": float(group_cx),
                    "center_y": float(group_cy),
                    "total_area": float(total_area),
                    "max_spread": float(max_spread),
                    "complexity": min(len(group_members) * 1.5 + max_spread / disk_radius * 2, 10.0),
                })
        
        return groups


# ============================================================
# Unified Advanced Detection Pipeline
# ============================================================

class AdvancedDetectionPipeline:
    """Unified pipeline integrating all advanced detection features.
    
    Pipeline stages:
    1. Image preprocessing (illumination correction, denoising)
    2. Solar disk detection
    3. Sunspot segmentation with group analysis
    4. Plage/penumbra extraction
    5. Prominence discrimination
    6. Feature merging and classification
    """
    
    def __init__(self):
        self.illumination_corrector = IlluminationCorrector()
        self.prominence_discriminator = ProminenceDiscriminator()
        self.plage_extractor = PlagePenumbraExtractor()
        self.group_segmenter = SunspotGroupSegmenter()
    
    def detect(self, image: np.ndarray,
               disk_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run full advanced detection pipeline.
        
        Args:
            image: Grayscale solar image
            disk_info: Dictionary with center_x, center_y, radius, etc.
            
        Returns:
            Comprehensive detection results
        """
        cx = disk_info.get("center_x", image.shape[1] / 2)
        cy = disk_info.get("center_y", image.shape[0] / 2)
        r = disk_info.get("radius", min(image.shape) * 0.4)
        
        # Create disk mask
        h, w = image.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        disk_mask = ((x_grid - cx)**2 + (y_grid - cy)**2 <= r**2).astype(np.uint8)
        
        # Step 1: Image preprocessing
        logger.info("Step 1: Image preprocessing...")
        corrected = self.illumination_corrector.correct_retinex(image)
        corrected = self.illumination_corrector.correct_limb_darkening(corrected, (cx, cy), r)
        corrected = self.illumination_corrector.enhance_local_contrast(corrected, clip_limit=1.5)
        corrected = self.illumination_corrector.denoise_bilateral(corrected, d=5)
        
        # Step 2: Sunspot detection with group segmentation
        logger.info("Step 2: Sunspot segmentation...")
        initial_sunspot_mask = self._detect_sunspot_candidates(corrected, disk_mask, (cx, cy), r)
        sunspots, groups = self.group_segmenter.segment_groups(
            corrected, disk_mask, (cx, cy), r, initial_sunspot_mask
        )
        
        # Step 3: Plage/penumbra extraction
        logger.info("Step 3: Plage/penumbra extraction...")
        faint_features = self.plage_extractor.extract_faint_features(
            corrected, disk_mask, (cx, cy), r
        )
        
        # Step 4: Prominence discrimination
        logger.info("Step 4: Prominence discrimination...")
        edge_candidates = self._detect_edge_candidates(corrected, disk_mask, (cx, cy), r)
        prominences = self.prominence_discriminator.discriminate(
            corrected, (cx, cy), r, edge_candidates
        )
        
        # Compile results
        results = {
            "sunspots": sunspots,
            "groups": groups,
            "faint_features": faint_features,
            "prominences": [p.to_dict() for p in prominences],
            "disk_info": {
                "center_x": cx,
                "center_y": cy,
                "radius": r,
            },
            "statistics": {
                "total_sunspots": len(sunspots),
                "total_groups": len(groups),
                "total_faint_features": len(faint_features),
                "total_prominences": len(prominences),
            },
        }
        
        logger.info(f"Detection complete: {results['statistics']}")
        return results
    
    def _detect_sunspot_candidates(self, image: np.ndarray,
                                    disk_mask: np.ndarray,
                                    disk_center: Tuple[float, float],
                                    disk_radius: float) -> np.ndarray:
        """Generate initial sunspot candidate mask."""
        cx, cy = disk_center
        r = disk_radius
        
        # Compute disk statistics
        disk_pixels = image[disk_mask > 0]
        disk_mean = np.mean(disk_pixels)
        disk_std = np.std(disk_pixels)
        
        # Threshold for sunspot detection
        threshold = disk_mean - 1.5 * disk_std
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask, mask, mask=disk_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _detect_edge_candidates(self, image: np.ndarray,
                                 disk_mask: np.ndarray,
                                 disk_center: Tuple[float, float],
                                 disk_radius: float) -> List[Dict]:
        """Detect candidate features near solar limb with enhanced sensitivity."""
        cx, cy = disk_center
        r = disk_radius
        h, w = image.shape
        
        # Create limb zone mask (80%-120% of disk radius for better coverage)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        inner_r = r * 0.80  # Expanded from 0.85
        outer_r = r * 1.20  # Expanded from 1.15
        limb_mask = ((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8)
        
        # Compute local limb statistics for better thresholding
        limb_pixels = image[limb_mask > 0]
        if len(limb_pixels) < 100:
            return []
            
        limb_mean = np.mean(limb_pixels)
        limb_std = np.std(limb_pixels)
        disk_mean = np.mean(image[disk_mask > 0])
        disk_std = np.std(image[disk_mask > 0])
        
        # Multi-scale thresholding for better prominence detection
        candidates = []
        
        # Scale 1: Bright features (prominences)
        for k in [0.6, 0.8, 1.0]:  # Multiple sensitivity levels
            bright_thresh = limb_mean + k * limb_std
            _, bright_mask = cv2.threshold(image, bright_thresh, 255, cv2.THRESH_BINARY)
            bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=limb_mask)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
            
            # Extract candidates at this scale
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 8:  # Lowered from 15 for better detection
                    continue
                    
                # Compute moments for shape analysis
                feat_mask = (labels == i).astype(np.uint8) * 255
                moments = cv2.moments(feat_mask)
                mu20 = moments.get("mu20", 0) / max(moments.get("m00", 1), 1)
                mu02 = moments.get("mu02", 0) / max(moments.get("m00", 1), 1)
                mu11 = moments.get("mu11", 0) / max(moments.get("m00", 1), 1)
                
                candidates.append({
                    "x": float(centroids[i][0]),
                    "y": float(centroids[i][1]),
                    "area": float(area),
                    "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                    "mean_intensity": float(np.mean(image[labels == i])),
                    "moments": {
                        "m00": moments.get("m00", 0),
                        "mu20": mu20,
                        "mu02": mu02,
                        "mu11": mu11,
                    },
                    "aspect_ratio": max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 
                                   max(min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]), 1),
                    "detection_scale": f"bright_k{k}",
                })
        
        # Scale 2: Dark features (filaments/dark prominences)
        for k in [0.6, 0.8]:
            dark_thresh = limb_mean - k * limb_std
            _, dark_mask = cv2.threshold(image, dark_thresh, 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=limb_mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                dark_mask, connectivity=8
            )
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 10:
                    continue
                    
                feat_mask = (labels == i).astype(np.uint8) * 255
                moments = cv2.moments(feat_mask)
                mu20 = moments.get("mu20", 0) / max(moments.get("m00", 1), 1)
                mu02 = moments.get("mu02", 0) / max(moments.get("m00", 1), 1)
                mu11 = moments.get("mu11", 0) / max(moments.get("m00", 1), 1)
                
                candidates.append({
                    "x": float(centroids[i][0]),
                    "y": float(centroids[i][1]),
                    "area": float(area),
                    "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                    "mean_intensity": float(np.mean(image[labels == i])),
                    "moments": {
                        "m00": moments.get("m00", 0),
                        "mu20": mu20,
                        "mu02": mu02,
                        "mu11": mu11,
                    },
                    "aspect_ratio": max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 
                                   max(min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]), 1),
                    "detection_scale": f"dark_k{k}",
                })
        
        # Merge overlapping detections (keep highest area)
        merged = self._merge_edge_candidates(candidates)
        
        logger.info(f"Edge candidates detected: {len(merged)} features")
        return merged
    
    def _merge_edge_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Merge overlapping edge candidates detected at multiple scales."""
        if not candidates:
            return []
            
        merged = []
        used = set()
        threshold = 15  # pixels
        
        for i, c1 in enumerate(candidates):
            if i in used:
                continue
            
            overlapping = [c1]
            for j, c2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                dist = math.sqrt((c1["x"] - c2["x"])**2 + (c1["y"] - c2["y"])**2)
                if dist < threshold:
                    overlapping.append(c2)
                    used.add(j)
            
            # Keep the candidate with largest area
            best = max(overlapping, key=lambda c: c["area"])
            if len(overlapping) > 1:
                best["scales_merged"] = len(overlapping)
            merged.append(best)
        
        return merged
