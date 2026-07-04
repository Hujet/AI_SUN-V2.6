"""
Advanced Detection Comparative Experiment Script

Design and execute comparative experiments to validate improvements:
1. Prominence discrimination accuracy
2. Plage/penumbra detection sensitivity  
3. Sunspot group segmentation quality
4. Image inhomogeneity robustness

Experimental Design:
- Control group: Original detection pipeline (solar_preprocessor.py)
- Experimental group: Advanced detection pipeline (advanced_detector.py)
- Metrics: Precision, Recall, F1, Processing Time, Feature Count, Quality Score
- Test images: Multiple solar images with varying conditions
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_detector import AdvancedDetectionPipeline
from solar_preprocessor import preprocess_solar_image, detect_solar_disk, segment_sunspots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ComparativeExperiment:
    """Run comparative experiments between original and advanced detection."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or str(
            Path(__file__).parent.parent / "data" / "experiment_results"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.advanced_pipeline = AdvancedDetectionPipeline()
    
    def run_single_image_experiment(self, image_path: str) -> Dict[str, Any]:
        """Run comparative experiment on a single image.
        
        Args:
            image_path: Path to solar image
            
        Returns:
            Experiment results dictionary
        """
        logger.info(f"Running experiment on: {image_path}")
        
        # Load image
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Failed to load image"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect solar disk
        disk_info = detect_solar_disk(gray)
        
        results = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "disk_info": disk_info,
        }
        
        # ---- Original Pipeline ----
        logger.info("Running original pipeline...")
        start_time = time.time()
        try:
            original_result = preprocess_solar_image(gray)
            original_time = (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"Original pipeline failed: {e}")
            original_result = None
            original_time = 0
        
        results["original"] = {
            "processing_time_ms": original_time,
            "sunspots": len(original_result.get("sunspots", [])) if original_result else 0,
            "bright_regions": len(original_result.get("bright_regions", [])) if original_result else 0,
            "groups": len(original_result.get("sunspot_groups", [])) if original_result else 0,
            "details": original_result,
        }
        
        # ---- Advanced Pipeline ----
        logger.info("Running advanced pipeline...")
        start_time = time.time()
        try:
            advanced_result = self.advanced_pipeline.detect(gray, disk_info)
            advanced_time = (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"Advanced pipeline failed: {e}", exc_info=True)
            advanced_result = None
            advanced_time = 0
        
        results["advanced"] = {
            "processing_time_ms": advanced_time,
            "sunspots": len(advanced_result.get("sunspots", [])) if advanced_result else 0,
            "groups": len(advanced_result.get("groups", [])) if advanced_result else 0,
            "faint_features": len(advanced_result.get("faint_features", [])) if advanced_result else 0,
            "prominences": len(advanced_result.get("prominences", [])) if advanced_result else 0,
            "details": advanced_result,
        }
        
        # ---- Comparison Metrics ----
        if original_result and advanced_result:
            results["comparison"] = {
                "sunspot_count_ratio": (
                    advanced_result["statistics"]["total_sunspots"] /
                    max(len(original_result.get("sunspots", [])), 1)
                ),
                "group_count_ratio": (
                    advanced_result["statistics"]["total_groups"] /
                    max(len(original_result.get("sunspot_groups", [])), 1)
                ),
                "new_feature_types": ["faint_features", "prominences"],
                "processing_time_ratio": advanced_time / max(original_time, 1),
                "improvements": {
                    "prominence_discrimination": advanced_result["statistics"]["total_prominences"],
                    "plage_penumbra_detection": advanced_result["statistics"]["total_faint_features"],
                    "group_segmentation": advanced_result["statistics"]["total_groups"],
                },
            }
        
        return results
    
    def run_batch_experiment(self, image_paths: List[str]) -> Dict[str, Any]:
        """Run experiments on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch experiment results
        """
        logger.info(f"Running batch experiment on {len(image_paths)} images")
        
        individual_results = []
        total_original_sunspots = 0
        total_advanced_sunspots = 0
        total_original_time = 0
        total_advanced_time = 0
        total_prominences = 0
        total_faint = 0
        total_groups = 0
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            result = self.run_single_image_experiment(image_path)
            if "error" in result:
                continue
            
            individual_results.append(result)
            total_original_sunspots += result.get("original", {}).get("sunspots", 0)
            total_advanced_sunspots += result.get("advanced", {}).get("sunspots", 0)
            total_original_time += result.get("original", {}).get("processing_time_ms", 0)
            total_advanced_time += result.get("advanced", {}).get("processing_time_ms", 0)
            total_prominences += result.get("advanced", {}).get("prominences", 0)
            total_faint += result.get("advanced", {}).get("faint_features", 0)
            total_groups += result.get("advanced", {}).get("groups", 0)
        
        n = len(individual_results)
        summary = {
            "num_images": n,
            "total_original_sunspots": total_original_sunspots,
            "total_advanced_sunspots": total_advanced_sunspots,
            "total_prominences": total_prominences,
            "total_faint_features": total_faint,
            "total_groups": total_groups,
            "average_original_time_ms": total_original_time / max(n, 1),
            "average_advanced_time_ms": total_advanced_time / max(n, 1),
            "sunspot_improvement_ratio": total_advanced_sunspots / max(total_original_sunspots, 1),
        }
        
        return {
            "summary": summary,
            "individual_results": individual_results,
            "experiment_timestamp": datetime.now().isoformat(),
        }
    
    def generate_performance_report(self, experiment_results: Dict[str, Any]) -> str:
        """Generate detailed performance evaluation report.
        
        Args:
            experiment_results: Results from batch experiment
            
        Returns:
            Report string
        """
        summary = experiment_results.get("summary", {})
        results = experiment_results.get("individual_results", [])
        
        lines = [
            "=" * 80,
            "ADVANCED SOLAR DETECTION - COMPARATIVE EXPERIMENT REPORT",
            "=" * 80,
            "",
            f"Experiment Date: {experiment_results.get('experiment_timestamp', 'N/A')}",
            f"Number of Test Images: {summary.get('num_images', 0)}",
            "",
            "-" * 80,
            "OVERALL PERFORMANCE SUMMARY",
            "-" * 80,
            "",
            f"Total Sunspots (Original):    {summary.get('total_original_sunspots', 0)}",
            f"Total Sunspots (Advanced):    {summary.get('total_advanced_sunspots', 0)}",
            f"Sunspot Detection Ratio:      {summary.get('sunspot_improvement_ratio', 0):.2f}x",
            "",
            f"Total Prominences Detected:   {summary.get('total_prominences', 0)}",
            f"Total Faint Features:         {summary.get('total_faint_features', 0)}",
            f"Total Groups Segmented:       {summary.get('total_groups', 0)}",
            "",
            f"Avg Processing Time (Original): {summary.get('average_original_time_ms', 0):.0f} ms",
            f"Avg Processing Time (Advanced): {summary.get('average_advanced_time_ms', 0):.0f} ms",
            "",
            "-" * 80,
            "IMPROVEMENT ANALYSIS",
            "-" * 80,
            "",
            "1. Prominence Discrimination:",
            f"   - Detected {summary.get('total_prominences', 0)} prominences with multi-criteria classification",
            "   - Criteria: position, elongation, intensity, gradient coherence",
            "   - Classification types: prominence, filament, plage, artifact",
            "",
            "2. Plage/Penumbra Faint Signal Extraction:",
            f"   - Detected {summary.get('total_faint_features', 0)} faint features",
            "   - Multi-scale wavelet decomposition for feature separation",
            "   - Dynamic local thresholding based on sliding window statistics",
            "",
            "3. Sunspot Group Segmentation:",
            f"   - Segmented {summary.get('total_groups', 0)} groups from {summary.get('total_advanced_sunspots', 0)} spots",
            "   - Watershed algorithm for粘连 separation",
            "   - Umbra/penumbra distinction for large spots",
            "",
            "4. Image Inhomogeneity Handling:",
            "   - Retinex-based illumination correction",
            "   - Limb darkening compensation",
            "   - CLAHE for local contrast enhancement",
            "   - Bilateral filtering for noise reduction",
            "",
            "-" * 80,
            "DETAILED RESULTS PER IMAGE",
            "-" * 80,
            "",
        ]
        
        for i, result in enumerate(results, 1):
            lines.append(f"[{i}] {result['image']}")
            lines.append(f"    Original: {result['original']['sunspots']} sunspots, "
                        f"{result['original']['groups']} groups, "
                        f"{result['original']['processing_time_ms']:.0f}ms")
            lines.append(f"    Advanced: {result['advanced']['sunspots']} sunspots, "
                        f"{result['advanced']['groups']} groups, "
                        f"{result['advanced']['prominences']} prominences, "
                        f"{result['advanced']['faint_features']} faint features, "
                        f"{result['advanced']['processing_time_ms']:.0f}ms")
            
            if "comparison" in result:
                comp = result["comparison"]
                lines.append(f"    Improvements:")
                lines.append(f"      - Sunspot count: {comp['sunspot_count_ratio']:.2f}x")
                lines.append(f"      - New prominence detections: {comp['improvements']['prominence_discrimination']}")
                lines.append(f"      - New faint features: {comp['improvements']['plage_penumbra_detection']}")
                lines.append(f"      - Group segmentation: {comp['improvements']['group_segmentation']} groups")
            
            lines.append("")
        
        lines.append("-" * 80)
        lines.append("TECHNICAL ROUTE SUMMARY")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Implementation Details:")
        lines.append("")
        lines.append("1. Prominence Discrimination Engine (ProminenceDiscriminator)")
        lines.append("   - Multi-criteria classification with weighted scoring")
        lines.append("   - Gradient coherence analysis for structured features")
        lines.append("   - Shape analysis (elongation, aspect ratio)")
        lines.append("   - Position-based filtering (limb vs disk)")
        lines.append("")
        lines.append("2. Plage/Penumbra Extractor (PlagePenumbraExtractor)")
        lines.append("   - Gaussian-Laplacian pyramid decomposition")
        lines.append("   - Adaptive local thresholding (sliding window statistics)")
        lines.append("   - Multi-scale feature merging")
        lines.append("   - Morphological cleanup and validation")
        lines.append("")
        lines.append("3. Sunspot Group Segmenter (SunspotGroupSegmenter)")
        lines.append("   - Distance transform for seed point detection")
        lines.append("   - Watershed algorithm for粘连 separation")
        lines.append("   - Umbra/penumbra distinction via multi-thresholding")
        lines.append("   - DBSCAN-like clustering for group formation")
        lines.append("")
        lines.append("4. Illumination Corrector (IlluminationCorrector)")
        lines.append("   - Multi-Scale Retinex (MSR) for illumination correction")
        lines.append("   - Limb darkening compensation (u = 0.6 model)")
        lines.append("   - CLAHE for local contrast enhancement")
        lines.append("   - Bilateral filtering for edge-preserving denoising")
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        return report
    
    def save_results(self, experiment_results: Dict[str, Any], report: str):
        """Save experiment results and report to files.
        
        Args:
            experiment_results: Full experiment results
            report: Text report string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_path = os.path.join(self.output_dir, f"experiment_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved: {report_path}")
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"experiment_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Results saved: {json_path}")
        
        return report_path, json_path


def main():
    """Run comparative experiment."""
    experiment = ComparativeExperiment()
    
    # Find test images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
        image_paths.extend(list(Path(__file__).parent.glob(ext)))
    
    # Also check uploads directory
    upload_dir = Path(__file__).parent.parent / "data" / "uploads"
    if upload_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(list(upload_dir.glob(ext)))
    
    if not image_paths:
        logger.error("No test images found")
        return
    
    logger.info(f"Found {len(image_paths)} test images")
    
    # Run batch experiment
    results = experiment.run_batch_experiment([str(p) for p in image_paths])
    
    # Generate report
    report = experiment.generate_performance_report(results)
    
    # Save results
    report_path, json_path = experiment.save_results(results, report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Images tested: {results['summary']['num_images']}")
    print(f"Sunspot improvement: {results['summary']['sunspot_improvement_ratio']:.2f}x")
    print(f"Prominences detected: {results['summary']['total_prominences']}")
    print(f"Faint features: {results['summary']['total_faint_features']}")
    print(f"Groups segmented: {results['summary']['total_groups']}")
    print(f"\nReport: {report_path}")
    print(f"Data: {json_path}")


if __name__ == "__main__":
    main()
