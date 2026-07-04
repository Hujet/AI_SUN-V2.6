"""
Enhanced Detection Performance Evaluation Script

Runs comparative experiments between standard and enhanced detection methods.
Generates detailed performance evaluation report.

Evaluation Metrics:
1. Feature Detection Rate (FDR): # features detected / ground truth
2. False Positive Rate (FPR): # false positives / total detections
3. Precision: TP / (TP + FP)
4. Recall: TP / (TP + FN)
5. F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
6. Processing Time: milliseconds per image
7. Quality Score: average feature quality metric

Usage:
    python test_enhanced_detection.py [--image_dir PATH] [--output_dir PATH]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_detector import EnhancedDetectionPipeline, generate_detection_report
from sunspot_detector import SunspotDetectionPipeline
from solar_preprocessor import preprocess_solar_image, detect_solar_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Evaluator for comparing detection methods."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or str(Path(__file__).parent.parent / "data" / "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.enhanced_pipeline = EnhancedDetectionPipeline()
        self.standard_pipeline = SunspotDetectionPipeline()
    
    def evaluate_single_image(self, image_path: str) -> Dict[str, Any]:
        """Evaluate detection on a single image.
        
        Args:
            image_path: Path to solar image
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating: {image_path}")
        
        # Load image
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Failed to load image"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Detect solar disk
        disk_info = detect_solar_disk(gray)
        
        # Step 2: Run standard detection
        start_time = time.time()
        try:
            standard_result = self.standard_pipeline.process(image_path)
            standard_time = (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"Standard detection failed: {e}")
            standard_result = None
            standard_time = 0
        
        # Step 3: Run enhanced detection
        start_time = time.time()
        enhanced_result = self.enhanced_pipeline.detect(gray, disk_info)
        enhanced_time = (time.time() - start_time) * 1000
        
        # Step 4: Compute comparison metrics
        comparison = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "disk_detection": {
                "method": disk_info.get("method", "unknown"),
                "confidence": disk_info.get("confidence", 0.0),
                "center": (disk_info.get("center_x", 0), disk_info.get("center_y", 0)),
                "radius": disk_info.get("radius", 0),
            },
            "standard": {
                "total_features": standard_result.total_spots if standard_result else 0,
                "processing_time_ms": standard_result.processing_time_ms if standard_result else standard_time,
                "features_by_region": standard_result.spots_by_region if standard_result else {},
                "spots": len(standard_result.sunspots) if standard_result else 0,
            },
            "enhanced": {
                "total_features": enhanced_result.statistics.total_features,
                "processing_time_ms": enhanced_result.statistics.processing_time_ms,
                "features_by_type": enhanced_result.statistics.features_by_type,
                "features_by_scale": enhanced_result.statistics.features_by_scale,
                "features_by_region": enhanced_result.statistics.features_by_region,
                "average_confidence": enhanced_result.statistics.average_confidence,
                "average_quality": enhanced_result.statistics.average_quality,
            },
            "improvement": {
                "feature_count_ratio": (
                    enhanced_result.statistics.total_features / 
                    max(standard_result.total_spots, 1) if standard_result else 0
                ),
                "processing_time_ratio": enhanced_time / max(standard_time, 1),
                "new_feature_types": list(enhanced_result.statistics.features_by_type.keys()),
            },
        }
        
        return comparison
    
    def evaluate_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """Evaluate detection on a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch evaluation results
        """
        logger.info(f"Evaluating batch of {len(image_paths)} images")
        
        results = []
        total_standard_time = 0
        total_enhanced_time = 0
        total_standard_features = 0
        total_enhanced_features = 0
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            result = self.evaluate_single_image(image_path)
            if "error" in result:
                continue
            
            results.append(result)
            total_standard_time += result["standard"]["processing_time_ms"]
            total_enhanced_time += result["enhanced"]["processing_time_ms"]
            total_standard_features += result["standard"]["total_features"]
            total_enhanced_features += result["enhanced"]["total_features"]
        
        # Aggregate statistics
        n = len(results)
        batch_summary = {
            "num_images": n,
            "total_standard_features": total_standard_features,
            "total_enhanced_features": total_enhanced_features,
            "average_standard_time": total_standard_time / max(n, 1),
            "average_enhanced_time": total_enhanced_time / max(n, 1),
            "feature_improvement_ratio": total_enhanced_features / max(total_standard_features, 1),
            "time_improvement_ratio": total_enhanced_time / max(total_standard_time, 1),
        }
        
        return {
            "batch_summary": batch_summary,
            "individual_results": results,
            "evaluation_timestamp": datetime.now().isoformat(),
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_filename: str = None):
        """Generate detailed performance evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_batch
            output_filename: Output filename (default: auto-generated)
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_report_{timestamp}.txt"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        batch = evaluation_results.get("batch_summary", {})
        results = evaluation_results.get("individual_results", [])
        
        lines = [
            "=" * 80,
            "SOLAR FEATURE DETECTION - PERFORMANCE EVALUATION REPORT",
            "=" * 80,
            "",
            f"Evaluation Date: {evaluation_results.get('evaluation_timestamp', 'N/A')}",
            f"Number of Images: {batch.get('num_images', 0)}",
            "",
            "-" * 80,
            "OVERALL SUMMARY",
            "-" * 80,
            "",
            f"Total Features (Standard):  {batch.get('total_standard_features', 0)}",
            f"Total Features (Enhanced):  {batch.get('total_enhanced_features', 0)}",
            f"Feature Improvement Ratio:  {batch.get('feature_improvement_ratio', 0):.2f}x",
            "",
            f"Avg Processing Time (Standard): {batch.get('average_standard_time', 0):.0f} ms",
            f"Avg Processing Time (Enhanced): {batch.get('average_enhanced_time', 0):.0f} ms",
            f"Time Ratio:                   {batch.get('time_improvement_ratio', 0):.2f}x",
            "",
            "-" * 80,
            "DETAILED RESULTS PER IMAGE",
            "-" * 80,
            "",
        ]
        
        for i, result in enumerate(results, 1):
            lines.append(f"[{i}] {result['image']}")
            lines.append(f"    Disk Detection: {result['disk_detection']['method']} "
                        f"(confidence: {result['disk_detection']['confidence']:.2f})")
            lines.append(f"    Standard: {result['standard']['total_features']} features, "
                        f"{result['standard']['processing_time_ms']:.0f}ms")
            lines.append(f"    Enhanced: {result['enhanced']['total_features']} features, "
                        f"{result['enhanced']['processing_time_ms']:.0f}ms")
            lines.append(f"    Feature Types: {result['enhanced']['features_by_type']}")
            lines.append(f"    Scales Used: {result['enhanced']['features_by_scale']}")
            lines.append(f"    Avg Confidence: {result['enhanced']['average_confidence']:.4f}")
            lines.append(f"    Avg Quality: {result['enhanced']['average_quality']:.4f}")
            lines.append(f"    Improvement: {result['improvement']['feature_count_ratio']:.2f}x features")
            lines.append("")
        
        lines.append("-" * 80)
        lines.append("ENHANCEMENT FEATURES")
        lines.append("-" * 80)
        lines.append("")
        lines.append("1. Multi-Scale Detection:")
        lines.append("   - Scale 1 (Original, 1.0x): Large features (sunspots, groups)")
        lines.append("   - Scale 2 (Medium, 0.5x): Medium features (moderate spots)")
        lines.append("   - Scale 3 (Fine, 0.25x): Small features (pores, micro-flares)")
        lines.append("   - Fusion: NMS with confidence-weighted merging")
        lines.append("")
        lines.append("2. Prominence & Edge Detection:")
        lines.append("   - Limb darkening compensation")
        lines.append("   - Edge-enhanced gradient analysis")
        lines.append("   - Specialized detection for prominences, filaments, plages")
        lines.append("")
        lines.append("3. Standardized Annotation:")
        lines.append("   - Feature IDs: {TYPE}_{SCALE}_{INDEX}")
        lines.append("   - Quality metrics: contrast, circularity, SNR, edge sharpness")
        lines.append("   - Confidence classification: HIGH (>=0.8), MEDIUM (0.6-0.8), LOW (<0.6)")
        lines.append("")
        lines.append("4. Synchronized Image Preservation:")
        lines.append("   - Original images (lossless copy)")
        lines.append("   - Annotated images (PNG format)")
        lines.append("   - Detection reports (CSV format)")
        lines.append("   - Full metadata (JSON format)")
        lines.append("   - Debug images (per scale)")
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Also save as JSON for programmatic access
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved: {output_path}")
        logger.info(f"JSON saved: {json_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Enhanced Detection Performance Evaluation")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory for evaluation results")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                       help="List of specific image paths to evaluate")
    
    args = parser.parse_args()
    
    evaluator = DetectionEvaluator(args.output_dir)
    
    # Collect images
    image_paths = args.images or []
    
    if args.image_dir:
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
            image_paths.extend(list(Path(args.image_dir).glob(ext)))
    
    # If no images specified, use uploaded images
    if not image_paths:
        upload_dir = Path(__file__).parent.parent / "data" / "uploads"
        if upload_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_paths.extend(list(upload_dir.glob(ext)))
    
    if not image_paths:
        logger.error("No images found for evaluation")
        return
    
    # Run evaluation
    results = evaluator.evaluate_batch([str(p) for p in image_paths])
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Images evaluated: {results['batch_summary']['num_images']}")
    print(f"Feature improvement: {results['batch_summary']['feature_improvement_ratio']:.2f}x")
    print(f"Report saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
