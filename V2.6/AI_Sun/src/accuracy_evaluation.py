"""
Solar Feature Detection Accuracy Evaluation System v1.0

Provides comprehensive accuracy assessment for AI-based solar feature detection:
- Prominence (日珥) detection accuracy metrics
- Flare (耀斑) detection accuracy metrics
- Feature extraction algorithm optimization
- Model training precision improvement tracking
- Statistical analysis of detection reliability

Key metrics:
- Precision (精确率): TP / (TP + FP)
- Recall (召回率): TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- False Positive Rate (误报率): FP / (FP + TN)
- False Negative Rate (漏报率): FN / (TP + FN)
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Single detection result with ground truth comparison."""
    feature_id: str = ""
    feature_type: str = ""  # prominence, flare, sunspot, etc.
    predicted: bool = True
    ground_truth: bool = False
    confidence: float = 0.0
    position: Dict[str, float] = field(default_factory=dict)
    size: float = 0.0
    is_correct: bool = False
    error_type: str = ""  # false_positive, false_negative, true_positive, true_negative


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for a specific feature type."""
    feature_type: str = ""
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy: float = 0.0
    mean_confidence: float = 0.0
    std_confidence: float = 0.0
    evaluation_date: str = ""
    sample_size: int = 0
    notes: str = ""

    def calculate(self):
        """Calculate all metrics from confusion matrix values."""
        tp = self.true_positives
        fp = self.false_positives
        tn = self.true_negatives
        fn = self.false_negatives
        
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0
        self.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        self.false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        self.accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        self.evaluation_date = datetime.now().isoformat()


@dataclass
class EvaluationReport:
    """Complete evaluation report for all feature types."""
    evaluation_id: str = ""
    evaluation_date: str = ""
    total_images_analyzed: int = 0
    prominence_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    flare_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    sunspot_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    overall_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    recommendations: List[str] = field(default_factory=list)
    improvement_actions: List[str] = field(default_factory=list)
    detailed_results: List[DetectionResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_date": self.evaluation_date,
            "total_images_analyzed": self.total_images_analyzed,
            "prominence_metrics": asdict(self.prominence_metrics),
            "flare_metrics": asdict(self.flare_metrics),
            "sunspot_metrics": asdict(self.sunspot_metrics),
            "overall_metrics": asdict(self.overall_metrics),
            "recommendations": self.recommendations,
            "improvement_actions": self.improvement_actions,
        }


# ---------------------------------------------------------------------------
# Accuracy Evaluation Engine
# ---------------------------------------------------------------------------

class AccuracyEvaluator:
    """
    Evaluates the accuracy of solar feature detection algorithms.
    
    Provides:
    - Ground truth comparison
    - Statistical analysis
    - Performance tracking over time
    - Optimization recommendations
    """
    
    def __init__(self, evaluation_dir: str = "AI_Sun/src/data/evaluations"):
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[DetectionResult] = []
        
    def add_detection_result(self, result: DetectionResult):
        """Add a single detection result for evaluation."""
        result.is_correct = (result.predicted == result.ground_truth)
        if result.predicted and not result.ground_truth:
            result.error_type = "false_positive"
        elif not result.predicted and result.ground_truth:
            result.error_type = "false_negative"
        elif result.predicted and result.ground_truth:
            result.error_type = "true_positive"
        else:
            result.error_type = "true_negative"
        self.results.append(result)
        
    def calculate_metrics(self, feature_type: str) -> AccuracyMetrics:
        """Calculate accuracy metrics for a specific feature type."""
        metrics = AccuracyMetrics(feature_type=feature_type)
        
        type_results = [r for r in self.results if r.feature_type == feature_type]
        metrics.sample_size = len(type_results)
        
        for r in type_results:
            if r.predicted and r.ground_truth:
                metrics.true_positives += 1
            elif r.predicted and not r.ground_truth:
                metrics.false_positives += 1
            elif not r.predicted and r.ground_truth:
                metrics.false_negatives += 1
            else:
                metrics.true_negatives += 1
                
        metrics.calculate()
        
        # Calculate confidence statistics
        confidences = [r.confidence for r in type_results if r.confidence > 0]
        if confidences:
            metrics.mean_confidence = sum(confidences) / len(confidences)
            if len(confidences) > 1:
                variance = sum((c - metrics.mean_confidence) ** 2 for c in confidences) / len(confidences)
                metrics.std_confidence = math.sqrt(variance)
                
        return metrics
        
    def generate_evaluation_report(self, evaluation_id: str = "") -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        report = EvaluationReport(
            evaluation_id=evaluation_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            evaluation_date=datetime.now().isoformat(),
            total_images_analyzed=len(set(r.feature_id for r in self.results)),
            prominence_metrics=self.calculate_metrics("prominence"),
            flare_metrics=self.calculate_metrics("flare"),
            sunspot_metrics=self.calculate_metrics("sunspot"),
            detailed_results=self.results,
        )
        
        # Calculate overall metrics
        overall = AccuracyMetrics(feature_type="overall")
        overall.true_positives = (report.prominence_metrics.true_positives + 
                                 report.flare_metrics.true_positives + 
                                 report.sunspot_metrics.true_positives)
        overall.false_positives = (report.prominence_metrics.false_positives + 
                                  report.flare_metrics.false_positives + 
                                  report.sunspot_metrics.false_positives)
        overall.true_negatives = (report.prominence_metrics.true_negatives + 
                                 report.flare_metrics.true_negatives + 
                                 report.sunspot_metrics.true_negatives)
        overall.false_negatives = (report.prominence_metrics.false_negatives + 
                                  report.flare_metrics.false_negatives + 
                                  report.sunspot_metrics.false_negatives)
        overall.calculate()
        report.overall_metrics = overall
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        report.improvement_actions = self._generate_improvement_actions(report)
        
        # Save report
        self._save_report(report)
        
        return report
        
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        # Prominence detection
        if report.prominence_metrics.recall < 0.7:
            recommendations.append("日珥检测召回率较低，建议增加日面边缘区域的检测灵敏度")
        if report.prominence_metrics.precision < 0.7:
            recommendations.append("日珥检测精确率较低，建议优化边缘特征提取算法，减少误报")
            
        # Flare detection
        if report.flare_metrics.recall < 0.7:
            recommendations.append("耀斑检测召回率较低，建议增强极亮区域的检测能力")
        if report.flare_metrics.precision < 0.7:
            recommendations.append("耀斑检测精确率较低，建议区分耀斑与其他亮区（谱斑、光斑）")
            
        # Overall
        if report.overall_metrics.f1_score < 0.75:
            recommendations.append("整体F1分数低于0.75，建议增加训练样本数量")
        if report.overall_metrics.mean_confidence < 0.7:
            recommendations.append("平均置信度较低，建议优化模型训练精度")
            
        if not recommendations:
            recommendations.append("检测性能良好，继续保持当前算法配置")
            
        return recommendations
        
    def _generate_improvement_actions(self, report: EvaluationReport) -> List[str]:
        """Generate specific improvement actions."""
        actions = [
            "增加样本数量：收集更多包含日珥和耀斑的太阳图像",
            "优化特征提取：改进边缘检测和亮度异常分析算法",
            "提升模型训练：使用更多标注数据训练AI模型",
            "多尺度检测：实施多尺度策略检测不同大小的特征",
            "交叉验证：使用多种检测方法相互验证结果",
            "持续监测：建立长期性能跟踪机制",
        ]
        return actions
        
    def _save_report(self, report: EvaluationReport):
        """Save evaluation report to file."""
        report_path = self.evaluation_dir / f"{report.evaluation_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation report saved: {report_path}")
        
    def load_ground_truth(self, ground_truth_file: str):
        """Load ground truth data for comparison."""
        gt_path = Path(ground_truth_file)
        if not gt_path.exists():
            logger.warning(f"Ground truth file not found: {ground_truth_file}")
            return
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        # Process ground truth data
        # Expected format: {"images": [{"id": "...", "features": [...]}]}
        for img in gt_data.get("images", []):
            img_id = img.get("id", "")
            for feat in img.get("features", []):
                self.add_detection_result(DetectionResult(
                    feature_id=img_id,
                    feature_type=feat.get("type", ""),
                    predicted=False,  # Will be updated when AI results are added
                    ground_truth=True,
                    position=feat.get("position", {}),
                    size=feat.get("size", 0.0),
                ))
                
    def compare_with_ai_results(self, ai_results: List[Dict]):
        """Compare AI detection results with ground truth."""
        for result in ai_results:
            self.add_detection_result(DetectionResult(
                feature_id=result.get("image_id", ""),
                feature_type=result.get("feature_type", ""),
                predicted=True,
                ground_truth=result.get("ground_truth", False),
                confidence=result.get("confidence", 0.0),
                position=result.get("position", {}),
                size=result.get("size", 0.0),
            ))


# ---------------------------------------------------------------------------
# Feature Extraction Optimization
# ---------------------------------------------------------------------------

class FeatureExtractionOptimizer:
    """
    Optimizes feature extraction algorithms for better accuracy.
    
    Provides:
    - Algorithm parameter tuning
    - Multi-scale detection strategies
    - Feature validation and filtering
    """
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
        
    def optimize_prominence_detection(self, current_params: Dict) -> Dict:
        """Optimize prominence detection parameters."""
        optimized = current_params.copy()
        
        # Increase edge detection sensitivity
        if "edge_threshold" in optimized:
            optimized["edge_threshold"] = max(optimized["edge_threshold"] * 0.9, 10)
            
        # Enhance limb region analysis
        if "limb_region_width" in optimized:
            optimized["limb_region_width"] = min(optimized["limb_region_width"] * 1.2, 50)
            
        # Improve brightness contrast
        if "brightness_contrast_ratio" in optimized:
            optimized["brightness_contrast_ratio"] = min(optimized["brightness_contrast_ratio"] * 1.1, 2.0)
            
        self.optimization_history.append({
            "type": "prominence",
            "timestamp": datetime.now().isoformat(),
            "params_before": current_params,
            "params_after": optimized,
        })
        
        return optimized
        
    def optimize_flare_detection(self, current_params: Dict) -> Dict:
        """Optimize flare detection parameters."""
        optimized = current_params.copy()
        
        # Increase brightness threshold for flare detection
        if "brightness_threshold" in optimized:
            optimized["brightness_threshold"] = min(optimized["brightness_threshold"] * 1.15, 255)
            
        # Enhance multi-scale analysis
        if "scales" in optimized:
            optimized["scales"] = [s * 0.9 for s in optimized["scales"] if s > 5]
            
        # Improve temporal consistency
        if "temporal_window" in optimized:
            optimized["temporal_window"] = max(optimized["temporal_window"] * 1.1, 1)
            
        self.optimization_history.append({
            "type": "flare",
            "timestamp": datetime.now().isoformat(),
            "params_before": current_params,
            "params_after": optimized,
        })
        
        return optimized
        
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations based on optimization history."""
        if not self.optimization_history:
            return ["暂无优化记录，建议先运行准确度评估"]
            
        recommendations = []
        for opt in self.optimization_history[-5:]:  # Last 5 optimizations
            recommendations.append(
                f"{opt['type']}检测参数已优化: "
                f"时间={opt['timestamp'][:19]}"
            )
            
        return recommendations


# ---------------------------------------------------------------------------
# Model Training Precision Tracker
# ---------------------------------------------------------------------------

class ModelPrecisionTracker:
    """
    Tracks model training precision over time.
    
    Provides:
    - Training history logging
    - Precision trend analysis
    - Model version comparison
    """
    
    def __init__(self, tracker_file: str = "AI_Sun/src/data/model_precision.json"):
        self.tracker_file = Path(tracker_file)
        self.history: List[Dict] = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load training history from file."""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
        
    def log_training_result(self, model_version: str, metrics: Dict):
        """Log a training result."""
        entry = {
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.history.append(entry)
        self._save_history()
        
    def get_precision_trend(self, last_n: int = 10) -> List[Dict]:
        """Get precision trend for last N training sessions."""
        return self.history[-last_n:]
        
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions."""
        v1_metrics = None
        v2_metrics = None
        
        for entry in self.history:
            if entry["model_version"] == version1:
                v1_metrics = entry["metrics"]
            if entry["model_version"] == version2:
                v2_metrics = entry["metrics"]
                
        if not v1_metrics or not v2_metrics:
            return {"error": "One or both versions not found"}
            
        comparison = {}
        for key in v1_metrics:
            if key in v2_metrics:
                comparison[key] = {
                    "version1": v1_metrics[key],
                    "version2": v2_metrics[key],
                    "improvement": v2_metrics[key] - v1_metrics[key],
                }
                
        return comparison
        
    def _save_history(self):
        """Save training history to file."""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create evaluator
    evaluator = AccuracyEvaluator()
    
    # Add sample detection results
    evaluator.add_detection_result(DetectionResult(
        feature_id="img_001",
        feature_type="prominence",
        predicted=True,
        ground_truth=True,
        confidence=0.85,
    ))
    
    evaluator.add_detection_result(DetectionResult(
        feature_id="img_002",
        feature_type="flare",
        predicted=True,
        ground_truth=False,
        confidence=0.72,
    ))
    
    # Generate report
    report = evaluator.generate_evaluation_report("test_eval_001")
    
    print(f"Prominence Precision: {report.prominence_metrics.precision:.2%}")
    print(f"Flare Recall: {report.flare_metrics.recall:.2%}")
    print(f"Overall F1: {report.overall_metrics.f1_score:.2%}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
