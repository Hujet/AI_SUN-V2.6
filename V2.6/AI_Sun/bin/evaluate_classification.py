import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class HaleClassificationEvaluator:
    HALE_CLASSES = ["Alpha", "Beta", "Beta-Gamma", "Gamma", "Delta", "Beta-Delta", "Unknown"]

    def __init__(self, ground_truth_file: Optional[str] = None):
        self.ground_truth_file = ground_truth_file
        self.ground_truth: Dict[str, str] = {}
        self.predictions: Dict[str, str] = {}
        self.confidences: Dict[str, float] = {}
        self.descriptions: Dict[str, str] = {}

        if ground_truth_file and os.path.exists(ground_truth_file):
            self.loadGroundTruth(ground_truth_file)

    def loadGroundTruth(self, filepath: str):
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    key = item.get("id") or item.get("filename") or item.get("name")
                    self.ground_truth[key] = item.get("hale_class", "Unknown")
        elif filepath.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get("id") or row.get("filename") or row.get("name") or row.get("noaa_id")
                    self.ground_truth[key] = row.get("hale_class", row.get("classification", "Unknown"))
        print(f"Loaded {len(self.ground_truth)} ground truth labels")

    def addPrediction(self, image_id: str, predicted_class: str, confidence: float = 1.0, description: str = ""):
        self.predictions[image_id] = predicted_class
        self.confidences[image_id] = confidence
        if description:
            self.descriptions[image_id] = description

    def loadPredictions(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                image_id = item.get("id") or item.get("filename") or item.get("name")
                predicted = item.get("predicted_class", item.get("hale_class", "Unknown"))
                confidence = item.get("confidence", 1.0)
                description = item.get("description", "")
                self.addPrediction(image_id, predicted, confidence, description)

    def loadResultsFromDeepseek(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                data = data["results"]
            for item in data:
                image_id = item.get("image_id") or item.get("filename") or item.get("image_path")
                predicted = item.get("classification", item.get("hale_class", item.get("predicted_class", "Unknown")))
                confidence = item.get("confidence", item.get("classification_confidence", 1.0))
                description = item.get("analysis", item.get("description", item.get("reasoning", "")))
                self.addPrediction(image_id, predicted, confidence, description)

    def normalizeClass(self, class_name: str) -> str:
        class_name = class_name.strip()
        class_mapping = {
            "alpha": "Alpha",
            "beta": "Beta",
            "beta-gamma": "Beta-Gamma",
            "beta-gamma-delta": "Beta-Gamma",
            "beta gamma": "Beta-Gamma",
            "gamma": "Gamma",
            "delta": "Delta",
            "beta-delta": "Beta-Delta",
            "unknown": "Unknown",
            "": "Unknown"
        }
        return class_mapping.get(class_name.lower(), class_name)

    def evaluate(self) -> Dict:
        if not self.ground_truth:
            print("No ground truth data available. Please load ground truth file.")
            return {}

        results = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
            "accuracy": 0.0,
            "by_class": {},
            "confusion_matrix": defaultdict(lambda: defaultdict(int)),
            "errors": []
        }

        for image_id, true_class in self.ground_truth.items():
            results["total"] += 1
            true_class_norm = self.normalizeClass(true_class)

            if image_id in self.predictions:
                pred_class = self.normalizeClass(self.predictions[image_id])
                pred_conf = self.confidences.get(image_id, 1.0)

                if pred_class == true_class_norm:
                    results["correct"] += 1
                    status = "correct"
                else:
                    results["incorrect"] += 1
                    status = "incorrect"
                    results["errors"].append({
                        "image_id": image_id,
                        "true_class": true_class_norm,
                        "predicted_class": pred_class,
                        "confidence": pred_conf,
                        "description": self.descriptions.get(image_id, "")
                    })

                results["confusion_matrix"][true_class_norm][pred_class] += 1
            else:
                results["missing"] += 1
                results["errors"].append({
                    "image_id": image_id,
                    "true_class": true_class_norm,
                    "predicted_class": "MISSING",
                    "error": "No prediction available"
                })

        for true_class in self.HALE_CLASSES:
            class_total = sum(1 for k, v in self.ground_truth.items()
                            if self.normalizeClass(v) == true_class and k in self.predictions)
            class_correct = sum(1 for k, v in self.ground_truth.items()
                               if self.normalizeClass(v) == true_class
                               and k in self.predictions
                               and self.normalizeClass(self.predictions[k]) == true_class)
            if class_total > 0:
                results["by_class"][true_class] = {
                    "total": class_total,
                    "correct": class_correct,
                    "accuracy": class_correct / class_total
                }

        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]

        return results

    def printReport(self, results: Dict):
        if not results:
            print("No evaluation results to display.")
            return

        print("\n" + "=" * 60)
        print("HALE CLASSIFICATION EVALUATION REPORT")
        print("=" * 60)

        print(f"\nOverall Performance:")
        print(f"  Total Samples:     {results['total']}")
        print(f"  Correct:           {results['correct']}")
        print(f"  Incorrect:         {results['incorrect']}")
        print(f"  Missing:           {results['missing']}")
        print(f"  Overall Accuracy:  {results['accuracy']:.2%}")

        if results.get("by_class"):
            print(f"\nPer-Class Performance:")
            print("-" * 50)
            print(f"{'Class':<15} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
            print("-" * 50)
            for cls, stats in sorted(results["by_class"].items()):
                print(f"{cls:<15} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']:.2%}")
            print("-" * 50)

        if results.get("confusion_matrix"):
            print(f"\nConfusion Matrix:")
            all_classes = sorted(set(self.HALE_CLASSES) | set(results["confusion_matrix"].keys()))
            header = "True\\Pred" + "".join(f"{c:>12}" for c in all_classes)
            print(header)
            print("-" * (12 * len(all_classes) + 12))
            for true_cls in all_classes:
                row = f"{true_cls:<12}"
                for pred_cls in all_classes:
                    count = results["confusion_matrix"][true_cls][pred_cls]
                    row += f"{count:>12}"
                print(row)

        if results.get("errors") and len(results["errors"]) <= 20:
            print(f"\nError Analysis:")
            for err in results["errors"]:
                if err.get("predicted_class") == "MISSING":
                    print(f"  [MISSING] {err['image_id']}: Expected {err['true_class']}")
                else:
                    print(f"  [ERROR] {err['image_id']}: Expected {err['true_class']}, Got {err['predicted_class']} (conf: {err.get('confidence', 0):.2f})")

        print("\n" + "=" * 60)

    def saveReport(self, results: Dict, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {filepath}")

    def generateSummaryCSV(self, results: Dict, filepath: str):
        if not results.get("errors"):
            print("No errors to save")
            return

        fieldnames = ["image_id", "true_class", "predicted_class", "confidence", "status", "description"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for err in results["errors"]:
                row = {
                    "image_id": err["image_id"],
                    "true_class": err["true_class"],
                    "predicted_class": err.get("predicted_class", ""),
                    "confidence": err.get("confidence", ""),
                    "status": "correct" if err.get("predicted_class") == "MISSING" else "incorrect",
                    "description": err.get("description", "")
                }
                writer.writerow(row)
        print(f"Summary CSV saved to: {filepath}")

def createSampleGroundTruth(output_dir: Path, num_samples: int = 10):
    sample_data = [
        {"id": "20240510_AR13664", "filename": "magnetogram_13664.jpg", "hale_class": "Beta-Gamma", "noaa_id": "13664"},
        {"id": "20240508_AR13663", "filename": "magnetogram_13663.jpg", "hale_class": "Beta", "noaa_id": "13663"},
        {"id": "20240505_AR13661", "filename": "magnetogram_13661.jpg", "hale_class": "Beta-Delta", "noaa_id": "13661"},
        {"id": "20240503_AR13658", "filename": "magnetogram_13658.jpg", "hale_class": "Alpha", "noaa_id": "13658"},
        {"id": "20240430_AR13652", "filename": "magnetogram_13652.jpg", "hale_class": "Beta", "noaa_id": "13652"},
        {"id": "20240428_AR13648", "filename": "magnetogram_13648.jpg", "hale_class": "Beta-Gamma", "noaa_id": "13648"},
        {"id": "20240425_AR13642", "filename": "magnetogram_13642.jpg", "hale_class": "Delta", "noaa_id": "13642"},
        {"id": "20240420_AR13635", "filename": "magnetogram_13635.jpg", "hale_class": "Alpha", "noaa_id": "13635"},
        {"id": "20240418_AR13630", "filename": "magnetogram_13630.jpg", "hale_class": "Beta", "noaa_id": "13630"},
        {"id": "20240415_AR13625", "filename": "magnetogram_13625.jpg", "hale_class": "Beta-Gamma", "noaa_id": "13625"},
    ]

    filepath = output_dir / "ground_truth.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sample_data[:num_samples], f, indent=2)
    print(f"Sample ground truth saved to: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Hale Classification Evaluation Tool")
    parser.add_argument("--ground-truth", "-g", type=str, help="Ground truth file (JSON or CSV)")
    parser.add_argument("--predictions", "-p", type=str, help="Predictions file (JSON)")
    parser.add_argument("--results-dir", "-r", type=str, help="Results output directory")
    parser.add_argument("--create-template", action="store_true", help="Create sample ground truth template")
    parser.add_argument("--output", "-o", type=str, help="Output file for evaluation report")

    args = parser.parse_args()

    if args.create_template:
        output_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent.parent / "resource" / "data"
        createSampleGroundTruth(output_dir)
        return

    evaluator = HaleClassificationEvaluator(ground_truth_file=args.ground_truth)

    if args.predictions:
        evaluator.loadPredictions(args.predictions)

    results = evaluator.evaluate()
    evaluator.printReport(results)

    if args.output:
        evaluator.saveReport(results, args.output)
        summary_file = args.output.replace(".json", "_errors.csv")
        evaluator.generateSummaryCSV(results, summary_file)

if __name__ == "__main__":
    main()