"""
Analysis Traceability Module

Provides complete traceability for every solar feature recognition operation:
- Input image metadata and pre-computation features
- Algorithm parameters and model configuration
- Intermediate computation results (raw model output, parsing details)
- Final recognition conclusion with confidence metrics
"""

import os
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class TraceabilityRecord:
    """Complete traceability record for one analysis."""

    def __init__(
        self,
        task_id: str,
        image_id: str,
        image_hash: str = "",
        input_metadata: Optional[Dict] = None,
        image_features: Optional[Dict] = None,
        algorithm_params: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        raw_model_output: str = "",
        parsing_intermediates: Optional[Dict] = None,
        final_result: Optional[Dict] = None,
        warnings: Optional[List[str]] = None,
        processing_steps: Optional[List[Dict]] = None,
        scientific_conclusion: str = "",
        flare_risk_assessment: Optional[Dict] = None,
    ):
        self.task_id = task_id
        self.image_id = image_id
        self.image_hash = image_hash
        self.input_metadata = input_metadata or {}
        self.image_features = image_features or {}
        self.algorithm_params = algorithm_params or {}
        self.model_config = model_config or {}
        self.raw_model_output = raw_model_output
        self.parsing_intermediates = parsing_intermediates or {}
        self.final_result = final_result or {}
        self.warnings = warnings or []
        self.processing_steps = processing_steps or []
        self.scientific_conclusion = scientific_conclusion
        self.flare_risk_assessment = flare_risk_assessment or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "image_id": self.image_id,
            "image_hash": self.image_hash,
            "input_metadata": self.input_metadata,
            "image_features": self.image_features,
            "algorithm_params": self.algorithm_params,
            "model_config": self.model_config,
            "raw_model_output": self.raw_model_output,
            "parsing_intermediates": self.parsing_intermediates,
            "final_result": self.final_result,
            "warnings": self.warnings,
            "processing_steps": self.processing_steps,
            "scientific_conclusion": self.scientific_conclusion,
            "flare_risk_assessment": self.flare_risk_assessment,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TraceabilityRecord":
        return TraceabilityRecord(**{k: v for k, v in data.items() if k in (
            "task_id", "image_id", "image_hash", "input_metadata",
            "image_features", "algorithm_params", "model_config",
            "raw_model_output", "parsing_intermediates", "final_result",
            "warnings", "processing_steps", "scientific_conclusion",
            "flare_risk_assessment", "created_at",
        )})


class TraceabilityStore:
    """Thread-safe persistent storage for traceability records."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent / "data" / "traceability.json"
        self._file = Path(storage_path)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._file.exists():
            self._write({"records": []})

    def _read(self) -> Dict:
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"records": []}

    def _write(self, data: Dict) -> None:
        tmp = self._file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self._file)

    def save(self, record: TraceabilityRecord) -> None:
        """Save a traceability record."""
        with self._lock:
            data = self._read()
            # Update if exists, append if new
            existing = [i for i, r in enumerate(data["records"]) if r.get("task_id") == record.task_id]
            if existing:
                data["records"][existing[0]] = record.to_dict()
            else:
                data["records"].append(record.to_dict())
            self._write(data)

    def get(self, task_id: str) -> Optional[Dict]:
        """Get traceability record by task_id."""
        with self._lock:
            data = self._read()
        for r in data.get("records", []):
            if r.get("task_id") == task_id:
                return r
        return None

    def list_all(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict]:
        """List all traceability records with optional date filter."""
        with self._lock:
            data = self._read()
        records = data.get("records", [])
        if start:
            records = [r for r in records if r.get("created_at", "") >= start]
        if end:
            records = [r for r in records if r.get("created_at", "") <= end]
        return records


def compute_image_hash(image_path: str) -> str:
    """Compute SHA-256 hash of image file for traceability."""
    sha256 = hashlib.sha256()
    try:
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "hash_unavailable"
