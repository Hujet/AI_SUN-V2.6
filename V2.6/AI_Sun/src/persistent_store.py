"""
Persistent Data Store Module

Provides thread-safe JSON file-based persistent storage for images,
analysis reports, and tasks. Replaces the previous in-memory dicts
to ensure data survives server restarts.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class PersistentStore:
    """Thread-safe JSON-based persistent key-value store.

    Each store instance manages one collection (images, reports, tasks)
    backed by a single JSON file.
    """

    def __init__(self, file_path: Path, collection_name: str = "items"):
        self._file_path = Path(file_path)
        self._collection_name = collection_name
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the data file if it does not exist."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            self._write({self._collection_name: {}})

    def _read(self) -> Dict:
        """Read the full JSON content from disk."""
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {self._collection_name: {}}

    def _write(self, data: Dict) -> None:
        """Write the full JSON content to disk atomically."""
        tmp_path = self._file_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, self._file_path)

    def get(self, key: str, default: Any = None) -> Optional[Dict]:
        """Get an item by key."""
        with self._lock:
            data = self._read()
            items = data.get(self._collection_name, {})
            return items.get(key, default)

    def set(self, key: str, value: Dict) -> None:
        """Set/update an item by key."""
        with self._lock:
            data = self._read()
            if self._collection_name not in data:
                data[self._collection_name] = {}
            data[self._collection_name][key] = value
            self._write(data)

    def delete(self, key: str) -> bool:
        """Delete an item by key. Returns True if item existed."""
        with self._lock:
            data = self._read()
            items = data.get(self._collection_name, {})
            if key in items:
                del items[key]
                self._write(data)
                return True
            return False

    def list_all(self) -> List[Dict]:
        """List all items with their IDs embedded."""
        with self._lock:
            data = self._read()
            return list(data.get(self._collection_name, {}).values())

    def filter(self, predicate) -> List[Dict]:
        """Filter items by a predicate function."""
        with self._lock:
            data = self._read()
            return [v for v in data.get(self._collection_name, {}).values() if predicate(v)]

    def count(self) -> int:
        """Return the total number of items."""
        with self._lock:
            data = self._read()
            return len(data.get(self._collection_name, {}))

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._write({self._collection_name: {}})


# ---------------------------------------------------------------------------
# Global store instances
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def get_images_store() -> PersistentStore:
    return PersistentStore(DATA_DIR / "images_store.json", "images")

def get_reports_store() -> PersistentStore:
    return PersistentStore(DATA_DIR / "reports_store.json", "reports")

def get_tasks_store() -> PersistentStore:
    return PersistentStore(DATA_DIR / "tasks_store.json", "tasks")

def get_analysis_history_store() -> PersistentStore:
    return PersistentStore(DATA_DIR / "analysis_history.json", "history")
