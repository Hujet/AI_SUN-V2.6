"""
Token Usage Tracker Module

Tracks DeepSeek API token consumption per request with persistent storage.
Supports real-time statistics, daily/weekly/monthly aggregation reports.
"""

import os
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class TokenUsageRecord:
    """Single token usage record for one API call."""

    def __init__(
        self,
        task_id: str = "",
        image_id: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_estimate: float = 0.0,
        timestamp: str = "",
    ):
        self.task_id = task_id
        self.image_id = image_id
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.cost_estimate = cost_estimate
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "image_id": self.image_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_estimate": self.cost_estimate,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TokenUsageRecord":
        return TokenUsageRecord(**{k: v for k, v in data.items() if k in (
            "task_id", "image_id", "model", "input_tokens",
            "output_tokens", "total_tokens", "cost_estimate", "timestamp",
        )})


class TokenUsageTracker:
    """Thread-safe token usage tracker with persistent JSON storage."""

    # Pricing estimates (DeepSeek Chat, per 1K tokens in RMB)
    PRICING = {
        "deepseek-chat": {"input": 0.001, "output": 0.002},
        "deepseek-reasoner": {"input": 0.004, "output": 0.016},
    }

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent / "data" / "token_usage.json"
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

    def record(
        self,
        task_id: str,
        image_id: str,
        model: str,
        usage: Dict[str, int],
        cost: float = 0.0,
    ) -> TokenUsageRecord:
        """Record token usage for one API call.

        Args:
            task_id: Associated analysis task ID
            image_id: Associated image ID
            model: Model name used
            usage: Dict with keys like "prompt_tokens", "completion_tokens", "total_tokens"
            cost: Pre-calculated cost (if 0, auto-estimated)
        """
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        if cost == 0:
            pricing = self.PRICING.get(model, self.PRICING.get("deepseek-chat", {"input": 0.001, "output": 0.002}))
            cost = round((input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"]), 6)

        record = TokenUsageRecord(
            task_id=task_id,
            image_id=image_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost,
        )

        with self._lock:
            data = self._read()
            data["records"].append(record.to_dict())
            self._write(data)

        return record

    def get_records(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Dict]:
        """Query token usage records with filters."""
        with self._lock:
            data = self._read()
        records = data.get("records", [])
        if start:
            records = [r for r in records if r.get("timestamp", "") >= start]
        if end:
            records = [r for r in records if r.get("timestamp", "") <= end]
        if task_id:
            records = [r for r in records if r.get("task_id") == task_id]
        return records

    def get_summary(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated token usage summary."""
        records = self.get_records(start=start, end=end)
        if not records:
            return {"total_requests": 0}

        total_input = sum(r.get("input_tokens", 0) for r in records)
        total_output = sum(r.get("output_tokens", 0) for r in records)
        total_cost = sum(r.get("cost_estimate", 0) for r in records)

        # Per-model breakdown
        model_breakdown: Dict[str, Dict] = {}
        for r in records:
            m = r.get("model", "unknown")
            if m not in model_breakdown:
                model_breakdown[m] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_cost": 0}
            model_breakdown[m]["requests"] += 1
            model_breakdown[m]["input_tokens"] += r.get("input_tokens", 0)
            model_breakdown[m]["output_tokens"] += r.get("output_tokens", 0)
            model_breakdown[m]["total_cost"] += r.get("cost_estimate", 0)

        return {
            "total_requests": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_estimate": round(total_cost, 4),
            "model_breakdown": model_breakdown,
            "avg_tokens_per_request": round((total_input + total_output) / len(records), 0),
        }

    def get_daily_report(self, days: int = 7) -> List[Dict]:
        """Get daily token usage for the last N days."""
        now = datetime.now()
        daily_data = {}
        for i in range(days):
            date_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_data[date_str] = {"requests": 0, "tokens": 0, "cost": 0}

        for r in self.get_records():
            day = r.get("timestamp", "")[:10]
            if day in daily_data:
                daily_data[day]["requests"] += 1
                daily_data[day]["tokens"] += r.get("total_tokens", 0)
                daily_data[day]["cost"] += r.get("cost_estimate", 0)

        return [
            {"date": d, **v} for d, v in sorted(daily_data.items(), reverse=True)
        ]

    def get_periodic_report(self, period: str = "weekly") -> Dict[str, Any]:
        """Generate periodic usage report (daily/weekly/monthly).

        Args:
            period: "daily", "weekly", or "monthly"

        Returns:
            Comprehensive report with usage statistics and trends
        """
        now = datetime.now()

        if period == "daily":
            start_date = (now - timedelta(days=1)).isoformat()
            title = "Daily Token Usage Report"
        elif period == "weekly":
            start_date = (now - timedelta(weeks=1)).isoformat()
            title = "Weekly Token Usage Report"
        elif period == "monthly":
            start_date = (now - timedelta(days=30)).isoformat()
            title = "Monthly Token Usage Report"
        else:
            start_date = (now - timedelta(weeks=1)).isoformat()
            title = "Weekly Token Usage Report"

        records = self.get_records(start=start_date)
        summary = self.get_summary(start=start_date)

        # Calculate trends
        daily_breakdown = {}
        for r in records:
            day = r.get("timestamp", "")[:10]
            if day not in daily_breakdown:
                daily_breakdown[day] = {"requests": 0, "tokens": 0, "cost": 0}
            daily_breakdown[day]["requests"] += 1
            daily_breakdown[day]["tokens"] += r.get("total_tokens", 0)
            daily_breakdown[day]["cost"] += r.get("cost_estimate", 0)

        # Sort by date
        sorted_days = sorted(daily_breakdown.keys())
        daily_list = [{"date": d, **daily_breakdown[d]} for d in sorted_days]

        # Calculate averages
        total_days = len(daily_list)
        avg_daily_tokens = summary.get("total_tokens", 0) / max(total_days, 1)
        avg_daily_cost = summary.get("total_cost_estimate", 0) / max(total_days, 1)

        # Find peak usage day
        peak_day = max(daily_list, key=lambda x: x["tokens"]) if daily_list else None

        return {
            "title": title,
            "period": period,
            "generated_at": now.isoformat(),
            "start_date": start_date,
            "summary": summary,
            "daily_breakdown": daily_list,
            "statistics": {
                "total_days": total_days,
                "average_daily_tokens": round(avg_daily_tokens, 0),
                "average_daily_cost": round(avg_daily_cost, 4),
                "peak_day": peak_day,
                "peak_tokens": peak_day["tokens"] if peak_day else 0,
                "peak_cost": peak_day["cost"] if peak_day else 0,
            },
            "model_breakdown": summary.get("model_breakdown", {}),
            "cost_analysis": {
                "total_cost": summary.get("total_cost_estimate", 0),
                "average_cost_per_request": round(
                    summary.get("total_cost_estimate", 0) / max(summary.get("total_requests", 1), 1), 4
                ),
                "projected_monthly_cost": round(avg_daily_cost * 30, 4),
            },
        }

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._write({"records": []})
