"""
User Feedback System

Collects and manages user feedback on AI model analysis results to continuously
improve feature detection accuracy and model selection strategy.
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(prefix="/api/v1/feedback", tags=["用户反馈"])

# Storage configuration
BASE_DIR = Path(__file__).parent.parent
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# Feedback index file
FEEDBACK_INDEX = FEEDBACK_DIR / "feedback_index.json"


def _load_index() -> List[Dict]:
    """Load the feedback index from disk."""
    if FEEDBACK_INDEX.exists():
        try:
            with open(FEEDBACK_INDEX, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_index(index: List[Dict]):
    """Save the feedback index to disk."""
    with open(FEEDBACK_INDEX, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    image_id: str
    analysis_id: str
    model_key: str
    
    # Verification results
    sunspot_verification: Optional[Dict[str, Any]] = None  # {correct_count, missed_count, false_positive_count}
    flare_verification: Optional[Dict[str, Any]] = None
    hale_verification: Optional[str] = None  # Correct Hale classification
    
    # Overall assessment
    accuracy_score: Optional[float] = None  # 0-1
    confidence_rating: Optional[str] = None  # low/medium/high
    comments: Optional[str] = None
    
    # Corrections
    corrections: Optional[List[Dict[str, Any]]] = None  # List of corrected features


@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for an analysis result."""
    feedback_id = str(uuid.uuid4())[:12]
    
    feedback = {
        'id': feedback_id,
        'image_id': request.image_id,
        'analysis_id': request.analysis_id,
        'model_key': request.model_key,
        'sunspot_verification': request.sunspot_verification,
        'flare_verification': request.flare_verification,
        'hale_verification': request.hale_verification,
        'accuracy_score': request.accuracy_score,
        'confidence_rating': request.confidence_rating,
        'comments': request.comments,
        'corrections': request.corrections,
        'submitted_at': datetime.now().isoformat(),
    }
    
    # Save feedback
    index = _load_index()
    index.append(feedback)
    _save_index(index)
    
    # Calculate accuracy metrics
    accuracy = request.accuracy_score or 0.0
    if request.sunspot_verification:
        correct = request.sunspot_verification.get('correct_count', 0)
        total = (correct + 
                request.sunspot_verification.get('missed_count', 0) + 
                request.sunspot_verification.get('false_positive_count', 0))
        if total > 0:
            accuracy = correct / total
    
    return {
        'success': True,
        'data': {
            'feedback_id': feedback_id,
            'accuracy': accuracy,
        },
        'message': '反馈已提交',
    }


@router.get("/statistics")
async def get_feedback_statistics(
    model_key: Optional[str] = Query(None, description="Filter by model"),
    days: int = Query(30, description="Last N days")
):
    """Get aggregated feedback statistics."""
    index = _load_index()
    
    # Filter by date range
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days)
    recent = [f for f in index if datetime.fromisoformat(f['submitted_at']) >= cutoff]
    
    # Filter by model if specified
    if model_key:
        recent = [f for f in recent if f.get('model_key') == model_key]
    
    if not recent:
        return {
            'success': True,
            'data': {
                'total_feedback': 0,
                'accuracy_by_model': {},
                'model_rankings': [],
            }
        }
    
    # Calculate accuracy by model
    model_stats = {}
    for feedback in recent:
        mk = feedback.get('model_key', 'unknown')
        if mk not in model_stats:
            model_stats[mk] = {'count': 0, 'total_accuracy': 0, 'ratings': []}
        
        model_stats[mk]['count'] += 1
        
        # Calculate accuracy
        accuracy = feedback.get('accuracy_score', 0.0)
        if feedback.get('sunspot_verification'):
            sv = feedback['sunspot_verification']
            correct = sv.get('correct_count', 0)
            total = correct + sv.get('missed_count', 0) + sv.get('false_positive_count', 0)
            if total > 0:
                accuracy = correct / total
        
        model_stats[mk]['total_accuracy'] += accuracy
        if feedback.get('confidence_rating'):
            model_stats[mk]['ratings'].append(feedback['confidence_rating'])
    
    # Calculate averages
    accuracy_by_model = {}
    for mk, stats in model_stats.items():
        avg_accuracy = stats['total_accuracy'] / stats['count'] if stats['count'] > 0 else 0
        accuracy_by_model[mk] = {
            'average_accuracy': round(avg_accuracy, 3),
            'total_feedback': stats['count'],
            'confidence_distribution': _count_ratings(stats['ratings']),
        }
    
    # Rank models by accuracy
    model_rankings = sorted(
        [{'model_key': k, **v} for k, v in accuracy_by_model.items()],
        key=lambda x: x['average_accuracy'],
        reverse=True
    )
    
    return {
        'success': True,
        'data': {
            'total_feedback': len(recent),
            'accuracy_by_model': accuracy_by_model,
            'model_rankings': model_rankings,
        }
    }


def _count_ratings(ratings: List[str]) -> Dict[str, int]:
    """Count rating distribution."""
    counts = {'high': 0, 'medium': 0, 'low': 0}
    for r in ratings:
        if r in counts:
            counts[r] += 1
    return counts


@router.get("/recent")
async def get_recent_feedback(limit: int = Query(20, description="Number of recent feedback to return")):
    """Get recent feedback submissions."""
    index = _load_index()
    # Sort by time (newest first)
    index.sort(key=lambda x: x['submitted_at'], reverse=True)
    return {
        'success': True,
        'data': {
            'items': index[:limit],
            'total': len(index),
        }
    }
