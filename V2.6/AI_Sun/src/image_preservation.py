"""
Synchronized Image Preservation Module

Provides standardized storage for:
- Original images (unchanged, lossless)
- Annotated images (with detection markers)
- Detection reports (CSV/JSON metadata)
- Quality metrics and traceability data

Storage Structure:
ai_sun/data/enhanced_results/
├── {session_id}/
│   ├── original_{image_id}.{ext}          # Original image
│   ├── annotated_{image_id}.png           # Annotated image
│   ├── report_{image_id}.csv              # Detection report
│   ├── metadata_{image_id}.json           # Full metadata
│   └── debug/
│       ├── multiscale_{scale}.png         # Scale debug images
│       └── prominence.png                 # Prominence debug image
"""

import logging
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Base storage directory
ENHANCED_RESULTS_DIR = Path(__file__).parent.parent / "data" / "enhanced_results"
ENHANCED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ImagePreservationManager:
    """Manages synchronized storage of original and annotated images."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or ENHANCED_RESULTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, image_id: str, original_path: str) -> Path:
        """Create a new session directory for storing detection results.
        
        Args:
            image_id: Unique identifier for the image
            original_path: Path to original image file
            
        Returns:
            Path to session directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{image_id}_{timestamp}"
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create debug subdirectory
        debug_dir = session_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created session: {session_dir}")
        return session_dir
    
    def save_original_image(
        self,
        session_dir: Path,
        image_id: str,
        original_path: str,
    ) -> str:
        """Save a copy of the original image (lossless).
        
        Args:
            session_dir: Session directory path
            image_id: Image identifier
            original_path: Path to original image
            
        Returns:
            Path to saved original image
        """
        original_ext = Path(original_path).suffix.lower()
        if original_ext not in ('.png', '.jpg', '.jpeg', '.tiff', '.tif'):
            original_ext = '.png'
        
        dest_path = session_dir / f"original_{image_id}{original_ext}"
        
        # Copy original file directly (no conversion)
        shutil.copy2(original_path, dest_path)
        
        logger.info(f"Saved original image: {dest_path}")
        return str(dest_path)
    
    def save_annotated_image(
        self,
        session_dir: Path,
        image_id: str,
        annotated_array: np.ndarray,
    ) -> str:
        """Save annotated image with detection markers.
        
        Args:
            session_dir: Session directory
            image_id: Image identifier
            annotated_array: Annotated image as numpy array
            
        Returns:
            Path to saved annotated image
        """
        dest_path = session_dir / f"annotated_{image_id}.png"
        
        # Save as PNG (lossless)
        cv2.imwrite(str(dest_path), annotated_array)
        
        logger.info(f"Saved annotated image: {dest_path}")
        return str(dest_path)
    
    def save_detection_report(
        self,
        session_dir: Path,
        image_id: str,
        report_content: str,
        report_format: str = "csv",
    ) -> str:
        """Save detection report.
        
        Args:
            session_dir: Session directory
            image_id: Image identifier
            report_content: Report content as string
            report_format: 'csv' or 'txt'
            
        Returns:
            Path to saved report
        """
        dest_path = session_dir / f"report_{image_id}.{report_format}"
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Saved detection report: {dest_path}")
        return str(dest_path)
    
    def save_metadata(
        self,
        session_dir: Path,
        image_id: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Save full detection metadata as JSON.
        
        Args:
            session_dir: Session directory
            image_id: Image identifier
            metadata: Metadata dictionary
            
        Returns:
            Path to saved metadata file
        """
        dest_path = session_dir / f"metadata_{image_id}.json"
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Saved metadata: {dest_path}")
        return str(dest_path)
    
    def save_debug_image(
        self,
        session_dir: Path,
        image_id: str,
        debug_name: str,
        debug_array: np.ndarray,
    ) -> str:
        """Save debug/intermediate image.
        
        Args:
            session_dir: Session directory
            image_id: Image identifier
            debug_name: Name of debug image (e.g., 'multiscale_original')
            debug_array: Debug image as numpy array
            
        Returns:
            Path to saved debug image
        """
        debug_dir = session_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        dest_path = debug_dir / f"{debug_name}_{image_id}.png"
        cv2.imwrite(str(dest_path), debug_array)
        
        return str(dest_path)
    
    def save_all(
        self,
        image_id: str,
        original_path: str,
        annotated_array: np.ndarray,
        report_content: str,
        metadata: Dict[str, Any],
        debug_images: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, str]:
        """Save all detection artifacts in one call.
        
        Args:
            image_id: Image identifier
            original_path: Path to original image
            annotated_array: Annotated image
            report_content: Detection report
            metadata: Full metadata
            debug_images: Optional debug images dict
            
        Returns:
            Dictionary mapping artifact type to file path
        """
        # Create session
        session_dir = self.create_session(image_id, original_path)
        
        saved_paths = {}
        
        # Save original
        saved_paths['original'] = self.save_original_image(
            session_dir, image_id, original_path
        )
        
        # Save annotated
        saved_paths['annotated'] = self.save_annotated_image(
            session_dir, image_id, annotated_array
        )
        
        # Save report
        saved_paths['report'] = self.save_detection_report(
            session_dir, image_id, report_content
        )
        
        # Save metadata
        saved_paths['metadata'] = self.save_metadata(
            session_dir, image_id, metadata
        )
        
        # Save debug images
        if debug_images:
            saved_paths['debug'] = {}
            for name, img_array in debug_images.items():
                path = self.save_debug_image(
                    session_dir, image_id, name, img_array
                )
                saved_paths['debug'][name] = path
        
        saved_paths['session_dir'] = str(session_dir)
        
        logger.info(f"All artifacts saved for {image_id}: {saved_paths.keys()}")
        return saved_paths
    
    def get_session_list(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent detection sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session info dictionaries
        """
        sessions = []
        
        for session_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            
            # Look for metadata file
            metadata_files = list(session_dir.glob("metadata_*.json"))
            if not metadata_files:
                continue
            
            try:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                
                sessions.append({
                    "session_id": session_dir.name,
                    "session_path": str(session_dir),
                    "timestamp": metadata.get("timestamp", ""),
                    "total_features": metadata.get("statistics", {}).get("total_features", 0),
                    "image_path": metadata.get("image_path", ""),
                })
            except Exception as e:
                logger.warning(f"Failed to read session {session_dir}: {e}")
            
            if len(sessions) >= limit:
                break
        
        return sessions
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a detection session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session details dictionary or None
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return None
        
        metadata_files = list(session_dir.glob("metadata_*.json"))
        if not metadata_files:
            return None
        
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            
            # List all files in session
            files = {
                'original': [],
                'annotated': [],
                'report': [],
                'debug': [],
            }
            
            for f in session_dir.iterdir():
                if f.is_file():
                    if f.name.startswith("original_"):
                        files['original'].append(str(f))
                    elif f.name.startswith("annotated_"):
                        files['annotated'].append(str(f))
                    elif f.name.startswith("report_"):
                        files['report'].append(str(f))
                    elif f.name.startswith("metadata_"):
                        files['metadata'] = str(f)
            
            debug_dir = session_dir / "debug"
            if debug_dir.exists():
                for f in debug_dir.iterdir():
                    if f.is_file():
                        files['debug'].append(str(f))
            
            return {
                "session_id": session_id,
                "metadata": metadata,
                "files": files,
            }
        except Exception as e:
            logger.error(f"Failed to read session {session_id}: {e}")
            return None
    
    def get_session_image(self, session_id: str, image_type: str) -> Optional[str]:
        """Get path to a specific image in a session.
        
        Args:
            session_id: Session identifier
            image_type: 'original' or 'annotated'
            
        Returns:
            Path to image file or None
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return None
        
        prefix = f"{image_type}_"
        for f in session_dir.iterdir():
            if f.is_file() and f.name.startswith(prefix):
                return str(f)
        
        return None
