"""
Solar Feature Image Annotator

Generates annotated solar images with:
- Individual sunspot markers (numbered circles)
- Sunspot group boundaries (dashed rectangles)
- Flare region markers (star symbols)
- Detection report overlay
- Interactive checkbox support data export
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Configure Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Output directory
ANNOTATED_DIR = Path(__file__).parent.parent / "data" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COLOR_MAP = {
    "sunspot": "#00FF00",       # Green for individual sunspots
    "sunspot_group": "#00BFFF",  # Cyan for sunspot groups
    "flare": "#FF4444",         # Red for flares
    "bright_region": "#FFD700", # Gold for bright regions
    "plage": "#00BFFF",         # Light blue for plage
    "filament": "#9370DB",      # Purple for filaments
    "coronal_hole": "#20B2AA",  # Teal for coronal holes
    "prominence": "#FFD700",    # Gold for prominences (bright limb features)
    "facula": "#DDA0DD",        # Plum for facula
}

MARKER_SHAPES = {
    "sunspot": "circle",
    "sunspot_group": "rectangle",
    "flare": "star",
    "bright_region": "diamond",
    "plage": "triangle",
    "filament": "line",
    "coronal_hole": "ellipse",
}


# ---------------------------------------------------------------------------
# Detection Report Data Structure
# ---------------------------------------------------------------------------

class DetectionReport:
    """Detection report with sunspot/flare counts and positions."""

    def __init__(self, image_id: str = "", image_path: str = ""):
        self.image_id = image_id
        self.image_path = image_path
        self.generated_at = datetime.now().isoformat()
        self.sunspots: List[Dict] = []
        self.sunspot_groups: List[Dict] = []
        self.flares: List[Dict] = []
        self.prominences: List[Dict] = []
        self.other_features: List[Dict] = []
        self.total_sunspots = 0
        self.total_flares = 0
        self.total_prominences = 0
        self.hale_classification = "Unknown"
        self.complexity_score = 0.0
        self.disk_info: Dict = {}
        self.img_width: int = 0
        self.img_height: int = 0

    def add_sunspot(self, label: str, position: Dict, size: float,
                    confidence: float, group_id: str = None,
                    umbra_size: float = 0, penumbra_size: float = 0,
                    additional_params: Dict = None) -> int:
        """Add an individual sunspot. Returns the spot index (1-based)."""
        idx = len(self.sunspots) + 1
        self.sunspots.append({
            "index": idx,
            "label": label,
            "position": position,
            "size_relative": size,
            "confidence": confidence,
            "group_id": group_id,
            "umbra_size": umbra_size,
            "penumbra_size": penumbra_size,
            "additional_params": additional_params or {},
            "checked": True,
        })
        self.total_sunspots = len(self.sunspots)
        return idx

    def add_sunspot_group(self, label: str, position: Dict, size: float,
                          confidence: float, spot_count: int = 0,
                          group_spots: List[Dict] = None,
                          additional_params: Dict = None) -> int:
        """Add a sunspot group. Returns the group index (1-based)."""
        idx = len(self.sunspot_groups) + 1
        self.sunspot_groups.append({
            "index": idx,
            "label": label,
            "position": position,
            "size_relative": size,
            "confidence": confidence,
            "spot_count": spot_count,
            "group_spots": group_spots or [],
            "additional_params": additional_params or {},
            "checked": True,
        })
        return idx

    def add_flare(self, label: str, position: Dict, size: float,
                  confidence: float, flare_class: str = "",
                  intensity: float = 0, additional_params: Dict = None) -> int:
        """Add a flare. Returns the flare index (1-based)."""
        idx = len(self.flares) + 1
        self.flares.append({
            "index": idx,
            "label": label,
            "position": position,
            "size_relative": size,
            "confidence": confidence,
            "flare_class": flare_class,
            "intensity": intensity,
            "additional_params": additional_params or {},
            "checked": True,
        })
        self.total_flares = len(self.flares)
        return idx

    def add_other(self, feature_type: str, label: str, position: Dict,
                  size: float, confidence: float) -> int:
        """Add other feature type."""
        idx = len(self.other_features) + 1
        self.other_features.append({
            "index": idx,
            "type": feature_type,
            "label": label,
            "position": position,
            "size_relative": size,
            "confidence": confidence,
            "checked": True,
        })
        return idx

    def add_prominence(self, label: str, position: Dict, size: float,
                       confidence: float, intensity: float = 0,
                       additional_params: Dict = None) -> int:
        """Add a prominence (日珥) detection. Returns the prominence index (1-based)."""
        idx = len(self.prominences) + 1
        self.prominences.append({
            "index": idx,
            "label": label,
            "position": position,
            "size_relative": size,
            "confidence": confidence,
            "intensity": intensity,
            "additional_params": additional_params or {},
            "checked": True,
        })
        self.total_prominences = len(self.prominences)
        return idx

    def to_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "generated_at": self.generated_at,
            "hale_classification": self.hale_classification,
            "complexity_score": self.complexity_score,
            "disk_info": self.disk_info,
            "summary": {
                "total_sunspots": self.total_sunspots,
                "total_flares": self.total_flares,
                "total_prominences": self.total_prominences,
                "total_sunspot_groups": len(self.sunspot_groups),
                "total_other_features": len(self.other_features),
                "total_features": (self.total_sunspots + self.total_flares +
                    self.total_prominences + len(self.sunspot_groups) +
                    len(self.other_features)),
            },
            "sunspots": self.sunspots,
            "sunspot_groups": self.sunspot_groups,
            "flares": self.flares,
            "prominences": self.prominences,
            "other_features": self.other_features,
        }

    def to_csv(self) -> str:
        """Generate CSV export of detection results."""
        lines = ["# Solar Feature Detection Report"]
        lines.append(f"# Image: {self.image_path}")
        lines.append(f"# Generated: {self.generated_at}")
        lines.append(f"# Hale Classification: {self.hale_classification}")
        lines.append(f"# Complexity Score: {self.complexity_score}")
        lines.append(f"# Total Sunspots: {self.total_sunspots}")
        lines.append(f"# Total Flares: {self.total_flares}")
        lines.append("")
        lines.append("Type,Index,Label,X,Y,Size,Confidence,Group_ID,Checked")

        for s in self.sunspots:
            pos = s.get("position", {})
            lines.append(f"sunspot,{s['index']},{s['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{s['size_relative']:.4f},{s['confidence']:.4f},{s.get('group_id','')},{s.get('checked',True)}")

        for g in self.sunspot_groups:
            pos = g.get("position", {})
            lines.append(f"sunspot_group,{g['index']},{g['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{g['size_relative']:.4f},{g['confidence']:.4f},,{g.get('checked',True)}")

        for f in self.flares:
            pos = f.get("position", {})
            lines.append(f"flare,{f['index']},{f['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{f['size_relative']:.4f},{f['confidence']:.4f},,{f.get('checked',True)}")

        for o in self.other_features:
            pos = o.get("position", {})
            lines.append(f"{o['type']},{o['index']},{o['label']},{pos.get('x',0):.4f},{pos.get('y',0):.4f},{o['size_relative']:.4f},{o['confidence']:.4f},,{o.get('checked',True)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image Annotation Functions
# ---------------------------------------------------------------------------

def generate_annotated_image(
    image_path: str,
    report: DetectionReport,
    output_id: str,
    show_checkboxes: bool = True,
) -> str:
    """Generate annotated solar image with individual feature markers.

    Each sunspot gets a numbered circle marker.
    Each flare gets a star marker.
    Sunspot groups get dashed boundary rectangles.
    """
    if not HAS_MATPLOTLIB:
        return ""

    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        fig, ax = plt.subplots(figsize=(14, 11))
        ax.imshow(img_array)

        # Use relative coordinates (0~1) to convert to pixel positions
        disk_info = report.disk_info or {}
        disk_cx = disk_info.get("center_x", w / 2)
        disk_cy = disk_info.get("center_y", h / 2)
        disk_r = disk_info.get("radius", min(w, h) / 2)
        disk_diameter = disk_r * 2

        # Normalized disk info (for coordinate conversion)
        norm_cx = disk_info.get("normalized_center_x", disk_cx / w if w > 0 else 0.5)
        norm_cy = disk_info.get("normalized_center_y", disk_cy / h if h > 0 else 0.5)
        norm_r = disk_info.get("normalized_radius", disk_r / min(w, h) if min(w, h) > 0 else 0.4)

        def _rel_to_pixel(rx: float, ry: float) -> Tuple[int, int]:
            """Convert relative coords (0~1, origin=top-left) to pixel coords."""
            px = int(rx * w)
            py = int(ry * h)
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            return px, py

        spot_counter = 0
        group_counter = 0
        flare_counter = 0
        prominence_counter = 0

        # Draw individual sunspots with rectangles
        for spot in report.sunspots:
            if not spot.get("checked", True):
                continue
            spot_counter += 1
            pos = spot.get("position", {})
            size_rel = spot.get("size_relative", 0.05)
            box_size = int(disk_diameter * size_rel)
            box_size = max(15, min(box_size, 200))  # clamp 15-200px
            x, y = _rel_to_pixel(pos.get("x", 0.5), pos.get("y", 0.5))
            color = COLOR_MAP.get("sunspot", "#00FF00")

            # Draw rectangle around sunspot
            rect = mpatches.Rectangle((x - box_size // 2, y - box_size // 2),
                                      box_size, box_size,
                                      fill=False, edgecolor=color, linewidth=2, alpha=0.9)
            ax.add_patch(rect)

            # Draw crosshair at center
            ax.plot(x, y, "+", color=color, markersize=10, markeredgewidth=2)

            # Number label
            label = f"#{spot_counter}"
            ax.annotate(label, xy=(x, y - box_size // 2 - 3), fontsize=9,
                        color="white", fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="square,pad=0.2",
                                  facecolor=color, edgecolor="white", alpha=0.9))

        # Draw sunspot groups
        for group in report.sunspot_groups:
            if not group.get("checked", True):
                continue
            group_counter += 1
            pos = group.get("position", {})
            size_rel = group.get("size_relative", 0.1)
            box_size = int(disk_diameter * size_rel)
            box_size = max(20, min(box_size, 300))  # groups can be larger
            x, y = _rel_to_pixel(pos.get("x", 0), pos.get("y", 0))
            color = COLOR_MAP.get("sunspot_group", "#00BFFF")

            # Draw dashed rectangle around group
            rect = mpatches.Rectangle((x - box_size // 2, y - box_size // 2),
                                      box_size, box_size,
                                      fill=False, edgecolor=color,
                                      linewidth=2, linestyle="--", alpha=0.8)
            ax.add_patch(rect)

            # Group label
            spot_count = group.get("spot_count", 0)
            label = f"G{group_counter}({spot_count}spots)"
            ax.annotate(label, xy=(x, y + box_size // 2 + 5), fontsize=8,
                        color=color, fontweight="bold",
                        ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="black", edgecolor=color, alpha=0.8))

        # Draw flares
        for flare in report.flares:
            if not flare.get("checked", True):
                continue
            flare_counter += 1
            pos = flare.get("position", {})
            size_rel = flare.get("size_relative", 0.08)
            size_px = int(disk_diameter * size_rel)
            size_px = max(10, min(size_px, 150))
            x, y = _rel_to_pixel(pos.get("x", 0), pos.get("y", 0))
            color = COLOR_MAP.get("flare", "#FF4444")

            # Draw flare region circle
            circle = mpatches.Circle((x, y), radius=size_px, fill=False,
                                     edgecolor=color, linewidth=2,
                                     linestyle="-.", alpha=0.9)
            ax.add_patch(circle)

            # Star marker at center
            ax.plot(x, y, "*", color=color, markersize=15,
                    markeredgecolor="white", markeredgewidth=1)

            # Flare label
            flare_class = flare.get("flare_class", "")
            label = f"Flare#{flare_counter}"
            if flare_class:
                label += f" ({flare_class})"
            ax.annotate(label, xy=(x, y - size_px - 3), fontsize=8,
                        color="white", fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, edgecolor="white", alpha=0.8))

        # Draw other features
        for feat in report.other_features:
            if not feat.get("checked", True):
                continue
            ftype = feat.get("type", "unknown")
            pos = feat.get("position", {})
            size_rel = feat.get("size_relative", 0.05)
            size_px = int(disk_diameter * size_rel)
            size_px = max(8, min(size_px, 120))
            x, y = _rel_to_pixel(pos.get("x", 0), pos.get("y", 0))
            color = COLOR_MAP.get(ftype, "#FFFFFF")

            circle = mpatches.Circle((x, y), radius=size_px, fill=False,
                                     edgecolor=color, linewidth=1.5,
                                     linestyle=":", alpha=0.7)
            ax.add_patch(circle)
            ax.plot(x, y, "o", color=color, markersize=6)

            label = feat.get("label", ftype)
            # Ensure "无" (none) label is clearly visible in black
            if not label or label.strip() in ("无", "None", "none", "?"):
                label = ftype.replace("_", " ").title()
            label_color = "#000000" if (not feat.get("label") or feat.get("label", "").strip() == "无") else color
            ax.annotate(label, xy=(x, y + size_px + 3), fontsize=7,
                        color=label_color, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="black", edgecolor=color, alpha=0.7))

        # Draw prominences (日珥) - limb features with arc-line markers
        for prom in report.prominences:
            if not prom.get("checked", True):
                continue
            prominence_counter += 1
            pos = prom.get("position", {})
            size_rel = prom.get("size_relative", 0.06)
            size_px = int(disk_diameter * size_rel)
            # Increase max size for prominences - they can be very large (up to 400px)
            size_px = max(15, min(size_px, 400))
            x, y = _rel_to_pixel(pos.get("x", 0), pos.get("y", 0))
            color = COLOR_MAP.get("prominence", "#FFD700")

            # Draw arc/ellipse for prominence on limb
            ellipse = mpatches.Ellipse((x, y), width=size_px * 1.5, height=size_px,
                                       angle=0, fill=False, edgecolor=color,
                                       linewidth=2.5, linestyle="-", alpha=0.9)
            ax.add_patch(ellipse)

            # Diamond marker
            ax.plot(x, y, "D", color=color, markersize=12,
                    markeredgecolor="white", markeredgewidth=1.5)

            # Label - use actual label from detection
            label = prom.get("label", f"Prom#{prominence_counter}")
            ax.annotate(label, xy=(x, y - size_px - 5), fontsize=9,
                        color="#FFFFFF", fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, edgecolor="white", alpha=0.9))

        # Legend
        legend_items = []
        if report.sunspots:
            legend_items.append(mpatches.Patch(color=COLOR_MAP["sunspot"],
                                                label=f"Sunspots ({report.total_sunspots})"))
        if report.sunspot_groups:
            legend_items.append(mpatches.Patch(color=COLOR_MAP["sunspot_group"],
                                                label=f"Groups ({len(report.sunspot_groups)})"))
        if report.flares:
            legend_items.append(mpatches.Patch(color=COLOR_MAP["flare"],
                                                label=f"Flares ({report.total_flares})"))
        if report.prominences:
            legend_items.append(mpatches.Patch(color=COLOR_MAP["prominence"],
                                                label=f"Prominences ({report.total_prominences})"))
        if report.other_features:
            legend_items.append(mpatches.Patch(color="#FFFFFF",
                                                label=f"Other ({len(report.other_features)})"))

        if legend_items:
            ax.legend(handles=legend_items, loc="upper right", fontsize=9,
                      framealpha=0.9, facecolor="black", edgecolor="white",
                      labelcolor="white")

        # Title with detection summary
        title = (f"Solar Feature Detection Report\n"
                 f"Hale: {report.hale_classification} | "
                 f"Complexity: {report.complexity_score:.1f}/10 | "
                 f"Sunspots: {report.total_sunspots} | "
                 f"Flares: {report.total_flares} | "
                 f"Prominences: {report.total_prominences}")
        ax.set_title(title, fontsize=13, color="white", pad=15, fontweight="bold")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="gray")

        # Remove axis labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        output_path = ANNOTATED_DIR / f"{output_id}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to generate annotated image: {e}", exc_info=True)
        return ""


def generate_detection_report_image(
    report: DetectionReport,
    output_id: str,
) -> str:
    """Generate a standalone detection report visualization (no background image).

    Shows a summary table of all detected features with checkboxes.
    """
    if not HAS_MATPLOTLIB:
        return ""

    try:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        fig.patch.set_facecolor("#1a1a2e")

        # Title
        ax.text(5, 9.5, "Solar Feature Detection Report", fontsize=18,
                color="white", ha="center", fontweight="bold")
        ax.text(5, 9.1, f"Image: {report.image_path}", fontsize=10,
                color="gray", ha="center")
        ax.text(5, 8.8, f"Generated: {report.generated_at[:19]}", fontsize=9,
                color="gray", ha="center")

        # Summary metrics
        y_pos = 8.3
        metrics = [
            ("Total Sunspots", str(report.total_sunspots), "#00FF00"),
            ("Total Flares", str(report.total_flares), "#FF4444"),
            ("Prominences", str(report.total_prominences), "#FFD700"),
            ("Sunspot Groups", str(len(report.sunspot_groups)), "#00BFFF"),
            ("Other Features", str(len(report.other_features)), "#FFFFFF"),
            ("Hale Class", report.hale_classification, "#FFD700"),
            ("Complexity", f"{report.complexity_score:.1f}/10", "#FFD700"),
        ]

        for i, (label, value, color) in enumerate(metrics):
            col = i % 3
            row = i // 3
            x = 1.5 + col * 3
            y = y_pos - row * 0.6
            ax.text(x, y, f"{label}: ", fontsize=10, color="gray", ha="right")
            ax.text(x + 0.1, y, value, fontsize=11, color=color, ha="left", fontweight="bold")

        # Feature table
        y_table = 5.5
        ax.text(0.5, y_table, "Detected Features", fontsize=14,
                color="white", fontweight="bold")
        y_table -= 0.4

        # Table headers
        headers = ["#", "Type", "Label", "X", "Y", "Size", "Confidence", "Checked"]
        col_widths = [0.4, 1.2, 2.5, 0.8, 0.8, 0.8, 1.0, 0.8]
        x_start = 0.5

        for i, (header, width) in enumerate(zip(headers, col_widths)):
            ax.text(x_start, y_table, header, fontsize=9, color="gray",
                    fontweight="bold")
            x_start += width

        y_table -= 0.35

        # Table rows
        row_idx = 1
        all_features = []

        for s in report.sunspots:
            all_features.append(("sunspot", s))
        for g in report.sunspot_groups:
            all_features.append(("sunspot_group", g))
        for f in report.flares:
            all_features.append(("flare", f))
        for p in report.prominences:
            all_features.append(("prominence", p))
        for o in report.other_features:
            all_features.append((o.get("type", "other"), o))

        for ftype, feat in all_features:
            if y_table < 0.5:
                break
            pos = feat.get("position", {})
            checked = "Yes" if feat.get("checked", True) else "No"
            color = COLOR_MAP.get(ftype, "#FFFFFF")

            row_data = [
                str(row_idx),
                ftype.replace("_", " ").title(),
                feat.get("label", "")[:20],
                f"{pos.get('x', 0):.3f}",
                f"{pos.get('y', 0):.3f}",
                f"{feat.get('size_relative', 0):.3f}",
                f"{feat.get('confidence', 0):.1%}",
                checked,
            ]

            for i, (val, width) in enumerate(zip(row_data, col_widths)):
                ax.text(x_start if i == 0 else 0.5 + sum(col_widths[:i]),
                        y_table, val, fontsize=8, color=color)

            y_table -= 0.3
            row_idx += 1

        # Footer
        ax.text(5, 0.3, f"Total: {row_idx - 1} features detected | "
                f"Sunspots: {report.total_sunspots} | Flares: {report.total_flares}",
                fontsize=10, color="gray", ha="center")

        plt.tight_layout()

        output_path = ANNOTATED_DIR / f"{output_id}_report.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to generate report image: {e}", exc_info=True)
        return ""


def generate_combined_report_image(
    image_path: str,
    annotated_path: str,
    report: DetectionReport,
    output_id: str,
) -> str:
    """Generate a combined report image showing original and annotated side by side.
    
    This is the main report image that clearly shows:
    - Left: Original solar image (待上传的原始太阳图片)
    - Right: AI-detected annotated image (AI检测后生成的特征标注图片)
    - Bottom: Detection summary statistics
    """
    if not HAS_MATPLOTLIB:
        return ""

    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Original image
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            axes[0].imshow(np.array(img))
        axes[0].set_title("Original Solar Image\n原始太阳图片", fontsize=14, 
                         color="white", fontweight="bold", pad=10)
        axes[0].axis("off")
        axes[0].set_facecolor("black")
        
        # Right: Annotated image
        if os.path.exists(annotated_path):
            ann_img = Image.open(annotated_path).convert("RGB")
            axes[1].imshow(np.array(ann_img))
        axes[1].set_title("AI Detection Result\nAI检测标注结果", fontsize=14,
                         color="white", fontweight="bold", pad=10)
        axes[1].axis("off")
        axes[1].set_facecolor("black")
        
        fig.patch.set_facecolor("#0a0a1a")
        fig.suptitle("Solar Feature Detection Report - 太阳特征检测报告", 
                    fontsize=18, color="white", fontweight="bold", y=0.98)
        
        # Add summary text at bottom
        summary_text = (f"Detected: {report.total_sunspots} sunspots, "
                       f"{len(report.sunspot_groups)} groups, "
                       f"{report.total_flares} flares | "
                       f"Hale: {report.hale_classification} | "
                       f"Complexity: {report.complexity_score:.1f}/10")
        fig.text(0.5, 0.02, summary_text, ha="center", fontsize=11,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#1a1a3e", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = ANNOTATED_DIR / f"{output_id}_combined.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to generate combined report image: {e}", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Build report from analysis features
# ---------------------------------------------------------------------------

def _convert_position_to_relative(
    position: Dict, 
    image_width: int, 
    image_height: int,
    pixel_position: Optional[Dict] = None,
    disk_info: Optional[Dict] = None,
    source: str = "",
) -> Dict:
    """Convert position coordinates to 0~1 relative coordinates for annotation.
    
    Coordinate systems:
    - AI features (source="ai"): position is already 0~1 image-relative, pixel_position is 0
    - CV features (source="cv"): position is disk-centered normalized (-1~+1), pixel_position has pixel coords
    
    Strategy:
    1. If pixel_position has valid pixel coords (>5): use them directly -> convert to 0~1
    2. If source is "ai" or coords are already 0~1: pass through
    3. If source is "cv" and disk_info available: convert disk-centered -> image-relative
    4. Fallback: heuristic based on value range
    """
    x = position.get("x", 0.5)
    y = position.get("y", 0.5)
    
    # Priority 1: Use pixel coordinates if they are valid (CV-detected features)
    if pixel_position and pixel_position.get("x", 0) > 5 and pixel_position.get("y", 0) > 5:
        px = pixel_position["x"]
        py = pixel_position["y"]
        return {
            "x": px / image_width if image_width > 0 else 0.5,
            "y": py / image_height if image_height > 0 else 0.5,
        }
    
    # Priority 2: AI features return 0~1 image-relative coords directly - pass through
    if source == "ai":
        return {"x": max(0.0, min(1.0, x)), "y": max(0.0, min(1.0, y))}
    
    # Priority 3: CV features with disk_info - convert disk-centered to image-relative
    if source == "cv" and disk_info and disk_info.get("detected"):
        disk_cx = disk_info.get("center_x", image_width / 2)
        disk_cy = disk_info.get("center_y", image_height / 2)
        disk_r = disk_info.get("radius", min(image_width, image_height) / 2)
        norm_cx = disk_cx / image_width if image_width > 0 else 0.5
        norm_cy = disk_cy / image_height if image_height > 0 else 0.5
        norm_r = disk_r / min(image_width, image_height) if min(image_width, image_height) > 0 else 0.4
        img_x = norm_cx + norm_r * x
        img_y = norm_cy + norm_r * y
        return {"x": max(0.0, min(1.0, img_x)), "y": max(0.0, min(1.0, img_y))}
    
    # Priority 4: Heuristic fallback
    # If values clearly outside 0~1, they're disk-centered -> convert
    if x < -0.1 or x > 1.1 or y < -0.1 or y > 1.1:
        if disk_info and disk_info.get("detected"):
            disk_cx = disk_info.get("center_x", image_width / 2)
            disk_cy = disk_info.get("center_y", image_height / 2)
            disk_r = disk_info.get("radius", min(image_width, image_height) / 2)
            norm_cx = disk_cx / image_width if image_width > 0 else 0.5
            norm_cy = disk_cy / image_height if image_height > 0 else 0.5
            norm_r = disk_r / min(image_width, image_height) if min(image_width, image_height) > 0 else 0.4
            img_x = norm_cx + norm_r * x
            img_y = norm_cy + norm_r * y
            return {"x": max(0.0, min(1.0, img_x)), "y": max(0.0, min(1.0, img_y))}
        # No disk info, assume pixel coords
        if abs(x) > 1.5 or abs(y) > 1.5:
            return {"x": x / image_width, "y": y / image_height}
    
    # Already 0~1 relative, pass through
    return {"x": max(0.0, min(1.0, x)), "y": max(0.0, min(1.0, y))}


def build_detection_report_from_cv(
    image_id: str,
    image_path: str,
    cv_result: Dict[str, Any],
    hale_classification: str = "Unknown",
    complexity_score: float = 0.0,
) -> DetectionReport:
    """Build a DetectionReport directly from CV preprocessing results.
    
    Converts CV sunspot dicts (x, y, radius, area, confidence) to
    DetectionReport sunspots with proper relative coordinates.
    Uses rectangle annotations for each individual sunspot.
    """
    report = DetectionReport(image_id, image_path)
    report.hale_classification = hale_classification
    report.complexity_score = complexity_score
    
    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception:
        img_width, img_height = 500, 500
    
    # Convert sunspots
    sunspots = cv_result.get("sunspots", [])
    for i, spot in enumerate(sunspots, 1):
        x_px = spot.get("x", 0)
        y_px = spot.get("y", 0)
        radius = spot.get("radius", 20)
        area = spot.get("area", 0)
        confidence = spot.get("confidence", 0.5)
        
        # Convert pixel coordinates to 0~1 relative
        position = {
            "x": x_px / img_width if img_width > 0 else 0.5,
            "y": y_px / img_height if img_height > 0 else 0.5,
        }
        
        # Size relative to image (use radius for annotation sizing)
        size = max(radius / min(img_width, img_height), 0.02)
        
        # Get additional params
        additional_params = {
            "area": area,
            "contrast": spot.get("contrast", 0),
            "brightness": spot.get("brightness", 0),
            "region": spot.get("region", "unknown"),
            "bbox_width": spot.get("bbox_width", 0),
            "bbox_height": spot.get("bbox_height", 0),
            "index": spot.get("index", i),
        }
        
        label = f"黑子#{i}"
        report.add_sunspot(
            label=label,
            position=position,
            size=size,
            confidence=confidence,
            additional_params=additional_params,
        )
    
    # Convert bright regions
    bright_regions = cv_result.get("bright_regions", [])
    for i, br in enumerate(bright_regions, 1):
        x_px = br.get("x", 0)
        y_px = br.get("y", 0)
        confidence = br.get("confidence", 0.5)
        br_type = br.get("type", "bright_region")
        
        position = {
            "x": x_px / img_width if img_width > 0 else 0.5,
            "y": y_px / img_height if img_height > 0 else 0.5,
        }
        size = max(br.get("width", 20) / img_width, 0.02)
        
        additional_params = {
            "brightness_ratio": br.get("brightness_ratio", 0),
            "area": br.get("area", 0),
            "index": br.get("index", i),
        }
        
        label = f"{br_type}#{i}"
        if br_type == "flare":
            report.add_flare(
                label=label,
                position=position,
                size=size,
                confidence=confidence,
                additional_params=additional_params,
            )
        else:
            report.add_other(
                feature_type=br_type,
                label=label,
                position=position,
                size=size,
                confidence=confidence,
            )
    
    # Convert sunspot groups
    groups = cv_result.get("sunspot_groups", [])
    for i, group in enumerate(groups, 1):
        cx = group.get("center_x", 0)
        cy = group.get("center_y", 0)
        confidence = group.get("confidence", 0.5)
        member_count = group.get("member_count", 0)
        
        position = {
            "x": cx / img_width if img_width > 0 else 0.5,
            "y": cy / img_height if img_height > 0 else 0.5,
        }
        size = max(group.get("max_spread", 30) / img_width, 0.05)
        
        additional_params = {
            "complexity": group.get("complexity", 0),
            "member_count": member_count,
        }
        
        label = f"群组#{i}"
        report.add_sunspot_group(
            label=label,
            position=position,
            size=size,
            confidence=confidence,
            spot_count=member_count,
            additional_params=additional_params,
        )
    
    report.total_sunspots = len(report.sunspots)
    report.total_flares = len(report.flares)
    
    return report


def build_detection_report(
    image_id: str,
    image_path: str,
    features: List[Dict],
    hale_classification: str = "Unknown",
    complexity_score: float = 0.0,
    disk_info: Optional[Dict] = None,
) -> DetectionReport:
    """Build a DetectionReport from analysis features.

    Separates features into sunspots, groups, flares, prominences, and others.
    Coordinates are expected in 0~1 relative format.
    """
    report = DetectionReport(image_id, image_path)
    report.hale_classification = hale_classification
    report.complexity_score = complexity_score
    if disk_info:
        report.disk_info = disk_info
    
    # Get image dimensions for coordinate conversion
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception:
        img_width, img_height = 500, 500

    report.img_width = img_width
    report.img_height = img_height

    spot_num = 0
    group_num = 0
    flare_num = 0
    prominence_num = 0

    for feat in features:
        ftype = feat.get("type", "unknown")
        label = feat.get("label", "")
        position = feat.get("position", {"x": 0.0, "y": 0.0})
        pixel_pos = feat.get("pixel_position", {})
        size = feat.get("size_relative", 0.05)
        confidence = feat.get("confidence", 0.5)
        params = feat.get("additional_params", {})
        
        # Convert position to 0~1 relative coordinates for annotation
        # Pass source to distinguish AI (0~1 image-relative) vs CV (disk-centered) coords
        source = params.get("source", "")
        position = _convert_position_to_relative(position, img_width, img_height, pixel_pos, disk_info, source)

        if ftype == "sunspot":
            spot_num += 1
            if not label:
                label = f"Sunspot #{spot_num}"
            # Include index in additional_params
            spot_params = {**params}
            if "index" not in spot_params and feat.get("index"):
                spot_params["index"] = feat["index"]
            report.add_sunspot(
                label=label,
                position=position,
                size=size,
                confidence=confidence,
                group_id=params.get("group_id"),
                umbra_size=params.get("umbra_size", 0),
                penumbra_size=params.get("penumbra_size", 0),
                additional_params=spot_params,
            )

        elif ftype == "sunspot_group":
            group_num += 1
            if not label:
                label = f"Sunspot Group #{group_num}"
            report.add_sunspot_group(
                label=label,
                position=position,
                size=size,
                confidence=confidence,
                spot_count=params.get("spot_count", 0),
                group_spots=params.get("group_spots", []),
            )

        elif ftype == "flare":
            flare_num += 1
            if not label:
                label = f"Flare #{flare_num}"
            report.add_flare(
                label=label,
                position=position,
                size=size,
                confidence=confidence,
                flare_class=params.get("flare_class", ""),
                intensity=params.get("intensity", 0),
            )

        elif ftype == "prominence":
            prominence_num += 1
            if not label:
                label = f"Prominence #{prominence_num}"
            report.add_prominence(
                label=label,
                position=position,
                size=size,
                confidence=confidence,
                intensity=params.get("intensity", 0),
                additional_params=params,
            )

        else:
            if not label:
                label = f"{ftype} #{len(report.other_features) + 1}"
            report.add_other(
                feature_type=ftype,
                label=label,
                position=position,
                size=size,
                confidence=confidence,
            )

    return report
