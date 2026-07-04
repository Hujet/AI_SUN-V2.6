"""
PDF Report Generator for Solar Feature Detection

Generates clean, professional PDF reports with:
- Adaptive layout preventing content overflow
- Centered images placed in text-free zones
- Full AI analysis summary
- Consistent font, spacing, and color scheme
- Batch multi-page export with cover page
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

logger = logging.getLogger(__name__)

CHINESE_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

# A4 dimensions in mm
PAGE_W = 210
PAGE_H = 297
MARGIN = 10
CONTENT_W = PAGE_W - 2 * MARGIN  # 190mm


class SolarReportPDF(FPDF):
    """Professional PDF with solar blue theme and Chinese support."""

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=8)
        if HAS_FPDF and os.path.exists(CHINESE_FONT_PATH):
            self.add_font("ZH", "", CHINESE_FONT_PATH, uni=True)
            self.add_font("ZH", "B", CHINESE_FONT_PATH, uni=True)
            self._zh = "ZH"
        else:
            self._zh = "Helvetica"
        self.set_left_margin(MARGIN)
        self.set_right_margin(MARGIN)

    def zh_font(self, bold=False, size=10):
        self.set_font(self._zh, "B" if bold else "", size)

    def section_bar(self, title: str, h: float = 6.5):
        self.set_fill_color(20, 60, 140)
        self.set_text_color(255, 255, 255)
        self.zh_font(bold=True, size=9)
        self.cell(CONTENT_W, h, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def kv_line(self, key: str, val: str, kw: float = 55):
        self.set_text_color(100, 100, 100)
        self.zh_font(size=7.5)
        self.cell(kw, 4.5, key)
        self.set_text_color(20, 40, 80)
        self.zh_font(bold=True, size=7.5)
        self.cell(0, 4.5, str(val), new_x="LMARGIN", new_y="NEXT")

    def body_text(self, text: str, size: float = 7.5, color: tuple = (50, 50, 50)):
        self.set_text_color(*color)
        self.zh_font(size=size)
        self.set_x(MARGIN)
        self.multi_cell(CONTENT_W, 4.2, text, align="L")
        self.ln(0.5)

    def header(self):
        pass

    def footer(self):
        self.set_y(-10)
        self.zh_font(size=6.5)
        self.set_text_color(160, 160, 160)
        self.cell(CONTENT_W / 2, 5, "AI Solar Detection System", align="L")
        self.cell(CONTENT_W / 2, 5, f"Page {self.page_no()}/{{nb}}", align="R")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_image(report_data: Dict) -> Optional[str]:
    """Try multiple image sources and return first existing one."""
    keys = ["combined_image_path", "annotated_image_path",
            "report_image_path", "original_image_path"]
    for key in keys:
        path = report_data.get(key, "")
        if path and os.path.exists(path):
            return path
    fn = report_data.get("image_info", {}).get("filename", "")
    if fn:
        p = Path(__file__).parent.parent / "data" / "uploads" / fn
        if p.exists():
            return str(p)
    return None


def _get_img_dims(img_path: str) -> tuple:
    """Return (width_px, height_px) or (800, 600) default."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path)
        return img.size
    except Exception:
        return (800, 600)


def _calc_image_size_on_page(img_path: str, max_w: float, max_h: float) -> tuple:
    """Calculate (w_mm, h_mm) that fit max_w x max_h while preserving aspect."""
    pw, ph = _get_img_dims(img_path)
    ratio = pw / ph
    w = min(max_w, max_h * ratio)
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    return w, h


# ---------------------------------------------------------------------------
# Single Report
# ---------------------------------------------------------------------------

def generate_single_report_pdf(report_data: Dict, output_path: str,
                                include_images: bool = True) -> str:
    """Generate a single-page PDF with adaptive layout.

    Layout: Title -> Info -> Stats -> AI Summary -> Warnings -> IMAGE (centered) -> Feature Table.
    If content overflows, images and/or table go to page 2.
    """
    if not HAS_FPDF:
        raise RuntimeError("fpdf2 required: pip install fpdf2")

    pdf = SolarReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    det = report_data.get("detection_report", {})
    ana = report_data.get("analysis", {})
    info = report_data.get("image_info", {})
    sm = det.get("summary", {})
    c = pdf._zh

    # ====== TITLE ======
    pdf.set_fill_color(15, 45, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.zh_font(bold=True, size=13)
    pdf.cell(CONTENT_W - 40, 7.5, "  Solar Feature Detection Report", fill=True)
    pdf.zh_font(size=7)
    pdf.cell(40, 7.5, report_data.get("generated_at", "")[:16].replace("T", " "),
             fill=True, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ====== BASIC INFO (compact) ======
    pdf.set_fill_color(240, 244, 250)
    pdf.set_text_color(50, 50, 50)
    pdf.zh_font(size=7)
    fname = info.get("filename", "N/A")
    display_name = fname[:50] + ("..." if len(fname) > 50 else "")
    pdf.cell(CONTENT_W, 5, f"  Image: {display_name}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(70, 70, 70)
    pdf.zh_font(size=6.5)
    model_name = report_data.get("analysis", {}).get("model_used",
                report_data.get("model_used", "AI"))
    pdf.cell(CONTENT_W, 5,
             f"  Size: {info.get('width',0)}x{info.get('height',0)}px  |  "
             f"Model: {model_name}  |  "
             f"Time: {report_data.get('generated_at','')[:19]}",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ====== DETECTION METRICS ======
    pdf.section_bar("Detection Results")
    total_other = len(det.get("other_features", []))
    metrics = [
        ("Sunspots", sm.get("total_sunspots", 0), (0, 160, 0)),
        ("Groups", len(det.get("sunspot_groups", [])), (0, 130, 200)),
        ("Flares", sm.get("total_flares", 0), (220, 60, 60)),
        ("Other", total_other, (120, 120, 120)),
        ("Hale", det.get("hale_classification", "?"), (200, 150, 0)),
        ("Score", det.get("complexity_score", 0), (200, 150, 0)),
    ]
    cell_w = CONTENT_W / 6
    for label, val, col in metrics:
        pdf.set_text_color(*col)
        pdf.zh_font(bold=True, size=12)
        pdf.cell(cell_w * 0.35, 7, str(val), align="C")
        pdf.set_text_color(90, 90, 90)
        pdf.zh_font(size=6.5)
        pdf.cell(cell_w * 0.65, 7, label)
    pdf.ln(9)

    # ====== AI ANALYSIS SUMMARY (full, not truncated) ======
    summary_text = ana.get("summary", "") or report_data.get("summary", "")
    if summary_text:
        pdf.section_bar("AI Analysis Summary")
        # Split long text into manageable paragraphs
        pdf.body_text(summary_text, size=7.5)

    # ====== WARNINGS ======
    warnings = ana.get("warnings", []) or det.get("warnings", [])
    if warnings:
        pdf.set_text_color(200, 30, 30)
        pdf.zh_font(bold=True, size=7)
        pdf.cell(CONTENT_W, 5, "Warnings:", new_x="LMARGIN", new_y="NEXT")
        for w in warnings[:4]:
            pdf.set_x(MARGIN)
            pdf.set_text_color(180, 40, 40)
            pdf.zh_font(size=7)
            pdf.multi_cell(CONTENT_W - 4, 3.8, f"  ! {w[:120]}")
        pdf.ln(2)

    # ====== IMAGE SECTION (centered in available space) ======
    if include_images:
        img_path = _find_best_image(report_data)
        if img_path:
            pdf.section_bar("Detection Visualization")

            # Calculate available height
            current_y = pdf.get_y()
            remaining = PAGE_H - current_y - 18  # 18 for footer + margin
            max_img_h = min(remaining * 0.55, 95)  # max 55% of remaining or 95mm

            img_w, img_h = _calc_image_size_on_page(img_path, CONTENT_W, max_img_h)

            # Center horizontally
            x_pos = MARGIN + (CONTENT_W - img_w) / 2
            try:
                pdf.image(img_path, x=x_pos, y=current_y + 1, w=img_w, h=img_h)
                pdf.set_y(current_y + img_h + 3)
            except Exception as e:
                logger.warning(f"Failed to embed image: {e}")
                pdf.set_text_color(160, 160, 160)
                pdf.zh_font(size=7)
                pdf.cell(CONTENT_W, 5, "[Image could not be loaded]",
                         new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.section_bar("Detection Visualization")
            pdf.set_text_color(160, 160, 160)
            pdf.zh_font(size=7)
            pdf.cell(CONTENT_W, 5, "[No image available]",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # ====== FEATURE TABLE (compact, adaptive) ======
    all_feats = []
    for items, ftype in [
        (det.get("sunspots", []), "Sunspot"),
        (det.get("sunspot_groups", []), "Group"),
        (det.get("flares", []), "Flare"),
        (det.get("other_features", []), "Other"),
    ]:
        for f in items:
            pos = f.get("position", {})
            all_feats.append((
                ftype,
                f.get("label", "")[:30],
                pos.get("x", 0),
                pos.get("y", 0),
                f.get("size_relative", 0),
                f.get("confidence", 0),
                f.get("flag", ""),
            ))

    if all_feats:
        # Check if enough space for table header + at least 5 rows
        needed = 12 + min(len(all_feats), 25) * 5
        current_y = pdf.get_y()
        remaining = PAGE_H - current_y - 15
        if remaining < needed:
            pdf.add_page()

        pdf.section_bar("Feature List")
        cols = [14, 6, 52, 19, 19, 19, 19, 17]
        headers = ["#", "T", "Label", "X", "Y", "Size", "Conf", "Flag"]
        type_map = {"Sunspot": "S", "Group": "G", "Flare": "F", "Other": "O"}

        # Header row
        pdf.set_fill_color(30, 50, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.zh_font(bold=True, size=6.5)
        for h, cw in zip(headers, cols):
            pdf.cell(cw, 5, h, fill=True, align="C")
        pdf.ln()

        for idx, (ftype, label, x, y, size, conf, flag) in enumerate(all_feats):
            if idx % 2 == 0:
                pdf.set_fill_color(238, 242, 252)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.zh_font(size=6.5)
            ts = type_map.get(ftype, "?")
            pdf.set_text_color(30, 30, 30)

            pdf.cell(cols[0], 4.5, str(idx + 1), fill=True, align="C")
            pdf.cell(cols[1], 4.5, ts, fill=True, align="C")
            pdf.cell(cols[2], 4.5, label, fill=True)
            pdf.cell(cols[3], 4.5, f"{x:.3f}", fill=True, align="C")
            pdf.cell(cols[4], 4.5, f"{y:.3f}", fill=True, align="C")
            pdf.cell(cols[5], 4.5, f"{size:.3f}", fill=True, align="C")

            # Confidence with color coding
            if conf >= 0.7:
                pdf.set_text_color(0, 140, 0)       # green: high confidence
            elif conf >= 0.4:
                pdf.set_text_color(200, 150, 0)      # orange: medium
            else:
                pdf.set_text_color(200, 50, 50)      # red: low
            pdf.cell(cols[6], 4.5, f"{conf:.2f}", fill=True, align="C")

            # Flag with color
            flag_colors = {
                "correct": (40, 140, 40),
                "false_positive": (200, 40, 40),
                "suspicious": (200, 140, 0),
                "missed": (140, 140, 140),
            }
            if flag and flag in flag_colors:
                pdf.set_text_color(*flag_colors[flag])
                pdf.zh_font(bold=True, size=6.5)
            elif flag:
                pdf.set_text_color(100, 100, 100)
                pdf.zh_font(size=6.5)
            else:
                pdf.set_text_color(60, 60, 60)
                pdf.zh_font(size=6.5)
            display_flag = flag[:4] if flag else "-"
            pdf.cell(cols[7], 4.5, display_flag, fill=True, align="C")
            pdf.set_text_color(30, 30, 30)
            pdf.ln()

    pdf.output(output_path)
    logger.info(f"PDF saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Batch PDF
# ---------------------------------------------------------------------------

def generate_batch_pdf(reports: List[Dict], output_path: str,
                        include_images: bool = True) -> str:
    """Batch PDF: cover page + one page per report."""
    if not HAS_FPDF:
        raise RuntimeError("fpdf2 required")

    pdf = SolarReportPDF()
    pdf.alias_nb_pages()
    c = pdf._zh

    # ====== COVER PAGE ======
    pdf.add_page()
    pdf.set_y(50)
    pdf.set_fill_color(15, 45, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.zh_font(bold=True, size=28)
    pdf.cell(CONTENT_W, 15, "Solar Feature Detection", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 15, "Batch Analysis Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.zh_font(size=14)
    pdf.cell(CONTENT_W, 10, f"{len(reports)} Reports", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_text_color(120, 120, 120)
    pdf.zh_font(size=10)
    pdf.cell(CONTENT_W, 8,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)

    # Aggregate stats
    total_s, total_f, total_feat = 0, 0, 0
    hale_map = {}
    for r in reports:
        d = r.get("detection_report", {})
        s = d.get("summary", {})
        total_s += s.get("total_sunspots", 0)
        total_f += s.get("total_flares", 0)
        det_feats = (d.get("sunspots", []) + d.get("sunspot_groups", []) +
                     d.get("flares", []) + d.get("other_features", []))
        total_feat += len(det_feats)
        h = d.get("hale_classification", "?")
        hale_map[h] = hale_map.get(h, 0) + 1

    x0 = MARGIN + 20
    pdf.set_xy(x0, pdf.get_y())
    pdf.set_text_color(20, 40, 80)
    pdf.zh_font(bold=True, size=12)
    pdf.cell(80, 8, f"Total Sunspots: {total_s}")
    pdf.cell(80, 8, f"Total Flares: {total_f}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(x0, pdf.get_y())
    pdf.cell(80, 8, f"Total Features: {total_feat}")
    avg = total_feat / max(len(reports), 1)
    pdf.cell(0, 8, f"Avg/Report: {avg:.1f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_xy(x0, pdf.get_y())
    pdf.zh_font(bold=True, size=10)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(CONTENT_W - 40, 7, "Hale Classification Distribution",
             new_x="LMARGIN", new_y="NEXT")
    for hale, cnt in sorted(hale_map.items(), key=lambda x: -x[1]):
        pdf.set_xy(x0 + 10, pdf.get_y())
        pdf.set_text_color(60, 60, 60)
        pdf.zh_font(size=9)
        pdf.cell(35, 6, hale)
        pdf.set_text_color(20, 40, 80)
        pdf.zh_font(bold=True, size=9)
        pct = cnt / max(len(reports), 1) * 100
        pdf.cell(0, 6, f"{cnt} reports ({pct:.0f}%)",
                 new_x="LMARGIN", new_y="NEXT")

    # ====== INDIVIDUAL REPORT PAGES ======
    for idx, rpt in enumerate(reports):
        pdf.add_page()

        det = rpt.get("detection_report", {})
        ana = rpt.get("analysis", {})
        info = rpt.get("image_info", {})
        sm = det.get("summary", {})

        # Header
        fname = info.get("filename", "?")
        pdf.set_fill_color(15, 45, 110)
        pdf.set_text_color(255, 255, 255)
        pdf.zh_font(bold=True, size=11)
        pdf.cell(120, 7, f"  #{idx+1}  {fname[:50]}", fill=True)
        pdf.zh_font(size=7)
        ts = rpt.get("generated_at", "")[:16].replace("T", " ")
        pdf.cell(CONTENT_W - 120, 7, f"  {ts}", fill=True, align="R",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Stats row
        pdf.zh_font(size=7.5)
        stats = [
            ("Sunspots", sm.get("total_sunspots", 0)),
            ("Flares", sm.get("total_flares", 0)),
            ("Groups", len(det.get("sunspot_groups", []))),
            ("Hale", det.get("hale_classification", "?")),
        ]
        for label, val in stats:
            pdf.set_text_color(100, 100, 100)
            pdf.cell(25, 5, f"{label}:")
            pdf.set_text_color(20, 40, 80)
            pdf.zh_font(bold=True, size=7.5)
            pdf.cell(20, 5, str(val))
            pdf.zh_font(size=7.5)
        pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # AI Summary
        summary_text = ana.get("summary", "") or rpt.get("summary", "")
        if summary_text:
            pdf.set_text_color(20, 40, 80)
            pdf.zh_font(bold=True, size=8)
            pdf.cell(CONTENT_W, 5, "AI Analysis:",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(50, 50, 50)
            pdf.zh_font(size=7)
            # Adaptive text with overflow handling
            pdf.set_x(MARGIN)
            pdf.multi_cell(CONTENT_W, 3.8, summary_text[:600])
            pdf.ln(2)

        # Warnings
        warns = ana.get("warnings", [])
        if warns:
            pdf.set_text_color(200, 30, 30)
            pdf.zh_font(bold=True, size=7)
            pdf.cell(CONTENT_W, 5, "Warnings:",
                     new_x="LMARGIN", new_y="NEXT")
            for w in warns[:2]:
                pdf.set_text_color(180, 40, 40)
                pdf.zh_font(size=6.5)
                pdf.set_x(MARGIN)
                pdf.multi_cell(CONTENT_W, 3.5, f"  {w[:90]}")
            pdf.ln(2)

        # Image
        if include_images:
            img_path = _find_best_image(rpt)
            if img_path:
                current_y = pdf.get_y()
                remaining = PAGE_H - current_y - 30
                max_h = min(remaining * 0.4, 75)
                img_w, img_h = _calc_image_size_on_page(img_path, CONTENT_W, max_h)
                x_pos = MARGIN + (CONTENT_W - img_w) / 2
                try:
                    pdf.image(img_path, x=x_pos, y=current_y + 1,
                              w=img_w, h=img_h)
                    pdf.set_y(current_y + img_h + 3)
                except Exception as e:
                    logger.warning(f"Batch image failed: {e}")
                    pdf.set_text_color(160, 160, 160)
                    pdf.zh_font(size=7)
                    pdf.cell(CONTENT_W, 5, "[Image unavailable]",
                             new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_text_color(160, 160, 160)
                pdf.zh_font(size=7)
                pdf.cell(CONTENT_W, 5, "[No image]",
                         new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

        # Feature table
        all_feats = []
        for items, ftype in [
            (det.get("sunspots", []), "S"),
            (det.get("sunspot_groups", []), "G"),
            (det.get("flares", []), "F"),
            (det.get("other_features", []), "O"),
        ]:
            for f in items:
                pos = f.get("position", {})
                all_feats.append((ftype, f.get("label", "")[:22],
                                  pos.get("x", 0), pos.get("y", 0),
                                  f.get("size_relative", 0),
                                  f.get("confidence", 0)))

        if all_feats:
            # Space check
            needed = 10 + min(len(all_feats), 20) * 4.5
            if PAGE_H - pdf.get_y() - 15 < needed:
                pdf.add_page()

            pdf.set_text_color(20, 40, 80)
            pdf.zh_font(bold=True, size=8)
            pdf.cell(CONTENT_W, 5, "Feature List:",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

            cols = [12, 5, 50, 16, 16, 16, 16]
            hdrs = ["#", "T", "Label", "X", "Y", "Size", "Conf"]
            pdf.set_fill_color(30, 50, 100)
            pdf.set_text_color(255, 255, 255)
            pdf.zh_font(bold=True, size=6.5)
            for h, cw in zip(hdrs, cols):
                pdf.cell(cw, 4.5, h, fill=True, align="C")
            pdf.ln()

            for i, (ft, lb, x, y, sz, cf) in enumerate(all_feats):
                pdf.set_fill_color(
                    (238, 242, 252) if i % 2 == 0 else (255, 255, 255))
                pdf.set_text_color(30, 30, 30)
                pdf.zh_font(size=6.5)
                pdf.cell(cols[0], 4.2, str(i+1), fill=True, align="C")
                pdf.cell(cols[1], 4.2, ft, fill=True, align="C")
                pdf.cell(cols[2], 4.2, lb, fill=True)
                pdf.cell(cols[3], 4.2, f"{x:.3f}", fill=True, align="C")
                pdf.cell(cols[4], 4.2, f"{y:.3f}", fill=True, align="C")
                pdf.cell(cols[5], 4.2, f"{sz:.3f}", fill=True, align="C")
                if cf >= 0.7:
                    pdf.set_text_color(0, 140, 0)
                elif cf >= 0.4:
                    pdf.set_text_color(200, 140, 0)
                else:
                    pdf.set_text_color(200, 50, 50)
                pdf.cell(cols[6], 4.2, f"{cf:.2f}", fill=True, align="C")
                pdf.ln()

    pdf.output(output_path)
    logger.info(f"Batch PDF saved: {output_path}")
    return output_path
