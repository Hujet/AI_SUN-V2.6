"""
Enhanced Solar Region Classifier v2.0

Provides AI-powered solar feature recognition with:
- Local CV-based preprocessing (disk detection, spot segmentation, feature clustering)
- Structured JSON output from the AI model
- Hybrid AI + CV feature detection for maximum accuracy
- Feature extraction with precise position coordinates and confidence scores
- Traceability recording for every analysis step
- Token usage tracking
"""

import logging
import os
import json
import re
import math
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import the preprocessing pipeline
from solar_preprocessor import (
    preprocess_solar_image,
    generate_feature_prompt,
)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SolarFeature:
    """One detected solar feature with position, type, and confidence."""
    feature_type: str = ""           # sunspot, flare, bright_region, plage, filament, coronal_hole
    label: str = ""                  # Human-readable label (e.g. "Hale Beta", "M-class Flare")
    # Position coordinates (multiple coordinate systems)
    position_x: float = 0.0          # Normalized disk coords (-1~+1), origin at disk center
    position_y: float = 0.0          # Normalized disk coords (-1~+1), origin at disk center
    pixel_x: float = 0.0             # Original pixel X coordinate
    pixel_y: float = 0.0             # Original pixel Y coordinate
    size_relative: float = 0.0       # Relative size (0-1, as fraction of solar disk)
    confidence: float = 0.0          # Detection confidence (0-1)
    description: str = ""            # Scientific description
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "type": self.feature_type,
            "label": self.label,
            "position": {"x": self.position_x, "y": self.position_y},
            "pixel_position": {"x": self.pixel_x, "y": self.pixel_y},
            "size_relative": self.size_relative,
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "additional_params": self.additional_params,
            "index": self.additional_params.get("index", None),
        }

    @staticmethod
    def from_dict(d: Dict) -> "SolarFeature":
        pos = d.get("position", {})
        pixel_pos = d.get("pixel_position", {})
        return SolarFeature(
            feature_type=d.get("type", d.get("feature_type", "")),
            label=d.get("label", ""),
            position_x=pos.get("x", d.get("position_x", 0.0)),
            position_y=pos.get("y", d.get("position_y", 0.0)),
            pixel_x=pixel_pos.get("x", d.get("pixel_x", 0.0)),
            pixel_y=pixel_pos.get("y", d.get("pixel_y", 0.0)),
            size_relative=d.get("size_relative", d.get("size", 0.0)),
            confidence=d.get("confidence", 0.5),
            description=d.get("description", ""),
            additional_params=d.get("additional_params", {}),
        )


@dataclass
class HaleClassification:
    alpha: float = 0.0
    beta: float = 0.0
    beta_gamma: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    beta_delta: float = 0.0
    unknown: float = 0.0

    def get_highest_probability(self) -> Tuple[str, float]:
        probs = {
            "Alpha": self.alpha, "Beta": self.beta, "Beta-Gamma": self.beta_gamma,
            "Gamma": self.gamma, "Delta": self.delta, "Beta-Delta": self.beta_delta,
            "Unknown": self.unknown,
        }
        best = max(probs, key=probs.get)
        return best, probs[best]

    def to_dict(self) -> Dict:
        return {
            "Alpha": self.alpha, "Beta": self.beta, "Beta-Gamma": self.beta_gamma,
            "Gamma": self.gamma, "Delta": self.delta, "Beta-Delta": self.beta_delta,
            "Unknown": self.unknown,
        }


@dataclass
class SolarRegionAnalysis:
    image_id: str = ""
    image_path: str = ""
    analysis_time: str = ""
    is_solar_image: bool = True
    region_count: int = 0
    features: List[SolarFeature] = field(default_factory=list)
    hale_classification: str = "Unknown"
    hale_distribution: Dict[str, float] = field(default_factory=dict)
    classification_confidence: float = 0.0
    confidence_level: str = "low"
    complexity_score: float = 0.0
    risk_level: str = "unknown"
    polarity_distribution: str = ""
    separation_pattern: str = ""
    reasoning: str = ""
    region_descriptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    # Traceability
    raw_model_output: str = ""
    intermediate_steps: Dict[str, Any] = field(default_factory=dict)
    # Token usage
    token_usage: Dict[str, int] = field(default_factory=dict)
    # Scientific conclusion (traceability requirement)
    scientific_conclusion: str = ""
    flare_risk_assessment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "analysis_time": self.analysis_time,
            "is_solar_image": self.is_solar_image,
            "region_count": self.region_count,
            "features": [f.to_dict() for f in self.features],
            "hale_classification": self.hale_classification,
            "hale_distribution": self.hale_distribution,
            "classification_confidence": round(self.classification_confidence, 4),
            "confidence_level": self.confidence_level,
            "complexity_score": round(self.complexity_score, 2),
            "risk_level": self.risk_level,
            "polarity_distribution": self.polarity_distribution,
            "separation_pattern": self.separation_pattern,
            "reasoning": self.reasoning,
            "region_descriptions": self.region_descriptions,
            "warnings": self.warnings,
            "recommended_actions": self.recommended_actions,
            "scientific_conclusion": self.scientific_conclusion,
            "flare_risk_assessment": self.flare_risk_assessment,
            "token_usage": self.token_usage,
        }

    def to_summary(self) -> str:
        lines = [
            f"太阳活动区分析报告", "=" * 40,
            f"图像: {self.image_path}", f"分析时间: {self.analysis_time}", "",
            f"Hale分类: {self.hale_classification}",
            f"置信度: {self.classification_confidence:.1%} ({self.confidence_level})", "",
            f"检测特征数: {len(self.features)}",
            f"复杂度评分: {self.complexity_score:.2f}/10.0",
        ]
        if self.scientific_conclusion:
            lines.append("")
            lines.append(f"科学结论: {self.scientific_conclusion}")
        if self.flare_risk_assessment:
            lines.append("")
            lines.append(f"耀斑风险评估: {self.flare_risk_assessment.get('flare_risk_level', '未知')}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.recommended_actions:
            lines.append("")
            for a in self.recommended_actions:
                lines.append(f"  → {a}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured JSON Prompt (v2.0 - Optimized for AI+CV Hybrid Detection)
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """你是一位专业的太阳物理学家和太阳活动区分析专家，拥有超过20年的太阳观测经验。

你将收到一份太阳图像的本地CV预处理报告，其中包含了：
- 日面边界检测结果（中心坐标、半径）
- 黑子候选区域（精确像素坐标、面积、对比度）
- 亮区候选（耀斑/谱斑/光斑位置）
- 暗条候选（日面内部暗长条状结构）
- 日珥候选（边缘明亮突出）
- 黑子群组聚类结果

你的任务是：
1. 独立观察太阳图像，识别所有可见的太阳活动特征
2. 根据预处理数据验证和精炼太阳特征分析，标记确认、误判或补充遗漏
3. 对活动区进行Hale分类（Alpha/Beta/Beta-Gamma/Gamma/Delta/Beta-Delta）
4. 评估每个特征的置信度，修正CV检测中可能的误报
5. 特别注意日珥(太阳边缘的明亮突出)、耀斑(极亮区域)、黑子(暗核区域)、暗条(日面暗长条状)的检测
6. 提供科学严谨的分析结论

=== Hale分类标准（Mt.Wilson/McIntosh体系） ===
- Alpha (α): 单极区域，简单紧凑。单一极性黑子群，无耀斑风险。
- Beta (β): 双极区域，正负极性明显分离。低-中等耀斑风险(C级约5-10%)。
- Beta-Gamma (βγ): 双极但不规则，极性混合。中等耀斑风险(C级约20-30%)。
- Gamma (γ): 复杂多极区域，无明显双极结构。高耀斑风险(40-50%)。
- Delta (δ): 相反极性本影在同一半影内。极高耀斑风险(M/X级>60%)。
- Beta-Delta (βδ): Beta区域中包含Delta子结构。极高耀斑风险。

=== 复杂度评分（0-10） ===
- 0-2: 非常简单（Alpha，单极，无活动）
- 3-4: 简单（Beta，双极，低活动）
- 5-6: 中等（Beta-Gamma，不规则双极）
- 7-8: 复杂（Gamma/Delta，多极或混合极性）
- 9-10: 极度复杂（Beta-Delta，多个Delta结构）

=== 特征类型 ===
- sunspot: 黑子（暗色区域，包含本影和半影）
- flare: 耀斑（明亮爆发区域，常在活动区附近，极亮且不规则）
- bright_region: 亮区（比周围更亮的区域）
- plage: 谱斑（色球层亮区，常在黑子群周围，面积较大）
- filament: 暗条（暗色线状结构，在日面内部）
- prominence: 日珥（太阳边缘的明亮突出，位于日面边界外侧）
- coronal_hole: 冕洞（大范围暗区）

=== 日珥(Prominence)与耀斑(Flare)的区分标准 ===
- 日珥：位于太阳边缘（日面边界外侧），呈弧形或柱状突出，亮度高于周围日冕
- 耀斑：位于日面内部或边缘，极亮区域，通常与黑子群相关，形状不规则
- 关键区别：日珥在日面外，耀斑在日面内或边缘

=== 准确度评估要求 ===
对每个检测到的特征，请提供：
1. 置信度评分（0-1）：基于特征清晰度、对比度、位置合理性
2. 特征类型确认：明确标注是日珥、耀斑还是其他类型
3. 误判风险评估：标注可能的误报或漏检

请基于提供的预处理数据分析，修正CV检测中可能的误差，给出最终的科学分析结果。"""

ANALYSIS_USER_PROMPT = """以下是太阳图像的CV预处理报告：

{{PREPROCESS_REPORT}}

请基于以上预处理数据进行分析，并以JSON格式返回结果。

重要提示：
1. 请先独立观察图像，识别所有可见的太阳活动特征
2. 然后参考CV检测结果，逐一验证每个候选特征
3. 对于CV检测的特征，请标记为confirmed(确认)、false_positive(误判)或不确定
4. 如果CV漏检了明显的特征，请在features中补充
5. 特别注意日珥(太阳边缘的明亮突出)和耀斑(极亮区域)

**关键要求**：
1. 坐标系统：所有坐标使用归一化日面坐标（-1到+1范围），原点为日面中心
   - x: 正值表示日面右侧（西），负值表示左侧（东）
   - y: 正值表示日面北侧（上），负值表示南侧（下）
   - 日面边界对应半径约1.0
2. 黑子逐个识别：每个黑子独立条目，包含精确位置
3. 黑子群组：空间邻近的黑子自动分组
4. 特征类型：sunspot / sunspot_group / flare / bright_region / plage / filament / prominence / coronal_hole

```json
{
  "image_type": "magnetogram|euv|white_light|hmi_aia_combo|unknown",
  "is_solar_image": true,
  "hale_classification": "Beta-Gamma",
  "hale_distribution": {
    "Alpha": 0.05, "Beta": 0.15, "Beta-Gamma": 0.65, "Gamma": 0.10, "Delta": 0.03, "Beta-Delta": 0.02, "Unknown": 0.0
  },
  "classification_confidence": 0.82,
  "complexity_score": 6.5,
  "polarity_distribution": "mixed bipolar with irregular boundaries",
  "separation_pattern": "partial_mixing",
  "region_count": 1,
  "sunspot_count": 4,
  "flare_count": 1,
  "features": [
    {
      "type": "sunspot",
      "label": "黑子 #1",
      "position": {"x": 0.15, "y": -0.10},
      "size_relative": 0.08,
      "confidence": 0.92,
      "description": "主黑子，清晰的本影和半影结构",
      "additional_params": {"is_group": false, "group_id": "G1", "region": "center"}
    },
    {
      "type": "sunspot_group",
      "label": "黑子群 G1 (4个黑子)",
      "position": {"x": 0.20, "y": -0.05},
      "size_relative": 0.25,
      "confidence": 0.88,
      "description": "包含4个黑子的活动区，呈现Beta-Gamma结构",
      "additional_params": {"is_group": true, "spot_count": 4, "group_id": "G1"}
    },
    {
      "type": "flare",
      "label": "Flare #1 (C级)",
      "position": {"x": 0.25, "y": -0.15},
      "size_relative": 0.10,
      "confidence": 0.70,
      "description": "活动区附近的C级耀斑候选",
      "additional_params": {"flare_class": "C", "intensity": 0.7, "near_group": "G1"}
    },
    {
      "type": "plage",
      "label": "谱斑 (亮发射区)",
      "position": {"x": 0.18, "y": 0.05},
      "size_relative": 0.15,
      "confidence": 0.65,
      "description": "活动区周围的亮发射区域",
      "additional_params": {"brightness_ratio": 1.3, "near_group": "G1"}
    }
  ],
  "reasoning": "基于CV预处理数据：检测到4个黑子形成1个群组(G1)，位于日面中心偏右区域。黑子群呈现不规则双极结构，符合Beta-Gamma分类特征。活动区附近检测到1个亮区候选，可能为C级耀斑。复杂度评分6.5反映中等复杂度。",
  "scientific_conclusion": "该活动区为Beta-Gamma型黑子群，包含4个黑子，中等耀斑风险。建议持续监测。",
  "warnings": [],
  "recommended_actions": ["建议关注NOAA空间天气预报中心"]
}
```

注意：只返回JSON，不要包含其他文字。坐标范围-1到+1（日面坐标系）。"""


# ---------------------------------------------------------------------------
# Image Pre-processing & Feature Extraction
# ---------------------------------------------------------------------------

def extract_image_features(image_path: str) -> Dict[str, Any]:
    """Extract quantitative pre-analysis features from the input image.

    Computes basic image statistics that serve as intermediate traceability data
    and can be used as fallback heuristics when the AI model is unavailable.

    Returns a dict with: dimensions, brightness_stats, contrast, polarity_ratio, etc.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("L")  # Grayscale
        arr = np.array(img)
        h, w = arr.shape
        total_pixels = h * w

        # Brightness statistics
        mean_brightness = float(np.mean(arr))
        std_brightness = float(np.std(arr))
        min_brightness = float(np.min(arr))
        max_brightness = float(np.max(arr))

        # Contrast (Michelson contrast)
        contrast = (max_brightness - min_brightness) / (max_brightness + min_brightness + 1e-8)

        # Histogram analysis for polarity detection (for magnetograms)
        histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
        mid_point = 128
        dark_pixels = int(np.sum(arr < mid_point))
        bright_pixels = int(np.sum(arr >= mid_point))
        polarity_ratio = dark_pixels / max(bright_pixels, 1)

        # Edge density (proxy for feature complexity)
        try:
            import cv2
            edges = cv2.Canny(arr, 50, 150)
            edge_density = float(np.sum(edges > 0) / total_pixels)
        except ImportError:
            # Fallback: use gradient-based approximation
            sobel_x = np.diff(arr, axis=1)
            sobel_y = np.diff(arr, axis=0)
            gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y[:, :-1] ** 2)
            edge_density = float(np.mean(gradient_mag > 20))

        # Complexity estimate from edge density and contrast
        heuristic_complexity = min(10.0, (edge_density * 50 + contrast * 10 + (std_brightness / 50)))

        return {
            "dimensions": {"width": w, "height": h, "total_pixels": total_pixels},
            "brightness": {
                "mean": round(mean_brightness, 2),
                "std": round(std_brightness, 2),
                "min": round(min_brightness, 2),
                "max": round(max_brightness, 2),
            },
            "contrast": round(contrast, 4),
            "polarity": {
                "dark_pixels": dark_pixels,
                "bright_pixels": bright_pixels,
                "ratio": round(polarity_ratio, 4),
            },
            "edge_density": round(edge_density, 4),
            "heuristic_complexity": round(heuristic_complexity, 2),
            "histogram_bins_16": [int(x) for x in np.histogram(arr, bins=16, range=(0, 256))[0]],
        }
    except Exception as e:
        return {"error": str(e), "dimensions": {"width": 0, "height": 0, "total_pixels": 0}}


# ---------------------------------------------------------------------------
# Structured JSON Response Parser
# ---------------------------------------------------------------------------

def parse_structured_json_response(text: str) -> Optional[Dict]:
    """Parse the AI's JSON response. Handles markdown code blocks and loose JSON."""
    if not text or not text.strip():
        return None

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find JSON object in text
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end <= start:
        return None

    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Legacy text-based parser (fallback)
# ---------------------------------------------------------------------------

class HaleClassificationParser:
    """Fallback parser for non-structured text responses."""

    HALE_CLASSES = ["Beta-Delta", "Beta-Gamma", "Delta", "Beta", "Gamma", "Alpha"]

    CLASS_INDICATORS = {
        "Alpha": ["单极", "单一极性", "只有一个极性", "单极性", "compact", "unipolar"],
        "Beta": ["双极", "两个极性", "分界清晰", "明显分离", "bipolar", "clear boundary"],
        "Beta-Gamma": ["不规则", "交错", "混合", "扭曲", "irregular", "mixed polarity", "beta-gamma"],
        "Gamma": ["gamma", "gamma-class", "γ"],
        "Delta": ["delta", "δ", "同半影", "挤压", "混合本影", "opposite polarity umbra"],
        "Beta-Delta": ["beta-delta", "beta delta", "β-δ"],
    }

    CONFIDENCE_PATTERNS = [
        r"置信[度率][：:]\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence[：:]\s*(\d+(?:\.\d+)?)\s*%?",
        r"可信度[：:]\s*(\d+(?:\.\d+)?)\s*%?",
    ]

    def parse_text_response(self, response_text: str) -> SolarRegionAnalysis:
        analysis = SolarRegionAnalysis()
        analysis.analysis_time = datetime.now().isoformat()
        analysis.image_path = "unknown"
        analysis.is_solar_image = self._check_if_solar_image(response_text)

        if not analysis.is_solar_image:
            analysis.hale_classification = "Not Applicable"
            analysis.reasoning = "The uploaded image does not appear to be a solar image"
            analysis.warnings.append("图像可能不是太阳图像")
            return analysis

        analysis.hale_classification = self._extract_classification(response_text)
        analysis.classification_confidence = self._extract_confidence(response_text)
        analysis.confidence_level = self._get_confidence_level(analysis.classification_confidence)
        analysis.reasoning = self._extract_reasoning(response_text)
        analysis.complexity_score = self._estimate_complexity(response_text)
        analysis.polarity_distribution = self._extract_polarity_info(response_text)
        analysis.separation_pattern = self._extract_separation_pattern(response_text)

        return analysis

    def _check_if_solar_image(self, text: str) -> bool:
        solar_keywords = ["太阳", "黑子", "日面", "solar", "sunspot", "sun", "magnetogram", "hmi", "aia", "euv"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in solar_keywords)

    def _extract_classification(self, text: str) -> str:
        text_upper = text.upper()
        for cls in self.HALE_CLASSES:
            if cls.upper() in text_upper:
                if cls == "Beta-Gamma" and ("BETA-DELTA" in text_upper or "BETA DELTA" in text_upper):
                    return "Beta-Delta"
                return cls
        indicators_found = {cls: 0 for cls in self.HALE_CLASSES}
        for cls, keywords in self.CLASS_INDICATORS.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    indicators_found[cls] += 1
        if max(indicators_found.values()) > 0:
            return max(indicators_found, key=indicators_found.get)
        return "Unknown"

    def _extract_confidence(self, text: str) -> float:
        for pattern in self.CONFIDENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value > 1:
                    value /= 100
                return min(max(value, 0.0), 1.0)
        return 0.5

    def _get_confidence_level(self, c: float) -> str:
        if c >= 0.85:
            return "high"
        if c >= 0.6:
            return "medium"
        return "low"

    def _extract_reasoning(self, text: str) -> str:
        indicators = ["因为", "依据", "根据", "由于", "判断", "分析", "reason", "because", "based on"]
        lines = text.split("\n")
        capture = False
        result = []
        for line in lines:
            ls = line.strip().lower()
            if any(ind in ls for ind in indicators):
                capture = True
            if capture:
                result.append(line.strip())
                if len(result) > 5:
                    break
        return " ".join(result) if result else text[:500]

    def _estimate_complexity(self, text: str) -> float:
        score = 5.0
        text_lower = text.lower()
        for kw in ["复杂", "混乱", "交错", "纠缠", "高度", "极复杂"]:
            if kw in text_lower:
                score += 1.5
        for kw in ["简单", "清晰", "明显", "分离", "规则"]:
            if kw in text_lower:
                score -= 1.0
        return min(max(score, 0.0), 10.0)

    def _extract_polarity_info(self, text: str) -> str:
        if "正极" in text and "负极" in text:
            return "Mixed polarity detected"
        if "正极" in text:
            return "Predominantly positive polarity"
        if "负极" in text:
            return "Predominantly negative polarity"
        return "Polarity information unclear"

    def _extract_separation_pattern(self, text: str) -> str:
        patterns = {
            "clear_separation": ["明显分开", "清晰分离", "分界明显", "clear separation"],
            "partial_mixing": ["部分混合", "有些不规则", "partial mixing"],
            "strong_mixing": ["严重混合", "混乱", "纠缠", "strongly mixed"],
        }
        for name, kws in patterns.items():
            if any(kw in text.lower() for kw in kws):
                return name
        return "pattern unclear"


# ---------------------------------------------------------------------------
# Main Classifier
# ---------------------------------------------------------------------------

class SolarClassifier:
    """AI-powered solar feature classifier with structured output,
    traceability, and token usage tracking."""

    def __init__(self, deepseek_client=None):
        self.deepseek_client = deepseek_client
        self.text_parser = HaleClassificationParser()
        self.analysis_history: List[SolarRegionAnalysis] = []

    def classify(
        self,
        image_path: str,
        task_id: str = "",
        image_id: str = "",
        context: str = "",
    ) -> SolarRegionAnalysis:
        """Classify a solar image with full traceability and token tracking.

        Pipeline (v2.0):
        1. Local CV preprocessing (disk detection, spot segmentation, feature clustering)
        2. Build prompt from preprocessing results
        3. Send to AI model (or fallback to heuristic)
        4. Parse structured JSON response
        5. Merge AI results with CV detection for hybrid output
        6. Record traceability data
        """
        steps: List[Dict] = []
        preprocess_result = None

        # Step 1: Local CV preprocessing
        try:
            from solar_preprocessor import load_image_cv2
            img_array = load_image_cv2(image_path)
            if img_array is None:
                raise ValueError(f"Failed to load image: {image_path}")
            preprocess_result = preprocess_solar_image(img_array)
            steps.append({
                "step": "cv_preprocessing",
                "sunspots_found": len(preprocess_result.get("sunspots", [])),
                "groups_found": len(preprocess_result.get("sunspot_groups", [])),
                "bright_regions_found": len(preprocess_result.get("bright_regions", [])),
                "disk_detected": preprocess_result.get("solar_disk", {}).get("detected", False),
                "time": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"CV preprocessing failed: {e}")
            # Fallback to legacy feature extraction
            preprocess_result = None
            steps.append({"step": "cv_preprocessing", "success": False, "error": str(e), "time": datetime.now().isoformat()})

        # Step 2: Build prompt
        if preprocess_result:
            feature_report = generate_feature_prompt(preprocess_result)
        else:
            # Legacy feature extraction as fallback
            img_features = extract_image_features(image_path)
            feature_report = self._legacy_feature_report(img_features)

        is_euv = "euv" in image_path.lower() or "aia" in image_path.lower()
        prompt_type = "EUV图像" if is_euv else "磁图"
        user_prompt = ANALYSIS_USER_PROMPT.replace("{{PREPROCESS_REPORT}}", feature_report)
        if context:
            user_prompt += f"\n\n额外上下文: {context}"

        steps.append({"step": "prompt_build", "prompt_type": prompt_type, "time": datetime.now().isoformat()})

        # Step 3: AI analysis
        raw_output = ""
        token_usage = {}
        structured_data = None

        if self.deepseek_client:
            response = self.deepseek_client.chat_with_image(
                image_path,
                f"{ANALYSIS_SYSTEM_PROMPT}\n\n{user_prompt}",
            )
            raw_output = self._extract_response_text(response)
            if hasattr(response, 'usage') and response.usage:
                token_usage = response.usage
            if hasattr(response, 'model') and response.model:
                token_usage["model"] = response.model

            structured_data = parse_structured_json_response(raw_output)

            steps.append({
                "step": "ai_analysis",
                "success": True,
                "raw_output_length": len(raw_output),
                "structured_json_parsed": structured_data is not None,
                "token_usage": token_usage,
                "time": datetime.now().isoformat(),
            })
        else:
            raw_output = "No DeepSeek client configured"
            steps.append({"step": "ai_analysis", "success": False, "fallback": True, "time": datetime.now().isoformat()})

        # Step 4: Build analysis result with hybrid AI+CV fusion
        analysis = self._build_hybrid_analysis(
            structured_data, raw_output, image_path, image_id,
            preprocess_result, token_usage,
        )

        # Step 5: Traceability
        steps.append({"step": "result_assembly", "time": datetime.now().isoformat()})
        analysis.intermediate_steps = {"processing_steps": steps}

        self.analysis_history.append(analysis)
        return analysis

    def _legacy_feature_report(self, img_features: Dict) -> str:
        """Generate a feature report from legacy image features."""
        lines = ["=== 太阳图像特征报告（启发式） ===", ""]
        dims = img_features.get("dimensions", {})
        bright = img_features.get("brightness", {})
        lines.append(f"图像尺寸: {dims.get('width', '?')}x{dims.get('height', '?')} 像素")
        lines.append(f"平均亮度: {bright.get('mean', 0):.1f}")
        lines.append(f"对比度: {img_features.get('contrast', 0):.3f}")
        lines.append(f"边缘密度: {img_features.get('edge_density', 0):.4f}")
        lines.append("")
        lines.append("注意：未启用CV预处理，分析基于基础图像统计。")
        return "\n".join(lines)

    def _build_hybrid_analysis(
        self,
        structured: Optional[Dict],
        raw_text: str,
        image_path: str,
        image_id: str,
        preprocess_result: Optional[Dict],
        token_usage: Dict,
    ) -> SolarRegionAnalysis:
        """Build analysis by fusing AI results with CV preprocessing data.

        Strategy:
        - If AI returned valid structured data: use AI results as primary, CV as validation
        - If AI failed: fall back to CV-based features
        - Always merge CV-detected features that AI may have missed
        """
        analysis = SolarRegionAnalysis()
        analysis.image_path = image_path
        analysis.image_id = image_id or Path(image_path).stem
        analysis.analysis_time = datetime.now().isoformat()
        analysis.raw_model_output = raw_text
        analysis.token_usage = token_usage

        # Get CV-detected features
        cv_features = []
        cv_spots = []
        cv_groups = []
        cv_bright = []
        cv_filaments = []
        cv_prominences = []

        if preprocess_result:
            cv_spots = preprocess_result.get("sunspots", [])
            cv_groups = preprocess_result.get("sunspot_groups", [])
            cv_bright = preprocess_result.get("bright_regions", [])
            cv_filaments = preprocess_result.get("filaments", [])
            cv_prominences = preprocess_result.get("prominences", [])

            disk_info = preprocess_result.get("solar_disk", {})
            disk_cx = disk_info.get("center_x", 0)
            disk_cy = disk_info.get("center_y", 0)
            disk_r = disk_info.get("radius", 1)

            # Convert CV sunspots to SolarFeature
            for i, spot in enumerate(cv_spots):
                group_id = None
                for g in cv_groups:
                    if i in g.get("member_indices", []):
                        group_id = g["id"]
                        break

                # Keep pixel coords and normalized disk coords separately
                px = spot.get("x", 0)
                py = spot.get("y", 0)
                norm_x = round((px - disk_cx) / max(disk_r, 1), 4)
                norm_y = round((py - disk_cy) / max(disk_r, 1), 4)

                cv_features.append(SolarFeature(
                    feature_type="sunspot",
                    label=f"黑子 #{i+1}",
                    position_x=norm_x,
                    position_y=norm_y,
                    pixel_x=px,
                    pixel_y=py,
                    size_relative=round(spot.get("radius", 0) / max(disk_r, 1), 3),
                    confidence=round(spot.get("confidence", 0.5), 3),
                    description=f"CV检测黑子，面积{spot.get('area', 0)}px²，对比度{spot.get('contrast', 0):.2f}",
                    additional_params={
                        "is_group": group_id is not None,
                        "group_id": group_id,
                        "region": spot.get("region", "unknown"),
                        "source": "cv",
                        "index": spot.get("index", i + 1),
                    },
                ))

            # Convert CV bright regions
            label_map = {"flare": "耀斑", "plage": "谱斑", "facula": "光斑", "bright_region": "亮区"}
            for i, region in enumerate(cv_bright):
                px = region.get("x", 0)
                py = region.get("y", 0)
                norm_x = round((px - disk_cx) / max(disk_r, 1), 4)
                norm_y = round((py - disk_cy) / max(disk_r, 1), 4)

                br_type = region.get("type", "bright_region").replace("_candidate", "")
                br_label = label_map.get(br_type, br_type)
                br_area = region.get("area", 0)
                br_ratio = region.get("brightness_ratio", 0)

                cv_features.append(SolarFeature(
                    feature_type=br_type,
                    label=br_label,
                    position_x=norm_x,
                    position_y=norm_y,
                    pixel_x=px,
                    pixel_y=py,
                    size_relative=round(math.sqrt(br_area) / max(disk_r, 1), 3),
                    confidence=round(region.get("confidence", 0.5), 3),
                    description=f"CV检测{br_label}，亮度比{br_ratio:.2f}",
                    additional_params={
                        "brightness_ratio": br_ratio,
                        "source": "cv",
                        "index": region.get("index", i + 1),
                    },
                ))

            # Convert CV filaments (dark elongated structures on disk)
            for i, fil in enumerate(cv_filaments):
                px = fil.get("x", 0)
                py = fil.get("y", 0)
                norm_x = round((px - disk_cx) / max(disk_r, 1), 4)
                norm_y = round((py - disk_cy) / max(disk_r, 1), 4)

                fil_area = fil.get("area", 0)
                fil_contrast = fil.get("brightness_contrast", 0)
                fil_aspect = fil.get("aspect_ratio", 0)

                cv_features.append(SolarFeature(
                    feature_type="filament",
                    label=f"暗条 #{i+1}",
                    position_x=norm_x,
                    position_y=norm_y,
                    pixel_x=px,
                    pixel_y=py,
                    size_relative=round(math.sqrt(fil_area) / max(disk_r, 1), 3),
                    confidence=round(fil.get("confidence", 0.5), 3),
                    description=f"CV检测暗条，面积{fil_area}px²，纵横比{fil_aspect:.1f}，亮度对比{fil_contrast:.2f}σ",
                    additional_params={
                        "area": fil_area,
                        "brightness_contrast": fil_contrast,
                        "elongation_score": fil.get("elongation_score", 0),
                        "aspect_ratio": fil_aspect,
                        "source": "cv",
                        "index": fil.get("index", i + 1),
                    },
                ))

            # Convert CV prominences (limb features)
            for i, prom in enumerate(cv_prominences):
                px = prom.get("x", 0)
                py = prom.get("y", 0)
                norm_x = round((px - disk_cx) / max(disk_r, 1), 4)
                norm_y = round((py - disk_cy) / max(disk_r, 1), 4)

                prom_type = prom.get("type", "prominence")
                prom_label = "日珥" if prom_type == "prominence" else "暗条"
                prom_area = prom.get("area", 0)
                prom_contrast = prom.get("brightness_contrast", 0)

                cv_features.append(SolarFeature(
                    feature_type=prom_type,
                    label=f"{prom_label} #{i+1}",
                    position_x=norm_x,
                    position_y=norm_y,
                    pixel_x=px,
                    pixel_y=py,
                    size_relative=round(math.sqrt(prom_area) / max(disk_r, 1), 3),
                    confidence=round(prom.get("confidence", 0.5), 3),
                    description=f"CV检测{prom_label}，距中心{prom.get('norm_distance', 0):.2f}R，亮度对比{prom_contrast:.2f}σ",
                    additional_params={
                        "norm_distance": prom.get("norm_distance", 0),
                        "brightness_contrast": prom_contrast,
                        "elongation_score": prom.get("elongation_score", 0),
                        "source": "cv",
                        "index": prom.get("index", i + 1),
                    },
                ))

        if structured:
            # === AI Structured JSON parsing (primary) ===
            analysis.is_solar_image = structured.get("is_solar_image", True)
            if not analysis.is_solar_image:
                analysis.hale_classification = "Not Applicable"
                analysis.reasoning = structured.get("reasoning", "图像不是太阳图像")
                analysis.warnings.append("图像可能不是太阳图像")
            else:
                analysis.hale_classification = structured.get("hale_classification", "Unknown")
                analysis.hale_distribution = structured.get("hale_distribution", {})
                analysis.classification_confidence = structured.get("classification_confidence", 0.5)
                analysis.confidence_level = self._confidence_level(analysis.classification_confidence)
                analysis.complexity_score = min(max(structured.get("complexity_score", 5.0), 0), 10)
                analysis.polarity_distribution = structured.get("polarity_distribution", "")
                analysis.separation_pattern = structured.get("separation_pattern", "")
                analysis.region_count = structured.get("region_count", 0)
                analysis.reasoning = structured.get("reasoning", "")
                analysis.scientific_conclusion = structured.get("scientific_conclusion", structured.get("reasoning", ""))

                # Calculate flare risk based on Hale classification
                hale = analysis.hale_classification
                if hale in ("Delta", "Beta-Delta"):
                    risk_level = "极高"
                elif hale == "Gamma":
                    risk_level = "高"
                elif hale == "Beta-Gamma":
                    risk_level = "中等"
                elif hale == "Beta":
                    risk_level = "低"
                else:
                    risk_level = "极低"

                analysis.flare_risk_assessment = {
                    "flare_risk_level": risk_level,
                    "estimated_flare_class": structured.get("estimated_flare_class", ""),
                    "hale_classification": analysis.hale_classification,
                    "complexity_score": analysis.complexity_score,
                }
                analysis.warnings = structured.get("warnings", [])
                analysis.recommended_actions = structured.get("recommended_actions", [])

            # Parse AI features
            ai_features = []
            ai_spot_count = 0
            for fd in structured.get("features", []):
                feat = SolarFeature.from_dict(fd)
                feat.additional_params["source"] = "ai"
                ai_features.append(feat)
                if feat.feature_type == "sunspot":
                    ai_spot_count += 1

            # Merge AI and CV features: prefer AI for classification, CV for position accuracy
            merged_features = []
            cv_used_indices = set()

            for ai_feat in ai_features:
                if ai_feat.feature_type == "sunspot":
                    # Try to match with nearest CV spot
                    best_match = None
                    best_dist = float('inf')
                    for j, cv_feat in enumerate(cv_features):
                        if cv_feat.feature_type != "sunspot" or j in cv_used_indices:
                            continue
                        dist = math.sqrt(
                            (ai_feat.position_x - cv_feat.position_x) ** 2 +
                            (ai_feat.position_y - cv_feat.position_y) ** 2
                        )
                        if dist < 0.3 and dist < best_dist:  # Within 0.3 normalized units
                            best_match = j
                            best_dist = dist

                    if best_match is not None:
                        # Merge: use AI label/confidence but CV precise position
                        cv_used_indices.add(best_match)
                        cv_feat = cv_features[best_match]
                        merged = SolarFeature(
                            feature_type="sunspot",
                            label=ai_feat.label,
                            position_x=cv_feat.position_x,  # Use CV precision (normalized -1~+1)
                            position_y=cv_feat.position_y,
                            pixel_x=cv_feat.pixel_x,
                            pixel_y=cv_feat.pixel_y,
                            size_relative=max(ai_feat.size_relative, cv_feat.size_relative),
                            confidence=max(ai_feat.confidence, cv_feat.confidence),
                            description=ai_feat.description,
                            additional_params={
                                **ai_feat.additional_params,
                                "source": "ai+cv",
                            },
                        )
                        merged_features.append(merged)
                    else:
                        merged_features.append(ai_feat)
                else:
                    # Non-sunspot features: validate against CV data before using AI type
                    # Prevent dark regions (sunspots) from being misclassified as flares/bright regions
                    matched_cv = None
                    for j, cv_feat in enumerate(cv_features):
                        if j in cv_used_indices:
                            continue
                        dist = math.sqrt(
                            (ai_feat.position_x - cv_feat.position_x) ** 2 +
                            (ai_feat.position_y - cv_feat.position_y) ** 2
                        )
                        if dist < 0.3:
                            matched_cv = j
                            break

                    if matched_cv is not None:
                        cv_feat = cv_features[matched_cv]
                        # If CV detected this as a sunspot but AI marked it as bright feature, trust CV
                        if cv_feat.feature_type == "sunspot" and ai_feat.feature_type in ("flare", "bright_region", "plage"):
                            # Keep as sunspot - AI made an error
                            cv_used_indices.add(matched_cv)
                            merged = SolarFeature(
                                feature_type="sunspot",
                                label=ai_feat.label,
                                position_x=cv_feat.position_x,
                                position_y=cv_feat.position_y,
                                pixel_x=cv_feat.pixel_x,
                                pixel_y=cv_feat.pixel_y,
                                size_relative=max(ai_feat.size_relative, cv_feat.size_relative),
                                confidence=max(ai_feat.confidence, cv_feat.confidence),
                                description=f"CV检测黑子（AI误标为{ai_feat.feature_type}，已修正）",
                                additional_params={
                                    **ai_feat.additional_params,
                                    "source": "ai+cv",
                                    "corrected_from": ai_feat.feature_type,
                                },
                            )
                            merged_features.append(merged)
                        else:
                            # Use AI type but with CV position
                            cv_used_indices.add(matched_cv)
                            merged = SolarFeature(
                                feature_type=ai_feat.feature_type,
                                label=ai_feat.label,
                                position_x=cv_feat.position_x,
                                position_y=cv_feat.position_y,
                                pixel_x=cv_feat.pixel_x,
                                pixel_y=cv_feat.pixel_y,
                                size_relative=max(ai_feat.size_relative, cv_feat.size_relative),
                                confidence=max(ai_feat.confidence, cv_feat.confidence),
                                description=ai_feat.description,
                                additional_params={
                                    **ai_feat.additional_params,
                                    "source": "ai+cv",
                                },
                            )
                            merged_features.append(merged)
                    else:
                        # No CV match, use AI feature as-is
                        merged_features.append(ai_feat)

            # Add unmatched CV features (lower threshold for filaments and prominences)
            for j, cv_feat in enumerate(cv_features):
                min_confidence = {
                    "sunspot": 0.6,
                    "flare": 0.5,
                    "bright_region": 0.4,
                    "plage": 0.4,
                    "facula": 0.4,
                    "filament": 0.25,
                    "prominence": 0.15,
                }.get(cv_feat.feature_type, 0.6)
                if j not in cv_used_indices and cv_feat.confidence > min_confidence:
                    merged_features.append(cv_feat)

            analysis.features = merged_features
            analysis.intermediate_steps["structured_json"] = structured

        else:
            # === Fallback: use CV features directly ===
            analysis.is_solar_image = True

            # Determine Hale class from CV data
            n_spots = len(cv_spots)
            n_groups = len(cv_groups)

            if n_spots >= 4 and n_groups >= 1:
                analysis.hale_classification = "Beta-Gamma"
                analysis.classification_confidence = 0.70
            elif n_spots >= 2:
                analysis.hale_classification = "Beta"
                analysis.classification_confidence = 0.75
            elif n_spots >= 1:
                analysis.hale_classification = "Alpha"
                analysis.classification_confidence = 0.65
            else:
                analysis.hale_classification = "Alpha"
                analysis.classification_confidence = 0.50

            analysis.confidence_level = self._confidence_level(analysis.classification_confidence)
            complexity = min(10.0, n_spots * 1.5 + n_groups * 2.0 + len(cv_bright) * 1.0)
            analysis.complexity_score = round(complexity, 1)
            analysis.polarity_distribution = "bipolar (estimated from CV analysis)"
            analysis.separation_pattern = "partial_mixing" if n_groups > 0 else "clear_separation"
            analysis.reasoning = (
                f"基于CV预处理分析：检测到{n_spots}个黑子，{n_groups}个群组，{len(cv_bright)}个亮区。"
                f"{'AI分析不可用，结果基于计算机视觉检测。' if not self.deepseek_client else 'AI分析返回非结构化结果，使用CV数据。'}"
            )

            if not self.deepseek_client:
                analysis.warnings.append("AI分析不可用，使用CV预处理结果")
            else:
                analysis.warnings.append(f"AI分析返回非结构化结果，使用CV数据。原始输出: {raw_text[:100]}")

            analysis.features = cv_features
            analysis.scientific_conclusion = (
                f"该活动区为{analysis.hale_classification}型，"
                f"包含{n_spots}个黑子" + (f"，{n_groups}个群组" if n_groups else "") + "。"
            )

        # Auto-add recommendations
        self._add_smart_recommendations(analysis)

        return analysis

    def _build_analysis(
        self,
        structured: Optional[Dict],
        raw_text: str,
        image_path: str,
        image_id: str,
        img_features: Dict,
        token_usage: Dict,
    ) -> SolarRegionAnalysis:
        """Legacy build method - delegates to hybrid analysis."""
        return self._build_hybrid_analysis(
            structured, raw_text, image_path, image_id, None, token_usage,
        )

    def _heuristic_analysis(
        self, image_path: str, image_id: str, img_features: Dict,
    ) -> SolarRegionAnalysis:
        """Fallback heuristic analysis when AI is not available."""
        analysis = SolarRegionAnalysis()
        analysis.image_path = image_path
        analysis.image_id = image_id or Path(image_path).stem
        analysis.analysis_time = datetime.now().isoformat()
        analysis.is_solar_image = True

        dims = img_features.get("dimensions", {})
        w = dims.get("width", 500)
        h = dims.get("height", 500)
        contrast = img_features.get("contrast", 0.5)
        edge_density = img_features.get("edge_density", 0.1)
        complexity = img_features.get("heuristic_complexity", 5.0)

        # Determine Hale class heuristically
        if contrast > 0.7 and edge_density > 0.3:
            analysis.hale_classification = "Beta-Gamma"
            analysis.classification_confidence = 0.65
        elif contrast > 0.5:
            analysis.hale_classification = "Beta"
            analysis.classification_confidence = 0.70
        else:
            analysis.hale_classification = "Alpha"
            analysis.classification_confidence = 0.60

        analysis.confidence_level = self._confidence_level(analysis.classification_confidence)
        analysis.complexity_score = min(max(complexity, 1.0), 10.0)
        analysis.polarity_distribution = "bipolar (estimated from contrast analysis)"
        analysis.separation_pattern = "partial_mixing" if edge_density > 0.2 else "clear_separation"
        analysis.reasoning = (
            f"基于图像特征的启发式分析：对比度 {contrast:.2f}，边缘密度 {edge_density:.4f}，"
            f"启发式复杂度 {complexity:.1f}/10。"
            f"建议配置 DeepSeek API 密钥以获得精确的 AI 分析结果。"
        )
        analysis.warnings.append("未配置 AI 分析引擎，当前结果为基于图像统计特征的启发式评估")
        analysis.recommended_actions.append("配置 DEEPSEEK_API_KEY 环境变量以启用 AI 分析")

        # Generate features from image stats
        analysis.features = self._generate_heuristic_features(img_features)

        return analysis

    def _generate_heuristic_features(self, img_features: Dict) -> List[SolarFeature]:
        """Generate approximate features from image statistics."""
        features = []
        dims = img_features.get("dimensions", {})
        contrast = img_features.get("contrast", 0.5)
        edge_density = img_features.get("edge_density", 0.1)
        polarity = img_features.get("polarity", {})
        pr = polarity.get("ratio", 1.0)

        # Main active region
        if contrast > 0.3:
            features.append(SolarFeature(
                feature_type="sunspot",
                label=f"Hale Beta{'-Gamma' if edge_density > 0.2 else ''} Region",
                position_x=0.5,
                position_y=0.45,
                size_relative=0.15 + edge_density * 0.1,
                confidence=min(contrast * 0.9, 0.85),
                description=f"Detected active region with contrast {contrast:.2f}",
            ))

        # Secondary region
        if edge_density > 0.15:
            features.append(SolarFeature(
                feature_type="bright_region",
                label="Secondary Region",
                position_x=0.65,
                position_y=0.55,
                size_relative=0.08,
                confidence=0.55,
                description="Secondary bright region detected",
            ))

        # Potential flare
        if contrast > 0.6:
            features.append(SolarFeature(
                feature_type="flare",
                label="M-class Flare Candidate",
                position_x=0.4,
                position_y=0.4,
                size_relative=0.1,
                confidence=min(contrast * 0.5, 0.7),
                description="Potential flare activity based on high contrast",
            ))

        return features

    def _confidence_level(self, c: float) -> str:
        if c >= 0.85:
            return "high"
        if c >= 0.6:
            return "medium"
        return "low"

    def _add_smart_recommendations(self, analysis: SolarRegionAnalysis):
        """Add recommendations based on analysis results."""
        if analysis.classification_confidence < 0.6:
            if not any("置信度较低" in str(w) for w in analysis.warnings):
                analysis.warnings.append("分类置信度较低，建议获取更高质量的图像")
                analysis.recommended_actions.append("使用 Helioviewer Select Area 功能截取高清局部图")

        if analysis.complexity_score > 7:
            if not any("复杂度较高" in str(w) for w in analysis.warnings):
                analysis.warnings.append("该活动区复杂度较高，可能伴随强烈耀斑活动")
                if not any("NOAA" in str(a) for a in analysis.recommended_actions):
                    analysis.recommended_actions.append("建议关注 NOAA Space Weather Prediction Center 的实时预报")

        if analysis.hale_classification in ("Beta-Gamma", "Delta", "Beta-Delta"):
            if not any("NOAA" in str(a) for a in analysis.recommended_actions):
                analysis.recommended_actions.append("建议关注 NOAA 空间天气预报中心的实时预警信息")

    def _extract_response_text(self, response) -> str:
        """Extract text from DeepSeekResponse or legacy dict."""
        if hasattr(response, 'success'):
            if not response.success:
                return f"[Error] {response.error_code}: {response.error_message}"
            return response.content or ""
        if isinstance(response, dict):
            if "error" in response:
                return f"Error: {response['error']}"
            try:
                choices = response.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return str(response)
            except Exception:
                return str(response)
        return str(response)
