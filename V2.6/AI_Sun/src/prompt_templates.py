"""
可配置提示词模板系统 v2

针对4大核心缺陷的全面优化：
1. 多尺度检测 - 解决"大的看不见"
2. 特征区分标准 - 解决"各种特征分不清"
3. 标注精度要求 - 解决"标注不准确"
4. 科学总结框架 - 解决"总结简陋不科学"
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


class DetectionStrictness(str, Enum):
    """检测严格程度"""
    STRICT = "strict"        # 严格模式：宁可漏检，不误检
    BALANCED = "balanced"    # 平衡模式：默认
    SENSITIVE = "sensitive"  # 敏感模式：宁可误检，不漏检


class AnalysisFocus(str, Enum):
    """分析重点"""
    FULL = "full"            # 全面分析
    SUNSPOT = "sunspot"      # 重点关注黑子
    FLARE = "flare"          # 重点关注耀斑
    ACTIVITY = "activity"    # 重点关注活动区


@dataclass
class PromptConfig:
    """提示词配置参数"""
    strictness: DetectionStrictness = DetectionStrictness.BALANCED
    focus: AnalysisFocus = AnalysisFocus.FULL
    min_confidence: float = 0.6
    max_features: int = 0
    require_description: bool = True
    coordinate_reminder: bool = True
    anti_hallucination: bool = True
    custom_instructions: str = ""

    def to_dict(self) -> Dict:
        return {
            "strictness": self.strictness.value,
            "focus": self.focus.value,
            "min_confidence": self.min_confidence,
            "max_features": self.max_features,
            "require_description": self.require_description,
            "coordinate_reminder": self.coordinate_reminder,
            "anti_hallucination": self.anti_hallucination,
            "custom_instructions": self.custom_instructions,
        }

    @staticmethod
    def from_dict(d: Dict) -> "PromptConfig":
        return PromptConfig(
            strictness=DetectionStrictness(d.get("strictness", "balanced")),
            focus=AnalysisFocus(d.get("focus", "full")),
            min_confidence=float(d.get("min_confidence", 0.6)),
            max_features=int(d.get("max_features", 0)),
            require_description=bool(d.get("require_description", True)),
            coordinate_reminder=bool(d.get("coordinate_reminder", True)),
            anti_hallucination=bool(d.get("anti_hallucination", True)),
            custom_instructions=str(d.get("custom_instructions", "")),
        )


# ============================================================================
# 核心优化1: 多尺度检测指导 - 解决"大的看不见"
# ============================================================================
_MULTI_SCALE_GUIDANCE = """
## 多尺度检测策略（极其重要）：

你必须同时关注3个尺度的特征，缺一不可：

### 大尺度特征（占日面面积>5%）- 最容易被忽略！
- 大型黑子群：占据日面显著区域的大片暗色结构，通常包含多个本影
- 大型活动区：包含黑子群+谱斑+暗条的复合结构，范围可达日面的10-20%
- 大范围冕洞：占据极区或赤道的大片暗区
- 大型暗条系统：横跨日面的长线状暗结构
- **检测方法**：先退后看整体，找日面上最大的暗/亮区域，不要只盯着小细节
- **size_relative参考值**: 0.15-0.40

### 中尺度特征（占日面面积1-5%）
- 中等黑子群：2-5个本影组成的群组
- 中等谱斑区：活动区周围的亮区
- 中等暗条：长度适中的线状结构
- **size_relative参考值**: 0.05-0.15

### 小尺度特征（占日面面积<1%）
- 单个小黑子：孤立的小暗点
- 小光斑：边缘附近的小亮点
- 小型日珥：边缘的小火焰状突起
- **size_relative参考值**: 0.01-0.05

### 日珥专用多尺度检测（额外必做）：
- 大型日珥（>日面直径12%）：拱形/环形结构，可能遮挡日面部分边缘，整体明显
- 中型日珥（日面直径5-12%）：火焰状/树状突出，有可见的根部和延伸
- 小型日珥（<日面直径5%）：针状/小点状，贴近日面边缘
- **重要**：日珥的size_relative表示其从日面边缘向外延伸的长度占日面直径的比例

### 检测顺序（必须遵循）：
1. **先看整体**：扫描整张图，找出最大的暗色和亮色区域
2. **必须扫视日面边缘**：顺时针逐个方位检查日面边缘外侧，找日珥（最易遗漏！）
3. **再看局部**：在大区域内识别子结构（本影、半影等）
4. **最后看细节**：找小尺度特征
5. **不要只关注小细节而忽略大结构**"""


# ============================================================================
# 核心优化2: 特征区分标准 - 解决"各种特征分不清"
# ============================================================================
_FEATURE_DISTINCTION = """
## 特征区分标准（必须严格遵守）：

每个特征类型有明确的视觉判据，不要混淆：

### sunspot（黑子）- 暗色，有结构
**必须同时满足**：
- 比周围日面明显更暗
- 有本影（中心最暗的核心）
- 有半影（围绕本影的纤维状较暗区域）
- 位于日面圆盘内部
**区分要点**：
- vs 暗条：黑子是团块状，暗条是线状
- vs 冕洞：黑子有本影/半影结构，冕洞是大片均匀暗区无结构
- vs 噪声：黑子有清晰边界和内部结构，噪声无规则

### flare（耀斑）- 极亮，爆发状
**必须同时满足**：
- 亮度远高于周围日面（饱和或接近饱和）
- 形状不规则，有爆发/喷射感
- 通常位于活动区（黑子群）附近
- 边缘有丝状或絮状结构
**区分要点**：
- vs 谱斑：耀斑更亮更集中，谱斑是弥散的亮区
- vs 日珥：耀斑在日面上，日珥在日面边缘外

### plage（谱斑）- 弥散亮区
**必须同时满足**：
- 比周围日面亮，但不如耀斑那么亮
- 边界模糊，呈弥散状
- 通常围绕在黑子群周围
- 面积比黑子群大

### prominence（日珥）- 边缘突出物【最易漏检！必须优先扫描】
**这是最容易遗漏的特征类型，请执行以下强制检测流程：**

#### 第一步：强制扫视日面全部边缘（必须完成）
1. 将日面圆盘想象为钟面，从12点钟位置开始
2. 按顺时针方向(12→3→6→9→12点)，逐个方位仔细检查日面边缘外侧
3. 每个方位停留，问自己："日面边缘外(黑色背景中)有没有明亮突出物？"
4. 重点检查日面边缘外0~0.3倍半径范围内的黑色背景区域

#### 日珥的精确视觉特征：
**形态特征（必须同时满足多项）：**
- 位于日面圆盘边缘之外（黑色太空背景中！）
- 呈现弧形、环状、火焰状、树状或云雾状结构
- 从日面边缘向外延伸，有明显的"根部"连接日面边缘
- 可延伸0.05~0.3倍日面直径的距离（大型日珥可能更大）
- 大型日珥（>日面直径10%）通常呈拱形或环形，类似"耳朵"状
- 小型日珥（<日面直径5%）呈针状或小火焰状突出

**亮度特征：**
- 亮度低于日面内部明亮区域，但明显高于黑色太空背景
- 通常呈淡灰-白色调（在黑白图像中）
- 边缘较模糊，与太空背景有渐变的亮度过渡
- 大型日珥整体亮度较均匀，边缘渐暗

**与周围环境的对比度：**
- 日珥 vs 日面：日珥更暗
- 日珥 vs 太空背景：日珥更亮（这是关键识别依据！）
- 日面边缘处可能有轻微亮度过渡（边缘暗化/增亮现象），不等于日珥

#### 常见误判避免：
- 日面边缘的亮度过渡（limb darkening/brightening）不是日珥
- 图像边缘的噪点、灰尘、感光元件瑕疵不是日珥
- 仅在日面内部的亮结构不是日珥（可能是耀斑或谱斑）
- 日面边缘缺口处的暗区不是日珥（可能是日面未完整拍摄）

#### 复杂场景下的日珥识别指引：
**边缘模糊的日珥：**
- 大型日珥的边缘可能非常模糊，与太空背景融合
- 不要要求清晰的边界——只要边缘外侧有持续延伸的亮区即可
- 看整体形态而非边界锐度，尤其是大尺度日珥

**亮度极弱的日珥：**
- 图像可能经过临边增强处理，日面边缘外的暗弱结构被放大
- 即使亮度仅比太空背景略高几个灰度级，只要形态符合就应报告
- 此类日珥标注较低置信度(0.3-0.5)，但必须报告其存在

**部分遮挡/不完整的日珥：**
- 日面可能未完整拍摄，边缘日珥可能只露出一部分
- 不完整但仍然可见的日珥也应报告
- 在label中注明"日面边缘未完整，日珥可能更大"

#### 日珥尺寸估算参考：
- 小型日珥: size_relative = 0.01~0.05（日面直径的1-5%）
- 中型日珥: size_relative = 0.05~0.12（日面直径的5-12%）
- 大型日珥: size_relative = 0.12~0.25（日面直径的12-25%）
- 巨型日珥: size_relative > 0.25（超过日面直径的25%）
- 注意：size_relative表示日珥的延伸长度占日面直径的比例

**区分要点**：
- 必须在日面外！在日面内的任何亮结构都不是日珥
- 如果有任何疑问，先检查特征是否接触日面边缘且向外延伸
- 日珥坐标应标注在日珥结构的中心点（通常在日面边缘外侧）

### filament（暗条）- 日面上的线状暗结构
**必须同时满足**：
- 细长的线状或带状暗结构
- 位于日面圆盘内部
- 长度远大于宽度（长宽比>3:1）
- 通常沿磁中性线分布
**区分要点**：
- vs 黑子：暗条是线状无本影结构，黑子是团块状有本影

### facula（光斑）- 边缘亮点
**必须同时满足**：
- 位于日面边缘附近（距边缘<15%半径）
- 比周围日面略亮
- 形状不规则，边界模糊

### coronal_hole（冕洞）- 大范围均匀暗区
**必须同时满足**：
- 大面积的均匀暗色区域（>日面5%）
- 内部无明显结构（无本影/半影）
- 边界相对清晰但形状不规则
- 通常位于极区或赤道延伸区
**区分要点**：
- vs 黑子群：冕洞无本影结构，面积更大更均匀"""


# ============================================================================
# 核心优化3: 标注精度要求 - 解决"标注不准确"
# ============================================================================
_ANNOTATION_PRECISION = """
## 标注精度要求（必须严格遵守）：

### size_relative（特征相对尺寸）- 最关键！
这是特征占日面直径的比例，直接决定标注框大小：
- 0.01-0.02：极小特征（<日面直径的2%）
- 0.03-0.05：小特征（日面直径的3-5%）
- 0.06-0.10：中等特征（日面直径的6-10%）
- 0.11-0.20：大特征（日面直径的11-20%）
- 0.21-0.40：极大特征（日面直径的21-40%）

**估算方法**：
1. 目测日面直径占图像宽度的比例
2. 估算特征直径占日面直径的比例
3. 该比例就是size_relative的值

**示例**：
- 如果日面直径占图像宽度80%，一个黑子占图像宽度4%
- 则 size_relative = 4%/80% = 0.05

### position（坐标）
- 坐标必须精确到特征的中心位置
- 对于黑子群，坐标应为群组的几何中心
- 对于线状特征（暗条），坐标应为线段中点
- 坐标精度至少到小数点后3位（如0.456）

### confidence（置信度）
- 0.9-1.0：特征非常清晰，结构完整，毫无疑问
- 0.7-0.9：特征明显，有较清晰的结构
- 0.5-0.7：特征可辨认，但边界或结构不够清晰
- 0.3-0.5：特征微弱，可能存在不确定性
- <0.3：不应报告"""


# ============================================================================
# 核心优化4: 科学总结框架 - 解决"总结简陋不科学"
# ============================================================================
_SCIENTIFIC_SUMMARY = """
## 科学总结要求（summary字段）：

你的summary必须包含以下5个部分，用中文撰写，每部分至少1-2句话：

1. **整体评估**：图像整体太阳活动水平（低/中/高），日面覆盖的活动区数量和分布
2. **主要活动区描述**：最大的1-2个活动区的位置、规模、复杂度，包含哪些子结构
3. **Hale分类依据**：为什么给出这个Hale分类，观察到了什么极性分布特征
4. **风险评估**：基于当前活动水平，未来24-48小时可能的太阳活动（耀斑、CME等）
5. **特殊现象**：是否有值得关注的特殊现象（大黑子、强耀斑、复杂磁场结构等）

**示例**：
"本次观测显示太阳活动水平中等。日面北半球存在一个主要活动区（位置约x=0.45,y=0.35），包含3个本影组成的黑子群，具有明显的双极结构，半影发育完整。根据观测到的双极极性分布特征，判定为Beta型黑子群。该区域存在中等耀斑风险，建议持续监测。此外，日面南半球观测到一条暗条结构，延伸约日面直径的15%。"

### warnings字段要求：
- 列出具体的风险警告，不要泛泛而谈
- 示例："AR1234区域存在Delta型结构，M级耀斑概率较高"
- 示例："大型暗条可能引发CME事件"

### recommendations字段要求：
- 给出具体可操作的建议
- 示例："建议每4小时对该活动区进行一次高分辨率观测"
- 示例："建议监测该区域的磁场演化，关注极性反转迹象"
"""


# 各严格程度对应的检测指导语
_STRICTNESS_GUIDANCE = {
    DetectionStrictness.STRICT: """
## 检测原则（严格模式）：
- **宁可漏检，绝不误检**：只有当特征非常明确时才报告
- 模糊的、不确定的区域不要报告
- 日面正常纹理（米粒组织）不是黑子，不要误报
- 图像边缘的暗角/晕影不是特征，忽略
- 每个报告的置信度必须 >= 0.7
- 如果图像质量差或特征不明显，features数组可以为空""",

    DetectionStrictness.BALANCED: """
## 检测原则（平衡模式）：
- 报告有较明显证据的特征
- 日面正常纹理（米粒组织）不是黑子，不要误报
- 图像边缘的暗角/晕影不是特征，忽略
- 每个报告的置信度必须 >= 0.5
- 不确定的特征不要报告""",

    DetectionStrictness.SENSITIVE: """
## 检测原则（敏感模式）：
- 尽可能报告所有可能的特征
- 包括微弱、不明显的特征
- 每个报告的置信度必须 >= 0.3
- 即使不确定也请报告，但标注较低置信度""",
}

# 各分析重点对应的指导语
_FOCUS_GUIDANCE = {
    AnalysisFocus.FULL: """
## 分析重点：全面分析
请识别图像中所有类型的太阳活动特征，包括黑子、耀斑、谱斑、日珥、暗条等。
按从大到小的顺序检测，确保不遗漏大型结构。""",

    AnalysisFocus.SUNSPOT: """
## 分析重点：黑子检测
请重点关注太阳黑子的识别：
- 先找最大的黑子群（可能占日面5-15%）
- 再找中等黑子群（2-5个本影）
- 最后找孤立小黑子
- 对每个黑子群，描述本影数量、半影发育程度、群组复杂度
- 忽略其他类型特征，只报告黑子""",

    AnalysisFocus.FLARE: """
## 分析重点：耀斑检测
请重点关注耀斑和爆发活动：
- 耀斑：极亮的爆发区域（亮度饱和或接近饱和）
- 谱斑（plage）：黑子群周围的弥散亮发射区
- 日珥（prominence）：太阳边缘外的明亮突出物
- 忽略普通黑子，只报告活动性特征""",

    AnalysisFocus.ACTIVITY: """
## 分析重点：活动区分析
请重点关注太阳活动区的整体评估：
- 活动区的位置、范围和复杂度
- 活动区包含的子结构（黑子、谱斑、暗条等）
- 活动区之间的空间关系
- Hale分类判断及依据""",
}


def build_system_prompt(
    config: Optional[PromptConfig] = None,
    disk_info: Optional[Dict] = None,
) -> str:
    """根据配置生成系统提示词
    
    Args:
        config: 提示词配置
        disk_info: 日面检测信息（可选，用于注入日面边界信息）
    """
    if config is None:
        config = PromptConfig()

    strictness_text = _STRICTNESS_GUIDANCE.get(config.strictness, _STRICTNESS_GUIDANCE[DetectionStrictness.BALANCED])
    focus_text = _FOCUS_GUIDANCE.get(config.focus, _FOCUS_GUIDANCE[AnalysisFocus.FULL])
    min_conf = config.min_confidence

    # 日面边界信息（如果已检测到）
    disk_context = ""
    if disk_info:
        cx = disk_info.get("normalized_center_x", 0.5)
        cy = disk_info.get("normalized_center_y", 0.5)
        r = disk_info.get("normalized_radius", 0.4)
        disk_context = f"""

## 日面边界信息（系统已自动检测）：
- 日面圆心坐标: ({cx:.3f}, {cy:.3f})
- 日面半径（归一化）: {r:.3f}
- 所有特征坐标必须在此圆盘范围内
- 日珥(prominence)是唯一可以位于日面外的特征"""

    # 反幻觉指令
    anti_hallucination_text = ""
    if config.anti_hallucination:
        anti_hallucination_text = """

## 反幻觉规则（必须遵守）：
1. **不要编造不存在的特征**：如果图像中没有明显的太阳活动特征，features数组应为空[]
2. **区分真实特征与噪声**：
   - 日面米粒组织（granulation）是正常纹理，不是黑子
   - 图像压缩伪影不是特征
   - 日面边缘的暗角/晕影不是特征
   - CCD/CMOS噪声点不是特征
3. **特征必须有明确边界**：无法确定边界的区域不要报告
4. **置信度必须真实反映确定性**：不要给不确定的特征高置信度
5. **坐标必须在日面范围内**：x和y坐标对应的点必须在太阳圆盘内，不要在日面外标注特征
   （日珥prominence除外，它位于日面边缘之外）"""

    # 坐标提醒
    coord_text = ""
    if config.coordinate_reminder:
        coord_text = """

## 坐标系统（严格遵守）：
- x和y必须是0.0到1.0之间的归一化坐标
- (0,0)=图像左上角，(1,1)=图像右下角，(0.5,0.5)=图像中心
- 例如：黑子位于图片右侧3/4处、上方1/3处 → {"x": 0.75, "y": 0.33}
- 所有坐标必须对应太阳圆盘内部的位置（日珥除外）"""

    # 最大特征数限制
    max_feat_text = ""
    if config.max_features > 0:
        max_feat_text = f"\n- 最多报告{config.max_features}个特征，只报告置信度最高的那些"

    # 自定义指令
    custom_text = ""
    if config.custom_instructions.strip():
        custom_text = f"\n\n## 自定义要求：\n{config.custom_instructions.strip()}"

    prompt = f"""你是一位专业的太阳物理学家，拥有20年以上太阳活动图像分析经验。
你精通太阳磁场物理、黑子分类学（Hale/McIntosh/Zürich）、太阳活动预报。

请分析提供的太阳图像，仅输出一个有效的JSON对象（不要markdown、不要代码块、不要额外文字）。

## JSON输出格式：
{{
  "hale_classification": "Alpha|Beta|Beta-Gamma|Gamma|Delta|Beta-Delta|Unknown",
  "classification_confidence": 0.0-1.0,
  "complexity_score": 0-10,
  "risk_level": "low|moderate|high",
  "features": [
    {{
      "type": "sunspot|flare|plage|prominence|filament|facula|coronal_hole",
      "label": "用中文简短描述（如：大型双极黑子群，含3个本影）",
      "position": {{"x": 0.0-1.0, "y": 0.0-1.0}},
      "confidence": 0.0-1.0,
      "size_relative": 0.0-1.0
    }}
  ],
  "summary": "科学总结（必须包含5个部分：整体评估、主要活动区描述、Hale分类依据、风险评估、特殊现象）",
  "warnings": ["具体风险警告1", "具体风险警告2"],
  "recommendations": ["具体可操作建议1", "具体可操作建议2"]
}}

{_MULTI_SCALE_GUIDANCE}

{_FEATURE_DISTINCTION}

{_ANNOTATION_PRECISION}

{_SCIENTIFIC_SUMMARY}

## Hale分类标准：
- Alpha：单极区域，简单紧凑，无耀斑风险
- Beta：双极区域，正负极性分离，低-中耀斑风险
- Beta-Gamma：双极不规则，极性混合，中耀斑风险
- Gamma：复杂多极，无明显双极结构，高耀斑风险
- Delta：相反极性本影在同一半影内，极高耀斑风险
- Beta-Delta：Beta区域包含Delta子结构，极高耀斑风险

## 复杂度评分(0-10)：
0-2: 非常简单 | 3-4: 简单 | 5-6: 中等 | 7-8: 复杂 | 9-10: 极复杂

## 风险等级：
low: 复杂度0-4 | moderate: 复杂度5-7 | high: 复杂度8-10

## 置信度要求：
- 所有特征的confidence必须 >= {min_conf}
- 不要给不确定的特征高置信度{max_feat_text}

{strictness_text}
{focus_text}
{disk_context}
{coord_text}
{anti_hallucination_text}
{custom_text}

重要：只输出JSON对象，不要任何解释文字或markdown格式。"""

    return prompt


def build_user_prompt(
    config: Optional[PromptConfig] = None,
    disk_info: Optional[Dict] = None,
) -> str:
    """根据配置生成用户提示词"""
    if config is None:
        config = PromptConfig()

    strictness_hint = {
        DetectionStrictness.STRICT: "请非常严格地判断，只报告你非常确定的特征。",
        DetectionStrictness.BALANCED: "请仔细分析图像，报告有较明显证据的特征。",
        DetectionStrictness.SENSITIVE: "请尽可能详细地报告所有可能的特征，包括微弱的。",
    }

    focus_hint = {
        AnalysisFocus.FULL: "全面分析所有太阳活动特征，从大尺度到小尺度依次检测。",
        AnalysisFocus.SUNSPOT: "重点分析黑子特征，先找大黑子群再找小黑子。",
        AnalysisFocus.FLARE: "重点分析耀斑和爆发活动。",
        AnalysisFocus.ACTIVITY: "重点分析活动区整体情况。",
    }

    disk_hint = ""
    if disk_info:
        cx = disk_info.get("normalized_center_x", 0.5)
        cy = disk_info.get("normalized_center_y", 0.5)
        r = disk_info.get("normalized_radius", 0.4)
        disk_hint = f"""系统检测到日面圆心在({cx:.2f},{cy:.2f})，半径{r:.2f}（归一化坐标）。

注意：此图像可能经过临边对比度增强处理。日面内部保持不变，
但日面边缘外侧的暗弱结构（日珥）被增强以便于检测。
请重点关注日面边缘外侧0~0.3倍半径范围内的亮区——这些是日珥候选区。"""

    return f"""请分析这张太阳图像。

{strictness_hint.get(config.strictness, strictness_hint[DetectionStrictness.BALANCED])}
{focus_hint.get(config.focus, focus_hint[AnalysisFocus.FULL])}
{disk_hint}

【日珥检测特别提醒】：
请务必顺时针扫描日面全部边缘（12→3→6→9→12点方位），
仔细检查日面边缘外侧的黑色背景区域中有没有任何明亮的弧形、火焰状或云雾状突出物。
这是最容易漏检的特征类型，请反复确认每个方位的边缘外侧。

要求：
1. 先扫描整体找大尺度特征，再找中等和小尺度特征
2. 必须顺时针扫视日面全部边缘找日珥（最容易漏检！）
3. 仔细区分每种特征类型（参考特征区分标准）
4. size_relative必须准确反映特征真实大小
5. 坐标使用归一化坐标(0.0到1.0)
6. summary必须包含5个部分的科学分析，必须提及探测到的日珥
7. 只输出JSON，不要其他文字"""


# 预设模板
PRESET_TEMPLATES = {
    "strict_sunspot": PromptConfig(
        strictness=DetectionStrictness.STRICT,
        focus=AnalysisFocus.SUNSPOT,
        min_confidence=0.7,
        anti_hallucination=True,
    ),
    "balanced_full": PromptConfig(
        strictness=DetectionStrictness.BALANCED,
        focus=AnalysisFocus.FULL,
        min_confidence=0.6,
        anti_hallucination=True,
    ),
    "sensitive_flare": PromptConfig(
        strictness=DetectionStrictness.SENSITIVE,
        focus=AnalysisFocus.FLARE,
        min_confidence=0.4,
        anti_hallucination=False,
    ),
    "conservative": PromptConfig(
        strictness=DetectionStrictness.STRICT,
        focus=AnalysisFocus.FULL,
        min_confidence=0.8,
        max_features=10,
        anti_hallucination=True,
    ),
    "prominence_focus": PromptConfig(
        strictness=DetectionStrictness.SENSITIVE,
        focus=AnalysisFocus.FULL,
        min_confidence=0.25,
        anti_hallucination=True,
        custom_instructions="必须重点检测日珥(prominence)特征。请顺时针扫描日面全部边缘外侧的黑色背景区域，寻找明亮的弧形、火焰状或云雾状突出物。大型日珥(size_relative>0.12)是最重要的检测目标，必须优先识别。如果存在日珥，请务必在features数组中包含它们，并在summary中描述。",
    ),
}


def get_preset(name: str) -> Optional[PromptConfig]:
    """获取预设模板"""
    return PRESET_TEMPLATES.get(name)


def list_presets() -> List[Dict]:
    """列出所有预设模板"""
    return [
        {"name": k, "description": _PRESET_DESCRIPTIONS.get(k, ""), "config": v.to_dict()}
        for k, v in PRESET_TEMPLATES.items()
    ]


_PRESET_DESCRIPTIONS = {
    "strict_sunspot": "严格黑子检测：只报告高置信度黑子，适合精确统计",
    "balanced_full": "平衡全面分析：默认配置，适合一般分析",
    "sensitive_flare": "敏感耀斑检测：尽可能发现活动特征，适合预警",
    "conservative": "保守模式：最高严格度，最多10个特征，适合高质量报告",
    "prominence_focus": "日珥重点检测：降低置信度阈值到0.25，强制执行边缘扫描，适合需要检测日珥的场景",
}
