# AI_Sun V2.6 版本更新说明

## 版本信息

| 项目 | 内容 |
|------|------|
| **版本号** | V2.6 |
| **发布日期** | 2026-07-04 |
| **上一版本** | V2.5 |
| **兼容性** | 向后兼容 V2.5，建议全新部署 |
| **最低 Python** | 3.9+ |

---

## 一、核心修复 (Critical Bug Fixes)

### 1.1 坐标转换系统重构

**问题描述：**
V2.5 中标注位置系统性偏移，所有标注框和标签偏离实际特征位置。日珥标注出现在完全错误的位置。

**根因：**
- `match_features_to_dark_regions()` 将 AI 的 0~1 归一化坐标覆盖为像素坐标，但未设置 `pixel_position` 字段
- 后续坐标转换逻辑执行 `max(0, min(1, pixel_value))` 将所有匹配特征钳制到右下角 (1.0, 1.0)
- AI 特征缺少 `source` 字段，坐标转换无法区分 AI 特征（0~1 图像相对坐标）与 CV 特征（盘心坐标）

**修复内容：**
- `solar_dark_region_detector.py`：`match_features_to_dark_regions()` 现在正确存储归一化坐标到 `position`，像素坐标到 `pixel_position`
- 新增 `pixel_to_norm()` 辅助函数
- `api/analyze.py`：移除将 AI 归一化坐标错误转换为像素坐标的代码
- `ai_model_adapter.py`：GLM 和 DeepSeek adapter 均为 AI 特征添加 `source="ai"` 标记
- `annotate_image.py`：`_convert_position_to_relative()` 基于 `source` 字段正确区分坐标系

**影响文件：**
- `src/solar_dark_region_detector.py`
- `src/api/analyze.py`
- `src/ai_model_adapter.py`
- `src/annotate_image.py`

---

### 1.2 日珥检测增强

**问题描述：**
V2.5 中大型日珥（特别是边缘火焰状结构）完全无法被检测，即使是非常明显的日珥也会漏检。

**修复内容：**
- **CV检测** (`solar_preprocessor.py`)：
  - 三重尺度局部对比度分析：小尺度 (0.025r), 中尺度 (0.06r), 大尺度 (0.15r)
  - 径向亮度归一化补偿临边增亮/减暗效应
  - 亮度对比阈值从 0.12 降至 0.05
  - 大尺度阈值从 1.0x std 降至 0.35x std
  - 搜索范围扩展至 0.90r ~ 2.0r
  - Clair 增强：对临边区域应用 CLAHE 提高 AI 模型可见度
- **AI检测** (`prompt_templates.py`)：
  - 新增日珥专用多尺度检测指引
  - 复杂场景指引：边缘模糊日珥、亮度极弱日珥、部分遮挡日珥
  - 新增 `prominence_focus` 预设模板（min_confidence=0.25）
- **置信度过滤** (`api/analyze.py`)：
  - 日珥置信度阈值降至 0.15
  - 耀斑置信度阈值 0.20
  - 默认阈值 0.30

**影响文件：**
- `src/solar_preprocessor.py`
- `src/prompt_templates.py`
- `src/api/analyze.py`

---

### 1.3 前端分析页面空白修复

**问题描述：**
检测完成后跳转到分析页面显示空白，overlay canvas 不绘制标注。

**根因：**
- `imgEl.onload` handler 在 `imgEl.src` 赋值之后才设置
- 浏览器缓存的图片不会触发 `onload` 回调

**修复内容：**
- `src/frontend/index.html`：`onload` handler 在 `src` 赋值之前设置
- 添加 `imgEl.complete` 检查处理已缓存图片
- 添加 `onerror` 处理防止网络错误导致空白
- 分析完成后自动隐藏 `processSteps`，显示 `analysisResult`

**影响文件：**
- `src/frontend/index.html`

---

## 二、性能与质量改进 (Enhancements)

### 2.1 依赖完整性

**问题：** `requirements.txt` 缺少 `fpdf2`、`scipy`、`scikit-learn`、`openai` 依赖

**修复：** 补充全部缺失依赖：
```
openai>=1.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
fpdf2>=2.7.0
```

### 2.2 标注尺寸优化

- 日珥标注框最大尺寸从 180px 增至 400px，适配大型日珥
- 标注编号使用全局顺序索引 (`displayIndex`)
- "无"标签文字颜色改为黑色 (`#000000`) 确保可见

### 2.3 PDF报告布局优化

- 自适应排版：图片放置前检查剩余空间，必要时分页
- 中文 SimHei 字体支持
- 置信度颜色编码：绿色 (≥0.7)、橙色 (≥0.4)、红色 (<0.4)

### 2.4 CORS 安全配置

- 移除通配符 origin `"*"`（与 `allow_credentials=True` 冲突）
- 显式列出允许的来源

---

## 三、架构优化

### 3.1 反幻觉规则强化

提示词中强制要求：
- 禁止编造 AR 编号等虚幻名词
- 仅分析当前图像内容
- 不假设图像拍摄日期
- 严格基于视觉证据报告特征

### 3.2 坐标系统标准化

建立统一的坐标转换流水线：
```
AI 特征: position(0~1 img-rel) --[source="ai"]--> pass-through
CV 特征: position(disk-centered) --[source="cv"]--> 转换为 0~1 img-rel
CV 像素: pixel_position(pixel coords) --[>5 check]--> 转换为 0~1 img-rel
```

---

## 四、已知问题与后续计划

| 编号 | 问题 | 优先级 | 计划版本 |
|------|------|--------|----------|
| 1 | AI 模型对小型黑子检测率偏低 | 高 | V2.7 |
| 2 | 暗条 (filament) 检测精度不足 | 中 | V2.7 |
| 3 | 批量分析内存占用需优化 | 低 | V2.8 |

---

## 五、文件变更清单

### 修改的文件

| 文件路径 | 变更类型 | 变更说明 |
|----------|----------|----------|
| `src/solar_dark_region_detector.py` | 重写 | 坐标转换逻辑修复 |
| `src/api/analyze.py` | 修改 | 移除错误坐标转换，添加日珥阈值 |
| `src/ai_model_adapter.py` | 修改 | 添加 source="ai" 标记 |
| `src/annotate_image.py` | 重写 | source 字段坐标系区分 |
| `src/solar_preprocessor.py` | 重写 | v3.0 日珥检测算法 |
| `src/prompt_templates.py` | 修改 | 日珥检测指引增强 |
| `src/frontend/index.html` | 修改 | 图片加载修复、processSteps 隐藏 |
| `src/solar_classifier.py` | 修改 | 置信度阈值调整 |
| `src/pdf_report_generator.py` | 重写 | 自适应排版 |
| `requirements.txt` | 修改 | 添加缺失依赖 |

### 新增的文件

| 文件路径 | 说明 |
|----------|------|
| `docs/CHANGELOG_V2.6.md` | 本文件 |
| `docs/FEATURES_V2.6.md` | 功能介绍文档 |

---

## 六、升级指南

### 从 V2.5 升级

1. **备份数据**：备份 `AI_Sun/data/` 目录下的分析历史和上传图像
2. **更新源代码**：替换所有源文件
3. **安装新依赖**：
   ```bash
   pip install -r requirements.txt
   ```
4. **清理缓存**：删除 `src/__pycache__/` 和 `src/api/__pycache__/`
5. **验证**：启动服务，上传测试图像，检查标注位置是否正确

### 新安装

```bash
# Windows
setup.bat

# Linux/macOS
chmod +x run.sh && ./run.sh
```
