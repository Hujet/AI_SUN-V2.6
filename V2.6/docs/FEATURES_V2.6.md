# AI_Sun V2.6 - 太阳活动区自动分析系统 功能介绍

## 系统概述

AI_Sun V2.6 是一个基于多模态大语言模型和计算机视觉技术的太阳活动区自动分析系统。系统能够自动检测、分类和标注太阳图像中的各类活动特征，生成详细的分析报告和可视化标注图像。

**核心能力：**
- 黑子 (Sunspot) / 黑子群 (Sunspot Group) 检测与分类
- 日珥 (Prominence) 检测 - 多尺度增强
- 耀斑 (Flare) / 谱斑 (Plage) / 光斑 (Facula) 识别
- 暗条 (Filament) / 冕洞 (Coronal Hole) 检测
- Hale 分类与太阳活动风险评估
- 自动标注图像生成与 PDF 报告导出

---

## 一、快速开始

### 1.1 环境要求

| 组件 | 最低版本 |
|------|----------|
| Python | 3.9+ |
| pip | 21.0+ |

### 1.2 一键安装

**Windows:**
```batch
setup.bat
```
**Linux/macOS:**
```bash
chmod +x run.sh && ./run.sh
```

脚本会自动完成：Python 检测 → 虚拟环境创建 → 依赖安装 → `.env` 配置引导 → 服务启动。

### 1.3 手动安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入您的 DEEPSEEK_API_KEY
python src/app.py
```

### 1.4 API Key 获取

需要 DeepSeek API Key：
1. 注册：https://platform.deepseek.com/
2. 获取 API Key
3. 填入 `.env` 文件的 `DEEPSEEK_API_KEY` 字段

---

## 二、核心功能

### 2.1 图像上传与分析

**Web界面：**
1. 打开 `http://localhost:8000`
2. 点击「上传图像」标签页
3. 选择太阳图像文件（支持 JPG/PNG）
4. 点击「开始分析」

**API调用：**
```bash
curl -X POST http://localhost:8000/api/v1/images/upload \
  -F "file=@solar_image.jpg"
```

**分析流程：**
```
图像上传 → 日面定位 → CV预处理 → AI视觉分析
→ CV特征匹配 → 置信度过滤 → 标注生成 → 报告输出
```

### 2.2 标注图像生成

系统自动生成带编号标注的分析图像：

| 特征类型 | 颜色 | 标记样式 |
|----------|------|----------|
| 黑子 | 绿色 (#00FF00) | 方形框 + 十字准星 |
| 黑子群 | 蓝色 (#00BFFF) | 虚线矩形 |
| 日珥 | 金色 (#FFD700) | 椭圆弧 + 菱形标记 |
| 耀斑 | 红色 (#FF4444) | 星形标记 |
| 谱斑/光斑 | 浅蓝 (#00BFFF) | 圆形标记 |

**标注信息：** 每个特征包含编号、类型、置信度和位置坐标，可在 Web 界面交互式查看。

### 2.3 Hale 分类

系统基于检测到的特征自动进行 Hale 磁分类：

| 分类 | 含义 |
|------|------|
| Alpha | 单极黑子群 |
| Beta | 双极黑子群 |
| Beta-Gamma | 复杂双极结构 |
| Gamma | 异常极性分布 |
| Delta | 相反极性本影共处同一半影 |
| Beta-Delta | Beta + Delta 复合型 |

### 2.4 风险等级评估

基于复杂度评分自动判定活动风险：
- **低风险** (评分 < 5)：简单的 Alpha/Beta 结构
- **中等风险** (评分 5-7)：Beta-Gamma 结构
- **高风险** (评分 ≥ 8)：Delta/Beta-Delta 结构

### 2.5 PDF 报告生成

详细分析报告包含：
- 原始图像与标注图像对比
- 检测特征清单（编号、类型、置信度、位置）
- Hale 分类与风险等级
- 量化指标统计
- AI 分析摘要与建议
- 置信度颜色编码

**导出格式：** 单份报告 PDF、批量导出 PDF

### 2.6 数据持久化与可追溯性

- **分析历史**：所有分析结果持久化存储
- **追溯记录**：完整的处理流程追溯（输入图像 → CV处理 → AI分析 → 标注）
- **Token 用量**：记录每次 API 调用的 Token 消耗
- **特征反馈**：支持对检测结果进行人工确认/标记

---

## 三、API 接口概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/images/upload` | POST | 上传太阳图像 |
| `/api/v1/analyze/{image_id}` | POST | 启动分析任务 |
| `/api/v1/analyze/{task_id}` | GET | 查询任务状态 |
| `/api/v1/analyze/{task_id}/report` | GET | 获取分析报告 |
| `/api/v1/analyze/{task_id}/annotated` | GET | 获取标注图像 |
| `/api/v1/analyze/{task_id}/report-pdf` | GET | 下载PDF报告 |
| `/api/v1/analyze/{task_id}/report-image` | GET | 获取报告图像 |
| `/api/v1/statistics/summary` | GET | 统计概览 |
| `/api/v1/statistics/history` | GET | 分析历史 |
| `/api/v1/api-keys/test` | POST | 测试API密钥 |
| `/api/v1/images/{image_id}/file` | GET | 获取原始图像 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger API文档 |

---

## 四、技术架构

### 4.1 整体架构

```
┌─────────────────────────────────────────────┐
│                Frontend (SPA)                │
│         index.html + Canvas Overlay          │
├─────────────────────────────────────────────┤
│           FastAPI Backend Server             │
│  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Images  │ │ Analyze  │ │ Statistics  │  │
│  │  API    │ │   API    │ │    API      │  │
│  └─────────┘ └──────────┘ └─────────────┘  │
├─────────────────────────────────────────────┤
│               Processing Layer              │
│  ┌──────────────┐ ┌──────────────────────┐  │
│  │ solar_       │ │ solar_               │  │
│  │ preprocessor │ │ classifier           │  │
│  │ (CV + disk)  │ │ (AI + classification)│  │
│  └──────────────┘ └──────────────────────┘  │
│  ┌──────────────┐ ┌──────────────────────┐  │
│  │ annotate_    │ │ pdf_report_          │  │
│  │ image        │ │ generator            │  │
│  └──────────────┘ └──────────────────────┘  │
├─────────────────────────────────────────────┤
│            Persistent Storage               │
│  ┌────────┐ ┌────────┐ ┌────────────────┐  │
│  │ tasks  │ │reports │ │ traceability   │  │
│  │ store  │ │ store  │ │ store          │  │
│  └────────┘ └────────┘ └────────────────┘  │
└─────────────────────────────────────────────┘
```

### 4.2 核心模块

| 模块 | 职责 |
|------|------|
| `solar_preprocessor.py` | CV预处理：日面定位、黑子/日检测 |
| `solar_classifier.py` | AI分类器：Hale分类、特征解析 |
| `solar_dark_region_detector.py` | CV暗区检测与AI-CV匹配 |
| `annotate_image.py` | 标注图像生成（matplotlib） |
| `pdf_report_generator.py` | PDF报告生成（fpdf2） |
| `ai_model_adapter.py` | AI模型统一接口（GLM/DeepSeek） |
| `prompt_templates.py` | 可配置提示词模板系统 |
| `persistent_store.py` | 线程安全JSON持久化存储 |
| `deepseek_client.py` | DeepSeek API客户端 |

### 4.3 坐标系统

```
┌────────────────────────────────────────┐
│  AI 特征坐标系 (0~1 图像相对)          │
│  (0,0) = 左上角, (1,1) = 右下角       │
│                                        │
│  CV 特征坐标系 (盘心归一化)            │
│  (0,0) = 日面中心, (±1,0) = 左右边缘  │
│                                        │
│  pixel_position (像素坐标)             │
│  原始图像像素坐标, CV检测精确位置       │
│                                        │
│  标注坐标系 (0~1 图像相对)             │
│  统一输出格式, 用于 matplotlib 标注    │
└────────────────────────────────────────┘
```

### 4.4 坐标转换流水线

```
AI 特征 → position(0~1 img-rel) → [source="ai"] → pass-through
CV 特征 → position(disk-centered) → [source="cv"] → disk_to_img()
CV 像素 → pixel_position(pixel) → [>5 check] → pixel/img_size()
```

---

## 五、配置说明

### 5.1 环境变量 (.env)

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥 |
| `DEEPSEEK_API_BASE_URL` | 否 | API 基础地址 (默认官方) |
| `DEEPSEEK_MODEL` | 否 | 模型名称 (默认 deepseek-chat) |
| `DEEPSEEK_TIMEOUT` | 否 | 请求超时秒数 (默认120) |
| `HOST` | 否 | 服务监听地址 (默认 0.0.0.0) |
| `PORT` | 否 | 服务端口 (默认 8000) |

### 5.2 检测参数 (Web界面)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 严格度 | balanced | strict/balanced/sensitive |
| 分析重点 | full | full/sunspot/flare/activity |
| 最低置信度 | 0.6 | AI特征最低置信度阈值 |
| 最多特征数 | 0(不限制) | 限制返回特征数量 |
| 反幻觉 | 开启 | 禁止编造不存在特征 |

---

## 六、技术亮点

### 6.1 三重尺度日珥检测 (V2.6)

- **小尺度** (0.025r sigma)：捕捉针状日珥
- **中尺度** (0.06r sigma)：捕捉火焰/灌木状日珥
- **大尺度** (0.15r sigma)：捕捉拱形/环形大日珥

配合径向亮度归一化+CLAHE临边增强，大幅提升日珥识别率。

### 6.2 AI+CV 混合检测

AI视觉分析提供语义理解，CV计算机视觉提供精确像素定位，二者互补确保检测精度和覆盖度。

### 6.3 可配置提示词系统

支持运行时调整AI检测的严格度、关注重点和置信度阈值，适应不同类型的太阳图像分析需求。

### 6.4 完整的可追溯性

每次分析都记录完整的处理流程，包括原始AI输出、中间步骤、坐标验证和人工反馈，确保分析结果可审查。

---

## 七、目录结构

```
AI_Sun/
├── src/                          # 源代码
│   ├── api/                      # API 路由
│   │   ├── analyze.py            # 分析核心 + 报告接口
│   │   ├── images.py             # 图像上传管理
│   │   ├── statistics.py         # 统计与可视化
│   │   ├── models.py             # AI 模型管理
│   │   └── ...
│   ├── data/                     # 运行数据
│   │   ├── uploads/              # 上传图像
│   │   ├── annotated/            # 标注图像
│   │   ├── reports/              # PDF 报告
│   │   └── ...
│   ├── frontend/                 # 前端
│   │   └── index.html            # SPA 单页应用
│   ├── app.py                    # 主入口
│   ├── solar_preprocessor.py     # CV预处理
│   ├── solar_classifier.py       # AI分类器
│   ├── annotate_image.py         # 标注生成
│   ├── pdf_report_generator.py   # PDF生成
│   ├── ai_model_adapter.py       # AI模型适配
│   ├── prompt_templates.py       # 提示词模板
│   └── persistent_store.py       # 数据持久化
├── docs/                         # 文档
│   ├── CHANGELOG_V2.6.md         # 版本更新说明
│   └── FEATURES_V2.6.md          # 功能介绍
├── .env.example                  # 环境配置模板
├── .gitignore                    # Git 忽略规则
├── requirements.txt              # Python 依赖
├── setup.bat                     # Windows 安装启动
└── run.sh                        # Linux/macOS 安装启动
```

---

## 八、常见问题

**Q: 分析结果为空或特征很少？**
A: 检查 `.env` 中 `DEEPSEEK_API_KEY` 是否正确配置，运行健康检查接口 `/health` 确认 API 连接状态。

**Q: 标注位置不准确？**
A: V2.6 已全面修复坐标转换问题。如仍有偏差，请确认图像中日面位于画面中央区域。

**Q: 如何提高日珥检测率？**
A: 在 Web 界面配置中选择 "prominence_focus" 预设模板，或降低最低置信度阈值。

**Q: 支持哪些图像格式？**
A: 支持 JPG、JPEG、PNG 格式。推荐使用清晰的太阳全日面图像。
