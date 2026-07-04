# SolarInsight - 太阳活动区自动分析系统

## 项目简介

**SolarInsight** 是一个基于多模态大语言模型（LLM）的太阳活动区自动分析系统，通过 Coze 平台实现太阳黑子磁场类型的智能分类。

### 核心功能

- 自动识别 HMI 磁图中的黑白磁极区域
- 基于 Hale 分类法进行磁场类型判断
- 支持多模态分析（磁图 + EUV 图）
- 提供详细的分类报告和置信度评估

---

## 目录结构

```
AI_Sun/
├── bin/                                    # 可执行工具
│   └── evaluate_classification.py          # 分类评估工具
├── src/                                    # 源代码
│   ├── helioviewer_client.py               # Helioviewer API 客户端
│   └── data_acquisition.py                 # 数据采集工具
├── resource/                               # 资源文件
│   ├── coze_config/                        # Coze 智能体配置
│   │   ├── bot_config.json                # 机器人配置
│   │   └── hale_classification_guide.md    # Hale分类知识库
│   └── data/                               # 太阳图像数据
├── doc/                                    # 文档
│   ├── research_plan.md                    # 研究方案
│   └── user_guide.md                       # 使用说明
└── README.md                               # 项目说明
```

---

## 快速开始

### 1. 数据获取

#### 使用 Helioviewer 网站手动获取（推荐新手）

1. 访问 [Helioviewer.org](https://helioviewer.org)
2. 选择观测日期（建议选择太阳活动高峰期）
3. 在 Data Sources 中选择 **SDO** → **HMI** → **Magnetogram**
4. 使用 **Select Area** 功能框选感兴趣的活动区
5. 点击 **Generate Screenshot** 下载图像

#### 使用 API 批量获取

```bash
# 下载磁图
python src/helioviewer_client.py "2024-05-10T12:00:00" magnetogram

# 下载 EUV 171 图
python src/helioviewer_client.py "2024-05-10T12:00:00" euv171

# 批量采集数据
python src/data_acquisition.py --start 2024-05-01 --end 2024-05-10 --count 20
```

### 2. Coze 智能体配置

1. 访问 [Coze](https://kouzi.ai-tab.cn) 并登录
2. 创建新 Bot，命名为 "SolarInsight"
3. 在 **人设与回复逻辑** 中粘贴 `resource/coze_config/bot_config.json` 中的 system_prompt
4. 上传知识库文件 `hale_classification_guide.md`
5. 保存并发布

### 3. 分类评估

```bash
# 创建评估模板
python bin/evaluate_classification.py --create-template

# 运行评估
python bin/evaluate_classification.py -g resource/data/ground_truth.json -p results/predictions.json -o results/evaluation_report.json
```

---

## Hale 分类说明

| 类型 | 描述 | 耀斑风险 |
|------|------|----------|
| **Alpha** | 单极性黑子 | 低 |
| **Beta** | 双极结构，分界清晰 | 中 |
| **Beta-Gamma** | 双极但分界不规则 | 高 |
| **Delta** | 相反极性本影挤在同一半影 | 极高 |

---

## 数据说明

### 推荐的太阳活动区

| 日期 | 活动区编号 | 事件 |
|------|------------|------|
| 2024-05-10 | AR13664 | X8.7 超级耀斑 |
| 2024-05-08 | AR13663 | X2.2 耀斑 |
| 2024-05-05 | AR13661 | M4.5 耀斑 |

### 文件命名规范

```
{日期}_{活动区编号}_{类型}.jpg
例如: 20240510_AR13664_HMI.jpg
```

---

## API 参考

### HelioviewerClient

```python
from src.helioviewer_client import HelioviewerClient

client = HelioviewerClient(api_token="your_token")

# 下载磁图
filepath, data = client.downloadMagnetogram("2024-05-10T12:00:00")

# 下载 EUV 图像
filepath, data = client.downloadEUVImage("2024-05-10T12:00:00", wavelength=171)
```

### SolarDataAcquirer

```python
from src.data_acquisition import SolarDataAcquirer

acquirer = SolarDataAcquirer()

# 采集历史事件
regions = acquirer.collectHistoricalEvents("2024-05-01", "2024-05-10", target_count=20)

# 批量下载
results = acquirer.batchDownload(regions)
```

### HaleClassificationEvaluator

```python
from bin.evaluate_classification import HaleClassificationEvaluator

evaluator = HaleClassificationEvaluator("ground_truth.json")
evaluator.loadPredictions("predictions.json")
results = evaluator.evaluate()
evaluator.printReport()
```

---

## 注意事项

1. **API Token**: Helioviewer API 是可选的，但某些功能需要 token
2. **图像质量**: 建议使用高分辨率局部图以获得最佳分析效果
3. **Coze 限制**: 注意平台的调用频率和文件大小限制
4. **科学严谨**: AI 分析结果仅供参考，重要决策请咨询专业太阳物理学家

---

## 许可证

本项目仅供研究和教育目的使用。

---

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系项目团队。