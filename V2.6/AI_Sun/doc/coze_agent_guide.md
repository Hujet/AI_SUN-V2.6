# Coze智能体配置说明

## 概述

本文档描述了SolarInsight项目中Coze智能体的配置和使用方法。

## Coze平台简介

Coze（扣子）是字节跳动推出的AI智能体平台，支持创建多模态AI应用。

平台地址: https://kouzi.ai-tab.cn

## 快速开始

### 1. 手动创建智能体（推荐）

1. 访问 Coze 并登录
2. 点击"创建Bot"
3. 填写基本信息:
   - Bot名称: SolarInsight
   - Bot描述: 太阳活动区自动分析助手
4. 在"人设与回复逻辑"中粘贴以下系统提示词:

```
# Role
你是一位专业的太阳物理观测助手，擅长通过视觉分析太阳动力学天文台(SDO)的图像。

# Skills
## 磁图识别
你能识别HMI磁图中的黑白区域。白色代表正磁极(Positive)，黑色代表负磁极(Negative)。灰色背景代表平静区，只有深黑和亮白才代表强磁场区域。

## 分类推理
基于Hale分类法，根据正负磁极的分布形态判断黑子类型

## EUV分析
能够对比极紫外图和磁图，建立磁场-辐射的物理关联

# Hale Classification Rules
- **Alpha**: 单极性黑子，只有一个极性区域，磁极分布简单
- **Beta**: 双极结构，正负极分开明显，磁极分界线清晰
- **Beta-Gamma**: 双极组，但正负极分界线不规则，互相交错，磁场复杂度增加
- **Delta**: 极端复杂配置，相反极性的本影被挤压在同一个半影内，通常与强耀斑相关

# Workflow
1. [图像确认] 首先确认这是否是一张太阳图像
2. [视觉描述] 详细描述你看到的黑子群位置、形态、大小
3. [磁极分析] 分析黑白区域的混合程度、空间关系
4. [分类判断] 根据Hale分类标准给出最可能的分类
5. [结论输出] 输出分类结果和置信度，提醒用户参考NOAA官方数据

# Constraints
- 如果图片模糊，建议用户去Helioviewer.org截取高清局部图
- 保持科学严谨的态度，明确标注分类的不确定性
- 如果无法确定分类，输出最可能的选项而非随意猜测
```

5. 上传知识库文件: `hale_classification_guide.md`
6. 保存并发布

### 2. API方式创建智能体

```python
from src.coze_client import CozeAPIClient, CozeBotManager

# 初始化客户端
client = CozeAPIClient(api_key="your_api_key")
manager = CozeBotManager(client)

# 创建Bot
result = manager.createSolarInsightBot()

# 发布Bot
manager.publishBot("SolarInsight")
```

## 获取API Key

1. 登录 Coze 平台
2. 进入"设置" → "API Settings"
3. 创建新的API Token
4. 保存Token并设置环境变量:
   ```bash
   export COZE_API_KEY="your_api_key"
   ```

## 使用智能体

### 通过Coze网页界面

1. 在 Coze 平台找到已发布的 SolarInsight Bot
2. 点击"使用"进入对话界面
3. 上传太阳磁图图片
4. 输入分析请求，如"请分析这张太阳磁图的Hale分类"

### 通过API调用

```python
from src.coze_client import CozeAPIClient, SolarClassifier

# 初始化
client = CozeAPIClient(api_key="your_api_key")
classifier = SolarClassifier(bot_id="your_bot_id", coze_client=client)

# 分析单张图像
result = classifier.classify("path/to/magnetogram.jpg")
print(f"Classification: {result.hale_classification}")
print(f"Confidence: {result.classification_confidence}")

# 批量分析
results = classifier.batch_classify(["img1.jpg", "img2.jpg"], output_dir="results/")
```

## 知识库配置

推荐上传以下文档到Coze知识库:

1. `hale_classification_guide.md` - Hale分类法完整指南
2. Hale分类标准论文（可选）
3. 太阳活动区观测手册（可选）

知识库启用RAG（检索增强生成）功能，AI会根据问题自动检索相关知识。

## 工作流配置（可选）

如果需要更精确的控制，可以在Coze中创建工作流:

```
Start (接收图片)
    ↓
LLM节点: 图像分析
    ↓
LLM节点: Hale分类判断
    ↓
Condition: 置信度 > 0.7?
    ↓ 是          ↓ 否
输出结果    请求补充信息或更高质量图片
```

## 常见问题

### Q: API调用失败怎么办？

A: 检查以下几点:
1. API Key是否正确
2. Bot是否已发布
3. 网络连接是否正常
4. 图片格式和大小是否符合要求

### Q: 分类结果不准确？

A: 优化建议:
1. 使用更高分辨率的局部图
2. 在提示词中增加更详细的判断标准
3. 启用知识库检索
4. 多次测试并收集bad case进行提示词迭代

### Q: 如何提高置信度？

A: 
1. 使用清晰的HMI磁图
2. 确保活动区在图像中心
3. 分辨率至少512x512
4. 避免噪声过多或过曝的图像

## 图像要求

| 参数 | 要求 |
|------|------|
| 格式 | JPG, PNG, JP2 |
| 最小分辨率 | 512x512 |
| 推荐分辨率 | 1024x1024 |
| 最大文件大小 | 10MB |
| 数据源 | SDO/HMI Magnetogram |

## 分类输出示例

```json
{
  "hale_classification": "Beta-Gamma",
  "classification_confidence": 0.85,
  "confidence_level": "high",
  "reasoning": "该活动区呈双极结构，但正负极分界线不规则，在多个位置存在极性交错，磁场复杂度较高。",
  "warnings": [],
  "recommended_actions": [
    "建议关注NOAA Space Weather Prediction Center的实时预报"
  ]
}
```

## 联系与支持

如有问题，请通过项目GitHub Issues页面反馈。