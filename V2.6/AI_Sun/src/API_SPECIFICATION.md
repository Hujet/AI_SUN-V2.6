# 太阳活动区自动分析系统 - API接口规范

## 概述

本文档定义了太阳活动区自动分析系统的RESTful API接口规范，包括接口列表、请求/响应格式、错误处理等内容。

---

## 一、基础信息

| 项目 | 值 |
|------|------|
| API版本 | v1.0 |
| 基础路径 | `/api/v1` |
| 认证方式 | API Key（可选） |
| 数据格式 | JSON |
| 字符编码 | UTF-8 |

---

## 二、接口列表

### 2.1 图像管理

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 上传图像 | POST | `/images` | 上传太阳观测图像 |
| 获取图像列表 | GET | `/images` | 获取已上传图像列表 |
| 获取图像详情 | GET | `/images/{id}` | 获取单个图像详情 |
| 删除图像 | DELETE | `/images/{id}` | 删除指定图像 |

### 2.2 分析服务

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 执行分析 | POST | `/analyze` | 对指定图像执行分析 |
| 获取分析状态 | GET | `/analyze/{task_id}` | 查询分析任务状态 |

### 2.3 报告管理

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 获取报告列表 | GET | `/reports` | 获取分析报告列表 |
| 获取报告详情 | GET | `/reports/{id}` | 获取单个报告详情 |
| 删除报告 | DELETE | `/reports/{id}` | 删除指定报告 |

### 2.4 案例库

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 获取案例列表 | GET | `/cases` | 获取预设案例列表 |
| 获取案例详情 | GET | `/cases/{id}` | 获取单个案例详情 |

---

## 三、接口详细定义

### 3.1 上传图像

**路径**: `POST /api/v1/images`

**功能**: 上传太阳观测图像文件

**请求头**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Content-Type | string | 是 | multipart/form-data |

**请求体**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image | file | 是 | 太阳图像文件（支持JPG/PNG/TIFF） |
| source | string | 否 | 数据源（如SDO/HMI、SOHO等） |
| wavelength | string | 否 | 波长信息（如171A、193A等） |
| timestamp | string | 否 | 观测时间（ISO 8601格式） |

**成功响应** (201 Created):
```json
{
  "success": true,
  "data": {
    "id": "img-20260320-120000",
    "filename": "sdo_image_20260320.jpg",
    "source": "SDO/HMI",
    "wavelength": "171A",
    "timestamp": "2026-03-20T12:00:00Z",
    "size": 1024000,
    "status": "uploaded",
    "created_at": "2026-03-20T12:00:05Z"
  },
  "message": "图像上传成功"
}
```

**失败响应** (400 Bad Request):
```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE",
    "message": "不支持的文件格式，请上传JPG、PNG或TIFF格式的图像"
  }
}
```

### 3.2 获取图像列表

**路径**: `GET /api/v1/images`

**功能**: 获取已上传图像的分页列表

**请求参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| page | int | 否 | 1 | 页码 |
| limit | int | 否 | 10 | 每页数量 |
| source | string | 否 | - | 按数据源筛选 |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "total": 45,
    "page": 1,
    "limit": 10,
    "items": [
      {
        "id": "img-20260320-120000",
        "filename": "sdo_image_20260320.jpg",
        "source": "SDO/HMI",
        "wavelength": "171A",
        "created_at": "2026-03-20T12:00:05Z"
      }
    ]
  }
}
```

### 3.3 获取图像详情

**路径**: `GET /api/v1/images/{id}`

**功能**: 获取单个图像的详细信息

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 图像ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "img-20260320-120000",
    "filename": "sdo_image_20260320.jpg",
    "source": "SDO/HMI",
    "wavelength": "171A",
    "timestamp": "2026-03-20T12:00:00Z",
    "size": 1024000,
    "status": "analyzed",
    "created_at": "2026-03-20T12:00:05Z",
    "last_analyzed_at": "2026-03-20T12:05:00Z"
  }
}
```

**失败响应** (404 Not Found):
```json
{
  "success": false,
  "error": {
    "code": "IMAGE_NOT_FOUND",
    "message": "指定的图像不存在"
  }
}
```

### 3.4 删除图像

**路径**: `DELETE /api/v1/images/{id}`

**功能**: 删除指定的图像文件

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 图像ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "message": "图像删除成功"
}
```

### 3.5 执行分析

**路径**: `POST /api/v1/analyze`

**功能**: 对指定图像执行太阳活动分析

**请求体**:
```json
{
  "image_id": "img-20260320-120000",
  "analysis_type": "full",
  "options": {
    "include_risk_assessment": true,
    "include_summary": true,
    "detailed_mode": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_id | string | 是 | 图像ID |
| analysis_type | string | 否 | 分析类型：basic/full，默认full |
| options.include_risk_assessment | bool | 否 | 是否包含风险评估，默认true |
| options.include_summary | bool | 否 | 是否包含摘要，默认true |
| options.detailed_mode | bool | 否 | 是否详细模式，默认false |

**成功响应** (202 Accepted):
```json
{
  "success": true,
  "data": {
    "task_id": "task-20260320-120500",
    "image_id": "img-20260320-120000",
    "status": "processing",
    "created_at": "2026-03-20T12:05:00Z"
  },
  "message": "分析任务已创建"
}
```

### 3.6 获取分析状态

**路径**: `GET /api/v1/analyze/{task_id}`

**功能**: 查询分析任务的执行状态

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| task_id | string | 是 | 任务ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "task_id": "task-20260320-120500",
    "image_id": "img-20260320-120000",
    "status": "completed",
    "progress": 100,
    "report_id": "rpt-20260320-120500",
    "created_at": "2026-03-20T12:05:00Z",
    "completed_at": "2026-03-20T12:05:30Z"
  }
}
```

**状态说明**:
| 状态 | 说明 |
|------|------|
| pending | 等待处理 |
| processing | 正在分析 |
| completed | 分析完成 |
| failed | 分析失败 |

### 3.7 获取报告列表

**路径**: `GET /api/v1/reports`

**功能**: 获取分析报告的分页列表

**请求参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| page | int | 否 | 1 | 页码 |
| limit | int | 否 | 10 | 每页数量 |
| risk_level | string | 否 | - | 按风险等级筛选 |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "total": 20,
    "page": 1,
    "limit": 10,
    "items": [
      {
        "id": "rpt-20260320-120500",
        "image_id": "img-20260320-120000",
        "risk_level": "moderate",
        "created_at": "2026-03-20T12:05:30Z"
      }
    ]
  }
}
```

### 3.8 获取报告详情

**路径**: `GET /api/v1/reports/{id}`

**功能**: 获取单个分析报告的详细内容

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 报告ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "rpt-20260320-120500",
    "image_id": "img-20260320-120000",
    "image_info": {
      "source": "SDO/HMI",
      "wavelength": "171A",
      "timestamp": "2026-03-20T12:00:00Z"
    },
    "analysis": {
      "sunspots": [
        {
          "id": "NOAA 3512",
          "position": {"x": 234, "y": 456},
          "size": "large",
          "magnetic_type": "beta-gamma",
          "complexity": "high"
        }
      ],
      "flares": [
        {
          "type": "M-class",
          "intensity": "moderate",
          "location": "active region NOAA 3512"
        }
      ],
      "bright_regions": [
        {
          "type": "plage",
          "location": "solar disk center",
          "size": "medium"
        }
      ],
      "risk_level": "moderate",
      "risk_score": 0.65,
      "recommendations": [
        "密切监测未来24小时活动",
        "准备应对可能的地磁暴"
      ]
    },
    "summary": "图像显示太阳盘面中心区域存在一个大型黑子群（NOAA 3512），磁类型为beta-gamma，具有较高的复杂性。该区域已观测到M级耀斑活动，建议密切监测。",
    "generated_at": "2026-03-20T12:05:30Z",
    "processing_time": 30.5
  }
}
```

### 3.9 删除报告

**路径**: `DELETE /api/v1/reports/{id}`

**功能**: 删除指定的分析报告

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 报告ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "message": "报告删除成功"
}
```

### 3.10 获取案例列表

**路径**: `GET /api/v1/cases`

**功能**: 获取预设太阳活动案例的列表

**请求参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| page | int | 否 | 1 | 页码 |
| limit | int | 否 | 10 | 每页数量 |
| type | string | 否 | - | 按案例类型筛选（sunspot/flare/cme） |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "total": 15,
    "page": 1,
    "limit": 10,
    "items": [
      {
        "id": "case-001",
        "title": "NOAA 3512 黑子群",
        "type": "sunspot",
        "source": "SDO/HMI",
        "wavelength": "171A",
        "description": "2026年3月20日观测到的大型黑子群，具有beta-gamma磁结构"
      }
    ]
  }
}
```

### 3.11 获取案例详情

**路径**: `GET /api/v1/cases/{id}`

**功能**: 获取单个预设案例的详细信息

**路径参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 是 | 案例ID |

**成功响应** (200 OK):
```json
{
  "success": true,
  "data": {
    "id": "case-001",
    "title": "NOAA 3512 黑子群",
    "type": "sunspot",
    "source": "SDO/HMI",
    "wavelength": "171A",
    "timestamp": "2026-03-20T12:00:00Z",
    "description": "2026年3月20日观测到的大型黑子群，具有beta-gamma磁结构，伴随M级耀斑活动",
    "image_url": "/cases/case-001.jpg",
    "ground_truth": {
      "sunspots": [
        {
          "id": "NOAA 3512",
          "magnetic_type": "beta-gamma",
          "complexity": "high"
        }
      ],
      "flares": [
        {
          "type": "M5.2",
          "time": "2026-03-20T11:30:00Z"
        }
      ]
    },
    "created_at": "2026-03-20T12:00:00Z"
  }
}
```

---

## 四、错误处理

### 4.1 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述信息",
    "detail": "可选的详细错误信息"
  }
}
```

### 4.2 错误码列表

| 错误码 | HTTP状态码 | 说明 |
|--------|-----------|------|
| INVALID_FILE | 400 | 无效的文件格式 |
| MISSING_PARAMETER | 400 | 缺少必需参数 |
| IMAGE_NOT_FOUND | 404 | 图像不存在 |
| REPORT_NOT_FOUND | 404 | 报告不存在 |
| CASE_NOT_FOUND | 404 | 案例不存在 |
| TASK_NOT_FOUND | 404 | 任务不存在 |
| ANALYSIS_FAILED | 500 | 分析任务失败 |
| INTERNAL_ERROR | 500 | 系统内部错误 |

---

## 五、数据模型定义

### 5.1 Image（图像）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 图像唯一标识 |
| filename | string | 原始文件名 |
| source | string | 数据源 |
| wavelength | string | 波长信息 |
| timestamp | string | 观测时间（ISO 8601） |
| size | int | 文件大小（字节） |
| status | string | 状态（uploaded/analyzed） |
| created_at | string | 创建时间 |
| last_analyzed_at | string | 最后分析时间（可选） |

### 5.2 Report（报告）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 报告唯一标识 |
| image_id | string | 关联图像ID |
| image_info | object | 图像元信息 |
| analysis | object | 分析结果 |
| summary | string | 分析摘要 |
| generated_at | string | 生成时间 |
| processing_time | float | 处理耗时（秒） |

### 5.3 Analysis（分析结果）

| 字段 | 类型 | 说明 |
|------|------|------|
| sunspots | array | 黑子列表 |
| flares | array | 耀斑列表 |
| bright_regions | array | 亮区列表 |
| risk_level | string | 风险等级 |
| risk_score | float | 风险评分（0-1） |
| recommendations | array | 建议列表 |

### 5.4 Sunspot（黑子）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | NOAA编号 |
| position | object | 位置坐标{x, y} |
| size | string | 大小（small/medium/large） |
| magnetic_type | string | 磁类型 |
| complexity | string | 复杂度（low/medium/high） |

### 5.5 Flare（耀斑）

| 字段 | 类型 | 说明 |
|------|------|------|
| type | string | 耀斑类型（如M-class, X-class） |
| intensity | string | 强度（weak/moderate/strong） |
| location | string | 位置描述 |

---

## 六、使用示例

### 6.1 cURL示例

**上传图像**:
```bash
curl -X POST http://localhost:8000/api/v1/images \
  -F "image=@sdo_image.jpg" \
  -F "source=SDO/HMI" \
  -F "wavelength=171A"
```

**执行分析**:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "img-20260320-120000",
    "analysis_type": "full",
    "options": {
      "include_risk_assessment": true,
      "include_summary": true
    }
  }'
```

**获取报告**:
```bash
curl http://localhost:8000/api/v1/reports/rpt-20260320-120500
```

---

**文档版本**: v1.0  
**创建日期**: 2026年5月