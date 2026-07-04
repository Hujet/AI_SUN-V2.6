"""
Solar Analysis API Key Management

Provides endpoints for managing external analysis API keys (DeepSeek, OpenAI, etc.),
including connectivity testing, format validation, and basic functionality verification.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/api-keys", tags=["API管理"])

DATA_DIR = Path(__file__).parent.parent / "data"
API_KEYS_FILE = DATA_DIR / "api_keys_store.json"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ApiKeyRecord(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., description="Service provider: deepseek, openai, qwen, custom")
    api_key: str = Field(..., min_length=5)
    api_base_url: Optional[str] = None
    model_name: str = Field(default="default", description="Default model name")
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""
    last_tested_at: str = ""
    test_status: str = "never_tested"  # never_tested | success | failed | timeout
    test_details: Dict[str, Any] = {}
    usage_count: int = 0


class ApiKeyUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    is_active: Optional[bool] = None


class ConnectivityTestResult(BaseModel):
    success: bool
    status: str
    response_time_ms: float = 0
    error_message: str = ""
    model_info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Storage Helpers
# ---------------------------------------------------------------------------

def _read_store() -> Dict:
    try:
        if API_KEYS_FILE.exists():
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read API keys store: {e}")
    return {"keys": [], "default_key": None}


def _write_store(data: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = API_KEYS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(str(tmp), str(API_KEYS_FILE))


# ---------------------------------------------------------------------------
# Connectivity Test
# ---------------------------------------------------------------------------

async def _test_connectivity(record: ApiKeyRecord) -> ConnectivityTestResult:
    """Test API key connectivity, response format, and basic functionality."""
    import httpx

    now = datetime.now().isoformat()
    start_time = time.time()

    base_url = record.api_base_url or ""
    api_key = record.api_key
    model = record.model_name or "default"

    try:
        if record.provider.lower() == "deepseek":
            return await _test_deepseek(api_key, base_url, model, start_time, now)
        elif record.provider.lower() == "openai":
            return await _test_openai(api_key, base_url, model, start_time, now)
        elif record.provider.lower() == "qwen":
            return await _test_qwen(api_key, base_url, model, start_time, now)
        else:
            return await _test_generic(api_key, base_url, model, start_time, now)
    except Exception as e:
        elapsed = round((time.time() - start_time) * 1000, 1)
        return ConnectivityTestResult(
            success=False, status="failed",
            response_time_ms=elapsed,
            error_message=str(e),
        )


async def _test_deepseek(api_key: str, base_url: str, model: str,
                         start_time: float, now: str) -> ConnectivityTestResult:
    """Test DeepSeek API connectivity."""
    import httpx

    endpoint = base_url.rstrip("/") if base_url else "https://api.deepseek.com/v1"

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Test 1: List models (connectivity + format check)
        try:
            resp = await client.get(
                f"{endpoint}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if resp.status_code == 200:
                models_data = resp.json()
                elapsed = round((time.time() - start_time) * 1000, 1)

                # Test 2: Basic chat completion (functionality check)
                try:
                    chat_resp = await client.post(
                        f"{endpoint}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": model or "deepseek-chat",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 10,
                        },
                        timeout=10.0
                    )
                    chat_ok = chat_resp.status_code == 200
                    chat_error = "" if chat_ok else chat_resp.text[:200]
                except Exception as ce:
                    chat_ok = False
                    chat_error = str(ce)

                return ConnectivityTestResult(
                    success=True,
                    status="success",
                    response_time_ms=elapsed,
                    model_info={
                        "model_id": model or "deepseek-chat",
                        "available_models_count": len(models_data.get("data", [])),
                        "chat_test_passed": chat_ok,
                    },
                )
            else:
                elapsed = round((time.time() - start_time) * 1000, 1)
                return ConnectivityTestResult(
                    success=False, status="failed",
                    response_time_ms=elapsed,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.ConnectError as e:
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=False, status="timeout",
                response_time_ms=elapsed,
                error_message=f"Connection failed: {str(e)}",
            )
        except httpx.TimeoutException as e:
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=False, status="timeout",
                response_time_ms=elapsed,
                error_message=f"Request timeout: {str(e)}",
            )


async def _test_openai(api_key: str, base_url: str, model: str,
                       start_time: float, now: str) -> ConnectivityTestResult:
    """Test OpenAI API connectivity."""
    import httpx

    endpoint = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{endpoint}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if resp.status_code == 200:
                models_data = resp.json()
                elapsed = round((time.time() - start_time) * 1000, 1)
                return ConnectivityTestResult(
                    success=True, status="success",
                    response_time_ms=elapsed,
                    model_info={
                        "model_id": model or "gpt-4o-mini",
                        "available_models_count": len(models_data.get("data", [])),
                    },
                )
            else:
                elapsed = round((time.time() - start_time) * 1000, 1)
                return ConnectivityTestResult(
                    success=False, status="failed",
                    response_time_ms=elapsed,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as e:
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=False, status="failed",
                response_time_ms=elapsed,
                error_message=str(e),
            )


async def _test_qwen(api_key: str, base_url: str, model: str,
                     start_time: float, now: str) -> ConnectivityTestResult:
    """Test Qwen/Tongyi API connectivity."""
    import httpx

    endpoint = base_url.rstrip("/") if base_url else "https://dashscope.aliyuncs.com/compatible-mode/v1"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{endpoint}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if resp.status_code == 200:
                models_data = resp.json()
                elapsed = round((time.time() - start_time) * 1000, 1)
                return ConnectivityTestResult(
                    success=True, status="success",
                    response_time_ms=elapsed,
                    model_info={
                        "model_id": model or "qwen-plus",
                        "available_models_count": len(models_data.get("data", [])),
                    },
                )
            else:
                elapsed = round((time.time() - start_time) * 1000, 1)
                return ConnectivityTestResult(
                    success=False, status="failed",
                    response_time_ms=elapsed,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as e:
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=False, status="failed",
                response_time_ms=elapsed,
                error_message=str(e),
            )


async def _test_generic(api_key: str, base_url: str, model: str,
                        start_time: float, now: str) -> ConnectivityTestResult:
    """Generic API connectivity test - just check if endpoint responds."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(base_url, headers={"Authorization": f"Bearer {api_key}"})
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=resp.status_code < 400,
                status="success" if resp.status_code < 400 else "failed",
                response_time_ms=elapsed,
                error_message="" if resp.status_code < 400 else f"HTTP {resp.status_code}",
                model_info={"endpoint": base_url},
            )
        except Exception as e:
            elapsed = round((time.time() - start_time) * 1000, 1)
            return ConnectivityTestResult(
                success=False, status="failed",
                response_time_ms=elapsed,
                error_message=str(e),
            )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/keys", summary="获取API密钥列表")
async def list_api_keys(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
):
    """List all registered API keys with optional filtering."""
    store = _read_store()
    keys = store.get("keys", [])

    if provider:
        keys = [k for k in keys if k.get("provider", "").lower() == provider.lower()]
    if is_active is not None:
        keys = [k for k in keys if k.get("is_active") == is_active]

    # Mask API keys for security
    safe_keys = []
    for k in keys:
        safe = dict(k)
        key_val = safe.get("api_key", "")
        safe["api_key_masked"] = key_val[:6] + "..." + key_val[-4:] if len(key_val) > 10 else "***"
        del safe["api_key"]
        safe_keys.append(safe)

    return {"success": True, "data": {"total": len(safe_keys), "items": safe_keys, "default_key": store.get("default_key")}}


@router.post("/keys", summary="添加API密钥")
async def create_api_key(req: ApiKeyRecord):
    """Add a new API key. Automatically tests connectivity."""
    import uuid

    store = _read_store()
    now = datetime.now().isoformat()

    key_id = req.id or f"key-{uuid.uuid4().hex[:8]}"
    record = {
        "id": key_id,
        "name": req.name,
        "provider": req.provider.lower(),
        "api_key": req.api_key,
        "api_base_url": req.api_base_url or "",
        "model_name": req.model_name or "default",
        "is_active": req.is_active,
        "created_at": now,
        "updated_at": now,
        "last_tested_at": "",
        "test_status": "never_tested",
        "test_details": {},
        "usage_count": 0,
    }

    # Auto-test connectivity
    test_result = await _test_connectivity(ApiKeyRecord(**record))
    record["last_tested_at"] = now
    record["test_status"] = test_result.status
    record["test_details"] = {
        "response_time_ms": test_result.response_time_ms,
        "error_message": test_result.error_message,
        "model_info": test_result.model_info,
    }

    store["keys"].append(record)
    _write_store(store)

    # Refresh model manager to pick up new API key
    try:
        from ai_model_adapter import get_model_manager
        manager = get_model_manager()
        manager.refresh_models()
        logger.info(f"Model manager refreshed after adding API key: {key_id}")
    except Exception as e:
        logger.warning(f"Failed to refresh model manager: {e}")

    safe = dict(record)
    safe["api_key_masked"] = req.api_key[:6] + "..." + req.api_key[-4:]
    del safe["api_key"]

    return {"success": True, "data": safe, "message": "API密钥已添加", "test_result": test_result.dict()}


@router.put("/keys/{key_id}", summary="更新API密钥")
async def update_api_key(key_id: str, req: ApiKeyUpdate):
    """Update an existing API key."""
    store = _read_store()
    keys = store.get("keys", [])

    idx = None
    for i, k in enumerate(keys):
        if k.get("id") == key_id:
            idx = i
            break

    if idx is None:
        raise HTTPException(status_code=404, detail=f"API密钥 {key_id} 不存在")

    now = datetime.now().isoformat()
    record = keys[idx]

    update_data = {k: v for k, v in req.dict().items() if v is not None}
    record.update(update_data)
    record["updated_at"] = now

    # If API key changed, re-test
    if req.api_key is not None:
        test_result = await _test_connectivity(ApiKeyRecord(**record))
        record["last_tested_at"] = now
        record["test_status"] = test_result.status
        record["test_details"] = {
            "response_time_ms": test_result.response_time_ms,
            "error_message": test_result.error_message,
        }

    store["keys"][idx] = record
    _write_store(store)

    # Refresh model manager to pick up updated API key
    try:
        from ai_model_adapter import get_model_manager
        manager = get_model_manager()
        manager.refresh_models()
        logger.info(f"Model manager refreshed after updating API key: {key_id}")
    except Exception as e:
        logger.warning(f"Failed to refresh model manager: {e}")

    return {"success": True, "data": {"id": key_id, "updated_at": now, "test_status": record.get("test_status")}}


@router.delete("/keys/{key_id}", summary="删除API密钥")
async def delete_api_key(key_id: str):
    """Delete an API key."""
    store = _read_store()
    keys = store.get("keys", [])

    new_keys = [k for k in keys if k.get("id") != key_id]
    if len(new_keys) == len(keys):
        raise HTTPException(status_code=404, detail=f"API密钥 {key_id} 不存在")

    # If deleted key was default, clear default
    if store.get("default_key") == key_id:
        store["default_key"] = None

    store["keys"] = new_keys
    _write_store(store)

    # Refresh model manager to remove deleted API key
    try:
        from ai_model_adapter import get_model_manager
        manager = get_model_manager()
        manager.refresh_models()
        logger.info(f"Model manager refreshed after deleting API key: {key_id}")
    except Exception as e:
        logger.warning(f"Failed to refresh model manager: {e}")

    return {"success": True, "message": "API密钥已删除"}


@router.post("/keys/{key_id}/test", summary="测试API连通性")
async def test_api_key(key_id: str):
    """Test API key connectivity, response format, and basic functionality."""
    store = _read_store()
    keys = store.get("keys", [])

    record = None
    idx = None
    for i, k in enumerate(keys):
        if k.get("id") == key_id:
            record = k
            idx = i
            break

    if record is None:
        raise HTTPException(status_code=404, detail=f"API密钥 {key_id} 不存在")

    test_record = ApiKeyRecord(**record)
    test_result = await _test_connectivity(test_record)

    now = datetime.now().isoformat()
    record["last_tested_at"] = now
    record["test_status"] = test_result.status
    record["test_details"] = {
        "response_time_ms": test_result.response_time_ms,
        "error_message": test_result.error_message,
        "model_info": test_result.model_info,
    }
    store["keys"][idx] = record
    _write_store(store)

    return {"success": True, "data": test_result.dict()}


@router.post("/keys/import", summary="批量导入API密钥")
async def import_api_keys(req: List[ApiKeyRecord]):
    """Import multiple API keys from a list. Each key is tested automatically."""
    import uuid

    store = _read_store()
    now = datetime.now().isoformat()
    results = []

    for r in req:
        key_id = r.id or f"key-{uuid.uuid4().hex[:8]}"
        record = {
            "id": key_id,
            "name": r.name,
            "provider": r.provider.lower(),
            "api_key": r.api_key,
            "api_base_url": r.api_base_url or "",
            "model_name": r.model_name or "default",
            "is_active": r.is_active,
            "created_at": now,
            "updated_at": now,
            "last_tested_at": "",
            "test_status": "never_tested",
            "test_details": {},
            "usage_count": 0,
        }

        test_result = await _test_connectivity(ApiKeyRecord(**record))
        record["last_tested_at"] = now
        record["test_status"] = test_result.status
        record["test_details"] = {
            "response_time_ms": test_result.response_time_ms,
            "error_message": test_result.error_message,
            "model_info": test_result.model_info,
        }

        store["keys"].append(record)
        results.append({"id": key_id, "name": r.name, "test_status": test_result.status})

    _write_store(store)
    return {"success": True, "data": {"imported": len(results), "results": results}}


@router.get("/keys/{key_id}", summary="获取单个API密钥详情")
async def get_api_key(key_id: str):
    """Get details of a specific API key."""
    store = _read_store()
    for k in store.get("keys", []):
        if k.get("id") == key_id:
            safe = dict(k)
            key_val = safe.get("api_key", "")
            safe["api_key_masked"] = key_val[:6] + "..." + key_val[-4:] if len(key_val) > 10 else "***"
            del safe["api_key"]
            return {"success": True, "data": safe}
    raise HTTPException(status_code=404, detail=f"API密钥 {key_id} 不存在")


@router.post("/keys/{key_id}/set-default", summary="设置为默认API密钥")
async def set_default_key(key_id: str):
    """Set a key as the default API key for the system."""
    store = _read_store()
    keys = store.get("keys", [])

    if not any(k.get("id") == key_id for k in keys):
        raise HTTPException(status_code=404, detail=f"API密钥 {key_id} 不存在")

    store["default_key"] = key_id
    _write_store(store)

    return {"success": True, "message": "已设置为默认API密钥", "data": {"default_key": key_id}}


@router.get("/default-key", summary="获取默认API密钥")
async def get_default_key():
    """Get the default API key info (masked)."""
    store = _read_store()
    default_id = store.get("default_key")
    if not default_id:
        return {"success": True, "data": None}

    for k in store.get("keys", []):
        if k.get("id") == default_id:
            safe = dict(k)
            key_val = safe.get("api_key", "")
            safe["api_key_masked"] = key_val[:6] + "..." + key_val[-4:] if len(key_val) > 10 else "***"
            del safe["api_key"]
            return {"success": True, "data": safe}

    return {"success": True, "data": None}
