"""
AI Model Adapter System

Provides a unified interface for AI models:
- GLM-4V (Zhipu AI) - Primary vision analysis model
- DeepSeek V3 - Text analysis (backup)
"""

import os
import json
import base64
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

API_KEYS_FILE = Path(__file__).parent / "data" / "api_keys_store.json"


def _load_api_keys_store() -> Dict:
    """Load API keys from the store."""
    try:
        if API_KEYS_FILE.exists():
            with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load API keys store: {e}")
    return {"keys": [], "default_key": None}


def _find_api_key_for_provider(provider: str) -> Optional[Dict]:
    """Find an active API key for the given provider."""
    store = _load_api_keys_store()
    for key in store.get("keys", []):
        if (key.get("provider", "").lower() == provider.lower() and 
            key.get("is_active", True)):
            return key
    return None


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str = ""
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    model_id: str = ""
    supports_vision: bool = True
    max_tokens: int = 4096
    temperature: float = 0.1
    description: str = ""
    strengths: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'provider': self.provider,
            'api_key': self.api_key[:8] + '****' if len(self.api_key) > 8 else '',
            'base_url': self.base_url,
            'model_id': self.model_id,
            'supports_vision': self.supports_vision,
            'description': self.description,
            'strengths': self.strengths,
            'enabled': self.enabled,
        }


@dataclass
class AIAnalysisResult:
    """Standardized result from AI model analysis."""
    success: bool = False
    model_name: str = ""
    provider: str = ""
    analysis_time: str = ""
    processing_time_ms: float = 0
    
    features: List[Dict] = field(default_factory=list)
    hale_classification: str = "Unknown"
    classification_confidence: float = 0.0
    complexity_score: float = 0.0
    risk_level: str = "unknown"
    
    sunspot_count: int = 0
    flare_count: int = 0
    bright_region_count: int = 0
    total_features: int = 0
    mean_confidence: float = 0.0
    
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_output: str = ""
    
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'model_name': self.model_name,
            'provider': self.provider,
            'analysis_time': self.analysis_time,
            'processing_time_ms': self.processing_time_ms,
            'features': self.features,
            'hale_classification': self.hale_classification,
            'classification_confidence': self.classification_confidence,
            'complexity_score': self.complexity_score,
            'risk_level': self.risk_level,
            'sunspot_count': self.sunspot_count,
            'flare_count': self.flare_count,
            'bright_region_count': self.bright_region_count,
            'total_features': self.total_features,
            'mean_confidence': self.mean_confidence,
            'summary': self.summary,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'raw_output': self.raw_output,
            'error': self.error,
        }


class BaseModelAdapter(ABC):
    """Abstract base class for AI model adapters."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def analyze_image(self, image_path: str, system_prompt: str, user_prompt: str) -> AIAnalysisResult:
        """Analyze a solar image and return structured results."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the model API is accessible."""
        pass
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _parse_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from model response text."""
        import re
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r'\{[\s\S]*\}', response_text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from response: {response_text[:200]}")
        return None


class GLMVisionAdapter(BaseModelAdapter):
    """Adapter for GLM-4V (Zhipu AI) vision model.
    
    Uses OpenAI-compatible API format.
    Base URL: https://open.bigmodel.cn/api/paas/v4
    Model: glm-4v-flash
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.base_url:
            config.base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not config.model_id:
            config.model_id = "glm-4v-flash"
    
    def analyze_image(self, image_path: str, system_prompt: str, user_prompt: str) -> AIAnalysisResult:
        start_time = time.time()
        result = AIAnalysisResult(
            model_name=self.config.name,
            provider=self.config.provider,
            analysis_time=datetime.now().isoformat(),
        )
        
        try:
            base64_image = self._encode_image_to_base64(image_path)
            
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            result.raw_output = response.choices[0].message.content
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            parsed = self._parse_json_from_response(result.raw_output)
            if parsed:
                self._populate_result(result, parsed)
                result.success = True
            else:
                result.error = "AI返回了非结构化结果，无法解析为JSON"
                logger.warning(f"GLM returned non-structured result: {result.raw_output[:300]}")
            
        except Exception as e:
            result.error = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"GLM analysis failed: {e}")
        
        return result
    
    def test_connection(self) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=10,
            )
            return response is not None
        except Exception as e:
            logger.error(f"GLM connection test failed: {e}")
            return False
    
    def _populate_result(self, result: AIAnalysisResult, parsed: Dict):
        result.hale_classification = parsed.get('hale_classification', 'Unknown')
        result.classification_confidence = float(parsed.get('classification_confidence', 0.0))
        result.complexity_score = float(parsed.get('complexity_score', 0.0))
        result.risk_level = parsed.get('risk_level', 'unknown')
        result.summary = parsed.get('summary', '')
        result.warnings = parsed.get('warnings', [])
        result.recommendations = parsed.get('recommendations', [])
        
        features = parsed.get('features', [])
        # Mark AI features with source="ai" so coordinate conversion knows
        # these are already 0~1 image-relative coords (not disk-centered)
        for f in features:
            params = f.get("additional_params", {})
            params["source"] = "ai"
            f["additional_params"] = params
        result.features = features
        result.total_features = len(features)
        result.sunspot_count = sum(1 for f in features if f.get('type') == 'sunspot')
        result.flare_count = sum(1 for f in features if f.get('type') in ['flare', 'flare_candidate'])
        result.bright_region_count = sum(1 for f in features if f.get('type') in ['bright_region', 'plage'])
        confidences = [f.get('confidence', 0) for f in features if f.get('confidence')]
        result.mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0


class DeepSeekAdapter(BaseModelAdapter):
    """Adapter for DeepSeek API models (text-only)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.base_url:
            config.base_url = "https://api.deepseek.com/v1"
        if not config.model_id:
            config.model_id = "deepseek-chat"
    
    def analyze_image(self, image_path: str, system_prompt: str, user_prompt: str) -> AIAnalysisResult:
        start_time = time.time()
        result = AIAnalysisResult(
            model_name=self.config.name,
            provider=self.config.provider,
            analysis_time=datetime.now().isoformat(),
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            result.raw_output = response.choices[0].message.content
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            parsed = self._parse_json_from_response(result.raw_output)
            if parsed:
                self._populate_result(result, parsed)
                result.success = True
            else:
                result.error = "AI返回了非结构化结果，无法解析为JSON"
                logger.warning(f"DeepSeek returned non-structured result: {result.raw_output[:300]}")
            
        except Exception as e:
            result.error = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"DeepSeek analysis failed: {e}")
        
        return result
    
    def test_connection(self) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=10,
            )
            return response is not None
        except Exception as e:
            logger.error(f"DeepSeek connection test failed: {e}")
            return False
    
    def _populate_result(self, result: AIAnalysisResult, parsed: Dict):
        result.hale_classification = parsed.get('hale_classification', parsed.get('haleClassification', 'Unknown'))
        result.classification_confidence = float(parsed.get('classification_confidence', parsed.get('classificationConfidence', 0.0)))
        result.complexity_score = float(parsed.get('complexity_score', parsed.get('complexityScore', 0.0)))
        result.risk_level = parsed.get('risk_level', parsed.get('riskLevel', 'unknown'))
        result.summary = parsed.get('summary', parsed.get('analysisSummary', ''))
        result.warnings = parsed.get('warnings', [])
        result.recommendations = parsed.get('recommendations', [])
        
        features = parsed.get('features', parsed.get('detectedFeatures', []))
        # Mark AI features with source="ai" so coordinate conversion knows
        # these are already 0~1 image-relative coords (not disk-centered)
        for f in features:
            params = f.get("additional_params", {})
            params["source"] = "ai"
            f["additional_params"] = params
        result.features = features
        result.total_features = len(features)
        result.sunspot_count = sum(1 for f in features if f.get('type') in ['sunspot', 'black_spot'])
        result.flare_count = sum(1 for f in features if f.get('type') in ['flare', 'flare_candidate'])
        result.bright_region_count = sum(1 for f in features if f.get('type') in ['bright_region', 'plage', 'Plage'])
        
        confidences = [float(f.get('confidence', f.get('conf', 0))) for f in features if f.get('confidence') or f.get('conf')]
        result.mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0


class ModelManager:
    """Manages AI model adapters."""
    
    DEFAULT_MODELS = {
        'glm': ModelConfig(
            name='GLM-4V 智谱视觉',
            provider='glm',
            api_key=os.environ.get('GLM_API_KEY', ''),
            base_url='https://open.bigmodel.cn/api/paas/v4',
            model_id='glm-4v-flash',
            supports_vision=True,
            max_tokens=1024,  # GLM-4V-Flash限制: [1, 1024]
            description='智谱GLM-4V-Flash - 图像视觉分析',
            strengths=['视觉分析', '太阳特征识别', '中文优化'],
            enabled=True,
        ),
        'deepseek': ModelConfig(
            name='DeepSeek V3',
            provider='deepseek',
            api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
            base_url='https://api.deepseek.com/v1',
            model_id='deepseek-chat',
            supports_vision=False,
            description='DeepSeek V3 - 文本分析（备用）',
            strengths=['Hale分类', '复杂度评估', '文本分析'],
            enabled=True,
        ),
    }
    
    def __init__(self):
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        for key, default_config in self.DEFAULT_MODELS.items():
            api_key_record = _find_api_key_for_provider(default_config.provider)
            
            config = ModelConfig(
                name=default_config.name,
                provider=default_config.provider,
                api_key='',
                base_url=default_config.base_url,
                model_id=default_config.model_id,
                supports_vision=default_config.supports_vision,
                max_tokens=default_config.max_tokens,
                description=default_config.description,
                strengths=default_config.strengths,
                enabled=False,
            )
            
            if api_key_record:
                config.api_key = api_key_record.get("api_key", "")
                if api_key_record.get("api_base_url"):
                    config.base_url = api_key_record["api_base_url"]
                if api_key_record.get("model_name") and api_key_record["model_name"] != "default":
                    config.model_id = api_key_record["model_name"]
                config.enabled = True
            elif os.environ.get(f'{default_config.provider.upper()}_API_KEY'):
                config.api_key = os.environ[f'{default_config.provider.upper()}_API_KEY']
                config.enabled = True
            
            self._configs[key] = config
            adapter = self._create_adapter(config)
            if adapter:
                self._adapters[key] = adapter
    
    def _create_adapter(self, config: ModelConfig) -> Optional[BaseModelAdapter]:
        if not config.api_key:
            logger.warning(f"Model {config.name} not enabled (no API key)")
            return None
        try:
            adapters = {
                'glm': GLMVisionAdapter,
                'deepseek': DeepSeekAdapter,
            }
            adapter_class = adapters.get(config.provider)
            if adapter_class:
                return adapter_class(config)
        except Exception as e:
            logger.error(f"Failed to create adapter for {config.name}: {e}")
        return None
    
    def refresh_models(self):
        logger.info("Refreshing model configurations...")
        self._adapters.clear()
        self._configs.clear()
        self._initialize_default_models()
        logger.info(f"Model refresh complete. {len(self._adapters)} models available.")
    
    def get_available_models(self) -> List[Dict]:
        result = []
        for key, config in self._configs.items():
            adapter = self._adapters.get(key)
            result.append({
                'key': key,
                **config.to_dict(),
                'available': adapter is not None and config.enabled,
            })
        return result
    
    def get_model(self, model_key: str) -> Optional[BaseModelAdapter]:
        return self._adapters.get(model_key)
    
    def test_all_connections(self) -> Dict[str, bool]:
        results = {}
        for key, adapter in self._adapters.items():
            results[key] = adapter.test_connection()
        return results


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
