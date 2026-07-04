"""
DeepSeek API Client Module

Provides a robust client for interacting with the DeepSeek API,
including request封装, error handling, timeout control, and retry logic.
"""

import os
import sys
import json
import base64
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from dataclasses import dataclass

import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
    else:
        # Also try the project root
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=True)
except ImportError:
    pass  # python-dotenv not installed, rely on os.environ

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class DeepSeekError(Exception):
    """Base exception for DeepSeek API errors."""
    pass


class DeepSeekAuthenticationError(DeepSeekError):
    """Raised when API key is invalid or missing."""
    pass


class DeepSeekRateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded."""
    pass


class DeepSeekTimeoutError(DeepSeekError):
    """Raised when API request times out."""
    pass


class DeepSeekServerError(DeepSeekError):
    """Raised when DeepSeek server returns a 5xx error."""
    pass


class DeepSeekInvalidRequestError(DeepSeekError):
    """Raised when the request payload is invalid."""
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API client.

    Reads from environment variables with sensible defaults.
    """
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0
    max_tokens: int = 2048
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        """Create config from environment variables."""
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise DeepSeekAuthenticationError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY environment variable or create a .env file "
                "with DEEPSEEK_API_KEY=your_key_here"
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            timeout=int(os.environ.get("DEEPSEEK_TIMEOUT", "120")),
            max_retries=int(os.environ.get("DEEPSEEK_MAX_RETRIES", "3")),
            retry_delay=float(os.environ.get("DEEPSEEK_RETRY_DELAY", "2.0")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.7")),
        )


# =============================================================================
# API Response Model
# =============================================================================

@dataclass
class DeepSeekResponse:
    """Standardized response wrapper for DeepSeek API calls."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    usage: Optional[Dict[str, int]] = None

    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> "DeepSeekResponse":
        """Parse the raw DeepSeek API response into a standardized format."""
        try:
            choices = response.get("choices", [])
            content = ""
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")

            return cls(
                success=True,
                data=response,
                content=content,
                usage=response.get("usage"),
                status_code=200,
            )
        except Exception as e:
            return cls(
                success=False,
                error_code="PARSE_ERROR",
                error_message=f"Failed to parse API response: {str(e)}",
                status_code=200,
            )

    @classmethod
    def error(cls, error_code: str, error_message: str, status_code: Optional[int] = None) -> "DeepSeekResponse":
        """Create an error response."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=error_message,
            status_code=status_code,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "content": self.content,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "usage": self.usage,
        }


# =============================================================================
# Main API Client
# =============================================================================

class DeepseekAPIClient:
    """Robust client for the DeepSeek API with error handling and retry logic."""

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig.from_env()
        self._validate_config()
        self.session = self._create_session()

    def _validate_config(self):
        """Validate the configuration."""
        if not self.config.api_key:
            raise DeepSeekAuthenticationError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        if not self.config.base_url:
            self.config.base_url = "https://api.deepseek.com/v1"

    def _create_session(self) -> requests.Session:
        """Create a requests session with default headers and timeout."""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Sun-Analysis/1.0",
        })
        return session

    # -------------------------------------------------------------------------
    # Core Request Method with Retry Logic
    # -------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        stream: bool = False,
    ) -> DeepSeekResponse:
        """Core request method with retry, timeout, and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "chat/completions")
            data: Request payload as dictionary
            stream: Whether to use streaming mode

        Returns:
            DeepSeekResponse with parsed result or error information
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    "DeepSeek API request [%d/%d]: %s %s",
                    attempt, self.config.max_retries, method, url
                )

                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    timeout=self.config.timeout,
                    stream=stream,
                )

                # Handle HTTP status codes
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        if "error" in json_data:
                            return self._handle_api_error(
                                json_data["error"],
                                response.status_code
                            )
                        return DeepSeekResponse.from_api_response(json_data)
                    except json.JSONDecodeError as e:
                        return DeepSeekResponse.error(
                            "INVALID_JSON",
                            f"Invalid JSON response: {str(e)}",
                            response.status_code,
                        )

                elif response.status_code == 401:
                    error = DeepSeekAuthenticationError(
                        "Invalid or expired API key. Please check your DEEPSEEK_API_KEY."
                    )
                    error.status_code = response.status_code
                    raise error
                elif response.status_code == 429:
                    raise DeepSeekRateLimitError(
                        "API rate limit exceeded. Please wait and try again."
                    )
                elif response.status_code == 400:
                    error_detail = self._extract_error_detail(response)
                    error = DeepSeekInvalidRequestError(
                        f"Invalid request: {error_detail}"
                    )
                    error.status_code = response.status_code
                    raise error
                elif 500 <= response.status_code < 600:
                    raise DeepSeekServerError(
                        f"DeepSeek server error (HTTP {response.status_code}): "
                        f"{self._extract_error_detail(response)}"
                    )
                else:
                    error_detail = self._extract_error_detail(response)
                    return DeepSeekResponse.error(
                        "HTTP_ERROR",
                        f"HTTP {response.status_code}: {error_detail}",
                        response.status_code,
                    )

            except requests.exceptions.Timeout as e:
                last_error = DeepSeekTimeoutError(
                    f"Request timed out after {self.config.timeout}s "
                    f"(attempt {attempt}/{self.config.max_retries})"
                )
                logger.warning(str(last_error))

            except (DeepSeekAuthenticationError, DeepSeekInvalidRequestError) as e:
                # Non-retryable errors
                error_code = type(e).__name__
                error_code = error_code.replace("DeepSeek", "")
                error_code = error_code.replace("Error", "")
                # Insert underscore between words
                error_code = "".join(
                    f"_{c}" if c.isupper() and i > 0 else c
                    for i, c in enumerate(error_code)
                ).upper()
                return DeepSeekResponse.error(
                    error_code,
                    str(e),
                    getattr(e, "status_code", None),
                )

            except (DeepSeekRateLimitError, DeepSeekServerError, requests.exceptions.ConnectionError) as e:
                last_error = e
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt, self.config.max_retries, str(e),
                )

            except requests.exceptions.RequestException as e:
                last_error = DeepSeekError(f"Request failed: {str(e)}")
                logger.warning(str(last_error))

            # Wait before retry (with exponential backoff)
            if attempt < self.config.max_retries:
                wait_time = self.config.retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %.1f seconds...", wait_time)
                time.sleep(wait_time)

        # All retries exhausted
        error_msg = str(last_error) if last_error else "Unknown error after all retries"
        return DeepSeekResponse.error("REQUEST_FAILED", error_msg)

    def _handle_api_error(self, error: Dict, status_code: int) -> DeepSeekResponse:
        """Handle API-level error responses."""
        error_msg = error.get("message", str(error))
        error_type = error.get("type", "API_ERROR")
        return DeepSeekResponse.error(error_type, error_msg, status_code)

    def _extract_error_detail(self, response: requests.Response) -> str:
        """Extract error detail from response body."""
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                error_field = error_data.get("error", str(error_data))
                if isinstance(error_field, dict):
                    return error_field.get("message", str(error_data))
                return str(error_field)
            return str(error_data)
        except (json.JSONDecodeError, AttributeError):
            try:
                return response.text[:500]
            except Exception:
                return "Unknown error"

    # -------------------------------------------------------------------------
    # Image Encoding
    # -------------------------------------------------------------------------

    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded string of the image

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the file is too large
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        max_size_mb = float(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"Image file too large: {file_size_mb:.1f}MB "
                f"(max: {max_size_mb}MB)"
            )

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # -------------------------------------------------------------------------
    # Chat Completion with Image
    # -------------------------------------------------------------------------

    def chat_with_image(
        self,
        image_path: str,
        question: str = "请分析这张太阳磁图，判断Hale分类类型并说明理由。",
    ) -> DeepSeekResponse:
        """Send image analysis request to DeepSeek API.

        Since DeepSeek API does not support image_url input, this method
        extracts image features and sends them as text description along
        with the question.

        Args:
            image_path: Path to the image file
            question: The question/prompt for analysis

        Returns:
            DeepSeekResponse with the model's response
        """
        # Extract image features to include in the prompt
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert("L")
            arr = np.array(img)
            h, w = arr.shape
            mean_brightness = float(np.mean(arr))
            std_brightness = float(np.std(arr))
            min_brightness = float(np.min(arr))
            max_brightness = float(np.max(arr))
            contrast = (max_brightness - min_brightness) / (max_brightness + min_brightness + 1e-8)

            mid_point = 128
            dark_pixels = int(np.sum(arr < mid_point))
            bright_pixels = int(np.sum(arr >= mid_point))
            polarity_ratio = dark_pixels / max(bright_pixels, 1)

            # Edge density
            try:
                import cv2
                edges = cv2.Canny(arr, 50, 150)
                edge_density = float(np.sum(edges > 0) / (h * w))
            except ImportError:
                sobel_x = np.diff(arr, axis=1)
                sobel_y = np.diff(arr, axis=0)
                gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y[:, :-1] ** 2)
                edge_density = float(np.mean(gradient_mag > 20))

            image_description = (
                f"图像尺寸: {w}x{h} 像素\n"
                f"平均亮度: {mean_brightness:.1f} (0-255)\n"
                f"亮度标准差: {std_brightness:.1f}\n"
                f"亮度范围: {min_brightness:.0f} - {max_brightness:.0f}\n"
                f"对比度: {contrast:.3f}\n"
                f"暗像素比例: {dark_pixels/(h*w):.2%}\n"
                f"亮像素比例: {bright_pixels/(h*w):.2%}\n"
                f"明暗比例: {polarity_ratio:.2f}\n"
                f"边缘密度: {edge_density:.4f}\n"
            )

            # Combine image description with question
            full_question = f"根据以下太阳图像的特征统计信息进行分析：\n\n{image_description}\n{question}"

        except Exception as e:
            logger.warning(f"Failed to extract image features: {e}, sending question as-is")
            full_question = question

        messages = [
            {
                "role": "user",
                "content": full_question,
            }
        ]

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        return self._request("POST", "chat/completions", data=data)

    # -------------------------------------------------------------------------
    # Text Chat Completion
    # -------------------------------------------------------------------------

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
    ) -> DeepSeekResponse:
        """Send a text message to the DeepSeek chat API.

        Args:
            message: User message content
            system_prompt: Optional system prompt to set context

        Returns:
            DeepSeekResponse with the model's response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        return self._request("POST", "chat/completions", data=data)

    def test(self) -> DeepSeekResponse:
        """Test the API connectivity with a minimal request.

        Returns:
            DeepSeekResponse indicating success/failure
        """
        return self.chat("OK", system_prompt="Reply with OK only.")

    # -------------------------------------------------------------------------
    # Response Extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_response_content(response: DeepSeekResponse) -> str:
        """Extract text content from a DeepSeekResponse.

        Args:
            response: DeepSeekResponse object

        Returns:
            Extracted text content, or error description if failed
        """
        if not response.success:
            return f"[Error] {response.error_code}: {response.error_message}"
        return response.content or ""

    @staticmethod
    def extract_response_text(response_dict: Dict[str, Any]) -> str:
        """Legacy: Extract text content from a raw API response dictionary.

        Args:
            response_dict: Raw API response dictionary

        Returns:
            Extracted text content
        """
        try:
            if "error" in response_dict:
                return f"Error: {response_dict['error']}"
            choices = response_dict.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
            return str(response_dict)
        except Exception as e:
            return str(response_dict)


# =============================================================================
# High-Level Solar Analyzer
# =============================================================================

class DeepseekSolarAnalyzer:
    """High-level analyzer that uses DeepSeek API for solar image analysis."""

    SYSTEM_PROMPT = """
你是一位专业的太阳物理学家，请分析以下太阳图像。

请按照以下结构进行分析：

1. 黑子区域分析
   - 黑子群编号（如NOAA编号）
   - 位置坐标
   - 大小和形态描述
   - 磁场类型（alpha/beta/gamma/beta-gamma等）

2. 亮区特征分析
   - 耀斑活动（类型、强度、位置）
   - 日珥/暗条特征
   - 谱斑区域

3. 磁场特征分析
   - 磁场强度评估
   - 极性分布
   - 复杂性判断

4. 活动风险评估
   - 风险等级（低/中/高/极高）
   - 未来24-48小时活动预测
   - 对地影响评估

5. 总结与建议
   - 简明摘要
   - 监测建议

请使用专业但易懂的语言进行描述，输出格式为结构化文本。
"""

    def __init__(self, api_client: DeepseekAPIClient):
        self.client = api_client

    def analyze_image(
        self,
        image_path: str,
        additional_context: str = "",
    ) -> DeepSeekResponse:
        """Analyze a solar image using the DeepSeek API.

        Args:
            image_path: Path to the solar image
            additional_context: Optional additional context for analysis

        Returns:
            DeepSeekResponse with analysis results
        """
        full_prompt = self.SYSTEM_PROMPT
        if additional_context:
            full_prompt += f"\n\n额外信息: {additional_context}"

        return self.client.chat_with_image(image_path, full_prompt)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the DeepSeek client."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek Solar Analysis Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image sun.jpg
  %(prog)s --image sun.jpg --output result.json
  %(prog)s --question "Describe this image"
        """,
    )
    parser.add_argument("--api-key", type=str, help="DeepSeek API key (overrides env)")
    parser.add_argument("--image", type=str, help="Image path to analyze")
    parser.add_argument(
        "--question",
        type=str,
        default="请分析这张太阳磁图，判断Hale分类类型并说明理由。",
        help="Question/prompt for analysis",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Override API key if provided via CLI
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key

    try:
        config = DeepSeekConfig.from_env()
        client = DeepseekAPIClient(config=config)
    except DeepSeekAuthenticationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.image:
        print(f"Analyzing image: {args.image}")
        result = client.chat_with_image(args.image, args.question)

        if result.success:
            content = client.extract_response_content(result)
            print("\nAnalysis Result:")
            print("=" * 60)
            print(content)
            print("=" * 60)

            if result.usage:
                print(f"\nToken Usage: {result.usage}")

            if args.output:
                output_data = {
                    "success": True,
                    "content": content,
                    "usage": result.usage,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {args.output}")
        else:
            print(f"Analysis failed: [{result.error_code}] {result.error_message}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Please specify --image option")
        parser.print_help()


if __name__ == "__main__":
    main()
