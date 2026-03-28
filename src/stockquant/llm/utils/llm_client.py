"""LLM API 客户端 - 支持多种LLM提供商"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = ""


class BaseLLMClient(ABC):
    """LLM 客户端基类"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or self._get_api_key()

    @abstractmethod
    def _get_api_key(self) -> str:
        """获取 API 密钥"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """生成文本"""
        pass

    def _validate_response(self, response: Dict) -> bool:
        """验证响应格式"""
        return True


class OpenAIClient(BaseLLMClient):
    """OpenAI API 客户端"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model, api_key)
        self.base_url = base_url or "https://api.openai.com/v1"

    def _get_api_key(self) -> str:
        """从环境变量获取 API 密钥"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """调用 OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    data = await response.json()

                    if "error" in data:
                        raise ValueError(f"API error: {data['error']}")

                    return LLMResponse(
                        content=data["choices"][0]["message"]["content"],
                        tokens_used=data.get("usage", {}).get("total_tokens", 0),
                        model=self.model,
                        finish_reason=data["choices"][0].get("finish_reason", ""),
                    )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API 客户端"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.base_url = "https://api.anthropic.com/v1"

    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """调用 Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # 提取系统消息
        system_message = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            payload["system"] = system_message

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    data = await response.json()

                    if "error" in data:
                        raise ValueError(f"API error: {data['error']}")

                    return LLMResponse(
                        content=data["content"][0]["text"],
                        tokens_used=data.get("usage", {}).get("input_tokens", 0)
                        + data.get("usage", {}).get("output_tokens", 0),
                        model=self.model,
                        finish_reason=data.get("stop_reason", ""),
                    )
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise


class DeepSeekClient(OpenAIClient):
    """DeepSeek API 客户端（兼容OpenAI格式）"""

    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )

    def _get_api_key(self) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return api_key


class KimiClient(OpenAIClient):
    """Kimi (Moonshot AI) API 客户端（兼容OpenAI格式）"""

    def __init__(self, model: str = "moonshot-v1-8k", api_key: Optional[str] = None):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )

    def _get_api_key(self) -> str:
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            raise ValueError("KIMI_API_KEY environment variable not set")
        return api_key


class MockLLMClient(BaseLLMClient):
    """模拟 LLM 客户端 - 用于测试"""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_history: List[Dict] = []

    def _get_api_key(self) -> str:
        return "mock-key"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """返回模拟响应"""
        prompt = messages[-1]["content"] if messages else ""
        self.call_history.append({"prompt": prompt, "messages": messages})

        # 根据提示词返回预设响应
        for key, response in self.responses.items():
            if key in prompt:
                return LLMResponse(content=response, tokens_used=100, model="mock")

        # 默认响应
        return LLMResponse(
            content='{"hypothesis": "mock hypothesis", "confidence": 0.8}',
            tokens_used=50,
            model="mock",
        )


def create_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMClient:
    """工厂函数创建 LLM 客户端"""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "claude": AnthropicClient,
        "deepseek": DeepSeekClient,
        "mock": MockLLMClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    client_class = providers[provider]

    if provider == "mock":
        return client_class()

    return client_class(model=model, api_key=api_key)
