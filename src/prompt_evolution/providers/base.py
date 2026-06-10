"""Model Provider 抽象基类。

如需自定义模型接入，请继承 ``BaseModelProvider``。
推荐使用 ``LiteLLMProvider``（见 ``litellm_provider.py``）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from prompt_evolution.core.base import BaseModelProvider


class BaseModelProvider(ABC):
    """所有模型提供商的抽象基类。

    统一屏蔽 OpenAI / Anthropic / Gemini / Ollama 等差异。
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """调用 LLM 生成文本，返回纯文本响应。"""
        ...

    @abstractmethod
    async def generate_with_logprobs(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """生成文本并返回 logprobs（部分优化算法需要）。"""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数。"""
        ...

    @abstractmethod
    def estimate_cost(self, prompt: str, completion: str) -> float:
        """估算一次调用的费用（美元）。"""
        ...
