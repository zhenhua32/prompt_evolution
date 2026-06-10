"""基于 LiteLLM 的统一模型提供商实现。

LiteLLM 支持 100+ 模型，统一为 OpenAI 兼容格式调用。
模型标识格式：``"openai/gpt-4o"``, ``"anthropic/claude-3-5-sonnet"`` 等。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import litellm
from loguru import logger

from prompt_evolution.core.base import BaseModelProvider


class LiteLLMProvider(BaseModelProvider):
    """基于 LiteLLM 的统一模型提供商。

    支持的模型格式：
        - ``"openai/gpt-4o"``
        - ``"anthropic/claude-3-5-sonnet"``
        - ``"gemini/gemini-pro"``
        - ``"deepseek/deepseek-chat"``
        - ``"ollama/llama3"``
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        # 可选：设置 LiteLLM 的全局配置
        if api_key:
            litellm.api_key = api_key
        if api_base:
            litellm.api_base = api_base
        # 费用追踪
        self._total_cost: float = 0.0

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    # ------------------------------------------------------------------
    # BaseModelProvider 接口实现
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """调用 LLM 生成文本，返回纯文本响应。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            text = response.choices[0].message.content or ""
            # 追踪费用
            if hasattr(response, "usage") and response.usage:
                self._total_cost += getattr(response.usage, "cost", 0.0) or 0.0
            return text
        except Exception as exc:
            logger.error("LiteLLM call failed: model={} err={}", self._model, exc)
            raise

    async def generate_with_logprobs(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """生成文本并返回 logprobs。"""
        messages = [{"role": "user", "content": prompt}]
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            logprobs=True,
            top_logprobs=kwargs.get("top_logprobs", 5),
            **{k: v for k, v in kwargs.items() if k != "top_logprobs"},
        )
        return {
            "text": response.choices[0].message.content,
            "logprobs": getattr(response.choices[0], "logprobs", None),
            "usage": response.usage,
        }

    def count_tokens(self, text: str) -> int:
        """使用 tiktoken 计算 token 数（LiteLLM 内部也用此方式）。"""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # 兜底：粗略估算（1 token ≈ 4 字符）
            return len(text) // 4

    def estimate_cost(self, prompt: str, completion: str) -> float:
        """使用 LiteLLM 的 cost_per_token 工具估算费用。"""
        try:
            prompt_tokens = self.count_tokens(prompt)
            completion_tokens = self.count_tokens(completion)
            cost = litellm.cost_per_token(
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return cost or 0.0
        except Exception:
            return 0.0
