"""基于 LiteLLM 的统一模型提供商实现。

LiteLLM 支持 100+ 模型，统一为 OpenAI 兼容格式调用。
模型标识格式：``"openai/gpt-4o"``, ``"anthropic/claude-3-5-sonnet"`` 等。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import litellm
from loguru import logger

from prompt_evolution.core.base import BaseModelProvider


def _has_known_provider(model: str) -> bool:
    """判断 LiteLLM 是否能直接识别模型 provider。"""
    try:
        litellm.get_llm_provider(model=model)
        return True
    except Exception:
        return False


def _should_use_openai_compatible_base(model: str) -> bool:
    """判断模型是否应继承 OpenAI 兼容 base_url 语义。"""
    if model.startswith("openai/"):
        return True
    return not _has_known_provider(model)


def _normalize_model(model: str, api_base: Optional[str]) -> str:
    """为 OpenAI 兼容接口补齐 LiteLLM 需要的 provider 前缀。"""
    if not api_base or _has_known_provider(model):
        return model
    return f"openai/{model}"


def _resolve_api_base(model: str, api_base: Optional[str] = None) -> Optional[str]:
    """按优先级解析 base_url：
    1. 显式传入的 api_base（最高优先级）
    2. 环境变量 OPENAI_BASE_URL（当模型使用 OpenAI 兼容接口）
    3. 环境变量 LITELLM_API_BASE（通用）
    4. 返回 None（使用 LiteLLM 默认）
    """
    if api_base:
        return api_base
    if _should_use_openai_compatible_base(model) and os.environ.get("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"]
    if os.environ.get("LITELLM_API_BASE"):
        return os.environ["LITELLM_API_BASE"]
    return None


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
        disable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        self._api_base = _resolve_api_base(model, api_base)
        self._model = _normalize_model(model, self._api_base)
        self._api_key = api_key or self._resolve_api_key(self._model)
        self._disable_thinking = disable_thinking
        # 费用追踪
        self._total_cost: float = 0.0
        think_hint = "（think 已关闭）" if disable_thinking else ""
        print(f"✅ LiteLLMProvider 初始化成功，model={self._model}, api_base={self._api_base}, api_key={'***' if self._api_key else None} {think_hint}")

    @staticmethod
    def _resolve_api_key(model: str) -> Optional[str]:
        """按模型前缀从环境变量读取对应的 API Key。"""
        env_map = {
            "openai/": "OPENAI_API_KEY",
            "anthropic/": "ANTHROPIC_API_KEY",
            "gemini/": "GEMINI_API_KEY",
            "deepseek/": "DEEPSEEK_API_KEY",
        }
        for prefix, env_var in env_map.items():
            if model.startswith(prefix):
                return os.environ.get(env_var) or None
        return os.environ.get("LITELLM_API_KEY") or None

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

        call_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._api_key:
            call_kwargs["api_key"] = self._api_key
        if self._api_base:
            call_kwargs["api_base"] = self._api_base
        if self._disable_thinking:
            call_kwargs["thinking"] = {"type": "disabled"}

        try:
            response = await litellm.acompletion(**call_kwargs, **kwargs)
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

        call_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": kwargs.get("top_logprobs", 5),
        }
        if self._api_key:
            call_kwargs["api_key"] = self._api_key
        if self._api_base:
            call_kwargs["api_base"] = self._api_base
        if self._disable_thinking:
            call_kwargs["thinking"] = {"type": "disabled"}

        # 移除已处理的参数
        for k in ["top_logprobs"]:
            kwargs.pop(k, None)

        response = await litellm.acompletion(**call_kwargs, **kwargs)
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
