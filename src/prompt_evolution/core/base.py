"""核心抽象基类 — 所有优化器 / 模型提供商 / 评估器均继承此处."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import OptimizationResult, PromptCandidate


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


class BaseMetric(ABC):
    """评估指标的抽象基类。"""

    name: str = "base_metric"

    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """计算指标分数，范围由具体指标定义（通常 0~1）。"""
        ...


class BaseEvaluator(ABC):
    """评估器的抽象基类。"""

    @abstractmethod
    async def evaluate(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        model_provider: BaseModelProvider,
    ) -> float:
        """评估一个 prompt 在给定数据集上的表现，返回 0~1 的分数。"""
        ...

    @abstractmethod
    def evaluate_batch(
        self,
        prompts: List[PromptCandidate],
        dataset: List[Dict[str, Any]],
        model_provider: BaseModelProvider,
        parallel: int = 5,
    ) -> List[float]:
        """批量评估多个 prompt（并发）。"""
        ...


class BaseOptimizer(ABC):
    """所有优化器的抽象基类。

    所有优化算法（APE / OPRO / EvoPrompt / …）均实现此接口，
    保证统一调用方式：``optimize() -> OptimizationResult``。
    """

    def __init__(
        self,
        model_provider: BaseModelProvider,
        evaluator: BaseEvaluator,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_provider = model_provider
        self.evaluator = evaluator
        self.config = config or {}
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    @abstractmethod
    async def optimize(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> OptimizationResult:
        """执行优化，返回完整结果。"""
        ...

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """优化器名称，用于 UI 展示和日志。"""
        ...

    # ------------------------------------------------------------------
    # 钩子（可选覆盖）
    # ------------------------------------------------------------------

    def on_iteration_start(self, iteration: int, **kwargs: Any) -> None:
        """每轮迭代开始时的回调（用于日志 / 进度条）。"""
        ...

    def on_iteration_end(
        self,
        iteration: int,
        candidates: List[PromptCandidate],
        **kwargs: Any,
    ) -> None:
        """每轮迭代结束时的回调。"""
        ...
