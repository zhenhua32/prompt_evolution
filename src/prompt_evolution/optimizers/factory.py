"""Optimizer 工厂 — 按名称创建优化器实例。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from prompt_evolution.core.base import BaseOptimizer, BaseEvaluator, BaseModelProvider
from prompt_evolution.optimizers.ape import APEOptimizer

# 名称 → 优化器类的映射
_NAME_MAP: dict[str, type[BaseOptimizer]] = {
    "ape": APEOptimizer,
    "ape-optimizer": APEOptimizer,
}


def list_optimizers() -> list[str]:
    """列出所有已注册的优化器名称。"""
    return list(_NAME_MAP.keys())


def register_optimizer(name: str, cls: type[BaseOptimizer]) -> None:
    """注册自定义优化器。"""
    _NAME_MAP[name.lower()] = cls


def create_optimizer(
    name: str,
    model_provider: BaseModelProvider,
    evaluator: BaseEvaluator,
    config: Optional[Dict[str, Any]] = None,
) -> BaseOptimizer:
    """按名称创建优化器实例。

    Args:
        name: 优化器名称，支持 ``"ape"`` 等。
        model_provider: 模型提供商实例。
        evaluator: 评估器实例。
        config: 算法特定配置字典。

    Returns:
        对应的优化器实例。

    Raises:
        ValueError: 未知优化器名称。
    """
    key = name.lower().strip()
    cls = _NAME_MAP.get(key)
    if cls is None:
        available = ", ".join(_NAME_MAP.keys()) or "（无）"
        raise ValueError(
            f"未知优化器: '{name}'。可用优化器: {available}"
        )
    return cls(
        model_provider=model_provider,
        evaluator=evaluator,
        config=config,
    )
