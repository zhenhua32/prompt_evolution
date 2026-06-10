"""Core 包 — 核心抽象层。"""

from __future__ import annotations

from .base import BaseOptimizer, BaseEvaluator, BaseModelProvider, BaseMetric
from .models import PromptCandidate, OptimizationResult

__all__ = [
    "BaseOptimizer",
    "BaseEvaluator",
    "BaseModelProvider",
    "BaseMetric",
    "PromptCandidate",
    "OptimizationResult",
]
