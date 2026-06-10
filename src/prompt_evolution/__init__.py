"""Prompt 迭代神器 - 一站式 Prompt 自动优化工具."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "prompt-evolution contributors"

from .core.models import PromptCandidate, OptimizationResult
from .core.base import BaseOptimizer, BaseEvaluator, BaseModelProvider

__all__ = [
    "PromptCandidate",
    "OptimizationResult",
    "BaseOptimizer",
    "BaseEvaluator",
    "BaseModelProvider",
]
