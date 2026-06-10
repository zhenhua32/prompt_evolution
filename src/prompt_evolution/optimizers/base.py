"""Optimizer 抽象基类。

所有优化算法（APE / OPRO / EvoPrompt / …）均继承此类，
保证统一调用方式：``optimize() -> OptimizationResult``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from prompt_evolution.core.base import BaseOptimizer as _BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate

# 重新导出，保持导入路径整洁
__all__ = ["BaseOptimizer", "PromptCandidate", "OptimizationResult"]


class BaseOptimizer(_BaseOptimizer, ABC):
    """所有优化器的抽象基类（已在 ``core/base.py`` 中定义，此处仅为导入便捷）。"""

    # 该类完全继承 ``prompt_evolution.core.base.BaseOptimizer``，
    # 此处不做任何额外定义，仅作为 ``optimizers.base`` 命名空间的锚点。
    pass
