"""Evaluation 包的基础定义 — 从 core 重新导出。"""

from __future__ import annotations

from prompt_evolution.core.base import BaseEvaluator, BaseMetric

# 重新导出，使 ``from prompt_evolution.evaluation.base import BaseEvaluator`` 可用
__all__ = ["BaseEvaluator", "BaseMetric"]
