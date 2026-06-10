"""Evaluation metrics sub-package.

导入本模块时自动注册所有内置指标（通过 BaseMetric 的 __init_subclass__ 机制）。
"""

from __future__ import annotations

from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric  # noqa: F401
from prompt_evolution.evaluation.metrics.base import BaseMetric, register_metric  # noqa: F401
from prompt_evolution.evaluation.metrics.exact_match import ExactMatchMetric  # noqa: F401
from prompt_evolution.evaluation.metrics.f1_score import F1ScoreMetric  # noqa: F401

# 导出注册表查询函数
from prompt_evolution.evaluation.metrics.base import (  # noqa: F401
    get_metric,
    list_metrics,
)

__all__ = [
    "AccuracyMetric",
    "BaseMetric",
    "ExactMatchMetric",
    "F1ScoreMetric",
    "get_metric",
    "list_metrics",
    "register_metric",
]
