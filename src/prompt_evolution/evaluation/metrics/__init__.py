"""Evaluation metrics sub-package.

导入本模块时通过 ``@register_metric`` 装饰器把所有内置指标注册到
``_REGISTRY``（见 ``base.py``）。新增指标时务必给类加上该装饰器，
否则 ``get_metric`` / ``list_metrics`` 将无法发现它。
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
