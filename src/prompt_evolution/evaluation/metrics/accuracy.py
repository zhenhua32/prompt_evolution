"""Accuracy 精确匹配指标。

如果 ``predictions[i]`` 与 ``references[i]`` 完全相同（忽略大小写和首尾空格），
计为正确，否则错误。最终返回正确率（0~1）。
"""

from __future__ import annotations

from prompt_evolution.evaluation.metrics.base import BaseMetric, register_metric


@register_metric
class AccuracyMetric(BaseMetric):
    """精确匹配准确率。"""

    name: str = "accuracy"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            1
            for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return correct / len(predictions)


# 便捷导出
__all__ = ["AccuracyMetric"]
