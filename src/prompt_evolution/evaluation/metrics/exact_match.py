"""ExactMatch 严格精确匹配指标。

与 Accuracy 不同，此指标不做任何归一化（大小写敏感、空格敏感）。
只有 ``predictions[i]`` 与 ``references[i]`` **完全一致** 才计为正确。
"""

from __future__ import annotations

from prompt_evolution.evaluation.metrics.base import BaseMetric, register_metric


@register_metric
class ExactMatchMetric(BaseMetric):
    """严格精确匹配（大小写敏感，不做任何归一化）。"""

    name: str = "exact_match"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            1 for pred, ref in zip(predictions, references) if pred == ref
        )
        return correct / len(predictions)


# 便捷导出
__all__ = ["ExactMatchMetric"]
