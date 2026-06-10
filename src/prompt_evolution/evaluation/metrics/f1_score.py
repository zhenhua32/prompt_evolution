"""F1 Score 指标（token 级别）。

对每个样本计算 token 级别的 Precision / Recall / F1，
最终返回所有样本 F1 的宏平均（macro-average）。

token 化策略：按空白字符切分（简单但有效，与 SQuAD 评测一致）。
"""

from __future__ import annotations

import re
from collections import Counter

from prompt_evolution.evaluation.metrics.base import BaseMetric, register_metric


def _tokenize(text: str) -> list[str]:
    """简单 token 化：小写 + 按空白/标点切分。"""
    return re.findall(r"\w+", text.lower())


@register_metric
class F1ScoreMetric(BaseMetric):
    """Token 级别 F1 Score（宏平均）。"""

    name: str = "f1_score"

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0

        f1_scores: list[float] = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(_tokenize(pred))
            ref_tokens = set(_tokenize(ref))

            if not ref_tokens:
                f1_scores.append(1.0 if not pred_tokens else 0.0)
                continue
            if not pred_tokens:
                f1_scores.append(0.0)
                continue

            overlap = pred_tokens & ref_tokens
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(ref_tokens)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)


# 便捷导出
__all__ = ["F1ScoreMetric"]
