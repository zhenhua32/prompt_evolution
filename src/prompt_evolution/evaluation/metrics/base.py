"""评估指标抽象基类与注册表。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseMetric(ABC):
    """评估指标的抽象基类。

    每个指标接收 ``predictions`` 和 ``references`` 两个列表，
    返回一个 float 分数（具体含义由指标定义，通常为 0~1）。
    """

    name: str = "base_metric"

    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """计算指标分数。"""
        ...


# ---------------------------------------------------------------------------
# 指标注册表（简单字典，便于按名称查找）
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(cls: type[BaseMetric]) -> type[BaseMetric]:
    """类装饰器 —— 将指标注册到全局注册表。"""
    _REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> type[BaseMetric] | None:
    """按名称获取指标类。"""
    return _REGISTRY.get(name)


def list_metrics() -> list[str]:
    """列出所有已注册的指标名称。"""
    return list(_REGISTRY.keys())
