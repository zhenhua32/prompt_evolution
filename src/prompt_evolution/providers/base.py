"""Model Provider 抽象基类。

如需自定义模型接入，请继承 ``BaseModelProvider``。
推荐使用 ``LiteLLMProvider``（见 ``litellm_provider.py``）。
"""

from __future__ import annotations

# 重导出 core.base 中的 BaseModelProvider，保持单一权威定义。
# 旧实现在此文件平行定义了同名类，导致两个不同抽象类并存：
# LiteLLMProvider 继承的是 core.base 版本，而 providers/__init__.py
# 导出的是本文件版本 —— 二者类型不兼容，会让外部子类化时困惑。
from prompt_evolution.core.base import BaseModelProvider

__all__ = ["BaseModelProvider"]
