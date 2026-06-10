"""Pytest 全局配置。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Generator

import pytest

# 确保 src/ 在 sys.path 中，使 ``import prompt_evolution`` 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Async 支持（pytest-asyncio）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """为 session 级 fixture 提供事件循环。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# 通用 fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_dataset() -> list[dict]:
    """返回一个小数据集，用于测试。"""
    return [
        {"input": "法国的首都是哪里？", "target": "巴黎"},
        {"input": "德国首都是哪里？", "target": "柏林"},
        {"input": "日本首都是哪里？", "target": "东京"},
    ]


@pytest.fixture()
def sample_prompt_candidate():
    """返回一个 PromptCandidate 实例。"""
    from prompt_evolution.core.models import PromptCandidate

    return PromptCandidate(
        id="test-001",
        instruction="你是一个有用的助手。请回答问题。",
        score=0.0,
    )
