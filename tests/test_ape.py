"""APE 优化器单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_evolution.core.models import PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.optimizers.ape import APEOptimizer
from prompt_evolution.providers.litellm_provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_provider() -> MagicMock:
    """创建一个 mock provider。"""
    provider = MagicMock(spec=LiteLLMProvider)
    provider.generate = AsyncMock(
        return_value=(
            "1. 你是一个专业助手，请准确回答问题。\n"
            "2. 你是一个知识丰富的AI，善于解答问题。\n"
            "3. 请作为助手，简洁明了地回答问题。"
        )
    )
    provider.count_tokens = MagicMock(return_value=100)
    provider.estimate_cost = MagicMock(return_value=0.001)
    return provider


@pytest.fixture()
def mock_evaluator() -> MagicMock:
    """创建一个 mock evaluator。"""
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate = AsyncMock(side_effect=[0.8, 0.9, 0.7])
    return evaluator


@pytest.fixture()
def ape_optimizer(mock_provider, mock_evaluator) -> APEOptimizer:
    """创建一个 APEOptimizer 实例。"""
    return APEOptimizer(
        model_provider=mock_provider,
        evaluator=mock_evaluator,
        config={"num_candidates": 3, "num_iterations": 1},
    )


@pytest.fixture()
def sample_dataset() -> list[dict]:
    return [
        {"input": "1+1=？", "target": "2"},
        {"input": "2+3=？", "target": "5"},
    ]


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestAPEOptimizer:
    """APE 优化器测试。"""

    def test_name(self, ape_optimizer):
        """测试 name 属性。"""
        assert ape_optimizer.name == "APE"

    def test_initialization(self, ape_optimizer):
        """测试初始化。"""
        assert ape_optimizer._num_candidates == 3
        assert ape_optimizer._num_iterations == 1
        assert ape_optimizer.model_provider is not None
        assert ape_optimizer.evaluator is not None

    @pytest.mark.asyncio()
    async def test_optimize(self, ape_optimizer, sample_dataset):
        """测试优化流程。"""
        initial = PromptCandidate(
            id="init", instruction="你是一个有用的助手。"
        )

        result = await ape_optimizer.optimize(
            initial_prompt=initial,
            dataset=sample_dataset,
            max_iterations=1,
        )

        assert result.best_prompt is not None
        assert result.best_prompt.score == 0.9  # mock 返回的最高分
        assert len(result.all_candidates) == 3
        assert result.num_candidates_evaluated == 3
        assert result.num_iterations == 1

    @pytest.mark.asyncio()
    async def test_generate_candidates(self, ape_optimizer, sample_dataset):
        """测试候选 prompt 生成。"""
        initial = PromptCandidate(
            id="init", instruction="你是一个有用的助手。"
        )

        candidates = await ape_optimizer._generate_candidates(
            initial_prompt=initial,
            dataset=sample_dataset,
            iteration=1,
        )

        assert len(candidates) == 3
        assert all(isinstance(c, PromptCandidate) for c in candidates)
        assert all(c.id for c in candidates)

    def test_on_iteration_start(self, ape_optimizer):
        """测试迭代开始钩子（默认无操作）。"""
        # 不应抛出异常
        ape_optimizer.on_iteration_start(1)

    def test_on_iteration_end(self, ape_optimizer):
        """测试迭代结束钩子（默认无操作）。"""
        candidates = [
            PromptCandidate(id="c1", instruction="test", score=0.8),
        ]
        # 不应抛出异常
        ape_optimizer.on_iteration_end(1, candidates)
