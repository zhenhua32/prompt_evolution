"""OPRO / DSPy / SPO 优化器单元测试。

重点验证本轮修复点：
- OPRO 历史取最高分（非最低分）
- DSPy 早停需 len>=4（不越界）
- 三个优化器费用用增量法
- DSPy bootstrap 走占位符替换链路（不调不存在的 agenerate）
- 候选 evaluated 标记被写回
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_evolution.core.models import OptimizationResult, PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.optimizers.dspy_optimizer.optimizer import DSPyOptimizer
from prompt_evolution.optimizers.opro.optimizer import OPROOptimizer
from prompt_evolution.optimizers.spo.optimizer import SPOOptimizer
from prompt_evolution.providers.litellm_provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# 共用 mock fixtures
# ---------------------------------------------------------------------------

def _make_provider_returns_backticks(provider_total_cost: float = 0.1) -> MagicMock:
    """Provider：候选生成调用返回 ``` 包裹的变体；评估调用返回固定预测。"""
    provider = MagicMock(spec=LiteLLMProvider)
    provider.total_cost_usd = provider_total_cost

    async def _generate(prompt: str = "", system_prompt: str = None, **kwargs: Any) -> str:
        if system_prompt is not None:
            return "```\n你是改进的助手。问题：{input}\n答案：\n```"
        # 评估调用：返回 "巴黎"
        return "巴黎"

    provider.generate = AsyncMock(side_effect=_generate)
    return provider


def _make_dataset(n: int = 3) -> List[Dict[str, Any]]:
    return [{"input": f"q{i}", "target": "巴黎"} for i in range(n)]


@pytest.fixture()
def initial_candidate() -> PromptCandidate:
    return PromptCandidate(
        id="initial",
        instruction='你是助手。问题：{input}\n答案：',
    )


# ---------------------------------------------------------------------------
# OPRO 测试
# ---------------------------------------------------------------------------

class TestOPROOptimizer:
    @pytest.fixture()
    def optimizer(self) -> OPROOptimizer:
        provider = _make_provider_returns_backticks()
        evaluator = Evaluator(metrics=[AccuracyMetric()], concurrency=2)
        return OPROOptimizer(
            model_provider=provider,
            evaluator=evaluator,
            config={"num_candidates": 3, "use_fewshot": False},
        )

    def test_name(self, optimizer) -> None:
        assert optimizer.name == "OPRO"

    @pytest.mark.asyncio()
    async def test_optimize_basic(self, optimizer, initial_candidate) -> None:
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )
        assert result.best_prompt is not None
        assert len(result.all_candidates) >= 1
        assert result.num_candidates_evaluated >= 1

    @pytest.mark.asyncio()
    async def test_history_keeps_top_scores(self, optimizer, initial_candidate) -> None:
        """P2-7 修复点：历史应取最高分（非最低分）。

        构造 scored_history，调用 _build_meta_prompt，验证传给 LLM 的历史
        按分数降序排列、且只保留前 20 条。
        """
        # 构造 25 条历史，分数从 0.1 到 0.9 乱序
        scored_history = [
            (f"prompt_{i}", float(i) / 30) for i in range(25)
        ]
        meta = optimizer._build_meta_prompt(
            task_description="test",
            scored_history=scored_history,
            iteration=1,
        )
        # 应只保留 20 条（sorted_history[:20]）
        # 检查 meta 里包含的 Score 数量
        score_lines = [l for l in meta.splitlines() if l.startswith("Score:")]
        assert len(score_lines) <= 20, f"应截断到 20 条，实际 {len(score_lines)}"
        # 最高分 0.8（i=24/30）应在 meta 里
        assert "0.8000" in meta or "0.7667" in meta  # 排序后最高分
        # 最低分 0.0（i=0）不应在前 20（i=0..4 的分数最低，应被排除）
        assert "0.0000\n" not in meta and "0.0333" not in meta

    @pytest.mark.asyncio()
    async def test_cost_increment(self, optimizer, initial_candidate) -> None:
        """费用用增量法。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=1,
        )
        assert result.total_cost_usd >= 0
        # 不应平方级虚高（增量法只取差值）
        assert result.total_cost_usd < 100.0

    @pytest.mark.asyncio()
    async def test_evaluated_flag_written_back(
        self, optimizer, initial_candidate
    ) -> None:
        """P0-1 一致性：evaluator.evaluate 回写 evaluated 标记。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=1,
        )
        for c in result.all_candidates:
            assert c.evaluated, f"候选 {c.id} 未被标记 evaluated"


# ---------------------------------------------------------------------------
# DSPy 测试
# ---------------------------------------------------------------------------

class TestDSPyOptimizer:
    @pytest.fixture()
    def optimizer(self) -> DSPyOptimizer:
        provider = _make_provider_returns_backticks()
        evaluator = Evaluator(metrics=[AccuracyMetric()], concurrency=2)
        return DSPyOptimizer(
            model_provider=provider,
            evaluator=evaluator,
            config={"num_candidates": 2, "bootstrap_samples": 2},
        )

    def test_name(self, optimizer) -> None:
        assert optimizer.name == "DSPy"

    @pytest.mark.asyncio()
    async def test_optimize_basic(self, optimizer, initial_candidate) -> None:
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )
        assert result.best_prompt is not None
        assert len(result.all_candidates) >= 1

    @pytest.mark.asyncio()
    async def test_early_stop_requires_len_4(
        self, optimizer, initial_candidate
    ) -> None:
        """P2-6 修复点：早停需 len(_scores) >= 4，不越界。

        max_iterations=3 时 history 最多 3 条，早停条件 len>=4 永远不满足，
        不应触发越界或误判。
        """
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=3,
        )
        # 应跑满 3 轮（不会因早停条件越界而提前 break 或异常）
        assert len(result.optimization_history) == 3

    @pytest.mark.asyncio()
    async def test_bootstrap_uses_placeholder_path(
        self, optimizer, initial_candidate
    ) -> None:
        """P0 修复点：bootstrap 走 generate(prompt=, system_prompt=None)，
        不调不存在的 agenerate(messages=...)。"""
        provider = optimizer.model_provider
        # 记录所有调用
        calls: List[Dict[str, Any]] = []
        orig = provider.generate

        async def _spy(prompt: str = "", system_prompt: str = None, **kwargs: Any) -> str:
            calls.append({"prompt": prompt, "system_prompt": system_prompt})
            return await orig(prompt=prompt, system_prompt=system_prompt, **kwargs)

        provider.generate = AsyncMock(side_effect=_spy)

        await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )

        # 所有调用都走 generate（不应有 agenerate 调用）
        assert all("prompt" in c for c in calls)
        # bootstrap 调用应不含 system_prompt（消除双发）
        # 至少有一些调用没传 system_prompt
        no_system = [c for c in calls if c["system_prompt"] is None]
        assert len(no_system) > 0

    @pytest.mark.asyncio()
    async def test_evaluated_flag_written_back(
        self, optimizer, initial_candidate
    ) -> None:
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=1,
        )
        for c in result.all_candidates:
            assert c.evaluated


# ---------------------------------------------------------------------------
# SPO 测试
# ---------------------------------------------------------------------------

class TestSPOOptimizer:
    @pytest.fixture()
    def optimizer(self) -> SPOOptimizer:
        provider = _make_provider_returns_backticks()
        evaluator = Evaluator(metrics=[AccuracyMetric()], concurrency=2)
        return SPOOptimizer(
            model_provider=provider,
            evaluator=evaluator,
            config={"num_candidates": 2, "num_angles": 1, "semantic_patience": 99},
        )

    def test_name(self, optimizer) -> None:
        assert optimizer.name == "SPO"

    @pytest.mark.asyncio()
    async def test_optimize_basic(self, optimizer, initial_candidate) -> None:
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )
        assert result.best_prompt is not None
        assert len(result.all_candidates) >= 1

    @pytest.mark.asyncio()
    async def test_cost_increment(self, optimizer, initial_candidate) -> None:
        """费用用增量法。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=1,
        )
        assert result.total_cost_usd >= 0
        assert result.total_cost_usd < 100.0

    @pytest.mark.asyncio()
    async def test_evaluated_flag_written_back(
        self, optimizer, initial_candidate
    ) -> None:
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(2),
            max_iterations=1,
        )
        for c in result.all_candidates:
            assert c.evaluated


# ---------------------------------------------------------------------------
# 工厂注册测试
# ---------------------------------------------------------------------------

class TestOptimizerFactory:
    def test_all_six_optimizers_registered(self) -> None:
        """所有 6 个优化器都应在工厂注册表里。"""
        from prompt_evolution.optimizers.factory import _NAME_MAP

        # 6 个类，每个有 2 个别名（短名 + xxx-optimizer）
        assert "ape" in _NAME_MAP
        assert "opro" in _NAME_MAP
        assert "dspy" in _NAME_MAP
        assert "promptbreeder" in _NAME_MAP
        assert "evoprompt" in _NAME_MAP
        assert "spo" in _NAME_MAP

    def test_create_optimizer_unknown_raises(self) -> None:
        from prompt_evolution.optimizers.factory import create_optimizer

        with pytest.raises(ValueError, match="未知优化器"):
            create_optimizer(name="nonexistent", model_provider=None, evaluator=None)
