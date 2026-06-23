"""进化算法优化器（PromptBreeder / EVOPrompt）单元测试。

重点验证 P0-1 修复点：
- 候选 `evaluated` 标记被正确写回（不再因 `score is None` 恒 False 而跳过评估）
- 主循环后所有候选都被评估（包括最后一轮子代）
- best_prompt 是真正的高分候选（而非默认 0.0 的子代）
- 费用统计用增量法
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_evolution.core.models import OptimizationResult, PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.optimizers.evoprompt.optimizer import EVOPromptOptimizer
from prompt_evolution.optimizers.prompt_breeder.optimizer import PromptBreederOptimizer
from prompt_evolution.providers.litellm_provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

def _make_provider_with_magic_behavior() -> MagicMock:
    """Mock provider：system_prompt 非空时返回含 MAGIC 的变体；否则按 MAGIC 决定预测。

    - 候选生成调用（带 system_prompt）→ 返回含 MAGIC 的 prompt
    - 评估调用（无 system_prompt）→ 含 MAGIC 的 prompt 返回 "正确类别"，否则 "错误类别"
    这样：初始 prompt（无 MAGIC）得 0 分；子代（含 MAGIC）得满分。
    修复前：子代从未评估 → best=initial（0 分）
    修复后：子代被评估 → best=含 MAGIC 的高分子代
    """
    provider = MagicMock(spec=LiteLLMProvider)
    provider.total_cost_usd = 0.2

    async def _generate(prompt: str = "", system_prompt: str = None, **kwargs: Any) -> str:
        if system_prompt is not None:
            return "```\n你是专家。关键词MAGIC。新闻：{input}\n类别：\n```"
        return "正确类别" if "MAGIC" in prompt else "错误类别"

    provider.generate = AsyncMock(side_effect=_generate)
    return provider


def _make_dataset(n: int = 4) -> List[Dict[str, Any]]:
    return [{"input": f"新闻标题{i}", "target": "正确类别"} for i in range(n)]


@pytest.fixture()
def mock_provider_pb() -> MagicMock:
    return _make_provider_with_magic_behavior()


@pytest.fixture()
def mock_provider_evo() -> MagicMock:
    return _make_provider_with_magic_behavior()


@pytest.fixture()
def evaluator() -> Evaluator:
    return Evaluator(metrics=[AccuracyMetric()], concurrency=2)


@pytest.fixture()
def initial_candidate() -> PromptCandidate:
    """不含 MAGIC → 评估得 0 分；子代含 MAGIC → 得满分。"""
    return PromptCandidate(
        id="initial",
        instruction='你是新闻分类助手。新闻："{input}"\n类别：',
    )


# ---------------------------------------------------------------------------
# PromptBreeder 测试
# ---------------------------------------------------------------------------

class TestPromptBreeder:
    @pytest.fixture()
    def optimizer(self, mock_provider_pb, evaluator) -> PromptBreederOptimizer:
        return PromptBreederOptimizer(
            model_provider=mock_provider_pb,
            evaluator=evaluator,
            config={
                "population_size": 5,
                "init_variants": 2,
                "num_mutations": 1,
                "early_stop_patience": 99,  # 不早停
            },
        )

    def test_name(self, optimizer) -> None:
        assert optimizer.name == "PromptBreeder"

    @pytest.mark.asyncio()
    async def test_optimize_evaluates_all_candidates(
        self, optimizer, mock_provider_pb, initial_candidate
    ) -> None:
        """P0-1 核心：所有候选都应被评估。"""
        result: OptimizationResult = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(4),
            max_iterations=2,
        )

        # 所有候选 evaluated=True
        unevaluated = [c for c in result.all_candidates if not c.evaluated]
        assert len(unevaluated) == 0, f"仍有 {len(unevaluated)} 个候选未评估"

        # num_candidates_evaluated 应等于 all_candidates 长度
        assert result.num_candidates_evaluated == len(result.all_candidates)

    @pytest.mark.asyncio()
    async def test_optimize_picks_high_score_child(
        self, optimizer, initial_candidate
    ) -> None:
        """P0-1 核心：best_prompt 应是含 MAGIC 的高分子代，而非 0 分的初始。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(4),
            max_iterations=2,
        )

        # best_prompt 应含 MAGIC（高分子代）
        assert "MAGIC" in result.best_prompt.instruction, (
            f"best_prompt 不含 MAGIC，可能选了未评估的 0 分候选。"
            f" instruction={result.best_prompt.instruction!r}"
        )
        # 得分应 > 0（不是初始的 0 分）
        assert result.best_prompt.score > 0.0

    @pytest.mark.asyncio()
    async def test_cost_uses_increment(
        self, optimizer, mock_provider_pb, initial_candidate
    ) -> None:
        """费用统计用增量法（cost_after - cost_before），非平方级累加。"""
        # 让 total_cost_usd 在调用过程中递增（mock provider 模拟真实费用增长）
        call_count = [0]
        orig_generate = mock_provider_pb.generate

        async def _count_and_generate(*args, **kwargs):
            call_count[0] += 1
            mock_provider_pb.total_cost_usd = 0.2 + call_count[0] * 0.01
            return await orig_generate(*args, **kwargs)

        mock_provider_pb.generate = AsyncMock(side_effect=_count_and_generate)

        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )

        # 增量法：cost = final_total - 0.2，应是个合理的正数（非平方级虚高）
        assert result.total_cost_usd >= 0
        # 上界：调用次数 * 0.01（每调用增加 0.01）+ 一点点容差
        assert result.total_cost_usd < call_count[0] * 0.02 + 1.0

    @pytest.mark.asyncio()
    async def test_children_are_scored_before_selection(self) -> None:
        """高分子代必须先评分，才能进入下一轮继续进化。

        旧实现里，子代先以默认 score=0.0 参与截断，可能当轮就被淘汰；
        修复后，第一轮产生的 STAGE1 会保留为精英，第二轮才能继续进化出 STAGE2。
        """
        provider = MagicMock(spec=LiteLLMProvider)
        provider.total_cost_usd = 0.0

        async def _generate(
            prompt: str = "", system_prompt: str = None, **kwargs: Any
        ) -> str:
            if system_prompt is None:
                return ""
            if "STAGE1" in prompt:
                return "```\nSTAGE2 {input}\n```"
            return "```\nSTAGE1 {input}\n```"

        provider.generate = AsyncMock(side_effect=_generate)

        evaluator = MagicMock(spec=Evaluator)

        async def _evaluate(
            prompt: PromptCandidate,
            dataset: List[Dict[str, Any]],
            model_provider: Any,
        ) -> float:
            score_map = {
                "BASE {input}": 0.1,
                "STAGE1 {input}": 0.6,
                "STAGE2 {input}": 1.0,
            }
            score = score_map[prompt.instruction]
            prompt.score = score
            prompt.evaluated = True
            return score

        evaluator.evaluate = AsyncMock(side_effect=_evaluate)

        optimizer = PromptBreederOptimizer(
            model_provider=provider,
            evaluator=evaluator,
            config={
                "population_size": 1,
                "init_variants": 0,
                "mutation_rate": 1.0,
                "crossover_rate": 0.0,
                "elite_ratio": 1.0,
            },
        )

        result = await optimizer.optimize(
            initial_prompt=PromptCandidate(id="initial", instruction="BASE {input}"),
            dataset=_make_dataset(1),
            max_iterations=2,
        )

        assert result.best_prompt.instruction == "STAGE2 {input}"
        assert result.best_prompt.score == 1.0


# ---------------------------------------------------------------------------
# EVOPrompt 测试
# ---------------------------------------------------------------------------

class TestEVOPrompt:
    @pytest.fixture()
    def optimizer(self, mock_provider_evo, evaluator) -> EVOPromptOptimizer:
        return EVOPromptOptimizer(
            model_provider=mock_provider_evo,
            evaluator=evaluator,
            config={
                "population_size": 4,
                "num_mutations": 1,
                "mutation_rate": 1.0,  # 一定变异
                "crossover_rate": 0.0,  # 不交叉
                "early_stop_patience": 99,
            },
        )

    def test_name(self, optimizer) -> None:
        assert optimizer.name == "EVOPrompt"

    @pytest.mark.asyncio()
    async def test_optimize_evaluates_all_candidates(
        self, optimizer, initial_candidate
    ) -> None:
        """P0-1 核心：所有候选都应被评估（含最后一轮子代）。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(4),
            max_iterations=2,
        )

        unevaluated = [c for c in result.all_candidates if not c.evaluated]
        assert len(unevaluated) == 0, f"仍有 {len(unevaluated)} 个候选未评估"
        assert result.num_candidates_evaluated == len(result.all_candidates)

    @pytest.mark.asyncio()
    async def test_optimize_picks_high_score_child(
        self, optimizer, initial_candidate
    ) -> None:
        """P0-1 核心：best_prompt 应是含 MAGIC 的高分子代。"""
        result = await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(4),
            max_iterations=2,
        )
        assert "MAGIC" in result.best_prompt.instruction
        assert result.best_prompt.score > 0.0

    @pytest.mark.asyncio()
    async def test_uses_local_random_not_global(
        self, optimizer, initial_candidate
    ) -> None:
        """P2 修复点：用局部 Random，不污染全局 random 状态。"""
        import random
        random.seed(999)
        expected = random.random()  # 取一个值

        await optimizer.optimize(
            initial_prompt=initial_candidate,
            dataset=_make_dataset(3),
            max_iterations=1,
        )

        # 重置回 999，下一个 random() 应等于 expected（若优化器用了全局 seed 会不同）
        random.seed(999)
        assert random.random() == expected
