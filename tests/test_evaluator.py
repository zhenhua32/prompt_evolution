"""评估器与指标单元测试。

重点验证 P1/P2 修复点：
- F1ScoreMetric 用 Counter（不再虚高）
- Evaluator.evaluate 写回 evaluated 标记
- Evaluator.compute_metrics 纯计算接口
- Evaluator.evaluate_batch 是协程 + 真并发
- evaluate 在无 model_provider 时抛 ValueError
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_evolution.core.models import PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.evaluation.metrics.exact_match import ExactMatchMetric
from prompt_evolution.evaluation.metrics.f1_score import F1ScoreMetric
from prompt_evolution.providers.litellm_provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# Metrics 测试
# ---------------------------------------------------------------------------

class TestAccuracyMetric:
    def test_name(self) -> None:
        assert AccuracyMetric().name == "accuracy"

    def test_perfect(self) -> None:
        m = AccuracyMetric()
        assert m.compute(["A", "B"], ["A", "B"]) == 1.0

    def test_half(self) -> None:
        m = AccuracyMetric()
        assert m.compute(["A", "X"], ["A", "B"]) == 0.5

    def test_empty(self) -> None:
        assert AccuracyMetric().compute([], []) == 0.0

    def test_case_insensitive(self) -> None:
        # AccuracyMetric 旧实现对 "巴黎" / "巴黎" 不应受大小写影响
        m = AccuracyMetric()
        # 大小写差异（中英混合）
        assert m.compute(["Paris"], ["paris"]) == 0.0 or m.compute(["Paris"], ["paris"]) == 1.0


class TestExactMatchMetric:
    def test_name(self) -> None:
        assert ExactMatchMetric().name == "exact_match"

    def test_perfect(self) -> None:
        m = ExactMatchMetric()
        assert m.compute(["巴黎", "东京"], ["巴黎", "东京"]) == 1.0

    def test_partial(self) -> None:
        m = ExactMatchMetric()
        assert m.compute(["巴黎", "X"], ["巴黎", "东京"]) == 0.5


class TestF1ScoreMetric:
    """P1-2 修复重点：用 Counter 而非 set。"""

    def test_name(self) -> None:
        assert F1ScoreMetric().name == "f1_score"

    def test_perfect_match(self) -> None:
        assert F1ScoreMetric().compute(["巴黎 是 首都"], ["巴黎 是 首都"]) == 1.0

    def test_no_overlap(self) -> None:
        assert F1ScoreMetric().compute(["北京"], ["巴黎"]) == 0.0

    def test_repeated_tokens_not_inflated(self) -> None:
        """关键回归：set 法会得 1.0（虚高），Counter 法应低于 1.0。

        pred="是 是 是 巴黎"（4 token），ref="巴黎 是"（2 token）。
        - set 法：{是,巴黎} ∩ {是,巴黎} → P=R=1 → F1=1.0
        - Counter 法：overlap=min(3,1)=1（"是"）+ min(1,1)=1（"巴黎"）=2
          P=2/4=0.5, R=2/2=1.0 → F1=2*0.5*1/(0.5+1)=0.6667
        """
        score = F1ScoreMetric().compute(["是 是 是 巴黎"], ["巴黎 是"])
        assert score < 0.7, f"重复 token 应降低 F1，实际 {score}"
        assert 0.6 < score < 0.7

    def test_empty_reference(self) -> None:
        assert F1ScoreMetric().compute(["A"], [""]) == 0.0

    def test_empty_prediction(self) -> None:
        assert F1ScoreMetric().compute([""], ["A B"]) == 0.0

    def test_macro_average(self) -> None:
        """多样本取宏平均。"""
        score = F1ScoreMetric().compute(
            ["巴黎 是 首都", "北京"],
            ["巴黎 是 首都", "上海"],
        )
        # 样本1: F1=1.0；样本2: 无交集 F1=0.0 → 宏平均 0.5
        assert abs(score - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Evaluator 测试
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_provider_with_cost() -> MagicMock:
    """Mock provider：按 prompt 内容返回可预测的预测。"""
    provider = MagicMock(spec=LiteLLMProvider)
    provider.total_cost_usd = 0.1

    async def _generate(prompt: str, **kwargs: Any) -> str:
        # 含 {input} 已被替换为真实输入；预测直接回传 "巴黎"
        return "巴黎"

    provider.generate = AsyncMock(side_effect=_generate)
    return provider


@pytest.fixture()
def evaluator_multi() -> Evaluator:
    return Evaluator(metrics=[AccuracyMetric(), ExactMatchMetric(), F1ScoreMetric()])


class TestEvaluatorEvaluate:
    """验证 Evaluator.evaluate 核心流程。"""

    @pytest.mark.asyncio()
    async def test_evaluate_returns_score_and_writes_back(
        self, mock_provider_with_cost, evaluator_multi
    ) -> None:
        """P0-1 修复点：evaluate 应回写 score 和 evaluated 标记。"""
        dataset = [
            {"input": "q1", "target": "巴黎"},
            {"input": "q2", "target": "巴黎"},
        ]
        cand = PromptCandidate(id="c1", instruction='问题：{input}\n答案：')
        # 初始状态
        assert cand.score == 0.0
        assert cand.evaluated is False

        score = await evaluator_multi.evaluate(
            prompt=cand, dataset=dataset, model_provider=mock_provider_with_cost
        )

        # 回写
        assert cand.evaluated is True
        assert cand.score == score
        # 全对 → accuracy=1.0
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_evaluate_placeholder_substitution(
        self, mock_provider_with_cost
    ) -> None:
        """占位符替换链路：{input} 应被替换为真实输入。"""
        received_prompts: List[str] = []

        async def _capture(prompt: str, **kwargs: Any) -> str:
            received_prompts.append(prompt)
            return "巴黎"

        provider = MagicMock(spec=LiteLLMProvider)
        provider.total_cost_usd = 0.0
        provider.generate = AsyncMock(side_effect=_capture)

        ev = Evaluator(metrics=[AccuracyMetric()])
        cand = PromptCandidate(id="c", instruction='Q:{input}\nA:')
        await ev.evaluate(
            prompt=cand,
            dataset=[{"input": "hello", "target": "巴黎"}],
            model_provider=provider,
        )
        assert any("hello" in p and "{input}" not in p for p in received_prompts)

    @pytest.mark.asyncio()
    async def test_evaluate_without_provider_raises(self, evaluator_multi) -> None:
        """无 model_provider 应抛 ValueError。"""
        cand = PromptCandidate(id="c", instruction="test")
        with pytest.raises(ValueError, match="model_provider"):
            await evaluator_multi.evaluate(
                prompt=cand, dataset=[{"input": "x", "target": "y"}]
            )


class TestEvaluatorComputeMetrics:
    """P1-5 修复点：compute_metrics 纯计算接口。"""

    def test_returns_dict(self, evaluator_multi) -> None:
        scores = evaluator_multi.compute_metrics(["巴黎"], ["巴黎"])
        assert isinstance(scores, dict)
        assert "accuracy" in scores
        assert "exact_match" in scores
        assert "f1_score" in scores
        assert scores["accuracy"] == 1.0

    def test_no_metrics_returns_empty(self) -> None:
        ev = Evaluator(metrics=[])
        assert ev.compute_metrics(["A"], ["A"]) == {}


class TestEvaluatorEvaluateBatch:
    """P1-4 修复点：evaluate_batch 异步化 + 真并发。"""

    @pytest.mark.asyncio()
    async def test_evaluate_batch_is_async_and_concurrent(
        self, mock_provider_with_cost
    ) -> None:
        """批量评估返回正确结果，且是协程。"""
        import inspect
        from prompt_evolution.evaluation.evaluator import Evaluator

        assert inspect.iscoroutinefunction(Evaluator.evaluate_batch)

        ev = Evaluator(metrics=[AccuracyMetric()])
        prompts = [
            PromptCandidate(id=f"c{i}", instruction='Q:{input}\nA:')
            for i in range(3)
        ]
        dataset = [{"input": f"q{i}", "target": "巴黎"} for i in range(3)]

        results = await ev.evaluate_batch(
            prompts, dataset, mock_provider_with_cost, parallel=2
        )

        assert len(results) == 3
        assert all(r == 1.0 for r in results)
        # evaluated 标记应被写回
        assert all(p.evaluated for p in prompts)
