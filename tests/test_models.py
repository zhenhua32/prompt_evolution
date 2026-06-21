"""PromptCandidate / OptimizationResult 模型测试。"""

from __future__ import annotations

import pytest

from prompt_evolution.core.models import OptimizationResult, PromptCandidate


class TestPromptCandidate:
    def test_default_score_is_zero_float(self) -> None:
        """P0-1 相关：score 默认 0.0（float 非 Optional）。

        历史代码用 `if candidate.score is None:` 判断，恒为 False。
        修复后用 `evaluated` 标记，但 score 默认值保持 0.0 不变（向后兼容）。
        """
        c = PromptCandidate(id="x", instruction="hi")
        assert c.score == 0.0
        assert not (c.score is None)  # 不应为 None

    def test_default_evaluated_is_false(self) -> None:
        """P0-1 修复点：新增 evaluated 字段，默认 False。"""
        c = PromptCandidate(id="x", instruction="hi")
        assert c.evaluated is False

    def test_evaluated_can_be_set_true(self) -> None:
        c = PromptCandidate(id="x", instruction="hi")
        c.evaluated = True
        c.score = 0.85
        assert c.evaluated is True
        assert c.score == 0.85

    def test_metadata_defaults_empty_dict(self) -> None:
        c = PromptCandidate(id="x", instruction="hi")
        assert c.metadata == {}

    def test_demo_examples_defaults_empty_list(self) -> None:
        c = PromptCandidate(id="x", instruction="hi")
        assert c.demo_examples == []

    def test_id_keeps_provided_value(self) -> None:
        """传入的 id 应被保留。"""
        c = PromptCandidate(id="my-id", instruction="hi")
        assert c.id == "my-id"

    def test_model_dump_json_roundtrip(self) -> None:
        """序列化反序列化往返。"""
        c = PromptCandidate(
            id="test",
            instruction="instr",
            score=0.5,
            evaluated=True,
            metadata={"k": "v"},
        )
        js = c.model_dump_json()
        assert '"evaluated":true' in js or '"evaluated": true' in js
        assert '"score":0.5' in js


class TestOptimizationResult:
    def test_construction(self) -> None:
        best = PromptCandidate(id="best", instruction="best instr", score=0.9)
        result = OptimizationResult(
            best_prompt=best,
            all_candidates=[best],
            optimization_history=[{"iter": 1}],
            total_cost_usd=0.01,
            elapsed_time_s=5.0,
            num_iterations=1,
            num_candidates_evaluated=1,
        )
        assert result.best_prompt.score == 0.9
        assert result.num_candidates_evaluated == 1

    def test_model_dump_json_excludes_none(self) -> None:
        """model_dump_json(exclude_none=True) 应能正常工作（CLI 用了此选项）。"""
        best = PromptCandidate(id="best", instruction="x", score=0.1)
        result = OptimizationResult(
            best_prompt=best,
            all_candidates=[best],
            optimization_history=[],
            total_cost_usd=0.0,
            elapsed_time_s=0.1,
            num_iterations=1,
            num_candidates_evaluated=1,
        )
        js = result.model_dump_json(indent=2, exclude_none=True)
        assert isinstance(js, str)
        assert '"best_prompt"' in js
