"""核心数据模型 — Pydantic 定义所有共享数据结构."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PromptCandidate(BaseModel):
    """一个 Prompt 候选对象."""

    id: str = Field(..., description="唯一标识")
    instruction: str = Field("", description="系统指令 / 主 prompt 文本")
    demo_examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Few-shot 示例，每项形如 {'role': 'user', 'content': '...'}",
    )
    score: float = Field(0.0, description="评估得分，越高越好")
    evaluated: bool = Field(
        False,
        description="是否已被 Evaluator 评估过。"
        "进化算法用此标记区分「未评估（score=0.0 默认值）」与「真实得分为 0.0」的候选，"
        "避免把未评估的候选误当作已评估而跳过。",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="算法自定义扩展字段"
    )

    model_config = {"extra": "allow"}


class OptimizationResult(BaseModel):
    """一次优化运行的完整结果."""

    best_prompt: PromptCandidate = Field(..., description="得分最高的 prompt")
    all_candidates: List[PromptCandidate] = Field(
        default_factory=list, description="所有历史候选 prompt"
    )
    optimization_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="每轮迭代的摘要记录"
    )
    total_cost_usd: float = Field(0.0, description="本次优化总 API 费用（美元）")
    elapsed_time_s: float = Field(0.0, description="总耗时（秒）")
    num_iterations: int = Field(0, description="实际迭代轮数")
    num_candidates_evaluated: int = Field(0, description="累计评估的候选 prompt 数")

    model_config = {"extra": "allow"}
