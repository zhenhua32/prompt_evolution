#!/usr/bin/env python3
"""验证 P0-1 修复：PromptBreeder / EvoPrompt 候选评估不再失效。

修复前：`if candidate.score is None:` 恒为 False（score 默认 0.0），子代从未被评估。
修复后：用 `evaluated` 标记判断，Evaluator 评估后自动设置 evaluated=True。

本脚本用 mock provider + mock evaluator 模拟一次完整进化流程，检查：
1. 子代候选的 evaluated 标记是否被正确设置
2. 子代的 score 是否不再是默认 0.0
3. 最终选出的 best_prompt 是否是真正得分最高的候选（而非随机 0.0 的子代）

用法：
    python scripts/verify_p0_fix.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# 让脚本能从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_evolution.core.models import PromptCandidate, OptimizationResult
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.optimizers.prompt_breeder.optimizer import PromptBreederOptimizer
from prompt_evolution.optimizers.evoprompt.optimizer import EVOPromptOptimizer


class MockModelProvider:
    """Mock provider：按 instruction 的内容返回可预测的预测结果。

    评估时 provider.generate(prompt=full_prompt) 收到的 full_prompt 是
    instruction.replace("{input}", user_input)。我们让「包含特定关键词的
    instruction」预测正确，从而得到高分；否则预测错误得 0 分。
    这样可以验证子代是否真的被评估了。
    """

    def __init__(self) -> None:
        self.total_cost_usd = 0.0
        self.call_count = 0

    async def generate(self, prompt: str = "", system_prompt: str = None, **kwargs: Any) -> str:
        self.call_count += 1
        # 如果是变异/交叉/变体生成调用（带 system_prompt），返回一个「含 magic 词」的 prompt
        if system_prompt is not None:
            return f"```\n你是专家。关键词MAGIC。新闻：{{input}}\n类别：\n```"
        # 否则是评估调用：prompt 里含 {input} 被替换成了真实输入
        # 数据集 target 是 "正确类别"，我们让含 MAGIC 的 prompt 预测正确
        if "MAGIC" in prompt:
            return "正确类别"
        return "错误类别"


def make_dataset(n: int = 6) -> List[Dict[str, Any]]:
    return [{"input": f"新闻标题{i}", "target": "正确类别"} for i in range(n)]


async def test_optimizer(name: str, optimizer_cls, expected_min_evaluated: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"测试 {name}")
    print(f"{'=' * 60}")

    provider = MockModelProvider()
    evaluator = Evaluator(metrics=[AccuracyMetric()], concurrency=3)
    optimizer = optimizer_cls(
        model_provider=provider,
        evaluator=evaluator,
        config={
            "population_size": 6,
            "init_variants": 3,
            "num_mutations": 1,
            "early_stop_patience": 10,  # 不早停，跑满轮数
        },
    )

    # 初始 prompt 不含 MAGIC → 得 0 分；子代含 MAGIC → 应得满分
    initial = PromptCandidate(
        id="initial",
        instruction='你是新闻分类助手。新闻："{input}"\n类别：',
    )
    dataset = make_dataset(6)

    result: OptimizationResult = await optimizer.optimize(
        initial_prompt=initial,
        dataset=dataset,
        max_iterations=2,
    )

    # 检查所有候选的 evaluated 标记
    all_cands = result.all_candidates
    evaluated_count = sum(1 for c in all_cands if c.evaluated)
    not_evaluated = [c for c in all_cands if not c.evaluated]

    print(f"  候选总数: {len(all_cands)}")
    print(f"  已评估 (evaluated=True): {evaluated_count}")
    print(f"  未评估 (evaluated=False): {len(not_evaluated)}")
    print(f"  provider.generate 调用次数: {provider.call_count}")
    print(f"  best_prompt score: {result.best_prompt.score:.4f}")
    print(f"  best_prompt 含 MAGIC: {'MAGIC' in result.best_prompt.instruction}")

    # 断言
    assert evaluated_count >= expected_min_evaluated, (
        f"{name}: 评估候选数 {evaluated_count} < 预期 {expected_min_evaluated}，"
        f"说明 evaluated 标记没生效，子代可能仍未被评估"
    )
    assert len(not_evaluated) == 0, (
        f"{name}: 仍有 {len(not_evaluated)} 个候选未评估"
    )
    assert result.best_prompt.score > 0.0, (
        f"{name}: best_prompt.score={result.best_prompt.score}，"
        f"说明子代没被评估（仍为默认 0.0），best 选不出含 MAGIC 的高分候选"
    )
    assert "MAGIC" in result.best_prompt.instruction, (
        f"{name}: best_prompt 不含 MAGIC，说明没选到高分子代"
    )
    print(f"  ✅ {name} 通过：子代已被正确评估，best_prompt 是真正的高分候选")


async def main() -> None:
    print("=" * 60)
    print("P0-1 修复验证：PromptBreeder / EvoPrompt 候选评估")
    print("=" * 60)
    print("原理：")
    print("  - 修复前 `if candidate.score is None:` 恒 False → 子代从不评估")
    print("  - 修复后用 `evaluated` 标记 → Evaluator 评估后置 True")
    print("  - 验证：子代 evaluated 应为 True，score 不再是 0.0，")
    print("    best_prompt 应是含 MAGIC 关键词的高分子代")

    await test_optimizer("PromptBreeder", PromptBreederOptimizer, expected_min_evaluated=4)
    await test_optimizer("EVOPrompt", EVOPromptOptimizer, expected_min_evaluated=4)

    print(f"\n{'=' * 60}")
    print("✅ 全部通过：P0-1 修复有效")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
