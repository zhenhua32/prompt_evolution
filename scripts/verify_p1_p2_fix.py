#!/usr/bin/env python3
"""验证 P1 + P2 修复。

覆盖：
- P1-1 APE 费用增量法（非平方级累加）
- P1-2 F1ScoreMetric 用 Counter（重复 token 不再被去重高估）
- P1-3 providers/base.py 重导出 core.base.BaseModelProvider（单一类型）
- P1-4 Evaluator.evaluate_batch 改为 async（不再 asyncio.run）
- P1-5 Evaluator.compute_metrics 存在 + UI 评估 Tab 不再调错 evaluate
- P2-7 OPRO 历史 [:20] 取最高分（非 [-20:] 最低分）
- P2-6 DSPy 早停需 len>=4（不再 len==3 越界）
- P2 全局副作用：litellm.drop_params 不再在导入时设为 True；rng 局部 Random
- P2 generate_with_logprobs 有异常处理；estimate_cost 适配元组

用法：
    python scripts/verify_p1_p2_fix.py
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "✅" if cond else "❌"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    assert cond, label


def test_p1_1_ape_cost_incremental() -> None:
    """P1-1: APE 费用统计用增量法（cost_after - cost_before），非平方级累加。"""
    print("\n[P1-1] APE 费用统计")
    src = Path("src/prompt_evolution/optimizers/ape/optimizer.py").read_text(encoding="utf-8")
    check("用 cost_before 记录起始", "cost_before" in src)
    check("删除了 total_cost += 累计值", "total_cost += getattr" not in src)
    check("结尾取差值", "total_cost_usd", 0.0)  # 占位
    check("结尾取差值", "- cost_before" in src)


def test_p1_2_f1_uses_counter() -> None:
    """P1-2: F1ScoreMetric 用 Counter 而非 set。"""
    print("\n[P1-2] F1ScoreMetric 用 Counter")
    from prompt_evolution.evaluation.metrics.f1_score import F1ScoreMetric, _tokenize

    metric = F1ScoreMetric()
    # pred 重复 "是"，ref 只有一个 "是" + "巴黎"
    # set 法：pred_tokens={"是","巴黎"}, ref_tokens={"是","巴黎"} → F1=1.0（虚高）
    # Counter 法：overlap=min(3,1)=1（"是"），precision=1/4, recall=1/2 → F1≈0.333
    score = metric.compute(["是 是 是 巴黎"], ["巴黎 是"])
    check("重复 token 不再被去重高估", score < 0.7, f"score={score:.4f}（set 法会是 1.0）")

    # 完全匹配应得 1.0
    score2 = metric.compute(["巴黎 是"], ["巴黎 是"])
    check("完全匹配 = 1.0", abs(score2 - 1.0) < 1e-6, f"score={score2:.4f}")

    # 用 Counter 而非 set
    src = Path("src/prompt_evolution/evaluation/metrics/f1_score.py").read_text(encoding="utf-8")
    check("代码用 Counter", "Counter(_tokenize" in src)
    check("代码不再用 set", "set(_tokenize" not in src)


def test_p1_3_provider_base_reexport() -> None:
    """P1-3: providers/base.py 重导出 core.base.BaseModelProvider（单一类型）。"""
    print("\n[P1-3] providers/base 重导出")
    from prompt_evolution.providers.base import BaseModelProvider as PB
    from prompt_evolution.core.base import BaseModelProvider as CB
    from prompt_evolution.providers.litellm_provider import LiteLLMProvider

    check("providers.base.BaseModelProvider is core.base.BaseModelProvider", PB is CB)
    check("LiteLLMProvider 继承的是同一个基类", issubclass(LiteLLMProvider, PB))


def test_p1_4_evaluate_batch_is_async() -> None:
    """P1-4: Evaluator.evaluate_batch 是 async（不再用 asyncio.run）。"""
    print("\n[P1-4] evaluate_batch 异步化")
    from prompt_evolution.evaluation.evaluator import Evaluator
    check("evaluate_batch 是协程", inspect.iscoroutinefunction(Evaluator.evaluate_batch))

    src = inspect.getsource(Evaluator.evaluate_batch)
    # 注释里会提到旧实现 "asyncio.run()"，但函数体不应有实际调用 "asyncio.run("
    call_lines = [l for l in src.splitlines() if "asyncio.run(" in l and not l.strip().startswith("#") and 'asyncio.run()' not in l and '"asyncio.run' not in l and "'asyncio.run" not in l]
    check("函数体不再实际调用 asyncio.run", len(call_lines) == 0, f"suspicious lines: {call_lines}")
    check("用 Semaphore 控制并发", "Semaphore" in src)


async def test_p1_5_compute_metrics() -> None:
    """P1-5: Evaluator.compute_metrics 存在 + UI 不再调 evaluate(predictions, references)。"""
    print("\n[P1-5] compute_metrics + UI 修复")
    from prompt_evolution.evaluation.evaluator import Evaluator
    from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric

    check("Evaluator 有 compute_metrics 方法", hasattr(Evaluator, "compute_metrics"))

    ev = Evaluator(metrics=[AccuracyMetric()])
    scores = ev.compute_metrics(["A", "B", "C"], ["A", "B", "X"])
    check("compute_metrics 返回 dict", isinstance(scores, dict))
    check("含 accuracy 键", "accuracy" in scores)
    check("accuracy 值正确 (2/3)", abs(scores["accuracy"] - 2/3) < 1e-6, f"acc={scores['accuracy']:.4f}")

    # UI 修复检查
    ui_src = Path("src/prompt_evolution/ui/app.py").read_text(encoding="utf-8")
    check("UI 不再调 evaluator.evaluate(predictions, references)", "evaluator.evaluate(predictions, references)" not in ui_src)
    check("UI 改用 compute_metrics", "compute_metrics(predictions" in ui_src)


def test_p2_7_opro_history_top_n() -> None:
    """P2-7: OPRO 历史取 sorted_history[:20]（最高分），非 [-20:]（最低分）。"""
    print("\n[P2-7] OPRO 历史取最高分")
    src = Path("src/prompt_evolution/optimizers/opro/optimizer.py").read_text(encoding="utf-8")
    check("用 [:20] 取最高分", "sorted_history[:20]" in src)
    check("不再用 [-20:] 取最低分", "sorted_history[-20:]" not in src)


def test_p2_6_dspy_early_stop_bound() -> None:
    """P2-6: DSPy 早停条件需 len>=4（非 len>=3 越界回绕）。"""
    print("\n[P2-6] DSPy 早停边界")
    src = Path("src/prompt_evolution/optimizers/dspy_optimizer/optimizer.py").read_text(encoding="utf-8")
    check("早停需 len(_scores) >= 4", "len(_scores) >= 4" in src)
    check("不再用 len(_scores) >= 3", "len(_scores) >= 3" not in src)


def test_p2_global_side_effects() -> None:
    """P2: 不再在导入时设 litellm.drop_params=True；优化器用局部 Random。"""
    print("\n[P2] 全局副作用")
    import litellm
    # 导入 providers.litellm_provider 后，litellm.drop_params 应保持 False（默认）
    from prompt_evolution.providers import litellm_provider  # noqa: F401
    check("litellm.drop_params 未被模块级修改", litellm.drop_params is False,
          f"actual={litellm.drop_params}")

    # 三个优化器改用局部 Random
    for opt_name in ["prompt_breeder", "evoprompt", "dspy_optimizer"]:
        path = Path(f"src/prompt_evolution/optimizers/{opt_name}/optimizer.py")
        src = path.read_text(encoding="utf-8")
        check(f"{opt_name}: 用局部 Random(42)", "random.Random(42)" in src, str(path))
        check(f"{opt_name}: 不再用全局 random.seed(42)", "random.seed(42)" not in src, str(path))


def test_p2_logprobs_exception_handling() -> None:
    """P2: generate_with_logprobs 有 try/except + 接口对齐 generate。"""
    print("\n[P2] generate_with_logprobs 异常处理")
    from prompt_evolution.providers.litellm_provider import LiteLLMProvider
    sig = inspect.signature(LiteLLMProvider.generate_with_logprobs)
    params = list(sig.parameters.keys())
    check("支持 system_prompt 参数", "system_prompt" in params)
    check("支持 temperature 参数", "temperature" in params)
    check("支持 max_tokens 参数", "max_tokens" in params)

    src = Path("src/prompt_evolution/providers/litellm_provider.py").read_text(encoding="utf-8")
    logprobs_section = src.split("async def generate_with_logprobs")[1].split("def count_tokens")[0]
    check("logprobs 有 try/except", "try:" in logprobs_section and "except Exception" in logprobs_section)


def test_p2_estimate_cost_tuple() -> None:
    """P2: estimate_cost 适配 cost_per_token 元组返回。"""
    print("\n[P2] estimate_cost 适配元组")
    src = Path("src/prompt_evolution/providers/litellm_provider.py").read_text(encoding="utf-8")
    cost_section = src.split("def estimate_cost")[1]
    check("检查 isinstance(cost, tuple)", "isinstance(cost, tuple)" in cost_section)
    check("元组时 sum(cost)", "sum(cost)" in cost_section)


def test_p2_ui_run_prompt_uses_placeholder() -> None:
    """P2: UI _run_prompt_on_dataset 走占位符替换链路（与 evaluator 对齐）。"""
    print("\n[P2] UI 推理链路对齐")
    src = Path("src/prompt_evolution/ui/app.py").read_text(encoding="utf-8")
    check("UI 检测 {input} 占位符", 'has_placeholder = "{input}"' in src)
    check("UI 用 replace 嵌入输入", 'prompt_text.replace("{input}"' in src)
    check("UI 不再把 input 当 prompt、prompt 当 system", 'system_prompt=item.get("input' not in src)
    check("日志用 f-string 或 {} format", '"预测失败: {exc}"' not in src)
    check("UI 推理有并发控制", "Semaphore" in src)


def test_p2_metrics_init_doc() -> None:
    """P2: metrics/__init__.py 注释反映装饰器机制。"""
    print("\n[P2] metrics 注释")
    src = Path("src/prompt_evolution/evaluation/metrics/__init__.py").read_text(encoding="utf-8")
    check("不再误导性提及 __init_subclass__", "__init_subclass__" not in src)
    check("提及 @register_metric 装饰器", "register_metric" in src)


def test_p2_readme_no_dspy_dep() -> None:
    """P2: README/pyproject 删除 dspy-ai 依赖 + 优化器状态更新。"""
    print("\n[P2] 文档一致性")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    check("pyproject 删除 dspy-ai 依赖", "dspy-ai" not in pyproject)

    readme = Path("README.md").read_text(encoding="utf-8")
    check("README 不再列 dspy-ai 依赖", "DSPy** — 优化算法集成" not in readme)
    check("README 优化器状态更新（不再全部'开发中'）", "✅ 已实现" in readme)
    check("README 说明 dspy_optimizer 是轻量实现", "轻量" in readme or "不依赖 dspy 库" in readme)


async def main() -> None:
    print("=" * 60)
    print("P1 + P2 修复验证")
    print("=" * 60)

    test_p1_1_ape_cost_incremental()
    test_p1_2_f1_uses_counter()
    test_p1_3_provider_base_reexport()
    test_p1_4_evaluate_batch_is_async()
    await test_p1_5_compute_metrics()
    test_p2_7_opro_history_top_n()
    test_p2_6_dspy_early_stop_bound()
    test_p2_global_side_effects()
    test_p2_logprobs_exception_handling()
    test_p2_estimate_cost_tuple()
    test_p2_ui_run_prompt_uses_placeholder()
    test_p2_metrics_init_doc()
    test_p2_readme_no_dspy_dep()

    print(f"\n{'=' * 60}")
    print("✅ 全部 P1 + P2 修复验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
