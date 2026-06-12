#!/usr/bin/env python3
"""新闻分类评测脚本 — 测试基础性能和各优化器效果。

用法:
    # 确保已设置 API Key
    export OPENAI_API_KEY=sk-xxx

    # 运行完整评测（所有优化器）
    python benchmark_news.py

    # 只跑指定优化器
    python benchmark_news.py --methods ape opro

    # 使用自定义模型
    python benchmark_news.py --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, find_dotenv

# 自动查找项目根目录的 .env（支持从任意目录运行脚本）
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("⚠️ 未找到 .env 文件，将只使用环境变量和命令行参数")

from prompt_evolution.core.models import PromptCandidate, OptimizationResult
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.optimizers.factory import create_optimizer, list_optimizers
from prompt_evolution.providers.litellm_provider import LiteLLMProvider

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TRAIN_FILE = Path("examples/ag_news_train.json")
TEST_FILE = Path("examples/ag_news_test.json")

INITIAL_PROMPT = """你是一个新闻分类专家。请根据输入的新闻标题，判断它属于以下哪个类别：

类别列表：科技、股票、体育、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座

只输出类别名称，不要输出任何其他内容。

新闻标题："{input}"
类别："""

RESULTS_FILE = Path("benchmark_results.json")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list[dict[str, Any]]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📊 结果已保存至: {RESULTS_FILE.resolve()}")


def enable_model_output_logging(provider: LiteLLMProvider) -> None:
    """在 benchmark 运行期间打印每次模型的原始输出。"""
    original_generate = provider.generate

    async def wrapped_generate(*args: Any, **kwargs: Any) -> str:
        text = await original_generate(*args, **kwargs)
        print("\n  ====== 模型原始输出 START ======")
        print(text if text else "<empty>")
        print("  ====== 模型原始输出 END ======\n")
        return text

    provider.generate = wrapped_generate  # type: ignore[method-assign]


async def evaluate_prompt(
    provider: LiteLLMProvider,
    prompt_instruction: str,
    dataset: list[dict[str, Any]],
) -> float:
    """用给定 prompt 在数据集上计算 Accuracy，逐条打印预测 vs 真实值（async）。"""
    correct = 0
    total = len(dataset)
    print(f"  🔍 开始评测 {total} 条数据...")

    for i, item in enumerate(dataset):
        input_text = item["input"]
        ground_truth = str(item["target"]).strip()

        # 填充 prompt 中的 {input} 占位符
        full_prompt = prompt_instruction.replace("{input}", input_text)

        # 调用模型
        try:
            response = await provider.generate(
                prompt=full_prompt,
                temperature=0.0,
                max_tokens=512,
            )
            prediction = response.strip()
        except Exception as e:
            print(f"  [{i+1}/{total}] ⚠️ 预测失败: {e}")
            continue

        # 判断正确与否
        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1

        # 每条都打印：预测值 vs 真实值
        status = "✅" if is_correct else "❌"
        print(f"  [{i+1}/{total}] {status} 预测: {prediction} | 真实: {ground_truth}")

    accuracy = correct / total if total > 0 else 0
    print(f"  📊 准确率: {accuracy:.4f} ({correct}/{total})")
    return accuracy


async def run_optimizer(
    method: str,
    provider: LiteLLMProvider,
    initial_prompt: str,
    train_data: list[dict[str, Any]],
    max_iterations: int = 3,
    num_candidates: int = 8,
) -> OptimizationResult:
    """运行指定优化器，返回结果。"""
    evaluator = Evaluator(metrics=[AccuracyMetric()])
    optimizer = create_optimizer(
        name=method,
        model_provider=provider,
        evaluator=evaluator,
        config={"num_candidates": num_candidates},
    )
    initial = PromptCandidate(id="initial", instruction=initial_prompt)
    return await optimizer.optimize(
        initial_prompt=initial,
        dataset=train_data,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="新闻分类 Prompt 优化评测")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"要评测的优化器，默认全部: {list(list_optimizers())}",
    )
    # --model 默认值：命令行 > .env 的 MODEL > openai/gpt-4o-mini
    default_model = os.environ.get("MODEL", "openai/gpt-4o-mini")
    parser.add_argument("--model", default=default_model, help="LiteLLM 模型标识（也可用 .env 中 MODEL）")
    parser.add_argument("--api-key", default=None, help="API Key（或用 .env 中 OPENAI_API_KEY）")
    # --base-url 默认值：命令行 > .env 的 OPENAI_BASE_URL
    default_base_url = os.environ.get("OPENAI_BASE_URL", None)
    parser.add_argument("--base-url", default=default_base_url, help="OpenAI 兼容 Base URL（也可用 .env 中 OPENAI_BASE_URL）")
    parser.add_argument("--max-iters", type=int, default=3, help="每个优化器最大迭代轮数")
    parser.add_argument("--num-candidates", type=int, default=8, help="每轮候选 prompt 数")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 baseline 评测")
    parser.add_argument("--train-samples", type=int, default=10, help="训练时使用的数据条数（默认 0 = 全部）")
    parser.add_argument("--eval-samples", type=int, default=10, help="评测时使用的数据条数（默认 100，用 0 表示全部）")
    parser.add_argument("--disable-thinking", action="store_true", help="关闭模型的 thinking/reasoning 输出（如 DeepSeek R1、Claude 等）")
    parser.add_argument("--print-model-output", action="store_true", help="打印每次模型调用的原始输出，便于排查 think 或格式问题")
    args = parser.parse_args()

    # .env 中的 DISABLE_THINKING=true 作为默认开启（命令行 --disable-thinking 可叠加）
    if os.environ.get("DISABLE_THINKING", "").lower() == "true":
        args.disable_thinking = True

    # 加载数据
    print(f"📂 加载数据集...")
    full_train_data = load_json(TRAIN_FILE)
    full_test_data = load_json(TEST_FILE)
    full_train_count = len(full_train_data)
    full_test_count = len(full_test_data)
    # 截取训练样本
    if args.train_samples and args.train_samples > 0 and args.train_samples < full_train_count:
        train_data = full_train_data[:args.train_samples]
    else:
        train_data = full_train_data
    # 截取评测样本
    if args.eval_samples and args.eval_samples > 0 and args.eval_samples < full_test_count:
        test_data = full_test_data[:args.eval_samples]
    else:
        test_data = full_test_data
    print(f"   训练集: {len(train_data)} 条（共 {full_train_count} 条，使用 --train-samples 调整）")
    print(f"   评测集: {len(test_data)} 条（共 {full_test_count} 条，使用 --eval-samples 调整）")

    # 初始化模型：API Key 来源 = 命令行 > .env 的 OPENAI_API_KEY
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and args.model.startswith("openai/"):
        print("⚠️ 警告：未提供 API Key，请设置 OPENAI_API_KEY 环境变量或用 --api-key")
    provider = LiteLLMProvider(model=args.model, api_key=api_key, api_base=args.base_url, disable_thinking=args.disable_thinking)
    if args.print_model_output:
        enable_model_output_logging(provider)
    print(f"   模型: {args.model}")
    if args.base_url:
        print(f"   Base URL: {args.base_url}")

    results: list[dict[str, Any]] = []

    # —— Baseline ——
    if not args.skip_baseline:
        print(f"\n{'=' * 60}")
        print("📊 Baseline（初始 Prompt 直接评测）")
        print(f"{'=' * 60}")
        t0 = time.time()
        baseline_score = await evaluate_prompt(provider, INITIAL_PROMPT, test_data)
        elapsed = time.time() - t0
        print(f"   Baseline Accuracy: {baseline_score:.4f}  ({elapsed:.1f}s)")
        results.append({
            "method": "baseline",
            "score": baseline_score,
            "elapsed_s": elapsed,
        })

    # —— 各优化器 ——
    methods = args.methods or list(list_optimizers())
    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"🚀 运行优化器: {method}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            result: OptimizationResult = await run_optimizer(
                method=method,
                provider=provider,
                initial_prompt=INITIAL_PROMPT,
                train_data=train_data,
                max_iterations=args.max_iters,
                num_candidates=args.num_candidates,
            )
            elapsed = time.time() - t0
            score = result.best_prompt.score if result.best_prompt else 0.0
            print(f"   ✅ 最优 Prompt 得分: {score:.4f}")
            print(f"   耗时: {elapsed:.1f}s")
            print(f"   最优 Prompt: {result.best_prompt.instruction[:100]}...")
            results.append({
                "method": method,
                "score": score,
                "elapsed_s": elapsed,
                "best_prompt": result.best_prompt.instruction if result.best_prompt else "",
                "num_candidates_evaluated": result.num_candidates_evaluated,
                "total_cost_usd": result.total_cost_usd,
            })
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            results.append({"method": method, "score": None, "error": str(e)})

    # —— 汇总 ——
    print(f"\n{'=' * 60}")
    print("📊 评测结果汇总")
    print(f"{'=' * 60}")
    print(f"{'方法':<20} {'Accuracy':<12} {'耗时(s)':<12}")
    print("-" * 60)
    for r in results:
        score_str = f"{r['score']:.4f}" if r.get("score") is not None else "ERROR"
        print(f"{r['method']:<20} {score_str:<12} {r.get('elapsed_s', 0):<12.1f}")
    print(f"{'=' * 60}")

    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
