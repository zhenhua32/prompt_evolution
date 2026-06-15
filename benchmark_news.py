#!/usr/bin/env python3
"""新闻分类评测脚本 — 测试基础性能和各优化器效果。

支持断点续评：
  - 运行时每完成一条评测立即写入 benchmark_checkpoint.json
  - 中断后重新运行，自动跳过已完成的样本
  - 使用 --no-checkpoint 可禁用此功能

用法:
    # 确保已设置 API Key
    export OPENAI_API_KEY=sk-xxx

    # 运行完整评测（所有优化器）
    python benchmark_news.py

    # 只跑指定优化器
    python benchmark_news.py --methods ape opro

    # 使用自定义模型 + 并发数
    python benchmark_news.py --model openai/gpt-4o --concurrency 10

    # 禁用断点续评（每次从头开始）
    python benchmark_news.py --no-checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from datetime import datetime

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
CHECKPOINT_FILE = Path("benchmark_checkpoint.json")


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


def enable_logging(provider: LiteLLMProvider, log_file: Path) -> None:
    """包装 provider.generate()，把完整 prompt 和模型返回写入日志文件。"""
    original_generate = provider.generate
    log_lock = asyncio.Lock()

    async def wrapped_generate(*args: Any, **kwargs: Any) -> str:
        prompt_text = kwargs.get("prompt", "")
        messages = kwargs.get("messages", None)
        text = await original_generate(*args, **kwargs)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with log_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"[{ts}] 模型调用\n")
                f.write("=" * 60 + "\n")
                f.write("\n--- PROMPT ---\n")
                if messages:
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        f.write(f"[{role}]\n{content}\n\n")
                else:
                    f.write(f"{prompt_text}\n")
                f.write("--- RESPONSE ---\n")
                f.write(f"{text if text else '<empty>'}\n")
                f.write("=" * 60 + "\n\n")

        return text

    provider.generate = wrapped_generate  # type: ignore[method-assign]
    print(f"📝 已启用模型调用日志，写入: {log_file.resolve()}")


def save_checkpoint(method_name: str, done: dict, total: int, checkpoint_file: Path) -> None:
    """将当前 done 字典写入 checkpoint 文件（保留其他方法的数据）。"""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                cp = json.load(f)
        except Exception:
            cp = {}
    else:
        cp = {}

    correct = sum(1 for v in done.values() if v.get("correct", False))
    cp[method_name] = {
        "done": [
            {"idx": idx, "prediction": v.get("prediction", ""), "correct": v.get("correct", False)}
            for idx, v in sorted(done.items())
        ],
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 评测函数（支持断点续评）
# ---------------------------------------------------------------------------


async def evaluate_prompt(
    provider: LiteLLMProvider,
    prompt_instruction: str,
    dataset: list[dict[str, Any]],
    concurrency: int = 5,
    log_file: Path | None = None,
    checkpoint_file: Path | None = None,
    method_name: str = "baseline",
) -> float:
    """用给定 prompt 在数据集上并发计算 Accuracy，支持断点续评。

    断点续评机制：
    - checkpoint_file 存在时，自动加载已完成的样本索引，跳过已评测的。
    - 每完成一条评测，立即追加写入 checkpoint，中断后下次可续跑。
    """
    total = len(dataset)

    # ── 加载 checkpoint ──
    done: dict[int, dict] = {}  # idx -> {prediction, correct}
    if checkpoint_file and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                cp = json.load(f)
            method_cp = cp.get(method_name, {})
            raw = method_cp.get("done", [])
            for item in raw:
                idx = item["idx"]
                done[idx] = {"prediction": item["prediction"], "correct": item["correct"]}
            if done:
                print(f"  📂 加载 checkpoint：已评测 {len(done)}/{total} 条，将跳过")
        except Exception as e:
            print(f"  ⚠️ 加载 checkpoint 失败: {e}，将从头开始")

    # 待评测的样本索引
    pending = [i for i in range(total) if i not in done]
    if not pending:
        correct = sum(1 for v in done.values() if v["correct"])
        accuracy = correct / total
        print(f"  ✅ checkpoint 已完整，跳过评测（Accuracy: {accuracy:.4f}）")
        return accuracy

    print(f"  🔍 开始评测 {len(pending)} 条新数据（并发数: {concurrency}，共 {total} 条）")

    semaphore = asyncio.Semaphore(concurrency)
    log_lock = asyncio.Lock()

    async def _eval_one(idx: int, item: dict[str, Any]) -> None:
        input_text = item["input"]
        ground_truth = str(item["target"]).strip()
        full_prompt = prompt_instruction.replace("{input}", input_text)

        async with semaphore:
            try:
                response = await provider.generate(
                    prompt=full_prompt,
                    temperature=0.0,
                    max_tokens=512,
                )
                prediction = response.strip()
            except Exception as e:
                print(f"  [{idx+1}/{total}] ⚠️ 预测失败: {e}")
                done[idx] = {"prediction": "", "correct": False, "error": str(e)}
                if checkpoint_file:
                    save_checkpoint(method_name, done, total, checkpoint_file)
                return

        is_correct = prediction == ground_truth
        status = "✅" if is_correct else "❌"
        print(f"  [{idx+1}/{total}] {status} 预测: {prediction} | 真实: {ground_truth}")
        done[idx] = {"prediction": prediction, "correct": is_correct}

        # 立即写入 checkpoint
        if checkpoint_file:
            save_checkpoint(method_name, done, total, checkpoint_file)

        # 写入日志
        if log_file:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            async with log_lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write(f"[{ts}] 评测样本 {idx+1}/{total}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"\n--- 输入新闻 ---\n{input_text}\n")
                    f.write(f"\n--- 模型预测 ---\n{prediction}\n")
                    f.write(f"\n--- 真实类别 ---\n{ground_truth}\n")
                    f.write(f"\n--- 结果 ---\n{'正确 ✅' if is_correct else '错误 ❌'}\n")
                    f.write("=" * 60 + "\n\n")

    # 并发评测待处理样本
    t0 = time.time()
    await asyncio.gather(*[_eval_one(i, dataset[i]) for i in pending])
    elapsed = time.time() - t0

    correct = sum(1 for v in done.values() if v.get("correct", False))
    accuracy = correct / total if total > 0 else 0
    print(f"  📊 准确率: {accuracy:.4f} ({correct}/{total})  （耗时 {elapsed:.1f}s）")
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
    default_model = os.environ.get("MODEL", "openai/gpt-4o-mini")
    parser.add_argument("--model", default=default_model, help="LiteLLM 模型标识（也可用 .env 中 MODEL）")
    parser.add_argument("--api-key", default=None, help="API Key（或用 .env 中 OPENAI_API_KEY）")
    default_base_url = os.environ.get("OPENAI_BASE_URL", None)
    parser.add_argument(
        "--base-url", default=default_base_url, help="OpenAI 兼容 Base URL（也可用 .env 中 OPENAI_BASE_URL）"
    )
    parser.add_argument("--max-iters", type=int, default=3, help="每个优化器最大迭代轮数")
    parser.add_argument("--num-candidates", type=int, default=8, help="每轮候选 prompt 数")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 baseline 评测")
    parser.add_argument("--train-samples", type=int, default=100, help="训练时使用的数据条数（默认 0 = 全部）")
    parser.add_argument("--eval-samples", type=int, default=40, help="评测时使用的数据条数（默认 100，用 0 表示全部）")
    parser.add_argument("--concurrency", type=int, default=5, help="并发请求数（默认 5，设 1 为完全串行）")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="关闭模型的 thinking/reasoning 输出（如 DeepSeek R1、Claude 等）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="benchmark.log",
        help="模型调用日志文件路径（默认 benchmark.log，设空字符串 '' 可关闭日志）",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default=None, help="checkpoint 文件路径（默认 benchmark_checkpoint.json）"
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="禁用断点续评（每次从头开始）")
    args = parser.parse_args()

    # .env 中的 DISABLE_THINKING=true 作为默认开启
    if os.environ.get("DISABLE_THINKING", "").lower() == "true":
        args.disable_thinking = True

    # 确定 checkpoint 文件
    checkpoint_file = None
    if not args.no_checkpoint:
        checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else CHECKPOINT_FILE

    # --no-checkpoint：删除已有 checkpoint
    if args.no_checkpoint and checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"🗑️  已删除旧 checkpoint，将从头开始评测")

    # 加载数据
    print(f"📂 加载数据集...")
    full_train_data = load_json(TRAIN_FILE)
    full_test_data = load_json(TEST_FILE)
    full_train_count = len(full_train_data)
    full_test_count = len(full_test_data)
    if args.train_samples and args.train_samples > 0 and args.train_samples < full_train_count:
        train_data = full_train_data[: args.train_samples]
    else:
        train_data = full_train_data
    if args.eval_samples and args.eval_samples > 0 and args.eval_samples < full_test_count:
        test_data = full_test_data[: args.eval_samples]
    else:
        test_data = full_test_data
    print(f"   训练集: {len(train_data)} 条（共 {full_train_count} 条）")
    print(f"   评测集: {len(test_data)} 条（共 {full_test_count} 条）")

    # 初始化模型
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and args.model.startswith("openai/"):
        print("⚠️ 警告：未提供 API Key，请设置 OPENAI_API_KEY 环境变量或用 --api-key")
    provider = LiteLLMProvider(
        model=args.model,
        api_key=api_key,
        api_base=args.base_url,
        disable_thinking=args.disable_thinking,
    )

    # 日志文件
    log_file = Path(args.log_file) if args.log_file else None
    if log_file:
        enable_logging(provider, log_file)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt Evolution Benchmark 日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"Base URL: {args.base_url or '(default)'}\n")
            f.write(f"\n{'=' * 60}\n\n")

    print(f"   模型: {args.model}")
    if args.base_url:
        print(f"   Base URL: {args.base_url}")
    if checkpoint_file:
        print(f"   Checkpoint: {checkpoint_file}（断点续评已启用）")

    results: list[dict[str, Any]] = []

    # —— Baseline ——
    if not args.skip_baseline:
        print(f"\n{'=' * 60}")
        print("📊 Baseline（初始 Prompt 直接评测）")
        print(f"{'=' * 60}")
        t0 = time.time()
        baseline_score = await evaluate_prompt(
            provider,
            INITIAL_PROMPT,
            test_data,
            concurrency=args.concurrency,
            log_file=log_file,
            checkpoint_file=checkpoint_file,
            method_name="baseline",
        )
        elapsed = time.time() - t0
        print(f"   Baseline Accuracy: {baseline_score:.4f}  ({elapsed:.1f}s)")
        results.append(
            {
                "method": "baseline",
                "score": baseline_score,
                "elapsed_s": elapsed,
            }
        )

    # —— 各优化器 ——
    methods = args.methods or list(list_optimizers())
    for method in methods:
        # 检查 checkpoint 是否已完整
        if checkpoint_file and checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                if method in cp and cp[method].get("accuracy") is not None:
                    acc = cp[method]["accuracy"]
                    print(f"\n{'=' * 60}")
                    print(f"✅ {method} 在 checkpoint 中已完整（Accuracy: {acc:.4f}），跳过")
                    print(f"{'=' * 60}")
                    results.append({"method": method, "score": acc, "from_checkpoint": True})
                    continue
            except Exception:
                pass

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
            results.append(
                {
                    "method": method,
                    "score": score,
                    "elapsed_s": elapsed,
                    "best_prompt": result.best_prompt.instruction if result.best_prompt else "",
                    "num_candidates_evaluated": result.num_candidates_evaluated,
                    "total_cost_usd": result.total_cost_usd,
                }
            )
            # 将优化器结果写入 checkpoint
            if checkpoint_file:
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        cp = json.load(f)
                except Exception:
                    cp = {}
                cp[method] = {
                    "done": [],  # 优化器不保存逐条结果
                    "score": score,
                    "elapsed_s": elapsed,
                    "accuracy": score,
                    "total": len(test_data),
                }
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(cp, f, ensure_ascii=False, indent=2)
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
        elapsed_str = f"{r.get('elapsed_s', 0):.1f}" if r.get("from_checkpoint") else f"{r.get('elapsed_s', 0):.1f}"
        print(f"{r['method']:<20} {score_str:<12} {elapsed_str:<12}")
    print(f"{'=' * 60}")

    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
