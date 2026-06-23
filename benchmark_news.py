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


def stratified_sample(
    data: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """分层采样：按原始类别分布抽取 n 条数据，防止分布偏移。

    对每个类别按其原始占比分配配额，随机无放回抽取。
    若 n 大于等于数据总量，直接返回全部数据。
    """
    if n <= 0:
        return []
    if n >= len(data):
        return data

    import random
    random.seed(seed)

    # 按类别分组
    from collections import defaultdict
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        groups[str(item["target"])].append(item)

    total = len(data)
    cats = list(groups.keys())

    # 按原始比例算浮点配额，再调整为整数
    float_quotas: list[tuple[str, float]] = [
        (cat, len(groups[cat]) / total * n) for cat in cats
    ]

    # 整数部分
    int_quotas: dict[str, int] = {cat: int(q) for cat, q in float_quotas}
    allocated = sum(int_quotas.values())
    remainder = n - allocated

    # 按小数部分从大到小分配余数
    remainders = sorted(float_quotas, key=lambda x: -(x[1] - int(x[1])))
    for cat, _ in remainders[:remainder]:
        int_quotas[cat] += 1

    # 从每类抽取
    sampled: list[dict[str, Any]] = []
    for cat, k in int_quotas.items():
        items = groups[cat]
        k = min(k, len(items))
        sampled.extend(random.sample(items, k))

    random.shuffle(sampled)
    return sampled


def print_data_distribution(data: list[dict[str, Any]], label: str) -> None:
    """打印数据集的类别分布（验证采样是否保持原始分布）。"""
    from collections import Counter
    c = Counter(str(d["target"]) for d in data)
    total = len(data)
    print(f"   {label} 类别分布（共 {total} 条）:")
    for cat, cnt in sorted(c.items()):
        print(f"     {cat}: {cnt} ({cnt / total * 100:.1f}%)")


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


async def select_best_on_holdout(
    provider: LiteLLMProvider,
    candidates: list,
    holdout_data: list[dict[str, Any]],
    concurrency: int = 5,
    top_k: int = 5,
) -> tuple[str, float]:
    """P1-2 修复：在 hold-out 验证集上重新评估 top-k 候选 prompt，选出最优。

    优化器内部在 train_data_fit 上搜索，报告的 best_prompt 可能过拟合 train。
    这里取 train 得分最高的 top_k 个候选，在独立的 hold-out 上重新评测，
    选 hold-out 得分最高者作为最终 best_prompt，降低过拟合风险。

    返回 (best_instruction, holdout_score)。
    """
    if not holdout_data or not candidates:
        # 无 hold-out 时退化为优化器报告的 best
        best = max(candidates, key=lambda c: c.score or 0.0) if candidates else None
        return (best.instruction if best else "", best.score if best else 0.0)

    # 按 train 得分取 top_k
    sorted_cands = sorted(candidates, key=lambda c: c.score or 0.0, reverse=True)
    top_candidates = sorted_cands[:top_k]

    print(f"   🔬 在 hold-out ({len(holdout_data)} 条) 上重新评估 top-{len(top_candidates)} 候选...")
    best_instruction = ""
    best_holdout_score = -1.0
    for i, cand in enumerate(top_candidates):
        score = await evaluate_prompt(
            provider,
            cand.instruction,
            holdout_data,
            concurrency=concurrency,
            log_file=None,
            checkpoint_file=None,
            method_name=f"_holdout_{i}",
        )
        print(f"      候选 {i + 1}: train={cand.score:.4f} → holdout={score:.4f}")
        if score > best_holdout_score:
            best_holdout_score = score
            best_instruction = cand.instruction

    return best_instruction, best_holdout_score


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
    parser.add_argument("--train-samples", type=int, default=400, help="训练时使用的数据条数（默认 400，用 0 表示全部，分层采样保持类别分布）。P1-2 修复：旧默认 200 样本过小导致选择偏差，现提升到 400 降低噪声")
    parser.add_argument("--eval-samples", type=int, default=100, help="评测时使用的数据条数（默认 100，用 0 表示全部，分层采样保持类别分布）")
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="从训练集中切出 hold-out 验证集的比例（默认 0.2）。P1-2 修复：优化器在 train 子集上搜索，"
        "在 hold-out 上选最优 prompt，再在 test 上报最终分，避免 train 选最优→test 暴露过拟合。设 0 禁用。",
    )
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
    # 分层采样：按原始类别比例抽取，防止分布偏移
    train_n = args.train_samples if args.train_samples and args.train_samples > 0 else full_train_count
    eval_n = args.eval_samples if args.eval_samples and args.eval_samples > 0 else full_test_count
    train_data = stratified_sample(full_train_data, min(train_n, full_train_count))
    test_data = stratified_sample(full_test_data, min(eval_n, full_test_count))

    # P1-2 修复：从训练集切出 hold-out 验证集。
    # 优化器在 train_data_fit（80%）上搜索候选，在 holdout（20%）上选最优 prompt，
    # 再在 test_data 上报最终分。避免「train 选最优 → test 暴露过拟合」。
    holdout_data: list[dict[str, Any]] = []
    train_data_fit: list[dict[str, Any]] = train_data
    if args.holdout_ratio and 0 < args.holdout_ratio < 1 and len(train_data) >= 20:
        import random as _rnd
        _rng = _rnd.Random(42)
        # 分层切分：按 target 分组，每组按比例切 hold-out
        from collections import defaultdict as _dd
        _groups: dict[str, list[dict[str, Any]]] = _dd(list)
        for item in train_data:
            _groups[str(item["target"])].append(item)
        train_data_fit = []
        holdout_data = []
        for cat, items in _groups.items():
            _rng.shuffle(items)
            n_holdout = max(1, int(len(items) * args.holdout_ratio)) if len(items) >= 5 else 0
            holdout_data.extend(items[:n_holdout])
            train_data_fit.extend(items[n_holdout:])
        _rng.shuffle(train_data_fit)
        _rng.shuffle(holdout_data)
        print(
            f"   🔬 hold-out 验证集: {len(holdout_data)} 条（用于选最优 prompt），"
            f"训练拟合集: {len(train_data_fit)} 条（用于优化器搜索）"
        )

    print(f"   训练集: {len(train_data)} 条（共 {full_train_count} 条）")
    print(f"   评测集: {len(test_data)} 条（共 {full_test_count} 条）")
    print_data_distribution(train_data, "训练集")
    print_data_distribution(test_data, "评测集")

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
        # 检查 checkpoint 是否已完整（必须有 test_accuracy 才算完成）
        if checkpoint_file and checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                method_cp = cp.get(method, {})
                test_acc = method_cp.get("test_accuracy")
                if test_acc is not None:
                    print(f"\n{'=' * 60}")
                    print(f"✅ {method} 在 checkpoint 中已完整（test_acc: {test_acc:.4f}），跳过")
                    print(f"{'=' * 60}")
                    results.append({
                        "method": method,
                        "score": test_acc,  # 主指标：测试集准确率
                        "train_score": method_cp.get("train_score"),
                        "elapsed_s": method_cp.get("elapsed_s", 0),
                        "from_checkpoint": True,
                    })
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
                train_data=train_data_fit,  # P1-2 修复：用拟合集训练（不含 hold-out）
                max_iterations=args.max_iters,
                num_candidates=args.num_candidates,
            )
            train_score = result.best_prompt.score if result.best_prompt else 0.0
            optimizer_best_instruction = result.best_prompt.instruction if result.best_prompt else INITIAL_PROMPT
            print(f"   ✅ 优化器报告得分 (train_fit): {train_score:.4f}")
            print(f"   耗时: {time.time() - t0:.1f}s")
            print(f"   优化器最优 Prompt: {optimizer_best_instruction[:100]}...")

            # P1-2 修复：在 hold-out 上重新评估 top-k 候选，选 hold-out 最优作为最终 best
            if holdout_data and result.all_candidates:
                best_instruction, holdout_score = await select_best_on_holdout(
                    provider,
                    result.all_candidates,
                    holdout_data,
                    concurrency=args.concurrency,
                    top_k=5,
                )
                print(f"   🔬 hold-out 最优得分: {holdout_score:.4f}")
                if not best_instruction:
                    best_instruction = optimizer_best_instruction
            else:
                best_instruction = optimizer_best_instruction

            # —— 关键：用 best_prompt 在测试集上评测，与 baseline 同口径 ——
            # 优化器报告的 train_score 是训练集得分，与 baseline 的测试集准确率不可比。
            # 必须在测试集上重新评测，才能公平比较优化器是否真的提升了 prompt。
            print(f"   📊 在测试集上评测最优 Prompt...")
            test_score = await evaluate_prompt(
                provider,
                best_instruction,
                test_data,
                concurrency=args.concurrency,
                log_file=log_file,
                checkpoint_file=checkpoint_file,
                method_name=f"{method}__test",
            )
            elapsed = time.time() - t0
            print(f"   📊 测试集准确率 (test): {test_score:.4f}")

            results.append(
                {
                    "method": method,
                    "score": test_score,  # 主指标：测试集准确率（与 baseline 同口径）
                    "train_score": train_score,  # 参考指标：优化器训练集得分
                    "elapsed_s": elapsed,
                    "best_prompt": best_instruction,
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
                    "train_score": train_score,
                    "test_accuracy": test_score,  # 主指标
                    "elapsed_s": elapsed,
                    "best_prompt": best_instruction,
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
    print(f"{'方法':<24} {'Test Acc':<12} {'Train Acc':<12} {'耗时(s)':<12}")
    print("-" * 60)
    for r in results:
        score_str = f"{r['score']:.4f}" if r.get("score") is not None else "ERROR"
        train_str = f"{r['train_score']:.4f}" if r.get("train_score") is not None else "-"
        elapsed_str = f"{r.get('elapsed_s', 0):.1f}"
        print(f"{r['method']:<24} {score_str:<12} {train_str:<12} {elapsed_str:<12}")
    print(f"{'=' * 60}")
    print("注：Test Acc = 最优 prompt 在测试集上的准确率（与 baseline 同口径，可对比）")
    print("    Train Acc = 优化器报告的训练集得分（仅供参考，反映过拟合情况）")

    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
