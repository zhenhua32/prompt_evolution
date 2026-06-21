#!/usr/bin/env python3
"""验证 evaluator 修复效果：对比 baseline / 修复前 evaluator / 修复后 evaluator。

不跑完整优化器（太慢），只验证评估链路本身是否正确：
1. baseline 链路：prompt.replace("{input}", x) 直接调 provider → 应得到 ~0.85
2. 修复后 evaluator 链路：Evaluator.evaluate(initial_prompt) → 应追平 baseline
3. 修复前 evaluator 链路的 bug（仅作对照，可选）

用法：
    python scripts/verify_evaluator_fix.py --eval-samples 50
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

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

from prompt_evolution.core.models import PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.providers.litellm_provider import LiteLLMProvider

TRAIN_FILE = Path("examples/ag_news_test.json")  # 用 test 集做评估（与 benchmark 一致）
INITIAL_PROMPT = """你是一个新闻分类专家。请根据输入的新闻标题，判断它属于以下哪个类别：

类别列表：科技、股票、体育、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座

只输出类别名称，不要输出任何其他内容。

新闻标题："{input}"
类别："""


def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(data, n, seed=42):
    if n <= 0 or n >= len(data):
        return data
    import random
    from collections import defaultdict
    random.seed(seed)
    groups: dict[str, list] = defaultdict(list)
    for item in data:
        groups[str(item["target"])].append(item)
    total = len(data)
    cats = list(groups.keys())
    float_quotas = [(c, len(groups[c]) / total * n) for c in cats]
    int_quotas = {c: int(q) for c, q in float_quotas}
    remainder = n - sum(int_quotas.values())
    remainders = sorted(float_quotas, key=lambda x: -(x[1] - int(x[1])))
    for c, _ in remainders[:remainder]:
        int_quotas[c] += 1
    sampled = []
    for c, k in int_quotas.items():
        sampled.extend(random.sample(groups[c], min(k, len(groups[c]))))
    random.shuffle(sampled)
    return sampled


async def baseline_eval(provider, dataset):
    """baseline 链路：直接 replace 占位符，与 benchmark_news.py 一致。"""
    semaphore = asyncio.Semaphore(5)
    predictions = [""] * len(dataset)
    references = [str(item["target"]).strip() for item in dataset]

    async def _one(idx, item):
        full_prompt = INITIAL_PROMPT.replace("{input}", item["input"])
        async with semaphore:
            try:
                resp = await provider.generate(prompt=full_prompt, temperature=0.0, max_tokens=512)
                predictions[idx] = resp.strip()
            except Exception as e:
                predictions[idx] = ""
                print(f"  [{idx}] baseline err: {e}")

    await asyncio.gather(*[_one(i, item) for i, item in enumerate(dataset)])
    correct = sum(1 for p, r in zip(predictions, references) if p.lower() == r.lower())
    return correct / len(dataset), predictions, references


async def evaluator_eval(provider, dataset):
    """修复后 evaluator 链路：通过 Evaluator 类评估同一个 initial_prompt。"""
    evaluator = Evaluator(metrics=[AccuracyMetric()], concurrency=5)
    candidate = PromptCandidate(id="initial", instruction=INITIAL_PROMPT)
    score = await evaluator.evaluate(prompt=candidate, dataset=dataset, model_provider=provider)
    return score


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-samples", type=int, default=50, help="评估样本数（默认 50）")
    parser.add_argument("--model", default=os.environ.get("MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", None))
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    if os.environ.get("DISABLE_THINKING", "").lower() == "true":
        args.disable_thinking = True

    full_data = load_json(TRAIN_FILE)
    dataset = stratified_sample(full_data, args.eval_samples)
    print(f"📂 数据集：{len(dataset)} 条（从 {len(full_data)} 条分层采样）")

    provider = LiteLLMProvider(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        api_base=args.base_url,
        disable_thinking=args.disable_thinking,
    )

    print(f"\n{'='*60}")
    print("1️⃣ baseline 链路（prompt.replace，与 benchmark_news.py 一致）")
    print(f"{'='*60}")
    t0 = time.time()
    baseline_acc, preds, refs = await baseline_eval(provider, dataset)
    elapsed = time.time() - t0
    print(f"   baseline accuracy = {baseline_acc:.4f}  ({elapsed:.1f}s)")
    # 打印前 5 条预测 vs 真实
    for i in range(min(5, len(preds))):
        mark = "✅" if preds[i].lower() == refs[i].lower() else "❌"
        print(f"   [{i}] {mark} pred={preds[i]!r} ref={refs[i]!r}")

    print(f"\n{'='*60}")
    print("2️⃣ 修复后 Evaluator 链路（含占位符替换 + 无双发 + 并发）")
    print(f"{'='*60}")
    t0 = time.time()
    eval_acc = await evaluator_eval(provider, dataset)
    elapsed = time.time() - t0
    print(f"   evaluator accuracy = {eval_acc:.4f}  ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print("📊 对比结论")
    print(f"{'='*60}")
    diff = eval_acc - baseline_acc
    if abs(diff) < 0.02:
        verdict = "✅ 修复成功：两条链路结果一致（差异 < 2pp）"
    elif diff > 0:
        verdict = f"⚠️ evaluator 比 baseline 高 {diff:+.4f}（可能并发顺序差异）"
    else:
        verdict = f"❌ 仍有差距：evaluator 比 baseline 低 {diff:+.4f}"
    print(f"   baseline  = {baseline_acc:.4f}")
    print(f"   evaluator = {eval_acc:.4f}")
    print(f"   差值      = {diff:+.4f}")
    print(f"   {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
