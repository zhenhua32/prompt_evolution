"""基础使用示例 — 最小可运行示例。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from prompt_evolution import PromptCandidate, create_optimizer
from prompt_evolution.evaluation import Evaluator
from prompt_evolution.evaluation.metrics import AccuracyMetric
from prompt_evolution.providers import LiteLLMProvider


async def main() -> None:
    """主函数。"""

    # 1. 准备数据集（如无 dataset.json，自动创建示例）
    dataset_path = Path("./examples/dataset.json")
    if not dataset_path.exists():
        dataset = [
            {"input": "1+1=？", "target": "2"},
            {"input": "2+3=？", "target": "5"},
            {"input": "10-4=？", "target": "6"},
            {"input": "3*5=？", "target": "15"},
        ]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已创建示例数据集：{dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 2. 初始化模型提供商（需配置 API Key）
    #    方式一：传入 api_key 参数
    #    方式二：在 .env 文件中配置 OPENAI_API_KEY 环境变量
    provider = LiteLLMProvider(
        model="openai/gpt-4o",
        api_key=None,  # 为 None 时会读取环境变量
    )

    # 3. 初始化评估器
    evaluator = Evaluator(metrics=[AccuracyMetric()])

    # 4. 创建优化器（当前支持：ape）
    optimizer = create_optimizer(
        name="ape",
        model_provider=provider,
        evaluator=evaluator,
        config={
            "num_candidates": 5,   # 每轮生成 5 个候选 prompt
            "num_iterations": 1,      # APE 通常 1 轮即可
        },
    )

    # 5. 构造初始 Prompt
    initial_prompt = PromptCandidate(
        id="initial",
        instruction="你是一个数学助手，请回答问题。",
    )

    # 6. 运行优化
    print(f"开始优化：方法={optimizer.name}, 数据集={len(dataset)} 条")
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        dataset=dataset,
        max_iterations=1,
    )

    # 7. 展示结果
    print("\n" + "=" * 60)
    print("优化完成！")
    print(f"最优 Prompt:\n{result.best_prompt.instruction}")
    print(f"得分: {result.best_prompt.score:.4f}")
    print(f"耗时: {result.elapsed_time_s:.1f}s")
    print(f"总费用: ${result.total_cost_usd:.4f}")
    print("=" * 60)

    # 8. 保存结果
    output_path = Path("./examples/result.json")
    output_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    print(f"结果已保存至：{output_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
