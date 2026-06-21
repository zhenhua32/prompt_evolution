"""评估器核心实现。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from prompt_evolution.core.base import BaseEvaluator, BaseMetric
from prompt_evolution.core.models import PromptCandidate
from prompt_evolution.evaluation.metrics.base import _REGISTRY, BaseMetric as _BaseMetric


class Evaluator(BaseEvaluator):
    """默认评估器实现。

    支持：
    - 确定型指标（Accuracy, ExactMatch 等，无需 LLM 调用）
    - LLM-as-Judge 指标（用 LLM 打分）
    - 多指标加权组合
    """

    def __init__(
        self,
        metrics: Optional[List[_BaseMetric]] = None,
        model_provider: Optional[Any] = None,
        concurrency: int = 5,
    ) -> None:
        """
        Args:
            metrics: 使用的指标列表，按 ``compute()`` 返回分数平均。
            model_provider: LLM 提供商（LLM-as-Judge 指标需要）。
            concurrency: 单次评估内的并发请求数（与 baseline 对齐，默认 5）。
        """
        self._metrics = metrics or []
        self._model_provider = model_provider
        self._concurrency: int = max(1, int(concurrency))

    # ------------------------------------------------------------------
    # BaseEvaluator 接口
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        model_provider: Optional[Any] = None,
    ) -> float:
        """评估一个 prompt 在数据集上的表现，返回 0~1 分数。"""
        provider = model_provider or self._model_provider
        if provider is None:
            raise ValueError("model_provider 不能为空，请在构造 Evaluator 或调用时传入。")

        # 1. 用 prompt + provider 在数据集上生成预测
        predictions = await self._generate_predictions(prompt, dataset, provider)

        # 2. 提取 ground truth
        references = [self._extract_reference(item) for item in dataset]

        # 3. 计算各指标分数，取平均
        if not self._metrics:
            # 无指标时默认返回 0.0 并警告
            logger.warning("Evaluator 未配置任何指标，返回分数 0.0")
            return 0.0

        scores = []
        for metric in self._metrics:
            try:
                score = metric.compute(predictions, references)
                scores.append(score)
                logger.debug("Metric '{}' score={:.4f}", metric.name, score)
            except Exception as exc:
                logger.error("Metric '{}' 计算失败: {}", metric.name, exc)
                scores.append(0.0)

        final_score = sum(scores) / len(scores)
        logger.info(
            "Evaluator: prompt_id={} final_score={:.4f}",
            prompt.id[:8],
            final_score,
        )
        return final_score

    def evaluate_batch(
        self,
        prompts: List[PromptCandidate],
        dataset: List[Dict[str, Any]],
        model_provider: Any,
        parallel: int = 5,
    ) -> List[float]:
        """批量评估多个 prompt（并发）。"""
        import asyncio

        async def _eval_one(p: PromptCandidate) -> float:
            return await self.evaluate(p, dataset, model_provider)

        async def _run_all() -> List[float]:
            # 简单并发控制（Python asyncio 不支持真正的 max_workers，这里用 gather）
            return await asyncio.gather(*(_eval_one(p) for p in prompts))

        return asyncio.run(_run_all())

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _generate_predictions(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        provider: Any,
    ) -> List[str]:
        """用给定 prompt 在数据集上生成所有预测。

        修复要点（与 baseline ``benchmark_news.py::evaluate_prompt`` 对齐）：
        1. **占位符替换**：若 ``prompt.instruction`` 含 ``{input}``，
           用 ``replace("{input}", user_input)`` 把用户输入嵌入到 instruction
           原位，保留 prompt 末尾的输出引导（如 ``\\n类别：``）。
           不含占位符时退化为 instruction + 用户输入的拼接（向后兼容）。
        2. **不传 system_prompt**：避免 instruction 在 system 和 user
           两条消息里重复发送（双发会破坏输出格式引导、稀释信号）。
        3. **并发**：用 ``asyncio.Semaphore`` 控制并发，与 baseline
           的并发行为对齐，显著缩短 SPO/OPRO 耗时。
        """
        # 预渲染 few-shot 上下文（如果有）
        fewshot_context = ""
        for ex in prompt.demo_examples:
            role = ex.get("role", "user")
            content = ex.get("content", "")
            fewshot_context += f"{role}: {content}\n"

        instruction = prompt.instruction
        has_placeholder = "{input}" in instruction

        semaphore = asyncio.Semaphore(self._concurrency)
        predictions: List[str] = ["" for _ in dataset]

        async def _predict(idx: int, item: Dict[str, Any]) -> None:
            user_input = self._extract_input(item)
            if has_placeholder:
                # 路径 A：占位符替换 — 与 baseline 完全一致
                full_prompt = instruction.replace("{input}", user_input)
                if fewshot_context:
                    # few-shot 加在 instruction 之前，避免破坏末尾输出引导
                    full_prompt = f"{fewshot_context}\n{full_prompt}"
            else:
                # 路径 B：兜底拼接 — 兼容未使用占位符的 prompt
                full_prompt = (
                    f"{instruction}\n\n{fewshot_context}用户输入：{user_input}"
                ).strip()

            async with semaphore:
                try:
                    pred = await provider.generate(
                        prompt=full_prompt,
                        system_prompt=None,
                        temperature=0.0,  # 评估时用确定性输出
                        max_tokens=512,
                    )
                    predictions[idx] = pred.strip()
                except Exception as exc:
                    logger.error("生成预测失败 (idx={}): {}", idx, exc)
                    predictions[idx] = ""

        await asyncio.gather(*(_predict(i, item) for i, item in enumerate(dataset)))
        return predictions

    @staticmethod
    def _extract_input(item: Dict[str, Any]) -> str:
        """从数据项中提取用户输入。"""
        for key in ("input", "question", "query", "text"):
            if key in item:
                return str(item[key])
        # 兜底：用第一个非 target/answer/label 的字段
        for k, v in item.items():
            if k not in ("target", "answer", "label", "output"):
                return str(v)
        return ""

    @staticmethod
    def _extract_reference(item: Dict[str, Any]) -> str:
        """从数据项中提取 ground truth。"""
        for key in ("target", "answer", "label", "output"):
            if key in item:
                return str(item[key])
        return ""
