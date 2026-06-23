"""APE (Automatic Prompt Engineer) 优化器实现。

论文：_Large Language Models Are Human-Level Prompt Engineers_
核心思想：
  1. 用 LLM 生成大量候选 prompt
  2. 在数据集上评估每个候选
  3. 选取得分最高的 prompt 作为输出
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


class APEOptimizer(BaseOptimizer):
    """APE (Automatic Prompt Engineer) 优化器。

    工作流程：
    1. **生成阶段**：用 LLM 根据初始 prompt + 数据集样本，
       生成 ``num_candidates`` 个候选 prompt。
    2. **评估阶段**：在每个候选上运行评估器，得到分数。
    3. **选择阶段**：返回得分最高的候选。

    配置参数（在 ``config`` 中传入）：
    - ``num_candidates`` (int)：生成的候选 prompt 数量，默认 10。
    - ``num_iterations`` (int)：迭代轮数，默认 1（APE 本身无迭代，
       但保留此参数以兼容统一接口）。
    - ``generation_temperature`` (float)：生成候选时的 temperature，默认 1.0。
    - ``prompt_gen_system`` (str)：用于生成候选 prompt 的系统指令。
    """

    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_provider=model_provider, evaluator=evaluator, config=config)
        self._num_candidates: int = self.config.get("num_candidates", 10)
        # P2-1 修复：默认迭代轮数从 1 改为 3，扩大搜索空间
        self._num_iterations: int = self.config.get("num_iterations", 3)
        self._generation_temperature: float = self.config.get("generation_temperature", 1.0)
        self._prompt_gen_system: str = self.config.get(
            "prompt_gen_system",
            # P0-2 / P1-1 修复：中文化 + 强制保留输出格式约束
            "你是一位资深 Prompt 工程师。你的任务是针对给定任务，"
            "撰写最清晰、最具体、最有效的指令 Prompt。",
        )
        # P0-2 修复：约束 LLM 保留 {input} 占位符 + 原始输出格式约束 + 末尾输出引导
        self._placeholder_constraint: str = self.config.get(
            "placeholder_constraint",
            "重要约束（必须遵守）：\n"
            "1. 新 Prompt 必须原样保留字面占位符 {input}（仅出现一次），"
            "用于评估时替换为实际用户输入，不得删除、改名或重复。\n"
            "2. 必须原样保留原 Prompt 中的输出格式约束"
            "（如「只输出类别名称，不要输出任何其他内容」），"
            "不得改变输出格式。\n"
            "3. 必须保留原 Prompt 末尾的输出引导"
            "（如「类别：」「答案：」），不得改写或删除。\n"
            "4. 不得添加「请逐步分析」「step by step」等推理引导，"
            "输出必须是裸答案，不带任何前缀或解释。",
        )

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "APE"

    async def optimize(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> OptimizationResult:
        """运行 APE 优化，返回最优 prompt。"""
        import time

        start_time = time.time()
        # 用增量法统计费用：开头记录 provider 累计费用，结尾取差值。
        # 旧实现 `total_cost += provider._total_cost` 是累加累计值（平方级虚高）。
        cost_before: float = getattr(self.model_provider, "total_cost_usd", 0.0)
        all_candidates: List[PromptCandidate] = []
        history: List[Dict[str, Any]] = []

        # P0-1 修复：评估初始 prompt 作为基准候选，保证 best_prompt 不会劣于 baseline。
        # 旧实现 all_candidates 从不含 initial_prompt，导致 APE 机制性必然返回新生成的
        # 候选（即使全部都比初始 prompt 差），是 APE 劣化最严重的根因。
        logger.info("APE: evaluating initial prompt as baseline candidate...")
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        all_candidates.append(initial_prompt)
        logger.info("APE initial prompt score: {:.4f}", initial_score)

        num_iterations = min(max_iterations, self._num_iterations)
        logger.info(
            "APE optimizer start: iterations={}, candidates_per_iter={}",
            num_iterations,
            self._num_candidates,
        )

        for iteration in range(1, num_iterations + 1):
            self.on_iteration_start(iteration)

            # 1. 生成候选 prompt
            candidates = await self._generate_candidates(
                initial_prompt=initial_prompt,
                dataset=dataset,
                iteration=iteration,
            )

            # 2. 评估所有候选
            for candidate in candidates:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score
                logger.debug("Candidate '{}' score={:.4f}", candidate.id[:8], score)

            all_candidates.extend(candidates)

            # 3. 记录本轮最优
            best_this_iter = max(candidates, key=lambda c: c.score)
            history.append({
                "iteration": iteration,
                "num_candidates": len(candidates),
                "best_score": best_this_iter.score,
                "best_prompt_id": best_this_iter.id,
            })
            logger.info(
                "APE iteration {}/{}: best_score={:.4f}",
                iteration,
                num_iterations,
                best_this_iter.score,
            )

            self.on_iteration_end(iteration, candidates)

        # 4. 选出全局最优
        best_prompt = max(all_candidates, key=lambda c: c.score)

        elapsed = time.time() - start_time
        total_cost: float = getattr(self.model_provider, "total_cost_usd", 0.0) - cost_before
        logger.info(
            "APE done: best_score={:.4f}, elapsed={:.1f}s, total_cost=${:.4f}",
            best_prompt.score,
            elapsed,
            total_cost,
        )

        return OptimizationResult(
            best_prompt=best_prompt,
            all_candidates=all_candidates,
            optimization_history=history,
            total_cost_usd=total_cost,
            elapsed_time_s=elapsed,
            num_iterations=num_iterations,
            num_candidates_evaluated=len(all_candidates),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _generate_candidates(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        iteration: int,
    ) -> List[PromptCandidate]:
        """用 LLM 生成 ``num_candidates`` 个候选 prompt。"""
        # P2-2 修复：示例采样改为分层覆盖多类别，避免旧实现 dataset[:3] 前 3 条全同类
        # （本项目数据集前 3 条全是"财经"），导致 LLM 生成的 prompt 偏科、泛化性差。
        dataset_sample = self._build_diverse_sample(dataset, max_samples=6)

        generation_prompt = (
            f"{self._prompt_gen_system}\n\n"
            f"以下是任务的若干示例：\n\n{dataset_sample}"
            f"当前使用的 prompt 是：\n```\n{initial_prompt.instruction}\n```\n\n"
            f"{self._placeholder_constraint}\n\n"
            f"请生成 {self._num_candidates} 条**不同风格**的改进版 prompt。"
            f"每条 prompt 用 ``` 包裹，编号 1 ~ {self._num_candidates}。"
        )

        response = await self.model_provider.generate(
            prompt=generation_prompt,
            system_prompt=self._prompt_gen_system,
            temperature=self._generation_temperature,
            max_tokens=2048,
        )

        # 解析 LLM 输出，提取候选 prompt
        import re

        candidates: List[PromptCandidate] = []
        # 匹配 ``` 代码块内的内容
        pattern = r"```(?:prompt)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if not matches:
            # 兜底：按行分割，尝试提取
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            for i, line in enumerate(lines[: self._num_candidates]):
                candidates.append(
                    PromptCandidate(
                        id=str(uuid.uuid4()),
                        instruction=line,
                        metadata={"source": "ape_fallback", "iteration": iteration},
                    )
                )
        else:
            for i, match in enumerate(matches[: self._num_candidates]):
                candidates.append(
                    PromptCandidate(
                        id=str(uuid.uuid4()),
                        instruction=match.strip(),
                        metadata={"source": "ape", "iteration": iteration},
                    )
                )

        # P2-3 修复：padding 不再加前缀标记，直接复用 initial_prompt.instruction。
        # 旧实现加 [variant n] 前缀会改变 prompt 开头（"你是一个新闻分类专家" →
        # "[variant 3]\n你是一个..."），干扰模型角色认知。
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=initial_prompt.instruction,
                    metadata={"source": "ape_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]

    @staticmethod
    def _build_diverse_sample(
        dataset: List[Dict[str, Any]], max_samples: int = 6
    ) -> str:
        """构造覆盖多类别的示例文本（P2-2 修复）。

        旧实现用 ``dataset[:3]``，本项目数据集前 3 条全是"财经"类，
        LLM 看到的示例只覆盖 14 类中的 1 类，生成的 prompt 容易偏科。
        现按 target 分组，每个类别抽 1 条，最多 max_samples 条，
        保证示例覆盖尽量多的类别。
        """
        from collections import defaultdict, OrderedDict

        groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for item in dataset:
            tgt = str(item.get("target", item.get("answer", "")))
            groups.setdefault(tgt, []).append(item)

        sampled: List[Dict[str, Any]] = []
        # 轮询各类别各取 1 条，直到达到 max_samples
        for tgt, items in groups.items():
            if items:
                sampled.append(items[0])
            if len(sampled) >= max_samples:
                break

        lines: List[str] = []
        for i, item in enumerate(sampled):
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))
            lines.append(f"输入：{inp}\n期望输出：{tgt}\n")
        return "\n".join(lines)
