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
        self._num_iterations: int = self.config.get("num_iterations", 1)
        self._generation_temperature: float = self.config.get("generation_temperature", 1.0)
        self._prompt_gen_system: str = self.config.get(
            "prompt_gen_system",
            "You are a helpful prompt engineer. "
            "Your task is to write the BEST possible instruction prompt "
            "for the given task. Make it clear, specific, and effective.",
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
        total_cost: float = 0.0
        all_candidates: List[PromptCandidate] = []
        history: List[Dict[str, Any]] = []

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
                total_cost += getattr(self.model_provider, "_total_cost", 0.0)
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
        # 构造生成候选的 prompt
        dataset_sample = ""
        for i, item in enumerate(dataset[:3]):  # 只用前 3 条作为示例
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))
            dataset_sample += f"输入：{inp}\n期望输出：{tgt}\n\n"

        generation_prompt = (
            f"{self._prompt_gen_system}\n\n"
            f"以下是任务的若干示例：\n\n{dataset_sample}"
            f"当前使用的 prompt 是：\n```\n{initial_prompt.instruction}\n```\n\n"
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

        # 如果解析出的候选不足，用初始 prompt 填充
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=initial_prompt.instruction + f" (variant {len(candidates) + 1})",
                    metadata={"source": "ape_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]
