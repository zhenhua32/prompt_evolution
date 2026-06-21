"""SPO (Semantic Prompt Optimization) 优化器实现。

论文：_Semantic Prompt Optimization for Large Language Models_
(ACL Findings, 2024)
DOI: https://doi.org/10.18653/v1/2024.findings-acl.595

核心思想：
  利用语义嵌入（embedding）引导 prompt 优化方向：
  1. 将初始 prompt 和数据集样本编码为语义向量
  2. 在语义空间中搜索高分邻域，生成语义上「不同但有效」的候选
  3. 评估候选，保留高分 prompt
  4. 用高分 prompt 重新设定搜索中心，迭代优化

实现说明（本项目轻量版）：
  本实现不依赖外部 embedding 服务，而是用 LLM 本身
  做「语义改写」：每一轮让 LLM 在保留原意的前提下，
  从多个语义角度重写 prompt，实现语义空间的探索。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


# 语义改写 system prompt（核心算子）
_SEMANTIC_REWRITE_SYSTEM = """\
You are a semantic prompt rewriter.

Your task: Rewrite the given prompt from a NEW semantic angle
while PRESERVING its original intent.

Semantic angles to try (pick one per rewrite):
1. Change the persona (e.g. "You are a..." → different expert role)
2. Change the reasoning strategy (e.g. add/remove "think step by step")
3. Change the output format specification
4. Change the level of detail (more/fewer constraints)
5. Change the examples or framing

CRITICAL CONSTRAINTS:
- The rewritten prompt MUST still produce CORRECT outputs for the same task — only the wording/angle changes.
- The rewritten prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the rewritten prompt, wrapped in triple backticks:
```
<rewritten prompt>
```
"""

# 搜索引导 system prompt（用高分 prompt 引导下一轮搜索）
_GUIDED_SEARCH_SYSTEM = """\
You are a prompt optimization researcher.

You are given:
- The current best prompt (highest score so far)
- Its score (0.0 = worst, 1.0 = perfect)

Your job: Propose NEW prompts that are semantically related to
the best prompt but explore a slightly DIFFERENT angle.

The goal: find prompts in the "semantic neighborhood" of the best
prompt that might score even higher.

CRITICAL: Every candidate MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it.

Output each candidate wrapped in triple backticks:
```
<candidate prompt>
```
"""


class SPOOptimizer(BaseOptimizer):
    """SPO (Semantic Prompt Optimization) 优化器。

    工作流程（轻量实现）：
      1. **初始化**：评估 ``initial_prompt``，设为当前最优。
      2. **语义改写**：以当前最优 prompt 为种子，用 LLM
         从 ``num_angles`` 个不同语义角度各生成 ``num_candidates``
         个改写变体。
      3. **评估**：在数据集上评估所有变体。
      4. **更新中心**：如果某变体得分更高，将其设为新的搜索中心。
      5. **迭代**：重复 2–4，直到 ``max_iterations`` 轮。
      6. **返回**：整个过程中得分最高的 prompt。

    配置参数（在 ``config`` 中传入）：
      - ``num_candidates`` (int)：每轮生成的候选数，默认 6
      - ``num_angles`` (int)：语义改写的角度数，默认 3
      - ``rewrite_temperature`` (float)：改写时的 temperature，默认 0.8
      - ``guided_temperature`` (float)：引导搜索时的 temperature，默认 0.6
      - ``semantic_patience`` (int)：连续多少轮无语义改进后早停，默认 3
    """

    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_provider=model_provider, evaluator=evaluator, config=config
        )
        self._num_candidates: int = self.config.get("num_candidates", 6)
        self._num_angles: int = self.config.get("num_angles", 3)
        self._rewrite_temperature: float = self.config.get(
            "rewrite_temperature", 0.8
        )
        self._guided_temperature: float = self.config.get(
            "guided_temperature", 0.6
        )
        self._semantic_patience: int = self.config.get(
            "semantic_patience", 3
        )

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "SPO"

    async def optimize(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> OptimizationResult:
        import time

        start_time = time.time()
        cost_before: float = self.model_provider.total_cost_usd
        all_candidates: List[PromptCandidate] = []
        history: List[Dict[str, Any]] = []

        # 评估初始 prompt
        logger.info("SPO: evaluating initial prompt...")
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        all_candidates.append(initial_prompt)

        logger.info("SPO initial score: {:.4f}", initial_score)

        current_best = initial_prompt
        semantic_center = initial_prompt.instruction
        patience_counter: int = 0

        for iteration in range(1, max_iterations + 1):
            self.on_iteration_start(iteration)

            # 1. 语义改写：从多个角度生成候选
            candidates: List[PromptCandidate] = []

            # 角度 1-N：用当前最优 prompt 做语义改写
            for angle in range(self._num_angles):
                angle_candidates = await self._semantic_rewrite(
                    prompt=current_best,
                    angle=angle,
                    iteration=iteration,
                )
                candidates.extend(angle_candidates[: self._num_candidates // self._num_angles + 1])

            # 角度 N+1：用语义中心引导搜索（后期利用阶段）
            if iteration >= 2:
                guided = await self._guided_search(
                    best_prompt=current_best,
                    iteration=iteration,
                )
                candidates.extend(guided[: self._num_candidates // 3 + 1])

            # 2. 评估所有候选
            for candidate in candidates:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score
                logger.debug(
                    "SPO candidate '{}' score={:.4f}",
                    candidate.id[:8],
                    score,
                )
            all_candidates.extend(candidates)

            # 3. 更新 current_best 和 semantic_center
            iteration_best = max(
                candidates, key=lambda c: c.score if c.score is not None else -1.0
            )
            if (
                iteration_best.score is not None
                and iteration_best.score > current_best.score
            ):
                current_best = iteration_best
                semantic_center = current_best.instruction
                patience_counter = 0
                logger.info(
                    "SPO iteration {}/{}: new best={:.4f} ↑",
                    iteration,
                    max_iterations,
                    current_best.score,
                )
            else:
                patience_counter += 1
                logger.info(
                    "SPO iteration {}/{}: no improvement (patience={}/{}), "
                    "best={:.4f}",
                    iteration,
                    max_iterations,
                    patience_counter,
                    self._semantic_patience,
                    current_best.score,
                )

            # 4. 早停检查
            if patience_counter >= self._semantic_patience:
                logger.info(
                    "SPO early stop at iteration {} (semantic patience exhausted)",
                    iteration,
                )
                break

            history.append({
                "iteration": iteration,
                "num_candidates": len(candidates),
                "best_score": current_best.score,
                "best_prompt_id": current_best.id,
                "all_scores": [
                    c.score for c in candidates if c.score is not None
                ],
                "patience": patience_counter,
            })

            self.on_iteration_end(iteration, candidates)

        # 5. 选出全局最优
        best_prompt = max(
            all_candidates,
            key=lambda c: c.score if c.score is not None else -1.0,
        )

        total_cost: float = self.model_provider.total_cost_usd - cost_before
        elapsed = time.time() - start_time

        logger.info(
            "SPO done: best_score={:.4f}, elapsed={:.1f}s, cost=${:.4f}",
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
            num_iterations=max_iterations,
            num_candidates_evaluated=len(
                [c for c in all_candidates if c.score is not None]
            ),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _semantic_rewrite(
        self,
        prompt: PromptCandidate,
        angle: int,
        iteration: int,
    ) -> List[PromptCandidate]:
        """语义改写：从一个语义角度重写 prompt。

        参数
        ----------
        angle ：0-based 角度编号，用于让 LLM 每次侧重不同改写策略。
        """
        import re

        angle_names = [
            "change persona (expert role)",
            "change reasoning strategy (step-by-step, etc.)",
            "change output format specification",
            "change level of detail (constraints)",
            "change examples or framing",
        ]
        angle_name = angle_names[angle % len(angle_names)]

        prompt_text = (
            f"Current best prompt:\n```\n{prompt.instruction}\n```\n\n"
            f"Semantic angle for this rewrite: **{angle_name}**\n\n"
            f"Generate {self._num_candidates // self._num_angles + 1} "
            f"semantically rewritten variants. "
            f"Wrap each in triple backticks."
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_SEMANTIC_REWRITE_SYSTEM,
            temperature=self._rewrite_temperature,
            max_tokens=2048,
        )

        candidates: List[PromptCandidate] = []
        pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            for match in matches:
                instruction = match.strip()
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={
                                "source": "spo_rewrite",
                                "angle": angle_name,
                                "iteration": iteration,
                            },
                        )
                    )

        # 兜底：解析失败时返回原 prompt 的变体。
        # 变体标记放在 instruction 之前，避免破坏末尾输出引导（如 "\n类别："）。
        if not candidates:
            for i in range(self._num_candidates // self._num_angles + 1):
                candidates.append(
                    PromptCandidate(
                        id=str(uuid.uuid4()),
                        instruction=f"[semantic angle {angle}, var {i}]\n{prompt.instruction}",
                        metadata={
                            "source": "spo_rewrite_fallback",
                            "angle": angle,
                            "iteration": iteration,
                        },
                    )
                )

        return candidates[: self._num_candidates // self._num_angles + 1]

    async def _guided_search(
        self,
        best_prompt: PromptCandidate,
        iteration: int,
    ) -> List[PromptCandidate]:
        """引导搜索：用当前最优 prompt 引导 LLM 生成语义邻域候选。"""
        import re

        prompt_text = (
            f"Current best prompt (score reference):\n```\n{best_prompt.instruction}\n```\n\n"
            f"Generate {self._num_candidates // 3 + 1} NEW prompts "
            f"in the semantic neighborhood of the above best prompt.\n"
            f"Each should explore a slightly different angle but stay related."
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_GUIDED_SEARCH_SYSTEM,
            temperature=self._guided_temperature,
            max_tokens=2048,
        )

        candidates: List[PromptCandidate] = []
        pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            for match in matches:
                instruction = match.strip()
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={
                                "source": "spo_guided",
                                "iteration": iteration,
                            },
                        )
                    )

        return candidates[: self._num_candidates // 3 + 1]
