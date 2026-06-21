"""PromptBreeder 优化器实现。

论文：_Large Language Models as Evolutionary Heuristics_
(Google DeepMind, 2023)
arXiv: https://arxiv.org/abs/2311.01918

核心思想：
  将 Prompt 优化视为进化算法问题：
  1. 维护一个 prompt 种群（population）
  2. 用 LLM 对高适应度 prompt 进行「变异」（mutation）
     和「交叉」（crossover），产生新一代 prompt
  3. 经过多代进化，种群整体适应度不断提升
  4. 返回历史最优 prompt

算法流程：
  1. 初始化种群：由 initial_prompt 的若干变体构成
  2. 评估种群中每个 prompt 的适应度（= 在数据集上的得分）
  3. 选择精英（top-k 高适应度 prompt）
  4. 对精英进行变异和交叉，生成子代，补充到种群
  5. 如果种群大小超过上限，淘汰低适应度个体
  6. 重复步骤 2-5，直到达到最大迭代轮数
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


# 变异用 system prompt
_MUTATION_SYSTEM = """You are a prompt mutation operator.

Your job: Given a prompt, produce a slightly modified version that
could perform better at the same task.

Mutation strategies (pick one or combine several):
- Rephrase for clarity
- Add/remove constraints or guidelines
- Change the output format specification
- Adjust the tone (more formal / more casual)
- Add a brief example if missing

CRITICAL: The mutated prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the mutated prompt text, wrapped in triple backticks:
```
<mutated prompt>
```
"""

# 交叉用 system prompt
_CROSSOVER_SYSTEM = """You are a prompt crossover operator.

Your job: Given two parent prompts, produce a new prompt that
combines the best aspects of both.

Crossover strategy:
- Take the clearest instructions from both
- Merge output format specifications
- Keep the most effective constraints from each
- Resolve contradictions by picking the clearer wording

CRITICAL: The crossed-over prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the crossed-over prompt, wrapped in triple backticks:
```
<crossed-over prompt>
```
"""

# 种群初始化时的变体生成 prompt
_INIT_VARIANT_SYSTEM = """You are generating variant prompts for an evolutionary search.

Given a base prompt, generate a variant that expresses the same
intent but with different wording, structure, or emphasis.

CRITICAL: The variant prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the variant prompt, wrapped in triple backticks:
```
<variant prompt>
```
"""


class PromptBreederOptimizer(BaseOptimizer):
    """PromptBreeder 进化算法优化器。

    工作流程：
      1. **初始化种群**：以 initial_prompt 为种子，用 LLM 生成
         ``population_size - 1`` 个变体，组成初始种群。
      2. **评估适应度**：评估种群中所有 prompt 的得分。
      3. **选择精英**：按得分排序，取前 ``elite_size`` 个。
      4. **变异（mutation）**：对每个精英 prompt，以 ``mutation_rate``
         的概率生成变异子代。
      5. **交叉（crossover）**：随机配对精英，以 ``crossover_rate``
         的概率生成交叉子代。
      6. **更新种群**：将子代加入种群，淘汰低分个体，保持
         种群大小不超过 ``population_size``。
      7. **迭代**：重复步骤 2-6，直到达到 ``max_iterations``。
      8. **返回**：历史最优 prompt。

    配置参数（在 ``config`` 中传入）：
      - ``population_size`` (int)：种群大小，默认 10
      - ``mutation_rate`` (float)：变异概率，默认 0.5
      - ``crossover_rate`` (float)：交叉概率，默认 0.5
      - ``elite_ratio`` (float)：精英比例（0~1），默认 0.2
      - ``init_variants`` (int)：初始化时生成的变体数，默认 5
      - ``mutation_temperature`` (float)：变异时的 temperature，默认 0.7
    """

    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_provider=model_provider, evaluator=evaluator, config=config)
        self._population_size: int = self.config.get("population_size", 10)
        self._mutation_rate: float = self.config.get("mutation_rate", 0.5)
        self._crossover_rate: float = self.config.get("crossover_rate", 0.5)
        self._elite_ratio: float = self.config.get("elite_ratio", 0.2)
        self._init_variants: int = self.config.get("init_variants", 5)
        self._mutation_temperature: float = self.config.get("mutation_temperature", 0.7)

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "PromptBreeder"

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

        # 用局部 Random 实例而非全局 random.seed，避免污染同进程其他随机逻辑。
        rng = random.Random(42)

        # 1. 初始化种群
        logger.info("PromptBreeder: initializing population...")
        population = await self._init_population(
            initial_prompt=initial_prompt,
            dataset=dataset,
        )
        all_candidates.extend(population)

        logger.info(
            "PromptBreeder start: iterations={}, population_size={}",
            max_iterations,
            len(population),
        )

        # 2. 进化主循环
        for iteration in range(1, max_iterations + 1):
            self.on_iteration_start(iteration)

            # 评估种群中所有未评估的 prompt。
            # 用 `evaluated` 标记而非 `score is None` 判断——后者恒为 False
            # （score 默认 0.0，是 float 非 Optional），会导致子代从未被评估。
            unevaluated = [p for p in population if not p.evaluated]
            for candidate in unevaluated:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                # Evaluator.evaluate 内部已设置 candidate.score 和 candidate.evaluated=True，
                # 这里保留赋值仅为日志可读性。
                candidate.score = score
                logger.debug(
                    "PromptBreeder pop member '{}' score={:.4f}",
                    candidate.id[:8],
                    score,
                )

            # 按适应度排序
            population.sort(key=lambda c: c.score, reverse=True)
            elite_size = max(1, int(len(population) * self._elite_ratio))
            elites = population[:elite_size]

            logger.info(
                "PromptBreeder iteration {}/{}: population={}, best_score={:.4f}, "
                "avg_score={:.4f}",
                iteration,
                max_iterations,
                len(population),
                population[0].score or 0.0,
                sum(c.score or 0.0 for c in population) / max(len(population), 1),
            )

            # 3. 生成子代
            children: List[PromptCandidate] = []

            # 变异
            for elite in elites:
                if rng.random() < self._mutation_rate:
                    child = await self._mutate(elite, iteration)
                    children.append(child)

            # 交叉
            if len(elites) >= 2:
                rng.shuffle(elites)
                for i in range(0, len(elites) - 1, 2):
                    if rng.random() < self._crossover_rate:
                        child = await self._crossover(
                            elites[i], elites[i + 1], iteration
                        )
                        children.append(child)

            # 4. 更新种群
            population.extend(children)
            all_candidates.extend(children)

            # 淘汰低分个体，保持种群大小
            population.sort(key=lambda c: c.score, reverse=True)
            population = population[: self._population_size]

            # 记录历史
            history.append({
                "iteration": iteration,
                "population_size": len(population),
                "num_children": len(children),
                "best_score": population[0].score or 0.0,
                "avg_score": sum(c.score or 0.0 for c in population)
                / max(len(population), 1),
                "best_prompt_id": population[0].id,
            })

            self.on_iteration_end(iteration, children)

        # 5. 最终评估：主循环结束后，对 all_candidates 中所有未评估的候选
        #    （主要是最后一轮产生的子代）做一次评估，确保 best_prompt 选择公平、
        #    num_candidates_evaluated 统计准确。
        pending_final = [c for c in all_candidates if not c.evaluated]
        if pending_final:
            logger.info(
                "PromptBreeder: final evaluation of {} pending candidates",
                len(pending_final),
            )
            for candidate in pending_final:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score

        # 6. 选出全局最优
        best_prompt = max(all_candidates, key=lambda c: c.score or 0.0)

        total_cost: float = self.model_provider.total_cost_usd - cost_before
        elapsed = time.time() - start_time

        logger.info(
            "PromptBreeder done: best_score={:.4f}, elapsed={:.1f}s, "
            "total_cost=${:.4f}",
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
            num_candidates_evaluated=len([c for c in all_candidates if c.evaluated]),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _init_population(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
    ) -> List[PromptCandidate]:
        """初始化种群：initial_prompt + 若干变体。"""
        population = [initial_prompt]

        # 先评估初始 prompt
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        logger.info(f"PromptBreeder initial score: {initial_score:.4f}")

        # 生成变体
        num_variants = min(self._init_variants, self._population_size - 1)
        if num_variants > 0:
            variants = await self._generate_variants(
                base_prompt=initial_prompt,
                num_variants=num_variants,
            )
            population.extend(variants)

        return population

    async def _generate_variants(
        self,
        base_prompt: PromptCandidate,
        num_variants: int,
    ) -> List[PromptCandidate]:
        """用 LLM 生成初始 prompt 的若干变体。"""
        import re

        prompt_text = (
            f"Base prompt:\n```\n{base_prompt.instruction}\n```\n\n"
            f"Generate {num_variants} variant prompts that express the same intent "
            f"but with different wording. Wrap each in triple backticks."
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_INIT_VARIANT_SYSTEM,
            temperature=1.0,
            max_tokens=2048,
        )

        candidates: List[PromptCandidate] = []
        pattern = r"```(?:prompt)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches[:num_variants]:
                instruction = match.strip()
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={"source": "breeder_init", "iteration": 0},
                        )
                    )

        # 如果变体不足，用简单变体填充。
        # 变体标记放在 instruction 之前，避免破坏末尾输出引导（如 "\n类别："）。
        while len(candidates) < num_variants:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=f"(variant {len(candidates) + 1})\n{base_prompt.instruction}",
                    metadata={"source": "breeder_padding", "iteration": 0},
                )
            )

        return candidates[:num_variants]

    async def _mutate(
        self, parent: PromptCandidate, iteration: int
    ) -> PromptCandidate:
        """对单个 prompt 进行变异，返回子代。"""
        import re

        prompt_text = (
            f"Parent prompt:\n```\n{parent.instruction}\n```\n\n"
            f"Mutate this prompt to create an improved variant."
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_MUTATION_SYSTEM,
            temperature=self._mutation_temperature,
            max_tokens=2048,
        )

        pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        instruction = parent.instruction  # 兜底
        if matches:
            instruction = matches[0].strip()

        return PromptCandidate(
            id=str(uuid.uuid4()),
            instruction=instruction,
            metadata={
                "source": "breeder_mutation",
                "parent_id": parent.id,
                "iteration": iteration,
            },
        )

    async def _crossover(
        self, parent_a: PromptCandidate, parent_b: PromptCandidate, iteration: int
    ) -> PromptCandidate:
        """对两个 prompt 进行交叉，返回子代。"""
        import re

        prompt_text = (
            f"Parent A:\n```\n{parent_a.instruction}\n```\n\n"
            f"Parent B:\n```\n{parent_b.instruction}\n```\n\n"
            f"Combine these two prompts into a new, improved prompt."
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_CROSSOVER_SYSTEM,
            temperature=self._mutation_temperature,
            max_tokens=2048,
        )

        pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        instruction = parent_a.instruction  # 兜底
        if matches:
            instruction = matches[0].strip()

        return PromptCandidate(
            id=str(uuid.uuid4()),
            instruction=instruction,
            metadata={
                "source": "breeder_crossover",
                "parent_a_id": parent_a.id,
                "parent_b_id": parent_b.id,
                "iteration": iteration,
            },
        )
