"""EVOPrompt 优化器实现。

论文：_EVOPrompt: Evolutionary Prompt Optimization for Enhancing
Large Language Model Performance Across Tasks_
(ACM SIGIR, 2024)
DOI: https://doi.org/10.1145/3626772.3657932

核心思想：
  将 Prompt 视为「基因」，用进化算法在多轮迭代中逐步优化：
  1. 初始化种群（初始 prompt + 若干变体）
  2. 评估种群中每个 prompt 的适应度（= 任务得分）
  3. 选择（Selection）：按适应度排序，保留精英
  4. 变异（Mutation）：对精英 prompt 做小幅改写
  5. 交叉（Crossover）：将两条精英 prompt 融合为一条新 prompt
  6. 用变异 + 交叉产生子代，补充进种群
  7. 淘汰低适应度个体，保持种群大小
  8. 重复 2-7，直到达到最大迭代轮数
  9. 返回历史最优 prompt

与 PromptBreeder 的区别：
  - EVOPrompt 使用更结构化的变异算子（插入/删除/改写指令段）
  - 支持 ``num_mutations`` 控制每条精英产生几个子代
  - 内置早停：连续 ``early_stop_patience`` 轮无提升时停止
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


# 变异算子 system prompt（针对性改写）
_MUTATION_SYSTEM = """\
You are an expert prompt mutation operator.

Given a prompt, rewrite it with ONE of the following mutations:
1. Rephrase the core instruction more clearly
2. Add a concrete example or constraint
3. Remove redundant or confusing wording
4. Make the output format specification more explicit
5. Adjust the tone or persona to better match the task

CRITICAL: The mutated prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the mutated prompt, wrapped in triple backticks:
```
<mutated prompt>
```
"""

# 交叉算子 system prompt（融合两条 prompt）
_CROSSOVER_SYSTEM = """\
You are a prompt crossover operator.

Given two parent prompts A and B, produce ONE new prompt that
intelligently combines the best parts of both.

Guidelines:
- Keep the clearest instruction from either parent
- Merge output format specs (resolve conflicts by picking the clearer one)
- Retain useful examples or constraints from both
- The result should be coherent, not a拼接 of two halves

CRITICAL: The crossed-over prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the crossed-over prompt, wrapped in triple backticks:
```
<crossed-over prompt>
```
"""

# 初始化种群时的变体生成（多样性启动）
_INIT_VARIANT_SYSTEM = """\
You are generating diverse prompt variants for evolutionary search.

Given a base prompt, produce a variant that:
- Expresses the same core intent
- Uses different wording, structure, or emphasis
- May add / remove examples or formatting instructions

CRITICAL: The variant prompt MUST contain the literal placeholder {input} exactly once. This placeholder is replaced with the actual user input at evaluation time. Do NOT remove, rename, or duplicate it. Keep it where the user's input should go (typically near the end, before the output cue).

Output ONLY the variant prompt, wrapped in triple backticks:
```
<variant prompt>
```
"""


class EVOPromptOptimizer(BaseOptimizer):
    """EVOPrompt 进化算法优化器。

    工作流程（与论文算法对应）：
      1. **初始化**：以 ``initial_prompt`` 为种子，用 LLM 生成
         ``population_size - 1`` 个语义不同的变体，构成初始种群。
      2. **评估**：对种群中所有 prompt 打分（在数据集上的准确率）。
      3. **选择**：按分数降序排列，取前 ``elite_size`` 个精英。
      4. **变异**：对每个精英，以 ``mutation_rate`` 概率生成
         ``num_mutations`` 个变异子代。
      5. **交叉**：随机配对精英，以 ``crossover_rate`` 概率生成
         交叉子代。
      6. **更新种群**：将子代加入种群，按分数淘汰低分个体，
         保持种群大小 ≤ ``population_size``。
      7. **迭代**：重复 2–6，直到 ``max_iterations`` 轮或早停触发。
      8. **返回**：整个搜索过程中得分最高的 prompt。

    配置参数（在 ``config`` 中传入）：
      - ``population_size`` (int)：种群大小，默认 12
      - ``elite_ratio`` (float)：精英比例（0~1），默认 0.3
      - ``mutation_rate`` (float)：每条精英被选中变异的概率，默认 0.6
      - ``crossover_rate`` (float)：每对精英被选中交叉的概率，默认 0.4
      - ``num_mutations`` (int)：每条精英变异时产生几个子代，默认 2
      - ``mutation_temperature`` (float)：变异时的 sampling temperature，默认 0.8
      - ``early_stop_patience`` (int)：连续多少轮无提升后早停，默认 4
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
        self._population_size: int = self.config.get("population_size", 12)
        self._elite_ratio: float = self.config.get("elite_ratio", 0.3)
        self._mutation_rate: float = self.config.get("mutation_rate", 0.6)
        self._crossover_rate: float = self.config.get("crossover_rate", 0.4)
        self._num_mutations: int = self.config.get("num_mutations", 2)
        self._mutation_temperature: float = self.config.get(
            "mutation_temperature", 0.8
        )
        self._early_stop_patience: int = self.config.get(
            "early_stop_patience", 4
        )

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "EVOPrompt"

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
        best_score_so_far: float = 0.0
        patience_counter: int = 0

        random.seed(42)

        # 1. 初始化种群
        logger.info("EVOPrompt: initializing population...")
        population = await self._init_population(initial_prompt, dataset)
        all_candidates.extend(population)

        logger.info(
            "EVOPrompt start: iterations={}, population_size={}",
            max_iterations,
            len(population),
        )

        # 2. 进化主循环
        for iteration in range(1, max_iterations + 1):
            self.on_iteration_start(iteration)

            # 评估种群中所有未评分的个体
            for candidate in population:
                if candidate.score is None:
                    score = await self.evaluator.evaluate(
                        prompt=candidate,
                        dataset=dataset,
                        model_provider=self.model_provider,
                    )
                    candidate.score = score

            # 按适应度降序排列
            population.sort(key=lambda c: c.score if c.score is not None else -1.0, reverse=True)

            best_this_iter = population[0]
            best_score_this_iter = best_this_iter.score or 0.0

            logger.info(
                "EVOPrompt iteration {}/{}: best={:.4f}, avg={:.4f}, "
                "population={}",
                iteration,
                max_iterations,
                best_score_this_iter,
                sum(c.score or 0.0 for c in population) / max(len(population), 1),
                len(population),
            )

            # 早停检查
            if best_score_this_iter > best_score_so_far + 1e-4:
                best_score_so_far = best_score_this_iter
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._early_stop_patience:
                    logger.info(
                        "EVOPrompt early stop at iteration {} (no improvement in {} rounds)",
                        iteration,
                        self._early_stop_patience,
                    )
                    break

            # 选择精英
            elite_size = max(1, int(len(population) * self._elite_ratio))
            elites = population[:elite_size]

            # 生成子代
            children: List[PromptCandidate] = []

            # 变异
            for elite in elites:
                if random.random() < self._mutation_rate:
                    for _ in range(self._num_mutations):
                        child = await self._mutate(elite, iteration)
                        children.append(child)

            # 交叉（随机配对精英）
            if len(elites) >= 2:
                random.shuffle(elites)
                for i in range(0, len(elites) - 1, 2):
                    if random.random() < self._crossover_rate:
                        child = await self._crossover(
                            elites[i], elites[i + 1], iteration
                        )
                        children.append(child)

            # 更新种群
            population.extend(children)
            all_candidates.extend(children)

            # 淘汰低分个体，保持种群大小
            population.sort(
                key=lambda c: c.score if c.score is not None else -1.0,
                reverse=True,
            )
            population = population[: self._population_size]

            # 记录历史
            history.append({
                "iteration": iteration,
                "population_size": len(population),
                "num_children": len(children),
                "best_score": best_this_iter.score,
                "best_prompt_id": best_this_iter.id,
                "avg_score": sum(c.score or 0.0 for c in population)
                / max(len(population), 1),
            })

            self.on_iteration_end(iteration, children)

        # 3. 选出全局最优
        best_prompt = max(
            all_candidates, key=lambda c: c.score if c.score is not None else -1.0
        )

        total_cost: float = self.model_provider.total_cost_usd - cost_before
        elapsed = time.time() - start_time

        logger.info(
            "EVOPrompt done: best_score={:.4f}, elapsed={:.1f}s, cost=${:.4f}",
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

    async def _init_population(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
    ) -> List[PromptCandidate]:
        """初始化种群：initial_prompt + 若干 LLM 生成的变体。"""
        population = [initial_prompt]

        # 评估初始 prompt
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        logger.info("EVOPrompt initial score: {:.4f}", initial_score)

        num_variants = min(
            self._num_mutations * 2,
            self._population_size - 1,
        )
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
        """用 LLM 生成初始 prompt 的若干语义变体。"""
        import re

        prompt_text = (
            f"Base prompt:\n```\n{base_prompt.instruction}\n```\n\n"
            f"Generate {num_variants} semantically DIFFERENT variant prompts.\n"
            f"Each must wrap in triple backticks: ```<variant>```"
        )

        response = await self.model_provider.generate(
            prompt=prompt_text,
            system_prompt=_INIT_VARIANT_SYSTEM,
            temperature=1.0,
            max_tokens=2048,
        )

        candidates: List[PromptCandidate] = []
        pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            for match in matches[:num_variants]:
                instruction = match.strip()
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={"source": "evoprompt_init", "iteration": 0},
                        )
                    )

        # 不足时用简单变体填充。
        # 变体标记放在 instruction 之前，避免破坏末尾输出引导（如 "\n类别："）。
        while len(candidates) < num_variants:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=f"[variant {len(candidates) + 1}]\n{base_prompt.instruction}",
                    metadata={"source": "evoprompt_pad", "iteration": 0},
                )
            )

        return candidates[:num_variants]

    async def _mutate(
        self, parent: PromptCandidate, iteration: int
    ) -> PromptCandidate:
        """对单个 prompt 执行变异算子，返回子代。"""
        import re

        prompt_text = (
            f"Parent prompt:\n```\n{parent.instruction}\n```\n\n"
            f"Mutate this prompt to create ONE improved variant."
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
                "source": "evoprompt_mutation",
                "parent_id": parent.id,
                "iteration": iteration,
            },
        )

    async def _crossover(
        self, parent_a: PromptCandidate, parent_b: PromptCandidate, iteration: int
    ) -> PromptCandidate:
        """对两个 prompt 执行交叉算子，返回子代。"""
        import re

        prompt_text = (
            f"Parent A:\n```\n{parent_a.instruction}\n```\n\n"
            f"Parent B:\n```\n{parent_b.instruction}\n```\n\n"
            f"Combine these two prompts into ONE improved prompt."
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
                "source": "evoprompt_crossover",
                "parent_a_id": parent_a.id,
                "parent_b_id": parent_b.id,
                "iteration": iteration,
            },
        )
