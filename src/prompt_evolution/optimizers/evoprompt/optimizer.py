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
# P0-2 / P1-1 / P1-3 修复：中文化 + 弱化对输出格式的破坏 + 强制保留格式约束
_MUTATION_SYSTEM = (
    '你是一位资深 Prompt 变异算子。\n\n'
    '给定一个 Prompt，用以下变异方式之一改写：\n'
    '1. 更清晰地重新表述核心指令\n'
    '2. 增加具体的示例或约束\n'
    '3. 删除冗余或令人困惑的措辞\n'
    '4. 调整语气或角色以更好地匹配任务\n'
    '（注意：不得改变输出格式）\n\n'
    '重要约束（必须遵守）：\n'
    '- 变异后的 Prompt 必须原样保留字面占位符 {input}（仅一次），评估时会替换为实际用户输入，不得删除、改名或重复\n'
    '- 必须原样保留原 Prompt 的输出格式约束（如「只输出类别名称」）和末尾输出引导（如「类别：」）\n'
    '- 不得改变输出格式，不得添加「请逐步分析」「step by step」等推理引导\n'
    '- 输出必须是裸答案，不带任何前缀或解释\n\n'
    '只输出变异后的 Prompt，用三反引号包裹：\n'
    '```\n'
    '<变异后的 prompt>\n'
    '```\n'
)

# 交叉算子 system prompt（融合两条 prompt）
# P0-2 / P1-1 修复：中文化 + 强制保留格式约束
_CROSSOVER_SYSTEM = (
    '你是一个 Prompt 交叉算子。\n\n'
    '给定两个父代 Prompt A 和 B，生成 ONE 个智能融合两者优点的新 Prompt。\n\n'
    '指引：\n'
    '- 取任一父代中最清晰的指令\n'
    '- 合并输出格式约束（冲突时选更清晰的一方）\n'
    '- 保留两者有用的示例或约束\n'
    '- 结果应连贯，不是两半的拼接\n\n'
    '重要约束（必须遵守）：\n'
    '- 交叉后的 Prompt 必须原样保留字面占位符 {input}（仅一次），评估时会替换为实际用户输入，不得删除、改名或重复\n'
    '- 必须原样保留原 Prompt 的输出格式约束（如「只输出类别名称」）和末尾输出引导（如「类别：」）\n'
    '- 不得改变输出格式，不得添加「请逐步分析」「step by step」等推理引导\n\n'
    '只输出交叉后的 Prompt，用三反引号包裹：\n'
    '```\n'
    '<交叉后的 prompt>\n'
    '```\n'
)

# 初始化种群时的变体生成（多样性启动）
# P0-2 / P1-1 修复：中文化 + 强制保留格式约束
_INIT_VARIANT_SYSTEM = (
    '你正在为进化搜索生成多样化的 Prompt 变体。\n\n'
    '给定一个基础 Prompt，生成一个变体：\n'
    '- 表达相同的核心意图\n'
    '- 使用不同的措辞、结构或侧重点\n'
    '- 可增减示例或格式说明（但不得改变输出格式约束）\n\n'
    '重要约束（必须遵守）：\n'
    '- 变体 Prompt 必须原样保留字面占位符 {input}（仅一次），评估时会替换为实际用户输入，不得删除、改名或重复\n'
    '- 必须原样保留原 Prompt 的输出格式约束（如「只输出类别名称」）和末尾输出引导（如「类别：」）\n'
    '- 不得改变输出格式，不得添加「请逐步分析」「step by step」等推理引导\n\n'
    '只输出变体 Prompt，用三反引号包裹：\n'
    '```\n'
    '<变体 prompt>\n'
    '```\n'
)


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

        # 用局部 Random 实例而非全局 random.seed，避免污染同进程其他随机逻辑。
        rng = random.Random(42)

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

            # 评估种群中所有未评分的个体。
            # 用 `evaluated` 标记而非 `score is None` 判断——后者恒为 False
            # （score 默认 0.0，是 float 非 Optional），会导致子代从未被评估。
            for candidate in population:
                if not candidate.evaluated:
                    score = await self.evaluator.evaluate(
                        prompt=candidate,
                        dataset=dataset,
                        model_provider=self.model_provider,
                    )
                    # Evaluator.evaluate 内部已设置 candidate.score 和 candidate.evaluated=True
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
                if rng.random() < self._mutation_rate:
                    for _ in range(self._num_mutations):
                        child = await self._mutate(elite, iteration)
                        children.append(child)

            # 交叉（随机配对精英）
            if len(elites) >= 2:
                rng.shuffle(elites)
                for i in range(0, len(elites) - 1, 2):
                    if rng.random() < self._crossover_rate:
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

        # 3. 最终评估：主循环结束后，对 all_candidates 中所有未评估的候选
        #    （主要是最后一轮产生的子代）做一次评估，确保 best_prompt 选择公平、
        #    num_candidates_evaluated 统计准确。
        pending_final = [c for c in all_candidates if not c.evaluated]
        if pending_final:
            logger.info(
                "EVOPrompt: final evaluation of {} pending candidates",
                len(pending_final),
            )
            for candidate in pending_final:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score

        # 4. 选出全局最优
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
                [c for c in all_candidates if c.evaluated]
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
            f"基础 Prompt:\n```\n{base_prompt.instruction}\n```\n\n"
            f"生成 {num_variants} 个语义上不同的变体 Prompt。\n"
            f"每个必须用三反引号包裹：```<变体>```"
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

        # P2-3 修复：padding 不再加 [variant n] 前缀，直接复用 base_prompt.instruction，
        # 避免前缀污染破坏 prompt 开头的角色认知。
        while len(candidates) < num_variants:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=base_prompt.instruction,
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
            f"父代 Prompt:\n```\n{parent.instruction}\n```\n\n"
            f"变异这个 Prompt，生成 ONE 个改进的变体。"
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
            f"父代 A:\n```\n{parent_a.instruction}\n```\n\n"
            f"父代 B:\n```\n{parent_b.instruction}\n```\n\n"
            f"将这两个 Prompt 融合为 ONE 个改进的 Prompt。"
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
