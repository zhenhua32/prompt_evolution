"""OPRO (Optimization by PROmpting) 优化器实现。

论文：_Large Language Models as Optimizers_ (Google DeepMind, 2023)
arXiv: https://arxiv.org/abs/2309.03409

核心思想：
  用 LLM 作为优化器，通过自然语言描述优化任务和历史尝试结果，
  引导 LLM 逐步生成更优的 prompt，最终找到最大化任务准确率的指令。

算法流程：
  1. 初始化 meta-prompt（包含任务描述和空历史）
  2. 用 LLM 基于 meta-prompt 生成 N 个候选 prompt
  3. 在数据集上评估每个候选 prompt 的得分
  4. 将候选 prompt 及其得分加入 meta-prompt 的历史部分
  5. 重复步骤 2-4，直到达到最大迭代次数
  6. 返回得分最高的 prompt
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


# 默认 meta-prompt 模板（instruction 部分）
_DEFAULT_META_INSTRUCTION = """You are an expert prompt optimizer.

Your task is to analyze the prompts and their scores below, then generate NEW prompts that are likely to achieve higher scores.

Scoring: higher is better (0.0 = worst, 1.0 = perfect).

Guidelines for generating better prompts:
- Analyze what makes high-scoring prompts effective
- Keep prompts clear, specific, and actionable
- Avoid copying existing prompts verbatim — improve them
- Consider different angles: clarity, structure, examples, constraints
"""

# 默认 meta-prompt 的 few-shot 示例（放在 history 之前，帮助 LLM 理解格式）
_DEFAULT_META_FEWSHOT = """Here are some examples of good optimization:

Example history:
```
Score: 0.92
Prompt: "You are a helpful math tutor. Solve the problem step by step, showing all work."
```
→
```
Score: 0.95
Prompt: "You are a patient math tutor. Solve each problem step by step with clear explanations. Show your reasoning before giving the final answer."
```

Example history:
```
Score: 0.45
Prompt: "Answer the question."
```
→
```
Score: 0.68
Prompt: "Read the question carefully. Think step by step. Provide your final answer in the format: #### <answer>"
```
"""

# 候选 prompt 生成时的 system prompt
_CANDIDATE_GEN_SYSTEM = """You are an expert prompt engineer tasked with writing optimized instruction prompts.

For each prompt you generate:
- Make it clear, specific, and effective for the task
- Include concrete instructions on output format when helpful
- Vary your approach across candidates (some with examples, some with step-by-step reasoning, etc.)
- Output each candidate prompt wrapped in triple backticks: ```<prompt>```
- Generate the exact number of candidates requested
"""


class OPROOptimizer(BaseOptimizer):
    """OPRO (Optimization by PROmpting) 优化器。

    工作流程：
      1. **初始化**：从初始 prompt 开始，评估其得分
      2. **构造 meta-prompt**：包含任务描述 + 历史 prompt 及其得分
      3. **生成候选**：用 LLM 基于 meta-prompt 生成 N 个新候选 prompt
      4. **评估候选**：在数据集上评估每个候选的得分
      5. **更新历史**：将新候选及其得分加入 meta-prompt
      6. **迭代**：重复步骤 3-5，直到达到最大迭代次数
      7. **返回**：得分最高的 prompt

    配置参数（在 ``config`` 中传入）：
      - ``num_candidates`` (int)：每轮生成的候选 prompt 数，默认 8
      - ``meta_instruction`` (str)：meta-prompt 的 instruction 部分
      - ``use_fewshot`` (bool)：是否在 meta-prompt 中加入 few-shot 示例，默认 True
      - ``generation_temperature`` (float)：生成候选时的 temperature，默认 0.7
      - ``max_prompt_len`` (int)：候选 prompt 最大字符数（超过则截断），默认 2048
    """

    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_provider=model_provider, evaluator=evaluator, config=config)
        self._num_candidates: int = self.config.get("num_candidates", 8)
        self._meta_instruction: str = self.config.get(
            "meta_instruction", _DEFAULT_META_INSTRUCTION
        )
        self._use_fewshot: bool = self.config.get("use_fewshot", True)
        self._generation_temperature: float = self.config.get("generation_temperature", 0.7)
        self._max_prompt_len: int = self.config.get("max_prompt_len", 2048)

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "OPRO"

    async def optimize(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> OptimizationResult:
        """运行 OPRO 优化，返回最优 prompt。"""
        import time

        start_time = time.time()
        cost_before: float = self.model_provider.total_cost_usd
        all_candidates: List[PromptCandidate] = []
        history: List[Dict[str, Any]] = []

        # 评估初始 prompt
        logger.info("OPRO: evaluating initial prompt...")
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        all_candidates.append(initial_prompt)

        logger.info(f"Initial prompt score: {initial_score:.4f}")

        # 历史记录：list of (prompt_text, score)
        scored_history: List[tuple[str, float]] = [
            (initial_prompt.instruction, initial_score)
        ]

        # 获取任务描述（从数据集前几条提取）
        task_description = self._build_task_description(dataset)

        logger.info(
            "OPRO optimizer start: iterations={}, candidates_per_iter={}",
            max_iterations,
            self._num_candidates,
        )

        for iteration in range(1, max_iterations + 1):
            self.on_iteration_start(iteration)

            # 1. 构造 meta-prompt
            meta_prompt = self._build_meta_prompt(
                task_description=task_description,
                scored_history=scored_history,
                iteration=iteration,
            )

            # 2. 用 LLM 生成候选 prompt
            candidates = await self._generate_candidates(
                meta_prompt=meta_prompt,
                iteration=iteration,
            )

            # 3. 评估所有候选
            for candidate in candidates:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score
                logger.debug(
                    "OPRO candidate '{}' score={:.4f}",
                    candidate.id[:8],
                    score,
                )

            all_candidates.extend(candidates)

            # 4. 更新历史（只保留得分 > 0 的候选，避免污染 meta-prompt）
            for candidate in candidates:
                if candidate.score > 0:
                    scored_history.append(
                        (candidate.instruction, candidate.score)
                    )

            # 5. 记录历史（按得分排序，方便查看）
            best_this_iter = max(candidates, key=lambda c: c.score)
            history.append({
                "iteration": iteration,
                "num_candidates": len(candidates),
                "best_score": best_this_iter.score,
                "best_prompt_id": best_this_iter.id,
                "all_scores": [c.score for c in candidates],
            })
            logger.info(
                "OPRO iteration {}/{}: best_score={:.4f}",
                iteration,
                max_iterations,
                best_this_iter.score,
            )

            self.on_iteration_end(iteration, candidates)

        # 6. 选出全局最优
        best_prompt = max(all_candidates, key=lambda c: c.score)

        # 计算本次优化产生的费用
        total_cost: float = self.model_provider.total_cost_usd - cost_before

        elapsed = time.time() - start_time
        logger.info(
            "OPRO done: best_score={:.4f}, elapsed={:.1f}s, total_cost=${:.4f}",
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
            num_candidates_evaluated=len(all_candidates),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_task_description(self, dataset: List[Dict[str, Any]]) -> str:
        """从数据集前几条样本构建任务描述。"""
        desc_parts: List[str] = [
            "Task: Given an input, generate the expected output.\n",
            "Below are example input-output pairs from the dataset:\n",
        ]
        for i, item in enumerate(dataset[:3]):
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))
            desc_parts.append(f"  Example {i + 1}:\n")
            desc_parts.append(f"    Input: {inp}\n")
            desc_parts.append(f"    Output: {tgt}\n")
        return "".join(desc_parts)

    def _build_meta_prompt(
        self,
        task_description: str,
        scored_history: List[tuple[str, float]],
        iteration: int,
    ) -> str:
        """构造用于生成候选 prompt 的 meta-prompt。

        Meta-prompt 结构：
          - 任务描述
          - （可选）few-shot 示例
          - 历史 prompt 及其得分（按得分从高到低排序）
          - 生成新候选的指令
        """
        parts: List[str] = []

        # 1. Instruction
        parts.append(self._meta_instruction.strip())
        parts.append("\n\n")

        # 2. 任务描述
        parts.append(task_description)
        parts.append("\n")

        # 3. Few-shot 示例（可选）
        if self._use_fewshot:
            parts.append(_DEFAULT_META_FEWSHOT)
            parts.append("\n")

        # 4. 历史 prompt 及其得分（按得分从高到低排序）
        sorted_history = sorted(scored_history, key=lambda x: x[1], reverse=True)
        # 只保留最近最多 20 条历史，避免超出 context window
        recent_history = sorted_history[-20:]

        parts.append("Here are the prompts tried so far and their scores:\n\n")
        for i, (prompt_text, score) in enumerate(recent_history):
            parts.append(f"```\nScore: {score:.4f}\nPrompt: {prompt_text}\n```\n\n")

        # 5. 生成指令
        parts.append(
            f"Based on the above, generate {self._num_candidates} NEW and IMPROVED prompts.\n"
        )
        parts.append(
            f"Each prompt should be different and aim to score higher than {recent_history[-1][1]:.4f}.\n"
        )
        parts.append(
            f"Output each candidate wrapped in triple backticks: ```<prompt>```\n"
        )
        parts.append(f"Number them 1 to {self._num_candidates}.\n")

        return "".join(parts)

    async def _generate_candidates(
        self,
        meta_prompt: str,
        iteration: int,
    ) -> List[PromptCandidate]:
        """用 LLM 基于 meta-prompt 生成候选 prompt。"""
        import re

        response = await self.model_provider.generate(
            prompt=meta_prompt,
            system_prompt=_CANDIDATE_GEN_SYSTEM,
            temperature=self._generation_temperature,
            max_tokens=2048,
        )

        # 解析 LLM 输出，提取候选 prompt
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
                        instruction=line[: self._max_prompt_len],
                        metadata={"source": "opro_fallback", "iteration": iteration},
                    )
                )
        else:
            for match in matches[: self._num_candidates]:
                instruction = match.strip()[: self._max_prompt_len]
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={"source": "opro", "iteration": iteration},
                        )
                    )

        # 如果解析出的候选不足，用初始 prompt 填充
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=(
                        f"Improved version of previous prompt. "
                        f"(iteration {iteration}, fill {len(candidates) + 1})"
                    ),
                    metadata={"source": "opro_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]
