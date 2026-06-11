"""DSPy 风格优化器实现。

参考 DSPy 的 teleprompter 设计理念：
  - Bootstrap：从数据集中采样生成 few-shot 示例
  - Propose：基于示例用 LLM 提出新的 instruction 候选
  - Evaluate：在数据集上评估每个候选
  - Select：选择得分最高的 prompt

与 DSPy 官方库的区别：
  本实现是轻量级的，不依赖 dspy 库，而是复用项目的
  BaseOptimizer / Evaluator / LiteLLMProvider 体系。
  核心思路与 DSPy 的 COPRO  teleprompter 一致。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from prompt_evolution.core.base import BaseOptimizer
from prompt_evolution.core.models import OptimizationResult, PromptCandidate


# 候选指令生成 system prompt
_PROPOSE_SYSTEM = """You are an expert prompt engineer optimizing instructions for an LLM task.

You will be given:
- A task description
- A few-shot example (input-output pairs)
- The current best instruction so far

Your job: Propose NEW instructions that could improve performance.
Focus on clarity, specificity, and alignment with the examples.

Output each candidate instruction wrapped in triple backticks:
```
<instruction>
```

Generate exactly the number of candidates requested.
"""

# Bootstrap few-shot 构建 prompt
_BOOTSTRAP_SYSTEM = """You are helping build a high-quality dataset.

Given an input and the expected output, write a clear instruction that tells an LLM how to produce the output from the input.

Output ONLY the instruction text, no explanation.
"""


class DSPyOptimizer(BaseOptimizer):
    """DSPy 风格优化器（轻量级，无 dspy 依赖）。

    算法流程（参考 COPRO）：
      1. **Bootstrap**：对每个训练样本，用当前 prompt 生成预测，
         收集正确的输入-输出对作为 few-shot 示例。
      2. **Propose**：将 task description + few-shot 示例喂给 LLM，
         生成 N 条新的 instruction 候选。
      3. **Evaluate**：在数据集上评估每条候选 instruction。
      4. **Select**：保留得分最高的 instruction，进入下一轮。
      5. **迭代**：重复步骤 1-4，直到达到 max_iterations。

    配置参数（在 ``config`` 中传入）：
      - ``num_candidates`` (int)：每轮生成的候选指令数，默认 6
      - ``bootstrap_samples`` (int)：bootstrap 阶段采样样本数，默认 8
      - ``propose_temperature`` (float)：生成候选时的 temperature，默认 0.7
      - ``bootstrap_max_attempts`` (int)：每个样本最多尝试次数，默认 3
    """

    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_provider=model_provider, evaluator=evaluator, config=config)
        self._num_candidates: int = self.config.get("num_candidates", 6)
        self._bootstrap_samples: int = self.config.get("bootstrap_samples", 8)
        self._propose_temperature: float = self.config.get("propose_temperature", 0.7)
        self._bootstrap_max_attempts: int = self.config.get("bootstrap_max_attempts", 3)

    # ------------------------------------------------------------------
    # BaseOptimizer 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "DSPy"

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
        logger.info("DSPy: evaluating initial prompt...")
        initial_score = await self.evaluator.evaluate(
            prompt=initial_prompt,
            dataset=dataset,
            model_provider=self.model_provider,
        )
        initial_prompt.score = initial_score
        all_candidates.append(initial_prompt)

        logger.info(f"DSPy initial score: {initial_score:.4f}")

        current_best = initial_prompt
        few_shots: List[Dict[str, str]] = []  # bootstrap 收集的示例

        for iteration in range(1, max_iterations + 1):
            self.on_iteration_start(iteration)

            # 1. Bootstrap：收集正确预测的 few-shot 示例
            few_shots = await self._bootstrap(
                prompt=current_best,
                dataset=dataset,
                num_samples=self._bootstrap_samples,
            )
            logger.info(
                "DSPy iteration {}/{}: bootstrapped {} examples",
                iteration,
                max_iterations,
                len(few_shots),
            )

            # 2. Propose：生成新候选 instruction
            candidates = await self._propose(
                task_description=self._build_task_description(dataset),
                few_shots=few_shots,
                current_best_instruction=current_best.instruction,
                iteration=iteration,
            )

            # 3. Evaluate
            for candidate in candidates:
                score = await self.evaluator.evaluate(
                    prompt=candidate,
                    dataset=dataset,
                    model_provider=self.model_provider,
                )
                candidate.score = score
                logger.debug(
                    "DSPy candidate '{}' score={:.4f}",
                    candidate.id[:8],
                    score,
                )
            all_candidates.extend(candidates)

            # 4. Select：更新 current_best
            iteration_best = max(candidates, key=lambda c: c.score)
            if iteration_best.score > current_best.score:
                current_best = iteration_best
                logger.info(
                    "DSPy iteration {}/{}: new best score={:.4f} (improved from {:.4f})",
                    iteration,
                    max_iterations,
                    current_best.score,
                    max(current_best.score - 0.001, 0),  # rough prev
                )
            else:
                logger.info(
                    "DSPy iteration {}/{}: no improvement, best={:.4f}",
                    iteration,
                    max_iterations,
                    current_best.score,
                )

            history.append({
                "iteration": iteration,
                "num_candidates": len(candidates),
                "best_score": iteration_best.score,
                "best_prompt_id": iteration_best.id,
                "all_scores": [c.score for c in candidates],
                "num_few_shots": len(few_shots),
            })

            self.on_iteration_end(iteration, candidates)

            # 早停：如果连续若干轮无提升可提前退出（简单实现：暂不早停）
            _scores = [h["best_score"] for h in history]
            if len(_scores) >= 3 and all(
                abs(_scores[-i] - _scores[-i - 1]) < 0.001 for i in range(1, 4)
            ):
                logger.info("DSPy early stopping: no improvement in 3 iterations")
                break

        # 5. 最终结果
        best_prompt = max(all_candidates, key=lambda c: c.score)
        total_cost: float = self.model_provider.total_cost_usd - cost_before
        elapsed = time.time() - start_time

        logger.info(
            "DSPy done: best_score={:.4f}, elapsed={:.1f}s, total_cost=${:.4f}",
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
        """从数据集前 3 条构建任务描述。"""
        parts: List[str] = [
            "Task: Given an input, generate the expected output.\n",
            "Example input-output pairs:\n",
        ]
        for i, item in enumerate(dataset[:3]):
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))
            parts.append(f"  Example {i + 1}: Input=`{inp}` → Output=`{tgt}`\n")
        return "".join(parts)

    async def _bootstrap(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, str]]:
        """Bootstrap 阶段：用当前 prompt 在数据集上推理，收集正确样本作为 few-shot。

        返回 list of ``{"input": ..., "output": ...}``。
        """
        import random

        random.seed(42)
        sampled = random.sample(dataset, min(num_samples, len(dataset)))

        few_shots: List[Dict[str, str]] = []
        for item in sampled:
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))

            # 用当前 prompt 推理
            try:
                messages = [
                    {"role": "system", "content": prompt.instruction},
                    {"role": "user", "content": inp},
                ]
                response = await self.model_provider.agenerate(messages=messages, temperature=0.0)
                pred = response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("DSPy bootstrap prediction failed: {exc}")
                pred = ""

            # 简单匹配：完全匹配或包含正确答案即视为正确（可配置 metric）
            if pred == tgt or tgt in pred:
                few_shots.append({"input": inp, "output": tgt})

        return few_shots

    async def _propose(
        self,
        task_description: str,
        few_shots: List[Dict[str, str]],
        current_best_instruction: str,
        iteration: int,
    ) -> List[PromptCandidate]:
        """Propose 阶段：基于 task + few-shots + 当前最优 instruction，生成新候选。"""
        import re

        # 构造 propose prompt
        parts: List[str] = []
        parts.append("## Task Description\n")
        parts.append(task_description)
        parts.append("\n")

        if few_shots:
            parts.append("## Few-Shot Examples (correct input-output pairs)\n")
            for i, ex in enumerate(few_shots[:5]):
                parts.append(f"  Example {i + 1}:\n")
                parts.append(f"    Input: {ex['input']}\n")
                parts.append(f"    Output: {ex['output']}\n")
            parts.append("\n")

        parts.append("## Current Best Instruction\n")
        parts.append(f"```\n{current_best_instruction}\n```\n\n")

        parts.append(
            f"## Your Job\n"
            f"Propose {self._num_candidates} NEW and IMPROVED instructions "
            f"based on the task and examples above.\n"
            f"Each should be different and aim to improve performance.\n"
            f"Wrap each candidate in triple backticks: ```<instruction>```\n"
        )

        propose_prompt = "".join(parts)

        response = await self.model_provider.generate(
            prompt=propose_prompt,
            system_prompt=_PROPOSE_SYSTEM,
            temperature=self._propose_temperature,
            max_tokens=2048,
        )

        # 解析候选
        candidates: List[PromptCandidate] = []
        pattern = r"```(?:prompt)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches[: self._num_candidates]:
                instruction = match.strip()
                if instruction:
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=instruction,
                            metadata={"source": "dspy_propose", "iteration": iteration},
                        )
                    )

        # 兜底：如果解析失败，按行提取
        if not candidates:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            for line in lines[: self._num_candidates]:
                if len(line) > 10:  # 过滤太短的行
                    candidates.append(
                        PromptCandidate(
                            id=str(uuid.uuid4()),
                            instruction=line,
                            metadata={"source": "dspy_fallback", "iteration": iteration},
                        )
                    )

        # 填充不足
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=current_best_instruction + f" (variation {len(candidates) + 1})",
                    metadata={"source": "dspy_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]
