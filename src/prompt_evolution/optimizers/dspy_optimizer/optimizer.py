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
# P0-2 / P1-1 修复：中文化 + 强制保留输出格式约束
_PROPOSE_SYSTEM = (
    '你是一位资深 Prompt 工程师，负责为 LLM 任务优化指令 Prompt。\n\n'
    '你将获得：\n'
    '- 任务描述\n'
    '- 若干 few-shot 示例（输入-输出对）\n'
    '- 当前最优指令\n\n'
    '你的任务：提出能够提升性能的新指令。聚焦于清晰度、具体性、与示例的对齐。\n\n'
    '重要约束（必须遵守）：\n'
    '- 每条指令必须原样保留字面占位符 {input}（仅一次），评估时会替换为实际用户输入，不得删除、改名或重复\n'
    '- 必须原样保留原 Prompt 的输出格式约束（如「只输出类别名称」）和末尾输出引导（如「类别：」）\n'
    '- 不得添加「请逐步分析」「step by step」等推理引导，输出必须是裸答案\n'
    '- 每条候选用三反引号包裹：\n'
    '```\n'
    '<instruction>\n'
    '```\n\n'
    '生成指定数量的候选。\n'
)

# Bootstrap few-shot 构建 prompt
_BOOTSTRAP_SYSTEM = (
    '你正在帮助构建高质量数据集。\n\n'
    '给定输入和期望输出，撰写一条清晰的指令，告诉 LLM 如何从输入产生该输出。\n\n'
    '只输出指令文本，不要解释。\n'
)


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

            # 早停：连续 3 轮相邻最优分数差均 < 0.001 时退出。
            # 旧实现 `range(1, 4)` 访问 `_scores[-1].._scores[-4]`，
            # len==3 时 `_scores[-4]` 越界回绕到 `_scores[-1]`（逻辑错误）。
            _scores = [h["best_score"] for h in history]
            if len(_scores) >= 4 and all(
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
        """从数据集构建任务描述（P2-2 修复：分层覆盖多类别）。"""
        parts: List[str] = [
            "任务：给定输入，生成对应的期望输出。\n",
            "示例输入-输出对：\n",
        ]
        # P2-2 修复：旧实现用 dataset[:3]，本项目数据集前 3 条全是"财经"类。
        # 现按 target 分组轮询采样，覆盖尽量多的类别。
        from collections import OrderedDict

        groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for item in dataset:
            tgt = str(item.get("target", item.get("answer", "")))
            groups.setdefault(tgt, []).append(item)

        sampled: List[Dict[str, Any]] = []
        for tgt, items in groups.items():
            if items:
                sampled.append(items[0])
            if len(sampled) >= 6:
                break

        for i, item in enumerate(sampled):
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))
            parts.append(f"  示例 {i + 1}: 输入=`{inp}` → 输出=`{tgt}`\n")
        return "".join(parts)

    async def _bootstrap(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, str]]:
        """Bootstrap 阶段：用当前 prompt 在数据集上推理，收集正确样本作为 few-shot。

        返回 list of ``{"input": ..., "output": ...}``。

        修复要点（与 evaluator 评估链路对齐）：
        1. 不再调用不存在的 ``agenerate(messages=...)``，改用
           ``generate(prompt=..., system_prompt=None, ...)``，与
           ``BaseModelProvider`` / ``LiteLLMProvider`` 接口一致。
        2. 占位符替换：若 ``instruction`` 含 ``{input}``，走
           ``instruction.replace("{input}", inp)``，与 evaluator 和
           baseline 完全对齐 —— 这样 bootstrap 收集的"正确样本"
           才能真实反映当前 prompt 在评估链路下的表现。
        3. 不再传 ``system_prompt=prompt.instruction``，避免
           instruction 在 system 和 user 两条消息里重复（双发会
           破坏输出格式引导）。
        """
        import random

        # 用局部 Random 实例而非全局 random.seed，避免污染同进程其他随机逻辑。
        rng = random.Random(42)
        sampled = rng.sample(dataset, min(num_samples, len(dataset)))

        instruction = prompt.instruction
        has_placeholder = "{input}" in instruction

        few_shots: List[Dict[str, str]] = []
        for item in sampled:
            inp = item.get("input", item.get("question", ""))
            tgt = item.get("target", item.get("answer", ""))

            # 与 evaluator 评估链路对齐：占位符替换或兜底拼接
            if has_placeholder:
                full_prompt = instruction.replace("{input}", inp)
            else:
                full_prompt = f"{instruction}\n\n用户输入：{inp}"

            try:
                response = await self.model_provider.generate(
                    prompt=full_prompt,
                    system_prompt=None,
                    temperature=0.0,
                    max_tokens=512,
                )
                pred = response.strip()
            except Exception as exc:
                logger.warning("DSPy bootstrap prediction failed: {}", exc)
                pred = ""

            # P2-4 修复：旧实现 `pred == tgt or tgt in pred` 过松——
            # 若 pred="该新闻属于财经类别"、tgt="财经"，`tgt in pred` 为真，
            # 会把格式错误的样本当作"正确 few-shot"收集，喂给 LLM 生成新候选时
            # 强化错误输出格式，形成恶性循环。现改为严格相等（与 AccuracyMetric 一致）。
            if pred.strip().lower() == tgt.strip().lower():
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
        parts.append("## 任务描述\n")
        parts.append(task_description)
        parts.append("\n")

        if few_shots:
            parts.append("## Few-Shot 示例（正确的输入-输出对）\n")
            for i, ex in enumerate(few_shots[:5]):
                parts.append(f"  示例 {i + 1}:\n")
                parts.append(f"    输入: {ex['input']}\n")
                parts.append(f"    输出: {ex['output']}\n")
            parts.append("\n")

        parts.append("## 当前最优指令\n")
        parts.append(f"```\n{current_best_instruction}\n```\n\n")

        parts.append(
            f"## 你的任务\n"
            f"基于上述任务和示例，提出 {self._num_candidates} 条新的、更优的指令。\n"
            f"每条应有差异，并力求提升性能。\n"
            f"每条候选用三反引号包裹：```<instruction>```\n"
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

        # P2-3 修复：padding 不再加 (variation n) 前缀，直接复用 current_best_instruction，
        # 避免前缀污染破坏 prompt 开头的角色认知。
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=current_best_instruction,
                    metadata={"source": "dspy_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]
