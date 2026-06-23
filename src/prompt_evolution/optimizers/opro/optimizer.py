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
# P0-2 / P1-1 修复：中文化 + 引导 LLM 保留输出格式约束
_DEFAULT_META_INSTRUCTION = """你是一位资深 Prompt 优化专家。

你的任务是分析下方已尝试的 Prompt 及其得分，然后生成更可能获得更高得分的新 Prompt。

评分规则：越高越好（0.0 = 最差，1.0 = 完美）。

生成更优 Prompt 的要点：
- 分析高分 Prompt 为什么有效
- 保持 Prompt 清晰、具体、可执行
- 不要照抄已有 Prompt，要真正改进
- 可从清晰度、结构、约束、示例等角度切入
- 重要：不得破坏原 Prompt 的输出格式约束和末尾输出引导
"""

# 默认 meta-prompt 的 few-shot 示例（放在 history 之前，帮助 LLM 理解格式）
# P1-1 修复：旧实现是英文数学辅导示例（"step by step"），与中文分类任务完全无关
# 且会诱导 LLM 生成 CoT 风格 prompt，破坏裸输出要求。现替换为同领域分类示例。
_DEFAULT_META_FEWSHOT = (
    '以下是优化的示例（中文分类任务）：\n\n'
    '示例历史：\n'
    '```\n'
    '得分: 0.83\n'
    'Prompt: "你是新闻分类专家。根据新闻标题判断属于以下哪个类别：科技、股票、体育、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座。只输出类别名称，不要输出任何其他内容。新闻标题：\\"{input}\\" 类别："\n'
    '```\n\n'
    '```\n'
    '得分: 0.88\n'
    'Prompt: "你是资深新闻编辑。请从下列14个类别中选出最贴切的一个：科技、股票、体育、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座。仅输出类别名，不要解释。新闻标题：\\"{input}\\" 类别："\n'
    '```\n\n'
    '反例（得分低，格式被破坏）：\n'
    '```\n'
    '得分: 0.45\n'
    'Prompt: "请分析这则新闻并给出分类。新闻标题：\\"{input}\\""\n'
    '```\n'
    '（错误原因：没有"只输出类别名"约束，没有末尾"类别："引导，模型会输出长篇分析）\n'
)

# 候选 prompt 生成时的 system prompt
# P0-2 修复：强制保留 {input} 占位符 + 原始输出格式约束 + 末尾输出引导
_CANDIDATE_GEN_SYSTEM = (
    '你是资深 Prompt 工程师，负责撰写优化后的指令 Prompt。\n\n'
    '每条 Prompt 必须遵守：\n'
    '- 清晰、具体、对任务有效\n'
    '- 必须原样保留字面占位符 {input}（仅一次），评估时会替换为实际用户输入\n'
    '- 必须原样保留原 Prompt 的输出格式约束（如「只输出类别名称」）和末尾输出引导（如「类别：」）\n'
    '- 不得添加「请逐步分析」「step by step」等推理引导，输出必须是裸答案\n'
    '- 不同候选之间要有差异化（角色、措辞、约束角度等）\n'
    '- 每条候选用三反引号包裹：```<prompt>```\n'
    '- 生成指定数量的候选\n'
)


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
        """从数据集构建任务描述（P2-2 修复：分层覆盖多类别）。"""
        desc_parts: List[str] = [
            "任务：给定输入，生成对应的期望输出。\n",
            "以下是数据集中的示例输入-输出对：\n",
        ]
        # P2-2 修复：旧实现用 dataset[:3]，本项目数据集前 3 条全是"财经"类，
        # 导致 LLM 看到的示例只覆盖 1 个类别。现按 target 分组轮询采样，
        # 覆盖尽量多的类别。
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
            desc_parts.append(f"  示例 {i + 1}:\n")
            desc_parts.append(f"    输入: {inp}\n")
            desc_parts.append(f"    输出: {tgt}\n")
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
        # 只保留得分最高的最多 20 条历史（喂给 LLM 当高质量参考）。
        # 旧实现用 [-20:] 取的是分数最低的 20 条，与注释意图相反。
        recent_history = sorted_history[:20]

        parts.append("以下是已尝试过的 Prompt 及其得分：\n\n")
        for i, (prompt_text, score) in enumerate(recent_history):
            parts.append(f"```\n得分: {score:.4f}\nPrompt: {prompt_text}\n```\n\n")

        # 5. 生成指令
        parts.append(
            f"基于以上信息，生成 {self._num_candidates} 条新的、更优的 Prompt。\n"
        )
        if recent_history:
            best_score = recent_history[0][1]
            parts.append(
                f"每条 Prompt 应有差异，并力求得分高于 {best_score:.4f}。\n"
            )
        parts.append(
            f"每条候选用三反引号包裹：```<prompt>```\n"
        )
        parts.append(f"编号 1 到 {self._num_candidates}。\n")

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

        # P2-3 修复：padding 不再加 [OPRO iteration n fill m] 前缀，
        # 改用含 {input} 占位符且带输出格式约束的中文模板，避免破坏评估链路。
        # 旧 padding 是英文通用模板，无类别列表、无中文、无格式约束，得分很低。
        _padding_template = (
            "你是一个有用的助手。请根据输入给出对应的输出，只输出结果本身。\n\n"
            "输入：{input}\n"
            "输出："
        )
        while len(candidates) < self._num_candidates:
            candidates.append(
                PromptCandidate(
                    id=str(uuid.uuid4()),
                    instruction=_padding_template,
                    metadata={"source": "opro_padding", "iteration": iteration},
                )
            )

        return candidates[: self._num_candidates]
