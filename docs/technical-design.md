# Prompt 迭代神器 — 技术方案

> 版本：v1.0 | 日期：2026-06-10 | 状态：待评审

---

## 1. 项目概述

| 项目 | 内容 |
|------|------|
| 项目名称 | `prompt_evolution` |
| 定位 | 一站式 Prompt 自动优化工具（Python 库 + Web UI） |
| 核心价值 | 集成所有主流 Prompt 优化算法，提供统一接口，支持多模型，可视化管理 |
| 目标用户 | AI 应用开发者、Prompt 工程师、AI 产品团队 |

---

## 2. 技术栈选型

| 层级 | 技术选型 | 理由 |
|------|----------|------|
| 语言 | Python 3.10+ | 所有优化算法均有 Python 实现 |
| LLM 统一接口 | **LiteLLM** | 支持 100+ LLM，OpenAI 兼容格式，统一异常处理 |
| Web UI (MVP) | **Gradio 5.0+** | Python 原生，开发最快，支持断点调试 |
| Web UI (未来) | React + FastAPI | MVP 验证后可选升级 |
| 优化算法集成 | 原生实现 + DSPy | DSPy 已内置多种优化器，直接复用 |
| 评估框架 | 自研 + 参考 DeepEval | DeepEval 风格，Pytest 式语法 |
| 数据验证 | Pydantic v2 | 类型安全，序列化友好 |
| 配置管理 | pydantic-settings | 支持 .env / yaml 配置 |
| 日志 | loguru | 简洁易用 |
| 异步 | asyncio + httpx | 支持并发评估 |

### 2.1 Web UI 框架决策：Gradio > Streamlit > React+FastAPI

**选择 Gradio 的理由：**

- 开发速度最快（纯 Python，无需前端知识）
- 与 ML/AI 工具生态无缝集成
- 支持断点调试（Streamlit 不支持）
- 内置聊天界面、代码编辑器、文件上传等组件
- 可导出为独立 Web 服务

**未来升级路径：** 当 UI 复杂度超过 Gradio 能力时，迁移至 React + FastAPI。

---

## 3. 项目目录结构

```
prompt_evolution/
├── pyproject.toml                  # 项目配置 + 依赖管理 (使用 uv)
├── README.md                        # 项目介绍 + 快速开始
├── .env.example                     # 环境变量模板
├── .gitignore
│
├── src/
│   └── prompt_evolution/            # 主包
│       ├── __init__.py
│       ├── __main__.py              # python -m prompt_evolution 入口
│       │
│       ├── core/                    # 核心抽象层
│       │   ├── __init__.py
│       │   ├── base.py             # 所有抽象基类定义
│       │   ├── models.py           # Pydantic 数据模型
│       │   └── exceptions.py       # 自定义异常类
│       │
│       ├── providers/              # 模型提供商实现
│       │   ├── __init__.py
│       │   ├── base.py             # BaseModelProvider 抽象类
│       │   ├── litellm_provider.py # 基于 LiteLLM 的统一提供商（推荐）
│       │   ├── openai_provider.py  # OpenAI 直连（备用）
│       │   ├── anthropic_provider.py
│       │   ├── gemini_provider.py
│       │   ├── deepseek_provider.py
│       │   ├── ollama_provider.py  # 本地模型支持
│       │   └── factory.py          # 提供商工厂函数
│       │
│       ├── optimizers/             # 优化算法实现
│       │   ├── __init__.py
│       │   ├── base.py             # BaseOptimizer 抽象类
│       │   │
│       │   ├── ape/                # APE (Automatic Prompt Engineer)
│       │   │   ├── __init__.py
│       │   │   └── optimizer.py
│       │   │
│       │   ├── opro/               # OPRO (Optimization by PROmpting)
│       │   │   ├── __init__.py
│       │   │   └── optimizer.py
│       │   │
│       │   ├── dspy_wrappers/      # DSPy 优化器封装
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── bootstrap_fewshot.py
│       │   │   ├── mipro_v2.py
│       │   │   ├── copro.py
│       │   │   ├── gepa.py
│       │   │   └── labeled_fewshot.py
│       │   │
│       │   ├── evoprompt/          # EvoPrompt (微软，进化算法)
│       │   │   ├── __init__.py
│       │   │   ├── optimizer.py
│       │   │   ├── genetic_algorithm.py
│       │   │   └── differential_evolution.py
│       │   │
│       │   ├── rlprompt/           # RLPrompt (强化学习)
│       │   │   ├── __init__.py
│       │   │   └── optimizer.py
│       │   │
│       │   ├── promptagent/        # PromptAgent (MCTS)
│       │   │   ├── __init__.py
│       │   │   └── optimizer.py
│       │   │
│       │   └── factory.py           # 优化器工厂函数
│       │
│       ├── evaluation/             # 评估框架
│       │   ├── __init__.py
│       │   ├── base.py             # BaseEvaluator 抽象类
│       │   ├── metrics/            # 评估指标
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── accuracy.py
│       │   │   ├── f1_score.py
│       │   │   ├── rouge.py
│       │   │   ├── bleu.py
│       │   │   ├── exact_match.py
│       │   │   ├── llm_judge.py
│       │   │   └── custom.py
│       │   ├── evaluator.py        # 核心评估逻辑
│       │   ├── dataset.py          # 数据集加载与处理
│       │   └── comparator.py       # 多候选 prompt 对比
│       │
│       ├── ui/                     # Web UI (Gradio)
│       │   ├── __init__.py
│       │   ├── app.py              # Gradio 主应用入口
│       │   ├── pages/              # 多页面/多标签
│       │   │   ├── __init__.py
│       │   │   ├── optimizer_page.py
│       │   │   ├── evaluator_page.py
│       │   │   ├── compare_page.py
│       │   │   └── history_page.py
│       │   ├── components/         # 可复用 UI 组件
│       │   │   ├── __init__.py
│       │   │   ├── model_selector.py
│       │   │   ├── optimizer_selector.py
│       │   │   ├── metric_selector.py
│       │   │   ├── dataset_uploader.py
│       │   │   └── result_viewer.py
│       │   └── state.py            # Gradio 状态管理
│       │
│       ├── storage/                # 结果持久化
│       │   ├── __init__.py
│       │   ├── base.py             # BaseStorage 抽象类
│       │   ├── json_storage.py     # JSON 文件存储 (轻量)
│       │   ├── sqlite_storage.py   # SQLite 存储 (推荐)
│       │   └── models.py           # 数据库 ORM 模型
│       │
│       └── utils/                  # 工具函数
│           ├── __init__.py
│           ├── config.py           # 配置管理
│           ├── logging.py          # 日志配置
│           ├── parallel.py         # 并发执行工具
│           └── formatting.py       # 输出格式化
│
├── tests/                          # 测试套件
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_providers/
│   ├── test_optimizers/
│   ├── test_evaluation/
│   └── test_integration/
│
├── examples/                       # 使用示例
│   ├── basic_usage.py
│   ├── optimize_with_ape.py
│   ├── optimize_with_dspy.py
│   ├── compare_optimizers.py
│   └── web_ui_demo.py
│
├── docs/                           # 文档
│   ├── index.md
│   ├── getting_started.md
│   ├── api_reference.md
│   ├── optimizer_guide.md
│   └── web_ui_guide.md
│
└── scripts/                        # 辅助脚本
    ├── setup.sh
    └── run_ui.sh
```

---

## 4. 核心模块设计

### 4.1 统一 Optimizer 抽象接口

所有优化器实现同一接口，确保统一调用方式。

```python
# src/prompt_evolution/core/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PromptCandidate(BaseModel):
    """Prompt 候选对象"""
    id: str
    instruction: str                # 系统指令部分
    demo_examples: List[Dict] = [] # few-shot 示例
    score: float = 0.0
    metadata: Dict[str, Any] = {}

class OptimizationResult(BaseModel):
    """优化结果"""
    best_prompt: PromptCandidate
    all_candidates: List[PromptCandidate]
    optimization_history: List[Dict]  # 每轮迭代记录
    total_cost: float                 # API 费用
    elapsed_time: float               # 耗时

class BaseOptimizer(ABC):
    """所有优化器的抽象基类"""

    @abstractmethod
    def __init__(
        self,
        model_provider: "BaseModelProvider",
        evaluator: "BaseEvaluator",
        config: Optional[Dict[str, Any]] = None
    ):
        ...

    @abstractmethod
    async def optimize(
        self,
        initial_prompt: PromptCandidate,
        dataset: List[Dict],
        max_iterations: int = 10,
        **kwargs
    ) -> OptimizationResult:
        """执行优化，返回最优 prompt"""
        ...

    @abstractmethod
    def name(self) -> str:
        """优化器名称"""
        ...

    # 可选钩子方法
    def on_iteration_start(self, iteration: int):
        """每轮迭代开始回调"""
        ...

    def on_iteration_end(self, iteration: int, candidates: List[PromptCandidate]):
        """每轮迭代结束回调"""
        ...
```

**优化器层次结构：**

```
BaseOptimizer (抽象基类)
    ├── APEOptimizer       (生成-筛选 两阶段)
    ├── OPROOptimizer      (元提示迭代优化)
    ├── DSPyOptimizer      (适配 DSPy 优化器)
    ├── EvoPromptOptimizer (进化算法)
    ├── RLPromptOptimizer  (强化学习)
    └── PromptAgentOptimizer (MCTS)
```

---

### 4.2 统一 Model Provider 抽象接口

通过 LiteLLM 统一接入 100+ 模型，屏蔽不同厂商 API 差异。

```python
# src/prompt_evolution/providers/base.py

class BaseModelProvider(ABC):
    """所有模型提供商的抽象基类"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """生成文本"""
        ...

    @abstractmethod
    async def generate_with_logprobs(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本并返回 logprobs（用于某些优化算法）"""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算 token 数"""
        ...

    @abstractmethod
    def estimate_cost(self, prompt: str, completion: str) -> float:
        """估算 API 调用费用"""
        ...

# 推荐使用 LiteLLM 实现统一接口
class LiteLLMProvider(BaseModelProvider):
    """基于 LiteLLM 的统一模型提供商"""

    def __init__(self, model: str, api_key: str = None, **kwargs):
        self.model = model  # 格式: "openai/gpt-4o", "anthropic/claude-3", ...
        self.client = litellm
        ...
```

**支持的模型（通过 LiteLLM）：**

| 厂商 | 示例模型标识 |
|------|-------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4-turbo` |
| Anthropic | `anthropic/claude-3-5-sonnet` |
| Google | `gemini/gemini-pro` |
| DeepSeek | `deepseek/deepseek-chat` |
| Ollama (本地) | `ollama/llama3` |

---

### 4.3 评估框架设计

参考 DeepEval 设计，支持确定型指标和 LLM-as-Judge。

```python
# src/prompt_evolution/evaluation/base.py

class BaseMetric(ABC):
    """评估指标抽象基类"""

    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算指标分数"""
        ...

class BaseEvaluator(ABC):
    """评估器抽象基类"""

    @abstractmethod
    async def evaluate(
        self,
        prompt: PromptCandidate,
        dataset: List[Dict],
        model_provider: BaseModelProvider
    ) -> float:
        """评估一个 prompt 在特定数据集上的表现"""
        ...

    def evaluate_batch(
        self,
        prompts: List[PromptCandidate],
        dataset: List[Dict],
        model_provider: BaseModelProvider,
        parallel: int = 5
    ) -> List[float]:
        """批量评估（并发）"""
        ...
```

**指标类型分层：**

```
指标类型
├── 确定型指标 (ExactMatch, F1, ROUGE, BLEU)  — 无需调用 LLM
├── 模型型指标 (LLM-as-Judge)                  — 需要调用 LLM 评分
└── 自定义指标 (用户提供的函数)                   — 灵活扩展
```

---

### 4.4 Web UI 功能模块设计（Gradio）

```python
# src/prompt_evolution/ui/app.py

import gradio as gr

def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""
    with gr.Blocks(title="Prompt 迭代神器") as app:

        # 标签 1: Prompt 优化
        with gr.Tab("Prompt 优化"):
            # - 选择优化器 (APE / OPRO / DSPy / EvoPrompt ...)
            # - 选择模型提供商
            # - 输入初始 prompt
            # - 上传数据集
            # - 配置优化参数
            # - 启动优化 (显示实时进度)
            # - 查看优化结果 (最优 prompt + 历史曲线)
            ...

        # 标签 2: Prompt 评估
        with gr.Tab("Prompt 评估"):
            # - 输入 prompt
            # - 选择评估指标
            # - 上传测试集
            # - 运行评估
            # - 查看详细报告
            ...

        # 标签 3: Prompt 对比
        with gr.Tab("Prompt 对比"):
            # - 输入多个 prompt
            # - 并排对比输出
            # - 雷达图展示各指标得分
            ...

        # 标签 4: 优化历史
        with gr.Tab("优化历史"):
            # - 查看历史优化记录
            # - 导出最优 prompt
            # - 复现实验
            ...

    return app
```

**UI 布局示意：**

```
┌─────────────────────────────────────────────────────┐
│               Prompt 迭代神器  v0.1.0              │
├──────────┬──────────┬──────────┬───────────────────┤
│ Prompt   │ Prompt   │ Prompt   │ 优化历史          │
│ 优化     │ 评估     │ 对比     │                   │
├──────────┴──────────┴──────────┴───────────────────┤
│                                                     │
│  选择优化器: [APE          ▼]                       │
│  选择模型:   [openai/gpt-4o          ▼]            │
│                                                     │
│  初始 Prompt:                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │ 你是一个有用的助手。                           │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  数据集文件: [选择文件...]          [开始优化]       │
│                                                     │
│  优化进度: [============================] 80%       │
│                                                     │
│  最优 Prompt:                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │ [优化后的 prompt 展示]                        │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  得分: 0.92   耗时: 45.3s   费用: $0.23           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. 集成的 Prompt 优化方法

### 5.1 各方法简介

| 方法 | 来源 | 核心思想 | 适用场景 |
|------|------|---------|---------|
| **APE** | GitHub: keirp/automatic_prompt_engineer | 生成候选 prompt 并筛选最优 | 快速原型，简单任务 |
| **OPRO** | Google DeepMind | 用 LLM 作为优化器迭代优化 | 复杂任务，需要多轮迭代 |
| **BootstrapFewShot** | DSPy (Stanford) | 自动生成 few-shot 示例 | 需要 few-shot 的场景 |
| **MIPROv2** | DSPy (Stanford) | 同时优化指令和演示 | 高精度要求场景 |
| **COPRO** | DSPy (Stanford) | 提示作为自动优化超参数 | 研究场景 |
| **GEPA** | DSPy (Stanford) | 遗传帕累托优化 | 多目标优化 |
| **EvoPrompt** | 微软 | 基于进化算法的 prompt 优化 | 大规模搜索空间 |
| **RLPrompt** | 学术论文 | 将 prompt 优化建模为强化学习 | 需要强化学习的场景 |
| **PromptAgent** | 学术论文 | 基于 MCTS 的提示优化 | 复杂推理任务 |

### 5.2 集成优先级

**Phase 1（MVP）：**
- ✅ APE — 最简单，生成-筛选两阶段
- ✅ 基础评估指标（Accuracy, Exact Match）

**Phase 2（算法扩展）：**
- ✅ DSPy 全系列优化器（BootstrapFewShot, MIPROv2, COPRO, GEPA）
- ✅ OPRO
- ✅ EvoPrompt
- ✅ 完善评估指标（ROUGE/BLEU/LLM-Judge）

**Phase 3（高级功能）：**
- RLPrompt（可选，复杂度高）
- PromptAgent（可选，复杂度高）
- 红队测试、CI/CD 集成

---

## 6. 实现路线图

### Phase 1: MVP 核心功能（4-6 周）

**目标：可用的最小闭环**

- [ ] 项目骨架搭建（目录结构、pyproject.toml、CI 配置）
- [ ] `BaseModelProvider` + `LiteLLMProvider` 实现
- [ ] `BaseOptimizer` 抽象接口定义
- [ ] 第一个优化器实现：**APE**
- [ ] 基础评估指标：Accuracy, Exact Match
- [ ] JSON 文件存储（保存优化结果）
- [ ] Gradio Web UI 基础版（单页面，优化 + 结果展示）
- [ ] Python API 基础版：`optimize(prompt, dataset, method="ape")`
- [ ] README + 快速开始文档

**Phase 1 交付物：** 可安装使用的 Python 库 + 基础 Web UI

---

### Phase 2: 算法扩展 + 评估完善（6-8 周）

**目标：集成所有主流优化算法**

- [ ] 集成 **DSPy 优化器**（BootstrapFewShot, MIPROv2, COPRO, GEPA）
- [ ] 实现 **OPRO** 优化器
- [ ] 实现 **EvoPrompt** 优化器（GA + DE 两种进化策略）
- [ ] 完善评估指标：ROUGE, BLEU, LLM-as-Judge
- [ ] 评估框架支持批量并发
- [ ] SQLite 存储（替代 JSON）
- [ ] Gradio Web UI 多标签版（优化/评估/对比/历史）
- [ ] 优化历史查看与导出
- [ ] 多模型对比功能
- [ ] 完整文档（API 参考 + 各优化器指南）

**Phase 2 交付物：** 支持 6+ 种优化算法，功能完整的 Web 应用

---

### Phase 3: 高级功能 + 生产就绪（4-6 周）

**目标：生产级工具**

- [ ] 实现 **RLPrompt** 优化器（可选）
- [ ] 实现 **PromptAgent** 优化器（可选）
- [ ] 红队测试功能（参考 promptfoo）
- [ ] CI/CD 集成（GitHub Actions 评估 Pipeline）
- [ ] 多用户支持（可选：迁移至 React + FastAPI）
- [ ] Docker 容器化部署
- [ ] 性能优化（缓存、并发控制）
- [ ] 完整测试覆盖（单元测试 + 集成测试）
- [ ] PyPI 发布

**Phase 3 交付物：** 生产级 Prompt 优化平台

---

## 7. 依赖清单

```toml
# pyproject.toml 核心依赖
[project]
name = "prompt-evolution"
version = "0.1.0"
description = "Prompt 迭代神器 - 一站式 Prompt 自动优化工具"
requires-python = ">=3.10"
dependencies = [
    "litellm>=1.0.0",
    "dspy-ai>=2.5.0",
    "gradio>=5.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "loguru>=0.7.0",
    "httpx>=0.27.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "rouge-score>=0.1.2",
    "nltk>=3.8.0",
    "tiktoken>=0.5.0",
    "pyyaml>=6.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-asyncio", "pytest-cov"]
```

---

## 8. Python API 使用示例

```python
import prompt_evolution as pe

# 初始化模型提供商
provider = pe.LiteLLMProvider(model="openai/gpt-4o", api_key="sk-...")

# 初始化评估器
evaluator = pe.Evaluator(metrics=[pe.metrics.Accuracy()])

# 选择优化器并优化
optimizer = pe.optimizers.APEOptimizer(
    model_provider=provider,
    evaluator=evaluator,
    config={"num_candidates": 10, "num_iterations": 3}
)

# 准备数据集
dataset = [
    {"input": "法国的首都是哪里？", "target": "巴黎"},
    {"input": "德国首都是哪里？", "target": "柏林"},
]

# 执行优化
result = await optimizer.optimize(
    initial_prompt=pe.PromptCandidate(instruction="你是一个有用的助手。"),
    dataset=dataset,
    max_iterations=5
)

print(f"最优 Prompt: {result.best_prompt.instruction}")
print(f"得分: {result.best_prompt.score}")
print(f"耗时: {result.elapsed_time}s")
print(f"费用: ${result.total_cost}")
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| DSPy API 变更频繁 | 高 | 封装适配层，隔离 DSPy 直接依赖 |
| 优化算法计算成本高 | 中 | 支持缓存、使用本地模型 (Ollama) |
| LiteLLM 异常映射不完整 | 低 | 自定义异常处理层 |
| Gradio 无法满足复杂 UI 需求 | 中 | Phase 2 评估是否迁移 React |
| 各优化算法论文复现难度大 | 高 | 优先使用官方开源实现，次优先参考论文 |

---

## 10. 参考资料

| 名称 | 类型 | 链接 |
|------|------|------|
| promptfoo | 评估框架 | https://github.com/promptfoo/promptfoo |
| DSPy | 优化框架 | https://github.com/stanfordnlp/dspy |
| APE | 论文+代码 | https://github.com/keirp/automatic_prompt_engineer |
| OPRO | 论文 | Google DeepMind (arXiv:2309.03409) |
| EvoPrompt | 论文+代码 | 微软 (arXiv:2309.08532) |
| DeepEval | 评估框架 | https://github.com/confident-ai/deepeval |
| LiteLLM | 模型统一接口 | https://github.com/BerriAI/litellm |

---

*本技术方案由架构师 Agent 编写，齐活林（Qi）整理归档。*
