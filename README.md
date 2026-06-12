# Prompt 迭代神器 🚀

一站式 Prompt 自动优化工具 —— 集成 GitHub 上最优秀的各种 Prompt 迭代方法，支持多模型、可视化评估、Web UI 一键优化。

---

## 特性

- **多算法集成**：APE、OPRO、DSPy（BootstrapFewShot / MIPROv2 / COPRO / GEPA）、PromptBreeder、EvoPrompt、SPO（持续增加中）
- **多模型支持**：通过 [LiteLLM](https://github.com/BerriAI/litellm) 统一接入 OpenAI / Anthropic / Gemini / DeepSeek / Ollama 等 100+ 模型
- **统一接口**：所有优化器实现同一 `BaseOptimizer` 接口，`optimize()` → `OptimizationResult`
- **评估框架**：内置 Accuracy、ExactMatch、F1、ROUGE、BLEU 等指标，支持 LLM-as-Judge
- **Web UI**：基于 Gradio 的可视化界面，Prompt 优化、评估、对比、历史记录一站式完成
- **CLI 工具**：`prompt-evol optimize` 命令行一键优化

---

## 快速开始

### 安装

```bash
# 从源码安装（推荐）
git clone https://github.com/your-username/prompt-evolution.git
cd prompt-evolution

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e ".[dev]"
```

### 配置 API Key

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 例如：OPENAI_API_KEY=sk-abc123
```

如果模型支持 reasoning / thinking 输出，且你希望请求时关闭它，也可以在 .env 中增加：

```bash
DISABLE_THINKING=true
```

### 配置 Base URL（OpenAI 兼容接口）

如果你使用 OpenAI 兼容的 API（如本地模型、第三方代理、DeepSeek 等），需要配置 `base_url`：

```bash
# 方式 1：在 .env 文件中配置（推荐）
# 编辑 .env，添加：
OPENAI_BASE_URL=http://localhost:8000/v1

# 方式 2：CLI 参数
prompt-evol optimize ... --base-url http://localhost:8000/v1

# 方式 3：Web UI 输入框
# 在「Base URL」输入框中填写自定义地址
```

**常见场景**：
- **Ollama 本地模型**：`OPENAI_BASE_URL=http://localhost:11434/v1`
- **DeepSeek API**：`OPENAI_BASE_URL=https://api.deepseek.com/v1`
- **自定义代理**：`OPENAI_BASE_URL=https://your-proxy.com/v1`

### CLI 使用

```bash
# 查看可用优化方法
prompt-evol list

# 运行 APE 优化（使用默认 OpenAI 地址）
prompt-evol optimize \
  --prompt "你是一个有用的助手。" \
  --dataset examples/dataset.json \
  --method ape \
  --model openai/gpt-4o \
  --max-iters 5 \
  --num-candidates 10 \
  -o result.json

# 运行 APE 优化（使用自定义 Base URL，如 Ollama 本地模型）
prompt-evol optimize \
  --prompt "你是一个有用的助手。" \
  --dataset examples/dataset.json \
  --method ape \
  --model openai/gpt-4o \
  --base-url http://localhost:11434/v1 \
  --max-iters 5 \
  -o result.json

# 运行时关闭 thinking / reasoning 输出
prompt-evol optimize \
  --prompt "你是一个有用的助手。" \
  --dataset examples/dataset.json \
  --method ape \
  --model openrouter/poolside/laguna-m.1:free \
  --disable-thinking

# 启动 Web UI
prompt-evol ui --port 7860
```

### Web UI 使用

```bash
python -m prompt_evolution ui
```

浏览器打开 `http://127.0.0.1:7860`，在界面中：
1. 选择优化算法和模型
2. （可选）填写 Base URL（OpenAI 兼容接口自定义地址）
3. （可选）勾选「关闭 thinking / reasoning」
4. （可选）填写 API Key
4. 输入初始 Prompt
5. 上传数据集（JSON 格式）
6. 点击「开始优化」
7. 查看最优 Prompt 和每轮迭代历史

---

## 数据集格式

数据集为 JSON 数组，每项包含 `input`（或 `question`）和 `target`（或 `answer`）：

```json
[
  {"input": "法国的首都是哪里？", "target": "巴黎"},
  {"input": "德国首都是哪里？", "target": "柏林"},
  {"input": "日本首都是哪里？", "target": "东京"}
]
```

---

## 新闻分类数据集（中文）

项目内置了一份中文新闻分类数据集，来自清华大学新闻分类语料，共 14 个类别：

> 科技、股票、体育、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座

文件已转换为 prompt_evolution 标准格式：

```bash
examples/ag_news_train.json   # 1400 条（每类 100 条）
examples/ag_news_test.json    # 420 条（每类 30 条）
```

格式示例：

```json
[{"input": "上证50ETF净申购突增", "target": "财经"}]
```

### 运行评测

使用 `benchmark_news.py` 一键评测基础 Prompt 和各优化器效果：

```bash
# 设置 API Key
export OPENAI_API_KEY=sk-xxx

# 运行完整评测（Baseline + 所有优化器）
python benchmark_news.py

# 只跑指定优化器
python benchmark_news.py --methods ape opro dspy

# 使用自定义模型和 Base URL
python benchmark_news.py \
  --model openai/gpt-4o-mini \
  --base-url http://localhost:11434/v1

# 跳过 baseline，只跑优化器
python benchmark_news.py --skip-baseline
```

评测完成后结果保存在 `benchmark_results.json`，终端会输出对比表格：

```
方法                  Accuracy    耗时(s)
============================================================
baseline              0.6523      5.2
ape                   0.7841     32.1
opro                 0.8012     45.3
...
```

---

## Python API

```python
import asyncio
from prompt_evolution import PromptCandidate, create_optimizer
from prompt_evolution.providers import LiteLLMProvider
from prompt_evolution.evaluation import Evaluator
from prompt_evolution.evaluation.metrics import AccuracyMetric

async def main():
    # 使用默认 OpenAI 地址
    provider = LiteLLMProvider(model="openai/gpt-4o", api_key="sk-...")

    # 或使用自定义 Base URL（OpenAI 兼容接口）
    provider = LiteLLMProvider(
        model="openai/gpt-4o",
        api_key="sk-...",
      api_base="http://localhost:8000/v1",
      disable_thinking=True,
    )

    evaluator = Evaluator(metrics=[AccuracyMetric()])
    optimizer = create_optimizer(
        name="ape",
        model_provider=provider,
        evaluator=evaluator,
        config={"num_candidates": 10},
    )

    initial = PromptCandidate(instruction="你是一个有用的助手。")
    dataset = [
        {"input": "1+1=？", "target": "2"},
        {"input": "2+3=？", "target": "5"},
    ]

    result = await optimizer.optimize(
        initial_prompt=initial,
        dataset=dataset,
        max_iterations=5,
    )

    print(f"最优 Prompt: {result.best_prompt.instruction}")
    print(f"得分: {result.best_prompt.score:.4f}")
    print(f"耗时: {result.elapsed_time_s:.1f}s")

asyncio.run(main())
```

---

## 项目结构

```
prompt_evolution/
├── src/prompt_evolution/   # 主包
│   ├── core/                  # 抽象基类（BaseOptimizer / BaseModelProvider / BaseEvaluator）
│   ├── providers/            # 模型提供商（LiteLLM / OpenAI / Anthropic / ...）
│   ├── optimizers/            # 优化算法实现
│   │   ├── ape/                # APE (Automatic Prompt Engineer)
│   │   ├── opro/               # OPRO (Optimization by PROmpting)
│   │   ├── dspy_optimizer/     # DSPy 优化器封装
│   │   ├── prompt_breeder/     # PromptBreeder (进化算法)
│   │   ├── evoprompt/          # EvoPrompt (进化 Prompt 优化)
│   │   └── spo/                # SPO (语义邻域搜索)
│   ├── evaluation/           # 评估框架
│   │   └── metrics/            # 评估指标（Accuracy / F1 / ROUGE / ...）
│   └── ui/                   # Web UI (Gradio)
├── examples/                  # 使用示例
├── tests/                    # 测试套件
└── docs/                     # 文档
```

---

## 支持的优化算法

| 方法 | 状态 | 说明 |
|------|------|------|
| APE | ✅ MVP | 生成-筛选两阶段，快速原型 |
| OPRO | 🚧 开发中 | 用 LLM 作为优化器迭代优化 |
| DSPy (BootstrapFewShot) | 🚧 开发中 | 自动生成 few-shot 示例 |
| DSPy (MIPROv2) | 🚧 开发中 | 同时优化指令和演示 |
| DSPy (COPRO) | 🚧 开发中 | 坐标上升优化指令与示例 |
| PromptBreeder | 🚧 开发中 | 进化算法 + 自我繁殖 Prompt |
| EvoPrompt | 📋 待开发 | 基于进化算法的 prompt 优化 |
| SPO | 📋 待开发 | 语义邻域搜索优化 |

---

## 开发路线图

- **Phase 1**（当前）：MVP 核心闭环 — APE + 基础评估 + Web UI
- **Phase 2**：算法扩展 — OPRO + EvoPrompt + DSPy 全系列 + 完善评估指标
- **Phase 3**：生产就绪 — 红队测试 + CI/CD + Docker + PyPI 发布

---

## 技术栈

- **Python 3.10+**
- **LiteLLM** — 多模型统一接口
- **Gradio** — Web UI
- **DSPy** — 优化算法集成
- **Pydantic v2** — 数据验证
- **loguru** — 日志

---

## 贡献

欢迎提交 PR！请先查看 [docs/technical-design.md](docs/technical-design.md) 了解架构设计。

---

## 许可证

[MIT License](LICENSE)
