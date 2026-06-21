# Prompt Evolution 项目代码全面 Review

> Review 范围：`src/`（33 个 .py，约 4055 行）、`tests/`、`benchmark_news.py`、`scripts/`、配置文件
> Review 日期：2026-06-21
> 测试现状：现有 9 个测试全部通过（`uv run --with pytest --with pytest-asyncio pytest`）

---

## 一、总体评价

项目架构清晰、抽象合理：`BaseOptimizer / BaseModelProvider / BaseEvaluator / BaseMetric` 四件套分层得当，7 个优化器统一实现 `optimize() -> OptimizationResult` 接口，工厂模式 + 注册表设计便于扩展。文档质量高，论文引用、算法流程说明都很到位。

**但存在若干严重缺陷，直接导致核心结论（"优化器都不如 baseline"）失真：**

1. 🔴 **PromptBreeder / EvoPrompt 两个优化器存在 dead-code bug，候选 prompt 永远不会被评估**（`score is None` 判断恒为 False）。
2. 🔴 **benchmark 评测方法论错误**：baseline 在测试集上评测，优化器却报告训练集得分，二者不可比。
3. 🟠 **APE 费用统计严重虚高**（累加累计成本而非增量）。
4. 🟠 **F1ScoreMetric 用 set 实现，数学上不是 F1**。

下面按严重程度分级详述。

---

## 二、🔴 严重问题（P0，影响正确性，必须修）

### P0-1. PromptBreeder / EvoPrompt 候选评估失效（dead code）

**位置**：`optimizers/prompt_breeder/optimizer.py:175,190`、`optimizers/evoprompt/optimizer.py:187`

**问题**：两个进化算法优化器用 `if candidate.score is None:` 判断候选是否已评估。但 `PromptCandidate.score` 在 `core/models.py:18` 定义为：

```python
score: float = Field(0.0, description="评估得分，越高越好")
```

类型是 `float`（非 `Optional[float]`），默认值 `0.0`，**永远不可能是 None**。

**复现**：
```
default score = 0.0
score is None ? False
```

**后果**（以 PromptBreeder 为例）：
- `_init_population` 只评估了 `initial_prompt`，其余变体保持 `score=0.0`。
- 主循环中 `unevaluated = [p for p in population if p.score is None]` 永远是空列表。
- 第二个 `if candidate.score is None:` 也永远为 False。
- **变异/交叉产生的子代从未被评估**，全部停留在 0.0。
- 最终 `max(all_candidates, key=lambda c: c.score or 0.0)` 只能选出初始 prompt（或某个 0.0 的变体）。

这完美解释了 `项目.md` 中的 benchmark 现象：`promptbreeder` 耗时仅 848s（SPO 为 24320s）——因为它几乎没有调用评估器。

**修复建议**：
- 方案 A（推荐）：将 `PromptCandidate.score` 改为 `Optional[float] = None`，用 `None` 表示"未评估"。需同步修改所有 `c.score or 0.0`、`c.score if c.score is not None else -1.0` 等处。
- 方案 B（最小改动）：删除 `is None` 判断，改为显式标记。给 `PromptCandidate` 增加 `evaluated: bool = False` 字段，或在 metadata 中打标。进化算法里改为 `if not getattr(candidate, "_evaluated", False):`。

无论哪种方案，修复后 PromptBreeder / EvoPrompt 的耗时和效果都会显著变化，需要重跑 benchmark。

---

### P0-2. benchmark 评测方法论错误（训练集 vs 测试集混比）

**位置**：`benchmark_news.py:491-501`

**问题**：
- `run_optimizer(... train_data=train_data ...)` 把**训练集**传给优化器。
- 优化器内部用 `Evaluator` 在该数据集上评估候选，`result.best_prompt.score` 是**训练集准确率**。
- baseline 调用 `evaluate_prompt(provider, INITIAL_PROMPT, test_data, ...)`，是**测试集准确率**。
- 汇总表把两者并列展示：

```python
score = result.best_prompt.score if result.best_prompt else 0.0
results.append({"method": method, "score": score, ...})
```

**后果**：`项目.md` 中"都不如 baseline"的结论建立在错误的对比上——优化器报的是 train acc，baseline 报的是 test acc，分布与样本量都不同，根本不可比。优化器即便找到了更好的 prompt，报告的也是它在训练集上的表现，而非泛化能力。

**修复建议**：benchmark 应对每个优化器做两步：
1. 在训练集上跑优化，得到 `best_prompt`。
2. 用 `best_prompt` 在**测试集**上评测，报告测试集准确率（与 baseline 同口径）。

即增加：
```python
test_score = await evaluate_prompt(provider, result.best_prompt.instruction, test_data, ...)
results.append({"method": method, "score": test_score, "train_score": score, ...})
```

这样优化器之间、优化器与 baseline 之间才可比。

---

## 三、🟠 重要问题（P1，影响准确性/可信度）

### P1-1. APE 费用统计严重虚高

**位置**：`optimizers/ape/optimizer.py:112`

```python
for candidate in candidates:
    score = await self.evaluator.evaluate(...)
    candidate.score = score
    total_cost += getattr(self.model_provider, "_total_cost", 0.0)  # ❌
```

`_total_cost` 是 provider 内部的**累计**费用。每评估一个候选就加一次累计值，相当于 `cost_1 + (cost_1+cost_2) + (cost_1+cost_2+cost_3) + ...`，呈平方级虚高。

**对比**：OPRO / DSPy / SPO 用的是正确的增量法：
```python
cost_before = self.model_provider.total_cost_usd
...
total_cost = self.model_provider.total_cost_usd - cost_before
```

**修复**：APE 应改为同样在开头记录 `cost_before`，结尾取差值。删掉循环里的 `total_cost += ...`。

---

### P1-2. F1ScoreMetric 用 set 实现，不是标准 F1

**位置**：`evaluation/metrics/f1_score.py:34-35`

```python
pred_tokens = set(_tokenize(pred))
ref_tokens = set(_tokenize(ref))
```

把 token 转 `set` 会丢失词频信息。例如 pred="是 是 是 巴黎"、ref="巴黎"，set 后都只剩 {"是","巴黎"}，precision 被严重高估。标准 token-level F1（SQuAD 风格）应使用 `Counter`（多重集）计算重叠。

**修复**：
```python
from collections import Counter
pred_tokens = Counter(_tokenize(pred))
ref_tokens = Counter(_tokenize(ref))
overlap = sum((pred_tokens & ref_tokens).values())  # 交集取最小频次
precision = overlap / max(sum(pred_tokens.values()), 1)
recall = overlap / max(sum(ref_tokens.values()), 1)
```

---

### P1-3. `providers/base.py` 重复定义 `BaseModelProvider`，与 `core/base.py` 不一致

**位置**：`providers/base.py:15` vs `core/base.py:11`

`providers/base.py` 重新定义了一个 `BaseModelProvider(ABC)`，没有从 `core.base` 继承，而是平行地复制了一份抽象方法。而 `litellm_provider.py` 实际 import 的是 `core.base.BaseModelProvider`。这导致：
- 存在两个同名但不同的抽象类，容易误导。
- `providers/__init__.py` 导出的是 `providers/base.py` 版本，但真实实现继承的是 `core/base.py` 版本。
- 若用户继承 `providers.base.BaseModelProvider`，类型检查会与 `LiteLLMProvider` 不兼容。

**修复**：删除 `providers/base.py` 的重复定义，改为从 `core.base` 重导出（类似 `optimizers/base.py` 的做法）。

---

### P1-4. `Evaluator.evaluate_batch` 在已有事件循环中会崩溃

**位置**：`evaluation/evaluator.py:102`

```python
def evaluate_batch(self, prompts, dataset, model_provider, parallel=5):
    ...
    return asyncio.run(_run_all())  # ❌
```

`asyncio.run()` 不能在已有运行中的事件循环内调用，会抛 `RuntimeError: This event loop is already running`。而 `evaluate` 是 async 的，意味着任何在 async 上下文里调用 `evaluate_batch` 的场景都会炸。此外 `parallel` 参数被完全忽略（注释也承认"用 gather"）。

**修复**：把 `evaluate_batch` 改为 `async def`，用 `asyncio.Semaphore(parallel)` 控制并发，或直接 `return await asyncio.gather(...)`。

---

### P1-5. UI 评估 Tab 调用 Evaluator 接口错误

**位置**：`ui/app.py:246`

```python
evaluator = Evaluator(metrics=[AccuracyMetric(), ExactMatchMetric(), F1ScoreMetric()])
scores = evaluator.evaluate(predictions, references)  # ❌
```

`Evaluator.evaluate` 的签名是 `evaluate(prompt, dataset, model_provider)`，不接受 `(predictions, references)`。这里传错了参数，且 `evaluate` 是协程（需 await）。同时后面 `for name, score in scores.items()` 期望返回 dict，但 `evaluate` 返回 float。

**后果**：UI 的"Prompt 评估" Tab 一点就报错。

**修复**：要么给 `Evaluator` 增加一个 `compute_metrics(predictions, references) -> dict[str, float]` 的纯计算方法供 UI 直接用；要么 UI 走完整的 `evaluate(prompt, dataset, provider)` 流程。前者更合理（评估 Tab 已自行生成 predictions）。

---

## 四、🟡 一般问题（P2，代码质量/健壮性）

### P2-1. `_run_prompt_on_dataset` 丢弃了 prompt 模板（UI 评估 Tab）

**位置**：`ui/app.py:96-112`

```python
resp = await provider.generate(prompt=item.get("input", ""), system_prompt=prompt_text, ...)
```

直接把原始 `input` 当 prompt，把用户填的 prompt 当 system_prompt。这与优化器/evaluator 的"占位符替换"链路不一致，导致 UI 评估结果与优化结果不可对照。

### P2-2. UI 日志字符串未格式化

**位置**：`ui/app.py:110`

```python
logger.warning("预测失败: {exc}")  # ❌ 应为 f"预测失败: {exc}" 或 "预测失败: {}", exc
```

`{exc}` 不会被替换，日志里永远是字面量 `{exc}`。

### P2-3. `generate_with_logprobs` 缺少异常处理与默认参数

**位置**：`providers/litellm_provider.py:157-187`

- 没有 `try/except`（`generate` 有）。
- 不接受 `system_prompt / temperature / max_tokens`，与 `generate` 接口不一致。
- `response.choices[0].message.content` 可能为 None，直接放进 dict 不安全。

### P2-4. `estimate_cost` 返回值处理可能出错

**位置**：`providers/litellm_provider.py:205-212`

`litellm.cost_per_token` 在较新版本返回 `(prompt_cost, completion_cost)` 元组。代码用 `cost or 0.0` 当标量处理，元组恒为真值，会导致返回元组而非 float。应 `return sum(cost)` 或按版本适配。

### P2-5. 全局可变状态 `litellm.drop_params = True`

**位置**：`providers/litellm_provider.py:17`

模块导入时修改 litellm 全局配置，影响整个进程的其他 litellm 使用者。建议改为每次调用传 `drop_params=True` 到 `acompletion`。

### P2-6. 早停逻辑边界 bug（DSPy）

**位置**：`optimizers/dspy_optimizer/optimizer.py:194-199`

```python
_scores = [h["best_score"] for h in history]
if len(_scores) >= 3 and all(abs(_scores[-i] - _scores[-i-1]) < 0.001 for i in range(1, 4)):
```

`range(1,4)` ⇒ i=1,2,3，访问 `_scores[-1].._scores[-4]`，需要 `len >= 4` 才安全。`len==3` 时 `_scores[-4]` 越界（Python 负索引会回绕到 _scores[-1]，逻辑错误但不会崩）。应改为 `len(_scores) >= 4` 或 `range(1,3)`。

### P2-7. OPRO 历史排序后取 `[-20:]` 语义混乱

**位置**：`optimizers/opro/optimizer.py:296-298`

```python
sorted_history = sorted(scored_history, key=lambda x: x[1], reverse=True)
recent_history = sorted_history[-20:]
```

先按分数降序，再取"最后 20 条"= **分数最低的 20 条**。这和注释"只保留最近最多 20 条历史"完全相反——等于把最差的 prompt 喂给 LLM 当参考。应改为 `sorted_history[:20]`（取最高的），或按时间序取最近 20 条。

### P2-8. `Evaluator.evaluate` 签名与基类 Liskov 不一致

基类 `BaseEvaluator.evaluate` 的 `model_provider` 是必填位置参数，`Evaluator.evaluate` 改成了 `Optional` 默认 None。虽能跑，但违反 LSP，且 `evaluate_batch` 仍是必填——不一致。

### P2-9. 测试覆盖严重不足

- 7 个优化器只有 APE 有测试（9 个用例）。
- OPRO / DSPy / PromptBreeder / EvoPrompt / SPO **零测试**——P0-1 的 bug 本该被测试拦住。
- Evaluator / 各 metric / UI 回调 **零测试**。
- `pyproject.toml` 声明了 dev 依赖（pytest 等），但 `.venv` 里没装（`uv run pytest` 失败，需 `--with pytest`），CI 配置缺失。

### P2-10. `evaluation/metrics/__init__.py` 注释误导

注释称"通过 BaseMetric 的 `__init_subclass__` 机制自动注册"，但实际靠 `@register_metric` 装饰器，`BaseMetric` 并未实现 `__init_subclass__`。若以后有人写新指标忘了加装饰器，将不会被注册，且注释会误导排查。

### P2-11. 文档与实现不一致

- `README.md` 称依赖 `dspy-ai`，但 `dspy_optimizer` 实际是"轻量级，不依赖 dspy 库"（见模块 docstring）。`dspy-ai` 可从依赖中移除（减重）。
- README 项目结构里列了 `providers/` 含 "OpenAI / Anthropic / ..."，实际只有 LiteLLM 一个实现。
- README 路线图称 APE 为 "✅ MVP"、其余 "🚧 开发中/📋 待开发"，但代码里 7 个优化器都已实现，状态描述过时。

### P2-12. `random.seed(42)` 全局固定，影响可复现性语义

PromptBreeder / EvoPrompt / DSPy 的 `_bootstrap` 都在方法内部 `random.seed(42)`，会污染全局 random 状态，影响同一进程内其他随机逻辑。应使用局部 `random.Random(42)` 实例。

---

## 五、🟢 优点与亮点

1. **抽象设计干净**：`BaseOptimizer` 的钩子（`on_iteration_start/end`）、统一 `OptimizationResult`、工厂 + 注册表，扩展新优化器成本很低（参考 `factory.py` 只需加一行映射）。
2. **占位符 `{input}` 约束链路统一**：评估器与各优化器都约束 prompt 保留 `{input}` 占位符，`evaluator._generate_predictions` 用 `replace` 嵌入输入，避免了 system/user 双发破坏输出引导——这点设计得很专业，`scripts/verify_evaluator_fix.py` 也专门验证了。
3. **LiteLLMProvider 的 base_url 解析逻辑周到**：`_resolve_api_base` 按优先级处理显式参数 / `OPENAI_BASE_URL` / `LITELLM_API_BASE`，并对未知 provider 自动补 `openai/` 前缀，覆盖了 Ollama / DeepSeek / 自建代理等场景，且有专门测试。
4. **benchmark 的断点续评机制实用**：逐条写 checkpoint、按 method 独立记录，对长评测很友好；分层采样保持类别分布也考虑到位。
5. **文档质量高**：每个优化器都引用了论文 + arXiv/DOI，算法流程注释清晰，对学术复现友好。

---

## 六、修复优先级建议

| 优先级 | 问题 | 预计工作量 |
|--------|------|-----------|
| P0 | PromptBreeder/EvoPrompt 评估失效（P0-1） | 小（改 score 默认值或评估判断） |
| P0 | benchmark 训练/测试集口径（P0-2） | 小（加一步测试集评测） |
| P1 | APE 费用统计（P1-1） | 极小 |
| P1 | F1 set→Counter（P1-2） | 极小 |
| P1 | providers/base 重复定义（P1-3） | 极小 |
| P1 | evaluate_batch asyncio.run（P1-4） | 小 |
| P1 | UI 评估 Tab 接口错误（P1-5） | 小 |
| P2 | OPRO 历史取反（P2-7） | 极小（但影响 OPRO 效果） |
| P2 | 测试补齐（P2-9） | 中（重点补进化算法 + evaluator） |

**建议修复顺序**：先修 P0-1 + P0-2 → 重跑 benchmark（此时结论才有意义）→ 再修 P1 → 补测试。

---

## 七、结论

项目骨架优秀、文档用心，但**两个 P0 级 bug 让当前的核心结论（"优化器不如 baseline"）不可信**：
- PromptBreeder / EvoPrompt 根本没在评估候选（P0-1）；
- 即便评估了，报的也是训练集分数，与 baseline 的测试集分数混比（P0-2）。

修完这两个问题后重跑 benchmark，很可能结论会反转——至少能给出公平的对比。其余 P1/P2 多为局部修复，不影响架构。

整体代码质量评分：**架构 A- / 正确性 C / 测试覆盖 D**。把正确性和测试补齐后，这是个很有潜力的项目。
