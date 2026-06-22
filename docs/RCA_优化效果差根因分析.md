# Prompt Evolution 优化效果劣于 Baseline 根因分析报告

> 任务：中文新闻分类（14 分类，1400 训练 / 420 测试），6 个优化器 12 个变体测试准确率均低于 baseline（baseline ≈ 0.85–0.86，优化器 ≈ 0.78–0.82）。
> 本报告基于对 `src/prompt_evolution/` 全部核心代码与 `benchmark_news.py` 的逐行审阅。

---

## 一、现象确认

`项目.md` 记录的两次评测结果：

| 方法 | Test Acc | 耗时(s) |
|------|---------|---------|
| baseline | 0.8500 / 0.8600 | 80.7 |
| ape | 0.7950 | 1582.1 |
| opro | 0.8150 | 2763.4 |
| spo | 0.8150 | 2889.6 |
| evoprompt | 0.7900 | — |
| promptbreeder | 0.7850 | 848.2 |
| dspy | 0.7900 | — |

**所有优化器变体的测试集准确率都低于 baseline**，且优化器耗时远高于 baseline。

---

## 二、评测口径验证（已排除的嫌疑）

在定位根因前，先排除「评测不公」的可能性——确认优化器与 baseline 是同口径比较。

| 维度 | baseline（`benchmark_news.py:evaluate_prompt`） | 优化器训练评估（`Evaluator` + `AccuracyMetric`） | 优化器测试评估 |
|------|------------------------------------------------|--------------------------------------------------|----------------|
| 渲染 | `prompt_instruction.replace("{input}", input_text)` (L264) | `instruction.replace("{input}", user_input)` (evaluator.py:176) | 同 baseline |
| system_prompt | 不传（默认 None） | `system_prompt=None` (evaluator.py:190) | 同 baseline |
| temperature | 0.0 (L271) | 0.0 (evaluator.py:191) | 同 baseline |
| max_tokens | 512 (L272) | 512 (evaluator.py:192) | 同 baseline |
| 匹配规则 | `prediction == ground_truth`（均 strip）(L281) | `pred.strip().lower() == ref.strip().lower()` (accuracy.py:28) | 同 baseline |

**结论 1**：单条预测逻辑完全一致，中文类别名无大小写问题，**匹配口径等价**。
**结论 2**：`benchmark_news.py:515-531` 已修复「混比」问题——优化器用 `best_instruction` 在 `test_data` 上重新评测，主指标是 `test_score`。所以「优化器劣于 baseline」是**真实测试集同口径**结果，结论可信，需要找真正的根因。

---

## 三、根因分析（按优先级）

### 🔴 P0-1：APE 不把 initial_prompt 纳入候选池（机制性劣化，最致命）

**位置**：`src/prompt_evolution/optimizers/ape/optimizer.py`

```python
# L86  all_candidates 初始化为空，从未加入 initial_prompt
all_candidates: List[PromptCandidate] = []

# L96-133  循环中只 extend 新生成的候选
all_candidates.extend(candidates)   # L116

# L136  只在新候选中选最优 —— 即使全部新候选都劣于 initial
best_prompt = max(all_candidates, key=lambda c: c.score)
```

**后果**：APE **机制上必然返回新生成的 prompt**。当初始 prompt 已很完善（baseline 0.85）时，LLM 随机生成的 8 个候选大概率都更差，但 APE 仍强制返回其中「最不差」的一个。这从结构上保证了 APE 大概率劣于 baseline。

**对比**：OPRO / DSPy / PromptBreeder / EVOPrompt / SPO 都在开头评估并加入 `initial_prompt`（如 `opro/optimizer.py:143-149`），best_prompt.train_score ≥ initial.train_score，至少在训练集上不会比 baseline 差。

---

### 🔴 P0-2：所有优化器的 meta-prompt 不约束保留输出格式约束

初始 prompt 能拿到 0.85 的关键在于 3 个结构要素：

```python
INITIAL_PROMPT = """你是一个新闻分类专家...类别列表：...
只输出类别名称，不要输出任何其他内容。   ← 输出格式约束
新闻标题："{input}"                       ← {input} 占位符
类别："""                                 ← 末尾输出引导
```

6 个优化器的 meta-prompt **都只约束保留 `{input}` 占位符**，但**没有任何一个约束保留「只输出类别名称」或末尾「类别：」引导**。LLM 在「改进/变异/改写」时普遍倾向：
- 改写输出引导（`类别：` → `答案：` / `分类结果：`）
- 加 CoT 引导（「请逐步分析后给出类别」）—— OPRO 尤甚
- 调整格式规格

这些改动让模型输出非裸类别名（如「财经类」「该新闻属于财经」），在 `AccuracyMetric` 的 `pred.strip().lower() == ref.strip().lower()` 严格相等下**判错**。

关键证据——各优化器的 meta-prompt 原文：

| 优化器 | 位置 | 问题表述 |
|--------|------|----------|
| APE | `ape/optimizer.py:48-62` | `_placeholder_constraint` 只约束 `{input}`，无格式约束 |
| OPRO | `opro/optimizer.py:75-76` | `_CANDIDATE_GEN_SYSTEM`：「Include concrete instructions on output format when helpful」「some with step-by-step reasoning」——**主动鼓励改格式和加推理** |
| DSPy | `dspy_optimizer/optimizer.py:27-45` | `_PROPOSE_SYSTEM` 只约束 `{input}`，无格式约束 |
| PromptBreeder | `prompt_breeder/optimizer.py:45` | 变异策略含「**Change the output format specification**」——主动鼓励改格式 |
| EVOPrompt | `evoprompt/optimizer.py:46` | 变异策略 4「**Make the output format specification more explicit**」——诱导改写「类别：」引导 |
| SPO | `spo/optimizer.py:41-42` | 角度 2「add/remove think step by step」+ 角度 3「**Change the output format specification**」——双重破坏 |

**后果**：LLM 生成的候选 prompt 经常破坏裸输出格式，即使格式彻底破坏的候选在 train 上得低分被淘汰，但 LLM 可能做「微小」格式改动，在 train 200 条上偶然多对几条被选中，在 test 100 条上暴露问题（过拟合 + 格式不稳）。

---

### 🟠 P1-1：meta-prompt 全英文，与中文任务 / few-shot 严重不一致

6 个优化器的 system prompt 和 meta-prompt **全是英文**，而初始 prompt、类别名、数据集**全是中文**。LLM 在英文指令下「改进」中文 prompt，可能：
- 生成全英文 prompt（类别名对照失败）
- 在中文 prompt 里夹杂英文说明

**最严重的是 OPRO**：`opro/optimizer.py:45-68` 的 `_DEFAULT_META_FEWSHOT` 是**英文数学辅导示例**：

```python
_DEFAULT_META_FEWSHOT = """Here are some examples of good optimization:
Example history:
Score: 0.92
Prompt: "You are a helpful math tutor. Solve the problem step by step, showing all work."
...
"""
```

这与中文 14 类新闻分类任务**完全无关**，且会诱导 LLM 生成「step by step」风格 prompt，对分类任务的裸输出要求是直接破坏。

---

### 🟠 P1-2：训练样本过小（200 条）导致选择偏差 / 过拟合

`benchmark_news.py:362` 默认 `--train-samples 200`，分层采样后 14 类每类约 14 条；test 默认 100 条每类约 7 条。

OPRO / DSPy / SPO / PromptBreeder / EVOPrompt 虽含 initial_prompt（best.train ≥ initial.train），但测试仍劣于 baseline，说明选中的是 **train 得分略高但 test 得分更低的新候选**——典型过拟合。机制：

```
train 200 条：新候选偶然多对 1-2 条（+0.5%~1%）→ 超过 initial → 被选中
test  100 条：同一 prompt 多错 5-6 条（-5%）→ 表现为 0.79 vs 0.85
```

temperature=0.0 虽确定性，但 train/test 是不同样本，统计噪声驱动选择偏差。样本越小，噪声越大。

---

### 🟠 P1-3：变异类优化器的变异算子主动鼓励破坏输出格式

PromptBreeder / EVOPrompt / SPO 的 mutation/rewrite system prompt 都含「Change the output format specification」（`prompt_breeder:45`、`evoprompt:46`、`spo:42`），SPO 还含「add/remove think step by step」（`spo:41`）。在分类任务里这些是减分操作，与 P0-2 叠加，破坏性最强。

---

### 🟡 P2-1：APE 迭代轮数默认为 1，搜索空间极小

`ape/optimizer.py:46` `num_iterations` 默认 1，`benchmark_news.py` 不覆盖。L89 `min(max_iterations, 1) = 1`，只生成 8 个候选评估一次就结束。叠加 P0-1（不含 initial），雪上加霜。

---

### 🟡 P2-2：生成候选的示例只用前 3 条，且前 3 条全是「财经」类

APE (`:170`)、OPRO (`:258`)、DSPy (`:235`) 都用 `dataset[:3]` 构造任务描述。实测 `examples/ag_news_train.json` 前 3 条 target 全是「财经」：

```json
[{"input": "从大宗商品价格下跌说开去", "target": "财经"},
 {"input": "建信基金获7亿美元QDII额度", "target": "财经"},
 {"input": "华宝兴业基金股东变更 领先资产持有49%", "target": "财经"}]
```

LLM 看到的示例只覆盖 14 类中的 1 类，生成的 prompt 可能偏财经、泛化性差。

---

### 🟡 P2-3：兜底 padding 在 instruction 前加标记前缀

当 LLM 生成的候选不足数量时，各优化器用前缀 padding：

| 优化器 | 位置 | padding 前缀 |
|--------|------|--------------|
| APE | `:226` | `[variant n]\n` |
| DSPy | `:381` | `(variation n)\n` |
| PromptBreeder | `:365` | `(variant n)\n` |
| EVOPrompt | `:398` | `[variant n]\n` |
| SPO | `:343` | `[semantic angle..., var n]\n` |

这些前缀改变 prompt 开头（「你是一个新闻分类专家」→「[variant 3]\n你是一个...」），可能干扰模型角色认知。OPRO 的 padding 改用通用英文模板（无类别列表），但因含 initial_prompt 不影响 best。

---

### 🟡 P2-4：DSPy bootstrap 匹配过松，收集错误格式 few-shot

`dspy_optimizer/optimizer.py:296`：

```python
if pred == tgt or tgt in pred:
    few_shots.append({"input": inp, "output": tgt})
```

`tgt in pred` 极松：若 pred = 「该新闻属于财经类别」，tgt = 「财经」，匹配成功，把这个**格式错误**的样本当作「正确 few-shot」收集。这些错误格式的 few-shot 喂给 LLM 生成新候选，会**强化错误输出格式**，形成恶性循环。

---

## 四、横向对比总结表

| 维度 | APE | OPRO | DSPy | PromptBreeder | EVOPrompt | SPO |
|------|-----|------|------|---------------|-----------|-----|
| meta-prompt 语言 | 英文 | 英文+数学few-shot | 英文 | 英文 | 英文 | 英文 |
| 保留 `{input}` 约束 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 保留输出格式约束 | ❌ | ❌ | ❌ | ❌（且鼓励改格式） | ❌（且鼓励改格式） | ❌（且鼓励改格式/加CoT） |
| 训练评估数据 | train 全量(200) | train 全量 | train 全量 | train 全量 | train 全量 | train 全量 |
| 每轮候选数 | 8 | 8 | 6 | 种群 10 | 种群 12 | ~9-12 |
| 迭代轮数 | **1（默认）** | 3 | 3(可早停) | 3 | 3(可早停) | 2-3(早停) |
| **含 initial_prompt** | **❌（致命）** | ✅ | ✅ | ✅ | ✅ | ✅ |
| padding 前缀污染 | [variant n] | 通用英文模板 | (variation n) | (variant n) | [variant n] | [semantic...] |
| 示例偏科(前3全财经) | 是 | 是 | 是 | — | — | — |
| 变异破坏格式风险 | 中 | 中 | 中 | **高** | **高** | **高** |

---

## 五、修复方案（按优先级）

### 修复 1（P0-1，最高性价比）：给 APE 补上 initial_prompt 基准候选

**位置**：`src/prompt_evolution/optimizers/ape/optimizer.py`，在 `optimize` 开头（L87 之后）加入：

```python
# 评估初始 prompt 作为基准，保证 best 不会劣于 baseline
initial_score = await self.evaluator.evaluate(
    prompt=initial_prompt, dataset=dataset, model_provider=self.model_provider,
)
initial_prompt.score = initial_score
all_candidates.append(initial_prompt)
```

仿照 `opro/optimizer.py:143-149`。改动约 5 行，预计 APE 立即追平 baseline。

### 修复 2（P0-2）：在所有 meta-prompt 增加输出格式保留约束

在每个优化器的 system prompt / placeholder_constraint 中追加（中英文皆可）：

```text
CRITICAL: 必须原样保留原 prompt 的输出格式约束与末尾输出引导
（如「只输出类别名称」和「类别：」）。不得改变输出格式，
不得添加「逐步分析/step by step」等推理引导，
不得在输出中增加前缀或解释。输出必须是裸类别名。
```

需修改的文件：6 个优化器的 system prompt 定义处。

### 修复 3（P1-1）：meta-prompt 中文化 + 替换 OPRO 数学 few-shot

将 6 个优化器的 system prompt 改为中文，few-shot 示例改为同领域分类任务示例。重点修改 `opro/optimizer.py:45-68` 的 `_DEFAULT_META_FEWSHOT`，例如：

```python
_DEFAULT_META_FEWSHOT = """以下是优化示例：
历史记录：
```
得分: 0.83
Prompt: "你是新闻分类专家。根据标题判断类别：科技、股票...只输出类别名。标题：{input} 类别："
```
```
得分: 0.88
Prompt: "你是资深编辑。从下列14类中选最贴切的：...。仅输出类别名，不要解释。标题：{input} 类别："
```
"""
```

### 修复 4（P1-2）：增大训练评估样本 + 引入 hold-out 验证

两种方案任选其一：

**方案 A（简单）**：benchmark 默认 `--train-samples 0`（用全部 1400 条训练），降低噪声。代价是耗时增加。

**方案 B（更稳）**：从 train_data 切出 hold-out 验证集（如 20%），优化器用剩余 80% 训练，最终在 hold-out 上选 best，再在 test 上报最终分。避免「train 选最优 → test 暴露过拟合」。

### 修复 5（P1-3）：弱化变异算子对输出格式的破坏

将 PromptBreeder / EVOPrompt / SPO 变异策略中的「Change the output format specification」「add think step by step」改为：

```text
变异策略（任选其一）：
- 重新表述核心指令，使其更清晰
- 调整角色/语气以更贴合任务
- 增加/精简约束（但不得改变输出格式）
- 保持原 prompt 的输出格式与末尾引导不变
```

### 修复 6（P2，批量清理）

- **P2-1**：APE `num_iterations` 默认改为 3
- **P2-2**：示例采样改为分层抽样覆盖多类别（`dataset[:3]` → 每类抽 1 条，共 14 条）
- **P2-3**：padding 不再加前缀，直接复用 `initial_prompt.instruction` 或 `parent.instruction`
- **P2-4**：DSPy bootstrap 匹配改为 `pred.strip() == tgt.strip()`（严格相等），避免收集错误格式 few-shot

---

## 六、验证建议（遵循小样本先验证）

按用户「先诊断后修复，小样本（50 条）快速验证通过后再全量」的习惯：

1. **先做修复 1 + 修复 2**（影响最大、改动最小）
2. 用小样本验证：`python benchmark_news.py --methods ape opro --train-samples 50 --eval-samples 50 --no-checkpoint`
3. 确认 APE / OPRO 至少追平 baseline 后，再逐步加修复 3-6
4. 最后全量基准测试：`python benchmark_news.py --no-checkpoint`（清空 checkpoint 从头跑）

---

## 七、根因影响度评估

| 根因 | 影响范围 | 对 APE | 对其余 5 个 | 修复难度 |
|------|----------|--------|-------------|----------|
| P0-1 APE 不含 initial | APE 独有 | 致命 | 无 | 极低（5行） |
| P0-2 格式约束丢失 | 全部 | 高 | 高 | 低 |
| P1-1 中英不一致/few-shot误导 | 全部 | 中 | 高（OPRO） | 中 |
| P1-2 小样本过拟合 | 5个(不含APE) | — | 高 | 中 |
| P1-3 变异破坏格式 | 3个变异类 | — | 高 | 低 |
| P2 其他 | 部分优化器 | 中 | 低 | 低 |

**核心结论**：现象「所有优化器劣于 baseline」由多重根因叠加导致。APE 劣化最严重且最确定（P0-1 机制性缺陷）；其余 5 个含 initial_prompt 但仍劣化，主要由 P0-2（格式约束丢失）+ P1-1（中英不一致）+ P1-2（小样本过拟合）叠加导致。修复 1-3 后重跑 benchmark，预计至少 APE 能追平 baseline，其余优化器劣化幅度显著收窄。
