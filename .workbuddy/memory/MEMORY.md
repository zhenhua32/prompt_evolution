# Prompt Evolution 项目长期记忆

## 项目概览
Prompt 迭代神器：集成 APE/OPRO/DSPy/PromptBreeder/EVOPrompt/SPO 6 类优化器，LiteLLM 统一接口，BaseOptimizer 基类 + create_optimizer 工厂，Gradio 三标签页 Web UI。

## 基准测试任务
中文新闻分类（14分类，1400训练/420测试）。baseline 准确率 ≈0.85-0.86。
- 评测脚本：`benchmark_news.py`（已修复 train/test 混比，主指标为 test_score）
- 默认参数：`--train-samples 200 --eval-samples 100 --max-iters 3 --num-candidates 8`

## 优化器效果劣于 baseline 的核心根因（2026-06-23 定位）
1. **APE 不含 initial_prompt**：`ape/optimizer.py` 的 all_candidates 从不加 initial，机制性必然劣化。其余 5 个优化器都含。
2. **格式约束丢失（全部 6 个）**：meta-prompt 只约束 `{input}`，不约束「只输出类别名称」+ 末尾引导。LLM 改写致输出漂移，AccuracyMetric 严格相等判错。
3. **中英不一致**：meta-prompt 全英文，OPRO few-shot 是数学示例。
4. **小样本过拟合**：train 200/test 100 噪声大。
5. **变异算子破坏格式**：Breeder/EVO/SPO 鼓励"改输出格式"。

## 关键评测口径
- Evaluator 渲染：`instruction.replace("{input}", user_input)`，无占位符走兜底拼接
- AccuracyMetric：`pred.strip().lower() == ref.strip().lower()`（严格相等）
- 优化器评估温度 0.0，max_tokens 512，system_prompt=None
- 详见 `docs/RCA_优化效果差根因分析.md`

## 开发习惯（用户偏好）
- 先诊断后修复，修复前分析根因
- 小样本(50条)快速验证通过后再全量基准
- P0>P1>P2 优先级，"开干""开修"推进执行
- 中文回复，结构化输出（表格、代码块）
