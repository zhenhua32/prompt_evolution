"""Gradio Web UI — Prompt 迭代神器主界面。"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from loguru import logger

from prompt_evolution.core.models import PromptCandidate
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.evaluation.metrics.exact_match import ExactMatchMetric
from prompt_evolution.evaluation.metrics.f1_score import F1ScoreMetric
from prompt_evolution.optimizers.factory import create_optimizer, list_optimizers
from prompt_evolution.providers.litellm_provider import LiteLLMProvider

# ---------------------------------------------------------------------------
# 全局状态 & 历史记录持久化
# ---------------------------------------------------------------------------

_HISTORY_FILE = Path(".prompt_evolution_history.jsonl")

_LAST_RESULT: Optional[Dict] = None


def _save_history(record: Dict) -> None:
    """将一条优化记录追加到本地历史文件。"""
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_history() -> List[Dict]:
    """读取本地历史文件，返回记录列表。"""
    if not _HISTORY_FILE.exists():
        return []
    records = []
    with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ---------------------------------------------------------------------------
# 模型调用辅助
# ---------------------------------------------------------------------------

def _get_provider(model_name: str, api_key: str, api_base: str = "") -> LiteLLMProvider:
    key = api_key.strip() or None
    base = api_base.strip() or None
    return LiteLLMProvider(model=model_name, api_key=key, api_base=base)


def _load_dataset(dataset_file: Any) -> Tuple[List[Dict], str]:
    """加载上传的数据集文件，返回 (data, status_msg)。"""
    if dataset_file is None:
        return [], "⚠️ 请先上传数据集文件（JSON 格式）。"
    try:
        path = Path(dataset_file) if isinstance(dataset_file, str) else Path(dataset_file.name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], f"❌ 数据集格式错误：顶级应为 JSON 数组，实际为 {type(data).__name__}"
        return data, f"✅ 数据集加载成功：{len(data)} 条"
    except Exception as exc:
        return [], f"❌ 数据集加载失败：{exc}"


async def _run_prompt_on_dataset(
    prompt_text: str, dataset: List[Dict], provider: LiteLLMProvider
) -> List[str]:
    """将 prompt 应用到数据集每条样本，返回模型预测列表。"""
    predictions: List[str] = []
    for item in dataset:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": item.get("input", "")},
        ]
        try:
            resp = await provider.agenerate(messages=messages, temperature=0.0)
            predictions.append(resp.choices[0].message.content.strip())
        except Exception as exc:
            logger.warning("预测失败: {exc}")
            predictions.append("")
    return predictions


# ---------------------------------------------------------------------------
# Tab 1 回调：Prompt 优化
# ---------------------------------------------------------------------------

async def run_optimization(
    initial_prompt: str,
    model_name: str,
    api_key: str,
    base_url: str,
    optimizer_name: str,
    dataset_file: Any,
    num_candidates: int,
    max_iters: int,
    progress=gr.Progress(),
) -> Tuple[str, str, str, Dict]:
    """运行优化，返回 (结果 Markdown, 状态, 历史文本, 结果字典)。"""
    global _LAST_RESULT

    if not initial_prompt.strip():
        return "", "❌ 请输入初始 Prompt。", "", {}
    data, msg = _load_dataset(dataset_file)
    if not data:
        return "", msg, "", {}

    progress(0.05, desc="初始化组件...")
    try:
        provider = _get_provider(model_name, api_key, base_url)
        evaluator = Evaluator(metrics=[AccuracyMetric(), ExactMatchMetric(), F1ScoreMetric()])
        optimizer = create_optimizer(
            name=optimizer_name,
            model_provider=provider,
            evaluator=evaluator,
            config={"num_candidates": num_candidates},
        )
    except Exception as exc:
        return "", f"❌ 初始化失败：{exc}", "", {}

    initial = PromptCandidate(id="initial", instruction=initial_prompt.strip())
    progress(0.1, desc="正在优化...")
    try:
        result = await optimizer.optimize(
            initial_prompt=initial,
            dataset=data,
            max_iterations=max_iters,
        )
    except Exception as exc:
        logger.exception("优化失败")
        return "", f"❌ 优化失败：{exc}", "", {}

    _LAST_RESULT = result.model_dump(mode="json")

    # 保存历史
    _save_history({
        "type": "optimization",
        "optimizer": optimizer_name,
        "model": model_name,
        "best_score": result.best_prompt.score,
        "best_prompt": result.best_prompt.instruction,
        "elapsed_time_s": result.elapsed_time_s,
        "num_candidates_evaluated": result.num_candidates_evaluated,
    })

    result_text = (
        f"## 最优 Prompt\n\n```\n{result.best_prompt.instruction}\n```\n\n"
        f"**得分：** {result.best_prompt.score:.4f}\n\n"
        f"**耗时：** {result.elapsed_time_s:.1f}s\n\n"
        f"**评估候选数：** {result.num_candidates_evaluated}"
    )

    history_text = ""
    for h in result.optimization_history:
        history_text += (
            f"轮次 {h.get('iteration', '?')}: "
            f"候选数={h.get('num_candidates', '?')}, "
            f"最优得分={h.get('best_score', 0.0):.4f}\n"
        )

    status = f"✅ 优化完成！最优得分：{result.best_prompt.score:.4f}"
    return result_text, status, history_text, _LAST_RESULT


def download_result(result_dict: Optional[Dict]) -> Optional[Tuple[str, str]]:
    """将结果字典保存为临时 JSON 文件供下载。"""
    if not result_dict:
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(result_dict, tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name, "optimization_result.json"


# ---------------------------------------------------------------------------
# Tab 2 回调：Prompt 评估
# ---------------------------------------------------------------------------

async def run_evaluation(
    prompt_text: str,
    model_name: str,
    api_key: str,
    base_url: str,
    dataset_file: Any,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """对单条 Prompt 在数据集上运行评估，返回 (结果 Markdown, 状态)。"""
    if not prompt_text.strip():
        return "", "❌ 请输入 Prompt。"
    data, msg = _load_dataset(dataset_file)
    if not data:
        return "", msg

    progress(0.1, desc="初始化模型...")
    try:
        provider = _get_provider(model_name, api_key, base_url)
    except Exception as exc:
        return "", f"❌ 模型初始化失败：{exc}"

    progress(0.3, desc="正在推理（可能耗时较长）...")
    try:
        predictions = await _run_prompt_on_dataset(prompt_text.strip(), data, provider)
    except Exception as exc:
        logger.exception("推理失败")
        return "", f"❌ 推理失败：{exc}"

    references = [item.get("target", item.get("output", "")) for item in data]

    progress(0.8, desc="计算评估指标...")
    evaluator = Evaluator(metrics=[AccuracyMetric(), ExactMatchMetric(), F1ScoreMetric()])
    scores = evaluator.evaluate(predictions, references)

    # 格式化结果
    lines = ["| 指标 | 得分 |", "|------|------|"]
    for name, score in scores.items():
        lines.append(f"| {name} | {score:.4f} |")
    result_md = "\n".join(lines)

    # 显示若干样本对比
    result_md += "\n\n### 样本对比（前 5 条）\n\n"
    for i, (pred, ref) in enumerate(zip(predictions[:5], references[:5])):
        result_md += f"**样本 {i+1}**\n- 预测：`{pred[:200]}`\n- 参考：`{ref[:200]}`\n\n"

    status = f"✅ 评估完成！共 {len(data)} 条，Accuracy={scores.get('accuracy', 0):.4f}"
    return result_md, status


# ---------------------------------------------------------------------------
# Tab 3 数据：优化历史
# ---------------------------------------------------------------------------

def _format_history_table() -> str:
    """将历史记录格式化为 Markdown 表格。"""
    records = _load_history()
    if not records:
        return "*暂无历史记录，快去「Prompt 优化」Tab 跑一次吧！*"

    lines = ["| 时间 | 优化器 | 模型 | 最优得分 | 耗时(s) |", "|------|--------|------|----------|---------|"]
    for r in reversed(records[-50:]):  # 最多显示最近 50 条
        ts = r.get("timestamp", "?")
        opt = r.get("optimizer", "?")
        model = r.get("model", "?")
        score = r.get("best_score", 0.0)
        elapsed = r.get("elapsed_time_s", 0.0)
        lines.append(f"| {ts} | {opt} | {model} | {score:.4f} | {elapsed:.1f} |")
    return "\n".join(lines)


def refresh_history() -> str:
    """刷新历史记录表格。"""
    return _format_history_table()


def view_best_prompt(history_table: str) -> Tuple[str, str]:
    """从最近一条历史记录中提取最优 Prompt 并显示。"""
    records = _load_history()
    if not records:
        return "", "暂无历史记录。"
    latest = records[-1]
    best_prompt = latest.get("best_prompt", "")
    return best_prompt, f"已加载最近一次优化的最优 Prompt（得分：{latest.get('best_score', 0):.4f}）"


# ---------------------------------------------------------------------------
# UI 布局
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """创建 Gradio 应用。"""

    with gr.Blocks(
        title="Prompt 迭代神器",
        theme=gr.themes.Soft(),
        css="""
        .optimize-btn {background-color: #2563eb !important; color: white !important;}
        .eval-btn {background-color: #059669 !important; color: white !important;}
        """,
    ) as app:

        gr.Markdown("# Prompt 迭代神器")
        gr.Markdown("集成多种 Prompt 自动优化算法，可视化评估，一键找到最优 Prompt。")

        with gr.Tabs():

            # ======================== Tab 1: Prompt 优化 ========================
            with gr.Tab(" Prompt 优化", id="tab-optimize"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 配置")

                        optimizer_dropdown = gr.Dropdown(
                            choices=list_optimizers(),
                            value=list_optimizers()[0] if list_optimizers() else None,
                            label="优化算法",
                            interactive=True,
                        )
                        model_textbox = gr.Textbox(
                            value="openai/gpt-4o",
                            label="模型标识（LiteLLM 格式）",
                            placeholder="openai/gpt-4o / anthropic/claude-3-5-sonnet / ...",
                        )
                        api_key_textbox = gr.Textbox(
                            value="",
                            label="API Key（可选，也可用 .env 文件配置）",
                            type="password",
                        )
                        base_url_textbox = gr.Textbox(
                            value="",
                            label="Base URL（可选，OpenAI 兼容接口自定义地址）",
                            placeholder="http://localhost:8000/v1  （留空使用默认地址）",
                        )
                        num_candidates_slider = gr.Slider(
                            minimum=3, maximum=50, value=10, step=1, label="候选 Prompt 数（APE）",
                        )
                        max_iters_slider = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1, label="最大迭代轮数",
                        )

                        gr.Markdown("### 初始 Prompt")
                        initial_prompt_textbox = gr.Textbox(
                            value="你是一个有用的助手。",
                            label="初始 Prompt（系统指令）",
                            lines=4,
                        )

                        gr.Markdown("### 数据集")
                        dataset_file_optimize = gr.File(
                            label="上传数据集（JSON 格式，数组，每项含 input 和 target）",
                            file_types=[".json"],
                        )
                        dataset_status = gr.Textbox(label="数据集状态", interactive=False)

                        optimize_btn = gr.Button("开始优化", elem_classes=["optimize-btn"])

                    with gr.Column(scale=2):
                        gr.Markdown("### 优化结果")
                        status_textbox = gr.Textbox(label="状态", interactive=False)
                        result_markdown = gr.Markdown(label="最优 Prompt")

                        gr.Markdown("### 优化历史")
                        history_textbox = gr.Textbox(
                            label="每轮迭代摘要", lines=8, interactive=False,
                        )

                        gr.Markdown("### 下载结果")
                        download_btn = gr.Button("下载完整结果（JSON）")
                        download_file = gr.File(label="下载")
                        hidden_state = gr.State(value={})

            # ======================== Tab 2: Prompt 评估 ========================
            with gr.Tab(" Prompt 评估", id="tab-eval"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 配置")
                        eval_model_textbox = gr.Textbox(
                            value="openai/gpt-4o",
                            label="模型标识（LiteLLM 格式）",
                        )
                        eval_api_key_textbox = gr.Textbox(
                            value="",
                            label="API Key（可选）",
                            type="password",
                        )
                        eval_base_url_textbox = gr.Textbox(
                            value="",
                            label="Base URL（可选，OpenAI 兼容接口自定义地址）",
                            placeholder="http://localhost:8000/v1  （留空使用默认地址）",
                        )
                        gr.Markdown("### Prompt")
                        eval_prompt_textbox = gr.Textbox(
                            value="你是一个有用的助手。请根据以下问题给出准确回答：",
                            label="Prompt（系统指令）",
                            lines=4,
                        )
                        gr.Markdown("### 数据集")
                        eval_dataset_file = gr.File(
                            label="上传数据集（JSON 格式）",
                            file_types=[".json"],
                        )
                        eval_dataset_status = gr.Textbox(label="数据集状态", interactive=False)
                        eval_btn = gr.Button("开始评估", elem_classes=["eval-btn"])

                    with gr.Column(scale=2):
                        gr.Markdown("### 评估结果")
                        eval_status_textbox = gr.Textbox(label="状态", interactive=False)
                        eval_result_markdown = gr.Markdown(label="指标得分")

            # ======================== Tab 3: 优化历史 ========================
            with gr.Tab(" 优化历史", id="tab-history"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 操作")
                        refresh_btn = gr.Button("刷新历史")
                        view_best_btn = gr.Button("查看最近最优 Prompt")
                        gr.Markdown("*(历史记录保存在 `~/.prompt_evolution/history.jsonl`，永久留存)*")

                    with gr.Column(scale=2):
                        gr.Markdown("### 历史记录")
                        history_table = gr.Markdown(_format_history_table())
                        gr.Markdown("### 最优 Prompt 查看")
                        best_prompt_textbox = gr.Textbox(
                            label="最优 Prompt", lines=6, interactive=False,
                        )
                        best_prompt_status = gr.Textbox(label="状态", interactive=False)

        # ======================== 事件绑定 ========================

        # 数据集上传后自动加载（Tab 1）
        dataset_file_optimize.change(
            fn=lambda f: _load_dataset(f)[1],
            inputs=[dataset_file_optimize],
            outputs=[dataset_status],
        )

        # 数据集上传后自动加载（Tab 2）
        eval_dataset_file.change(
            fn=lambda f: _load_dataset(f)[1],
            inputs=[eval_dataset_file],
            outputs=[eval_dataset_status],
        )

        # 点击"开始优化"（Tab 1）
        optimize_btn.click(
            fn=run_optimization,
            inputs=[
                initial_prompt_textbox,
                model_textbox,
                api_key_textbox,
                base_url_textbox,
                optimizer_dropdown,
                dataset_file_optimize,
                num_candidates_slider,
                max_iters_slider,
            ],
            outputs=[result_markdown, status_textbox, history_textbox, hidden_state],
        )

        # 点击"下载结果"（Tab 1）
        download_btn.click(
            fn=download_result,
            inputs=[hidden_state],
            outputs=[download_file],
        )

        # 点击"开始评估"（Tab 2）
        eval_btn.click(
            fn=run_evaluation,
            inputs=[
                eval_prompt_textbox,
                eval_model_textbox,
                eval_api_key_textbox,
                eval_base_url_textbox,
                eval_dataset_file,
            ],
            outputs=[eval_result_markdown, eval_status_textbox],
        )

        # 点击"刷新历史"（Tab 3）
        refresh_btn.click(
            fn=refresh_history,
            inputs=[],
            outputs=[history_table],
        )

        # 点击"查看最近最优 Prompt"（Tab 3）
        view_best_btn.click(
            fn=view_best_prompt,
            inputs=[history_table],
            outputs=[best_prompt_textbox, best_prompt_status],
        )

    return app


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """启动 Web UI。"""
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    launch_ui()
