"""CLI 入口实现 — 支持 ``prompt-evol`` 和 ``prompt-evolution`` 命令。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from loguru import logger

from prompt_evolution import __version__
from prompt_evolution.core.models import PromptCandidate, OptimizationResult
from prompt_evolution.evaluation.evaluator import Evaluator
from prompt_evolution.evaluation.metrics.accuracy import AccuracyMetric
from prompt_evolution.optimizers.factory import create_optimizer, list_optimizers
from prompt_evolution.providers.litellm_provider import LiteLLMProvider

# 加载 .env
load_dotenv()

app = typer.Typer(
    name="prompt-evolution",
    help="Prompt 迭代神器 — 一站式 Prompt 自动优化工具",
    add_completion=False,
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 子命令：optimize
# ---------------------------------------------------------------------------

optimize_app = typer.Typer(help="运行 Prompt 优化")
app.add_typer(optimize_app, name="optimize")


@optimize_app.callback(invoke_without_command=True)
def optimize(
    ctx: typer.Context,
    initial_prompt: str = typer.Option(..., "--prompt", "-p", help="初始 Prompt 文本。"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="数据集 JSON 文件路径。"),
    method: str = typer.Option("ape", "--method", "-m", help=f"优化方法，可选: {', '.join(list_optimizers())}"),
    model: str = typer.Option("openai/gpt-4o", "--model", help="LiteLLM 模型标识。"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API Key（也可用环境变量）。"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL（OpenAI 兼容接口自定义地址）。"),
    max_iterations: int = typer.Option(5, "--max-iters", help="最大迭代轮数。"),
    num_candidates: int = typer.Option(10, "--num-candidates", help="每轮候选 Prompt 数（APE）。"),
    disable_thinking: Optional[bool] = typer.Option(
        None,
        "--disable-thinking/--enable-thinking",
        help="关闭或开启模型的 thinking/reasoning 输出；默认读取 DISABLE_THINKING。",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="结果输出 JSON 路径。"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """运行 Prompt 优化并输出最优结果。"""
    if ctx.invoked_subcommand is not None:
        return

    if verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # 读取数据集
    if not dataset.exists():
        typer.echo(f"错误：数据集文件不存在: {dataset}", err=True)
        raise typer.Exit(code=1)
    with open(dataset, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 初始化组件
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key and model.startswith("openai/"):
        typer.echo("警告：未提供 API Key，请设置 OPENAI_API_KEY 环境变量或用 --api-key。", err=True)

    # 解析 base_url：优先用命令行参数，其次用环境变量
    api_base = base_url or os.environ.get("OPENAI_BASE_URL", "") or None
    if disable_thinking is None:
        disable_thinking = _env_flag("DISABLE_THINKING")

    provider = LiteLLMProvider(
        model=model,
        api_key=key,
        api_base=api_base,
        disable_thinking=disable_thinking,
    )
    evaluator = Evaluator(metrics=[AccuracyMetric()])
    optimizer = create_optimizer(
        name=method,
        model_provider=provider,
        evaluator=evaluator,
        config={"num_candidates": num_candidates},
    )

    # 构造初始 PromptCandidate
    initial = PromptCandidate(id="initial", instruction=initial_prompt)

    typer.echo(f"开始优化：方法={method}, 模型={model}, 数据集={len(data)} 条")
    if api_base:
        typer.echo(f"Base URL: {api_base}")
    if disable_thinking:
        typer.echo("Thinking: disabled")
    typer.echo("正在运行，请稍候...")

    # 运行优化
    result: OptimizationResult = asyncio.run(
        optimizer.optimize(
            initial_prompt=initial,
            dataset=data,
            max_iterations=max_iterations,
        )
    )

    # 展示结果
    typer.echo("\n" + "=" * 60)
    typer.echo("优化完成！")
    typer.echo(f"最优 Prompt:\n{result.best_prompt.instruction}")
    typer.echo(f"得分: {result.best_prompt.score:.4f}")
    typer.echo(f"耗时: {result.elapsed_time_s:.1f}s")
    typer.echo(f"总费用: ${result.total_cost_usd:.4f}")
    typer.echo(f"评估候选数: {result.num_candidates_evaluated}")
    typer.echo("=" * 60)

    # 保存结果
    out_path = output or Path("./optimization_result.json")
    out_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    typer.echo(f"结果已保存至: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# 子命令：list
# ---------------------------------------------------------------------------

@app.command("list")
def list_methods() -> None:
    """列出所有可用的优化方法。"""
    typer.echo("可用优化方法：")
    for name in list_optimizers():
        typer.echo(f"  - {name}")


# ---------------------------------------------------------------------------
# 子命令：ui
# ---------------------------------------------------------------------------

@app.command("ui")
def launch_ui(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址。"),
    port: int = typer.Option(7860, "--port", "-p", help="监听端口。"),
    share: bool = typer.Option(False, "--share", help="创建公开分享链接。"),
) -> None:
    """启动 Web UI（Gradio）。"""
    try:
        from prompt_evolution.ui.app import create_app
    except ImportError:
        typer.echo("错误：Web UI 依赖未安装，请安装 gradio: pip install gradio", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"启动 Web UI：http://{host}:{port}")
    app_ui = create_app()
    app_ui.launch(server_name=host, server_port=port, share=share)


# ---------------------------------------------------------------------------
# 主命令：version
# ---------------------------------------------------------------------------

@app.command("version")
def version() -> None:
    """显示版本信息。"""
    typer.echo(f"prompt-evolution {__version__}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
