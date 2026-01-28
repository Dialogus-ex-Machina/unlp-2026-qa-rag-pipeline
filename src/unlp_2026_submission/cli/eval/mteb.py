from typing import Annotated
import logging
import math
import typer
from mteb.results import ModelResult
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from unlp_2026_submission.config import Config
from unlp_2026_submission.evals.mteb_retrieval import evaluate_mteb, MTEBTaskName

app = typer.Typer()

@app.command('mteb')
def evaluate_mteb_command(
        embeddings_model_name: Annotated[
            str,
            typer.Option("--embedding-model", "-em")
        ] = None,
        task_name: Annotated[
            MTEBTaskName,
            typer.Option("--task", "-t")
        ] = MTEBTaskName.QA,
        logging_level: Annotated[int, typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Run mteb for a given tasks
    """
    logging.basicConfig(level=logging_level)

    config = Config(
        embeddings_model_name=embeddings_model_name
    )
    result = evaluate_mteb(
        config=config,
        task_name=task_name,
    )

    print_results(result)


def print_results(result: ModelResult):
    console = Console()

    # ---- model header ----
    console.print(
        Panel.fit(
            f"[bold]Model:[/bold] {result.model_name}",
            title="Evaluation",
        )
    )

    # ---- loop over tasks ----
    for task_result in result.task_results:
        task_name = task_result.task_name
        metrics = task_result.scores["test"][0]

        table = Table(
            title=f"Task: {task_name} | split: test",
            show_lines=False,
        )

        table.add_column("Metric", style="bold cyan", no_wrap=True)
        table.add_column("Value", justify="right")

        for k, v in metrics.items():
            if isinstance(v, float):
                value = "NaN" if math.isnan(v) else f"{v:.5f}"
            else:
                value = str(v)

            table.add_row(k, value)

        console.print(table)
