# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Rich UI for SigMamba.

All TUI output (logs, progress bars, tables, panels) is routed through the console instance.
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console(highlight=False, log_path=False, force_terminal=True, force_interactive=True)


def make_training_progress() -> Progress:
    """
    Returns a Progress instance styled for the main training loop.
    Custom fields expected per task: loss (float) and auc (float).
    """
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}", justify="left"),
        BarColumn(bar_width=36, style="cyan", complete_style="bright_cyan"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("[yellow]loss[/yellow] [white]{task.fields[loss]}"),
        TextColumn("[green] auc[/green] [white]{task.fields[auc]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
        expand=False,
    )


def make_test_progress() -> Progress:
    """Returns a Progress instance for iterating over test videos."""
    return Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("[bold magenta]{task.description}", justify="left"),
        BarColumn(bar_width=36, style="magenta", complete_style="bright_magenta"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
        expand=False,
    )


def make_progress(color: str = "blue") -> Progress:
    """
    Generic progress bar for utility scripts (extractor, list generator).

    Args:
        color: Rich color string applied to the spinner and bar.
    """
    return Progress(
        SpinnerColumn(style=color),
        TextColumn(f"[bold {color}]{{task.description}}", justify="left"),
        BarColumn(bar_width=36, style=color, complete_style=f"bright_{color}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=8,
        expand=False,
    )


def print_run_summary(
    model_name: str,
    backend: str,
    total_params: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    feature_size: int,
    model_dim: int,
    model_depth: int,
    seg_num: int,
    lambda_sparse: float,
    lambda_smooth: float,
) -> None:
    """Prints a startup panel with model configuration."""
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column("key", style="bold cyan", no_wrap=True)
    t.add_column("value", style="white")

    t.add_row("Backend", backend)
    t.add_row("Device", str(device))
    t.add_row("Parameters", f"{total_params:,}")
    t.add_row("Feature dim", str(feature_size))
    t.add_row("Model dim", str(model_dim))
    t.add_row("Depth", str(model_depth))
    t.add_row("Segments", str(seg_num))
    t.add_row("Batch size", str(batch_size))
    t.add_row("Learning rate", str(learning_rate))
    t.add_row("Sparsity weight", str(lambda_sparse))
    t.add_row("Smoothness weight", str(lambda_smooth))

    console.print(Panel(t, title=f"[bold white]{model_name}[/bold white]", border_style="cyan", expand=False))


def print_extractor_summary(
    model_id: str,
    dataset_path: str,
    save_dir: str,
    num_gpus: int,
    batch_size: int,
    frame_interval: int,
    total: int,
    done: int,
    remaining: int,
) -> None:
    """Prints a startup panel for the feature extractor."""
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column("key", style="bold yellow", no_wrap=True)
    t.add_column("value", style="white")

    t.add_row("Model", model_id)
    t.add_row("Input", dataset_path)
    t.add_row("Output", save_dir)
    t.add_row("GPUs", str(num_gpus))
    t.add_row("Batch size", str(batch_size))
    t.add_row("Frame interval", f"1 / {frame_interval}")
    t.add_row("Total videos", str(total))
    t.add_row("Already done", str(done))
    t.add_row("Remaining", str(remaining))

    console.print(
        Panel(
            t, title="[bold white]SigMamba Dataset Feature Extractor[/bold white]", border_style="yellow", expand=False
        )
    )


def print_val_results(
    step: int,
    auc: float,
    ap: float,
    f1: float,
    best: float,
    metric: str,
    print_to: Console | None = None,
) -> None:
    """
    Prints validation-results row.

    print_to can be a progress.console so the output is rendered
    above the live progress bar rather than below it.
    """
    out = print_to or console
    is_best = {"AUC": auc, "AP": ap, "F1": f1}[metric] >= best
    star = " [bold yellow]★ best[/bold yellow]" if is_best else ""
    out.log(
        f"[cyan]step {step:>5}[/cyan]  "
        f"AUC [bold white]{auc:.4f}[/bold white]  "
        f"AP [bold white]{ap:.4f}[/bold white]  "
        f"F1 [bold white]{f1:.4f}[/bold white]"
        f"{star}"
    )


def print_checkpoint_saved(name: str, step: int, duration: str, print_to: Console | None = None) -> None:
    """Logs checkpoint-saved confirmation."""
    out = print_to or console
    out.log(f"[green]Checkpoint saved[/green] [white]{name!r}[/white]  step {step}  elapsed {duration}")


def print_export_summary(meta: dict, vision_model_id: str) -> None:
    """Prints a panel with checkpoint metadata before export begins."""
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column("key", style="bold yellow", no_wrap=True)
    t.add_column("value", style="white")

    t.add_row("Step", str(meta.get("step", "?")))
    t.add_row("AUC", str(meta.get("auc", "?")))
    t.add_row("AP", str(meta.get("average_precision", "?")))
    t.add_row("F1", str(meta.get("f1_score", "?")))
    t.add_row("Duration", str(meta.get("training_duration", "?")))
    t.add_row("Vision", vision_model_id)

    console.print(Panel(t, title="[bold cyan]Checkpoint Metadata[/bold cyan]", border_style="cyan", expand=False))


def print_export_success(output_dir: str) -> None:
    """Prints the success panel and usage snippet after a completed export."""
    console.print(
        Panel(
            "[bold white]Model ready for distribution.[/bold white]",
            title="[bold magenta]Export Complete...[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )
