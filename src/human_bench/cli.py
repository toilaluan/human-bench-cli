import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from human_bench.catalog import export_catalog, load_catalog
from human_bench.config import load_config
from human_bench.manifest import build_manifest, load_manifest
from human_bench.reporting import export_comparison, export_run, read_run
from human_bench.runner import evaluate_run

app = typer.Typer(help="Evaluate human-centric T2I generations from external image folders.")
prompts_app = typer.Typer(help="Export the fixed v1 prompt catalog.")
manifest_app = typer.Typer(help="Construct a prompt/sample/image mapping.")
app.add_typer(prompts_app, name="prompts")
app.add_typer(manifest_app, name="manifest")


def fail(exc: Exception):
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1) from exc


@prompts_app.command("export")
def prompts_export(
    output: Annotated[Path, typer.Option(help="Destination JSON or JSONL file.")],
    rich: Annotated[bool, typer.Option(help="Include facts and contrasts as JSONL.")] = False,
):
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        export_catalog(output, rich)
        typer.echo(f"Exported 160 fixed prompts to {output}")
    except (OSError, ValueError) as exc:
        fail(exc)


@manifest_app.command("build")
def manifest_build(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    try:
        count = build_manifest(folder, output)
        typer.echo(f"Mapped {count} images to {output}")
    except (OSError, ValueError) as exc:
        fail(exc)


@app.command("validate")
def validate(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)], exploratory: bool = False
):
    """Validate every image and prompt/sample coverage without calling an API."""
    try:
        items = load_manifest(manifest, load_catalog(), exploratory)
        typer.echo(
            f"Valid: {len(items)} images, {len({i.prompt_id for i in items})} prompts; "
            f"{'exploratory' if exploratory else 'standard'} protocol"
        )
    except (OSError, ValueError) as exc:
        fail(exc)


@app.command("evaluate")
def evaluate(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    resume: bool = False,
    preflight_only: bool = False,
):
    """Evaluate, checkpoint, and export results; resume preserves completed stages."""
    try:
        loaded = load_config(config)
        completed = -1

        def progress(rows):
            nonlocal completed
            count = sum(r["status"] == "complete" for r in rows)
            errors = sum(r["status"] == "error" for r in rows)
            if count != completed or errors:
                typer.echo(f"Completed {count}/{len(rows)}; errors {errors}")
                completed = count

        summary = asyncio.run(
            evaluate_run(
                loaded, manifest, output, resume, preflight_only=preflight_only, progress=progress
            )
        )
        typer.echo(
            f"{summary['status']}: {summary['completed_images']}/"
            f"{summary['expected_images']} images. Results: {output}"
        )
        if summary["status"] == "incomplete":
            raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("Interrupted. Checkpoints are saved; restart with --resume.", err=True)
        raise typer.Exit(130) from None
    except (OSError, ValueError, RuntimeError) as exc:
        fail(exc)


@app.command("report")
def report(folder: Annotated[Path, typer.Argument(exists=True, file_okay=False)]):
    """Re-export reports from a run database without inference."""
    try:
        read_run(folder)
        summary = export_run(folder)
        typer.echo(
            json.dumps(
                {k: summary[k] for k in ("name", "status", "completed_images", "expected_images")},
                indent=2,
            )
        )
    except (OSError, ValueError) as exc:
        fail(exc)


@app.command("compare")
def compare(
    folders: Annotated[list[Path], typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    """Compare compatible, complete runs and export paired confidence intervals."""
    try:
        export_comparison(folders, output)
        typer.echo(f"Comparison exported to {output} and {output.with_suffix('.csv')}")
    except (OSError, ValueError) as exc:
        fail(exc)


@app.command("review")
def review(
    runs_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    host: str = "127.0.0.1",
    port: int = 7860,
):
    """Launch a read-only Gradio app. No public sharing or model inference."""
    from human_bench.review import create_app

    try:
        create_app(runs_dir).launch(server_name=host, server_port=port, share=False)
    except (OSError, ValueError, RuntimeError) as exc:
        fail(exc)
