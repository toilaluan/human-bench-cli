"""Read-only Gradio inspection of checkpoints, including runs still in progress."""

import json
from pathlib import Path

from human_bench.images import prepare_image
from human_bench.metrics import compare_rows, summarize
from human_bench.reporting import read_run

EXPORTS = (
    "results.jsonl",
    "results.csv",
    "summary.json",
    "summary.csv",
    "run.json",
    "attempts.json",
)


def discover_runs(root: Path) -> list[str]:
    return sorted(str(path.parent.resolve()) for path in root.rglob("run.sqlite"))


def filter_rows(rows, family="all", difficulty="all", status="all"):
    selected = []
    for row in rows:
        if family != "all" and row["family"] != family:
            continue
        if difficulty != "all" and row["difficulty"] != difficulty:
            continue
        scores = row.get("scores", {})
        matches = {
            "all": True,
            "adherence failures": scores.get("strict_adherence") == 0,
            "anatomy defects": any(
                a["verdict"] == "fail" for a in row.get("evaluation", {}).get("anatomy", [])
            ),
            "uncertain": any(
                a["verdict"] == "unassessable"
                for a in (
                    row.get("evaluation", {}).get("anatomy", [])
                    + row.get("evaluation", {}).get("facts", [])
                )
            ),
            "errors": row.get("status") == "error",
            "pending": row.get("status") == "pending",
        }
        if matches.get(status, False):
            selected.append(row)
    return selected


def inspect_row(metadata: dict, row: dict | None):
    if row is None:
        return None, "", "", {}
    prompt = next(p["prompt"] for p in metadata["catalog"] if p["prompt_id"] == row["prompt_id"])
    try:
        image = prepare_image(Path(row["image_path"]))
        image_error = None
    except (OSError, ValueError) as exc:
        image, image_error = None, str(exc)
    details = {
        "status": row["status"],
        "error": row.get("error"),
        "image_error": image_error,
        "scores": row.get("scores"),
        "evaluation": row.get("evaluation"),
        "image_sha256": row["image_sha256"],
    }
    return image, prompt, row.get("caption", ""), details


def create_app(root: Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the review app with: uv sync --extra review") from exc

    root = root.resolve()

    def checked_run(folder):
        path = Path(folder).resolve()
        if not path.is_relative_to(root):
            raise gr.Error("Select a run inside the configured runs directory")
        return read_run(path)

    def refresh():
        choices = discover_runs(root)
        return [gr.Dropdown(choices=choices) for _ in range(3)]

    def browse(folder, family, difficulty, status):
        if not folder:
            return {}, [], None, [], None, "", "", {}
        metadata, rows = checked_run(folder)
        selected = filter_rows(rows, family, difficulty, status)
        table = [
            [
                r["prompt_id"],
                r["sample_id"],
                r["family"],
                r["difficulty"],
                r["status"],
                r.get("scores", {}).get("strict_adherence"),
                r.get("scores", {}).get("anatomy_defect_free"),
                r.get("scores", {}).get("anatomy_assessable_fraction"),
            ]
            for r in selected
        ]
        downloads = [
            str(Path(folder) / name) for name in EXPORTS if (Path(folder) / name).is_file()
        ]
        first = inspect_row(metadata, selected[0] if selected else None)
        return (
            summarize(metadata, rows, bootstrap=False),
            table,
            (metadata, selected),
            downloads,
            *first,
        )

    def select_row(state, event: gr.SelectData):
        if not state:
            return None, "", "", {}
        metadata, rows = state
        index = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
        return inspect_row(metadata, rows[index] if 0 <= index < len(rows) else None)

    def comparison(left, right):
        if not left or not right:
            return {}, gr.Dropdown(choices=[], value=None)
        lmeta, lrows = checked_run(left)
        rmeta, rrows = checked_run(right)
        try:
            result = compare_rows([(lmeta, lrows), (rmeta, rrows)])
        except ValueError as exc:
            result = {"comparison_unavailable": str(exc), "inspection_available": True}
        shared = sorted(
            {(r["prompt_id"], r["sample_id"]) for r in lrows}
            & {(r["prompt_id"], r["sample_id"]) for r in rrows}
        )
        choices = [(f"{pid} / {sid}", json.dumps([pid, sid])) for pid, sid in shared]
        return result, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

    def show_pair(left, right, key):
        if not left or not right or not key:
            return None, "", "", {}, None, "", "", {}
        pid, sid = json.loads(key)
        views = []
        for folder in (left, right):
            meta, rows = checked_run(folder)
            row = next((r for r in rows if (r["prompt_id"], r["sample_id"]) == (pid, sid)), None)
            views.extend(inspect_row(meta, row))
        return tuple(views)

    choices = discover_runs(root)
    with gr.Blocks(title="Human Generation Benchmark") as app:
        gr.Markdown("# Human Generation Benchmark\nInspect results and compare T2I model runs.")
        refresh_button = gr.Button("Refresh runs")
        with gr.Tab("Inspect a run"):
            run = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Run")
            with gr.Row():
                family = gr.Dropdown(
                    ["all", *[f"T{i}" for i in range(1, 9)]], value="all", label="Family"
                )
                difficulty = gr.Dropdown(
                    ["all", "easy", "medium", "hard", "very_hard"], value="all", label="Difficulty"
                )
                status = gr.Dropdown(
                    [
                        "all",
                        "adherence failures",
                        "anatomy defects",
                        "uncertain",
                        "errors",
                        "pending",
                    ],
                    value="all",
                    label="Filter",
                )
            reload_button = gr.Button("Reload results")
            summary = gr.JSON(label="Summary and breakdowns (intervals are in exported reports)")
            table = gr.Dataframe(
                headers=[
                    "Prompt",
                    "Sample",
                    "Family",
                    "Difficulty",
                    "Status",
                    "Adherence",
                    "Visible anatomy",
                    "Anatomy coverage",
                ],
                interactive=False,
                label="Images — select a row to inspect",
            )
            state = gr.State()
            with gr.Row():
                image = gr.Image(label="Generated image", interactive=False)
                with gr.Column():
                    prompt = gr.Textbox(label="Generation prompt", interactive=False)
                    caption = gr.Textbox(label="Blind caption", interactive=False)
            detail = gr.JSON(label="Verdicts, evidence, anatomy, and diagnostics")
            downloads = gr.File(
                label="Latest exported results", file_count="multiple", interactive=False
            )
        with gr.Tab("Compare runs"):
            with gr.Row():
                left = gr.Dropdown(choices=choices, label="Reference run")
                right = gr.Dropdown(choices=choices, label="Comparison run")
            compare_button = gr.Button("Compare and align images")
            comparison_result = gr.JSON(label="Metrics and paired differences")
            key = gr.Dropdown(label="Shared prompt and sample")
            with gr.Row():
                with gr.Column():
                    li = gr.Image(label="Reference", interactive=False)
                    lp = gr.Textbox(label="Prompt", interactive=False)
                    lc = gr.Textbox(label="Blind caption", interactive=False)
                    ld = gr.JSON(label="Reference findings")
                with gr.Column():
                    ri = gr.Image(label="Comparison", interactive=False)
                    rp = gr.Textbox(label="Prompt", interactive=False)
                    rc = gr.Textbox(label="Blind caption", interactive=False)
                    rd = gr.JSON(label="Comparison findings")
        refresh_button.click(refresh, outputs=[run, left, right])
        inputs = [run, family, difficulty, status]
        outputs = [summary, table, state, downloads, image, prompt, caption, detail]
        for component in inputs:
            component.change(browse, inputs=inputs, outputs=outputs)
        reload_button.click(browse, inputs=inputs, outputs=outputs)
        app.load(browse, inputs=inputs, outputs=outputs)
        table.select(select_row, inputs=[state], outputs=[image, prompt, caption, detail])
        compare_button.click(
            comparison, inputs=[left, right], outputs=[comparison_result, key]
        ).then(show_pair, inputs=[left, right, key], outputs=[li, lp, lc, ld, ri, rp, rc, rd])
        key.change(show_pair, inputs=[left, right, key], outputs=[li, lp, lc, ld, ri, rp, rc, rd])
    return app
