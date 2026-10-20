import csv
import json
import os
import tempfile
from pathlib import Path

from human_bench.metrics import compare_rows, summarize
from human_bench.storage import Store


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                    if key in fields
                }
            )


def read_run(folder: Path) -> tuple[dict, list[dict]]:
    if not (folder / "run.sqlite").is_file():
        raise ValueError(f"No run database in {folder}")
    store = Store(folder)
    try:
        meta = store.metadata()
        if "config" not in meta:
            raise ValueError(f"Run is not initialized: {folder}")
        return meta, store.results()
    finally:
        store.close()


def export_run(folder: Path) -> dict:
    store = Store(folder)
    try:
        meta, rows, attempts = store.metadata(), store.results(), store.attempts()
    finally:
        store.close()
    summary = summarize(meta, rows, attempts)
    write_json(folder / "run.json", meta)
    write_json(folder / "summary.json", summary)
    with (folder / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    flattened = [
        {**{k: v for k, v in row.items() if k != "scores"}, **row.get("scores", {})} for row in rows
    ]
    fields = [
        "prompt_id",
        "sample_id",
        "image_path",
        "image_sha256",
        "family",
        "difficulty",
        "status",
        "error",
        "caption",
        "evaluation",
        "embedding_metrics",
    ]
    fields += sorted({k for row in rows for k in row.get("scores", {})})
    write_csv(folder / "results.csv", flattened, fields)
    summary_rows = []
    for group, metrics in [
        ("overall", summary["metrics"]),
        *summary["by_family"].items(),
        *summary["by_difficulty"].items(),
    ]:
        for metric, value in metrics.items():
            summary_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": value["mean"],
                    "ci95": value["ci95"],
                    "eligible_prompts": value["eligible_prompts"],
                }
            )
    write_csv(
        folder / "summary.csv",
        summary_rows,
        ["group", "metric", "mean", "ci95", "eligible_prompts"],
    )
    write_json(folder / "attempts.json", attempts)
    return summary


def export_comparison(folders: list[Path], output: Path) -> dict:
    comparison = compare_rows([read_run(folder) for folder in folders])
    write_json(output, comparison)
    rows = [
        {"run": run["name"], "metric": metric, **values}
        for run in comparison["runs"]
        for metric, values in run["metrics"].items()
    ]
    write_csv(
        output.with_suffix(".csv"), rows, ["run", "metric", "mean", "ci95", "eligible_prompts"]
    )
    return comparison
