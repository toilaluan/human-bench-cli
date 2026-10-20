"""Image-dependent scores, with equal prompt/family weighting and clustered intervals."""

from collections import defaultdict

import numpy as np

from human_bench.schemas import REGIONS, Evaluation, PromptSpec


def image_scores(prompt: PromptSpec, evaluation: dict, embeddings: dict | None = None) -> dict:
    checked = Evaluation.model_validate(evaluation).validate_for(prompt)
    verdicts = {f.fact_id: f.verdict for f in checked.facts}
    scores = {
        "strict_adherence": float(all(v == "pass" for v in verdicts.values())),
        "fact_adherence": sum(v == "pass" for v in verdicts.values()) / len(verdicts),
        "fact_assessable_fraction": sum(v != "unassessable" for v in verdicts.values())
        / len(verdicts),
        "no_human_rate": float(not checked.human_present),
    }
    groups = defaultdict(list)
    for f in prompt.facts:
        groups[f.capability].append(verdicts[f.fact_id])
    for capability, values in groups.items():
        scores[f"{capability}_adherence"] = sum(v == "pass" for v in values) / len(values)
    anatomy = checked.anatomy
    assessable = [a for a in anatomy if a.verdict != "unassessable"]
    scores["anatomy_defect_free"] = (
        sum(a.verdict == "pass" for a in assessable) / len(assessable) if assessable else None
    )
    scores["anatomy_assessable_fraction"] = len(assessable) / len(anatomy) if anatomy else 0.0
    for region in REGIONS:
        findings = [a for a in anatomy if a.region == region]
        visible = [a for a in findings if a.verdict != "unassessable"]
        scores[f"anatomy_{region}_defect_free"] = (
            sum(a.verdict == "pass" for a in visible) / len(visible) if visible else None
        )
        scores[f"anatomy_{region}_coverage"] = len(visible) / len(findings) if findings else 0.0
    scores.update(embeddings or {})
    return scores


def _prompt_groups(rows: list[dict], metric: str) -> dict[str, list[float]]:
    samples = defaultdict(list)
    families = {}
    for row in rows:
        value = row.get("scores", {}).get(metric)
        if row.get("status") != "complete" or value is None:
            continue
        samples[row["prompt_id"]].append(value)
        families[row["prompt_id"]] = row["family"]
    grouped = defaultdict(list)
    for prompt_id in sorted(samples):
        grouped[families[prompt_id]].append(float(np.mean(samples[prompt_id])))
    return dict(grouped)


def aggregate(rows: list[dict], bootstrap: bool = True) -> dict:
    metrics = sorted({key for row in rows for key in row.get("scores", {})})
    result = {}
    for metric in metrics:
        groups = _prompt_groups(rows, metric)
        if not groups:
            result[metric] = {"mean": None, "ci95": None, "eligible_prompts": 0}
            continue
        means = [float(np.mean(values)) for _, values in sorted(groups.items())]
        ci = None
        if bootstrap:
            rng = np.random.default_rng(0)
            draws = []
            for _, values in sorted(groups.items()):
                values = np.asarray(values)
                indices = rng.integers(0, len(values), (2000, len(values)))
                draws.append(values[indices].mean(axis=1))
            ci = np.quantile(np.mean(draws, axis=0), [0.025, 0.975]).tolist()
        result[metric] = {
            "mean": float(np.mean(means)),
            "ci95": ci,
            "eligible_prompts": sum(map(len, groups.values())),
        }
    return result


def summarize(
    metadata: dict, rows: list[dict], attempts: list[dict] | None = None, bootstrap: bool = True
) -> dict:
    complete = [r for r in rows if r.get("status") == "complete"]
    attempts = attempts or []
    usage = defaultdict(int)
    for attempt in attempts:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (attempt.get("usage") or {}).get(key)
            if isinstance(value, int):
                usage[key] += value
    return {
        "name": metadata["config"]["name"],
        "status": metadata.get("status", "incomplete"),
        "exploratory": metadata["config"]["exploratory"],
        "catalog_hash": metadata["catalog_hash"],
        "profile_hash": metadata["profile_hash"],
        "expected_images": len(metadata.get("manifest", rows)),
        "completed_images": len(complete),
        "error_images": sum(r.get("status") == "error" for r in rows),
        "completed_prompts": len({r["prompt_id"] for r in complete}),
        "metrics": aggregate(rows, bootstrap),
        "by_family": {
            family: aggregate([r for r in rows if r["family"] == family], bootstrap)
            for family in sorted({r["family"] for r in rows})
        },
        "by_difficulty": {
            level: aggregate([r for r in rows if r["difficulty"] == level], bootstrap)
            for level in sorted({r["difficulty"] for r in rows})
        },
        "catalog_diagnostics": metadata.get("catalog_diagnostics", {}),
        "usage": dict(usage),
        "request_attempts": len(attempts),
        "inference_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts),
        "returned_models": sorted(
            {a["returned_model"] for a in attempts if a.get("returned_model")}
        ),
        "notes": [
            "Metrics cover completed images; incomplete runs are not standard results.",
            "Anatomy is conditional on visibility; inspect coverage and no-human rate.",
            "Confidence intervals resample prompts within families, not individual images.",
            "Embedding similarities and margins are diagnostics, not calibrated probabilities.",
        ],
    }


def compare_rows(run_data: list[tuple[dict, list[dict]]]) -> dict:
    if len(run_data) < 2:
        raise ValueError("Comparison requires at least two runs")
    reference_meta, reference = run_data[0]
    keys = {(r["prompt_id"], r["sample_id"]) for r in reference}
    for meta, rows in run_data:
        if meta["catalog_hash"] != reference_meta["catalog_hash"]:
            raise ValueError("Cannot compare runs with different catalogs")
        if meta["profile_hash"] != reference_meta["profile_hash"]:
            raise ValueError("Cannot compare runs with different evaluator profiles")
        if meta["config"]["exploratory"] != reference_meta["config"]["exploratory"]:
            raise ValueError("Cannot compare standard and exploratory protocols")
        if meta.get("status") != "complete" or any(r.get("status") != "complete" for r in rows):
            raise ValueError(
                "Comparison requires complete runs; inspect incomplete runs separately"
            )
        if {(r["prompt_id"], r["sample_id"]) for r in rows} != keys:
            raise ValueError("Comparison requires matching prompt/sample coverage")
    comparisons = []
    reference_by_key = {(r["prompt_id"], r["sample_id"]): r for r in reference}
    for meta, rows in run_data[1:]:
        deltas = []
        for row in rows:
            before = reference_by_key[(row["prompt_id"], row["sample_id"])]["scores"]
            deltas.append(
                {
                    **row,
                    "scores": {
                        k: v - before[k]
                        for k, v in row["scores"].items()
                        if v is not None and before.get(k) is not None
                    },
                }
            )
        comparisons.append(
            {"name": meta["config"]["name"], "delta_vs_reference": aggregate(deltas)}
        )
    return {
        "reference": reference_meta["config"]["name"],
        "runs": [
            {"name": meta["config"]["name"], "metrics": aggregate(rows)} for meta, rows in run_data
        ],
        "paired_comparisons": comparisons,
    }
