import json
from collections import Counter
from importlib.resources import files
from pathlib import Path

from human_bench.config import digest
from human_bench.schemas import PromptSpec

CATALOG_VERSION = "human-bench-v1"


def load_catalog(path: Path | None = None) -> list[PromptSpec]:
    source = path if path else files("human_bench").joinpath("data/prompts.v1.jsonl")
    prompts = [
        PromptSpec.model_validate_json(line) for line in source.read_text().splitlines() if line
    ]
    ids = [p.prompt_id for p in prompts]
    if len(ids) != len(set(ids)) or len({p.prompt for p in prompts}) != len(prompts):
        raise ValueError("Catalog contains duplicate IDs or prompt texts")
    for p in prompts:
        fact_ids = [f.fact_id for f in p.facts]
        if len(fact_ids) != len(set(fact_ids)) or p.contrast_fact_id not in fact_ids:
            raise ValueError(f"Invalid fact bindings for {p.prompt_id}")
        if p.prompt in {p.counterfactual, p.negation}:
            raise ValueError(f"Unchanged contrast for {p.prompt_id}")
    if path is None and Counter(p.family for p in prompts) != {f"T{i}": 20 for i in range(1, 9)}:
        raise ValueError("Bundled v1 catalog must contain exactly 20 prompts per family")
    return prompts


def catalog_hash(prompts: list[PromptSpec]) -> str:
    return digest([p.model_dump() for p in prompts])


def export_catalog(path: Path, rich: bool = False) -> None:
    prompts = load_catalog()
    if rich:
        path.write_text("".join(p.model_dump_json() + "\n" for p in prompts))
    else:
        path.write_text(json.dumps({p.prompt_id: p.prompt for p in prompts}, indent=2) + "\n")
