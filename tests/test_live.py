"""Opt-in actual endpoint check; no network or credentials required by default."""

import os
from pathlib import Path

import pytest

from human_bench.config import load_config
from human_bench.runner import evaluate_run


@pytest.mark.skipif(not os.getenv("HUMAN_BENCH_LIVE_CONFIG"), reason="Live endpoint not configured")
async def test_live_vlm_preflight(tmp_path, manifest_factory, prompts):
    config = load_config(Path(os.environ["HUMAN_BENCH_LIVE_CONFIG"]))
    config.exploratory = True
    # Replace the synthetic image with a real fixture using HUMAN_BENCH_LIVE_MANIFEST
    # if checking model accuracy rather than transport and schema compatibility.
    manifest = (
        Path(os.environ["HUMAN_BENCH_LIVE_MANIFEST"])
        if os.getenv("HUMAN_BENCH_LIVE_MANIFEST")
        else manifest_factory([(prompts[0].prompt_id, 0)])
    )
    result = await evaluate_run(config, manifest, tmp_path / "live", preflight_only=True)
    assert result["status"] == "preflight_passed"
