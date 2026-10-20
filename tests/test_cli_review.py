import copy
import json
from pathlib import Path

import pytest
from conftest import evaluation_for
from typer.testing import CliRunner

from human_bench.cli import app
from human_bench.config import RunConfig, load_config
from human_bench.metrics import image_scores
from human_bench.review import create_app, discover_runs, filter_rows, inspect_row
from human_bench.storage import Store


def test_cli_export_validate_and_help(tmp_path, prompts, manifest_factory):
    cli = CliRunner()
    output = tmp_path / "prompts.json"
    result = cli.invoke(app, ["prompts", "export", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(output.read_text())) == 160
    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    assert cli.invoke(app, ["validate", "--manifest", str(manifest)]).exit_code == 1
    result = cli.invoke(app, ["validate", "--manifest", str(manifest), "--exploratory"])
    assert result.exit_code == 0, result.output
    assert "1 images" in result.output


def test_config_examples_and_secret_protection():
    for path in Path("configs").glob("*.yaml"):
        load_config(path)
    with pytest.raises(ValueError, match="Credentials"):
        RunConfig.model_validate({"vlm": {"extra_body": {"api_key": "do-not-store"}}})
    with pytest.raises(ValueError, match="api_key_env"):
        RunConfig.model_validate({"vlm": {"base_url": "https://secret@example.com/v1"}})
    with pytest.raises(ValueError, match="managed"):
        RunConfig.model_validate({"vlm": {"request_kwargs": {"messages": []}}})


def test_review_filters_and_image_inspection(prompts, manifest_factory):
    from human_bench.manifest import load_manifest

    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    item = load_manifest(manifest, prompts, True)[0]
    evaluation = evaluation_for(prompts[0])
    evaluation["anatomy"][0]["verdict"] = "fail"
    evaluation["facts"][0]["verdict"] = "unassessable"
    row = {
        **item.model_dump(),
        "family": "T1",
        "difficulty": "easy",
        "status": "complete",
        "evaluation": evaluation,
        "caption": "A person.",
        "scores": image_scores(prompts[0], evaluation),
    }
    for status in ("anatomy defects", "adherence failures", "uncertain"):
        assert filter_rows([row], status=status) == [row]
    assert not filter_rows([row], family="T2")
    assert not filter_rows([row], difficulty="hard")
    assert not filter_rows([row], status="errors")
    meta = {"catalog": [p.model_dump() for p in prompts]}
    image, prompt, caption, detail = inspect_row(meta, row)
    assert image.size == (24, 40) and prompt == prompts[0].prompt and caption == "A person."
    assert detail["evaluation"] == evaluation


def test_gradio_app_callbacks_and_downloads(tmp_path, prompts, manifest_factory):
    pytest.importorskip("gradio")
    from human_bench.manifest import load_manifest

    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    item = load_manifest(manifest, prompts, True)[0]
    store = Store(tmp_path / "runs" / "first")
    meta = {
        "config": {"name": "A", "exploratory": True},
        "catalog_hash": "c",
        "profile_hash": "p",
        "catalog": [p.model_dump() for p in prompts],
        "manifest": [item.model_dump()],
        "status": "complete",
    }
    for key, value in meta.items():
        store.set_meta(key, value)
    row = {
        **item.model_dump(),
        "family": "T1",
        "difficulty": "easy",
        "status": "complete",
        "caption": "A person.",
        "evaluation": evaluation_for(prompts[0]),
        "scores": image_scores(prompts[0], evaluation_for(prompts[0])),
    }
    store.save_result(row)
    store.close()
    from human_bench.reporting import export_run

    export_run(tmp_path / "runs" / "first")
    other = Store(tmp_path / "runs" / "second")
    for key, value in meta.items():
        if key == "config":
            value = copy.deepcopy(value)
            value["name"] = "B"
        other.set_meta(key, value)
    other.save_result(row)
    other.close()
    root = tmp_path / "runs"
    assert len(discover_runs(root)) == 2
    interface = create_app(root)
    functions = {fn.fn.__name__: fn.fn for fn in interface.fns.values()}
    browse = functions["browse"](str(root / "first"), "all", "all", "all")
    assert browse[0]["completed_images"] == 1
    assert len(browse[1]) == 1 and browse[4].size == (24, 40)
    assert len(browse[3]) == 6 and all(Path(path).is_file() for path in browse[3])
    compared, selector = functions["comparison"](str(root / "first"), str(root / "second"))
    assert compared["reference"] == "A"
    pair = functions["show_pair"](str(root / "first"), str(root / "second"), selector.value)
    assert pair[0].size == pair[4].size == (24, 40)
    with pytest.raises(Exception, match="configured runs directory"):
        functions["browse"](str(tmp_path), "all", "all", "all")
