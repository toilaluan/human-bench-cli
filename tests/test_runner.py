import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from human_bench.config import EmbeddingConfig, RunConfig, VLMConfig
from human_bench.reporting import read_run
from human_bench.runner import evaluate_run
from human_bench.schemas import REGIONS


@pytest.fixture
def api_server():
    state = {"requests": [], "fail_after": None, "bad_request": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, payload))
            failed = (
                state["fail_after"] is not None and len(state["requests"]) > state["fail_after"]
            )
            status = 400 if state["bad_request"] else 500 if failed else 200
            if status != 200:
                result = {"error": {"message": "Test API failure", "type": "server_error"}}
            elif self.path.endswith("/embeddings"):
                result = {
                    "object": "list",
                    "model": payload["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [
                                1 + n for n in hashlib.sha256(text.encode()).digest()[:3]
                            ],
                        }
                        for i, text in enumerate(payload["input"])
                    ],
                    "usage": {
                        "prompt_tokens": len(payload["input"]),
                        "total_tokens": len(payload["input"]),
                    },
                }
            else:
                text = payload["messages"][1]["content"][1]["text"]
                if "Required facts:\n" in text:
                    facts = json.loads(
                        text.split("Required facts:\n", 1)[1].split("\nReturn JSON", 1)[0]
                    )
                    output = {
                        "human_present": True,
                        "facts": [
                            {"fact_id": f["fact_id"], "verdict": "pass", "evidence": "Visible."}
                            for f in facts
                        ],
                        "anatomy": [
                            {
                                "entity_id": "p1",
                                "region": r,
                                "verdict": "pass",
                                "evidence": "Visible.",
                            }
                            for r in REGIONS
                        ],
                    }
                else:
                    output = {"caption": "One visible person walking beside a chair."}
                result = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(output)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def config_for(server, embedding="none", **overrides):
    return RunConfig(
        exploratory=True,
        attempts=1,
        concurrency=1,
        vlm=VLMConfig(
            base_url=server["url"],
            model="mock-vlm",
            extra_body={},
            request_kwargs={},
            response_format="json_schema",
        ),
        embedding=EmbeddingConfig(
            backend=embedding,
            base_url=server["url"],
            model="mock-embedding",
            text_prefix="",
            batch_size=16,
        ),
        **overrides,
    )


async def test_real_sdk_round_trip_exports_and_free_resume(
    tmp_path, prompts, manifest_factory, api_server
):
    manifest = manifest_factory([(prompts[0].prompt_id, 0), (prompts[1].prompt_id, 0)])
    output = tmp_path / "run"
    config = config_for(api_server, embedding="api")
    summary = await evaluate_run(config, manifest, output, prompts=prompts[:2])
    assert summary["status"] == "complete" and summary["completed_images"] == 2
    assert summary["metrics"]["strict_adherence"]["mean"] == 1
    assert summary["metrics"]["adapted_agreement"]["mean"] is None
    assert len([r for r in api_server["requests"] if r[0].endswith("chat/completions")]) == 4
    assert all(
        (output / file).exists()
        for file in [
            "run.json",
            "summary.json",
            "results.csv",
            "results.jsonl",
            "summary.csv",
            "attempts.json",
        ]
    )
    meta, rows = read_run(output)
    assert meta["catalog_diagnostics"]["prompt_distinction"]
    assert rows[0]["embedding_metrics"]["image_contrast_margin"] is None
    requests = len(api_server["requests"])
    await evaluate_run(config, manifest, output, resume=True, prompts=prompts[:2])
    assert len(api_server["requests"]) == requests
    assert "data:image" not in (output / "attempts.json").read_text()


async def test_failure_resume_reuses_successful_work(
    tmp_path, prompts, manifest_factory, api_server
):
    manifest = manifest_factory([(prompts[0].prompt_id, 0), (prompts[1].prompt_id, 0)])
    config, output = config_for(api_server), tmp_path / "run"
    api_server["fail_after"] = 2
    summary = await evaluate_run(config, manifest, output, prompts=prompts[:2])
    assert summary["status"] == "incomplete" and summary["error_images"] == 1
    assert summary["completed_images"] == 1
    requests = len(api_server["requests"])
    api_server["fail_after"] = None
    resumed = await evaluate_run(config, manifest, output, resume=True, prompts=prompts[:2])
    assert resumed["status"] == "complete"
    assert len(api_server["requests"]) - requests == 2


async def test_resume_rejects_changes_without_mutating_run(
    tmp_path, prompts, manifest_factory, api_server
):
    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    config, output = config_for(api_server), tmp_path / "run"
    await evaluate_run(config, manifest, output, prompts=prompts[:1])
    original = read_run(output)
    before = len(api_server["requests"])
    changed = config.model_copy(update={"image_max_edge": 1024})
    with pytest.raises(ValueError, match="Resume rejected"):
        await evaluate_run(changed, manifest, output, resume=True, prompts=prompts[:1])
    assert read_run(output) == original
    assert len(api_server["requests"]) == before
    with pytest.raises(ValueError, match="already contains"):
        await evaluate_run(config, manifest, output, prompts=prompts[:1])
    assert read_run(output)[0]["status"] == "complete"


async def test_invalid_manifest_never_calls_api(tmp_path, prompts, manifest_factory, api_server):
    manifest = manifest_factory([("unknown-prompt", 0)])
    with pytest.raises(ValueError, match="Unknown prompt"):
        await evaluate_run(config_for(api_server), manifest, tmp_path / "run")
    assert not api_server["requests"]


async def test_preflight_only_and_non_retryable_api_error(
    tmp_path, prompts, manifest_factory, api_server
):
    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    config = config_for(api_server)
    result = await evaluate_run(
        config, manifest, tmp_path / "probe", preflight_only=True, prompts=prompts[:1]
    )
    assert result["status"] == "preflight_passed"
    assert len(api_server["requests"]) == 2
    api_server["requests"].clear()
    api_server["bad_request"] = True
    with pytest.raises(RuntimeError, match="Preflight failed"):
        await evaluate_run(
            config.model_copy(update={"attempts": 3}),
            manifest,
            tmp_path / "failed",
            prompts=prompts[:1],
        )
    assert len(api_server["requests"]) == 1
    meta, rows = read_run(tmp_path / "failed")
    assert meta["status"] == "incomplete" and rows[0]["status"] == "error"


async def test_cancellation_checkpoints_then_resumes(
    tmp_path, prompts, manifest_factory, api_server
):
    manifest = manifest_factory([(prompts[0].prompt_id, 0), (prompts[1].prompt_id, 0)])
    config, output = config_for(api_server), tmp_path / "run"
    task = None

    def progress(rows):
        if sum(r["status"] == "complete" for r in rows) == 1:
            task.cancel()

    task = asyncio.create_task(
        evaluate_run(config, manifest, output, prompts=prompts[:2], progress=progress)
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert read_run(output)[0]["status"] == "incomplete"
    before = len(api_server["requests"])
    result = await evaluate_run(config, manifest, output, resume=True, prompts=prompts[:2])
    assert result["status"] == "complete"
    assert len(api_server["requests"]) - before == 2


def test_cli_evaluate_report_compare_end_to_end(tmp_path, prompts, manifest_factory, api_server):
    from typer.testing import CliRunner

    from human_bench.cli import app
    from human_bench.reporting import read_run

    manifest = manifest_factory([(prompts[0].prompt_id, 0)])
    config = config_for(api_server)
    config_path = tmp_path / "config.json"
    cli = CliRunner()
    folders = [tmp_path / "a", tmp_path / "b"]
    for name, folder in zip(("A", "B"), folders, strict=True):
        config.name = name
        config_path.write_text(config.model_dump_json())
        result = cli.invoke(
            app,
            [
                "evaluate",
                "--config",
                str(config_path),
                "--manifest",
                str(manifest),
                "--output",
                str(folder),
            ],
        )
        assert result.exit_code == 0, result.output
        assert read_run(folder)[0]["status"] == "complete"
    result = cli.invoke(app, ["report", str(folders[0])])
    assert result.exit_code == 0, result.output
    output = tmp_path / "comparison.json"
    result = cli.invoke(app, ["compare", *map(str, folders), "--output", str(output)])
    assert result.exit_code == 0, result.output
    comparison = json.loads(output.read_text())
    assert comparison["reference"] == "A"
    assert comparison["paired_comparisons"][0]["name"] == "B"
    assert output.with_suffix(".csv").exists()
