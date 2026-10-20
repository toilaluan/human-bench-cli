import json
from types import SimpleNamespace

import pytest
from conftest import FakeVLM, evaluation_for

from human_bench.config import EmbeddingConfig, RunConfig, VLMConfig
from human_bench.embeddings import Embedder
from human_bench.inference import VLM, EvaluationError
from human_bench.storage import Store


async def test_portable_request_blindness_cache_and_retries(tmp_path, prompts):
    config = RunConfig(attempts=2, vlm=VLMConfig(extra_body={}, response_format="json_schema"))
    store = Store(tmp_path / "run")
    sdk = FakeVLM(
        [
            "not JSON",
            '{"caption":"One person is walking."}',
            json.dumps(evaluation_for(prompts[0])),
            '{"caption":"Changed image."}',
        ]
    )
    vlm = VLM(config, store, "profile", client=sdk)
    try:
        caption = await vlm.request("caption", "data:image/png;base64,AA==", "hash")
        assert caption["caption"] == "One person is walking."
        assert await vlm.request("caption", "ignored", "hash") == caption
        assert len(sdk.calls) == 2
        assert prompts[0].prompt not in sdk.calls[0]["messages"][1]["content"][1]["text"]
        await vlm.request("evaluation", "data:image/png;base64,AA==", "hash", prompts[0])
        assert sdk.calls[-1]["response_format"]["json_schema"]["strict"] is True
        assert prompts[0].prompt in sdk.calls[-1]["messages"][1]["content"][1]["text"]
        await vlm.request("caption", "data:image/png;base64,AA==", "changed-hash")
        assert len(sdk.calls) == 4
        assert store.attempts()[0]["error"]
        assert "base64" not in json.dumps(store.attempts())
    finally:
        store.close()


async def test_changed_profile_does_not_reuse_response(tmp_path):
    store = Store(tmp_path)
    config = RunConfig()
    try:
        one = FakeVLM(['{"caption":"A."}'])
        two = FakeVLM(['{"caption":"B."}'])
        assert (await VLM(config, store, "a", client=one).request("caption", "image", "h"))[
            "caption"
        ] == "A."
        assert (await VLM(config, store, "b", client=two).request("caption", "image", "h"))[
            "caption"
        ] == "B."
        assert len(two.calls) == 1
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["json_object", "prompt_json"])
async def test_explicit_output_modes(tmp_path, mode):
    store = Store(tmp_path)
    sdk = FakeVLM(['{"caption":"A person."}'])
    config = RunConfig(vlm=VLMConfig(response_format=mode, extra_body={}))
    try:
        await VLM(config, store, mode, sdk).request("caption", "image", "hash")
        if mode == "json_object":
            assert sdk.calls[0]["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in sdk.calls[0]
    finally:
        store.close()


async def test_invalid_fact_coverage_is_an_error(tmp_path, prompts):
    store = Store(tmp_path)
    data = evaluation_for(prompts[0])
    data["facts"].pop()
    sdk = FakeVLM([json.dumps(data)])
    try:
        with pytest.raises(EvaluationError, match="every required fact"):
            await VLM(RunConfig(attempts=1), store, "p", sdk).request(
                "evaluation", "image", "hash", prompts[0]
            )
        assert store.cached("hash") is None
    finally:
        store.close()


async def test_refusals_are_not_success(tmp_path):
    class RefusingSDK(FakeVLM):
        async def create(self, **kwargs):
            return SimpleNamespace(
                model="mock",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=None, refusal="Cannot judge"),
                    )
                ],
            )

    store = Store(tmp_path)
    try:
        with pytest.raises(EvaluationError, match="Refused"):
            await VLM(RunConfig(attempts=1), store, "p", RefusingSDK([])).request(
                "caption", "image", "hash"
            )
        assert store.attempts()[0]["refusal"] == "Cannot judge"
    finally:
        store.close()


async def test_local_gemma_multimodal_contract_without_weights(tmp_path):
    class LocalModel:
        def __init__(self):
            self.inputs = []

        def encode(self, values, **kwargs):
            import numpy as np

            self.inputs.append((values, kwargs))
            return np.ones((len(values), 768))

    from PIL import Image

    store = Store(tmp_path)
    embedder = Embedder(RunConfig(embedding=EmbeddingConfig(backend="local_gemma")), store)
    embedder.model = LocalModel()
    try:
        await embedder.texts(["A person."])
        await embedder.image(Image.new("RGB", (16, 16)), "image", 2048)
        assert embedder.model.inputs[0][0] == ["task: sentence similarity | query: A person."]
        assert embedder.model.inputs[1][0][0]["text"] == "<|image|>"
        assert embedder.model.inputs[0][1]["prompt"] == ""
        assert len(await embedder.image(Image.new("RGB", (16, 16)), "image", 2048)) == 768
        assert len(embedder.model.inputs) == 2
    finally:
        store.close()
