import csv
from types import SimpleNamespace

import pytest
from PIL import Image

from human_bench.catalog import load_catalog
from human_bench.schemas import REGIONS


@pytest.fixture
def prompts():
    return load_catalog()


@pytest.fixture
def manifest_factory(tmp_path):
    def make(entries):
        rows = []
        for index, (prompt_id, sample_id) in enumerate(entries):
            image_path = tmp_path / f"image-{index}.png"
            Image.new("RGB", (24, 40), (index % 256, 20, 100)).save(image_path)
            rows.append(
                {"prompt_id": prompt_id, "sample_id": str(sample_id), "image_path": image_path.name}
            )
        path = tmp_path / "manifest.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt_id", "sample_id", "image_path"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    return make


def evaluation_for(prompt, verdict="pass", human_present=True):
    return {
        "human_present": human_present,
        "facts": [
            {"fact_id": f.fact_id, "verdict": verdict, "evidence": "Visible evidence."}
            for f in prompt.facts
        ],
        "anatomy": [
            {
                "entity_id": "p1",
                "region": region,
                "verdict": "pass",
                "evidence": "Visible coherent region.",
            }
            for region in REGIONS
        ]
        if human_present
        else [],
    }


class FakeVLM:
    """Exercise real request caching/retries with a fake SDK response producer."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            model="mock-vision-v1",
            _request_id="mock-request",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=reply, refusal=None)
                )
            ],
        )

    async def close(self):
        pass
