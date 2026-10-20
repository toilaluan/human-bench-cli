import csv
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from human_bench.catalog import catalog_hash, export_catalog, load_catalog
from human_bench.images import prepare_image
from human_bench.manifest import build_manifest, load_manifest


def test_frozen_catalog(prompts):
    assert len(prompts) == 160
    assert Counter(p.family for p in prompts) == {f"T{i}": 20 for i in range(1, 9)}
    assert len({p.prompt for p in prompts}) == 160
    assert all(p.contrast_fact_id in {f.fact_id for f in p.facts} for p in prompts)
    assert all(p.prompt != p.counterfactual != p.negation for p in prompts)
    assert catalog_hash(prompts) == catalog_hash(load_catalog())
    assert (
        catalog_hash(prompts) == "f757bebd5ed9886d617f820cf29db23a7b37416dd11c153c6b36cfeaf19e52e1"
    )
    subprocess.run([sys.executable, "scripts/build_catalog.py", "--check"], check=True)


def test_export_and_reload(tmp_path, prompts):
    output = tmp_path / "catalog.jsonl"
    export_catalog(output, rich=True)
    assert load_catalog(output) == prompts


def test_standard_and_exploratory_coverage(manifest_factory, prompts):
    manifest = manifest_factory([(p.prompt_id, sample) for p in prompts for sample in range(4)])
    assert len(load_manifest(manifest, prompts)) == 640
    subset = manifest_factory([(prompts[0].prompt_id, 0)])
    with pytest.raises(ValueError, match="Standard run"):
        load_manifest(subset, prompts)
    assert len(load_manifest(subset, prompts, exploratory=True)) == 1


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "missing", "corrupt"])
def test_invalid_input(kind, manifest_factory, prompts):
    pid = "unknown" if kind == "unknown" else prompts[0].prompt_id
    entries = [(pid, 0), (pid, 0)] if kind == "duplicate" else [(pid, 0)]
    manifest = manifest_factory(entries)
    if kind == "missing":
        (manifest.parent / "image-0.png").unlink()
    elif kind == "corrupt":
        (manifest.parent / "image-0.png").write_text("not an image")
    with pytest.raises(ValueError):
        load_manifest(manifest, prompts, exploratory=True)


def test_manifest_builder_relative_paths(tmp_path, prompts):
    folder = tmp_path / "external-model" / "nested"
    folder.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(folder / f"{prompts[0].prompt_id}__3.png")
    output = tmp_path / "bench" / "manifest.csv"
    assert build_manifest(folder.parent, output) == 1
    item = load_manifest(output, prompts, exploratory=True)[0]
    assert item.sample_id == "3"
    assert Path(item.image_path).parent == folder
    with output.open() as f:
        assert not Path(next(csv.DictReader(f))["image_path"]).is_absolute()


def test_image_preprocessing_preserves_source(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (80, 40), (0, 0, 0, 0)).save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = prepare_image(path, 40)
    assert prepared.mode == "RGB" and prepared.size == (40, 20)
    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert prepare_image(path, 2048).size == (80, 40)


def test_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (60, 30))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert prepare_image(path).size == (30, 60)
