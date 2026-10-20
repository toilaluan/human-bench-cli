import csv
from pathlib import Path

from PIL import Image

from human_bench.images import file_sha256
from human_bench.schemas import ImageItem, PromptSpec

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def build_manifest(folder: Path, output: Path) -> int:
    import os

    rows = []
    for path in sorted(folder.resolve().rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "__" not in path.stem:
            raise ValueError(f"Expected <prompt_id>__<sample_id>, found {path.name}")
        prompt_id, sample_id = path.stem.rsplit("__", 1)
        rows.append(
            {
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "image_path": os.path.relpath(path, output.resolve().parent),
            }
        )
    if not rows:
        raise ValueError("No supported images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "sample_id", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_manifest(
    path: Path, prompts: list[PromptSpec], exploratory: bool = False
) -> list[ImageItem]:
    known = {p.prompt_id for p in prompts}
    items, seen = [], set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"prompt_id", "sample_id", "image_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires prompt_id, sample_id, image_path columns")
        for line, row in enumerate(reader, 2):
            if any(
                not (row.get(k) or "").strip() for k in ("prompt_id", "sample_id", "image_path")
            ):
                raise ValueError(f"Empty required value on manifest line {line}")
            prompt_id, sample_id = row["prompt_id"].strip(), row["sample_id"].strip()
            key = (prompt_id, sample_id)
            if prompt_id not in known:
                raise ValueError(f"Unknown prompt ID on line {line}: {prompt_id}")
            if key in seen:
                raise ValueError(f"Duplicate sample key on line {line}: {key}")
            seen.add(key)
            image_path = (path.resolve().parent / row["image_path"]).resolve()
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Unreadable image on line {line}: {image_path}: {exc}") from exc
            items.append(
                ImageItem(
                    prompt_id=prompt_id,
                    sample_id=sample_id,
                    image_path=str(image_path),
                    image_sha256=file_sha256(image_path),
                )
            )
    if not items:
        raise ValueError("Manifest is empty")
    if not exploratory:
        expected = {(p, str(s)) for p in known for s in range(4)}
        if seen != expected:
            raise ValueError(
                f"Standard run requires samples 0–3 for all {len(known)} prompts; "
                f"missing={len(expected - seen)}, unexpected={len(seen - expected)}. "
                "Use exploratory mode for subsets."
            )
    return sorted(items, key=lambda item: (item.prompt_id, item.sample_id))
