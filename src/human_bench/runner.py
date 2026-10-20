import asyncio
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from filelock import FileLock

from human_bench import __version__
from human_bench.catalog import CATALOG_VERSION, catalog_hash, load_catalog
from human_bench.config import RunConfig, digest
from human_bench.embeddings import Embedder, embedding_metrics, prompt_distinction
from human_bench.images import file_sha256, image_data_url, prepare_image
from human_bench.inference import VLM, safe_error
from human_bench.manifest import load_manifest
from human_bench.metrics import image_scores
from human_bench.prompts import TEMPLATE_HASH
from human_bench.reporting import export_run, write_json
from human_bench.schemas import PromptSpec
from human_bench.storage import Store


def environment_versions(local_embedding: bool) -> dict:
    packages = ["openai", "pillow", "numpy", "pydantic"]
    if local_embedding:
        packages += ["sentence-transformers", "transformers", "torch"]
    versions = {"human-bench": __version__, "python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def evaluator_profile(config: RunConfig) -> dict:
    return {
        "vlm": config.vlm.model_dump(),
        "embedding": config.embedding.model_dump(),
        "image_max_edge": config.image_max_edge,
        "preprocessing": "rgb-exif-white-png-v1",
        "template_hash": TEMPLATE_HASH,
        "metric_version": "v1",
        "versions": environment_versions(config.embedding.backend == "local_gemma"),
    }


async def evaluate_run(
    config: RunConfig,
    manifest: Path,
    output: Path,
    resume: bool = False,
    prompts: list[PromptSpec] | None = None,
    preflight_only: bool = False,
    progress=None,
    vlm_factory=VLM,
    embedder_factory=Embedder,
) -> dict:
    prompts = prompts if prompts is not None else load_catalog()
    # Validate all inputs before opening a client or making a paid API request.
    items = load_manifest(manifest, prompts, config.exploratory)
    profile = evaluator_profile(config)
    profile_hash = digest(profile)
    identity = digest(
        {
            "profile": profile_hash,
            "catalog": catalog_hash(prompts),
            "manifest": [i.model_dump() for i in items],
            "exploratory": config.exploratory,
            "name": config.name,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".evaluate.lock"), timeout=0):
        store = Store(output)
        vlm, embedder = None, None
        initialized = False
        try:
            previous = store.metadata()
            if previous:
                if not resume:
                    raise ValueError(
                        "Output already contains a run; use --resume or a new directory"
                    )
                if previous.get("identity") != identity:
                    raise ValueError(
                        "Resume rejected: inputs, catalog, or evaluator profile changed"
                    )
            else:
                store.initialize(
                    {
                        "identity": identity,
                        "config": config.model_dump(),
                        "catalog_version": CATALOG_VERSION,
                        "catalog_hash": catalog_hash(prompts),
                        "catalog": [p.model_dump() for p in prompts],
                        "profile": profile,
                        "profile_hash": profile_hash,
                        "manifest": [i.model_dump() for i in items],
                        "created_at": datetime.now(UTC).isoformat(),
                        "status": "incomplete",
                    }
                )
            initialized = True
            store.set_meta(
                "last_execution",
                {
                    "started_at": datetime.now(UTC).isoformat(),
                    "concurrency": config.concurrency,
                    "timeout_seconds": config.timeout_seconds,
                    "attempts": config.attempts,
                    "requests_per_minute": config.requests_per_minute,
                },
            )
            by_id = {p.prompt_id: p for p in prompts}
            for item in items:
                if store.result(item.prompt_id, item.sample_id) is None:
                    prompt = by_id[item.prompt_id]
                    store.save_result(
                        {
                            **item.model_dump(),
                            "family": prompt.family,
                            "difficulty": prompt.difficulty,
                            "status": "pending",
                        }
                    )
            if all(r["status"] == "complete" for r in store.results()):
                store.set_meta("status", "complete")
                return export_run(output)
            store.set_meta("status", "running")
            write_json(output / "run.json", store.metadata())
            vlm = vlm_factory(config, store, profile_hash)
            embedder = (
                embedder_factory(config, store) if config.embedding.backend != "none" else None
            )
            if embedder and config.embedding.backend == "api":
                # One configured API budget spans both clients, even on a shared endpoint.
                embedder.limiter = vlm.limiter
            vectors = {}

            async def inspect(item):
                path = Path(item.image_path)
                if await asyncio.to_thread(file_sha256, path) != item.image_sha256:
                    raise ValueError(f"Image changed after validation: {path}")
                image = await asyncio.to_thread(prepare_image, path, config.image_max_edge)
                data_url = image_data_url(image)
                prompt = by_id[item.prompt_id]
                caption = await vlm.request("caption", data_url, item.image_sha256)
                evaluation = await vlm.request("evaluation", data_url, item.image_sha256, prompt)
                return image, caption, evaluation

            # First real example probes both schemas and image input. Its outputs remain cached.
            first = next(
                item
                for item in items
                if store.result(item.prompt_id, item.sample_id)["status"] != "complete"
            )
            try:
                await inspect(first)
                if embedder:
                    p = by_id[first.prompt_id]
                    await embedder.texts([p.prompt, p.counterfactual, p.negation])
                store.set_meta("preflight", "passed")
            except Exception as exc:
                row = store.result(first.prompt_id, first.sample_id)
                store.save_result({**row, "status": "error", "error": safe_error(exc)})
                raise RuntimeError(
                    f"Preflight failed; check endpoint, schema mode, and model: {exc}"
                ) from exc
            if preflight_only:
                store.set_meta("status", "preflight_passed")
                return export_run(output)
            if embedder:
                texts = [text for p in prompts for text in (p.prompt, p.counterfactual, p.negation)]
                all_vectors = await embedder.texts(texts)
                vectors = {
                    p.prompt_id: all_vectors[i * 3 : i * 3 + 3] for i, p in enumerate(prompts)
                }
                distinctions = {pid: prompt_distinction(*value) for pid, value in vectors.items()}
                store.set_meta(
                    "catalog_diagnostics",
                    {
                        "prompt_distinction": distinctions,
                        "prompt_distinction_mean": sum(distinctions.values()) / len(distinctions),
                        "note": "Text-only catalog diagnostic; not a T2I model score.",
                    },
                )
            semaphore = asyncio.Semaphore(config.concurrency)
            progress_rows = {
                (row["prompt_id"], row["sample_id"]): {
                    "prompt_id": row["prompt_id"],
                    "sample_id": row["sample_id"],
                    "status": row["status"],
                }
                for row in store.results()
            }

            async def process(item):
                async with semaphore:
                    row = store.result(item.prompt_id, item.sample_id)
                    if row["status"] == "complete":
                        return
                    try:
                        image, caption, evaluation = await inspect(item)
                        emb = None
                        if embedder:
                            caption_vector = (await embedder.texts([caption["caption"]]))[0]
                            image_vector = (
                                await embedder.image(
                                    image,
                                    item.image_sha256,
                                    config.image_max_edge,
                                )
                                if embedder.supports_images
                                else None
                            )
                            emb = embedding_metrics(
                                *vectors[item.prompt_id], caption_vector, image_vector
                            )
                        scores = image_scores(by_id[item.prompt_id], evaluation, emb)
                        store.save_result(
                            {
                                **row,
                                "status": "complete",
                                "error": None,
                                "caption": caption["caption"],
                                "evaluation": evaluation,
                                "embedding_metrics": emb,
                                "scores": scores,
                            }
                        )
                    except Exception as exc:
                        store.save_result({**row, "status": "error", "error": safe_error(exc)})
                    if progress:
                        progress_rows[(item.prompt_id, item.sample_id)]["status"] = store.result(
                            item.prompt_id, item.sample_id
                        )["status"]
                        progress(list(progress_rows.values()))

            async with asyncio.TaskGroup() as tasks:
                for item in items:
                    tasks.create_task(process(item))
            complete = all(r["status"] == "complete" for r in store.results())
            store.set_meta("status", "complete" if complete else "incomplete")
            store.set_meta("finished_at", datetime.now(UTC).isoformat())
            return export_run(output)
        except BaseException:
            # A failed resume identity check must never alter the existing run.
            if initialized:
                store.set_meta("status", "incomplete")
                export_run(output)
            raise
        finally:
            if vlm:
                await vlm.close()
            if embedder:
                await embedder.close()
            store.close()
