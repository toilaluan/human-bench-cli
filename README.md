# Human Generation Benchmark

A reproducible benchmark for human-centric text-to-image generation. It reads
images produced **outside this repository**, evaluates them through a vision model,
and reports prompt adherence, visible anatomy, and embedding diagnostics separately.

The fixed v1 catalog adapts the eight template families in
[HumanBench (Jain, Vatsa, and Singh, WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Jain_HumanBench_Two_Heads_No_Legs_But_Mostly_Human_the_State_WACV_2026_paper.html).
It has **160 prompts: 20 per family**. Standard comparisons use **four independent
images per prompt**, totaling **640 images per T2I model**. See the
[catalog provenance](src/human_bench/data/README.md) for adaptations and limitations.

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra review
```

Add `--extra embedding` for local EmbeddingGemma 2, and `--extra dev` for tests.
Keep vLLM in a separate environment on the GPU host. The benchmark itself needs
no GPU when using remote APIs or CPU embeddings. `uv.lock` pins the resolved
client, review, test, and optional embedding dependencies.

Use `uv run --no-sync human-bench ...` after installation so uv preserves the
extras you selected. The executable also works as `.venv/bin/human-bench`.

## Prepare externally generated images

```bash
uv run --no-sync human-bench prompts export --output prompts.json
uv run --no-sync human-bench prompts export --rich --output catalog.jsonl
```

`prompts.json` is the fixed `{prompt_id: prompt}` mapping. Send these exact strings
to your T2I generation system. The benchmark never generates images. Use four
independent samples, with sample IDs `0`, `1`, `2`, and `3`; record your T2I model,
revision, seeds, and generation settings alongside the external image dataset.

Organize image names as `<prompt_id>__<sample_id>.png` (JPEG, WebP, BMP, and TIFF
also work), then build a manifest:

```bash
uv run --no-sync human-bench manifest build /path/to/model-images --output manifest.csv
uv run --no-sync human-bench validate --manifest manifest.csv
```

Alternatively, supply a CSV with arbitrary filenames:

```csv
prompt_id,sample_id,image_path
hb-v1-t1-001,0,images/arbitrary-name.png
hb-v1-t1-001,1,/absolute/path/another-image.jpg
```

Relative image paths resolve against the manifest directory. Unknown IDs, duplicate
prompt/sample keys, missing or corrupt images, and nonstandard coverage are rejected
before API inference. For subsets or a different number of samples, set
`exploratory: true` in your config and validate with `--exploratory`.

| Family | Capability | Structural difficulty |
| --- | --- | --- |
| T1 | Single-person action | Easy |
| T2 | Two people with different actions | Hard |
| T3 | Material applied to a human figure | Easy |
| T4 | Person–object spatial relation | Medium |
| T5 | Spatial relation plus object material | Hard |
| T6 | Exact count of people | Easy |
| T7 | Count bound to an action | Medium |
| T8 | Two groups with distinct counts and actions | Very hard |

## Configure and evaluate

Copy and edit one of the profiles:

- [`configs/qwen-local.yaml`](configs/qwen-local.yaml): Qwen VLM API plus local
  EmbeddingGemma 2. The default embedder uses CPU; install the `embedding` extra.
- [`configs/api.yaml`](configs/api.yaml): separate configurable vision and text
  embedding endpoints through the OpenAI client. Replace the endpoint/model
  placeholders and use parameters supported by your provider.
- [`configs/vlm-only.yaml`](configs/vlm-only.yaml): an exploratory VLM-only run,
  without embedding dependencies or metrics.

Set credentials in the environment, using the names configured by `api_key_env`.
For example, export `VLM_API_KEY` and `EMBEDDING_API_KEY` in your shell. Local
servers without authentication use the client's `EMPTY` placeholder. Credentials
are not included in run snapshots.

```bash
uv run --no-sync human-bench evaluate --config configs/qwen-local.yaml \
  --manifest manifest.csv --output runs/model-a --preflight-only

uv run --no-sync human-bench evaluate --config configs/qwen-local.yaml \
  --manifest manifest.csv --output runs/model-a --resume
```

Preflight uses the first real image to verify image input and both output schemas,
plus a small text embedding request if enabled. It makes billable requests on
paid endpoints. Its successful outputs are cached for the full evaluation. You
can also start the full evaluation directly without `--preflight-only`.

Each image gets a blind factual caption and a separate checklist/anatomy judgment.
The generation prompt is absent from the caption request. Both stages use the
same versioned evaluator templates across providers. Choose `json_schema`,
`json_object`, or `prompt_json` in advance; there is no silent fallback between
protocols. All output modes receive local Pydantic validation.

The runner caps request concurrency, optionally spaces requests by
`requests_per_minute`, and makes at most three attempts for transient failures or
invalid outputs. It retains the first valid response. Bad authentication and
unsupported request parameters stop the preflight without repeated requests.
`max_tokens`, thinking settings, and other model options belong in the profile's
`request_kwargs` and `extra_body`; omit options unsupported by the chosen model.

### Self-hosted Qwen on H100

[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) supports image input
and vLLM serving. Use the [official serving recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
to provision a separate serving environment. The recipe's hardware checks focus
on text serving; validate the actual image/schema workflow on your target hardware.

From that environment on the GPU host:

```bash
# Set the exact checkpoint commit and copy the same value into vlm.revision.
export VLM_REVISION=your-checkpoint-commit
bash scripts/serve_vllm.sh
```

The script defaults to one GPU, a 32K context, four concurrent sequences, and 0.85
memory utilization. Override `VLM_TP`, `VLM_CONTEXT`, `VLM_MAX_SEQS`, `VLM_MEMORY`,
`VLM_HOST`, and `VLM_PORT` for the actual hardware. Additional CLI options pass
through to vLLM. No fixed H100 count or memory capacity is assumed. Record the
serving version/container digest and checkpoint revision with your results.

These are starting settings, not measured H100 throughput or memory guarantees.
The development checks exercise a local test API; actual H100 inference requires
the target server and the preflight above.

### Embedding backends

Local [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) produces
768-dimensional text and image embeddings. The supplied profile pins its checkpoint.
The adapter disables the unused audio encoder, uses BF16 on supported CUDA devices
and FP32 otherwise, and normalizes vectors. FP16 is not used. SentenceTransformers
6.1+ is required for this checkpoint's ordered multimodal inputs.

The API adapter supports text embeddings through `embeddings.create`. It uses the
same text prefix on prompts, captions, counterfactuals, and negations; configure an
empty prefix for models that do not require one. It does not assume that a standard
text embeddings endpoint accepts images. Image-based metrics are explicitly null
for API embeddings. VLM-only runs omit embedding diagnostics entirely.

## Read results

Every run directory contains a SQLite checkpoint, the exact catalog and config
snapshot, `results.jsonl`, `results.csv`, `summary.json`, `summary.csv`, and
`attempts.json`. Raw response text, validated outputs, returned model identifiers,
token usage, latency, and errors are recorded. Uploaded image data and credential
values are not stored in request logs. Images remain in their original folder.

| Metric | Interpretation |
| --- | --- |
| Strict adherence | All required facts pass; uncertainty prevents success |
| Fact adherence | Fraction of required facts that pass |
| Capability adherence | Separate counts, actions, relations, materials, entities, and bindings |
| Anatomy defect-free | Fraction of assessable person/region findings with no visible defect |
| Anatomy coverage | Fraction of person/region findings that are assessable |
| No-human rate | Fraction of evaluated images without human-shaped figures |
| Caption alignment | Prompt–blind-caption cosine similarity |
| Caption contrast margin | Prompt–caption minus counterfactual–caption similarity |
| Negation contrast margin | Prompt–caption minus negation–caption similarity |
| Image contrast margin | Image–prompt minus image–counterfactual similarity |
| Adapted Agreement | Mean caption and image contrast margins, when both exist |

Anatomy checks cover heads, torsos, arms, legs, hands, and feet for each depicted
person. Hidden or tiny regions are unassessable, not automatically defective.
Intentional materials and illustration styles are allowed. No-human images receive
no anatomy pass. Always read anatomy rates alongside coverage and no-human rate;
a conditional anatomy score alone does not establish whole-body correctness.

Average images within a prompt, then prompts within a family, then families equally.
Report 95% confidence intervals from 2,000 prompt bootstrap resamples within families,
with random seed zero. Region metrics use eligible prompts and report their count.
Scores are fractions, not percentages; embedding similarities and margins are raw
values, not probabilities. Adherence and anatomy are separate headline results.

The original paper's Distinction formula uses prompt texts alone. It appears under
`catalog_diagnostics`, never in model rankings. Our adapted Agreement uses one blind
captioner and upgraded embeddings, so it is not the paper's reproduced metric.
See [the paper's equations 1–4](https://openaccess.thecvf.com/content/WACV2026/papers/Jain_HumanBench_Two_Heads_No_Legs_But_Mostly_Human_the_State_WACV_2026_paper.pdf).
Newer evaluators are candidate upgrades; this tool does not establish their accuracy
without independent validation.

### Resume and compare

Use `--resume` after interruption or API failure. Successful request stages and
embeddings are cached. Changed image contents/paths, catalog, evaluator templates,
preprocessing, model profiles, or relevant library versions require a new run
directory. Concurrency, timeout, retry count, and rate limit can change on resume;
the latest execution settings are recorded separately.

API errors leave the run incomplete. Reports describe only completed images and
include coverage; technical errors are not silently converted into generation
failures. Re-export without inference, and compare compatible complete runs:

```bash
uv run --no-sync human-bench report runs/model-a
uv run --no-sync human-bench compare runs/model-a runs/model-b --output comparison.json
```

Comparisons require matching catalog, evaluator profile, exploratory/standard mode,
and prompt/sample coverage. Differences use paired prompt bootstrap intervals.
API providers may update model aliases internally; pin immutable model IDs where
available. Seeds and temperature zero do not guarantee identical remote inference.

## Gradio review

```bash
uv run --no-sync human-bench review --runs-dir runs
```

Open `http://127.0.0.1:7860`. Browse completed or running checkpoints; filter by
family, difficulty, adherence failures, visible defects, uncertainty, errors, or
pending images. Select a row to inspect the image, prompt, blind caption, evidence,
and anatomy findings. Compare shared prompt/sample pairs side by side and download
the latest exported results. Refresh runs or reload results to see new checkpoints.

The app is read-only: no evaluation jobs, annotations, or human-label collection.
It binds to localhost with public sharing disabled. For a remote host, use an SSH
tunnel, for example `ssh -L 7860:127.0.0.1:7860 gpu-host`.

## Development checks

```bash
uv sync --extra dev --extra review
uv run --no-sync ruff check src scripts tests
uv run --no-sync ruff format --check src scripts tests
uv run --no-sync pytest -q
uv run --no-sync python scripts/build_catalog.py --check
uv run --no-sync python -m build --wheel
```

Tests cover real-client HTTP contracts against a local mock API, retries, caching,
interruption and resume, input validation, geometry and aggregation, exports, and
Gradio callbacks. They do not download model weights or use paid endpoints.

For an opt-in check against a real configured endpoint:

```bash
HUMAN_BENCH_LIVE_CONFIG=configs/vlm-only.yaml \
HUMAN_BENCH_LIVE_MANIFEST=manifest.csv \
uv run --no-sync pytest -q tests/test_live.py
```

The live check verifies image input and output schemas, not evaluator accuracy.
It is skipped unless `HUMAN_BENCH_LIVE_CONFIG` is set.
