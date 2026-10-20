import asyncio
import os

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

from human_bench.config import RunConfig, digest
from human_bench.inference import RateLimiter, attempt_embedding
from human_bench.storage import Store


def normalize(vectors) -> list[list[float]]:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embedding backend returned invalid or non-finite vectors")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Embedding backend returned a zero vector")
    return (matrix / lengths).tolist()


class Embedder:
    def __init__(self, config: RunConfig, store: Store, client=None):
        self.config, self.store = config, store
        self.spec = config.embedding
        self.profile = digest(self.spec.model_dump())
        self.limiter = RateLimiter(
            config.requests_per_minute if self.spec.backend == "api" else None
        )
        self.lock = asyncio.Lock()
        self.dimension = None
        self.model = None
        self.client = client
        if self.spec.backend == "api" and self.client is None:
            self.client = AsyncOpenAI(
                base_url=self.spec.base_url,
                api_key=os.environ.get(self.spec.api_key_env, "EMPTY"),
                timeout=config.timeout_seconds,
                max_retries=0,
            )

    @property
    def supports_images(self):
        return self.spec.backend == "local_gemma"

    def _local_model(self):
        if self.model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install local embeddings with: uv sync --extra embedding"
                ) from exc
            use_bf16 = self.spec.device.startswith("cuda") and torch.cuda.is_bf16_supported()
            self.model = SentenceTransformer(
                self.spec.model,
                revision=self.spec.revision,
                device=self.spec.device,
                config_kwargs={"audio_config": None},
                model_kwargs={"torch_dtype": torch.bfloat16 if use_bf16 else torch.float32},
            )
        return self.model

    def _encode_local(self, values):
        return (
            self._local_model()
            .encode(
                values,
                batch_size=self.spec.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                prompt="",
            )
            .tolist()
        )

    def _check(self, values):
        result = normalize(values)
        dim = len(result[0])
        if self.dimension is not None and dim != self.dimension:
            raise ValueError("Embedding dimensionality changed within a run")
        if self.spec.backend == "local_gemma" and dim != 768:
            raise ValueError("The local Gemma protocol requires 768-dimensional embeddings")
        if self.spec.dimensions is not None and dim != self.spec.dimensions:
            raise ValueError("Embedding dimensionality does not match the requested dimensions")
        self.dimension = dim
        return result

    async def texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self.lock:
            keys = [digest({"embedding_profile": self.profile, "text": t}) for t in texts]
            unique = {key: i for i, key in enumerate(keys)}
            missing = [i for key, i in unique.items() if self.store.cached(key) is None]
            # Cache inputs independently; changed captions do not invalidate prompt vectors.
            for offset in range(0, len(missing), self.spec.batch_size):
                indices = missing[offset : offset + self.spec.batch_size]
                values = [self.spec.text_prefix + texts[i] for i in indices]

                async def operation(values=values):
                    if self.spec.backend == "local_gemma":
                        vectors = await asyncio.to_thread(self._encode_local, values)
                        metadata = {
                            "returned_model": self.spec.model,
                            "revision": self.spec.revision,
                        }
                    elif self.spec.backend == "api":
                        kwargs = (
                            {"dimensions": self.spec.dimensions} if self.spec.dimensions else {}
                        )
                        response = await self.client.embeddings.create(
                            model=self.spec.model,
                            input=values,
                            encoding_format="float",
                            **kwargs,
                        )
                        ordered = sorted(response.data, key=lambda item: item.index)
                        if [item.index for item in ordered] != list(range(len(values))):
                            raise ValueError("Incomplete or duplicated embedding response indices")
                        vectors = [item.embedding for item in ordered]
                        metadata = {
                            "returned_model": response.model,
                            "usage": response.usage.model_dump(),
                        }
                    else:
                        raise ValueError("Embeddings are disabled")
                    if len(vectors) != len(values):
                        raise ValueError("Wrong number of embedding vectors")
                    return self._check(vectors), metadata

                vectors = await attempt_embedding(
                    self.config,
                    self.store,
                    self.limiter,
                    "text_embedding",
                    digest([keys[i] for i in indices]),
                    operation,
                )
                for i, vector in zip(indices, vectors, strict=True):
                    self.store.cache(keys[i], vector)
            return self._check([self.store.cached(key) for key in keys])

    async def image(self, image: Image.Image, image_hash: str, max_edge: int) -> list[float]:
        if not self.supports_images:
            raise ValueError("This embedding backend does not support images")
        async with self.lock:
            key = digest(
                {
                    "embedding_profile": self.profile,
                    "image": image_hash,
                    "max_edge": max_edge,
                    "preprocessing": "rgb-exif-white-png-v1",
                }
            )
            cached = self.store.cached(key)
            if cached is not None:
                return self._check([cached])[0]

            async def operation():
                # Gemma's multimodal inputs use image placeholders, with no text task prefix.
                values = [{"text": "<|image|>", "image": [image]}]
                vectors = self._check(await asyncio.to_thread(self._encode_local, values))
                return vectors, {"returned_model": self.spec.model, "revision": self.spec.revision}

            vectors = await attempt_embedding(
                self.config,
                self.store,
                self.limiter,
                "image_embedding",
                key,
                operation,
            )
            self.store.cache(key, vectors[0])
            return vectors[0]

    async def close(self):
        if self.client:
            await self.client.close()


def cosine(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Cannot compare vectors of different dimensions")
    return float(np.clip(np.dot(x, y), -1, 1))


def embedding_metrics(prompt, counterfactual, negation, caption, image=None) -> dict:
    alignment = cosine(prompt, caption)
    margin = alignment - cosine(counterfactual, caption)
    image_margin = (
        cosine(image, prompt) - cosine(image, counterfactual) if image is not None else None
    )
    return {
        "caption_alignment": alignment,
        "caption_contrast_margin": margin,
        "negation_contrast_margin": alignment - cosine(negation, caption),
        "image_contrast_margin": image_margin,
        "adapted_agreement": (margin + image_margin) / 2 if image_margin is not None else None,
    }


def prompt_distinction(prompt, counterfactual, negation) -> float:
    return (
        1
        - (
            cosine(prompt, counterfactual)
            + cosine(prompt, negation)
            + cosine(counterfactual, negation)
        )
        / 3
    )
