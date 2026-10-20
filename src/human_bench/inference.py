import asyncio
import os
import time
from collections.abc import Callable

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from human_bench.config import RunConfig, digest
from human_bench.prompts import SYSTEM, TEMPLATE_HASH, request_text
from human_bench.schemas import Caption, Evaluation, PromptSpec
from human_bench.storage import Store


class EvaluationError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, rpm: float | None):
        self.interval = 60 / rpm if rpm else 0
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait(self):
        async with self.lock:
            await asyncio.sleep(max(0, self.next_start - time.monotonic()))
            self.next_start = time.monotonic() + self.interval


def retryable(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            ValueError,
            ValidationError,
            EvaluationError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
        ),
    ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)


def safe_error(exc: Exception) -> str:
    # Never persist HTTP exception messages: they can echo headers or request payloads.
    if isinstance(exc, openai.APIStatusError):
        return f"{type(exc).__name__}: HTTP {exc.status_code}"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return type(exc).__name__
    return f"{type(exc).__name__}: {str(exc)[:1200]}"


class VLM:
    def __init__(self, config: RunConfig, store: Store, profile_hash: str, client=None):
        self.config, self.store, self.profile_hash = config, store, profile_hash
        self.client = client or AsyncOpenAI(
            base_url=config.vlm.base_url,
            api_key=os.environ.get(config.vlm.api_key_env, "EMPTY"),
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self.limiter = RateLimiter(config.requests_per_minute)
        self.locks: dict[str, asyncio.Lock] = {}

    async def close(self):
        await self.client.close()

    async def request(
        self, stage: str, image_url: str, image_hash: str, prompt: PromptSpec | None = None
    ) -> dict:
        schema_class = Evaluation if stage == "evaluation" else Caption
        text = request_text(stage, prompt)
        key = digest(
            {
                "profile": self.profile_hash,
                "image": image_hash,
                "template": TEMPLATE_HASH,
                "stage": stage,
                "text": text,
            }
        )
        async with self.locks.setdefault(key, asyncio.Lock()):
            cached = self.store.cached(key)
            if cached is not None:
                return cached
            schema = schema_class.model_json_schema()
            kwargs = dict(self.config.vlm.request_kwargs)
            mode = self.config.vlm.response_format
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_class.__name__.lower(),
                        "strict": True,
                        "schema": schema,
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            if self.config.vlm.extra_body:
                kwargs["extra_body"] = self.config.vlm.extra_body
            for number in range(1, self.config.attempts + 1):
                await self.limiter.wait()
                start = time.monotonic()
                record = {"attempt": number, "requested_model": self.config.vlm.model}
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.vlm.model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                    {"type": "text", "text": text},
                                ],
                            },
                        ],
                        **kwargs,
                    )
                    record.update(
                        {
                            "returned_model": response.model,
                            "request_id": getattr(response, "_request_id", None),
                            "usage": response.usage.model_dump() if response.usage else None,
                        }
                    )
                    if not response.choices:
                        raise EvaluationError("No response choices")
                    choice = response.choices[0]
                    record.update(
                        {
                            "raw_text": choice.message.content,
                            "refusal": getattr(choice.message, "refusal", None),
                            "finish_reason": choice.finish_reason,
                        }
                    )
                    if choice.finish_reason != "stop" or record["refusal"]:
                        raise EvaluationError("Refused or incomplete evaluator response")
                    parsed = schema_class.model_validate_json(choice.message.content or "")
                    if isinstance(parsed, Evaluation):
                        parsed.validate_for(prompt)
                    value = parsed.model_dump()
                    record.update(
                        {
                            "elapsed_seconds": time.monotonic() - start,
                            "output": value,
                            "error": None,
                        }
                    )
                    self.store.attempt(stage, key, record)
                    self.store.cache(key, value)
                    return value
                except Exception as exc:
                    error = safe_error(exc)
                    record.update({"elapsed_seconds": time.monotonic() - start, "error": error})
                    self.store.attempt(stage, key, record)
                    if number == self.config.attempts or not retryable(exc):
                        raise EvaluationError(f"{stage} failed: {error}") from exc
                    await asyncio.sleep(min(2 ** (number - 1), 4))
        raise AssertionError("Unreachable")


async def attempt_embedding(
    config: RunConfig, store: Store, limiter: RateLimiter, stage: str, key: str, operation: Callable
):
    for number in range(1, config.attempts + 1):
        await limiter.wait()
        start = time.monotonic()
        try:
            vectors, metadata = await operation()
            store.attempt(
                stage,
                key,
                {
                    "attempt": number,
                    "elapsed_seconds": time.monotonic() - start,
                    "error": None,
                    **metadata,
                },
            )
            return vectors
        except Exception as exc:
            store.attempt(
                stage,
                key,
                {
                    "attempt": number,
                    "elapsed_seconds": time.monotonic() - start,
                    "error": safe_error(exc),
                },
            )
            if number == config.attempts or not retryable(exc):
                raise EvaluationError(f"{stage} failed: {safe_error(exc)}") from exc
            await asyncio.sleep(min(2 ** (number - 1), 4))
