import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from human_bench.schemas import StrictModel


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class VLMConfig(StrictModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    revision: str | None = None
    api_key_env: str = "VLM_API_KEY"
    response_format: Literal["json_schema", "json_object", "prompt_json"] = "json_schema"
    request_kwargs: dict = Field(default_factory=lambda: {"temperature": 0, "max_tokens": 8192})
    extra_body: dict = Field(
        default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}}
    )

    @field_validator("request_kwargs")
    @classmethod
    def reserved_fields(cls, value: dict) -> dict:
        if set(value) & {"model", "messages", "response_format", "extra_body", "stream", "n"}:
            raise ValueError(
                "Model, messages, response format, stream and n are managed by the runner"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def no_url_credentials(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("base_url must be an HTTP(S) endpoint")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Use api_key_env, not credentials or query parameters in base_url")
        return value.rstrip("/")


class EmbeddingConfig(StrictModel):
    backend: Literal["local_gemma", "api", "none"] = "local_gemma"
    model: str = "google/embeddinggemma-2"
    revision: str | None = None
    base_url: str = "http://localhost:8001/v1"
    api_key_env: str = "EMBEDDING_API_KEY"
    device: str = "cpu"
    batch_size: int = Field(default=16, ge=1)
    dimensions: int | None = Field(default=None, ge=1)
    text_prefix: str = "task: sentence similarity | query: "

    _base_url = field_validator("base_url")(VLMConfig.no_url_credentials.__func__)


class RunConfig(StrictModel):
    name: str = "unnamed-t2i-model"
    exploratory: bool = False
    concurrency: int = Field(default=4, ge=1, le=256)
    requests_per_minute: float | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=180, gt=0)
    attempts: int = Field(default=3, ge=1, le=3)
    image_max_edge: int = Field(default=2048, ge=64, le=16384)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @model_validator(mode="after")
    def credentials_only_in_environment(self):
        forbidden = {
            "api_key",
            "apikey",
            "authorization",
            "password",
            "access_token",
            "secret",
            "extra_headers",
            "default_headers",
            "token",
        }

        def inspect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in forbidden:
                        raise ValueError("Credentials must use api_key_env, never inline config")
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        inspect(self.model_dump())
        if self.embedding.backend == "local_gemma":
            if self.embedding.model != "google/embeddinggemma-2":
                raise ValueError("local_gemma requires google/embeddinggemma-2")
            if self.embedding.dimensions not in (None, 768):
                raise ValueError("The Gemma protocol uses the full 768 dimensions")
        return self


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()) or {})
