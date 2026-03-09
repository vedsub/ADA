from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeVar, TypedDict

from pydantic import BaseModel

StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class LLMClientConfig:
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @staticmethod
    def from_env() -> "LLMClientConfig":
        return LLMClientConfig(
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )


def _build_langfuse_openai_client(config: LLMClientConfig) -> Any:
    try:
        from langfuse.openai import OpenAI as LangfuseOpenAI

        return LangfuseOpenAI(
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    except ImportError:
        from langfuse.openai import openai as langfuse_openai

        return langfuse_openai.OpenAI(
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )


class OpenAILLMClient:
    """
    Centralized LLM client for all model calls.

    All requests are routed via Langfuse's OpenAI wrapper so prompts, outputs,
    timings, and cost telemetry are captured by Langfuse automatically.
    """

    def __init__(self, config: LLMClientConfig | None = None) -> None:
        self.config = config or LLMClientConfig.from_env()
        self.client = _build_langfuse_openai_client(self.config)

    def generate_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate free-form text output from chat messages."""
        response = self.client.chat.completions.create(
            model=model or self.config.chat_model,
            messages=list(messages),
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI completion returned empty content.")
        return content.strip()

    def generate_structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponseT],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StructuredResponseT:
        """
        Generate guaranteed JSON output using a Pydantic response model.

        Preferred path uses `beta.chat.completions.parse`; fallback requests a
        JSON object and validates it with Pydantic.
        """
        chosen_model = model or self.config.chat_model
        chosen_temperature = (
            self.config.temperature if temperature is None else temperature
        )

        parse_api = getattr(
            getattr(getattr(self.client, "beta", None), "chat", None), "completions", None
        )
        if parse_api is not None and hasattr(parse_api, "parse"):
            response = self.client.beta.chat.completions.parse(
                model=chosen_model,
                messages=list(messages),
                temperature=chosen_temperature,
                max_tokens=max_tokens,
                response_format=response_model,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Structured completion returned no parsed payload.")
            return parsed

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=list(messages),
            temperature=chosen_temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Structured completion returned empty content.")
        return response_model.model_validate_json(content)

    def generate_embeddings(
        self,
        text: str | Sequence[str],
        *,
        model: str | None = None,
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings for semantic similarity search.

        Uses `text-embedding-3-small` by default.
        """
        is_single = isinstance(text, str)
        input_payload = [text] if is_single else list(text)
        if not input_payload:
            raise ValueError("Embedding input cannot be empty.")

        response = self.client.embeddings.create(
            model=model or self.config.embedding_model,
            input=input_payload,
        )
        vectors = [item.embedding for item in response.data]
        return vectors[0] if is_single else vectors


_default_llm_client: OpenAILLMClient | None = None


def get_default_llm_client() -> OpenAILLMClient:
    global _default_llm_client
    if _default_llm_client is None:
        _default_llm_client = OpenAILLMClient()
    return _default_llm_client
