"""
Abstract base class and data classes for the LLMClient interface.

This module defines the LLMClient abstract base class (ABC) that all LLM
backend clients must implement. It enables dependency injection: the
orchestrator depends on the abstraction (LLMClient), not the concrete
OllamaClient, allowing easy swapping of backends and straightforward testing.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Data classes -- shared between interface and implementations
# ---------------------------------------------------------------------------


@dataclass
class GenerationResponse:
    """Response from a non-streaming text-generation call."""
    response: str
    model: str
    done: bool
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding call."""
    embedding: list[float]
    model: str


@dataclass
class ToolCall:
    """Represents a tool call returned by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    message: str
    model: str
    done: bool
    tool_calls: list[ToolCall] | None = None
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract interface for any LLM backend (Ollama, OpenAI, etc.).

    Subclasses **must** implement `generate`, `embeddings`, and `chat`.
    Concrete helpers such as ``list_models``, ``check_models``,
    ``is_available``, and ``close`` may be provided for convenience.
    """

    # -- Core LLM protocol (abstract) ---------------------------------------

    @abstractmethod
    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        num_ctx: int = 8192,
        stream: bool = False,
        images: list[str] | None = None,
    ) -> GenerationResponse | AsyncGenerator[str, None]:
        """Generate text from a model.

        In non-streaming mode returns a ``GenerationResponse``.  In
        streaming mode returns an *async generator* yielding partial
        strings.
        """

    @abstractmethod
    async def embeddings(
        self,
        model: str,
        prompt: str,
    ) -> EmbeddingResponse:
        """Generate an embedding vector for the given text."""

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        num_ctx: int = 8192,
        tool_executor: Any | None = None,
        max_tool_calls: int = 5,
    ) -> ChatResponse:
        """Chat with a model, optionally using tool calls."""

    # -- Lifecycle / utilities (non-abstract, overridable) ------------------

    async def list_models(self) -> list[str]:
        """Return list of available model names (default: NotImplementedError)."""
        raise NotImplementedError("Backend does not implement list_models")

    async def check_models(self, model_names: list[str]) -> list[str]:
        """Return model names from *model_names* that are NOT available."""
        available = await self.list_models()
        return [name for name in model_names if name not in available]

    async def is_available(self) -> bool:
        """Health-check the backend (default: NotImplementedError)."""
        raise NotImplementedError("Backend does not implement is_available")

    async def close(self) -> None:
        """Release backend resources (default: no-op)."""
