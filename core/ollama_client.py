import httpx
import json
import sys
from typing import AsyncGenerator
from dataclasses import dataclass


@dataclass
class GenerationResponse:
    response: str
    model: str
    done: bool
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass
class EmbeddingResponse:
    embedding: list[float]
    model: str


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        num_ctx: int = 8192,
        stream: bool = False,
    ) -> GenerationResponse | AsyncGenerator[str, None]:
        client = await self._get_client()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system

        if stream:
            return self._stream_generate(client, payload)
        else:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            full_response = ""
            full_thinking = ""
            final_data = {}
            for line in lines:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if chunk := data.get("response"):
                            full_response += chunk
                        if chunk := data.get("thinking"):
                            full_thinking += chunk
                        if data.get("done"):
                            final_data = data
                    except json.JSONDecodeError:
                        continue
            if not final_data and lines:
                try:
                    final_data = json.loads(lines[0])
                except json.JSONDecodeError:
                    final_data = {}
            combined = full_response
            if full_thinking and not full_response:
                combined = full_thinking
            return GenerationResponse(
                response=combined,
                model=final_data.get("model", model),
                done=final_data.get("done", True),
                total_duration=final_data.get("total_duration"),
                prompt_eval_count=final_data.get("prompt_eval_count"),
                eval_count=final_data.get("eval_count"),
            )

    async def _stream_generate(
        self, client: httpx.AsyncClient, payload: dict
    ) -> AsyncGenerator[str, None]:
        async with client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if chunk := data.get("response"):
                            yield chunk
                    except json.JSONDecodeError:
                        # Skip malformed lines — Ollama streaming can produce
                        # partial lines, empty objects, or server error output
                        continue

    async def embeddings(self, model: str, prompt: str) -> EmbeddingResponse:
        client = await self._get_client()
        payload = {
            "model": model,
            "prompt": prompt,
        }
        response = await client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(
                f"Ollama /api/embeddings returned non-JSON response: "
                f"{response.text[:200]}"
            ) from e
        return EmbeddingResponse(
            embedding=data.get("embedding", []),
            model=data.get("model", model),
        )

    async def list_models(self) -> list[str]:
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(
                f"Ollama /api/tags returned non-JSON response: "
                f"{response.text[:200]}"
            ) from e
        return [model["name"] for model in data.get("models", [])]

    async def check_models(self, model_names: list[str]) -> list[str]:
        """
        Check which of the given model names exist in Ollama.

        :param model_names: Names to check (e.g. ["gemma4:e4b", "qwen3.5:9b"])
        :returns: List of model names that were NOT found in Ollama.
        """
        available = await self.list_models()
        missing = [name for name in model_names if name not in available]
        return missing

    async def is_available(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            print(f"Warning: Ollama health check failed: {e}", file=sys.stderr)
            return False
