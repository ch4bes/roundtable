import base64
import httpx
import json
import sys
from pathlib import Path
from typing import AsyncGenerator
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Allowed image file types and their magic-byte signatures
# (used to prevent local file disclosure when --image is passed)
# ---------------------------------------------------------------------------
_ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
}

# format -> (bytes_prefix, min_read_len)
_IMAGE_MAGIC: dict[str, tuple[bytes, int]] = {
    "PNG": (b"\x89PNG", 4),
    "JPG": (b"\xff\xd8\xff", 3),
    "WEBP": (b"RIFF", 8),  # RIFF....WEBP
    "GIF": (b"GIF8", 4),
    "BMP": (b"BM", 2),
    "TIFF_LE": (b"II\x2A\x00", 4),
    "TIFF_BE": (b"MM\x00\x2A", 4),
}


def _is_valid_image_path(path: Path) -> tuple[bool, str]:
    """Check that *path* looks like a genuine image file.

    Returns ``(True, "")`` when the file passes all checks, or
    ``(False, reason)`` when validation fails.
    """
    # Extension whitelist
    ext = path.suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXTENSIONS:
        return False, f"extension {ext!r} not in allowed set"

    # Magic-byte verification
    try:
        with open(path, "rb") as f:
            # Read enough bytes for the longest header check (WEBP = 8)
            header = f.read(8)
    except (OSError, PermissionError) as e:
        return False, f"cannot read file: {e}"

    for _fmt, (prefix, min_len) in _IMAGE_MAGIC.items():
        if len(header) >= min_len and header[:len(prefix)] == prefix:
            # WEBP needs an extra check for the "WEBP" payload tag at offset 8
            if _fmt == "WEBP" and len(header) >= 12 and header[8:12] != b"WEBP":
                continue
            return True, ""

    return False, f"file header does not match any supported image format"


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

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _supports_think_param(self, model: str) -> bool:
        """Check if the model supports the 'think' parameter.
        
        Qwen3 models support the think parameter, other models may not.
        """
        model_lower = model.lower()
        return "qwen3" in model_lower or "qwen2.5" in model_lower

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
        client = await self._get_client()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }
        
        # Only add 'think' parameter for models that support it (Qwen3)
        if self._supports_think_param(model):
            payload["think"] = False
        
        if system:
            payload["system"] = system

        if images:
            # images can be file paths (will be read and base64 encoded) or base64 strings
            encoded_images = []
            for img in images:
                p = Path(img)
                if p.exists() and p.is_file():
                    # Validate before reading to prevent local file disclosure.
                    ok, reason = _is_valid_image_path(p)
                    if ok:
                        with open(p.resolve(), "rb") as f:
                            encoded_images.append(
                                base64.b64encode(f.read()).decode("utf-8")
                            )
                    else:
                        print(
                            f"Warning: Skipping {img!r} - {reason}",
                            file=sys.stderr,
                        )
                    continue

                # Path does not exist or is not a regular file;
                # assume it's already base64 encoded.
                encoded_images.append(img)
            if encoded_images:
                payload["images"] = encoded_images

        if stream:
            return self._stream_generate(client, payload)
        else:
            try:
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Try to extract the error message from Ollama's response
                error_msg = e.response.text
                try:
                    error_data = json.loads(error_msg)
                    ollama_error = error_data.get("error", "Unknown error")
                    raise RuntimeError(
                        f"Ollama API error: {ollama_error}\n"
                        f"Model: {model}\n"
                        f"This may indicate the model doesn't exist or doesn't support the requested parameters."
                    ) from e
                except (json.JSONDecodeError, ValueError):
                    raise RuntimeError(
                        f"Ollama API error: HTTP {e.response.status_code}\n"
                        f"Response: {error_msg[:500]}\n"
                        f"Model: {model}"
                    ) from e
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
        try:
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
        except httpx.HTTPStatusError as e:
            error_msg = e.response.text
            try:
                error_data = json.loads(error_msg)
                ollama_error = error_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"Ollama API error (streaming): {ollama_error}\n"
                    f"Model: {payload.get('model')}\n"
                    f"This may indicate the model doesn't exist or doesn't support the requested parameters."
                ) from e
            except (json.JSONDecodeError, ValueError):
                raise RuntimeError(
                    f"Ollama API error (streaming): HTTP {e.response.status_code}\n"
                    f"Response: {error_msg[:500]}\n"
                    f"Model: {payload.get('model')}"
                ) from e

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
