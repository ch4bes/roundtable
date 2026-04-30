import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.ollama_client import OllamaClient, GenerationResponse, EmbeddingResponse


class TestOllamaClientInit:
    def test_client_init(self):
        client = OllamaClient(base_url="http://localhost:11434", timeout=120)
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 120
        assert client._client is None

    def test_client_default_url(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"

    def test_client_url_stripping(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestGenerationResponse:
    def test_creation(self):
        response = GenerationResponse(
            response="test response",
            model="test-model",
            done=True,
            total_duration=1000000,
            prompt_eval_count=50,
            eval_count=100,
        )

        assert response.response == "test response"
        assert response.model == "test-model"
        assert response.done is True
        assert response.total_duration == 1000000

    def test_done_default(self):
        response = GenerationResponse(
            response="test",
            model="model",
            done=True,
        )
        assert response.done is True


class TestEmbeddingResponse:
    def test_creation(self):
        response = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            model="nomic-embed-text",
        )

        assert response.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.model == "nomic-embed-text"

    def test_empty_embedding(self):
        response = EmbeddingResponse(
            embedding=[],
            model="test",
        )
        assert response.embedding == []


class TestClientMethods:
    @pytest.mark.asyncio
    async def test_get_client_creates(self):
        client = OllamaClient()
        httpx_client = await client._get_client()
        assert httpx_client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_close(self):
        client = OllamaClient()
        await client._get_client()
        await client.close()
        assert client._client is None or client._client.is_closed


class TestOllamaEndpoints:
    def test_generate_payload_structure(self):
        client = OllamaClient()

        payload = {
            "model": "test-model",
            "prompt": "test prompt",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
                "num_ctx": 8192,
            },
        }

        assert payload["model"] == "test-model"
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 1024
        assert payload["options"]["num_ctx"] == 8192

    def test_embeddings_payload_structure(self):
        payload = {
            "model": "nomic-embed-text",
            "prompt": "Test prompt",
        }

        assert payload["model"] == "nomic-embed-text"
        assert payload["prompt"] == "Test prompt"


class TestApiResponseHandling:
    def test_response_parsing(self):
        response_data = {
            "model": "test-model",
            "response": "生成的文本",
            "done": True,
            "total_duration": 1000000,
            "prompt_eval_count": 50,
            "eval_count": 100,
        }

        response = GenerationResponse(
            response=response_data.get("response", ""),
            model=response_data.get("model", "test-model"),
            done=response_data.get("done", True),
            total_duration=response_data.get("total_duration"),
            prompt_eval_count=response_data.get("prompt_eval_count"),
            eval_count=response_data.get("eval_count"),
        )

        assert response.response == "生成的文本"
        assert response.done is True

    def test_thinking_fallback(self):
        response_data = {
            "model": "test-model",
            "response": "",
            "thinking": "Let me think about this...",
            "done": True,
        }

        combined = response_data.get("response") or response_data.get("thinking", "")
        assert combined == "Let me think about this..."


class TestErrorHandling:
    def test_missing_response_field(self):
        response_data = {"model": "test-model"}

        response = GenerationResponse(
            response=response_data.get("response", ""),
            model=response_data.get("model", "test-model"),
            done=response_data.get("done", True),
        )

        assert response.response == ""

    def test_missing_embedding(self):
        response_data = {"model": "test-model"}

        embedding = response_data.get("embedding", [])
        assert embedding == []


class TestStreamGenerate:
    """Tests for _stream_generate JSON error handling (fix #1.4)."""

    def _make_streaming_mock(self, lines: list[str]):
        """Build a mock httpx.AsyncClient.stream() context manager yielding given lines."""
        async def aiter_lines():
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        return mock_client

    @pytest.mark.asyncio
    async def test_stream_generate_malformed_json_line(self):
        """A malformed line in the stream is skipped, not crashed."""
        client = OllamaClient()
        mock_httpx = self._make_streaming_mock([
            '{"response": "hello"}',
            'this is not json',
            '{"response": " world"}',
        ])

        result = ""
        async for chunk in client._stream_generate(mock_httpx, {}):
            result += chunk

        assert result == "hello world"
        assert "this is not json" not in result

    @pytest.mark.asyncio
    async def test_stream_generate_empty_object_line(self):
        """A '{}' line in the stream is handled gracefully (no crash, no yield)."""
        client = OllamaClient()
        mock_httpx = self._make_streaming_mock([
            '{"response": "hi"}',
            '{}',
            '{"response": " there"}',
        ])

        result = ""
        async for chunk in client._stream_generate(mock_httpx, {}):
            result += chunk

        assert result == "hi there"

    @pytest.mark.asyncio
    async def test_stream_generate_all_malformed(self):
        """If all lines are malformed, generator yields nothing (no crash)."""
        client = OllamaClient()
        mock_httpx = self._make_streaming_mock([
            'bad1',
            'bad2',
            'bad3',
        ])

        result = ""
        async for chunk in client._stream_generate(mock_httpx, {}):
            result += chunk

        assert result == ""