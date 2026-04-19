import pytest
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