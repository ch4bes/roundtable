"""Tests for Ollama client error responses and config validation - Issue #13 error path coverage."""
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from core.ollama_client import OllamaClient
from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig


class TestOllamaClientErrorResponses:
    """Test Ollama client handles error responses gracefully."""

    @pytest.mark.asyncio
    async def test_generate_model_not_found_404(self):
        """When model doesn't exist (404), should raise appropriate error."""
        client = OllamaClient()

        # Mock a 404 response
        with patch.object(client, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = '{"error": "model not found"}'
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response,
            )

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Should raise an exception
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate(model="nonexistent-model", prompt="test")

    @pytest.mark.asyncio
    async def test_generate_server_error_500(self):
        """When Ollama returns 500, should raise appropriate error."""
        client = OllamaClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=mock_response,
            )

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await client.generate(model="model1", prompt="test")

    @pytest.mark.asyncio
    async def test_generate_timeout(self):
        """When request times out, should raise timeout error."""
        client = OllamaClient(timeout=1)  # 1 second timeout

        with patch.object(client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.TimeoutException):
                await client.generate(model="model1", prompt="test prompt")

    @pytest.mark.asyncio
    async def test_embeddings_model_not_found(self):
        """When embedding model doesn't exist, should handle error gracefully."""
        client = OllamaClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.headers = {"content-type": "application/json"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response,
            )

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await client.embeddings(model="nonexistent-embedding", prompt="test")

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self):
        """When Ollama is not running, should handle connection error."""
        client = OllamaClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            # list_models should raise or return empty list
            with pytest.raises(httpx.ConnectError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_generate_connection_refused(self):
        """When Ollama is not running, connection should be refused."""
        client = OllamaClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await client.generate(model="model1", prompt="test")


class TestConfigValidation:
    """Test config validation with invalid values - Issue #13 error path coverage."""

    def test_temperature_below_min(self):
        """Temperature below 0 should raise validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            ModelConfig(
                name="test-model",
                temperature=-0.5,
            )

    def test_temperature_above_max(self):
        """Temperature above 2 should raise validation error."""
        with pytest.raises(Exception):
            ModelConfig(
                name="test-model",
                temperature=2.5,
            )

    def test_temperature_at_min_boundary(self):
        """Temperature at 0 should be valid."""
        config = ModelConfig(
            name="test-model",
            temperature=0.0,
        )
        assert config.temperature == 0.0

    def test_temperature_at_max_boundary(self):
        """Temperature at 2 should be valid."""
        config = ModelConfig(
            name="test-model",
            temperature=2.0,
        )
        assert config.temperature == 2.0

    def test_max_tokens_zero_raises_error(self):
        """max_tokens of 0 should raise validation error (must be > 0)."""
        with pytest.raises(Exception):
            ModelConfig(
                name="test-model",
                max_tokens=0,
            )

    def test_max_tokens_negative_raises_error(self):
        """Negative max_tokens should raise validation error."""
        with pytest.raises(Exception):
            ModelConfig(
                name="test-model",
                max_tokens=-100,
            )

    def test_num_ctx_zero_raises_error(self):
        """num_ctx of 0 should raise validation error (must be > 0)."""
        with pytest.raises(Exception):
            ModelConfig(
                name="test-model",
                num_ctx=0,
            )

    def test_single_model_raises_error(self):
        """Config with only 1 model should raise validation error (need 2+)."""
        with pytest.raises(Exception):
            Config(
                models=[
                    ModelConfig(name="model1"),
                    # Only one model - should fail
                ],
                discussion=DiscussionConfig(max_rounds=5),
                context=ContextConfig(mode="summary_only"),
            )

    def test_two_models_valid(self):
        """Config with exactly 2 models should be valid."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=5),
            context=ContextConfig(mode="summary_only"),
        )
        assert len(config.models) == 2

    def test_consensus_threshold_below_zero(self):
        """Consensus threshold below 0 should raise validation error."""
        with pytest.raises(Exception):
            DiscussionConfig(consensus_threshold=-0.1)

    def test_consensus_threshold_above_one(self):
        """Consensus threshold above 1 should raise validation error."""
        with pytest.raises(Exception):
            DiscussionConfig(consensus_threshold=1.5)

    def test_last_n_responses_zero_raises_error(self):
        """last_n_responses of 0 should raise validation error (must be > 0)."""
        with pytest.raises(Exception):
            ContextConfig(last_n_responses=0)

    def test_last_n_responses_negative_raises_error(self):
        """Negative last_n_responses should raise validation error."""
        with pytest.raises(Exception):
            ContextConfig(last_n_responses=-1)

    def test_max_rounds_zero_raises_error(self):
        """max_rounds of 0 should raise validation error (must be > 0)."""
        with pytest.raises(Exception):
            DiscussionConfig(max_rounds=0)

    def test_invalid_context_mode_raises_error(self):
        """Invalid context mode should raise validation error."""
        with pytest.raises(Exception):
            ContextConfig(mode="invalid_mode")

    def test_invalid_consensus_mode_raises_error(self):
        """Invalid consensus mode should raise validation error."""
        from core.config import ConsensusConfig
        with pytest.raises(Exception):
            ConsensusConfig(mode="invalid_mode")

    def test_invalid_consensus_method_raises_error(self):
        """Invalid consensus method should raise validation error."""
        from core.config import ConsensusConfig
        with pytest.raises(Exception):
            ConsensusConfig(method="invalid_method")

    def test_invalid_rotation_order_raises_error(self):
        """Invalid rotation order should raise validation error."""
        with pytest.raises(Exception):
            DiscussionConfig(rotation_order="invalid_order")


class TestConfigLoadInvalid:
    """Test Config.load() with invalid JSON files."""

    def test_load_nonexistent_file_returns_default(self):
        """Loading nonexistent file should return default config."""
        config = Config.load("/nonexistent/path/config.json")
        assert config is not None
        assert isinstance(config, Config)

    def test_load_invalid_json_raises_error(self):
        """Loading invalid JSON should raise error."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                Config.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestOllamaClientNonJsonResponse:
    """Test handling of non-JSON responses from Ollama."""

    @pytest.mark.asyncio
    async def test_generate_non_json_response(self):
        """When Ollama returns HTML instead of JSON, should handle gracefully."""
        client = OllamaClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "<html>Error</html>"
            mock_response.strip.return_value = "<html>Error</html>"

            # raise_for_status doesn't raise for 200
            mock_response.raise_for_status = MagicMock()

            # Simulate what happens when json() is called on HTML
            def json_side_effect():
                raise ValueError("Expecting value")

            mock_response.json = json_side_effect
            mock_response.text.strip.return_value = "<html>Error</html>"
            mock_response.text.strip().split.return_value = ["<html>Error</html>"]

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Should handle gracefully (not crash)
            result = await client.generate(model="model1", prompt="test")
            # May return empty or partial result, but shouldn't crash
            assert result is not None