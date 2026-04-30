import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from main import check_ollama
from core.config import Config


class TestCheckOllama:
     """Tests for main.check_ollama() — verifies correct client lifecycle."""

     @pytest.fixture
     def config(self):
         return Config()

     @pytest.fixture
     def mock_ollama_client(self):
         """Mock OllamaClient that tracks call order."""
         mock = AsyncMock()
         mock.is_available = AsyncMock(return_value=True)
         mock.list_models = AsyncMock(return_value=["gemma4:e4b", "qwen3.5:9b"])
         mock.close = AsyncMock()
         return mock

     @pytest.mark.asyncio
     async def test_check_ollama_available_lists_models(
         self, config, mock_ollama_client
     ):
         """When Ollama is available, list_models is called and models returned."""
         with patch("core.OllamaClient", return_value=mock_ollama_client):
             result = await check_ollama(config)

         assert result is True
         mock_ollama_client.is_available.assert_called_once()
         mock_ollama_client.list_models.assert_called_once()

     @pytest.mark.asyncio
     async def test_check_ollama_unavailable(self, config, mock_ollama_client):
         """When Ollama is unavailable, list_models is not called."""
         mock_ollama_client.is_available = AsyncMock(return_value=False)

         with patch("core.OllamaClient", return_value=mock_ollama_client):
             result = await check_ollama(config)

         assert result is False
         mock_ollama_client.list_models.assert_not_called()

     @pytest.mark.asyncio
     async def test_close_called_after_list_models(
         self, config, mock_ollama_client
     ):
         """close() is called AFTER list_models(), not before.

         Regression test: previously close() was called before list_models(),
         which caused the closed client to be used for a subsequent API call.
         """
         # Track call order
         call_order = []
         original_is_available = mock_ollama_client.is_available
         original_list_models = mock_ollama_client.list_models
         original_close = mock_ollama_client.close

         async def track_is_available():
             call_order.append("is_available")
             return await original_is_available()

         async def track_list_models():
             call_order.append("list_models")
             return await original_list_models()

         async def track_close():
             call_order.append("close")
             return await original_close()

         mock_ollama_client.is_available = track_is_available
         mock_ollama_client.list_models = track_list_models
         mock_ollama_client.close = track_close

         with patch("core.OllamaClient", return_value=mock_ollama_client):
             await check_ollama(config)

         # Verify correct order: is_available -> list_models -> close
         assert call_order == ["is_available", "list_models", "close"], (
             f"Expected is_available -> list_models -> close, got {call_order}"
         )

     @pytest.mark.asyncio
     async def test_close_called_on_exception(self, config, mock_ollama_client):
         """close() is always called even if list_models raises."""
         mock_ollama_client.list_models = AsyncMock(
             side_effect=RuntimeError("simulated failure")
         )

         with patch("core.OllamaClient", return_value=mock_ollama_client):
             with pytest.raises(RuntimeError, match="simulated failure"):
                 await check_ollama(config)

         # close() must still have been called (finally block)
         mock_ollama_client.close.assert_called_once()


class TestCheckOllamaModelsTruncation:
     """Verify that model listing handles large model counts correctly."""

     @pytest.fixture
     def config(self):
         return Config()

     @pytest.mark.asyncio
     async def test_many_models_shows_truncation(self, capsys, config):
         """When there are >10 models, output shows truncation message."""
         mock_client = AsyncMock()
         mock_client.is_available = AsyncMock(return_value=True)
         mock_client.list_models = AsyncMock(
             return_value=[f"model{i}" for i in range(15)]
         )
         mock_client.close = AsyncMock()

         with patch("core.OllamaClient", return_value=mock_client):
             await check_ollama(config)

         captured = capsys.readouterr()
         assert "model0" in captured.out
         assert "model9" in captured.out
         assert "and 5 more" in captured.out
         # model14 should NOT be listed
         assert "model14" not in captured.out

     @pytest.mark.asyncio
     async def test_few_models_no_truncation(self, capsys, config):
         """When there are <=10 models, all are listed with no truncation message."""
         mock_client = AsyncMock()
         mock_client.is_available = AsyncMock(return_value=True)
         mock_client.list_models = AsyncMock(
             return_value=[f"model{i}" for i in range(3)]
         )
         mock_client.close = AsyncMock()

         with patch("core.OllamaClient", return_value=mock_client):
             await check_ollama(config)

         captured = capsys.readouterr()
         assert "model0" in captured.out
         assert "model1" in captured.out
         assert "model2" in captured.out
         assert "and" not in captured.out.lower()   # no truncation message
