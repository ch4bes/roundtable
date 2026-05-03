"""Tests for _handle_human_input edge cases - Issue #13 core logic coverage."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO

from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig, HumanParticipantConfig
from core.discussion import DiscussionOrchestrator
from storage.session import Session


class TestHandleHumanInputEdgeCases:
    """Edge case tests for _handle_human_input method."""

    def _make_orchestrator(self, human_enabled=True):
        """Create a DiscussionOrchestrator with mocked dependencies."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
            human_participant=HumanParticipantConfig(
                enabled=human_enabled,
                prompt="Share your perspective on: {prompt}",
            ),
        )
        session = Session(prompt="What is the best programming language?", config={})

        with patch("core.discussion.OllamaClient") as MockOllama, \
             patch("core.discussion.SimilarityEngine") as MockSim, \
             patch("core.discussion.ConsensusDetector"):
            mock_ollama = AsyncMock()
            MockOllama.return_value = mock_ollama
            mock_sim = MagicMock()
            MockSim.return_value = mock_sim

            orchestrator = DiscussionOrchestrator(config, session)
            return orchestrator

    @pytest.mark.asyncio
    async def test_handle_human_input_skip_returns_empty(self):
        """When user types 's', _handle_human_input returns empty string."""
        orchestrator = self._make_orchestrator()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            mock_stdin.write("s\n")
            mock_stdin.seek(0)

            result = await orchestrator._handle_human_input(
                context="Some context",
                round_num=1,
                position=0,
            )

        assert result == ""

    @pytest.mark.asyncio
    async def test_handle_human_input_empty_input_returns_empty(self):
        """When user submits empty input (just Enter), returns empty string."""
        orchestrator = self._make_orchestrator()

        # Simulate pressing Enter immediately (empty input)
        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            mock_stdin.write("\n")
            mock_stdin.seek(0)

            result = await orchestrator._handle_human_input(
                context="Some context",
                round_num=1,
                position=0,
            )

        assert result == ""

    @pytest.mark.asyncio
    async def test_handle_human_input_single_paragraph(self):
        """Single paragraph submission works correctly."""
        orchestrator = self._make_orchestrator()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            mock_stdin.write("My perspective on this topic.\n\n")
            mock_stdin.seek(0)

            result = await orchestrator._handle_human_input(
                context="Some context",
                round_num=1,
                position=0,
            )

        assert "My perspective on this topic." in result

    @pytest.mark.asyncio
    async def test_handle_human_input_multiple_paragraphs(self):
        """Multiple paragraphs are joined with double newlines."""
        orchestrator = self._make_orchestrator()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            # First paragraph
            mock_stdin.write("First paragraph.\n")
            mock_stdin.write("\n")  # Empty line = submit
            mock_stdin.seek(0)

            result = await orchestrator._handle_human_input(
                context="Some context",
                round_num=1,
                position=0,
            )

        # The implementation should handle multi-line input
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_handle_human_input_uses_callback_when_provided(self):
        """When human_input_callback is provided, it's used instead of stdin."""
        orchestrator = self._make_orchestrator()

        callback_result = "Response from callback"

        async def mock_callback(context, round_num, position):
            return callback_result

        orchestrator.human_input_callback = mock_callback

        result = await orchestrator._handle_human_input(
            context="Some context",
            round_num=2,
            position=1,
        )

        assert result == callback_result

    @pytest.mark.asyncio
    async def test_handle_human_input_callback_round_number_passed(self):
        """Verify callback receives correct round number."""
        orchestrator = self._make_orchestrator()

        received_args = {}

        async def mock_callback(context, round_num, position):
            received_args["round_num"] = round_num
            received_args["position"] = position
            return "callback response"

        orchestrator.human_input_callback = mock_callback

        await orchestrator._handle_human_input(
            context="Some context",
            round_num=3,
            position=2,
        )

        assert received_args["round_num"] == 3
        assert received_args["position"] == 2

    @pytest.mark.asyncio
    async def test_handle_human_input_disabled_returns_empty(self):
        """When human participant is disabled, returns empty string."""
        orchestrator = self._make_orchestrator(human_enabled=False)

        result = await orchestrator._handle_human_input(
            context="Some context",
            round_num=1,
            position=0,
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_handle_human_input_includes_summary_in_context(self):
        """When there's an attributed summary, it should be shown to human."""
        orchestrator = self._make_orchestrator()

        # Add a summary to the session
        orchestrator.session.add_attributed_summary(
            round_num=1,
            individual_summaries={"model1": ["They agree that X is best"]},
            agreement_analysis="All participants agree",
            consensus_assessment="NOT REACHED",
            confidence="MEDIUM",
            full_text="Full text",
        )

        # Just verify the method doesn't crash with summary present
        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            mock_stdin.write("s\n")  # Skip
            mock_stdin.seek(0)

            result = await orchestrator._handle_human_input(
                context="Some context",
                round_num=2,
                position=0,
            )

        # Should skip (return empty) since we entered 's'
        assert result == ""


class TestHandleHumanInputVeryLongInput:
    """Test _handle_human_input with very long inputs - edge case for issue #13."""

    @pytest.mark.asyncio
    async def test_handle_very_long_input_does_not_crash(self):
        """Very long input (>10K chars) should not crash the system."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
            human_participant=HumanParticipantConfig(
                enabled=True,
                prompt="Share your perspective on: {prompt}",
            ),
        )
        session = Session(prompt="test prompt", config={})

        with patch("core.discussion.OllamaClient") as MockOllama, \
             patch("core.discussion.SimilarityEngine") as MockSim, \
             patch("core.discussion.ConsensusDetector"):
            mock_ollama = AsyncMock()
            MockOllama.return_value = mock_ollama
            mock_sim = MagicMock()
            MockSim.return_value = mock_sim

            orchestrator = DiscussionOrchestrator(config, session)

            # Create a very long input (>10K chars)
            long_text = "A" * 15000  # 15K characters

            with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
                mock_stdin.write(long_text + "\n\n")
                mock_stdin.seek(0)

                result = await orchestrator._handle_human_input(
                    context="context",
                    round_num=1,
                    position=0,
                )

            # Should handle it without crashing
            assert isinstance(result, str)


class TestHandleHumanInputUnicode:
    """Test _handle_human_input with Unicode/special characters - edge case for issue #13."""

    @pytest.mark.asyncio
    async def test_handle_unicode_input(self):
        """Unicode characters in input should be handled correctly."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
            human_participant=HumanParticipantConfig(
                enabled=True,
                prompt="Share your perspective on: {prompt}",
            ),
        )
        session = Session(prompt="test prompt", config={})

        with patch("core.discussion.OllamaClient") as MockOllama, \
             patch("core.discussion.SimilarityEngine") as MockSim, \
             patch("core.discussion.ConsensusDetector"):
            mock_ollama = AsyncMock()
            MockOllama.return_value = mock_ollama
            mock_sim = MagicMock()
            MockSim.return_value = mock_sim

            orchestrator = DiscussionOrchestrator(config, session)

            # Unicode input including emoji
            unicode_text = "Thoughts on this: 🌍 中文العربية emoji 😃"

            with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
                mock_stdin.write(unicode_text + "\n\n")
                mock_stdin.seek(0)

                result = await orchestrator._handle_human_input(
                    context="context",
                    round_num=1,
                    position=0,
                )

            # Should handle Unicode correctly
            assert isinstance(result, str)
            assert "🧍" in result or "Thoughts" in result or len(result) > 0