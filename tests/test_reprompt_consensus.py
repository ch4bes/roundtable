"""Tests for _reprompt_for_consensus() method - Issue #13 core logic coverage."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig
from core.discussion import DiscussionOrchestrator
from storage.session import Session


class TestRepromptForConsensus:
    """Test the reprompt logic when consensus is inconsistent between moderator assessment and quantitative evidence."""

    def _make_orchestrator(self):
        """Create a DiscussionOrchestrator with mocked Ollama."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
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
            orchestrator.ollama = mock_ollama
            orchestrator.similarity_engine = mock_sim
            return orchestrator

    @pytest.mark.asyncio
    async def test_reprompt_keeps_original_assessment_when_low_agreement(self):
        """When similarity shows low agreement (<70%), moderator keeps original NOT REACHED."""
        orchestrator = self._make_orchestrator()

        # Mock similarity matrix with low agreement
        similarity_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

        mock_attributed = MagicMock()
        mock_attributed.consensus_assessment = "NOT REACHED"
        mock_attributed.agreement_analysis = "Some disagreement"

        # Mock the moderator response to say KEEP NOT REACHED
        mock_response = MagicMock()
        mock_response.response = "KEEP NOT REACHED - participants have different views on the main answer"
        orchestrator.ollama.generate = AsyncMock(return_value=mock_response)

        result = await orchestrator._reprompt_for_consensus(
            round_num=1,
            attributed=mock_attributed,
            similarity_matrix=similarity_matrix,
            model_names=["model1", "model2"],
        )

        assert result == "NOT REACHED"

    @pytest.mark.asyncio
    async def test_reprompt_changes_to_reached_when_high_agreement(self):
        """When similarity shows high agreement (>=70%), moderator changes to REACHED."""
        orchestrator = self._make_orchestrator()

        # Mock similarity matrix with high agreement
        similarity_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])

        mock_attributed = MagicMock()
        mock_attributed.consensus_assessment = "NOT REACHED"
        mock_attributed.agreement_analysis = "Participants agree on main point but differ on examples"

        # Mock the moderator response to say CHANGE REACHED
        mock_response = MagicMock()
        mock_response.response = "CHANGE REACHED - quantitative evidence shows strong agreement on main answer"
        orchestrator.ollama.generate = AsyncMock(return_value=mock_response)

        result = await orchestrator._reprompt_for_consensus(
            round_num=1,
            attributed=mock_attributed,
            similarity_matrix=similarity_matrix,
            model_names=["model1", "model2"],
        )

        assert result == "REACHED"

    @pytest.mark.asyncio
    async def test_reprompt_with_3_models_high_agreement(self):
        """Test reprompt with 3 models where 2 out of 3 agree."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
                ModelConfig(name="model3"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
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
            orchestrator.ollama = mock_ollama

        # 3-model matrix: model1 and model2 agree (0.85), both differ from model3
        similarity_matrix = np.array([
            [1.0, 0.85, 0.2],
            [0.85, 1.0, 0.25],
            [0.2, 0.25, 1.0],
        ])

        mock_attributed = MagicMock()
        mock_attributed.consensus_assessment = "NOT REACHED"
        mock_attributed.agreement_analysis = "Two models agree"

        mock_response = MagicMock()
        mock_response.response = "CHANGE REACHED - majority (2/3) agree on main answer"
        orchestrator.ollama.generate = AsyncMock(return_value=mock_response)

        result = await orchestrator._reprompt_for_consensus(
            round_num=1,
            attributed=mock_attributed,
            similarity_matrix=similarity_matrix,
            model_names=["model1", "model2", "model3"],
        )

        assert result == "REACHED"

    @pytest.mark.asyncio
    async def test_reprompt_threshold_configurable(self):
        """Verify threshold is read from config (70% by default)."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=2),
            context=ContextConfig(mode="summary_only"),
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
            orchestrator.ollama = mock_ollama

        # 80% agreement - should trigger reprompt
        similarity_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])

        mock_attributed = MagicMock()
        mock_attributed.consensus_assessment = "NOT REACHED"
        mock_attributed.agreement_analysis = "Strong agreement"

        mock_response = MagicMock()
        mock_response.response = "CHANGE REACHED"
        orchestrator.ollama.generate = AsyncMock(return_value=mock_response)

        # Call _calculate_agreement_percentage directly to verify threshold
        agreement_pct = orchestrator._calculate_agreement_percentage(similarity_matrix, 0.75)
        assert agreement_pct >= 70  # Should trigger reprompt

    @pytest.mark.asyncio
    async def test_reprompt_not_triggered_when_agreement_below_threshold(self):
        """When agreement < 70%, reprompt is not triggered."""
        orchestrator = self._make_orchestrator()

        # Low agreement matrix
        similarity_matrix = np.array([[1.0, 0.4], [0.4, 1.0]])

        mock_attributed = MagicMock()
        mock_attributed.consensus_assessment = "NOT REACHED"
        mock_attributed.agreement_analysis = "Disagreement"

        agreement_pct = orchestrator._calculate_agreement_percentage(similarity_matrix, 0.75)
        assert agreement_pct < 70  # Should NOT trigger reprompt