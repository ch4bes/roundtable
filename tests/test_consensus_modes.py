import pytest
import numpy as np
from core.config import Config, ModelConfig, ConsensusConfig, DiscussionConfig
from core.consensus import ConsensusDetector, ConsensusResult


class TestConsensusModesConfig:
    def test_moderator_decides_mode(self):
        config = Config(consensus=ConsensusConfig(mode="moderator_decides"))
        assert config.consensus.mode == "moderator_decides"

    def test_programmatic_decides_mode(self):
        config = Config(
            consensus=ConsensusConfig(
                mode="programmatic_decides",
                threshold=0.8,
            )
        )
        assert config.consensus.mode == "programmatic_decides"
        assert config.consensus.threshold == 0.8


class TestProgrammaticConsensus:
    def test_pairwise_full_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.88],
            [0.9, 0.88, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_pairwise_partial_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is False

    def test_clustering_majority(self):
        detector = ConsensusDetector(threshold=0.7, method="clustering")
        matrix = np.array([
            [1.0, 0.8, 0.75],
            [0.8, 1.0, 0.72],
            [0.75, 0.72, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_clustering_no_majority(self):
        detector = ConsensusDetector(threshold=0.85, method="clustering")
        matrix = np.array([
            [1.0, 0.9, 0.3, 0.3],
            [0.9, 1.0, 0.3, 0.3],
            [0.3, 0.3, 1.0, 0.9],
            [0.3, 0.3, 0.9, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is False


class TestConsensusConfigValidation:
    def test_consensus_method_pairwise(self):
        config = Config(
            discussion=DiscussionConfig(
                consensus_method="pairwise",
                consensus_threshold=0.9,
            )
        )
        assert config.discussion.consensus_method == "pairwise"
        assert config.discussion.consensus_threshold == 0.9

    def test_consensus_method_clustering(self):
        config = Config(
            discussion=DiscussionConfig(
                consensus_method="clustering",
                consensus_threshold=0.75,
            )
        )
        assert config.discussion.consensus_method == "clustering"
        assert config.discussion.consensus_threshold == 0.75


class TestConsensusEdgeCases:
    def test_single_response_pairwise(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([[1.0]])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_single_response_clustering(self):
        detector = ConsensusDetector(threshold=0.85, method="clustering")
        matrix = np.array([[1.0]])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_two_responses_agree(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([
            [1.0, 0.95],
            [0.95, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_two_responses_disagree(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([
            [1.0, 0.4],
            [0.4, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is False


class TestConsensusThresholdBehavior:
    def test_threshold_0(self):
        detector = ConsensusDetector(threshold=0.0, method="pairwise")
        matrix = np.array([
            [1.0, 0.1],
            [0.1, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is True

    def test_threshold_1(self):
        detector = ConsensusDetector(threshold=1.0, method="pairwise")
        matrix = np.array([
            [1.0, 0.99],
            [0.99, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is False

    def test_threshold_half(self):
        detector = ConsensusDetector(threshold=0.5, method="pairwise")
        matrix = np.array([
            [1.0, 0.6],
            [0.6, 1.0],
        ])

        result = detector.detect(matrix)
        assert result.reached is True


class TestGetSimilarPairs:
    def test_find_pairs_above_threshold(self):
        detector = ConsensusDetector(threshold=0.8)
        matrix = np.array([
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ])
        names = ["model1", "model2", "model3"]

        pairs = detector.get_similar_pairs(matrix, names)

        assert len(pairs) >= 1

    def test_no_pairs_below_threshold(self):
        detector = ConsensusDetector(threshold=0.95)
        matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])
        names = ["a", "b"]

        pairs = detector.get_similar_pairs(matrix, names)
        assert len(pairs) == 0


class TestConsensusResult:
    def test_result_dataclass(self):
        result = ConsensusResult(
            reached=True,
            percentage=100.0,
            agreeing_pairs=3,
            total_pairs=3,
            method="pairwise",
            details={"test": "data"},
        )

        assert result.reached is True
        assert result.percentage == 100.0
        assert result.method == "pairwise"