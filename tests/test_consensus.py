import pytest
import numpy as np
from core.consensus import ConsensusDetector, ConsensusResult


class TestPairwiseConsensus:
    def test_full_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array(
            [
                [1.0, 0.9, 0.95],
                [0.9, 1.0, 0.92],
                [0.95, 0.92, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is True
        assert result.percentage == 100.0
        assert result.agreeing_pairs == 3
        assert result.total_pairs == 3

    def test_no_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array(
            [
                [1.0, 0.5, 0.6],
                [0.5, 1.0, 0.7],
                [0.6, 0.7, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is False
        assert result.percentage == 0.0
        assert result.agreeing_pairs == 0

    def test_partial_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array(
            [
                [1.0, 0.9, 0.5],
                [0.9, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is False
        assert result.percentage == pytest.approx(33.33, rel=1)
        assert result.agreeing_pairs == 1
        assert result.total_pairs == 3

    def test_single_response(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array([[1.0]])

        result = detector.detect(matrix)

        assert result.reached is True
        assert result.percentage == 100.0

    def test_two_responses_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array(
            [
                [1.0, 0.9],
                [0.9, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is True
        assert result.percentage == 100.0

    def test_two_responses_no_consensus(self):
        detector = ConsensusDetector(threshold=0.85, method="pairwise")
        matrix = np.array(
            [
                [1.0, 0.5],
                [0.5, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is False


class TestClusteringConsensus:
    def test_single_cluster(self):
        detector = ConsensusDetector(threshold=0.7, method="clustering")
        matrix = np.array(
            [
                [1.0, 0.8, 0.75],
                [0.8, 1.0, 0.72],
                [0.75, 0.72, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is True
        assert result.percentage == 100.0

    def test_multiple_clusters(self):
        detector = ConsensusDetector(threshold=0.85, method="clustering")
        matrix = np.array(
            [
                [1.0, 0.9, 0.3],
                [0.9, 1.0, 0.3],
                [0.3, 0.3, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is True
        assert result.percentage == pytest.approx(66.67, rel=1)

    def test_no_clear_majority(self):
        detector = ConsensusDetector(threshold=0.85, method="clustering")
        matrix = np.array(
            [
                [1.0, 0.9, 0.3, 0.3],
                [0.9, 1.0, 0.3, 0.3],
                [0.3, 0.3, 1.0, 0.9],
                [0.3, 0.3, 0.9, 1.0],
            ]
        )

        result = detector.detect(matrix)

        assert result.reached is False
        assert result.percentage == 50.0


class TestGetSimilarPairs:
    def test_get_similar_pairs(self):
        detector = ConsensusDetector(threshold=0.8)
        matrix = np.array(
            [
                [1.0, 0.9, 0.5],
                [0.9, 1.0, 0.6],
                [0.5, 0.6, 1.0],
            ]
        )
        model_names = ["A", "B", "C"]

        pairs = detector.get_similar_pairs(matrix, model_names)

        assert len(pairs) == 1
        assert pairs[0] == ("A", "B", 0.9)

    def test_no_similar_pairs(self):
        detector = ConsensusDetector(threshold=0.9)
        matrix = np.array(
            [
                [1.0, 0.5, 0.6],
                [0.5, 1.0, 0.7],
                [0.6, 0.7, 1.0],
            ]
        )
        model_names = ["A", "B", "C"]

        pairs = detector.get_similar_pairs(matrix, model_names)

        assert len(pairs) == 0
