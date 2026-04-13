import numpy as np
from typing import Literal
from dataclasses import dataclass


@dataclass
class ConsensusResult:
    reached: bool
    percentage: float
    agreeing_pairs: int
    total_pairs: int
    method: Literal["pairwise", "clustering"]
    details: dict | None = None


class ConsensusDetector:
    def __init__(
        self, threshold: float = 0.85, method: Literal["pairwise", "clustering"] = "pairwise"
    ):
        self.threshold = threshold
        self.method = method

    def detect(self, similarity_matrix: np.ndarray) -> ConsensusResult:
        if self.method == "pairwise":
            return self._detect_pairwise(similarity_matrix)
        else:
            return self._detect_clustering(similarity_matrix)

    def _detect_pairwise(self, similarity_matrix: np.ndarray) -> ConsensusResult:
        n = similarity_matrix.shape[0]
        if n < 2:
            return ConsensusResult(
                reached=True,
                percentage=100.0,
                agreeing_pairs=0,
                total_pairs=0,
                method="pairwise",
            )

        upper_tri_indices = np.triu_indices(n, k=1)
        pairs = similarity_matrix[upper_tri_indices]

        if len(pairs) == 0:
            return ConsensusResult(
                reached=True,
                percentage=100.0,
                agreeing_pairs=0,
                total_pairs=0,
                method="pairwise",
            )

        agreeing_pairs = int(np.sum(pairs >= self.threshold))
        total_pairs = len(pairs)
        percentage = (agreeing_pairs / total_pairs) * 100
        reached = agreeing_pairs == total_pairs

        return ConsensusResult(
            reached=reached,
            percentage=percentage,
            agreeing_pairs=agreeing_pairs,
            total_pairs=total_pairs,
            method="pairwise",
            details={
                "pair_values": pairs.tolist(),
                "threshold": self.threshold,
            },
        )

    def _detect_clustering(self, similarity_matrix: np.ndarray) -> ConsensusResult:
        n = similarity_matrix.shape[0]
        if n < 2:
            return ConsensusResult(
                reached=True,
                percentage=100.0,
                agreeing_pairs=0,
                total_pairs=0,
                method="clustering",
            )

        clusters = self._cluster_responses(similarity_matrix)
        if not clusters:
            return ConsensusResult(
                reached=False,
                percentage=0.0,
                agreeing_pairs=0,
                total_pairs=0,
                method="clustering",
            )

        largest_cluster_size = max(len(c) for c in clusters)
        percentage = (largest_cluster_size / n) * 100
        reached = largest_cluster_size > (n / 2)

        return ConsensusResult(
            reached=reached,
            percentage=percentage,
            agreeing_pairs=largest_cluster_size,
            total_pairs=n,
            method="clustering",
            details={
                "clusters": [len(c) for c in clusters],
                "largest_cluster": largest_cluster_size,
            },
        )

    def _cluster_responses(self, similarity_matrix: np.ndarray) -> list[list[int]]:
        n = similarity_matrix.shape[0]
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i, j] >= self.threshold:
                        cluster.append(j)
                        visited[j] = True
                clusters.append(cluster)

        unvisited = [i for i in range(n) if not visited[i]]
        for i in unvisited:
            clusters.append([i])

        return clusters

    def get_similar_pairs(
        self, similarity_matrix: np.ndarray, model_names: list[str]
    ) -> list[tuple[str, str, float]]:
        n = similarity_matrix.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]
                if sim >= self.threshold:
                    pairs.append((model_names[i], model_names[j], sim))
        return pairs
