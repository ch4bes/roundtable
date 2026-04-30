import httpx
import numpy as np
from typing import List
from dataclasses import dataclass
from .ollama_client import OllamaClient, EmbeddingResponse


@dataclass
class SimilarityResult:
    matrix: np.ndarray
    model_names: List[str]


class SimilarityEngine:
    def __init__(
        self,
        ollama_client: OllamaClient,
        embedding_model: str = "nomic-embed-text",
        use_embeddings: bool = True,
    ):
        self.ollama = ollama_client
        self.embedding_model = embedding_model
        self.use_embeddings = use_embeddings
        self._cache: dict[str, list[float]] = {}

    async def get_embedding(self, text: str) -> list[float]:
        if not self.use_embeddings:
            raise RuntimeError("Embeddings are disabled")

        cache_key = f"{self.embedding_model}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        response: EmbeddingResponse = await self.ollama.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        embedding = response.embedding
        self._cache[cache_key] = embedding
        return embedding

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    async def calculate_similarity_matrix(
        self, texts: List[str], model_names: List[str]
    ) -> SimilarityResult:
        n = len(texts)
        if n == 0:
            return SimilarityResult(matrix=np.array([]), model_names=[])

        if self.use_embeddings:
            try:
                embeddings = []
                for text in texts:
                    embedding = await self.get_embedding(text)
                    embeddings.append(embedding)

                matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            matrix[i, j] = 1.0
                        elif i < j:
                            sim = self.cosine_similarity(embeddings[i], embeddings[j])
                            matrix[i, j] = sim
                            matrix[j, i] = sim

                return SimilarityResult(matrix=matrix, model_names=model_names)
            except (httpx.HTTPError, RuntimeError) as e:
                print(f"Warning: Embedding generation failed ({e}), falling back to text-based similarity")
                self.use_embeddings = False

        return await self._calculate_text_similarity_matrix(texts, model_names)

    async def _calculate_text_similarity_matrix(
        self, texts: List[str], model_names: List[str]
    ) -> SimilarityResult:
        n = len(texts)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    sim = self._text_similarity(texts[i], texts[j])
                    matrix[i, j] = sim
                    matrix[j, i] = sim

        return SimilarityResult(matrix=matrix, model_names=model_names)

    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    async def calculate_pairwise_similarities(
        self, texts: List[str]
    ) -> list[tuple[int, int, float]]:
        n = len(texts)
        if n < 2:
            return []

        if self.use_embeddings:
            try:
                embeddings = []
                for text in texts:
                    embedding = await self.get_embedding(text)
                    embeddings.append(embedding)

                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = self.cosine_similarity(embeddings[i], embeddings[j])
                        pairs.append((i, j, sim))

                return pairs
            except (httpx.HTTPError, RuntimeError) as e:
                print(f"Warning: Pairwise embedding failed ({e}), falling back to text-based similarity")
                self.use_embeddings = False

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._text_similarity(texts[i], texts[j])
                pairs.append((i, j, sim))

        return pairs

    def clear_cache(self):
        self._cache.clear()
