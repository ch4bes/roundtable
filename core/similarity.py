import httpx
import re
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
        self._max_cache_size = 100
        self._cache_order: list[str] = []  # FIFO ordering for eviction

    async def get_embedding(self, text: str) -> list[float]:
        if not self.use_embeddings:
            raise RuntimeError("Embeddings are disabled")

        # Initialize cache attributes if not present (handles __new__ bypass in tests)
        if not hasattr(self, '_cache_order'):
            self._cache_order = []
        if not hasattr(self, '_cache'):
            self._cache = {}
        if not hasattr(self, '_max_cache_size'):
            self._max_cache_size = 100

        cache_key = f"{self.embedding_model}:{text}"
        if cache_key in self._cache:
            self._move_to_end(cache_key)
            return self._cache[cache_key]

        response: EmbeddingResponse = await self.ollama.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        embedding = response.embedding
        self._cache[cache_key] = embedding
        self._cache_order.append(cache_key)
        self._evict_if_needed()
        return embedding

    def _move_to_end(self, key: str) -> None:
        """Move a key to the end of the cache order (most recently used)."""
        if key in self._cache_order:
            self._cache_order.remove(key)
            self._cache_order.append(key)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        while len(self._cache) > self._max_cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

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
        # NOTE: Sequential embedding calls. Ollama's /api/embeddings does not support
        # batching. If Ollama adds a batch endpoint, replace with asyncio.gather() for
        # parallel embedding retrieval:
        #   embeddings = await asyncio.gather(*[self.get_embedding(t) for t in texts])
        n = len(texts)
        if n == 0:
            return SimilarityResult(matrix=np.array([]), model_names=[])

        if self.use_embeddings:
            try:
                embeddings = [await self.get_embedding(text) for text in texts]
                return await self._build_similarity_matrix(texts, model_names, embeddings)
            except (httpx.HTTPError, RuntimeError) as e:
                print(f"Warning: Embedding generation failed ({e}), falling back to text-based similarity")
                self.use_embeddings = False

        return await self._build_similarity_matrix(texts, model_names, None)

    async def _build_similarity_matrix(
        self,
        texts: List[str],
        model_names: List[str],
        embeddings: List[list[float]] | None,
    ) -> SimilarityResult:
        """
        Build a full NxN symmetric similarity matrix.
        
        If embeddings are provided, uses cosine similarity. Otherwise falls back to
        Jaccard text similarity.
        """
        n = len(texts)
        matrix = np.zeros((n, n))

        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                if embeddings is not None:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                else:
                    sim = self._text_similarity(texts[i], texts[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return SimilarityResult(matrix=matrix, model_names=model_names)

    async def _calculate_text_similarity_matrix(
        self, texts: List[str], model_names: List[str]
    ) -> SimilarityResult:
        """Delegate to _build_similarity_matrix with no embeddings (text fallback)."""
        return await self._build_similarity_matrix(texts, model_names, None)

    def _text_similarity(self, text1: str, text2: str) -> float:
        # Use re.findall(r"\w+", ...) for better tokenization:
        # - Strips punctuation: "hello," -> "hello"
        # - Splits hyphenated: "state-of-the-art" -> ["state", "of", "the", "art"]
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

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
        # Initialize cache attributes if not present (handles __new__ bypass in tests)
        if not hasattr(self, '_cache'):
            self._cache = {}
        if not hasattr(self, '_cache_order'):
            self._cache_order = []
        self._cache.clear()
        self._cache_order.clear()
