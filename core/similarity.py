import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass

import httpx
import numpy as np

from .exceptions import DimensionMismatchError
from .llm_client import EmbeddingResponse, LLMClient

logger = logging.getLogger(__name__)

# Number of similarity calculations to wait in text-fallback mode before
# re-testing whether the embedding service has recovered.
_EMBEDDING_RETRY_AFTER: int = 5


@dataclass
class SimilarityResult:
    matrix: np.ndarray
    model_names: list[str]


class SimilarityEngine:
    def __init__(
        self,
        ollama_client: LLMClient,
        embedding_model: str = "nomic-embed-text",
        use_embeddings: bool = True,
        dimension: int | None = None,
    ):
        self.ollama = ollama_client
        self.embedding_model = embedding_model
        self.use_embeddings = use_embeddings
        self._dimension = dimension
        self._default_dimension = (
            1024  # Fallback if dimension not set and can't be auto-detected
        )
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = 100
        self._cache_order: list[str] = []  # FIFO ordering for eviction
        # Tracks calls made while use_embeddings is False so we can retry
        # periodically instead of staying in fallback mode permanently.
        self._fallback_call_count: int = 0

    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Returns a zero vector for empty text, uses caching when available,
        and gracefully falls back to a zero vector if the Ollama API fails.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the text embedding.

        Raises:
            RuntimeError: If embeddings have been disabled after a failure.
        """
        if not self.use_embeddings:
            raise RuntimeError("Embeddings are disabled")

        # Handle empty text - return a zero vector to avoid shape mismatches
        if not text or not text.strip():
            dim = self._dimension if self._dimension else self._default_dimension
            logger.warning(
                "Empty text provided for embedding, using zero vector (dim=%d)", dim
            )
            return [0.0] * dim

        # Initialize cache attributes if not present (handles __new__ bypass in tests)
        if not hasattr(self, "_cache_order"):
            self._cache_order = []
        if not hasattr(self, "_cache"):
            self._cache = {}
        if not hasattr(self, "_max_cache_size"):
            self._max_cache_size = 100

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"{self.embedding_model}:{text_hash}"
        if cache_key in self._cache:
            self._move_to_end(cache_key)
            return self._cache[cache_key]

        response: EmbeddingResponse = await self.ollama.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        embedding = response.embedding

        # Handle empty embedding response
        if not embedding or len(embedding) == 0:
            dim = self._dimension if self._dimension else self._default_dimension
            logger.warning("Empty embedding returned, using zero vector (dim=%d)", dim)
            embedding = [0.0] * dim

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
        """Compute cosine similarity between two vectors.

        Raises DimensionMismatchError if vectors have different lengths.
        Returns 0.0 for empty inputs.
        """
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) != len(vec2):
            raise DimensionMismatchError(len(vec1), len(vec2))

        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    async def calculate_similarity_matrix(
        self, texts: list[str], model_names: list[str]
    ) -> SimilarityResult:
        # NOTE: Sequential embedding calls. Ollama's /api/embeddings does not support
        # batching. If Ollama adds a batch endpoint, replace with asyncio.gather() for
        # parallel embedding retrieval:
        #   embeddings = await asyncio.gather(*[self.get_embedding(t) for t in texts])
        n = len(texts)
        if n == 0:
            return SimilarityResult(matrix=np.array([]), model_names=[])

        # Filter out empty texts and track which models have empty content
        valid_indices = []
        valid_texts = []
        valid_names = []
        empty_models = []
        for i, (text, name) in enumerate(zip(texts, model_names)):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
                valid_names.append(name)
            else:
                empty_models.append(name)

        if empty_models:
            logger.warning(
                "%d model(s) returned empty content: %s — using zero similarity",
                len(empty_models),
                ", ".join(empty_models),
            )

        # If all texts are empty, return a matrix with zeros
        if not valid_texts:
            return SimilarityResult(matrix=np.zeros((n, n)), model_names=model_names)

        # Periodically re-test the embedding service when in fallback mode
        # so a transient outage does not disable embeddings for the entire session.
        if not self.use_embeddings:
            self._fallback_call_count += 1
            if self._fallback_call_count >= _EMBEDDING_RETRY_AFTER:
                logger.info(
                    "Re-testing embedding service after %d fallback calls ...",
                    self._fallback_call_count,
                )
                self.use_embeddings = True
                self._fallback_call_count = 0

        if self.use_embeddings:
            try:
                embeddings = await asyncio.gather(
                    *[self.get_embedding(text) for text in valid_texts]
                )
                # Build full matrix with zero vectors for empty texts
                full_embeddings = []
                emb_idx = 0
                for i in range(n):
                    if i in valid_indices:
                        full_embeddings.append(embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        # Use zero vector for empty text
                        dim = (
                            self._dimension
                            if self._dimension
                            else self._default_dimension
                        )
                        full_embeddings.append([0.0] * dim)
                return await self._build_similarity_matrix(
                    texts, model_names, full_embeddings
                )
            except (httpx.HTTPError, RuntimeError) as e:
                logger.warning(
                    "Embedding generation failed (%s); falling back to text-based similarity",
                    e,
                )
                self.use_embeddings = False
                self._fallback_call_count = 0

        return await self._build_similarity_matrix(texts, model_names, None)

    async def _build_similarity_matrix(
        self,
        texts: list[str],
        model_names: list[str],
        embeddings: list[list[float]] | None,
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
                    try:
                        sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    except DimensionMismatchError:
                        sim = self._text_similarity(texts[i], texts[j])
                else:
                    sim = self._text_similarity(texts[i], texts[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return SimilarityResult(matrix=matrix, model_names=model_names)

    async def _calculate_text_similarity_matrix(
        self, texts: list[str], model_names: list[str]
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
        self, texts: list[str]
    ) -> list[tuple[int, int, float]]:
        n = len(texts)
        if n < 2:
            return []

        if self.use_embeddings:
            try:
                embeddings = await asyncio.gather(
                    *[self.get_embedding(text) for text in texts]
                )

                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        try:
                            sim = self.cosine_similarity(embeddings[i], embeddings[j])
                        except DimensionMismatchError:
                            sim = self._text_similarity(texts[i], texts[j])
                        pairs.append((i, j, sim))

                return pairs
            except (httpx.HTTPError, RuntimeError) as e:
                logger.warning(
                    "Pairwise embedding failed (%s); falling back to text-based similarity",
                    e,
                )
                self.use_embeddings = False

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._text_similarity(texts[i], texts[j])
                pairs.append((i, j, sim))

        return pairs

    def clear_cache(self) -> None:
        # Initialize cache attributes if not present (handles __new__ bypass in tests)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if not hasattr(self, "_cache_order"):
            self._cache_order = []
        self._cache.clear()
        self._cache_order.clear()
