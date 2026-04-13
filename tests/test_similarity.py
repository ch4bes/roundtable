import pytest
import numpy as np
from core.similarity import SimilarityEngine
from core.ollama_client import OllamaClient, EmbeddingResponse


class MockOllamaClient:
    def __init__(self, embeddings_map: dict[str, list[float]]):
        self.embeddings_map = embeddings_map
        self.call_count = 0

    async def embeddings(self, model: str, prompt: str) -> EmbeddingResponse:
        self.call_count += 1
        if prompt in self.embeddings_map:
            return EmbeddingResponse(
                embedding=self.embeddings_map[prompt],
                model=model,
            )
        return EmbeddingResponse(
            embedding=[0.0] * 10,
            model=model,
        )


class TestCosineSimilarity:
    def test_identical_vectors(self):
        engine = SimilarityEngine.__new__(SimilarityEngine)
        engine.use_embeddings = True
        vec = [1.0, 2.0, 3.0]
        sim = engine.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        engine = SimilarityEngine.__new__(SimilarityEngine)
        engine.use_embeddings = True
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = engine.cosine_similarity(vec1, vec2)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        engine = SimilarityEngine.__new__(SimilarityEngine)
        engine.use_embeddings = True
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        sim = engine.cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_partial_similarity(self):
        engine = SimilarityEngine.__new__(SimilarityEngine)
        engine.use_embeddings = True
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 1.0, 0.0]
        sim = engine.cosine_similarity(vec1, vec2)
        assert 0.5 < sim < 0.8


@pytest.mark.asyncio
async def test_similarity_matrix():
    embeddings = {
        "text a": [1.0, 0.0, 0.0],
        "text b": [1.0, 0.1, 0.0],
        "text c": [0.0, 1.0, 0.0],
    }
    mock_client = MockOllamaClient(embeddings)
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = mock_client
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    texts = ["text a", "text b", "text c"]
    model_names = ["model_a", "model_b", "model_c"]

    result = await engine.calculate_similarity_matrix(texts, model_names)

    assert result.matrix.shape == (3, 3)
    assert result.model_names == model_names

    for i in range(3):
        assert abs(result.matrix[i, i] - 1.0) < 1e-6

    assert result.matrix[0, 1] > 0.9
    assert result.matrix[0, 2] < 0.1
    assert result.matrix[1, 2] < 0.1


@pytest.mark.asyncio
async def test_embedding_cache():
    embeddings = {
        "cached text": [1.0, 2.0, 3.0],
    }
    mock_client = MockOllamaClient(embeddings)
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = mock_client
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    await engine.get_embedding("cached text")
    await engine.get_embedding("cached text")

    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_pairwise_similarities():
    embeddings = {
        "text 1": [1.0, 0.0, 0.0],
        "text 2": [0.0, 1.0, 0.0],
        "text 3": [0.0, 0.0, 1.0],
    }
    mock_client = MockOllamaClient(embeddings)
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = mock_client
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    pairs = await engine.calculate_pairwise_similarities(["text 1", "text 2", "text 3"])

    assert len(pairs) == 3
    for i, j, sim in pairs:
        assert abs(sim) < 1e-6


def test_text_similarity_fallback():
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.use_embeddings = False

    text1 = "the quick brown fox jumps over the lazy dog"
    text2 = "the quick brown fox jumps over the lazy dog"
    text3 = "completely different words"

    sim1 = engine._text_similarity(text1, text2)
    sim2 = engine._text_similarity(text1, text3)

    assert sim1 == 1.0
    assert sim2 < 0.3
