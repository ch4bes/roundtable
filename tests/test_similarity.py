import httpx
import pytest

from core.exceptions import DimensionMismatchError
from core.ollama_client import EmbeddingResponse
from core.similarity import SimilarityEngine


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

    def test_cosine_similarity_raises_on_dimension_mismatch(self):
        engine = SimilarityEngine.__new__(SimilarityEngine)
        engine.use_embeddings = True
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        with pytest.raises(DimensionMismatchError) as exc_info:
            engine.cosine_similarity(vec1, vec2)
        assert exc_info.value.dim1 == 2
        assert exc_info.value.dim2 == 3


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


# ── Fallback-on-embedding-failure tests (§1.9) ──────────────────────────


class MockOllamaClientFailing:
    """Mock client that raises on every embeddings() call."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.call_count = 0

    async def embeddings(self, model: str, prompt: str) -> EmbeddingResponse:
        self.call_count += 1
        raise self.exc


@pytest.mark.asyncio
async def test_similarity_fallback_logs_error_message(caplog):
    """When embedding fails, the warning is logged, not silently swallowed."""
    import logging

    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = MockOllamaClientFailing(httpx.HTTPError("Connection refused"))
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    with caplog.at_level(logging.WARNING, logger="core.similarity"):
        result = await engine.calculate_similarity_matrix(
            ["text1", "text2", "text3"], ["m1", "m2", "m3"]
        )

    assert not engine.use_embeddings  # fallback triggered
    assert result.matrix.shape == (3, 3)
    assert any("Connection refused" in r.message for r in caplog.records)
    assert any("falling back" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_pairwise_similarity_fallback_logs_error(caplog):
    """When pairwise embedding fails, the warning is logged."""
    import logging

    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = MockOllamaClientFailing(httpx.HTTPError("Timeout"))
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    with caplog.at_level(logging.WARNING, logger="core.similarity"):
        await engine.calculate_pairwise_similarities(["text1", "text2"])

    assert not engine.use_embeddings
    assert any("Timeout" in r.message for r in caplog.records)
    assert any("falling back" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_similarity_fallback_uses_jaccard_after_http_error():
    """After embedding fails, Jaccard similarity is used for the matrix."""
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = MockOllamaClientFailing(httpx.HTTPError("500"))
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    result = await engine.calculate_similarity_matrix(
        ["hello world", "world hello"], ["m1", "m2"]
    )

    # Jaccard of {"hello", "world"} and {"world", "hello"} = 1.0
    assert result.matrix[0, 1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_similarity_embeddings_disabled_after_failure():
    """use_embeddings stays False after one failure for the lifetime of the engine."""
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = MockOllamaClientFailing(httpx.HTTPError("fail"))
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    await engine.calculate_similarity_matrix(["x"], ["m1"])
    assert not engine.use_embeddings

    # Subsequent calls should NOT try embeddings at all:
    engine.ollama = MockOllamaClientFailing(httpx.HTTPError("should not be called"))
    await engine.calculate_similarity_matrix(["y"], ["m2"])
    assert engine.ollama.call_count == 0


@pytest.mark.asyncio
async def test_similarity_matrix_fallback_on_mismatched_embeddings():
    """The similarity matrix should fall back to text similarity on dimension mismatch."""
    embeddings = {
        "text a": [1.0, 0.0],
        "text b": [1.0, 0.0, 0.0],  # Mismatch!
    }
    mock_client = MockOllamaClient(embeddings)
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = mock_client
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    # Using a a and b that are exactly identical text for easy Jaccard check
    texts = ["same text", "same text"]
    embeddings_mismatched = {"same text": [1.0, 0.0]}
    # We need different embeddings for different texts to test mismatch.
    # But the MockOllamaClient's embeddings() uses prompt as key.
    # So we need different prompts that return different lengths.

    texts = ["text a", "text b"]
    model_names = ["m1", "m2"]

    # we already defined 'embeddings' dictionary at top of test
    result = await engine.calculate_similarity_matrix(texts, model_names)

    # text a and text b are different, so Jaccard will be < 1.0
    # But they should have SOME similarity if we use words that overlap.
    # Let's redefine for a clear Jaccard result.

    embeddings_mismatched = {
        "hello world": [1.0, 0.0],
        "hello universe": [1.0, 0.0, 0.0],
    }
    mock_client.embeddings_map = embeddings_mismatched

    result = await engine.calculate_similarity_matrix(
        ["hello world", "hello universe"], model_names
    )

    # Jaccard of {"hello", "world"} and {"hello", "universe"} = 1 / 3 = 0.333
    assert result.matrix[0, 1] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_pairwise_fallback_on_mismatched_embeddings():
    """Pairwise similarity should fall back to text similarity on dimension mismatch."""
    embeddings_mismatched = {
        "hello world": [1.0, 0.0],
        "hello universe": [1.0, 0.0, 0.0],
    }
    mock_client = MockOllamaClient(embeddings_mismatched)
    engine = SimilarityEngine.__new__(SimilarityEngine)
    engine.ollama = mock_client
    engine.embedding_model = "test-model"
    engine.use_embeddings = True
    engine._cache = {}

    pairs = await engine.calculate_pairwise_similarities(
        ["hello world", "hello universe"]
    )

    # One pair, similarity should be Jaccard 1/3
    assert len(pairs) == 1
    assert pairs[0][2] == pytest.approx(1 / 3)
