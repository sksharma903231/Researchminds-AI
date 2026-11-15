import numpy as np
import pytest


class MockEmbeddingModel:
    """Lightweight deterministic mock embedding model for testing without downloading weights."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            # Deterministic vector based on hash of text
            seed = sum(ord(c) for c in text) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class MockLLMResponse:
    """Mock LLM response object mimicking LangChain message output."""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


class MockLLM:
    """Mock LLM that returns a preset answer or echoes context evidence."""

    def __init__(self, fixed_answer: str = "Test answer grounded in context."):
        self.fixed_answer = fixed_answer
        self.last_prompt = None

    def invoke(self, prompt: str) -> MockLLMResponse:
        self.last_prompt = prompt
        return MockLLMResponse(self.fixed_answer)


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel(dimension=16)


@pytest.fixture
def mock_llm():
    return MockLLM()
