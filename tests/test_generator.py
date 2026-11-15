import pytest

from src.generation.generator import AnswerGenerator, UNSUPPORTED_ANSWER_MESSAGE


def test_generator_with_mock_llm(mock_llm):
    generator = AnswerGenerator(mode="cloud", llm=mock_llm)
    context = "[Document: LightRAG, Page 1]\nLightRAG incorporates dual-level graph structures."
    question = "How does LightRAG work?"

    answer = generator.generate(context=context, question=question)
    assert answer == "Test answer grounded in context."
    assert mock_llm.last_prompt is not None
    assert "LightRAG incorporates dual-level graph structures." in mock_llm.last_prompt


def test_generator_unsupported_question_on_empty_context(mock_llm):
    generator = AnswerGenerator(mode="cloud", llm=mock_llm)

    # When context is empty
    answer = generator.generate(context="", question="What is the capital of Mars?")
    assert answer == UNSUPPORTED_ANSWER_MESSAGE

    # When context is whitespace
    answer_whitespace = generator.generate(context="   \n  ", question="What is the recipe for cake?")
    assert answer_whitespace == UNSUPPORTED_ANSWER_MESSAGE


def test_generator_empty_question(mock_llm):
    generator = AnswerGenerator(mode="cloud", llm=mock_llm)
    answer = generator.generate(context="Some context", question="")
    assert "Please provide a valid" in answer


def test_generator_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        AnswerGenerator(mode="cloud")


def test_generator_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        AnswerGenerator(mode="invalid_mode")
