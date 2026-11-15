from unittest.mock import MagicMock, patch

from main import run_rag


def test_run_rag_success():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "paper_id": "p_test",
            "title": "Test Paper",
            "page": 1,
            "score": 0.95,
            "text": "This is test evidence text.",
        }
    ]
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Synthesized test answer."

    with patch("main.DocumentRetriever", return_value=mock_retriever), \
         patch("main.AnswerGenerator", return_value=mock_generator):
        answer = run_rag(query="What is testing?", mode="local")
        assert answer == "Synthesized test answer."
        mock_retriever.search.assert_called_once_with("What is testing?", top_k=3)
        mock_generator.generate.assert_called_once()


def test_run_rag_no_chunks_unsupported():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with patch("main.DocumentRetriever", return_value=mock_retriever):
        answer = run_rag(query="Unknown topic?", mode="local")
        assert "not have enough information" in answer
