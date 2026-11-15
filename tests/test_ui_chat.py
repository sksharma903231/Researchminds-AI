from unittest.mock import MagicMock, patch

from src.ui.chat import load_systems


def test_load_systems():
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    with patch("src.ui.chat.DocumentRetriever", return_value=mock_retriever), \
         patch("src.ui.chat.AnswerGenerator", return_value=mock_generator):
        # Call the underlying function if decorated or normal
        fn = getattr(load_systems, "__wrapped__", load_systems)
        ret, gen = fn("local")
        assert ret == mock_retriever
        assert gen == mock_generator
