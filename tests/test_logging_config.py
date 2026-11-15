import logging
from src.logging_config import configure_logging


def test_configure_logging():
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("sentence_transformers").level == logging.WARNING
