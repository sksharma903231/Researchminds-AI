import logging

from src.generation.generator import AnswerGenerator
from src.logging_config import configure_logging
from src.retrieval.retriever import DocumentRetriever

configure_logging()
logger = logging.getLogger(__name__)


def run_rag(query: str, mode: str = "local") -> None:
    """Retrieve relevant paper chunks and print a generated answer."""
    print(f"\n{'='*50}")
    print(f"Executing RAG Pipeline | Model: {mode.upper()}")
    print(f"Query: {query}")
    print(f"{'='*50}\n")

    logger.info("Initializing retrieval and generation modules.")
    retriever = DocumentRetriever()
    generator = AnswerGenerator(mode=mode)

    logger.info("Retrieving the top three semantic matches.")
    retrieved_chunks = retriever.search(query, top_k=3)

    if not retrieved_chunks:
        logger.error("No chunks were returned; answer generation skipped.")
        return

    logger.info("Formatting retrieved chunks for generation.")
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Source ID: {chunk['paper_id']} - "
            f"Match Score: {chunk['score']:.4f}]\n{chunk['text']}"
        )

    full_context = "\n\n".join(context_blocks)

    logger.info("Generating answer.")
    final_answer = generator.generate(context=full_context, question=query)

    print("\n" + "=" * 50)
    print("FINAL SYNTHESIZED ANSWER:")
    print("=" * 50)
    print(final_answer)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_query = "What is GraphRAG?"
    run_rag(query=test_query, mode="local")
