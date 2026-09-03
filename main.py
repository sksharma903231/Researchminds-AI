import logging
from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import AnswerGenerator

# Suppress overly verbose HTTP logs from third-party libraries for cleaner terminal output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_rag(query: str, mode: str = "local"):
    print(f"\n{'='*50}")
    print(f"Executing RAG Pipeline | Model: {mode.upper()}")
    print(f"Query: {query}")
    print(f"{'='*50}\n")
    
    # 1. Initialize Systems
    logging.info("Booting Retrieval and Generation modules...")
    retriever = DocumentRetriever()
    generator = AnswerGenerator(mode=mode)
    
    # 2. Retrieve Context
    logging.info("Querying FAISS database for top 3 semantic matches...")
    retrieved_chunks = retriever.search(query, top_k=3)
    
    if not retrieved_chunks:
        logging.error("Database returned zero chunks. Aborting generation.")
        return
        
    # 3. Format Context for the LLM
    logging.info("Formatting retrieved chunks into context payload...")
    context_blocks = []
    for rank, chunk in enumerate(retrieved_chunks):
        context_blocks.append(f"[Source ID: {chunk['paper_id']} - Match Score: {chunk['score']:.4f}]\n{chunk['text']}")
    
    full_context = "\n\n".join(context_blocks)
    
    # 4. Generate Final Answer
    logging.info("Synthesizing final answer. This may take a moment on local hardware...")
    final_answer = generator.generate(context=full_context, question=query)
    
    print("\n" + "="*50)
    print("FINAL SYNTHESIZED ANSWER:")
    print("="*50)
    print(final_answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    # The actual question targeting your ingested research papers
    test_query = "What is GraphRAG?"
    
    # Executing the end-to-end pipeline
    run_rag(query=test_query, mode="local")