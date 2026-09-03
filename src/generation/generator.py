import os
import logging
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnswerGenerator:
    def __init__(self, mode: str = "cloud"):
        """
        Initializes the LLM based on the user's selected mode.
        mode: 'cloud' (Groq/API) or 'local' (Ollama)
        """
        self.mode = mode
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert research assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I do not have enough information."
            
Context:
{context}

Question: {question}
Answer:"""
        )
        
        if self.mode == "cloud":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is missing.")
            logging.info("Initializing Cloud API (Groq)...")
            self.llm = ChatGroq(temperature=0.1, model_name="openai/gpt-oss-20b")
            
        elif self.mode == "local":
            logging.warning("Initializing Local LLM via Ollama. Monitor your VRAM usage closely.")
            # Updated to the modern LangChain Ollama integration
            self.llm = OllamaLLM(model="llama3", temperature=0.1)
            
        else:
            raise ValueError("Invalid mode. Choose 'cloud' or 'local'.")

    def generate(self, context: str, question: str) -> str:
        prompt = self.prompt_template.format(context=context, question=question)
        logging.info(f"Generating answer using {self.mode} model...")
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

if __name__ == "__main__":
    # Test Local Mode
    generator = AnswerGenerator(mode="local")
    
    mock_context = "GraphRAG is a novel approach by Microsoft that uses knowledge graphs."
    mock_query = "Who developed GraphRAG?"
    
    response = generator.generate(mock_context, mock_query)
    print(f"\nResponse: {response}")