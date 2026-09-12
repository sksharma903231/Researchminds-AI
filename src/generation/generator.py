import logging
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generate answers from retrieved research-paper context."""

    def __init__(self, mode: str = "cloud"):
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
            logger.info("Initializing Groq model.")
            self.llm = ChatGroq(
                groq_api_key=api_key,
                temperature=0.1,
                model_name="openai/gpt-oss-20b",
            )
            
        elif self.mode == "local":
            logger.info("Initializing local Ollama model.")
            self.llm = OllamaLLM(model="llama3", temperature=0.1)
            
        else:
            raise ValueError("Invalid mode. Choose 'cloud' or 'local'.")

    def generate(self, context: str, question: str) -> str:
        prompt = self.prompt_template.format(context=context, question=question)
        logger.info("Generating answer with %s mode.", self.mode)

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response
