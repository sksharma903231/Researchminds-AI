import logging
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

UNSUPPORTED_ANSWER_MESSAGE = (
    "I do not have enough information in the provided research papers to answer this question."
)

DEFAULT_PROMPT_TEMPLATE = """You are an expert research assistant. Answer the user's question using ONLY the provided research paper context below.
If the provided context does not contain enough information to answer the question factually and accurately, say: "I do not have enough information in the provided research papers to answer this question."
Do not speculate, make assumptions, or extrapolate beyond the provided context.
Whenever answering, cite the relevant paper title or ID and page number provided in the context.

Context:
{context}

Question: {question}
Answer:"""


class AnswerGenerator:
    """Generate answers from retrieved research-paper context."""

    def __init__(
        self,
        mode: str = "cloud",
        llm: object = None,
        model_name: str | None = None,
    ):
        self.mode = mode
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=DEFAULT_PROMPT_TEMPLATE,
        )

        if llm is not None:
            self.llm = llm
            return

        if self.mode == "cloud":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is missing.")
            resolved_model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info("Initializing Groq model '%s'.", resolved_model)
            self.llm = ChatGroq(
                groq_api_key=api_key,
                temperature=0.1,
                model_name=resolved_model,
            )

        elif self.mode == "local":
            resolved_model = model_name or os.getenv("OLLAMA_MODEL", "llama3")
            logger.info("Initializing local Ollama model '%s'.", resolved_model)
            self.llm = OllamaLLM(model=resolved_model, temperature=0.1)

        else:
            raise ValueError("Invalid mode. Choose 'cloud' or 'local'.")

    def generate(self, context: str, question: str) -> str:
        """Generate an answer strictly grounded in the provided context."""
        if not question or not question.strip():
            return "Please provide a valid research question."

        if not context or not context.strip():
            logger.info("No context provided; returning unsupported answer message.")
            return UNSUPPORTED_ANSWER_MESSAGE

        prompt = self.prompt_template.format(
            context=context.strip(),
            question=question.strip(),
        )
        logger.info("Generating answer with %s mode.", self.mode)

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer.strip()

