import streamlit as st
import logging
from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import AnswerGenerator

# Suppress verbose backend logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

st.set_page_config(page_title="ResearchMinds AI", layout="wide")
st.title("📚 ResearchMinds RAG Assistant (Public V1)")

@st.cache_resource
def load_systems():
    retriever = DocumentRetriever()
    generator = AnswerGenerator(mode="cloud") 
    return retriever, generator

try:
    retriever, generator = load_systems()
except ValueError:
    st.error("Deployment Configuration Error: The server is missing the required API keys. Check Streamlit Secrets.")
    st.stop()
except Exception as e:
    st.error(f"System Boot Failure: {e}")
    st.stop()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
if prompt := st.chat_input("Ask a question about the ingested research..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating response via Groq..."):
            retrieved_chunks = retriever.search(prompt, top_k=3)
            
            if not retrieved_chunks:
                st.warning("No relevant context found in the database.")
            else:
                context_blocks = [f"[Source: {c['paper_id']}]\n{c['text']}" for c in retrieved_chunks]
                full_context = "\n\n".join(context_blocks)
                
                answer = generator.generate(context=full_context, question=prompt)
                st.markdown(answer)
                
                with st.expander("View Retrieved Database Chunks"):
                    for block in context_blocks:
                        st.text(block)
                        st.divider()
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})