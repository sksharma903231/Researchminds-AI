import streamlit as st
import logging
from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import AnswerGenerator

# Suppress verbose backend logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure the web page
st.set_page_config(page_title="ResearchMinds AI", layout="wide")
st.title("📚 ResearchMinds RAG Assistant")

# Sidebar Configuration
st.sidebar.header("System Configuration")
selected_mode = st.sidebar.radio("LLM Engine", ["local", "cloud"])

# Cache the heavy models so they don't reload every time you click a button
@st.cache_resource
def load_systems(mode):
    retriever = DocumentRetriever()
    generator = AnswerGenerator(mode=mode)
    return retriever, generator

try:
    retriever, generator = load_systems(selected_mode)
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
if prompt := st.chat_input("Ask a question about your research papers..."):
    
    # 1. Display User Question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Process and Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Searching vectors and synthesizing answer..."):
            
            # Retrieve
            retrieved_chunks = retriever.search(prompt, top_k=3)
            
            if not retrieved_chunks:
                st.warning("No relevant context found in the database.")
            else:
                # Format context
                context_blocks = [f"[Source: {c['paper_id']}]\n{c['text']}" for c in retrieved_chunks]
                full_context = "\n\n".join(context_blocks)
                
                # Generate
                answer = generator.generate(context=full_context, question=prompt)
                
                # Render Answer
                st.markdown(answer)
                
                # Render Citations inside a dropdown toggle
                with st.expander("View Retrieved Database Chunks"):
                    for block in context_blocks:
                        st.text(block)
                        st.divider()
                        
                # Save to memory
                st.session_state.messages.append({"role": "assistant", "content": answer})