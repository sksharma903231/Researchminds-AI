import streamlit as st

from src.logging_config import configure_logging
from src.ui.chat import render_chat


configure_logging()
st.set_page_config(page_title="ResearchMinds AI", layout="wide")
st.title("📚 ResearchMinds RAG Assistant")

st.sidebar.header("System configuration")
selected_mode = st.sidebar.radio("LLM engine", ["local", "cloud"])

render_chat(selected_mode, "Ask a question about your research papers...")
