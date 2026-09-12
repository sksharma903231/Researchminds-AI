import streamlit as st

from src.logging_config import configure_logging
from src.ui.chat import render_chat


configure_logging()
st.set_page_config(page_title="ResearchMinds AI", layout="wide")
st.title("📚 ResearchMinds RAG Assistant")

render_chat("cloud", "Ask a question about the ingested research...")
