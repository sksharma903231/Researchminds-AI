import streamlit as st

from src.generation.generator import AnswerGenerator
from src.retrieval.retriever import DocumentRetriever


@st.cache_resource
def load_systems(mode: str) -> tuple[DocumentRetriever, AnswerGenerator]:
    """Load and cache the retriever and selected generator."""
    return DocumentRetriever(), AnswerGenerator(mode=mode)


def render_chat(mode: str, prompt_placeholder: str) -> None:
    """Render the shared Streamlit chat interface."""
    try:
        retriever, generator = load_systems(mode)
    except ValueError as error:
        st.error(f"Deployment configuration error: {error}")
        st.stop()
    except Exception as error:
        st.error(f"System boot failure: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(prompt_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching vectors and synthesizing an answer..."):
                retrieved_chunks = retriever.search(prompt, top_k=3)

                if not retrieved_chunks:
                    fallback_answer = (
                        "I do not have enough information in the provided research papers to answer this question."
                    )
                    st.info(fallback_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": fallback_answer}
                    )
                    return

                context_blocks = []
                for chunk in retrieved_chunks:
                    doc_title = chunk.get("title") or chunk.get("paper_id", "Unknown")
                    page_str = f"Page {chunk['page']}" if chunk.get("page") else "Page N/A"
                    score_val = chunk.get("score", 0.0)
                    context_blocks.append(
                        f"[Document: {doc_title} | {page_str} | ID: {chunk.get('paper_id')} | Score: {score_val:.4f}]\n{chunk['text']}"
                    )

                full_context = "\n\n".join(context_blocks)
                answer = generator.generate(context=full_context, question=prompt)
                st.markdown(answer)

                with st.expander("🔍 View Retrieved Sources & Evidence Chunks"):
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        doc_title = chunk.get("title") or chunk.get("paper_id", "Unknown")
                        page_str = f"Page {chunk['page']}" if chunk.get("page") else "Page N/A"
                        score_val = chunk.get("score", 0.0)
                        st.markdown(f"**Source {i}: {doc_title} ({page_str})** — *Score: {score_val:.4f}*")
                        st.text(chunk["text"])
                        st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

