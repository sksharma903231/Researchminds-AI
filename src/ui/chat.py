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
                    st.warning("No relevant context found in the database.")
                    return

                context_blocks = [
                    f"[Source: {chunk['paper_id']}]\n{chunk['text']}"
                    for chunk in retrieved_chunks
                ]
                answer = generator.generate(
                    context="\n\n".join(context_blocks), question=prompt
                )
                st.markdown(answer)

                with st.expander("View retrieved database chunks"):
                    for block in context_blocks:
                        st.text(block)
                        st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
