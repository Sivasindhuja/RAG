import streamlit as st
from rag import ask_question


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Research Intelligence Agent",
    page_icon="🤖"
)


# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🤖 AI Research RAG")

    st.markdown("""
    Research-paper RAG system for:

    - Transformers
    - LoRA
    - RAG
    - RLHF
    - DistilBERT
    - Switch Transformers
    - LLM Agents
    """)

    st.divider()

    st.subheader("📊 System")

    st.success("Hybrid Retrieval")

    st.success("Cohere Reranking")

    st.success("Hierarchical Retrieval")

    st.success("Grounded Generation")

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------- MAIN UI ---------------- #

st.title("AI Research Intelligence Agent")

st.info(
    "Ask questions about transformers, RAG, LoRA, RLHF, LLM agents, and modern AI architectures."
)


# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- USER INPUT ---------------- #

if prompt := st.chat_input("How does LoRA reduce trainable parameters?"):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        message_placeholder.markdown(
            "🔍 Searching research papers..."
        )

        try:

            answer, docs = ask_question(prompt)

            message_placeholder.markdown(answer)

            with st.expander("📚 View Retrieved Sources"):

                for i, doc in enumerate(docs):

                    paper = doc.metadata.get("paper", "Unknown")

                    page = doc.metadata.get("page", "Unknown")

                    st.write(f"### {paper} — Page {page}")

                    st.caption(doc.page_content[:1500])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(f"Error: {e}")

            message_placeholder.markdown(
                "Sorry, something went wrong."
            )

