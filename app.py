import streamlit as st
from src.rag import ask_question
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="SatCom Intelligence Agent", page_icon="🛰️")

# --- SIDEBAR: Project Info & Metrics ---
with st.sidebar:
    st.title("🛰️ Project Info")
    st.markdown("""
    **SatCom-NGP RAG Agent**
    Validated with Gemma-3 and Ragas.
    """)
    
    st.divider()
    st.subheader("📊 Performance Metrics")
    st.metric(label="Faithfulness", value="1.00", help="Perfect grounding in PDF")
    st.metric(label="Context Recall", value="0.60", delta="-0.40", delta_color="inverse")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("SatCom Intelligence Explorer")
st.info("Ask questions about the India Satellite Communication Regulatory Policy.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the licensing authority for GMPCS?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(" *Searching policy documents...*")
        
        try:
            # Call your existing RAG logic
            answer, docs = ask_question(prompt)
            
            # Display the answer
            message_placeholder.markdown(answer)
            
            # Show Sources in an Expander
            with st.expander("View Cited Sources"):
                for i, doc in enumerate(docs):
                    page = doc.metadata.get("page", "Unknown")
                    st.write(f"**Source {i+1} (Page {page}):**")
                    st.caption(doc.page_content)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            message_placeholder.markdown("Sorry, I encountered an error processing your request.")