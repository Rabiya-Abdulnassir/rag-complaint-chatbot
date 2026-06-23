import streamlit as st
from src.rag_pipeline import ask_question
#from rag_pipeline import ask_question  # import your existing RAG function

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CrediTrust RAG Chatbot",
    layout="centered"
)

st.title("💬 CrediTrust Complaint Assistant (RAG)")
st.write("Ask questions about banking complaints and issues.")

# ============================================================
# SESSION STATE (to store chat history)
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# USER INPUT
# ============================================================

question = st.text_input("Enter your question:")

col1, col2 = st.columns(2)

ask_btn = col1.button("Ask")
clear_btn = col2.button("Clear Chat")

# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_btn:
    st.session_state.chat_history = []
    st.rerun()

# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_btn and question:

    with st.spinner("Thinking..."):

        result = ask_question(question)

        answer = result["answer"]
        sources = result["sources"]

        st.session_state.chat_history.append(
            (question, answer, sources)
        )

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for q, a, s in reversed(st.session_state.chat_history):

    st.markdown("### ❓ Question")
    st.write(q)

    st.markdown("### 🧠 Answer")
    st.write(a)

    # ========================================================
    # SOURCES (KEY REQUIREMENT)
    # ========================================================

    st.markdown("### 📚 Sources Used")

    if s:
        for i, src in enumerate(s):
            st.markdown(f"**Source {i+1}:**")
            st.write(src["text"])
            st.markdown("---")
    else:
        st.write("No sources found.")

    st.markdown("---")