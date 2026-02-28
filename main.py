# main.py
import streamlit as st
from utils.db import ensure_cache_indexes, ensure_questions_indexes

st.set_page_config(page_title="RAG Quiz Generator", page_icon="🎓")

if "indexes_ensured" not in st.session_state:
    ensure_cache_indexes()
    ensure_questions_indexes()
    st.session_state["indexes_ensured"] = True

st.title("🎓 RAG-Based MCQ Quiz Generator")

if st.session_state.get("authenticated", False):
    st.success(f"Welcome back, {st.session_state['user_email']}!")
    st.page_link("pages/3_Quiz_Generator.py", label="Go to Quiz Generator", icon="📝")
    st.page_link("pages/4_Previous_Questions.py", label="Question Bank", icon="📚")
else:
    st.info("Please log in or register to use the quiz generator.")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_Login.py", label="Login", icon="🔐")
    with col2:
        st.page_link("pages/2_Register.py", label="Register", icon="📝")