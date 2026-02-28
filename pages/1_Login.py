# pages/1_Login.py
import streamlit as st
from utils.auth import login_user

st.set_page_config(page_title="Login", page_icon="🔐")

# If already logged in, redirect to quiz generator
if st.session_state.get("authenticated", False):
    st.switch_page("pages/3_Quiz_Generator.py")

st.title("🔐 Faculty Login")

with st.form("login_form"):
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")
    
    if submitted:
        success, message, user = login_user(email, password)
        if success:
            st.session_state["authenticated"] = True
            st.session_state["user_email"] = user["email"]
            st.session_state["user_id"] = str(user["_id"])
            st.success(message)
            st.switch_page("pages/3_Quiz_Generator.py")
        else:
            st.error(message)

st.markdown("Don't have an account? [Register here](/2_Register)")