# pages/2_Register.py
import streamlit as st
from utils.auth import register_user

st.set_page_config(page_title="Register", page_icon="📝")

# If already logged in, redirect to quiz generator
if st.session_state.get("authenticated", False):
    st.switch_page("pages/3_Quiz_Generator.py")

st.title("📝 Faculty Registration")

with st.form("register_form"):
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    submitted = st.form_submit_button("Register")
    
    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(email, password)
            if success:
                st.success(message)
                st.info("Please login with your new credentials.")
                # Optionally auto-redirect to login
                st.switch_page("pages/1_Login.py")
            else:
                st.error(message)

st.markdown("Already have an account? [Login here](/1_Login)")