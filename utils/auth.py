# utils/auth.py
import streamlit as st
import bcrypt
from utils.db import get_user_by_email, create_user
import logging

logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    Truncates to 72 bytes silently (bcrypt's max).
    """
    # bcrypt operates on bytes, max 72 bytes
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = password.encode('utf-8')[:72]
    try:
        return bcrypt.checkpw(password_bytes, password_hash.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def register_user(email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success: bool, message: str).
    """
    if not email or not password:
        return False, "Email and password are required."
    
    if len(password) > 72:
        # Optionally warn, but we truncate silently
        logger.warning(f"Password longer than 72 bytes; will be truncated.")
    
    existing = get_user_by_email(email)
    if existing:
        return False, "Email already registered."
    
    try:
        password_hash = hash_password(password)
        create_user(email, password_hash)
        logger.info(f"New user registered: {email}")
        return True, "Registration successful! You can now log in."
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False, "An error occurred during registration."

def login_user(email: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Log in a user.
    Returns (success: bool, message: str, user: dict or None).
    """
    if not email or not password:
        return False, "Email and password are required.", None
    
    user = get_user_by_email(email)
    if not user:
        return False, "Email not found.", None
    
    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect password.", None
    
    # Don't return the password hash
    user.pop("password_hash", None)
    return True, "Login successful!", user

def logout():
    """Clear authentication session."""
    for key in ["authenticated", "user_email", "user_id"]:
        if key in st.session_state:
            del st.session_state[key]