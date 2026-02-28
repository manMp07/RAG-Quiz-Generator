# utils/db.py
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_db():
    """Return MongoDB database connection (cached)."""
    try:
        client = MongoClient(st.secrets["MONGODB_URI"])
        # Verify connection
        client.admin.command('ping')
        logger.info("Connected to MongoDB")
        return client[st.secrets.get("DB_NAME", "quiz_app")]
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        st.error("Database connection failed. Please check your credentials.")
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred.")
        st.stop()

def get_user_by_email(email: str):
    """Return user document from 'users' collection by email."""
    db = get_db()
    return db.users.find_one({"email": email.lower()})

def create_user(email: str, password_hash: str):
    """Insert a new user into 'users' collection."""
    db = get_db()
    user = {
        "email": email.lower(),
        "password_hash": password_hash,
        "created_at": __import__("datetime").datetime.utcnow()
    }
    result = db.users.insert_one(user)
    return result.inserted_id

def ensure_cache_indexes():
    """Create indexes for faiss_cache collection."""
    db = get_db()
    db.faiss_cache.create_index([("pdf_hash", 1), ("user_id", 1)], unique=True)


def save_questions(user_id: str, course: str, questions: list):
    """
    Save a list of questions to the 'questions' collection.
    Each question dict should contain:
        question_text, options (list of 4), correct_answer, difficulty
    """
    db = get_db()
    for q in questions:
        doc = {
            "user_id": user_id,
            "course": course,
            "question_text": q["question"],
            "options": q["options"],
            "correct_answer": q["answer"],
            "difficulty": q.get("difficulty", "Not Specified"),
            "created_at": __import__("datetime").datetime.utcnow()
        }
        db.questions.insert_one(doc)

def get_questions_by_user(user_id: str, course: str = None):
    """
    Retrieve all questions for a user, optionally filtered by course.
    Returns a list of dicts (including MongoDB _id as string).
    """
    db = get_db()
    query = {"user_id": user_id}
    if course and course != "None":
        query["course"] = course
    cursor = db.questions.find(query).sort("created_at", -1)
    questions = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # convert ObjectId to string for JSON serialization
        questions.append(doc)
    return questions

def ensure_questions_indexes():
    db = get_db()
    db.questions.create_index([("user_id", 1), ("course", 1)])