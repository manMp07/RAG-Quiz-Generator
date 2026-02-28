# utils/storage.py
import streamlit as st
import pickle
import faiss
import numpy as np  # <-- add this import
from utils.db import get_db
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_faiss_to_mongodb(pdf_hash: str, index: faiss.Index, chunks: list, user_id: str):
    """
    Save FAISS index and chunks to MongoDB for a specific user.
    """
    db = get_db()
    
    # Serialize FAISS index to bytes
    faiss_data = faiss.serialize_index(index)
    # Convert numpy array to bytes if needed
    if isinstance(faiss_data, np.ndarray):
        faiss_data = faiss_data.tobytes()
    elif not isinstance(faiss_data, bytes):
        faiss_data = bytes(faiss_data)
    
    # Serialize chunks
    chunks_data = pickle.dumps(chunks)
    
    # Store in MongoDB (upsert to replace if exists)
    db.faiss_cache.update_one(
        {"pdf_hash": pdf_hash, "user_id": user_id},
        {
            "$set": {
                "faiss_index": faiss_data,
                "chunks": chunks_data,
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    logger.info(f"Saved FAISS cache for PDF {pdf_hash} (user {user_id})")

def load_faiss_from_mongodb(pdf_hash: str, user_id: str):
    """
    Load FAISS index and chunks from MongoDB for a specific user.
    Returns (index, chunks) or (None, None) if not found.
    """
    db = get_db()
    cached = db.faiss_cache.find_one({
        "pdf_hash": pdf_hash,
        "user_id": user_id
    })
    
    if not cached:
        return None, None
    
    try:
        # Deserialize FAISS index from bytes
        faiss_data = cached["faiss_index"]
        index = faiss.deserialize_index(faiss_data)
        
        # Restore chunks
        chunks = pickle.loads(cached["chunks"])
        
        logger.info(f"Loaded FAISS cache for PDF {pdf_hash} (user {user_id})")
        return index, chunks
    except Exception as e:
        logger.error(f"Failed to load cached FAISS: {e}")
        return None, None

def delete_faiss_from_mongodb(pdf_hash: str, user_id: str):
    """Delete a cached FAISS entry for a user (optional cleanup)."""
    db = get_db()
    result = db.faiss_cache.delete_one({"pdf_hash": pdf_hash, "user_id": user_id})
    return result.deleted_count > 0