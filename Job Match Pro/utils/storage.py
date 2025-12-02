"""Local storage utilities for Job Match Pro.

Manages local JD JSON storage for persistence and display.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime

# Path to local JD store
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
JD_STORE_PATH = os.path.join(DATA_DIR, "jd_store.json")


def _ensure_data_dir():
    """Ensure data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def _ensure_store_file():
    """Ensure JD store file exists."""
    _ensure_data_dir()
    if not os.path.exists(JD_STORE_PATH):
        with open(JD_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)


def load_jd_store() -> List[Dict]:
    """
    Load all stored JDs from local JSON file.
    
    Returns:
        List of JD dictionaries
    """
    _ensure_store_file()
    try:
        with open(JD_STORE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_jd_to_store(jd_data: Dict) -> None:
    """
    Add new JD to local store.
    
    Args:
        jd_data: JD dictionary with id, job_title, description, 
                 pinecone_vector_id, filename, created_at
    """
    _ensure_store_file()
    
    # Load existing data
    jds = load_jd_store()
    
    # Check for duplicates by id
    existing_ids = {jd.get('id') for jd in jds}
    if jd_data.get('id') in existing_ids:
        # Update existing
        jds = [jd_data if jd.get('id') == jd_data.get('id') else jd for jd in jds]
    else:
        # Add new
        jds.append(jd_data)
    
    # Save back
    with open(JD_STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(jds, f, indent=2, ensure_ascii=False)


def delete_jd_from_store(jd_id: str) -> bool:
    """
    Remove JD from local store by ID.
    
    Args:
        jd_id: The JD's unique ID
        
    Returns:
        True if deletion was successful, False if JD not found
    """
    _ensure_store_file()
    
    jds = load_jd_store()
    original_count = len(jds)
    
    # Filter out the JD to delete
    jds = [jd for jd in jds if jd.get('id') != jd_id]
    
    if len(jds) == original_count:
        return False  # JD not found
    
    # Save back
    with open(JD_STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(jds, f, indent=2, ensure_ascii=False)
    
    return True


def get_jd_count() -> int:
    """
    Get count of stored JDs.
    
    Returns:
        Number of JDs in store
    """
    return len(load_jd_store())


def get_jd_by_id(jd_id: str) -> Optional[Dict]:
    """
    Get a specific JD by ID.
    
    Args:
        jd_id: The JD's unique ID
        
    Returns:
        JD dictionary or None if not found
    """
    jds = load_jd_store()
    for jd in jds:
        if jd.get('id') == jd_id:
            return jd
    return None


def clear_jd_store() -> None:
    """Clear all JDs from local store."""
    _ensure_store_file()
    with open(JD_STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump([], f)
