"""Helper functions for processing job descriptions (JD) uploads."""

import os
import json
import uuid
import re
from datetime import datetime
from typing import Optional, Dict, List
import dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from text import smart_extract_pdf, clean_text
from jsonify_JD import parse_single_job_with_openai

# Load environment variables
dotenv.load_dotenv()

# Initialize APIs lazily to avoid errors if .env is not loaded
_openai_client = None
_pc = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY environment variable. "
                "Please set this in your .env file."
            )
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def get_pinecone_client():
    """Get or create Pinecone client."""
    global _pc
    if _pc is None:
        pc_api_key = os.getenv("PC_API_KEY")
        if not pc_api_key:
            raise ValueError(
                "Missing PC_API_KEY environment variable. "
                "Please set this in your .env file."
            )
        _pc = Pinecone(api_key=pc_api_key)
    return _pc

# Constants
JDS_FOLDER = os.path.join(os.path.dirname(__file__), "jds")
METADATA_FILE = os.path.join(os.path.dirname(__file__), "output", "jd_metadata.json")
INDEX_NAME = "job-match-pro"
EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-small dimension


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(JDS_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)


def get_pinecone_index():
    """Get or create Pinecone index."""
    pc_client = get_pinecone_client()
    # Check if index exists, if not create it
    try:
        index = pc_client.Index(INDEX_NAME)
        return index
    except Exception:
        # Index doesn't exist, create it
        pc_client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        return pc_client.Index(INDEX_NAME)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Ensure it ends with .pdf
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    return filename


def save_jd_pdf(uploaded_file, jd_title: str) -> str:
    """Save uploaded PDF to jds folder with sanitized filename."""
    ensure_directories()
    
    # Create filename from title
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', jd_title)
    safe_title = safe_title.strip('. ')[:100]  # Limit length
    filename = sanitize_filename(f"{safe_title}_{uuid.uuid4().hex[:8]}.pdf")
    file_path = os.path.join(JDS_FOLDER, filename)
    
    # Save file (reset file pointer to beginning in case it was read before)
    uploaded_file.seek(0)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    return file_path


def extract_jd_text(pdf_path: str) -> str:
    """Extract and clean text from JD PDF using existing text.py functions."""
    extracted = smart_extract_pdf(pdf_path)
    cleaned = clean_text(extracted)
    return cleaned


def parse_jd_text(jd_text: str) -> dict:
    """Parse JD text using existing jsonify_JD.py function."""
    return parse_single_job_with_openai(jd_text)


def generate_jd_embedding(parsed_jd: dict) -> list:
    """Generate embedding for a single JD using OpenAI."""
    description = parsed_jd.get("description", "")
    
    # Validate description
    if not description:
        raise ValueError("JD description is empty. Cannot generate embedding.")
    
    if not isinstance(description, str):
        description = str(description)
    
    # Ensure description is not just whitespace
    description = description.strip()
    if not description:
        raise ValueError("JD description is empty after cleaning. Cannot generate embedding.")
    
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=description
        )
        
        # Extract vector from response
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("No embedding returned from API")
            
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")


def store_in_pinecone(jd_data: dict, embedding: list, pinecone_id: str) -> None:
    """Store JD embedding in Pinecone."""
    index = get_pinecone_index()
    
    index.upsert(
        namespace="__default__",
        vectors=[{
            "id": pinecone_id,
            "values": embedding,
            "metadata": {
                "job_title": jd_data.get("job_title", ""),
                "job_description": jd_data.get("description", ""),
            },
        }],
    )


def load_jd_metadata() -> List[Dict]:
    """Load JD metadata from JSON file."""
    ensure_directories()
    
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_jd_metadata(metadata: List[Dict]) -> None:
    """Save JD metadata to JSON file."""
    ensure_directories()
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def add_jd_to_metadata(jd_info: Dict) -> None:
    """Add a new JD to metadata file."""
    metadata = load_jd_metadata()
    metadata.append(jd_info)
    save_jd_metadata(metadata)


def delete_jd_from_metadata(jd_id: str) -> Optional[Dict]:
    """Delete JD from metadata and return deleted entry."""
    metadata = load_jd_metadata()
    
    for i, jd in enumerate(metadata):
        if jd.get("id") == jd_id:
            deleted = metadata.pop(i)
            save_jd_metadata(metadata)
            return deleted
    return None


def delete_jd_from_pinecone(pinecone_id: str) -> None:
    """Delete JD vector from Pinecone."""
    try:
        index = get_pinecone_index()
        index.delete(ids=[pinecone_id], namespace="__default__")
    except Exception as e:
        # Log error but don't fail if vector doesn't exist
        print(f"Warning: Could not delete from Pinecone: {e}")


def process_jd_sync(uploaded_file, jd_title: str) -> Dict:
    """
    Main synchronous processing function that chains all steps:
    1. Save PDF
    2. Extract text
    3. Parse with OpenAI
    4. Generate embedding
    5. Store in Pinecone
    6. Save metadata
    
    Returns dict with processing result and metadata.
    """
    jd_id = str(uuid.uuid4())
    pinecone_id = str(uuid.uuid4())
    
    try:
        # Step 1: Save PDF
        file_path = save_jd_pdf(uploaded_file, jd_title)
        
        # Step 2: Extract text
        jd_text = extract_jd_text(file_path)
        
        if not jd_text or len(jd_text.strip()) < 50:
            raise ValueError("Extracted text is too short or empty. Please check the PDF.")
        
        # Step 3: Parse with OpenAI
        parsed_jd = parse_jd_text(jd_text)
        
        # Step 4: Generate embedding
        embedding = generate_jd_embedding(parsed_jd)
        
        # Step 5: Store in Pinecone
        store_in_pinecone(parsed_jd, embedding, pinecone_id)
        
        # Step 6: Save metadata
        jd_metadata = {
            "id": jd_id,
            "title": jd_title,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "pinecone_id": pinecone_id,
            "parsed_data": parsed_jd
        }
        add_jd_to_metadata(jd_metadata)
        
        return {
            "success": True,
            "metadata": jd_metadata,
            "message": "JD processed and stored successfully"
        }
        
    except Exception as e:
        # Cleanup on failure
        error_msg = str(e)
        
        # Try to clean up file if it was saved
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        # Try to clean up Pinecone if vector was stored
        try:
            delete_jd_from_pinecone(pinecone_id)
        except:
            pass
        
        return {
            "success": False,
            "error": error_msg,
            "message": f"Failed to process JD: {error_msg}"
        }


def delete_jd(jd_id: str) -> Dict:
    """Delete JD: remove file, Pinecone vector, and metadata."""
    metadata = load_jd_metadata()
    jd_to_delete = None
    
    for jd in metadata:
        if jd.get("id") == jd_id:
            jd_to_delete = jd
            break
    
    if not jd_to_delete:
        return {"success": False, "message": "JD not found"}
    
    try:
        # Delete PDF file
        file_path = jd_to_delete.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from Pinecone
        pinecone_id = jd_to_delete.get("pinecone_id")
        if pinecone_id:
            delete_jd_from_pinecone(pinecone_id)
        
        # Delete from metadata
        delete_jd_from_metadata(jd_id)
        
        return {"success": True, "message": "JD deleted successfully"}
        
    except Exception as e:
        return {"success": False, "message": f"Error deleting JD: {str(e)}"}

