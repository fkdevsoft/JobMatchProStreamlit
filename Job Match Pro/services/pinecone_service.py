"""Pinecone service for Job Match Pro.

Handles all Pinecone vector database operations.
"""

import os
import uuid
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PC_API_KEY"))

INDEX_NAME = "job-match-pro"
NAMESPACE = "jd_namespace"


def init_pinecone_index(index_name: str = INDEX_NAME):
    """
    Initialize or get existing Pinecone index.
    
    Args:
        index_name: Name of the index
        
    Returns:
        Pinecone Index object
    """
    # Create index if it doesn't exist
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=768,  # Google text-embedding-004 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled",
            tags={"environment": "development"}
        )
    
    return pc.Index(index_name)


def upsert_jd(jd_json: dict, embedding: List[float], index_name: str = INDEX_NAME) -> str:
    """
    Upsert a Job Description to Pinecone.
    
    Args:
        jd_json: JD data with job_title, description, id
        embedding: 768-dimensional embedding vector
        index_name: Name of the Pinecone index
        
    Returns:
        Vector ID for later reference/deletion
    """
    index = init_pinecone_index(index_name)
    
    # Generate a unique vector ID
    vector_id = str(uuid.uuid4())
    
    # Prepare metadata (Pinecone has metadata size limits)
    metadata = {
        "job_title": jd_json.get("job_title", "Unknown"),
        "job_description": jd_json.get("description", "")[:8000],  # Truncate if needed
        "local_id": jd_json.get("id", ""),
        "filename": jd_json.get("filename", ""),
    }
    
    # Upsert to Pinecone
    index.upsert(
        namespace=NAMESPACE,
        vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": metadata,
        }]
    )
    
    return vector_id


def query_with_cv(
    cv_embedding: List[float], 
    top_k: int = 10,
    index_name: str = INDEX_NAME
) -> List[Dict]:
    """
    Query Pinecone with CV embedding to find matching JDs.
    
    Args:
        cv_embedding: 768-dimensional CV embedding vector
        top_k: Number of top matches to return
        index_name: Name of the Pinecone index
        
    Returns:
        List of matched JDs with similarity scores
    """
    index = init_pinecone_index(index_name)
    
    results = index.query(
        namespace=NAMESPACE,
        vector=cv_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    
    # Format results
    matches = []
    for match in results.get('matches', []):
        matches.append({
            "vector_id": match['id'],
            "job_title": match['metadata'].get('job_title', 'N/A'),
            "job_description": match['metadata'].get('job_description', 'N/A'),
            "similarity_score": match['score'],
            "local_id": match['metadata'].get('local_id', ''),
            "filename": match['metadata'].get('filename', ''),
        })
    
    return matches


def delete_jd_by_id(vector_id: str, index_name: str = INDEX_NAME) -> bool:
    """
    Delete a JD from Pinecone by vector ID.
    
    Args:
        vector_id: The Pinecone vector ID
        index_name: Name of the Pinecone index
        
    Returns:
        True if deletion was successful
    """
    try:
        index = init_pinecone_index(index_name)
        index.delete(
            ids=[vector_id],
            namespace=NAMESPACE
        )
        return True
    except Exception as e:
        print(f"Failed to delete vector {vector_id}: {e}")
        return False


def delete_jds_by_ids(vector_ids: List[str], index_name: str = INDEX_NAME) -> bool:
    """
    Delete multiple JDs from Pinecone by vector IDs.
    
    Args:
        vector_ids: List of Pinecone vector IDs
        index_name: Name of the Pinecone index
        
    Returns:
        True if deletion was successful
    """
    if not vector_ids:
        return True
        
    try:
        index = init_pinecone_index(index_name)
        index.delete(
            ids=vector_ids,
            namespace=NAMESPACE
        )
        return True
    except Exception as e:
        print(f"Failed to delete vectors: {e}")
        return False


def get_index_stats(index_name: str = INDEX_NAME) -> Dict:
    """
    Get statistics about the Pinecone index.
    
    Args:
        index_name: Name of the Pinecone index
        
    Returns:
        Dictionary with index statistics
    """
    try:
        index = init_pinecone_index(index_name)
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "namespaces": stats.get('namespaces', {}),
        }
    except Exception as e:
        return {"error": str(e)}
