"""Embedding service for Job Match Pro.

Generates embeddings using Google's text-embedding model.
"""

import os
import time
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to generate embedding for
        task_type: "RETRIEVAL_DOCUMENT" for JDs, "RETRIEVAL_QUERY" for CVs
        
    Returns:
        768-dimensional embedding vector
    """
    try:
        response = client.models.embed_content(
            model="text-embedding-004", # dimension = 786
            contents=[text],
            config=types.EmbedContentConfig(task_type=task_type)
        )
        
        return response.embeddings[0].values
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")


def generate_embeddings_batch(
    texts: List[str], 
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 10,
    delay_seconds: float = 2.0
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts with rate limiting.
    
    Args:
        texts: List of texts to generate embeddings for
        task_type: "RETRIEVAL_DOCUMENT" for JDs, "RETRIEVAL_QUERY" for CVs
        batch_size: Number of texts per API call
        delay_seconds: Delay between batches to avoid rate limits
        
    Returns:
        List of 768-dimensional embedding vectors
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            
            batch_embeddings = [entry.values for entry in response.embeddings]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limit protection
            if i + batch_size < len(texts):
                time.sleep(delay_seconds)
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings for batch {i}: {e}")
    
    return all_embeddings
