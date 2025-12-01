import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client lazily
_client = None

def get_openai_client():
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY environment variable. "
                "Please set this in your .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client

# For backward compatibility
client = None  # Will be initialized when needed

# Load job descriptions from JSON file
with open("./output/CV_parsed.json", "r", encoding="utf-8") as f:
    extracted_jobs = json.load(f)

def get_embeddings_batched(jobs, batch_size=20):
    """
    Batches jobs to save API calls, but sleeps to save Token Limits.
    """
    all_embeddings = []
    
    # Process in chunks (batches)
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        batch_texts = [j["description"] for j in batch]
        
        try:
            print(f"Processing batch {i} to {i+batch_size}...")
            
            # ONE API call for 'batch_size' documents
            client = get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            
            # Extract vectors from response
            batch_vectors = [entry.embedding for entry in response.data]
            
            # Match them back to the job objects
            for job, vector in zip(batch, batch_vectors):
                job["embedding"] = vector
                all_embeddings.append(job)

            # --- CRITICAL: RATE LIMIT PROTECTION ---
            # We sleep 2 seconds between batches to stay under the 
            # 30,000 Tokens Per Minute limit.
            time.sleep(2) 

        except Exception as e:
            print(f"❌ Batch failed: {e}")
            # Optional: Add retry logic here
            
    return all_embeddings

# Run it
print("Starting Batch Embedding...")
processed_data = get_embeddings_batched(extracted_jobs, batch_size=10)

# Save
df = pd.DataFrame(processed_data)
df.to_pickle("./output/embedded_CV.pkl")
df.to_csv("./output/embedded_CV.csv", index=False)
print(f"Saved {len(df)} embedded CV entries.")