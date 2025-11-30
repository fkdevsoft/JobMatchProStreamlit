import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

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
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch_texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            # Extract vectors from response
            batch_vectors = [entry.values for entry in response.embeddings]
            
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