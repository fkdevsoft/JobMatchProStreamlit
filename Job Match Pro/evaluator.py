import dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
import pandas as pd
import json
from google import genai

dotenv.load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PC_API_KEY"))

# Initialize Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CV_embeddings = pd.DataFrame(pd.read_pickle("./output/embedded_CV.pkl"))
JD_embeddings = pd.DataFrame(pd.read_pickle("./output/embedded_JD.pkl"))

JD_titles, JD_descriptions, JD_embedding = JD_embeddings.columns
CV_description, CV_embedding = CV_embeddings.columns

job_size = 5

index_name = "job-match-pro"

# Create index if needed
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={"environment": "development"}
    )

# Get index handle
index = pc.Index(index_name)

# Upsert JD vectors
"""for row_id, row in JD_embeddings.iterrows():
    index.upsert(
        namespace="__default__",
        vectors=[{
            "id": str(uuid.uuid4()),
            "values": row[JD_embedding],
            "metadata": {
                "job_title": row[JD_titles],
                "job_description": row[JD_descriptions],
            },
        }],
    )

print("Index updated successfully.")"""

# Query with first CV embedding
query_vector = CV_embeddings.loc[0, CV_embedding]
results = index.query(
    namespace="__default__",
    vector=query_vector,
    top_k=job_size,
    include_metadata=True,
    include_values=False,
)
print("Query Processed Successfully!")

# Format Pinecone results for LLM
pinecone_results_formatted = []
for match in results['matches']:
    pinecone_results_formatted.append({
        "job_title": match['metadata'].get('job_title', 'N/A'),
        "job_description": match['metadata'].get('job_description', 'N/A'),
        "similarity_score": match['score']
    })

# Batch processing configuration
BATCH_SIZE = 5
all_evaluation_results = []

# Split results into batches of 10
def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

batches = list(chunk_list(pinecone_results_formatted, BATCH_SIZE))
total_batches = len(batches)

print(f"\nTotal jobs to process: {len(pinecone_results_formatted)}")
print(f"Processing in {total_batches} batch(es) of {BATCH_SIZE} jobs each...")

# Process each batch
for batch_num, batch in enumerate(batches, 1):
    print(f"\n--- Processing Batch {batch_num}/{total_batches} ({len(batch)} jobs) ---")
    
    # Construct the LLM prompt for this batch
    llm_prompt = f"""You are a Resume to JD matching expert. 
Analyze the following information and produce a JSON output with EXACTLY the structure given below.

Inputs Provided:
1. Candidate Resume
2. Job Description (JD)
3. My Pinecone query results (batch {batch_num} of {total_batches}, containing {len(batch)} jobs)
4. The parsed Resume description

Your Task:
- Read the resume and extract all technical skills, tools, cloud platforms, programming languages, software engineering concepts, and relevant experience.
- Read the JD and extract required skills, preferred skills, role responsibilities, experience level, and job category.
- Perform a detailed comparison between Resume Skills and JD Requirements.
- Use the similarity score as only one part of the evaluation — do NOT rely on it alone.
- Compute an overall "match_score" using your reasoning:
  - 40% → Skill overlap (very important)
  - 40% → Experience/responsibility match
  - 10% → Title relevance
  - 10% → Category alignment

Scoring Instructions:
1. similarity (0 to 100):
   - Estimate semantic similarity between the resume and JD.
   - Consider responsibilities, technical fit, relevance, and experience alignment.
   - DO NOT output cosine values; convert to a human-readable score.

2. match_score (0 to 100):
   - Combine:
     - semantic similarity,
     - skill overlap,
     - relevance of responsibilities,
     - experience alignment,
     - category fitness.
   - Higher means stronger match.

3. category_match:
   - "Yes" if the resume belongs to the same category as JD (Software Dev, DevOps, Data, QA, UIUX, Cloud).
   - Otherwise "No".

4. skills_matched:
   - Extract skills from JD.
   - List only the ones actually present in the resume.

5. skills_missing:
   - Extract remaining JD-required skills not found in resume.

6. verdict:
   - Provide a CLEAR, HUMAN explanation of:
     - Why the match score was given,
     - Why the similarity score was given,
     - Why skills matched / missing matter,
     - Whether experience is relevant,
     - Whether responsibilities align,
     - Any weaknesses or mismatches.

IMPORTANT:
Output MUST be valid JSON with EXACTLY this structure:

{{
     "job_title": "",
     "similarity": "",
     "match_score": "",
     "category_match": "",
     "skills_matched": [],
     "skills_missing": [],
     "verdict": ""
}}

Now begin.

**Pinecone Query Results (Batch {batch_num}/{total_batches}):**
""" + json.dumps(batch, indent=2) + """

**Parsed Resume:**
""" + json.dumps(CV_embeddings["description"].tolist(), indent=2) + """

Please analyze and provide the output as a valid JSON array containing the evaluation for each job in this batch.
"""

    # Call Gemini API for this batch
    print(f"Sending batch {batch_num} to Gemini API...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=llm_prompt
    )

    # Extract the response text
    llm_response = response.text
    print(f"Batch {batch_num} response received.")

    # Parse JSON from response
    try:
        json_start = llm_response.find('[')
        json_end = llm_response.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = llm_response[json_start:json_end]
            batch_results = json.loads(json_str)
        else:
            batch_results = json.loads(llm_response)
        
        # Add batch results to the main list
        all_evaluation_results.extend(batch_results)
        print(f"Batch {batch_num}: Successfully parsed {len(batch_results)} evaluations.")
        
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON from batch {batch_num}. Error: {e}")
        # Store raw response for this batch
        all_evaluation_results.append({
            "batch": batch_num,
            "error": "JSON parse failed",
            "raw_response": llm_response
        })

# Save all results to JSON file
output_file = "./output/evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(all_evaluation_results, f, indent=2)

print(f"\n{'='*50}")
print(f"All batches processed!")
print(f"Total evaluations: {len(all_evaluation_results)}")
print(f"Results saved to {output_file}")