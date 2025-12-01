"""Helper functions for CV matching with job descriptions."""

import os
import json
import re
from typing import List, Dict
import dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from text import smart_extract_pdf, clean_text
from jsonify_CV import parse_cv_with_openai
from jd_processor import get_openai_client, get_pinecone_index, EMBEDDING_DIMENSION

# Load environment variables
dotenv.load_dotenv()

# Constants
INDEX_NAME = "job-match-pro"
TOP_K_JOBS = 10  # Number of top matching jobs to retrieve
BATCH_SIZE = 5  # Number of jobs to evaluate per LLM batch


def extract_cv_text(pdf_path: str) -> str:
    """Extract and clean text from CV PDF using existing text.py functions."""
    extracted = smart_extract_pdf(pdf_path)
    cleaned = clean_text(extracted)
    return cleaned


def parse_cv_text(cv_text: str) -> dict:
    """Parse CV text using existing jsonify_CV.py function."""
    return parse_cv_with_openai(cv_text)


def generate_cv_embedding(parsed_cv: dict) -> list:
    """Generate embedding for CV using OpenAI."""
    description = parsed_cv.get("description", "")
    
    # Handle case where description might be a dict or other type
    if isinstance(description, dict):
        # If description is a dict, convert to JSON string
        import json
        description = json.dumps(description, ensure_ascii=False)
    elif description is None:
        raise ValueError("CV description is None. The CV parsing may have failed.")
    elif not isinstance(description, str):
        description = str(description)
    
    # Ensure description is a string and not just whitespace
    if not isinstance(description, str):
        raise ValueError(f"CV description is not a string after conversion. Type: {type(description)}")
    
    description = description.strip()
    if not description:
        raise ValueError(f"CV description is empty after cleaning. Parsed CV keys: {list(parsed_cv.keys())}")
    
    # Additional validation: ensure description has minimum length
    if len(description) < 10:
        raise ValueError(f"CV description is too short ({len(description)} chars). Minimum 10 characters required.")
    
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
        error_msg = str(e)
        # Provide more context in error message
        if "input" in error_msg.lower() or "invalid" in error_msg.lower():
            raise Exception(f"Failed to generate CV embedding: {error_msg}. Description length: {len(description)}, Preview: {description[:100]}...")
        raise Exception(f"Failed to generate CV embedding: {error_msg}")


def query_pinecone_for_jobs(cv_embedding: list, top_k: int = TOP_K_JOBS) -> List[Dict]:
    """Query Pinecone to find matching job descriptions."""
    try:
        index = get_pinecone_index()
        
        results = index.query(
            namespace="__default__",
            vector=cv_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )
        
        # Format Pinecone results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                "job_title": match['metadata'].get('job_title', 'N/A'),
                "job_description": match['metadata'].get('job_description', 'N/A'),
                "similarity_score": match['score']
            })
        
        return formatted_results
        
    except Exception as e:
        raise Exception(f"Failed to query Pinecone: {str(e)}")


def chunk_list(lst: List, chunk_size: int):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def evaluate_matches_with_llm(
    parsed_cv: dict,
    pinecone_results: List[Dict],
    batch_size: int = BATCH_SIZE
) -> List[Dict]:
    """
    Evaluate CV matches with job descriptions using LLM.
    Uses the same evaluation logic as evaluator.py
    """
    all_evaluation_results = []
    
    # Split results into batches
    batches = list(chunk_list(pinecone_results, batch_size))
    total_batches = len(batches)
    
    cv_description = parsed_cv.get("description", "")
    
    # Process each batch
    for batch_num, batch in enumerate(batches, 1):
        # Construct the LLM prompt (same as evaluator.py)
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
""" + json.dumps([cv_description], indent=2) + """

Please analyze and provide the output as a valid JSON array containing the evaluation for each job in this batch.
"""
        
        try:
            # Call OpenAI API for this batch
            client = get_openai_client()
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are a Resume to JD matching expert. Analyze the information and produce JSON output with the exact structure specified."},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.3
            )
            
            # Extract the response text
            llm_response = response.choices[0].message.content
            
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
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON from batch {batch_num}. Error: {e}")
                # Store raw response for this batch
                all_evaluation_results.append({
                    "batch": batch_num,
                    "error": "JSON parse failed",
                    "raw_response": llm_response
                })
                
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            all_evaluation_results.append({
                "batch": batch_num,
                "error": str(e)
            })
    
    return all_evaluation_results


def process_cv_and_match(uploaded_file, top_k: int = TOP_K_JOBS) -> Dict:
    """
    Main function to process CV and find matching jobs.
    
    Steps:
    1. Save CV PDF temporarily
    2. Extract text
    3. Parse CV with OpenAI
    4. Generate embedding
    5. Query Pinecone for matching jobs
    6. Evaluate matches with LLM
    7. Clean up temporary file
    
    Returns dict with results.
    """
    import tempfile
    import uuid
    
    temp_file_path = None
    
    try:
        # Step 1: Save uploaded file temporarily
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"cv_{uuid.uuid4().hex[:8]}.pdf")
        
        uploaded_file.seek(0)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Step 2: Extract text
        cv_text = extract_cv_text(temp_file_path)
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise ValueError("Extracted CV text is too short or empty. Please check the PDF.")
        
        # Step 3: Parse CV
        parsed_cv = parse_cv_text(cv_text)
        
        # Validate parsed CV has description
        if not parsed_cv or not isinstance(parsed_cv, dict):
            raise ValueError("Failed to parse CV. Invalid response from AI.")
        
        description = parsed_cv.get("description", "")
        
        # Handle case where description might be a dict or other type
        if isinstance(description, dict):
            # If description is a dict, try to extract text from it or convert to JSON string
            import json
            description = json.dumps(description, ensure_ascii=False)
        elif not isinstance(description, str):
            description = str(description) if description is not None else ""
        
        # Ensure description is a string and not empty
        if not description or (isinstance(description, str) and not description.strip()):
            raise ValueError(f"CV parsing returned invalid description. Type: {type(description)}, Keys in parsed_cv: {list(parsed_cv.keys())}")
        
        # Step 4: Generate embedding
        cv_embedding = generate_cv_embedding(parsed_cv)
        
        # Step 5: Query Pinecone
        pinecone_results = query_pinecone_for_jobs(cv_embedding, top_k=top_k)
        
        if not pinecone_results:
            return {
                "success": True,
                "message": "No matching jobs found in the database.",
                "parsed_cv": parsed_cv,
                "matches": []
            }
        
        # Step 6: Evaluate matches with LLM
        evaluation_results = evaluate_matches_with_llm(parsed_cv, pinecone_results)
        
        # Sort by match_score descending
        for result in evaluation_results:
            try:
                # Convert match_score to float for sorting
                score_str = str(result.get("match_score", "0")).strip('%')
                result["match_score_num"] = float(score_str) if score_str.replace('.', '').isdigit() else 0
            except:
                result["match_score_num"] = 0
        
        evaluation_results.sort(key=lambda x: x.get("match_score_num", 0), reverse=True)
        
        return {
            "success": True,
            "message": f"Found {len(evaluation_results)} matching jobs",
            "parsed_cv": parsed_cv,
            "matches": evaluation_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to process CV: {str(e)}"
        }
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

