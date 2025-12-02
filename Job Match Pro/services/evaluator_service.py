"""Evaluator service for Job Match Pro.

Evaluates CV against matched JDs using Gemini AI.
"""

import json
import os
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BATCH_SIZE = 5


def evaluate_matches(cv_description: str, matched_jds: List[Dict]) -> List[Dict]:
    """
    Evaluate CV against matched JDs using Gemini AI.
    
    Args:
        cv_description: Filtered CV's description text
        matched_jds: List of matched JDs from Pinecone query
        
    Returns:
        List of evaluation results with scores and analysis
    """
    if not matched_jds:
        return []
    
    all_results = []
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(matched_jds), BATCH_SIZE), 1):
        batch = matched_jds[i:i + BATCH_SIZE]
        total_batches = (len(matched_jds) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Prepare batch data for prompt
        batch_data = [{
            "job_title": jd.get("job_title", "N/A"),
            "job_description": jd.get("job_description", "N/A"),
            "similarity_score": jd.get("similarity_score", 0)
        } for jd in batch]
        
        # Construct prompt
        prompt = f"""You are a Resume to JD matching expert. 
        Analyze the following information and produce a JSON output with EXACTLY the structure given below.
        
        Inputs Provided:
        1. Candidate Resume
        2. Job Description (JD)
        3. Pinecone query results (batch {batch_num} of {total_batches}, containing {len(batch)} jobs)
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
           - Provide a short verdict: "Excellent Match", "Good Match", "Moderate", or "Poor Match"
           - Based on match_score: 85+ = Excellent, 70-84 = Good, 50-69 = Moderate, <50 = Poor
        
        7. explanation:
           - Provide a dict with two keys:
             - "match_accuracy": detailed explanation of why the score was given
             - "score_interpretation": summary of strengths and weaknesses
        
        IMPORTANT:
        Output MUST be valid JSON with EXACTLY this structure:
        
        {{
             "job_title": "",
             "similarity": 0,
             "match_score": 0,
             "category_match": "",
             "skills_matched": [],
             "skills_missing": [],
             "verdict": "",
             "explanation": {{
                 "match_accuracy": "",
                 "score_interpretation": ""
             }}
        }}
        
        Now begin.
        
        **Pinecone Query Results (Batch {batch_num}/{total_batches}):**
        {json.dumps(batch_data, indent=2)}
        
        **Parsed Resume:**
        {cv_description}
        
        Please analyze and provide the output as a valid JSON array containing the evaluation for each job in this batch.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            llm_response = response.text
            
            # Parse JSON from response
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                batch_results = json.loads(json_str)
            else:
                # Try parsing as single object
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    batch_results = [json.loads(json_str)]
                else:
                    batch_results = json.loads(llm_response)
            
            # Ensure batch_results is a list
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            
            # Add vector_id and local_id from original batch for reference
            for j, result in enumerate(batch_results):
                if j < len(batch):
                    result["vector_id"] = batch[j].get("vector_id", "")
                    result["local_id"] = batch[j].get("local_id", "")
            
            all_results.extend(batch_results)
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON from batch {batch_num}. Error: {e}")
            # Add error placeholder for this batch
            for jd in batch:
                all_results.append({
                    "job_title": jd.get("job_title", "N/A"),
                    "similarity": 0,
                    "match_score": 0,
                    "category_match": "N/A",
                    "skills_matched": [],
                    "skills_missing": [],
                    "verdict": "Error",
                    "explanation": {
                        "match_accuracy": "Failed to evaluate",
                        "score_interpretation": str(e)
                    },
                    "vector_id": jd.get("vector_id", ""),
                    "local_id": jd.get("local_id", ""),
                })
        except Exception as e:
            print(f"Error evaluating batch {batch_num}: {e}")
            raise
    
    # Sort by match_score descending
    all_results.sort(
        key=lambda x: float(str(x.get("match_score", 0)).strip('%')), 
        reverse=True
    )
    
    return all_results
