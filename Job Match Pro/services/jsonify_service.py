"""JSONification service for Job Match Pro.

Converts raw JD and CV text into structured JSON using Gemini API.
"""

import json
import os
import re
import uuid
from datetime import datetime
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def jsonify_jd(jd_text: str, filename: str = None) -> list[dict]:
    """
    Convert Job Description text to structured JSON using Gemini.
    Handles files with one or multiple JDs.
    
    Args:
        jd_text: Raw JD text extracted from file (may contain multiple JDs)
        filename: Original filename (optional)
        
    Returns:
        List of dictionaries, each with job_title, description, id, and metadata
    """
    prompt = f"""Parse this document containing ONE OR MORE Job Descriptions and return a JSON ARRAY of jobs.

Each job in the array must have ONLY TWO fields:
1. "job_title"
2. "description"

The "description" field must be a single, highly information-dense string optimized for semantic search.

STRICT RULES:
- Identify ALL separate job descriptions in the document.
- Extract ONLY job-relevant details for each job.
- Include all responsibilities and requirements.
- Compress long sentences into short, keyword-dense statements.
- Normalize skill names (React.js not ReactJS, Node.js not Node, etc.)
- Remove filler text, HR language, or benefits.
- Do NOT include location, salary, company intro, or cultural statements.
- Do NOT output bullet symbols — convert into compact text blocks.

Each "description" must follow EXACTLY this structure:

Responsibilities:
<Concise list of responsibilities in 3 to 6 compressed lines. Each line should start with a verb.>

Requirements:
<Skill Set: programming languages, frameworks, databases, tools, cloud, DevOps, etc.>
<Technologies: list>
<Experience Level: junior/mid/senior>
<Category: Software Development / DevOps / Data / QA / UIUX / Cloud>

Return ONLY a valid JSON ARRAY (even if there's only one job):
[
  {{
    "job_title": "...",
    "description": "Responsibilities: ... Requirements: ..."
  }},
  {{
    "job_title": "...",
    "description": "Responsibilities: ... Requirements: ..."
  }}
]

Document with Job Description(s):
{jd_text}

JSON Array:
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Handle different response formats
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            response_text = response.candidates[0].content.parts[0].text.strip()
        else:
            raise ValueError(f"Unexpected response format: {type(response)}")
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        parsed_json = json.loads(response_text)
        
        # Ensure it's a list
        if isinstance(parsed_json, dict):
            # Single JD returned as object, wrap in list
            parsed_json = [parsed_json]
        elif not isinstance(parsed_json, list):
            raise ValueError(f"Expected list or dict, got {type(parsed_json)}")
        
        # Add metadata to each JD
        result = []
        for idx, job_dict in enumerate(parsed_json):
            if not isinstance(job_dict, dict):
                continue
            
            job_dict["id"] = str(uuid.uuid4())
            job_dict["filename"] = filename or "unknown.txt"
            job_dict["created_at"] = datetime.now().isoformat()
            result.append(job_dict)
        
        if not result:
            raise ValueError("No valid job descriptions found in response")
        
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from Gemini response: {e}\nResponse: {response_text[:500]}")
    except Exception as e:
        raise RuntimeError(f"Failed to JSONify JD: {e}")


def jsonify_cv(cv_text: str) -> dict:
    """
    Convert CV/Resume text to structured JSON using Gemini.
    
    Args:
        cv_text: Raw CV text extracted from PDF
        
    Returns:
        Dictionary with description field optimized for semantic search
    """
    prompt = """Parse this CV/Resume and produce a JSON with ONLY one field called "description".

The "description" field should be a single structured, skill-dense text optimized for semantic search.

STRICT RULES:
- Do NOT include soft skills unless they are job-relevant.
- Do NOT include personal details (address, phone, hobbies).
- Extract ONLY job-relevant information.
- Convert everything into a short, compact, dense description.
- Do NOT create long sentences. Prefer keyword-rich, concise phrasing.

The "description" must follow EXACTLY this format and order:

Summary:
<2 to 3 lines max summarizing the candidate's role, expertise, domains>

Education:
<Degree> — <Institution> (<Duration>), CGPA: <score>

Skill Set:
<Programming Languages: ...>
<Frameworks: ...>
<Cloud: ...>
<Databases: ...>
<DevOps Tools: ...>
<Other Tools: ...>

Experience:
- <Role/Project Name> (<Duration>): <Short description>. Technologies: <tech stack>
- <Role/Project Name>: <Short description>. Technologies: <tech stack>

Certifications:
<List all certifications separated by comma>

IMPORTANT:
- Use clean labels for skills (Python, React.js, SQL, AWS, Docker, etc.)
- Remove duplicates
- Normalize tech names (e.g., "JS" → "JavaScript")
- The output must be very compact and highly information-dense for embeddings.

Return ONLY a valid JSON object with one key: "description".

CV Content:
""" + cv_text + """

JSON:
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Handle different response formats
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            response_text = response.candidates[0].content.parts[0].text.strip()
        else:
            raise ValueError(f"Unexpected response format: {type(response)}")
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from Gemini response: {e}\nResponse: {response_text[:500]}")
    except Exception as e:
        raise RuntimeError(f"Failed to JSONify CV: {e}")
