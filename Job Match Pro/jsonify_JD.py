import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and configure OpenAI
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


def read_job_descriptions(file_path: str) -> str:
    """Read the job descriptions from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_jobs(job_text: str) -> list[str]:
    """Split the job descriptions into individual jobs."""
    # Split by numbered job entries (1., 2., 3., etc.)
    jobs = re.split(r'\n(?=\d+\.\s)', job_text.strip())
    return [job.strip() for job in jobs if job.strip()]


def parse_single_job_with_openai(job_text: str) -> dict:
    """Use OpenAI API to parse a single job description into JSON format."""
    
    prompt = f"""Parse this Job Description and return a JSON with ONLY TWO fields:
1. "job_title"
2. "description"

The "description" field must be a single, highly information-dense string optimized for semantic search.

STRICT RULES:
- Extract ONLY job-relevant details.
- Include all responsibilities and requirements.
- Compress long sentences into short, keyword-dense statements.
- Normalize skill names (React.js not ReactJS, Node.js not Node, etc.)
- Remove filler text, HR language, or benefits.
- Do NOT include location, salary, company intro, or cultural statements.
- Do NOT output bullet symbols — convert into compact text blocks.

The "description" must follow EXACTLY this structure:

Responsibilities:
<Concise list of responsibilities in 3 to 6 compressed lines. Each line should start with a verb.>

Requirements:
<Skill Set: programming languages, frameworks, databases, tools, cloud, DevOps, etc.>
<Technologies: list>
<Experience Level: junior/mid/senior>
<Category: Software Development / DevOps / Data / QA / UIUX / Cloud>

Return ONLY a valid JSON with:
{{
  "job_title": "...",
  "description": "Responsibilities: ... Requirements: ..."
}}

Job:
{job_text}

JSON:
"""
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that parses job descriptions into structured JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response (handle markdown code blocks if present)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        job_dict = json.loads(response_text)
        return job_dict
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response_text}")
        raise


def parse_jobs_with_openai(job_text: str) -> list[dict]:
    """Use OpenAI API to parse job descriptions into JSON format."""
    
    # Split into individual jobs
    individual_jobs = split_jobs(job_text)
    print(f"Found {len(individual_jobs)} job descriptions to parse")
    
    jobs_list = []
    for i, job in enumerate(individual_jobs, 1):
        print(f"  Parsing job {i}/{len(individual_jobs)}...")
        try:
            parsed_job = parse_single_job_with_openai(job)
            jobs_list.append(parsed_job)
        except Exception as e:
            print(f"  Failed to parse job {i}: {e}")
    
    return jobs_list


def save_to_json(data: list[dict], output_path: str) -> None:
    """Save the parsed job data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {len(data)} jobs to {output_path}")


def main():
    # File paths
    input_file = "./output/Job_Descriptions.txt"
    output_file = "./output/job_descriptions.json"
    
    print("Reading job descriptions...")
    job_text = read_job_descriptions(input_file)
    
    print("Parsing jobs with OpenAI API...")
    jobs = parse_jobs_with_openai(job_text)
    
    print(f"Parsed {len(jobs)} job descriptions")
    
    # Save to JSON file
    save_to_json(jobs, output_file)
    
    # Print preview
    print("\nPreview of parsed jobs:")
    for job in jobs:
        print(f"- {job.get('job_title', 'Unknown')}")


if __name__ == "__main__":
    main()
