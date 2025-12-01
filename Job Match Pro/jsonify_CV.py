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


def read_cv_content(file_path: str) -> str:
    """Read the CV content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_cv_with_openai(cv_text: str) -> dict:
    """Use OpenAI API to parse CV into JSON format."""
    
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
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that parses CVs/resumes into structured JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            print(f"Could not find JSON in response: {response_text[:500]}")
            raise ValueError("No JSON found in response")
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response_text[:500]}")
        raise


def save_cv_json(cv_data: list[dict], output_path: str) -> None:
    """Save the parsed CV data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cv_data, f, indent=2, ensure_ascii=False)
    print(f"CV JSON saved to: {output_path}")


def main():
    # File paths
    input_file = "./output/CV_contents.txt"
    output_file = "./output/CV_parsed.json"
    
    print("Reading CV content...")
    cv_content = read_cv_content(input_file)
    
    print("Parsing CV with OpenAI API...")
    cv_data = [parse_cv_with_openai(cv_content)]
    
    print("Saving parsed CV to JSON...")
    save_cv_json(cv_data, output_file)
    
    print("\nParsed CV:")
    print(f"  Description preview: {cv_data[0].get('description', '')[:200]}...")


if __name__ == "__main__":
    main()
