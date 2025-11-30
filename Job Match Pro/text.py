"""PDF text extraction with smart extractor selection.

Provides functions to extract text from PDFs using PyMuPDF (primary)
or pdfplumber (fallback), with automatic selection based on extraction quality.
"""

import argparse
import os
import sys
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

# --- Begin inlined text_cleaner.py ---
import re
from typing import Dict, Tuple

# Precompiled regex patterns
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")
CID_RE = re.compile(r"\(cid:\d+\)")


def mask_entities(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    i = 0

    def mk_placeholder(name: str) -> str:
        nonlocal i
        i += 1
        return f"__{name.upper()}_{i}__"

    def mask_fn_email(m):
        key = mk_placeholder('email')
        mapping[key] = m.group(0)
        return key

    text = EMAIL_RE.sub(mask_fn_email, text)

    def mask_fn_url(m):
        key = mk_placeholder('url')
        mapping[key] = m.group(0)
        return key

    text = URL_RE.sub(mask_fn_url, text)

    def mask_fn_phone(m):
        key = mk_placeholder('phone')
        mapping[key] = m.group(0)
        return key

    text = PHONE_RE.sub(mask_fn_phone, text)

    return text, mapping


def unmask_entities(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def normalize_punctuation(text: str) -> str:
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'[""«»„‟]', '"', text)
    text = re.sub(r"[''‛']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'…', '...', text)
    text = re.sub(r'[ð§]', '', text)
    text = re.sub(r'\bI\s*\+', '+', text)
    text = re.sub(r'-\s*-\s*', '- ', text)
    return text


def remove_cid_noise(text: str) -> str:
    return CID_RE.sub('', text)


def remove_page_markers(text: str) -> str:
    text = re.sub(r'(?m)^\s*\d+\s*/\s*\d+\s*$', '', text)
    text = re.sub(r'(?i)(?m)^\s*page\s+\d+\s+of\s+\d+\s*$', '', text)
    return text


def fix_hyphenation(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'([a-z])\n\s*([a-z])', r'\1 \2', text)
    return text


def join_split_words(text: str) -> str:
    text = re.sub(r'\b(proven)(track)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(in)(fast)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Adept)(at)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b([A-Z]{2,})(models?|systems?|applications?|tools?|platforms?|solutions?)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcutting(task)\b', r'cutting \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(fatigue)(detection)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(secure)(database)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(enforce)(speed)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(speed)(limits)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(accident)(rates)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(AI)(based)\b', r'\1-\2', text)
    text = re.sub(r'\b(utilization)(insights)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(SQL)(backed)\b', r'\1-\2', text)
    text = re.sub(r'\b(transaction)(speed)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(detection)(accuracy)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(rates)(by)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(based)(route|gate|data)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(RFID)(based)\b', r'\1-\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(backed)(data)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(speed)(by)\b', r'\1 \2', text, flags=re.IGNORECASE)
    return text


def conservative_case_split(text: str) -> str:
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z][a-z])', ' ', text)
    text = re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', text)
    return text


def fix_bullets(text: str) -> str:
    text = re.sub(r'(?m)^[ \t]*[○•·§#]+\s*', '- ', text)
    text = re.sub(r'([^\n])\n- ', r'\1\n\n- ', text)
    return text


def restore_headings(text: str) -> str:
    sections = [
        'Summary', 'Education', 'Experience', 'Projects', 'Achievements', 'Skills',
        'Technical Skills', 'Publications', 'Certifications', 'Relevant Coursework', 
        'Projects & Achievements', 'Coding Profile'
    ]
    for sec in sections:
        text = re.sub(rf'(?m)^\s*{re.escape(sec)}\s*$', f"\n\n{sec}\n", text)
    return text


def fix_merged_lines(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z][a-z]+,\s*India)', r'\1\n\2', text)
    text = re.sub(r'(\w)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', r'\1\n\2', text)
    text = re.sub(r'(\d+)\s+K\+', r'\1K+', text)
    text = re.sub(r'(\d+)\s+K\+', r'\1K+', text)
    text = re.sub(r'(\d+)\s+(st|nd|rd|th)\b', r'\1\2', text)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'([a-z,])\n\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]{2,})\n\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n')]
    text = '\n'.join(lines)
    return text


def fix_double_bullets(text: str) -> str:
    text = re.sub(r'^-\s*-\s+', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'\n-\s*-\s+', '\n- ', text)
    return text


def fix_phone_prefix(text: str) -> str:
    text = re.sub(r'\bI\s*\+(\d)', r'+\1', text)
    return text


def clean_text(raw_text: str) -> str:
    if not raw_text:
        return raw_text

    text = raw_text
    text = remove_cid_noise(text)
    text = remove_page_markers(text)
    text = normalize_punctuation(text)
    text, mapping = mask_entities(text)
    text = fix_hyphenation(text)
    text = join_split_words(text)
    text = conservative_case_split(text)
    text = fix_merged_lines(text)
    text = fix_bullets(text)
    text = restore_headings(text)
    text = collapse_whitespace(text)
    text = unmask_entities(text, mapping)
    text = fix_double_bullets(text)
    text = fix_phone_prefix(text)
    return text.strip()

# --- End inlined text_cleaner.py ---

def extract_with_pymupdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        print(f"[PyMuPDF ERROR] {e}")
        return ""


def extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print(f"[pdfplumber ERROR] {e}")
        return ""


def smart_extract_pdf(pdf_path: str, threshold: int = 200) -> str:
    """Extract text from PDF, choosing best extractor based on output length.
    
    Args:
        pdf_path: Path to PDF file.
        threshold: Minimum characters for PyMuPDF to be considered successful.
    
    Returns:
        Extracted text string.
    """
    print("➡ Extracting text using PyMuPDF...")
    pymu_text = extract_with_pymupdf(pdf_path)

    # If PyMuPDF extracted enough text → use it
    if len(pymu_text) > threshold:
        print("✔ Using PyMuPDF result (primary extractor).")
        return pymu_text

    print("⚠ PyMuPDF extracted too little. Trying pdfplumber...")

    plumber_text = extract_with_pdfplumber(pdf_path)

    if len(plumber_text) > len(pymu_text):
        print("✔ Using pdfplumber result (fallback).")
        return plumber_text

    print("✔ Using PyMuPDF result (fallback was not better).")
    return pymu_text


def choose_pdf_interactively() -> Optional[str]:
    """If no file is provided on the command line, allow the user to choose one interactively.
    Lists PDFs found in current working directory and prompts the user to select one.
    Returns the chosen file path or None if none exists or user cancels.
    """
    pdfs = [f for f in os.listdir(os.getcwd()) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("No PDFs found in the current directory. Please place your PDF in this folder and rerun.")
        return None

    print("Found the following PDFs in the current directory:")
    for i, p in enumerate(pdfs, start=1):
        print(f"  {i}. {p}")
    choice = input("Enter the number of the PDF to use (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        return None
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(pdfs):
            print("Invalid selection")
            return None
        return pdfs[idx]
    except Exception:
        print("Invalid input")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from PDF using PyMuPDF and pdfplumber fallback')
    parser.add_argument('pdf', nargs='?', help='Path to the PDF file to extract')
    parser.add_argument('--out', '-o', help='Output .txt file (defaults to <pdf>.txt)')
    parser.add_argument('--threshold', '-t', type=int, default=200, help='Threshold for primary extractor to be considered sufficient')

    args = parser.parse_args()

    pdf_file = args.pdf
    if not pdf_file:
        pdf_file = choose_pdf_interactively()
        if not pdf_file:
            print('No PDF selected. Exiting.')
            sys.exit(1)

    if not os.path.exists(pdf_file):
        print("❌ File not found:", pdf_file)
        sys.exit(1)

    print("\n=== PDF TEXT EXTRACTION STARTED ===\n")

    extracted_text = smart_extract_pdf(pdf_file, threshold=args.threshold)
    cleaned_text = clean_text(extracted_text)

    print("\n=== EXTRACTION COMPLETE ===\n")
    print(cleaned_text)
    
    # Save to .txt file
    output_file = args.out if args.out else pdf_file.replace(".pdf", ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    saved_path = os.path.abspath(output_file)
    print(f"\n✔ Saved cleaned text to: {saved_path}")
