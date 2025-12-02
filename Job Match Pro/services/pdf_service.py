"""Document extraction service for Job Match Pro.

Supports PDF, TXT, DOC, and DOCX files.
"""

import os
import sys
import tempfile

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from existing text.py module (ignore linter warning - runtime import)
from text import smart_extract_pdf, clean_text


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # Fallback: read as binary and decode with errors ignored
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX support. "
            "Install it with: pip install python-docx"
        )
    
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return '\n'.join(paragraphs)


def extract_text_from_doc(file_path: str) -> str:
    """Extract text from a DOC file using antiword or fallback methods."""
    import subprocess
    
    # Try antiword first (if installed)
    try:
        result = subprocess.run(
            ['antiword', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Try using python-docx (works for some .doc files)
    try:
        return extract_text_from_docx(file_path)
    except Exception:
        pass
    
    # Try textract if available
    try:
        import textract
        return textract.process(file_path).decode('utf-8')
    except ImportError:
        pass
    except Exception:
        pass
    
    raise RuntimeError(
        "Could not extract text from .doc file. "
        "Please convert to .docx or .pdf format, or install antiword/textract."
    )


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract and clean text from an uploaded file.
    
    Supports: PDF, TXT, DOC, DOCX
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Cleaned extracted text string
    """
    filename = uploaded_file.name.lower()
    
    # Determine file extension
    if filename.endswith('.pdf'):
        suffix = '.pdf'
        extractor = smart_extract_pdf
    elif filename.endswith('.txt'):
        suffix = '.txt'
        extractor = extract_text_from_txt
    elif filename.endswith('.docx'):
        suffix = '.docx'
        extractor = extract_text_from_docx
    elif filename.endswith('.doc'):
        suffix = '.doc'
        extractor = extract_text_from_doc
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported: PDF, TXT, DOC, DOCX")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Extract text
        extracted_text = extractor(tmp_path)
        
        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)
        
        return cleaned_text
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {filename}: {e}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Alias for backward compatibility
extract_text_from_pdf = extract_text_from_file


def extract_text_from_path(file_path: str) -> str:
    """
    Extract and clean text from a file path.
    
    Supports: PDF, TXT, DOC, DOCX
    
    Args:
        file_path: Path to the file
        
    Returns:
        Cleaned extracted text string
    """
    file_path_lower = file_path.lower()
    
    try:
        if file_path_lower.endswith('.pdf'):
            extracted_text = smart_extract_pdf(file_path)
        elif file_path_lower.endswith('.txt'):
            extracted_text = extract_text_from_txt(file_path)
        elif file_path_lower.endswith('.docx'):
            extracted_text = extract_text_from_docx(file_path)
        elif file_path_lower.endswith('.doc'):
            extracted_text = extract_text_from_doc(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        cleaned_text = clean_text(extracted_text)
        return cleaned_text
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from file: {e}")
