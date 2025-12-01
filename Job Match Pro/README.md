# PDF Text Extractor

A small script to extract text from PDFs using PyMuPDF (primary extractor) and pdfplumber (fallback).

Quick start

1. Install dependencies

```powershell
pip install -r requirements.txt
```

Note: If you're developing or running tests, install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

2. Place your PDF in the project folder `C:\Users\Admin\Desktop\HR`.

3. Run the script:

```powershell
python text.py your_resume.pdf
```

Run via Streamlit (web UI)

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run Streamlit app:

```powershell
streamlit run streamlit_app.py
```

3. Visit the local URL printed by Streamlit (usually http://localhost:8501) and use the upload UI to try a PDF.

The Streamlit app calls the same `smart_extract_pdf` function as `text.py`.
```

If you run the script with no arguments, it will list all PDFs in the working directory and prompt you to choose one interactively:

```powershell
python text.py
```

Useful flags

- `--out` or `-o`: specify the output txt file name
- `--threshold` or `-t`: set the minimum number of characters from the primary extractor required; fallback is used if extraction returns less

Examples

```powershell
# run and auto-save to your_resume.txt
python text.py your_resume.pdf

# run and output to a specified file
python text.py your_resume.pdf -o my_file.txt

# run interactive selection and always use primary extractor if > 500 characters
python text.py -t 500
```

Design notes

- The script attempts to extract text with PyMuPDF first. If the extracted length is below the threshold (default 200), it tries `pdfplumber` as a fallback.
- If PyMuPDF or pdfplumber fails, the script prints helpful error messages.

Let me know which PDF you'd like me to run the extractor on (if you upload it into the workspace), or I can improve the CLI or add batch-processing.

Environment variables / Cloud APIs
---------------------------------
- Set `OPENAI_API_KEY` in a `.env` file for OpenAI API access (used for embeddings and text generation).
- Set `PC_API_KEY` in a `.env` file for Pinecone vector database access.
- The `python-dotenv` package is used to load environment variables.
- Get your OpenAI API key from: https://platform.openai.com/api-keys
- Get your Pinecone API key from: https://app.pinecone.io/

Notes
-----
- `requirements.txt` contains the detected runtime packages used by this project.
- If a package requires a specific version for your environment, pin versions in `requirements.txt` or use a virtual environment.