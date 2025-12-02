# Job Match Pro 💼

An AI-powered job matching application that uses semantic search and LLM evaluation to match CVs against job descriptions.

## Features

- **📤 Upload Multiple JDs**: Upload job description PDFs and automatically process them with AI
- **📄 CV Matching**: Upload your CV to find the best matching jobs
- **🤖 AI-Powered Analysis**: Uses Google Gemini for intelligent parsing and evaluation
- **🔍 Semantic Search**: Pinecone vector database for accurate similarity matching
- **📊 Detailed Results**: Comprehensive match scores, skills analysis, and AI explanations

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Upload JD  │───▶│  JSONify    │───▶│   Embed     │───▶ Pinecone
│   (PDF)     │    │  (Gemini)   │    │ (text-004)  │    (Upsert)
└─────────────┘    └─────────────┘    └─────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Upload CV  │───▶│  JSONify    │───▶│   Query     │───▶ Pinecone
│   (PDF)     │    │  (Gemini)   │    │ (text-004)  │    (Search)
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │  Evaluate   │───▶ Results
                                      │  (Gemini)   │
                                      └─────────────┘
```

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PC_API_KEY=your_pinecone_api_key
```

### 3. Run the Application

```powershell
streamlit run main_app.py
```

### 4. Open in Browser

Visit http://localhost:8501

## Usage Flow

1. **Upload JDs**: Start by uploading one or more job description PDFs
2. **Review JDs**: View uploaded JDs in the sidebar, delete if needed
3. **Upload CV**: Once JDs exist, upload your CV for matching
4. **View Results**: See detailed match analysis with scores and recommendations

## Project Structure

```
Job Match Pro/
├── main_app.py              # Main Streamlit application
├── services/
│   ├── pdf_service.py       # PDF text extraction
│   ├── jsonify_service.py   # AI-powered JSON conversion
│   ├── embedding_service.py # Vector embedding generation
│   ├── pinecone_service.py  # Vector database operations
│   └── evaluator_service.py # AI match evaluation
├── utils/
│   └── storage.py           # Local JD storage management
├── data/
│   └── jd_store.json        # Persistent JD storage
├── pages/
│   └── results.py           # Standalone results page
├── text.py                  # PDF extraction utilities
└── requirements.txt         # Python dependencies
```

## API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| Google Gemini | JSONification & Evaluation | [Google AI Studio](https://aistudio.google.com/) |
| Pinecone | Vector Database | [Pinecone Console](https://www.pinecone.io/) |

## Legacy Scripts

The following scripts are available for standalone use:

### PDF Text Extraction

```powershell
python text.py your_resume.pdf
```

### Original Streamlit App (PDF extraction only)

```powershell
streamlit run streamlit_app.py
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
- If you use `generate_embeddings.py` or `CVPARSER/CVpar_ai.py`, you should set `GOOGLE_API_KEY` in a `.env` file (see `CVPARSER/.env` or create your own). The `python-dotenv` package is used to load that environment variable.

Notes
-----
- `requirements.txt` contains the detected runtime packages used by this project.
- If a package requires a specific version for your environment, pin versions in `requirements.txt` or use a virtual environment.