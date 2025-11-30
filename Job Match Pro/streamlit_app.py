"""Streamlit web app for PDF text extraction and cleaning."""

import os

import streamlit as st

from text import smart_extract_pdf, clean_text

# Define folders
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "pdfs")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output")

def main():
    st.set_page_config(page_title="PDF Text Extractor", layout="wide")

    st.title("📄 PDF Text Extractor (PyMuPDF + pdfplumber)")

    st.markdown("Upload a PDF; the cleaned extracted text will be shown below.")

    # Ensure folders exist
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Keep only the uploader control and the cleaned output on the frontend
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"]) 

    if uploaded_file is not None:
        # Save the uploaded file to pdfs folder
        tmp_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Automatically extract and clean text immediately after upload
        try:
            with st.spinner("Extracting and cleaning text..."):
                extracted = smart_extract_pdf(tmp_path)
                cleaned = clean_text(extracted)

            if not extracted:
                st.warning("Extraction returned no text; try a higher threshold or a different extractor")
            else:
                st.success("✅ Extraction complete!")
                
                # Show cleaned text prominently on front page
                st.subheader("📝 Cleaned Extracted Text")
                st.text_area("Cleaned Text", value=cleaned, height=400, label_visibility="collapsed")

                # Save output to output folder
                output_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name.replace(".pdf", ".txt"))
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                st.info(f"💾 Saved to: {output_path}")

        except Exception as e:
            st.error(f"Error while extracting: {e}")

    else:
        st.info('👆 Drag a PDF here or click to select one')


if __name__ == "__main__":
    main()
