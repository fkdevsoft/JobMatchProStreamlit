"""Streamlit page for JD (Job Description) upload and management."""

import streamlit as st
from datetime import datetime
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jd_processor import (
    load_jd_metadata,
    process_jd_sync,
    delete_jd
)

# Page config
st.set_page_config(
    page_title="JD Upload",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .jd-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .jd-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
    }
    .delete-btn {
        background-color: #dc3545;
        color: white;
        border: none;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .modal-content {
        background: white;
        padding: 2rem;
        border-radius: 0.5rem;
        max-width: 600px;
        width: 90%;
        max-height: 90vh;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'show_add_modal' not in st.session_state:
        st.session_state.show_add_modal = False
    if 'jd_title' not in st.session_state:
        st.session_state.jd_title = ""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'delete_confirm_id' not in st.session_state:
        st.session_state.delete_confirm_id = None


def show_add_modal():
    """Display the Add JD modal/popup."""
    st.markdown("### ➕ Add New Job Description")
    st.markdown("---")
    
    # JD Title input
    jd_title = st.text_input(
        "Job Title",
        value=st.session_state.get('jd_title', ''),
        max_chars=150,
        help="Enter the job title (max 150 characters)"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload JD PDF",
        type=['pdf'],
        help="Upload a PDF file containing the job description"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            # Validation
            if not jd_title or not jd_title.strip():
                st.error("Please enter a job title")
                return
            
            if not uploaded_file:
                st.error("Please upload a PDF file")
                return
            
            if not uploaded_file.name.lower().endswith('.pdf'):
                st.error("Please upload a PDF file")
                return
            
            # Process JD synchronously
            with st.spinner("Processing JD..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Saving PDF
                status_text.text("📁 Saving PDF...")
                progress_bar.progress(10)
                
                # Step 2: Extracting text
                status_text.text("📄 Extracting text from PDF...")
                progress_bar.progress(30)
                
                # Step 3: Parsing with AI
                status_text.text("🤖 Parsing job description with OpenAI...")
                progress_bar.progress(50)
                
                # Step 4: Generating embedding
                status_text.text("🔢 Generating embedding...")
                progress_bar.progress(70)
                
                # Step 5: Storing in Pinecone
                status_text.text("💾 Storing in database...")
                progress_bar.progress(90)
                
                # Process the JD
                result = process_jd_sync(uploaded_file, jd_title.strip())
                
                progress_bar.progress(100)
                
                if result["success"]:
                    status_text.success("✅ JD processed and stored successfully!")
                    st.session_state.show_add_modal = False
                    st.session_state.jd_title = ""
                    st.session_state.uploaded_file = None
                    st.rerun()
                else:
                    status_text.error(f"❌ {result.get('message', 'Failed to process JD')}")
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_add_modal = False
            st.session_state.jd_title = ""
            st.session_state.uploaded_file = None
            st.rerun()


def display_jd_grid():
    """Display grid/table of uploaded JDs."""
    metadata = load_jd_metadata()
    
    if not metadata:
        st.info("📭 No job descriptions uploaded yet. Click 'Add JD' to upload your first JD.")
        return
    
    # Convert to DataFrame for display
    df_data = []
    for jd in metadata:
        df_data.append({
            "Upload Date": jd.get("upload_date", "N/A"),
            "Job Title": jd.get("title", "N/A"),
            "ID": jd.get("id", ""),
            "Pinecone ID": jd.get("pinecone_id", ""),
        })
    
    df = pd.DataFrame(df_data)
    
    # Display with delete buttons
    st.markdown("### 📋 Uploaded Job Descriptions")
    
    # Create a grid display
    for idx, jd in enumerate(metadata):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{jd.get('title', 'N/A')}**")
        
        with col2:
            st.write(f"📅 {jd.get('upload_date', 'N/A')}")
        
        with col3:
            # Delete button with confirmation
            delete_key = f"delete_{jd.get('id')}"
            jd_id = jd.get('id')
            
            if st.session_state.delete_confirm_id == jd_id:
                # Show confirmation
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Confirm", key=f"confirm_yes_{jd_id}", type="primary"):
                        result = delete_jd(jd_id)
                        st.session_state.delete_confirm_id = None
                        if result["success"]:
                            st.success("✅ JD deleted successfully")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('message', 'Failed to delete')}")
                with col_no:
                    if st.button("❌ Cancel", key=f"confirm_no_{jd_id}"):
                        st.session_state.delete_confirm_id = None
                        st.rerun()
            else:
                if st.button("🗑️ Delete", key=delete_key, type="secondary"):
                    st.session_state.delete_confirm_id = jd_id
                    st.rerun()
        
        st.markdown("---")


def main():
    """Main function for JD upload page."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="jd-header">📄 Job Description Upload</h1>', unsafe_allow_html=True)
    
    # Add button at top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕ Add JD", type="primary", use_container_width=True):
            st.session_state.show_add_modal = True
            st.rerun()
    
    st.markdown("---")
    
    # Show modal if needed
    if st.session_state.show_add_modal:
        with st.container():
            st.info("📝 Fill in the form below to add a new Job Description")
            show_add_modal()
        st.markdown("---")
    
    # Display JD grid
    display_jd_grid()


if __name__ == "__main__":
    main()

