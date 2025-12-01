"""Home page for Job Match Pro application."""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Job Match Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .home-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin: 2rem 0 1rem 0;
    }
    .home-subtitle {
        font-size: 1.3rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        height: 100%;
    }
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .feature-description {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for home page."""
    # Header
    st.markdown('<h1 class="home-header">💼 Job Match Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="home-subtitle">AI-Powered Resume to Job Description Matching System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    st.markdown("### 🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📄 JD Upload</div>
            <div class="feature-description">
                Upload and manage job descriptions. The system automatically extracts, parses, and stores them in the vector database for matching.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 CV Match</div>
            <div class="feature-description">
                Upload your CV to find matching job opportunities. Get AI-powered analysis with match scores, skills analysis, and detailed insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Analytics</div>
            <div class="feature-description">
                View comprehensive match analysis with interactive charts, skill comparisons, and AI-generated verdicts for each job match.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 📖 Quick Start Guide")
    
    st.markdown("""
    **1. Upload Job Descriptions**
    - Navigate to **JD Upload** page from the sidebar
    - Click "Add JD" and upload job description PDFs
    - The system will automatically process and store them
    
    **2. Match Your CV**
    - Go to **CV Match** page from the sidebar
    - Upload your CV/Resume as a PDF
    - Click "Find Matching Jobs" to get AI-powered matches
    
    **3. Review Results**
    - View match scores and detailed analysis
    - See skills matched and missing
    - Read AI-generated verdicts for each match
    """)
    
    st.markdown("---")
    
    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **AI & ML:**
        - OpenAI GPT-4o-mini
        - Text Embeddings (text-embedding-3-small)
        - Vector Similarity Search
        """)
    
    with tech_col2:
        st.markdown("""
        **Database:**
        - Pinecone Vector Database
        - JSON Metadata Storage
        """)
    
    with tech_col3:
        st.markdown("""
        **Framework:**
        - Streamlit Web App
        - Plotly Visualizations
        - PDF Processing (PyMuPDF, pdfplumber)
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown(
        '<p style="text-align: center; color: #6B7280; font-size: 0.9rem;">Built with ❤️ using Streamlit | Powered by OpenAI & Pinecone</p>',
        unsafe_allow_html=True
    )

# Call main function
main()
