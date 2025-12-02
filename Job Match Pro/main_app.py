"""Job Match Pro - Main Streamlit Application.

A fully connected application for matching CVs against Job Descriptions
using AI-powered semantic search and evaluation.
"""

import os
import sys

k_size = 5  # Number of top matches to retrieve

# Ensure the project root is in the path for imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import time
from datetime import datetime

# Import services
from services import (
    extract_text_from_pdf,
    jsonify_jd,
    jsonify_cv,
    generate_embedding,
    upsert_jd,
    query_with_cv,
    delete_jd_by_id,
    evaluate_matches,
)

# Import utilities
from utils import (
    load_jd_store,
    save_jd_to_store,
    delete_jd_from_store,
    get_jd_count,
)

# Page Configuration
st.set_page_config(
    page_title="Job Match Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* JD Card styling */
    .jd-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #667eea;
    }
    
    .jd-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E3A8A;
    }
    
    .jd-meta {
        font-size: 0.85rem;
        color: #6B7280;
    }
    
    /* Status badges */
    .status-success {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .status-warning {
        background: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .status-error {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Progress steps */
    .step-complete {
        color: #10B981;
    }
    
    .step-active {
        color: #3B82F6;
        font-weight: 600;
    }
    
    .step-pending {
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "cv_description" not in st.session_state:
    st.session_state.cv_description = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload_jd"
if "processing" not in st.session_state:
    st.session_state.processing = False


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.markdown("## 🧭 Navigation")
    
    jd_count = get_jd_count()
    
    # Navigation buttons
    if st.sidebar.button("📤 Upload JDs", use_container_width=True, 
                         type="primary" if st.session_state.current_page == "upload_jd" else "secondary"):
        st.session_state.current_page = "upload_jd"
        st.rerun()
    
    if st.sidebar.button("📄 Upload CV", use_container_width=True,
                         type="primary" if st.session_state.current_page == "upload_cv" else "secondary",
                         disabled=jd_count == 0):
        st.session_state.current_page = "upload_cv"
        st.rerun()
    
    if st.sidebar.button("📊 View Results", use_container_width=True,
                         type="primary" if st.session_state.current_page == "results" else "secondary",
                         disabled=st.session_state.evaluation_results is None):
        st.session_state.current_page = "results"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Status info
    st.sidebar.markdown("### 📈 Status")
    st.sidebar.metric("Job Descriptions", jd_count)
    
    if st.session_state.evaluation_results:
        st.sidebar.metric("Matches Found", len(st.session_state.evaluation_results))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="text-align: center; color: #6B7280; font-size: 0.8rem;">Job Match Pro v1.0</p>',
        unsafe_allow_html=True
    )


# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt", "doc", "docx"]


def render_jd_upload_page():
    """Render the JD upload page."""
    st.markdown('<h1 class="main-header">📤 Upload Job Descriptions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents containing job descriptions to match against</p>', 
                unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📁 Upload New JDs")
    st.caption("Supported formats: PDF, TXT, DOC, DOCX")
    uploaded_files = st.file_uploader(
        "Drag and drop JD files here or click to browse",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="jd_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if already processing
            file_key = f"processed_{uploaded_file.name}"
            if file_key in st.session_state:
                continue
            
            with st.expander(f"📄 Processing: {uploaded_file.name}", expanded=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract text
                    status_text.markdown("⏳ **Step 1/4:** Extracting text from document...")
                    jd_text = extract_text_from_pdf(uploaded_file)
                    progress_bar.progress(20)
                    
                    if not jd_text or len(jd_text.strip()) < 50:
                        st.error("❌ Could not extract sufficient text from document")
                        continue
                    
                    # Step 2: JSONify (returns list of JDs)
                    status_text.markdown("⏳ **Step 2/4:** Parsing JD(s) with AI...")
                    jd_list = jsonify_jd(jd_text, filename=uploaded_file.name)
                    progress_bar.progress(40)
                    
                    status_text.markdown(f"📋 Found **{len(jd_list)}** job description(s). Processing...")
                    
                    # Process each JD in the list
                    successful_jds = []
                    for idx, jd_json in enumerate(jd_list):
                        # Step 3: Generate embedding for each JD
                        status_text.markdown(f"⏳ **Step 3/4:** Generating embedding ({idx + 1}/{len(jd_list)})...")
                        embedding = generate_embedding(
                            jd_json["description"],
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        progress_bar.progress(40 + int(30 * (idx + 1) / len(jd_list)))
                        
                        # Step 4: Upsert to Pinecone
                        status_text.markdown(f"⏳ **Step 4/4:** Storing in vector database ({idx + 1}/{len(jd_list)})...")
                        vector_id = upsert_jd(jd_json, embedding)
                        jd_json["pinecone_vector_id"] = vector_id
                        progress_bar.progress(70 + int(25 * (idx + 1) / len(jd_list)))
                        
                        # Save to local store
                        save_jd_to_store(jd_json)
                        successful_jds.append(jd_json["job_title"])
                    
                    progress_bar.progress(100)
                    
                    # Show success message with all job titles
                    if len(successful_jds) == 1:
                        status_text.markdown(f"✅ **Complete!** Job Title: **{successful_jds[0]}**")
                    else:
                        job_titles = ", ".join([f"**{title}**" for title in successful_jds])
                        status_text.markdown(f"✅ **Complete!** Added {len(successful_jds)} jobs: {job_titles}")
                    
                    st.session_state[file_key] = True
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    progress_bar.progress(0)
        
        # Clear uploader after processing
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    
    # Display existing JDs
    st.markdown("### 📋 Previously Uploaded JDs")
    
    jds = load_jd_store()
    
    if not jds:
        st.info("👆 No JDs uploaded yet. Upload your first job description above!")
    else:
        st.markdown(f"*{len(jds)} job description(s) available*")
        
        for idx, jd in enumerate(jds):
            # Ensure JD has an id (for legacy data)
            jd_id = jd.get('id', f"legacy_{idx}")
            
            col1, col2, col3 = st.columns([5, 2, 1])
            
            with col1:
                created_at = jd.get('created_at', 'N/A')
                created_display = created_at[:10] if created_at and created_at != 'N/A' else 'N/A'
                st.markdown(f"""
                <div class="jd-card">
                    <div class="jd-title">🏢 {jd.get('job_title', 'Unknown')}</div>
                    <div class="jd-meta">📎 {jd.get('filename', 'N/A')} • 📅 {created_display}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                with st.expander("👁️ Preview"):
                    desc = jd.get('description', 'No description')
                    st.text(desc[:500] + "..." if len(desc) > 500 else desc)
            
            with col3:
                if st.button("🗑️", key=f"del_{jd_id}", help="Delete this JD"):
                    # Delete from Pinecone if vector_id exists
                    vector_id = jd.get("pinecone_vector_id")
                    if vector_id:
                        delete_jd_by_id(vector_id)
                    
                    # Delete from local store
                    if jd.get('id'):
                        delete_jd_from_store(jd["id"])
                    else:
                        # For legacy JDs without id, remove by index
                        jds_current = load_jd_store()
                        jds_current = [j for i, j in enumerate(jds_current) if i != idx]
                        import json
                        from utils.storage import JD_STORE_PATH
                        with open(JD_STORE_PATH, 'w', encoding='utf-8') as f:
                            json.dump(jds_current, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Deleted: {jd.get('job_title')}")
                    time.sleep(0.5)
                    st.rerun()


def render_cv_upload_page():
    """Render the CV upload page."""
    st.markdown('<h1 class="main-header">📄 Upload Your CV</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your resume to find matching job opportunities</p>', 
                unsafe_allow_html=True)
    
    jd_count = get_jd_count()
    
    # Check prerequisites
    if jd_count == 0:
        st.warning("⚠️ Please upload at least one Job Description first!")
        st.info("👉 Go to 'Upload JDs' page to add job descriptions.")
        if st.button("📤 Go to Upload JDs"):
            st.session_state.current_page = "upload_jd"
            st.rerun()
        return
    
    # Status info
    st.success(f"✅ {jd_count} Job Description(s) available for matching")
    
    st.markdown("---")
    
    # File uploader
    st.markdown("### 📎 Upload Your Resume")
    st.caption("Supported formats: PDF, TXT, DOC, DOCX")
    uploaded_cv = st.file_uploader(
        "Drag and drop your CV here or click to browse",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=False,
        key="cv_uploader"
    )
    
    if uploaded_cv:
        st.markdown("---")
        st.markdown("### 🔄 Processing Pipeline")
        
        # Create progress tracking
        steps = [
            ("📄 Extract text from CV", "extract"),
            ("🤖 Parse CV with AI", "jsonify"),
            ("🔢 Generate embedding", "embed"),
            ("🔍 Query matching jobs", "query"),
            ("📊 Evaluate matches with AI", "evaluate"),
        ]
        
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            step_status = st.empty()
            
            try:
                # Step 1: Extract text
                step_status.markdown("⏳ **Step 1/5:** Extracting text from document...")
                cv_text = extract_text_from_pdf(uploaded_cv)
                progress_bar.progress(20)
                
                if not cv_text or len(cv_text.strip()) < 50:
                    st.error("❌ Could not extract sufficient text from document")
                    return
                
                # Step 2: JSONify CV
                step_status.markdown("⏳ **Step 2/5:** Parsing CV with AI...")
                cv_json = jsonify_cv(cv_text)
                st.session_state.cv_description = cv_json.get("description", "")
                progress_bar.progress(40)
                
                # Step 3: Generate embedding
                step_status.markdown("⏳ **Step 3/5:** Generating embedding...")
                cv_embedding = generate_embedding(
                    cv_json["description"],
                    task_type="RETRIEVAL_QUERY"
                )
                progress_bar.progress(60)
                
                # Step 4: Query Pinecone
                step_status.markdown("⏳ **Step 4/5:** Finding matching jobs...")
                matches = query_with_cv(cv_embedding, top_k=min(jd_count, k_size))
                progress_bar.progress(80)
                
                if not matches:
                    st.warning("⚠️ No matches found in the database.")
                    return
                
                # Step 5: Evaluate matches
                step_status.markdown("⏳ **Step 5/5:** Evaluating matches with AI (this may take a moment)...")
                evaluation_results = evaluate_matches(cv_json["description"], matches)
                progress_bar.progress(100)
                
                # Store results in session state
                st.session_state.evaluation_results = evaluation_results
                
                step_status.markdown("✅ **Complete!** Redirecting to results...")
                
            except Exception as e:
                st.error(f"❌ Error processing CV: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Show summary and redirect
        with results_container:
            time.sleep(1)
            
            st.success(f"🎉 Found and evaluated {len(st.session_state.evaluation_results)} matching jobs!")
            
            # Quick preview
            if st.session_state.evaluation_results:
                best_match = st.session_state.evaluation_results[0]
                st.info(f"🏆 Best Match: **{best_match.get('job_title')}** - "
                       f"Match Score: **{best_match.get('match_score')}%**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Detailed Results", type="primary", use_container_width=True):
                    st.session_state.current_page = "results"
                    st.rerun()
            with col2:
                if st.button("📄 Upload Different CV", use_container_width=True):
                    st.session_state.evaluation_results = None
                    st.rerun()


def render_results_page():
    """Render the evaluation results page."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.markdown('<h1 class="main-header">📊 Job Match Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Resume to Job Description Matching Analysis</p>', 
                unsafe_allow_html=True)
    
    # Check if results exist
    if st.session_state.evaluation_results is None:
        st.warning("⚠️ No evaluation results available.")
        st.info("👉 Please upload a CV first to see matching results.")
        if st.button("📄 Go to Upload CV"):
            st.session_state.current_page = "upload_cv"
            st.rerun()
        return
    
    data = st.session_state.evaluation_results
    
    # Summary metrics
    st.markdown("### 📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_jobs = len(data)
    excellent = len([d for d in data if "excellent" in str(d.get("verdict", "")).lower()])
    good = len([d for d in data if "good" in str(d.get("verdict", "")).lower() and "excellent" not in str(d.get("verdict", "")).lower()])
    moderate = len([d for d in data if "moderate" in str(d.get("verdict", "")).lower()])
    poor = len([d for d in data if "poor" in str(d.get("verdict", "")).lower()])
    
    with col1:
        st.metric("Total Jobs Analyzed", total_jobs)
    with col2:
        st.metric("Excellent/Good Matches", excellent + good, 
                  delta=f"{((excellent+good)/total_jobs*100):.0f}%" if total_jobs > 0 else "0%")
    with col3:
        avg_score = sum([float(str(d.get("match_score", 0)).strip('%')) for d in data]) / total_jobs if total_jobs > 0 else 0
        st.metric("Avg Match Score", f"{avg_score:.1f}%")
    with col4:
        category_matches = len([d for d in data if d.get("category_match") == "Yes"])
        st.metric("Category Matches", category_matches)
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 🎯 Match Verdict Distribution")
        verdict_counts = {
            "Excellent Match": excellent,
            "Good Match": good,
            "Moderate": moderate,
            "Poor Match": poor
        }
        verdict_counts = {k: v for k, v in verdict_counts.items() if v > 0}
        
        if verdict_counts:
            colors_map = {
                "Excellent Match": "#10B981",
                "Good Match": "#3B82F6",
                "Moderate": "#F59E0B",
                "Poor Match": "#EF4444"
            }
            chart_colors = [colors_map.get(k, "#888888") for k in verdict_counts.keys()]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(verdict_counts.keys()),
                values=list(verdict_counts.values()),
                hole=0.4,
                marker_colors=chart_colors,
                textinfo='label+percent',
                textfont_size=12
            )])
            fig_pie.update_layout(
                showlegend=True,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        st.markdown("### 📊 Match Scores by Job")
        df = pd.DataFrame(data)
        
        if "match_score" in df.columns:
            df["match_score_num"] = df["match_score"].apply(
                lambda x: float(str(x).strip('%')) if x else 0
            )
            df = df.sort_values("match_score_num", ascending=False)
            
            # Color based on verdict
            def get_color(verdict):
                verdict_lower = str(verdict).lower()
                if "excellent" in verdict_lower:
                    return "#10B981"
                elif "good" in verdict_lower:
                    return "#3B82F6"
                elif "moderate" in verdict_lower:
                    return "#F59E0B"
                else:
                    return "#EF4444"
            
            df["color"] = df["verdict"].apply(get_color)
            
            fig_bar = go.Figure(data=[go.Bar(
                x=df["job_title"],
                y=df["match_score_num"],
                marker_color=df["color"],
                text=df["match_score_num"].round(1).astype(str) + "%",
                textposition="outside"
            )])
            fig_bar.update_layout(
                xaxis_title="Job Title",
                yaxis_title="Match Score (%)",
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis_range=[0, 100]
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed job cards
    st.markdown("### 📋 Detailed Job Analysis")
    
    for idx, job in enumerate(data):
        verdict = job.get("verdict", "N/A")
        verdict_lower = str(verdict).lower()
        
        if "excellent" in verdict_lower:
            icon = "🏆"
        elif "good" in verdict_lower:
            icon = "✅"
        elif "moderate" in verdict_lower:
            icon = "⚠️"
        else:
            icon = "❌"
        
        with st.expander(f"{icon} {job.get('job_title', 'N/A')} — {verdict}", expanded=False):
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("#### 📊 Scores")
                
                similarity = job.get("similarity", 0)
                if isinstance(similarity, str):
                    similarity = float(similarity.strip('%')) if similarity else 0
                
                match_score = job.get("match_score", 0)
                if isinstance(match_score, str):
                    match_score_num = float(match_score.strip('%')) if match_score else 0
                else:
                    match_score_num = float(match_score) if match_score else 0
                
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=match_score_num,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Match Score", 'font': {'size': 16}},
                    number={'suffix': "%", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [0, 50], 'color': '#FEE2E2'},
                            {'range': [50, 70], 'color': '#FEF3C7'},
                            {'range': [70, 85], 'color': '#DBEAFE'},
                            {'range': [85, 100], 'color': '#D1FAE5'}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(t=30, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{idx}")
                
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric("Similarity", f"{similarity}%")
                with score_col2:
                    cat_match = job.get("category_match", "No")
                    st.metric("Category Match", "✅ Yes" if cat_match == "Yes" else "❌ No")
            
            with col_right:
                st.markdown("#### 🛠️ Skills Analysis")
                
                # Matched skills
                st.markdown("**✅ Skills Matched:**")
                matched = job.get("skills_matched", [])
                if matched:
                    skills_html = " ".join([
                        f'<span style="background:#D1FAE5;color:#065F46;padding:0.25rem 0.5rem;border-radius:0.5rem;margin:0.1rem;display:inline-block;font-size:0.85rem;">{skill}</span>' 
                        for skill in matched
                    ])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("None identified")
                
                st.markdown("")
                
                # Missing skills
                st.markdown("**❌ Skills Missing:**")
                missing = job.get("skills_missing", [])
                if missing:
                    skills_html = " ".join([
                        f'<span style="background:#FEE2E2;color:#991B1B;padding:0.25rem 0.5rem;border-radius:0.5rem;margin:0.1rem;display:inline-block;font-size:0.85rem;">{skill}</span>' 
                        for skill in missing
                    ])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("None - all skills matched!")
            
            # Explanation
            explanation = job.get("explanation", {})
            if explanation:
                st.markdown("#### 💡 AI Analysis")
                
                if isinstance(explanation, dict):
                    if "match_accuracy" in explanation:
                        st.info(f"**Match Analysis:** {explanation['match_accuracy']}")
                    if "score_interpretation" in explanation:
                        st.success(f"**Interpretation:** {explanation['score_interpretation']}")
                else:
                    st.info(str(explanation))
    
    st.markdown("---")
    
    # Quick comparison table
    st.markdown("### 📑 Quick Comparison Table")
    
    df_table = pd.DataFrame([{
        "Job Title": d.get("job_title", "N/A"),
        "Match Score": f"{d.get('match_score', 0)}%",
        "Similarity": f"{d.get('similarity', 0)}%",
        "Category": "✅" if d.get("category_match") == "Yes" else "❌",
        "Skills ✓": len(d.get("skills_matched", [])),
        "Skills ✗": len(d.get("skills_missing", [])),
        "Verdict": d.get("verdict", "N/A")
    } for d in data])
    
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Upload New CV", use_container_width=True):
            st.session_state.evaluation_results = None
            st.session_state.current_page = "upload_cv"
            st.rerun()
    
    with col2:
        if st.button("📤 Add More JDs", use_container_width=True):
            st.session_state.current_page = "upload_jd"
            st.rerun()
    
    with col3:
        # Download results as JSON
        import json
        results_json = json.dumps(st.session_state.evaluation_results, indent=2)
        st.download_button(
            "📥 Download Results",
            data=results_json,
            file_name="job_match_results.json",
            mime="application/json",
            use_container_width=True
        )


def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()
    
    # Route to appropriate page
    if st.session_state.current_page == "upload_jd":
        render_jd_upload_page()
    elif st.session_state.current_page == "upload_cv":
        render_cv_upload_page()
    elif st.session_state.current_page == "results":
        render_results_page()
    else:
        render_jd_upload_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6B7280; font-size: 0.85rem;">'
        'Built with ❤️ using Streamlit | Powered by AI Resume Matching</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
