"""Job Match Pro - Results Page (Standalone).

This page can be used for multi-page Streamlit apps.
For the unified single-file app, see main_app.py.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Job Match Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-matched {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
    }
    .skill-missing {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def load_results():
    """Load evaluation results from session state or file."""
    # Try session state first
    if "evaluation_results" in st.session_state and st.session_state.evaluation_results:
        return st.session_state.evaluation_results
    
    # Fallback to file
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "evaluation_results.json"
    )
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    
    return None


def main():
    st.markdown('<h1 class="main-header">📊 Job Match Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Resume to Job Description Matching Analysis</p>', 
                unsafe_allow_html=True)
    
    data = load_results()
    
    if not data:
        st.warning("⚠️ No evaluation results available.")
        st.info("Please run the main application to generate results.")
        return
    
    # Process data to normalize verdicts
    for d in data:
        current_verdict = d.get("verdict", "")
        match_score = d.get("match_score", 0)
        
        if isinstance(match_score, str):
            try:
                match_score = float(match_score.strip('%'))
            except:
                match_score = 0
        
        # Normalize verdict if needed
        if len(str(current_verdict)) > 30:
            if match_score >= 85:
                d["verdict"] = "Excellent Match"
            elif match_score >= 70:
                d["verdict"] = "Good Match"
            elif match_score >= 50:
                d["verdict"] = "Moderate"
            else:
                d["verdict"] = "Poor Match"
    
    # Sort by match score
    data.sort(key=lambda x: float(str(x.get("match_score", 0)).strip('%') or 0), reverse=True)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    unique_verdicts = list(set([d.get("verdict", "N/A") for d in data]))
    verdict_filter = st.sidebar.multiselect(
        "Filter by Verdict",
        options=unique_verdicts,
        default=unique_verdicts
    )
    
    filtered_data = [d for d in data if d.get("verdict") in verdict_filter]
    
    # Summary metrics
    st.markdown("### 📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_jobs = len(data)
    excellent = len([d for d in data if "excellent" in str(d.get("verdict", "")).lower()])
    good = len([d for d in data if "good" in str(d.get("verdict", "")).lower()])
    
    with col1:
        st.metric("Total Jobs", total_jobs)
    with col2:
        st.metric("Top Matches", excellent + good)
    with col3:
        avg_score = sum([float(str(d.get("match_score", 0)).strip('%') or 0) for d in data]) / max(total_jobs, 1)
        st.metric("Avg Score", f"{avg_score:.1f}%")
    with col4:
        category_matches = len([d for d in data if d.get("category_match") == "Yes"])
        st.metric("Category Match", category_matches)
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 🎯 Verdict Distribution")
        verdict_counts = {}
        for d in data:
            v = d.get("verdict", "Unknown")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        colors = {
            "Excellent Match": "#10B981",
            "Good Match": "#3B82F6",
            "Moderate": "#F59E0B",
            "Poor Match": "#EF4444"
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(verdict_counts.keys()),
            values=list(verdict_counts.values()),
            hole=0.4,
            marker_colors=[colors.get(k, "#888") for k in verdict_counts.keys()]
        )])
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.markdown("### 📊 Scores by Job")
        df = pd.DataFrame(data)
        df["score_num"] = df["match_score"].apply(lambda x: float(str(x).strip('%') or 0))
        
        fig = go.Figure(data=[go.Bar(
            x=df["job_title"],
            y=df["score_num"],
            marker_color="#3B82F6"
        )])
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Job cards
    st.markdown("### 📋 Detailed Analysis")
    
    for job in filtered_data:
        with st.expander(f"🏢 {job.get('job_title')} — {job.get('verdict')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Match Score", f"{job.get('match_score')}%")
                st.metric("Similarity", f"{job.get('similarity')}%")
                st.metric("Category", "✅" if job.get("category_match") == "Yes" else "❌")
            
            with col2:
                st.markdown("**Skills Matched:**")
                matched = job.get("skills_matched", [])
                if matched:
                    st.markdown(" ".join([f'`{s}`' for s in matched]))
                
                st.markdown("**Skills Missing:**")
                missing = job.get("skills_missing", [])
                if missing:
                    st.markdown(" ".join([f'`{s}`' for s in missing]))
            
            explanation = job.get("explanation", {})
            if explanation and isinstance(explanation, dict):
                st.info(explanation.get("match_accuracy", ""))
    
    # Download
    st.markdown("---")
    st.download_button(
        "📥 Download Results JSON",
        data=json.dumps(data, indent=2),
        file_name="job_match_results.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()
