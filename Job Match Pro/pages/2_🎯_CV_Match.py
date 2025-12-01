"""Streamlit page for CV matching with job descriptions."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_matcher import process_cv_and_match

# Page config
st.set_page_config(
    page_title="CV Match",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .cv-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .match-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
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
    .verdict-excellent {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
    }
    .verdict-good {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
    }
    .verdict-moderate {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
    }
    .verdict-poor {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def get_verdict_class(match_score: str) -> str:
    """Get CSS class for verdict based on match score."""
    try:
        score = float(str(match_score).strip('%'))
        if score >= 85:
            return "verdict-excellent"
        elif score >= 70:
            return "verdict-good"
        elif score >= 50:
            return "verdict-moderate"
        else:
            return "verdict-poor"
    except:
        return "verdict-moderate"


def get_verdict_text(match_score: str) -> str:
    """Get verdict text based on match score."""
    try:
        score = float(str(match_score).strip('%'))
        if score >= 85:
            return "Excellent Match"
        elif score >= 70:
            return "Good Match"
        elif score >= 50:
            return "Moderate Match"
        else:
            return "Poor Match"
    except:
        return "Moderate Match"


def display_match_results(results: dict):
    """Display match results in a formatted way."""
    matches = results.get("matches", [])
    
    if not matches:
        st.info("No matching jobs found. Try uploading more job descriptions.")
        return
    
    # Summary metrics
    st.markdown("### 📊 Match Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_matches = len(matches)
    excellent = len([m for m in matches if get_verdict_text(m.get("match_score", "0")) == "Excellent Match"])
    good = len([m for m in matches if get_verdict_text(m.get("match_score", "0")) == "Good Match"])
    avg_score = sum([float(str(m.get("match_score", "0")).strip('%')) for m in matches if str(m.get("match_score", "0")).strip('%').replace('.', '').isdigit()]) / total_matches if total_matches > 0 else 0
    
    with col1:
        st.metric("Total Matches", total_matches)
    with col2:
        st.metric("Excellent/Good", excellent + good, delta=f"{((excellent+good)/total_matches*100):.0f}%" if total_matches > 0 else "0%")
    with col3:
        st.metric("Avg Match Score", f"{avg_score:.1f}%")
    with col4:
        category_matches = len([m for m in matches if m.get("category_match") == "Yes"])
        st.metric("Category Matches", category_matches)
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 🎯 Match Score Distribution")
        verdict_counts = {}
        for match in matches:
            verdict = get_verdict_text(match.get("match_score", "0"))
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        if verdict_counts:
            colors_map = {
                "Excellent Match": "#10B981",
                "Good Match": "#3B82F6",
                "Moderate Match": "#F59E0B",
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
        st.markdown("### 📈 Match Scores by Job")
        df = pd.DataFrame([{
            "Job Title": m.get("job_title", "N/A"),
            "Match Score": float(str(m.get("match_score", "0")).strip('%')) if str(m.get("match_score", "0")).strip('%').replace('.', '').isdigit() else 0
        } for m in matches])
        
        df = df.sort_values("Match Score", ascending=False)
        
        color_map = {
            "Excellent Match": "#10B981",
            "Good Match": "#3B82F6",
            "Moderate Match": "#F59E0B",
            "Poor Match": "#EF4444"
        }
        df["color"] = df["Match Score"].apply(
            lambda x: "#10B981" if x >= 85 else "#3B82F6" if x >= 70 else "#F59E0B" if x >= 50 else "#EF4444"
        )
        
        fig_bar = go.Figure(data=[go.Bar(
            x=df["Job Title"],
            y=df["Match Score"],
            marker_color=df["color"],
            text=df["Match Score"].round(1).astype(str) + "%",
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
    
    # Detailed match cards
    st.markdown("### 📋 Detailed Match Analysis")
    
    for idx, match in enumerate(matches, 1):
        with st.expander(
            f"🏢 {idx}. {match.get('job_title', 'N/A')} — {get_verdict_text(match.get('match_score', '0'))}",
            expanded=(idx == 1)  # Expand first match by default
        ):
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("#### 📊 Scores")
                
                similarity = match.get("similarity", 0)
                match_score = match.get("match_score", "0%")
                
                # Gauge chart for match score
                try:
                    score_num = float(str(match_score).strip('%'))
                except:
                    score_num = 0
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_num,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Match Score", 'font': {'size': 16}},
                    number={'suffix': "%", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#3B82F6"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 40], 'color': '#FEE2E2'},
                            {'range': [40, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#D1FAE5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': score_num
                        }
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(t=30, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Match score and category
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric("Match Score", match_score)
                with score_col2:
                    cat_match = match.get("category_match", "No")
                    st.metric("Category Match", "✅ Yes" if cat_match == "Yes" else "❌ No")
            
            with col_right:
                st.markdown("#### 🛠️ Skills Analysis")
                
                # Matched skills
                st.markdown("**✅ Skills Matched:**")
                matched = match.get("skills_matched", [])
                if matched:
                    skills_html = " ".join([f'<span class="skill-matched">{skill}</span>' for skill in matched])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("None")
                
                st.markdown("")
                
                # Missing skills
                st.markdown("**❌ Skills Missing:**")
                missing = match.get("skills_missing", [])
                if missing:
                    skills_html = " ".join([f'<span class="skill-missing">{skill}</span>' for skill in missing])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("None")
            
            # Verdict/Explanation
            verdict = match.get("verdict", "")
            if verdict:
                st.markdown("#### 💡 AI Analysis")
                st.info(verdict)
            
            st.markdown("---")
    
    # Comparison table
    st.markdown("### 📑 Quick Comparison Table")
    
    df_table = pd.DataFrame([{
        "Job Title": m.get("job_title", "N/A"),
        "Match Score": m.get("match_score", "N/A"),
        "Similarity": f"{m.get('similarity', 0)}%" if isinstance(m.get('similarity'), (int, float)) else str(m.get('similarity', 'N/A')),
        "Category Match": "✅" if m.get("category_match") == "Yes" else "❌",
        "Skills Matched": len(m.get("skills_matched", [])),
        "Skills Missing": len(m.get("skills_missing", [])),
        "Verdict": get_verdict_text(m.get("match_score", "0"))
    } for m in matches])
    
    st.dataframe(df_table, use_container_width=True, hide_index=True)


def main():
    """Main function for CV match page."""
    # Header
    st.markdown('<h1 class="cv-header">🎯 CV Job Matcher</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;">Upload your CV to find matching job opportunities</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CV Upload section
    st.markdown("### 📄 Upload Your CV")
    uploaded_file = st.file_uploader(
        "Upload CV/Resume (PDF)",
        type=['pdf'],
        help="Upload a PDF file containing your CV/Resume"
    )
    
    # Number of matches to retrieve
    top_k = st.sidebar.slider(
        "Number of matches to retrieve",
        min_value=5,
        max_value=20,
        value=10,
        help="Select how many top matching jobs to retrieve and analyze"
    )
    
    if uploaded_file is not None:
        if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
            # Process CV and find matches
            with st.spinner("Processing CV and finding matches..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extracting text
                status_text.text("📄 Extracting text from CV...")
                progress_bar.progress(10)
                
                # Step 2: Parsing CV
                status_text.text("🤖 Parsing CV with AI...")
                progress_bar.progress(30)
                
                # Step 3: Generating embedding
                status_text.text("🔢 Generating embedding...")
                progress_bar.progress(50)
                
                # Step 4: Searching Pinecone
                status_text.text("🔍 Searching for matching jobs...")
                progress_bar.progress(70)
                
                # Step 5: Evaluating matches
                status_text.text("📊 Evaluating matches with AI...")
                progress_bar.progress(90)
                
                # Process the CV
                result = process_cv_and_match(uploaded_file, top_k=top_k)
                
                progress_bar.progress(100)
                
                if result["success"]:
                    status_text.success("✅ CV processed and matches found!")
                    st.markdown("---")
                    
                    # Display parsed CV summary
                    parsed_cv = result.get("parsed_cv", {})
                    if parsed_cv:
                        with st.expander("📝 Parsed CV Summary", expanded=False):
                            st.text(parsed_cv.get("description", "N/A"))
                    
                    # Display match results
                    display_match_results(result)
                    
                else:
                    status_text.error(f"❌ {result.get('message', 'Failed to process CV')}")
                    if result.get("error"):
                        st.error(f"Error: {result.get('error')}")
    
    else:
        st.info("👆 Upload a PDF CV/Resume to find matching job opportunities")
        
        # Show instructions
        with st.expander("ℹ️ How it works", expanded=True):
            st.markdown("""
            **CV Matching Process:**
            
            1. **Upload CV**: Upload your CV/Resume as a PDF file
            2. **Extract Text**: The system extracts and cleans text from your CV
            3. **Parse with AI**: AI analyzes and structures your CV data
            4. **Generate Embedding**: Creates a vector representation of your CV
            5. **Search Jobs**: Finds matching job descriptions from the database
            6. **AI Evaluation**: Analyzes each match and provides:
               - Match score (0-100%)
               - Skills matched/missing
               - Category match
               - Detailed verdict and reasoning
            
            **Tips for best results:**
            - Ensure your CV is clear and well-formatted
            - Include relevant technical skills and experience
            - Make sure job descriptions are uploaded in the JD Upload page
            """)


if __name__ == "__main__":
    main()

