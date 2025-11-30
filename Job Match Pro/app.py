import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Job Match Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Improve visibility and sizing for the main header across themes */
    :root { --main-header-color: #1E3A8A; }
    @media (prefers-color-scheme: dark) {
        :root { --main-header-color: #FFFFFF; }
    }
    .main-header {
        font-size: 2.8rem !important; /* ~45px, larger and more visible */
        font-weight: 800 !important;
        color: var(--main-header-color) !important;
        text-align: center;
        margin: 0.35rem 0 0.6rem 0;
        line-height: 1.15 !important;
        letter-spacing: 0.2px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    @media (prefers-color-scheme: dark) {
        .sub-header { color: #CBD5E1 !important; }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
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

# Load data
@st.cache_data
def load_data():
    with open("./output/evaluation_results.json", "r") as f:
        data = json.load(f)
    
    # Post-process data to fix issues
    for d in data:
        # Fix verdict if it's a long explanation
        current_verdict = d.get("verdict", "")
        match_score = d.get("match_score", 0)
        
        # If verdict is long, it's likely an explanation
        if len(current_verdict) > 20:
            # Move to explanation structure if needed
            if "explanation" not in d or not isinstance(d["explanation"], dict):
                d["explanation"] = {
                    "match_accuracy": current_verdict,
                    "score_interpretation": f"Match Score: {match_score}"
                }
            
            # Calculate short verdict based on match_score
            try:
                score = float(str(match_score).strip('%'))
            except (ValueError, TypeError):
                score = 0
                
            if score >= 85:
                d["verdict"] = "Excellent Match"
            elif score >= 70:
                d["verdict"] = "Good Match"
            elif score >= 50:
                d["verdict"] = "Moderate"
            else:
                d["verdict"] = "Poor Match"
    
    # Sort by match_score descending (best matches first)
    data.sort(key=lambda x: float(str(x.get("match_score", 0)).strip('%')), reverse=True)
    
    return data

data = load_data()

# Header
st.markdown('<h1 class="main-header">💼 Job Match Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Resume to Job Description Matching Analysis</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Filters")

# Get unique verdicts from data to ensure we capture all variations
unique_verdicts = sorted(list(set([d.get("verdict", "N/A") for d in data])))
# Define standard order if possible
standard_order = ["Excellent Match", "Good Match", "Moderate", "Poor Match", "Poor"]
# Sort unique_verdicts based on standard_order
unique_verdicts.sort(key=lambda x: standard_order.index(x) if x in standard_order else 999)

verdict_filter = st.sidebar.multiselect(
    "Filter by Verdict",
    options=unique_verdicts,
    default=unique_verdicts
)

# Filter data and sort by match_score descending
filtered_data = [d for d in data if d.get("verdict") in verdict_filter]
filtered_data.sort(key=lambda x: float(str(x.get("match_score", 0)).strip('%')), reverse=True)

# Summary metrics
st.markdown("### 📊 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

total_jobs = len(data)
excellent = len([d for d in data if d.get("verdict") == "Excellent Match"])
good = len([d for d in data if d.get("verdict") == "Good Match"])
moderate = len([d for d in data if d.get("verdict") == "Moderate"])
# Handle both "Poor" and "Poor Match"
poor = len([d for d in data if d.get("verdict") in ["Poor", "Poor Match"]])

with col1:
    st.metric("Total Jobs Analyzed", total_jobs, delta=None)
with col2:
    st.metric("Excellent/Good Matches", excellent + good, delta=f"{((excellent+good)/total_jobs*100):.0f}%" if total_jobs > 0 else "0%")
with col3:
    avg_similarity = sum([d.get("similarity", 0) for d in data]) / total_jobs if total_jobs > 0 else 0
    st.metric("Avg Similarity Score", f"{avg_similarity:.1f}%")
with col4:
    category_matches = len([d for d in data if d.get("category_match") == "Yes"])
    st.metric("Category Matches", category_matches)

st.markdown("---")

# Charts row
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("### 🎯 Match Verdict Distribution")
    # Group Poor and Poor Match for the chart
    verdict_counts = {
        "Excellent Match": excellent, 
        "Good Match": good, 
        "Moderate": moderate, 
        "Poor Match": poor
    }
    # Filter out zero counts to make chart cleaner
    verdict_counts = {k: v for k, v in verdict_counts.items() if v > 0}
    
    colors_map = {
        "Excellent Match": "#10B981", 
        "Good Match": "#3B82F6", 
        "Moderate": "#F59E0B", 
        "Poor Match": "#EF4444",
        "Poor": "#EF4444"
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
    st.markdown("### 📈 Similarity Scores by Job")
    df = pd.DataFrame(data)
    df["similarity_pct"] = df["similarity"]  # Already 0-100 scale
    
    # Color based on verdict
    color_map = {
        "Excellent Match": "#10B981",
        "Good Match": "#3B82F6", 
        "Moderate": "#F59E0B",
        "Poor": "#EF4444",
        "Poor Match": "#EF4444"
    }
    df["color"] = df["verdict"].map(color_map).fillna("#888888")
    
    # Sort by match_score for chart display (best first)
    df["match_score_num"] = df["match_score"].apply(lambda x: float(str(x).strip('%')))
    df = df.sort_values("match_score_num", ascending=False)
    
    fig_bar = go.Figure(data=[go.Bar(
        x=df["job_title"],
        y=df["similarity_pct"],
        marker_color=df["color"],
        text=df["similarity_pct"].round(1).astype(str) + "%",
        textposition="outside"
    )])
    fig_bar.update_layout(
        xaxis_title="Job Title",
        yaxis_title="Similarity Score (%)",
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        yaxis_range=[0, 100],
        xaxis={'categoryorder': 'array', 'categoryarray': df["job_title"].tolist()}
    )
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Detailed job cards
st.markdown("### 📋 Detailed Job Analysis")

for job in filtered_data:
    with st.expander(f"🏢 {job.get('job_title', 'N/A')} — {job.get('verdict', 'N/A')}", expanded=False):
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Scores
            st.markdown("#### 📊 Scores")
            
            similarity = job.get("similarity", 0)
            match_score = job.get("match_score", "0%")
            
            # Gauge chart for similarity
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=similarity * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Similarity Score", 'font': {'size': 16}},
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
                        'value': similarity * 100
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
                cat_match = job.get("category_match", "No")
                st.metric("Category Match", "✅ Yes" if cat_match == "Yes" else "❌ No")
        
        with col_right:
            # Skills
            st.markdown("#### 🛠️ Skills Analysis")
            
            # Matched skills
            st.markdown("**✅ Skills Matched:**")
            matched = job.get("skills_matched", [])
            if matched:
                skills_html = " ".join([f'<span class="skill-matched">{skill}</span>' for skill in matched])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.write("None")
            
            st.markdown("")
            
            # Missing skills
            st.markdown("**❌ Skills Missing:**")
            missing = job.get("skills_missing", [])
            if missing:
                skills_html = " ".join([f'<span class="skill-missing">{skill}</span>' for skill in missing])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.write("None")
            
            # Skills comparison chart
            if matched or missing:
                fig_skills = go.Figure(data=[
                    go.Bar(name='Matched', x=['Skills'], y=[len(matched)], marker_color='#10B981'),
                    go.Bar(name='Missing', x=['Skills'], y=[len(missing)], marker_color='#EF4444')
                ])
                fig_skills.update_layout(
                    barmode='group',
                    height=150,
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_skills, use_container_width=True)
        
        # Explanation section
        explanation = job.get("explanation", {})
        if explanation:
            st.markdown("#### 💡 AI Analysis")
            
            if "match_accuracy" in explanation:
                with st.container():
                    st.markdown("**Match Accuracy:**")
                    st.info(explanation["match_accuracy"])
            
            if "score_interpretation" in explanation:
                with st.container():
                    st.markdown("**Score Interpretation:**")
                    st.info(explanation["score_interpretation"])

# Comparison table
st.markdown("---")
st.markdown("### 📑 Quick Comparison Table")

df_table = pd.DataFrame([{
    "Job Title": d.get("job_title", "N/A"),
    "Similarity": f"{d.get('similarity', 0):.1%}",
    "Match Score": d.get("match_score", "N/A"),
    "Category Match": "✅" if d.get("category_match") == "Yes" else "❌",
    "Skills Matched": len(d.get("skills_matched", [])),
    "Skills Missing": len(d.get("skills_missing", [])),
    "Verdict": d.get("verdict", "N/A")
} for d in filtered_data])

# Style the dataframe
def color_verdict(val):
    if val == "Excellent Match":
        return 'background-color: #D1FAE5; color: #065F46'
    elif val == "Good Match":
        return 'background-color: #DBEAFE; color: #1E40AF'
    elif val == "Moderate":
        return 'background-color: #FEF3C7; color: #92400E'
    elif val in ["Poor", "Poor Match"]:
        return 'background-color: #FEE2E2; color: #991B1B'
    return ''

styled_df = df_table.style.applymap(color_verdict, subset=['Verdict'])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #6B7280; font-size: 0.9rem;">Built with love❤️ using Streamlit | Powered by AI Resume Matching</p>',
    unsafe_allow_html=True
)
