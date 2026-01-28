import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import MODEL_INFO, SAMPLE_METRICS

st.set_page_config(page_title="Home - Glaucoma Detection", page_icon="🏠", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_css():
    is_dark = st.session_state.theme == 'dark'
    bg_main = '#0f172a' if is_dark else '#ffffff'
    bg_card = '#1e293b' if is_dark else '#f8fafc'
    text_primary = '#f1f5f9' if is_dark else '#1e293b'
    text_secondary = '#94a3b8' if is_dark else '#64748b'
    border_color = '#475569' if is_dark else '#e2e8f0'
    accent = '#3b82f6'
    
    return f"""
    <style>
        .stApp {{ background-color: {bg_main}; }}
        .page-title {{ color: {accent}; font-size: 2.2rem; font-weight: 700; text-align: center; padding: 1rem 0; border-bottom: 2px solid {border_color}; margin-bottom: 2rem; }}
        .card {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
        .stat-box {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 10px; padding: 1rem; text-align: center; }}
        .stat-number {{ font-size: 2rem; font-weight: 700; color: {accent}; }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

st.markdown('<div class="page-title">🏠 Glaucoma Detection System</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📖 About Glaucoma", "📊 Dataset", "🤖 Models", "📈 Performance"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Glaucoma?
        
        Glaucoma is a group of eye conditions that damage the optic nerve, 
        often caused by abnormally high eye pressure. It's the leading cause 
        of irreversible blindness worldwide.
        
        **Key Facts:**
        - Affects 80+ million people globally
        - Often called "silent thief of sight"
        - Early detection is crucial
        - Vision loss cannot be recovered
        
        **Types:**
        1. Open-Angle (most common)
        2. Angle-Closure (emergency)
        3. Normal-Tension
        4. Secondary
        """)
    
    with col2:
        st.markdown("""
        ### Risk Factors
        
        - Age over 60
        - Family history
        - High eye pressure
        - Diabetes
        - High myopia
        - Previous eye injury
        
        ### Symptoms
        
        **Early:** Often none
        
        **Advanced:**
        - Blind spots
        - Tunnel vision
        - Eye pain
        - Blurred vision
        - Halos around lights
        """)

with tab2:
    st.markdown("### Dataset Structure")
    
    st.code("""
Fundus_Scanes_Sorted/
├── Train/
│   ├── glaucoma/
│   └── normal/
└── Validation/
    ├── glaucoma/
    └── normal/
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Classification", "Binary")
    with col2:
        st.metric("Classes", "2")
    with col3:
        st.metric("Format", "JPG/PNG")
    with col4:
        st.metric("Input Size", "224/299")

with tab3:
    st.markdown("### Available Models (11 Total)")
    
    all_models = list(MODEL_INFO.keys())
    
    st.markdown(f"""
    **All models available in dropdown:**
    {', '.join(all_models)}
    """)
    
    model_df = pd.DataFrame([
        {'Model': name, 'Params': info['params'], 'Input': info['input_size']}
        for name, info in MODEL_INFO.items()
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    selected = st.selectbox("View model details:", all_models)
    if selected:
        info = MODEL_INFO[selected]
        st.info(f"**{selected}:** {info['description']}")

with tab4:
    st.markdown("### Model Performance")
    st.caption("Sample metrics - train models for actual performance")
    
    metric = st.selectbox("Metric:", ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
    
    data = pd.DataFrame([
        {'Model': name, metric: m[metric]}
        for name, m in SAMPLE_METRICS.items()
    ])
    
    fig = px.bar(data, x='Model', y=metric, color=metric, color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    radar_models = st.multiselect("Radar comparison:", all_models, default=['ResNet50', 'DenseNet121', 'GlaucoNet'])
    
    if radar_models:
        cats = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
        fig = go.Figure()
        for m in radar_models:
            vals = [SAMPLE_METRICS[m][c] for c in cats] + [SAMPLE_METRICS[m][cats[0]]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill='toself', name=m))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
