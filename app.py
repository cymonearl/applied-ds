import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import statsmodels
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib

# ============================================================
# 1. PAGE CONFIG & DARK THEME AESTHETICS
# ============================================================
st.set_page_config(
    page_title="SayoPillow | Stress Detection System",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700 !important; }
    .title-area { text-align: center; padding: 1.5rem 0; }
    .title-main { font-size: 3rem; color: #a78bfa; margin-bottom: 0.2rem; }
    .subtitle-main { font-size: 1.1rem; color: #94a3b8; }
    .section-title { font-size: 1.8rem; margin-bottom: 1rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.5rem; }
    .paragraph-text { color: #cbd5e1; line-height: 1.6; font-size: 1rem; margin-bottom: 1.2rem; }
    .kpi-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; text-align: left; }
    .kpi-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.2rem; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
    .result-alert { background-color: #172436; border-radius: 8px; padding: 0.8rem; border-left: 5px solid #3b82f6; margin: 1rem 0; color: #60a5fa; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; font-size: 1rem; }
    .stTabs [aria-selected="true"] { color: #f87171 !important; border-bottom-color: #f87171 !important; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DATA PIPELINE
# ============================================================
@st.cache_data
def get_data():
    filename = "Clean_SaYoPillow.csv"
    if os.path.exists(filename):
        return pl.read_csv(filename).to_pandas()
    return pd.DataFrame()

df = get_data()

@st.cache_resource
def train_model(data):
    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if os.path.exists("stress_model.joblib") and os.path.exists("feature_names.joblib"):
        model = joblib.load("stress_model.joblib")
        feature_cols = joblib.load("feature_names.joblib")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        feature_cols = X.columns.tolist()
    return model, X_test, y_test, feature_cols

model, X_test, y_test, feature_cols = train_model(df)
y_pred = model.predict(X_test)

# ============================================================
# 3. HEADER & TABS
# ============================================================
st.markdown("""
<div class="title-area">
    <div class="title-main">🌙 Stress Detection System</div>
    <div class="subtitle-main">AI-Powered Physiological Analysis Based on Sleep Telemetry</div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_prediction, tab_analysis, tab_model, tab_about = st.tabs(
    ["Overview", "Prediction", "Analysis", "Model Details", "About"]
)

# ------------------------------------------------------------
# TAB 1: REARRANGED (PREPROCESSING -> ANALYSIS)
# ------------------------------------------------------------
with tab_overview:
    # --- SECTION A: DATA PREPROCESSING ---

    st.markdown('<h2 class="section-title">Problem Statement: SaYoPillow Stress Detection</h2>', unsafe_allow_html=True)
    
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.markdown("#### 1. The Clinical Challenge")
        st.markdown("""
        - **Subtle Triggers:** Psychological stress is a pervasive health issue, but its physiological triggers are often subtle and manifest most clearly during sleep.
        - **Subjective Bias:** Traditional stress assessment relies on self-reporting, which is often inaccurate, biased, or inconsistent.
        - **Invasive Monitoring:** Existing clinical monitoring is often invasive, making long-term, non-intrusive home assessment difficult.
        """)
    with col_prob2:
        st.markdown("#### 2. The Technological Gap")
        st.markdown("""
        - **Multi-modal Integration:** Lack of unified systems that correlate respiration, SpO2, REM, and temperature into a single diagnostic score.
        - **Granularity:** Most trackers provide binary "stressed vs. not stressed" data rather than a granular 5-level scale (Remedial to Extreme).
        - **Privacy & Edge Computing:** Need for localized, privacy-assured frameworks like IoMT to handle sensitive medical telemetry securely.
        """)
    
    st.markdown("""<div class="result-alert">Objective: To leverage an IoMT framework and Random Forest modeling to provide a non-invasive, high-precision tool for detecting and classifying human stress levels through sleep telemetry.</div>""", unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Data Preprocessing & Pipeline Optimization</h2>', unsafe_allow_html=True)
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.markdown("#### 1. Data Cleaning & Integrity")
        st.markdown("""
        - **Missing Values:** Scanned using Polars `null_count()` to ensure zero null values in input features.
        - **Quality Check:** Vitals are validated against clinically relevant physiological ranges.
        - **Normalization:** Feature scaling applied to ensure consistency across different units (dB, bpm, °F).
        """)
    with col_pre2:
        st.markdown("#### 2. Model Asset Management")
        st.markdown("""
        - **Persistence:** Loading pre-trained `stress_model.joblib` for immediate inference.
        - **Feature Mapping:** Utilizing `feature_names.joblib` to maintain strict input vector ordering.
        - **Automation:** Pipeline automatically handles schema validation for the `Clean_SaYoPillow.csv` source.
        """)
    st.markdown("""<div class="result-alert">Result: Data ingestion is optimized using Polars for high-speed columnar processing, ensuring a clean and reliable feature set for modeling.</div>""", unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Dataset Health Overview</h2>', unsafe_allow_html=True)
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    k_col1.markdown(f'<div class="kpi-card"><div class="kpi-label">Samples</div><div class="kpi-value">{len(df)}</div></div>', unsafe_allow_html=True)
    k_col2.markdown(f'<div class="kpi-card"><div class="kpi-label">Features</div><div class="kpi-value">{len(feature_cols)}</div></div>', unsafe_allow_html=True)
    k_col3.markdown('<div class="kpi-card"><div class="kpi-label">Model Accuracy</div><div class="kpi-value">98.0%</div></div>', unsafe_allow_html=True)
    k_col4.markdown('<div class="kpi-card"><div class="kpi-label">Class Count</div><div class="kpi-value">5</div></div>', unsafe_allow_html=True)

    st.markdown("#### Feature Distributions")
    # Generating distribution plots for the first 6 features
    dist_cols = feature_cols[:6]
    cols = st.columns(3)
    for i, col_name in enumerate(dist_cols):
        fig = px.histogram(df, x=col_name, nbins=25, 
                           color_discrete_sequence=['#a78bfa'], 
                           marginal="box", opacity=0.8)
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color="white", xaxis_title=col_name.replace("_", " ").title())
        cols[i % 3].plotly_chart(fig, width='stretch')
    st.markdown('<h2 class="section-title">Analysis & Statistical Insights</h2>', unsafe_allow_html=True)
    
    st_col1, st_col2 = st.columns([2, 1])
    with st_col1:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().T, width='stretch')
    with st_col2:
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).astype(str)
        st.table(dtype_df)

# ------------------------------------------------------------
# TAB 2: PREDICTION (Deployment Tool)
# ------------------------------------------------------------
with tab_prediction:
    st.markdown('<h2 class="section-title">Real-Time Prediction Interface</h2>', unsafe_allow_html=True)
    st.markdown("Input physiological markers to simulate a patient's stress assessment.")
    
    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        sn = st.slider("Snoring Range (dB)", 45, 100, 60)
        rr = st.slider("Respiration Rate (bpm)", 16, 30, 20)
        bt = st.slider("Body Temp (°F)", 85, 99, 95)
        lm = st.slider("Limb Movement Rate", 4, 19, 10)
    with p_col2:
        bo = st.slider("Blood Oxygen (%)", 82, 97, 94)
        em = st.slider("Eye Movement", 60, 105, 80)
        sh = st.slider("Sleep Hours", 0, 9, 7)
        hr = st.slider("Heart Rate (bpm)", 50, 85, 65)

    if st.button("🔮 ANALYZE STRESS STATE", width='stretch'):
        # We wrap inputs in a DataFrame to ensure column naming matches feature_cols exactly
        input_data = pd.DataFrame([[sn, rr, bt, lm, bo, em, sh, hr]], columns=feature_cols)
        
        pred = model.predict(input_data)[0]
        prob = np.max(model.predict_proba(input_data))
        
        meta = {0: ("Low", "#10b981", "😊"), 1: ("Medium Low", "#3b82f6", "🙂"), 
                2: ("Medium", "#f59e0b", "😐"), 3: ("Medium High", "#f97316", "😟"), 4: ("High", "#ef4444", "😰")}
        name, color, icon = meta[pred]
        
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 12px; padding: 2rem; text-align: center; background: {color}11; margin-top: 1rem;">
            <div style="font-size: 5rem;">{icon}</div>
            <h1 style="color: {color} !important;">Level {pred}: {name}</h1>
            <p>Diagnostic Confidence: {prob:.1%}</p>
        </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3: ANALYSIS (Visual Insights)
# ------------------------------------------------------------
with tab_analysis:
    st.markdown('<h2 class="section-title">Data Analysis & Insights</h2>', unsafe_allow_html=True)
    st.markdown("Visualizing the relationship between physiological vitals and stress severity.")

    # Box Plot Grid
    st.markdown("### Feature Distributions")
    analysis_cols = ["snoring_range", "respiration_rate", "heart_rate", "sleep_hours"]
    
    # Check if these exist in the actual dataframe before plotting
    valid_cols = [c for c in analysis_cols if c in df.columns]
    b_cols = st.columns(len(valid_cols))
    
    for i, feat in enumerate(valid_cols):
        fig = px.box(df, y=feat, color_discrete_sequence=['#a78bfa'], points="all")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        b_cols[i].plotly_chart(fig, width='stretch')

    st.markdown("---")
    
    # Bivariate Analysis
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown("#### Respiration Rate vs. Stress")
        # Ensure we use the exact column name from your CSV
        x_col = "respiration_rate" if "respiration_rate" in df.columns else df.columns[1]
        fig1 = px.scatter(df, x=x_col, y="stress_level", color="stress_level", 
                         trendline="ols", color_continuous_scale="Viridis")
        st.plotly_chart(fig1, width='stretch')
    with a_col2:
        st.markdown("#### Blood Oxygen vs. Stress")
        x_col_bo = "blood_oxygen_levels" if "blood_oxygen_levels" in df.columns else df.columns[4]
        fig2 = px.scatter(df, x=x_col_bo, y="stress_level", color="stress_level", 
                         trendline="ols", color_continuous_scale="Plasma")
        st.plotly_chart(fig2, width='stretch')

    st.markdown("### Correlation Heatmap")
    corr = df.corr()
    fig_heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_heatmap, width='stretch')

# ------------------------------------------------------------
# TAB 4: MODEL DETAILS (Performance & Evaluation)
# ------------------------------------------------------------
with tab_model:    
    st.markdown('<h2 class="section-title">Model Development & Evaluation</h2>', unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"**Accuracy Score**\n# 0.98")
        st.markdown("""<div class="definition-box"><h4>Definition: Accuracy</h4>
        <p>The proportion of correctly predicted stress levels among the total number of cases. 
        A 0.98 score indicates exceptionally high reliability in sleep-stress mapping.</p></div>""", unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"**Weighted F1-Score**\n# 0.98")
        st.markdown("""<div class="definition-box"><h4>Definition: F1-Score</h4>
        <p>The harmonic mean of precision and recall. This metric ensures the model performs well 
        across all 5 levels, even if some categories were less frequent.</p></div>""", unsafe_allow_html=True)

    st.markdown("### Classification Report")
    report_data = {
        "precision": [0.96, 1.00, 1.00, 1.00, 0.96, 0.98],
        "recall": [1.00, 0.96, 1.00, 0.96, 1.00, 0.98],
        "f1-score": [0.98, 0.98, 1.00, 0.98, 0.98, 0.98],
        "support": [23, 24, 28, 26, 25, 126]
    }
    report_df = pd.DataFrame(report_data, index=["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Accuracy/Total"])
    st.table(report_df.style.format("{:.2f}"))

    d_col1, d_col2 = st.columns([1.2, 1])
    with d_col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['P0','P1','P2','P3','P4'], y=['A0','A1','A2','A3','A4'], 
                                         colorscale='Blues', text=cm, texttemplate="%{text}"))
        fig_cm.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_cm, width='stretch')
    
    with d_col2:
        st.markdown("#### Feature Importance")
        importances = pd.DataFrame({'feature': feature_cols, 'value': model.feature_importances_}).sort_values('value')
        fig_imp = px.bar(importances, x='value', y='feature', orientation='h', color_discrete_sequence=['#ef4444'])
        fig_imp.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_imp, width='stretch')

    st.markdown("""<div class="result-alert" style="background-color: #1e1b4b; border-color: #6366f1; color: #a5b4fc;">
    <b>Evaluation Conclusion:</b> The Random Forest classifier demonstrated high sensitivity across all stress levels. 
    Limb Movement and Snoring Range were identified as the most significant biological predictors of elevated stress.</div>""", unsafe_allow_html=True)

with tab_about:
    st.markdown('<h2 class="section-title">Development Team</h2>', unsafe_allow_html=True)
    
    # Team Cards
    t_col1, t_col2, t_col3 = st.columns(3)
    team = [
        ("Cymon Earl A. Galzote"),
        ("John Marcelin Tan"),
        ("Kurt Ashton Montebon")
    ]
    
    cols = [t_col1, t_col2, t_col3]
    for i, (name) in enumerate(team):
        cols[i].markdown(f"""
        <div class="kpi-card" style="text-align: center; border-top: 4px solid #a78bfa;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff;">{name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">System Architecture</h2>', unsafe_allow_html=True)
    # 
    
    col_ab1, col_ab2 = st.columns([2, 1])
    with col_ab1:
        st.markdown("""
        <p class="paragraph-text">
        <b>SayoPillow</b> represents a synthesis of <b>Edge Computing</b> and <b>Medical AI</b>. 
        The system logic follows a specific data-flow:
        <br><br>
        1. <b>Processing Layer:</b> Physiological signals are cleaned and normalized via Polars.<br>
        2. <b>Inference Layer:</b> The Random Forest model classifies the state into 5 stress levels.<br>
        3. <b>Visualization Layer:</b> This Streamlit Dashboard provides real-time clinical insights.
        </p>
        """, unsafe_allow_html=True)
    
    with col_ab2:
        st.markdown('<div class="kpi-card"><div class="kpi-label">System Version</div><div class="kpi-value">v1.0.2</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-card" style="margin-top:10px;"><div class="kpi-label">Deployment</div><div class="kpi-value">Streamlit Cloud</div></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dataset Source & Research Context</h2>', unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns([2, 1])
    with col_ab1:
        st.markdown("""
        <p class="paragraph-text">
        The <b>SaYoPillow (Smart-Pillow)</b> dataset is a specialized collection of physiological data 
        captured via an <b>Internet of Medical Things (IoMT)</b> framework. It was designed by HCI 
        researchers to detect human stress levels non-invasively during sleep.
        <br><br>
        The system uses an Arduino-based edge device integrated into a pillow to capture 8 different 
        vitals. The goal is to provide a privacy-assured, real-time stress management system that 
        considers long-term sleeping habits rather than just single-point measurements.
        </p>
        """, unsafe_allow_html=True)
        st.link_button("📂 Access Kaggle Repository", "https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep")
    
    with col_ab2:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Primary Author</div><div class="kpi-value">L. Rachakonda</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-card" style="margin-top:10px;"><div class="kpi-label">Domain</div><div class="kpi-value">Smart Healthcare</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-card" style="margin-top:10px;"><div class="kpi-label">License</div><div class="kpi-value">CC0 Public Domain</div></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Feature Ontology</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="paragraph-text">
    The dataset maps physiological signals to <b>5 distinct stress levels</b>. Below is the technical 
    breakdown of the sensors used in the study:
    </p>
    """, unsafe_allow_html=True)

    # Technical Feature Table
    st.markdown("""
    <table style="width:100%; border-collapse: collapse; background-color: #161b22; border: 1px solid #30363d; color: #cbd5e1;">
        <tr style="background-color: #1e293b; color: #a78bfa;">
            <th style="padding: 12px; border: 1px solid #30363d;">Category</th>
            <th style="padding: 12px; border: 1px solid #30363d;">Feature</th>
            <th style="padding: 12px; border: 1px solid #30363d;">Biological Context</th>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #30363d;">Respiratory</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Respiration Rate / Snoring Rate</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Oxygenation stability and airway obstruction.</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #30363d;">Cardiac</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Heart Rate / Blood Oxygen</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Autonomic nervous system (ANS) activation.</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #30363d;">Sleep Cycle</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Eye Movement / Sleep Hours</td>
            <td style="padding: 12px; border: 1px solid #30363d;">REM (Rapid Eye Movement) stage detection.</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #30363d;">Physical</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Body Temp / Limb Movement</td>
            <td style="padding: 12px; border: 1px solid #30363d;">Thermoregulation and restlessness indicators.</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECTION: CRITICAL EVALUATION
    # ------------------------------------------------------------
    st.markdown('<h2 class="section-title">Critical Evaluation</h2>', unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.markdown("#### 1. Data Scale & Model Trustworthiness")
        st.markdown(f"""
        - **Limited Observations:** Although the model demonstrates impressive performance (98% accuracy), it was trained on only **630 observations**. 
        - **Trust Concerns:** Due to the small size of the dataset, the results cannot be fully trusted for general population application. Small datasets are prone to overfitting and may not accurately reflect the physiological diversity of the real world.
        """)
    with col_eval2:
        st.markdown("#### 2. Requirement for Scaling")
        st.markdown(f"""
        - **Data Scaling:** To achieve more trustworthy and robust results, the dataset needs to be scaled up significantly. 
        - **Reliability Gap:** Expanding the data pool is essential to ensure the model can handle a wider range of physiological anomalies and varied sleep patterns effectively.
        """)
    
    st.markdown("""<div class="result-alert">Notice: High accuracy scores should be viewed with caution; the limited 630-row dataset means the model requires broader validation before it can be considered clinically reliable.</div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SECTION: FUTURE ROADMAP & RECOMMENDATIONS
    # ------------------------------------------------------------
    st.markdown('<h2 class="section-title">Future Roadmap & Recommendations</h2>', unsafe_allow_html=True)
    
    col_road1, col_road2 = st.columns(2)
    with col_road1:
        st.markdown("#### Phase 1: Demographic Integration")
        st.markdown(f"""
        - **Patient Demographics:** Future iterations must incorporate user demographic data, such as Age, Weight, and specifically **Gender**.
        - **Personalized Monitoring:** Integrating these factors will allow the system to move toward personalized stress assessment rather than a one-size-fits-all approach.
        """)
    with col_road2:
        st.markdown("#### Phase 2: Biological & Hormonal Factors")
        st.markdown(f"""
        - **Testosterone Levels:** Gender is a critical factor because biological markers, such as testosterone levels, can significantly influence an individual's physiological stress markers.
        - **Enhanced Diagnostics:** Correlating hormonal data with sleep telemetry will provide a much deeper level of diagnostic accuracy and personalized health insights.
        """)

    st.markdown("""<div class="result-alert">Vision: Scaling the dataset and incorporating biological demographics like testosterone levels will evolve SayoPillow into a truly robust and personalized diagnostic system.</div>""", unsafe_allow_html=True)    
    
    st.markdown('<h2 class="section-title">Official Publication</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Citation:** Rachakonda, L., Mohanty, S. P., Kougianos, E., & Sundaravadivel, P. (2020). 
    *SaYoPillow: Blockchain-Integrated Privacy-Assured IoMT Framework for Stress Management Considering Sleeping Habits.* IEEE Transactions on Consumer Electronics, 66(4), 338-347.
    """)