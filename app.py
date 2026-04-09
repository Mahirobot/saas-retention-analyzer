import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SaaS Growth Analyzer", layout="wide")

# --- DATA GENERATION (Cached so it doesn't reload constantly) ---
@st.cache_data
def load_synthetic_data():
    np.random.seed(42)
    n_users = 10000
    start_date = datetime(2025, 1, 1)

    user_ids = [f"USR-{i:05d}" for i in range(n_users)]
    signup_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_users)]
    test_group = np.random.choice(['Control', 'Variant'], size=n_users, p=[0.5, 0.5])

    feature_A_clicks = np.random.poisson(lam=3, size=n_users)
    feature_B_clicks = np.random.poisson(lam=1.5, size=n_users)

    onboarding_step_reached = []
    for group in test_group:
        step = 1
        if np.random.rand() < 0.90: step = 2  
        pass_step_3_prob = 0.60 if group == 'Variant' else 0.35
        if step == 2 and np.random.rand() < pass_step_3_prob: step = 3
        if step == 3 and np.random.rand() < 0.80: step = 4
        if step == 4 and np.random.rand() < 0.85: step = 5
        onboarding_step_reached.append(step)

    churned = []
    for i in range(n_users):
        churn_prob = 0.50 
        if onboarding_step_reached[i] == 5: churn_prob -= 0.20
        if feature_B_clicks[i] > 2: churn_prob -= 0.25
        if feature_A_clicks[i] > 5: churn_prob += 0.10
        churn_prob = max(0.05, min(0.95, churn_prob)) 
        churned.append(np.random.rand() < churn_prob)

    return pd.DataFrame({
        'user_id': user_ids,
        'test_group': test_group,
        'onboarding_step_reached': onboarding_step_reached,
        'feature_A_clicks': feature_A_clicks,
        'feature_B_clicks': feature_B_clicks,
        'churned': churned
    })

data = load_synthetic_data()

# --- HEADER ---
st.title("SaaS Growth & Retention Analyzer")
st.markdown("A data-driven deep dive into user onboarding behavior, A/B testing, and churn prediction.")

# --- THE BUSINESS PROBLEM & CONTEXT ---
st.markdown("### 🎯 The Business Problem")
st.write("""
Product teams often face three critical blind spots: **Activation Bottlenecks** (where do users drop off?), **Feature ROI** (does a new UI actually convert?), and **Churn Drivers** (what behaviors predict cancellation?). This dashboard answers these questions.
""")

with st.expander("📊 Read about the Data & Methodology"):
    st.write("""
    **The Data Context:**
    To demonstrate these analytics without exposing proprietary company data, this project uses a Python script to generate a highly realistic, synthetic dataset of **10,000 SaaS users**. 
    
    **Data Dictionary:**
    * `onboarding_step_reached`: Step 1 through 5 (representing the onboarding funnel).
    * `test_group`: 'Control' vs 'Variant' (simulating an A/B test for a new UI).
    * `feature_A_clicks` & `feature_B_clicks`: Simulated engagement metrics.
    * `churned`: Boolean indicating if the user ultimately canceled their subscription.
    """)

# Top-level metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Users Evaluated", f"{len(data):,}")
col2.metric("Overall Churn Rate", f"{data['churned'].mean() * 100:.1f}%")
col3.metric("Onboarding Completion", f"{(len(data[data['onboarding_step_reached'] == 5]) / len(data)) * 100:.1f}%")

st.divider()

# --- TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs(["📊 Funnel Optimization (EDA)", "🧪 A/B Testing", "📉 Churn Prediction (ML)"])

# --- TAB 1: FUNNEL ---
with tab1:
    st.header("Exploratory Data Analysis: Onboarding Funnel")
    st.write("Tracking user drop-off across the 5-step onboarding process.")
    
    steps = ['1: Signup', '2: Profile Setup', '3: Connect DB', '4: Build App', '5: Deploy']
    values = [len(data[data['onboarding_step_reached'] >= i]) for i in range(1, 6)]
    
    fig_funnel = go.Figure(go.Funnel(
        y=steps, x=values, textinfo="value+percent initial+percent previous"
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    st.info("**Product Bet:** The funnel reveals a massive 50%+ drop-off between Step 2 and Step 3 (Connect DB). We should prioritize simplifying the database integration UI to unclog this bottleneck.")

# --- TAB 2: A/B TESTING ---
with tab2:
    st.header("Hypothesis Testing: New Onboarding UI")
    st.write("We ran an A/B test changing the 'Connect DB' screen. Did the 'Variant' group complete onboarding at a higher rate?")
    
    data['completed_onboarding'] = data['onboarding_step_reached'] == 5
    contingency_table = pd.crosstab(data['test_group'], data['completed_onboarding'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    control_conv = contingency_table.loc['Control', True] / contingency_table.loc['Control'].sum() * 100
    variant_conv = contingency_table.loc['Variant', True] / contingency_table.loc['Variant'].sum() * 100
    
    c1, c2 = st.columns(2)
    c1.metric("Control Group Conversion", f"{control_conv:.1f}%")
    c2.metric("Variant Group Conversion", f"{variant_conv:.1f}%")
    
    st.write(f"**Statistical Significance (P-Value):** `{p_value:.5f}`")
    
    if p_value < 0.05:
        st.success("The Variant UI significantly improved onboarding completion. Recommend deploying Variant to 100% of traffic.")
    else:
        st.warning(" No statistically significant difference. We should form a new hypothesis.")

# --- TAB 3: CHURN ---
with tab3:
    st.header("Behavioral Modeling: Why are users churning?")
    st.write("Training a Decision Tree classifier to identify which features correlate most strongly with user retention.")
    
    ml_data = data.copy()
    le = LabelEncoder()
    ml_data['test_group'] = le.fit_transform(ml_data['test_group'])
    X = ml_data[['test_group', 'onboarding_step_reached', 'feature_A_clicks', 'feature_B_clicks']]
    y = ml_data['churned']
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    st.info("**Product Bet:** The model confirms that engaging with 'Feature B' and finishing onboarding are the highest predictors of retention. We should trigger targeted lifecycle emails promoting Feature B to users at risk of churn.")
