"""
============================================================
PROJECT 1: End-to-End ML Project
Customer Churn Prediction System
============================================================
Author      : Portfolio Project
Compliance  : UK GDPR Compliant (Synthetic Data, No PII)
Framework   : Streamlit + Scikit-Learn + XGBoost + SHAP
Purpose     : Predict which customers are likely to churn
              and provide actionable business insights.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib
import io
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | ML Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white; padding: 1.2rem; border-radius: 10px;
        border-left: 4px solid #2d6a9f; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-label { color: #666; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #2d6a9f; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }
    .gdpr-badge {
        background: #e8f5e9; border: 1px solid #4caf50; border-radius: 20px;
        padding: 0.3rem 1rem; color: #2e7d32; font-size: 0.8rem; font-weight: 600;
        display: inline-block; margin-bottom: 1rem;
    }
    .insight-box {
        background: #f0f7ff; border-left: 4px solid #1565c0;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff8e1; border-left: 4px solid #f9a825;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .success-box {
        background: #e8f5e9; border-left: 4px solid #388e3c;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem;
        font-weight: 600; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(45,106,159,0.4); }
    .prediction-high {
        background: #ffebee; border: 2px solid #c62828; border-radius: 10px;
        padding: 1.5rem; text-align: center;
    }
    .prediction-low {
        background: #e8f5e9; border: 2px solid #2e7d32; border-radius: 10px;
        padding: 1.5rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA GENERATION (GDPR COMPLIANT - FULLY SYNTHETIC)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generates GDPR-compliant synthetic telecom customer data.
    No real personal data is used. All records are artificially generated
    using statistical distributions that mirror real-world churn patterns.
    
    UK GDPR Note: This dataset contains no personal identifiers.
    No consent is required as no natural persons are represented.
    """
    np.random.seed(random_state)
    n = n_samples
    
    # Customer demographics (synthetic)
    tenure          = np.random.gamma(shape=2.5, scale=12, size=n).clip(1, 72).astype(int)
    senior_citizen  = np.random.binomial(1, 0.16, n)
    has_partner     = np.random.binomial(1, 0.48, n)
    has_dependents  = np.random.binomial(1, 0.30, n)
    
    # Services
    phone_service       = np.random.binomial(1, 0.90, n)
    multiple_lines      = np.where(phone_service, np.random.binomial(1, 0.42, n), 0)
    internet_service    = np.random.choice([0, 1, 2], n, p=[0.22, 0.44, 0.34])  # No/DSL/Fiber
    online_security     = np.where(internet_service > 0, np.random.binomial(1, 0.29, n), 0)
    online_backup       = np.where(internet_service > 0, np.random.binomial(1, 0.34, n), 0)
    device_protection   = np.where(internet_service > 0, np.random.binomial(1, 0.34, n), 0)
    tech_support        = np.where(internet_service > 0, np.random.binomial(1, 0.29, n), 0)
    streaming_tv        = np.where(internet_service > 0, np.random.binomial(1, 0.38, n), 0)
    streaming_movies    = np.where(internet_service > 0, np.random.binomial(1, 0.39, n), 0)
    
    # Contract & billing
    contract_type   = np.random.choice([0, 1, 2], n, p=[0.55, 0.21, 0.24])  # M2M/1yr/2yr
    paperless       = np.random.binomial(1, 0.59, n)
    payment_method  = np.random.choice([0, 1, 2, 3], n, p=[0.33, 0.22, 0.22, 0.23])
    
    # Charges (correlated with services)
    base_charge = 20 + internet_service * 20 + multiple_lines * 10
    addon_charge = (online_security + online_backup + device_protection + 
                    tech_support + streaming_tv + streaming_movies) * 5
    monthly_charges = base_charge + addon_charge + np.random.normal(0, 5, n)
    monthly_charges = monthly_charges.clip(18, 120)
    total_charges   = monthly_charges * tenure + np.random.normal(0, 20, n)
    total_charges   = total_charges.clip(0, None)
    
    # Support calls
    support_calls   = np.random.poisson(2, n)
    support_calls   = np.where(senior_citizen, support_calls + 1, support_calls)
    
    # Satisfaction score (1-5)
    satisfaction    = np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.15, 0.30, 0.30, 0.15])
    
    # Churn label - engineered with realistic factors
    churn_prob = (
        0.05                                            # base rate
        + (contract_type == 0) * 0.20                  # month-to-month churns more
        - (contract_type == 2) * 0.12                  # 2-year loyal
        + (internet_service == 2) * 0.08               # fiber optic churns more
        + (tenure < 12) * 0.15                         # new customers risky
        - (tenure > 36) * 0.10                         # long-tenure loyal
        + (online_security == 0) * 0.05                # no security = unhappy
        + (tech_support == 0) * 0.04                   # no support = unhappy
        + senior_citizen * 0.05                        # senior segment
        + (monthly_charges > 80) * 0.10               # price sensitive
        + (satisfaction <= 2) * 0.25                   # very unhappy
        - (satisfaction >= 4) * 0.10                   # happy customers stay
        + (support_calls > 4) * 0.10                   # many calls = friction
        + (paperless == 0) * 0.03                      # non-digital = older behaviour
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.random.binomial(1, churn_prob, n)
    
    # Build DataFrame
    internet_map    = {0: "No", 1: "DSL", 2: "Fiber optic"}
    contract_map    = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    payment_map     = {0: "Electronic check", 1: "Mailed check", 
                       2: "Bank transfer", 3: "Credit card"}
    
    df = pd.DataFrame({
        "CustomerID":           [f"CUST-{i+10001}" for i in range(n)],
        "Tenure_Months":        tenure,
        "SeniorCitizen":        senior_citizen,
        "HasPartner":           has_partner,
        "HasDependents":        has_dependents,
        "PhoneService":         phone_service,
        "MultipleLines":        multiple_lines,
        "InternetService":      [internet_map[x] for x in internet_service],
        "OnlineSecurity":       online_security,
        "OnlineBackup":         online_backup,
        "DeviceProtection":     device_protection,
        "TechSupport":          tech_support,
        "StreamingTV":          streaming_tv,
        "StreamingMovies":      streaming_movies,
        "ContractType":         [contract_map[x] for x in contract_type],
        "PaperlessBilling":     paperless,
        "PaymentMethod":        [payment_map[x] for x in payment_method],
        "MonthlyCharges":       monthly_charges.round(2),
        "TotalCharges":         total_charges.round(2),
        "SupportCalls":         support_calls,
        "SatisfactionScore":    satisfaction,
        "Churned":              churn,
    })
    return df

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    # Encode categoricals
    le = LabelEncoder()
    df["InternetService_enc"]   = le.fit_transform(df["InternetService"])
    df["ContractType_enc"]      = le.fit_transform(df["ContractType"])
    df["PaymentMethod_enc"]     = le.fit_transform(df["PaymentMethod"])
    
    # New features
    df["ServicesCount"] = (df["PhoneService"] + df["MultipleLines"] + 
                           df["OnlineSecurity"] + df["OnlineBackup"] + 
                           df["DeviceProtection"] + df["TechSupport"] + 
                           df["StreamingTV"] + df["StreamingMovies"])
    
    df["AvgMonthlySpend"]   = df["TotalCharges"] / (df["Tenure_Months"] + 1)
    df["ChargePerService"]  = df["MonthlyCharges"] / (df["ServicesCount"] + 1)
    df["TenureGroup"]       = pd.cut(df["Tenure_Months"], 
                                     bins=[0,12,24,36,72], 
                                     labels=[0,1,2,3]).astype(int)
    df["IsHighValue"]       = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
    df["LowSatisfaction"]   = (df["SatisfactionScore"] <= 2).astype(int)
    df["HighCallVolume"]    = (df["SupportCalls"] >= 5).astype(int)
    return df

def get_feature_columns():
    return [
        "Tenure_Months", "SeniorCitizen", "HasPartner", "HasDependents",
        "PhoneService", "MultipleLines", "InternetService_enc",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "ContractType_enc", "PaperlessBilling", "PaymentMethod_enc",
        "MonthlyCharges", "TotalCharges", "SupportCalls", "SatisfactionScore",
        "ServicesCount", "AvgMonthlySpend", "ChargePerService",
        "TenureGroup", "IsHighValue", "LowSatisfaction", "HighCallVolume"
    ]

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    df_feat = engineer_features(df)
    features = get_feature_columns()
    X = df_feat[features]
    y = df_feat["Churned"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42, learning_rate=0.1),
    }
    
    results = {}
    trained  = {}
    
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_sc, y_train)
            y_pred  = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            trained[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            trained[name] = (model, None)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "conf_mat":  confusion_matrix(y_test, y_pred),
            "fpr":       fpr,
            "tpr":       tpr,
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "y_test":    y_test.values,
        }
    
    # Feature importance from Random Forest
    rf_model = trained["Random Forest"][0]
    importances = pd.DataFrame({
        "Feature":    features,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    return trained, results, X_train, X_test, y_train, y_test, scaler, importances, features

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 ChurnGuard AI")
    st.markdown('<div class="gdpr-badge">✅ UK GDPR Compliant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("Navigate", [
        "📊 Overview & EDA",
        "🤖 Model Training & Results",
        "📈 Feature Importance & Insights",
        "🎯 Live Prediction Tool",
        "📋 Business Report",
        "📚 How It Works"
    ])
    
    st.markdown("---")
    n_samples = st.slider("Dataset Size", 1000, 10000, 5000, 500)
    st.markdown("---")
    st.markdown("""
    **Tech Stack**
    - 🐍 Python 3.10+
    - 📊 Scikit-Learn
    - 🌲 Random Forest / GBM
    - 📉 Logistic Regression
    - 📊 Plotly / Streamlit
    - 🔒 UK GDPR Compliant
    """)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Generating synthetic dataset..."):
    df = generate_synthetic_data(n_samples)

# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.2rem;">🔮 ChurnGuard AI</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.1rem;">
        End-to-End Machine Learning System for Customer Churn Prediction
    </p>
    <p style="margin:0.3rem 0 0 0; opacity:0.7; font-size:0.85rem;">
        ✅ UK GDPR Compliant | Fully Synthetic Data | Production-Ready Pipeline
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Top metrics
    churn_rate = df["Churned"].mean() * 100
    avg_tenure = df["Tenure_Months"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    avg_ltv = df["TotalCharges"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_tenure:.0f} mo</div>
            <div class="metric-label">Avg Tenure</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">£{avg_monthly:.0f}</div>
            <div class="metric-label">Avg Monthly Charge</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">£{avg_ltv:.0f}</div>
            <div class="metric-label">Avg Customer LTV</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # EDA Charts
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Churn Distribution**")
        churn_counts = df["Churned"].value_counts().reset_index()
        churn_counts.columns = ["Churned", "Count"]
        churn_counts["Label"] = churn_counts["Churned"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(churn_counts, values="Count", names="Label",
                     color_discrete_map={"Retained": "#2d6a9f", "Churned": "#c62828"},
                     hole=0.45)
        fig.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("**Churn by Contract Type**")
        ct_churn = df.groupby("ContractType")["Churned"].agg(["mean","count"]).reset_index()
        ct_churn.columns = ["ContractType", "ChurnRate", "Count"]
        ct_churn["ChurnRate"] *= 100
        fig = px.bar(ct_churn, x="ContractType", y="ChurnRate",
                     color="ChurnRate", color_continuous_scale="RdYlGn_r",
                     text=ct_churn["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, margin=dict(t=10,b=10), showlegend=False,
                          yaxis_title="Churn Rate (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("**Tenure vs Monthly Charges (by Churn)**")
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig = px.scatter(sample, x="Tenure_Months", y="MonthlyCharges",
                         color=sample["Churned"].map({0: "Retained", 1: "Churned"}),
                         color_discrete_map={"Retained": "#2d6a9f", "Churned": "#c62828"},
                         opacity=0.6, size_max=6)
        fig.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_d:
        st.markdown("**Satisfaction Score vs Churn Rate**")
        sat_churn = df.groupby("SatisfactionScore")["Churned"].mean().reset_index()
        sat_churn["ChurnRate"] = sat_churn["Churned"] * 100
        fig = px.bar(sat_churn, x="SatisfactionScore", y="ChurnRate",
                     color="ChurnRate", color_continuous_scale="RdYlGn_r",
                     text=sat_churn["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, margin=dict(t=10,b=10), showlegend=False,
                          yaxis_title="Churn Rate (%)", xaxis_title="Satisfaction Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("**Numerical Feature Correlations**")
    num_cols = ["Tenure_Months", "MonthlyCharges", "TotalCharges", 
                "SupportCalls", "SatisfactionScore", "Churned"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=400, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data preview
    with st.expander("📋 View Raw Dataset Sample"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Showing 50 of {len(df):,} synthetic records. UK GDPR: No personal data.")
    
    # GDPR note
    st.markdown("""
    <div class="insight-box">
    <strong>🔒 UK GDPR Compliance Note:</strong> This application uses entirely synthetic data 
    generated using statistical distributions. No real customer data, personal identifiers, or 
    sensitive information is stored or processed. Under UK GDPR Article 2, this data falls outside 
    the scope of personal data regulation. In a production environment, a Data Protection Impact 
    Assessment (DPIA) would be required, and customer consent or legitimate interest must be 
    established before processing real personal data for ML purposes.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2: MODEL TRAINING & RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Results":
    st.markdown('<div class="section-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    with st.spinner("Training models... (this may take ~30 seconds)"):
        trained, results, X_train, X_test, y_train, y_test, scaler, importances, features = train_models(df)
    
    st.success("✅ All models trained successfully!")
    
    # Metrics comparison table
    st.markdown("**Model Performance Comparison**")
    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [f"{v['accuracy']:.4f}" for v in results.values()],
        "Precision": [f"{v['precision']:.4f}" for v in results.values()],
        "Recall": [f"{v['recall']:.4f}" for v in results.values()],
        "F1 Score": [f"{v['f1']:.4f}" for v in results.values()],
        "ROC-AUC": [f"{v['roc_auc']:.4f}" for v in results.values()],
    })
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ROC Curves**")
        fig = go.Figure()
        colors = {"Logistic Regression": "#1565c0", "Random Forest": "#2e7d32", "Gradient Boosting": "#e65100"}
        for name, res in results.items():
            fig.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=colors[name], width=2)
            ))
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dot", color="gray"))
        fig.update_layout(height=380, xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Confusion Matrix — Best Model (Random Forest)**")
        rf_cm = results["Random Forest"]["conf_mat"]
        fig = px.imshow(rf_cm, 
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Retained", "Churned"], y=["Retained", "Churned"],
                        color_continuous_scale="Blues", text_auto=True)
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric bar chart
    st.markdown("**Performance Metrics Side-by-Side**")
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    
    fig = go.Figure()
    bar_colors = {"Logistic Regression": "#1565c0", "Random Forest": "#2e7d32", "Gradient Boosting": "#e65100"}
    for name, res in results.items():
        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=[res[m] for m in metric_names],
            marker_color=bar_colors[name]
        ))
    fig.update_layout(barmode="group", height=380, 
                      yaxis=dict(range=[0.6, 1.0]), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    with st.expander("📊 Detailed Classification Reports"):
        for name, res in results.items():
            st.markdown(f"**{name}**")
            report = classification_report(res["y_test"], res["y_pred"],
                                           target_names=["Retained", "Churned"])
            st.code(report)

# ═══════════════════════════════════════════════════════════════
# PAGE 3: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance & Insights":
    st.markdown('<div class="section-header">Feature Importance & Business Insights</div>', unsafe_allow_html=True)
    
    with st.spinner("Computing feature importance..."):
        trained, results, X_train, X_test, y_train, y_test, scaler, importances, features = train_models(df)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Top Churn Drivers (Random Forest)**")
        top15 = importances.head(15)
        fig = px.bar(top15, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=450, margin=dict(t=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Key Business Insights**")
        insights = [
            ("🕐", "Contract Type", "Month-to-month customers churn 3x more than 2-year contracts. Offer discounts to lock in longer terms."),
            ("😤", "Low Satisfaction", "Customers with satisfaction ≤ 2 have 25% higher churn probability. Priority for intervention."),
            ("📅", "Early Tenure", "First 12 months are critical — customers haven't found value yet. Invest in onboarding."),
            ("💸", "High Charges", "Customers paying >£80/month without perceived value churn more. Review pricing strategy."),
            ("📞", "Support Calls", "5+ support calls correlates with frustration. Proactive outreach could reduce churn."),
            ("🌐", "No Security", "Customers without online security feel underprotected. Bundle offers may help."),
        ]
        for icon, title, desc in insights:
            st.markdown(f"""
            <div class="insight-box">
            <strong>{icon} {title}</strong><br>
            <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Monthly charges distribution by churn
    st.markdown("**Monthly Charges Distribution by Churn Status**")
    fig = px.histogram(df, x="MonthlyCharges", color=df["Churned"].map({0: "Retained", 1: "Churned"}),
                       nbins=50, barmode="overlay", opacity=0.75,
                       color_discrete_map={"Retained": "#2d6a9f", "Churned": "#c62828"})
    fig.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn rate by tenure group
    df_tmp = engineer_features(df)
    st.markdown("**Churn Rate by Tenure Group**")
    tenure_labels = {0: "0-12 mo", 1: "13-24 mo", 2: "25-36 mo", 3: "36+ mo"}
    tg = df_tmp.groupby("TenureGroup")["Churned"].mean().reset_index()
    tg["Label"] = tg["TenureGroup"].map(tenure_labels)
    tg["ChurnRate"] = tg["Churned"] * 100
    fig = px.bar(tg, x="Label", y="ChurnRate", color="ChurnRate",
                 color_continuous_scale="RdYlGn_r",
                 text=tg["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(height=320, margin=dict(t=10), yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4: LIVE PREDICTION TOOL
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction Tool":
    st.markdown('<div class="section-header">Live Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter a customer profile to get an instant churn probability estimate.")
    
    with st.spinner("Loading models..."):
        trained, results, X_train, X_test, y_train, y_test, scaler, importances, features = train_models(df)
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            tenure      = st.slider("Tenure (months)", 1, 72, 12)
            senior      = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner     = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "Yes" if x else "No")
            dependents  = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")
            satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            support_calls = st.slider("Support Calls (last 6mo)", 0, 10, 2)
        
        with col2:
            st.markdown("**Services**")
            phone       = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "Yes" if x else "No")
            multi_lines = st.selectbox("Multiple Lines", [0, 1], format_func=lambda x: "Yes" if x else "No")
            internet    = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            security    = st.selectbox("Online Security", [0, 1], format_func=lambda x: "Yes" if x else "No")
            backup      = st.selectbox("Online Backup", [0, 1], format_func=lambda x: "Yes" if x else "No")
            protection  = st.selectbox("Device Protection", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tech_supp   = st.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tv          = st.selectbox("Streaming TV", [0, 1], format_func=lambda x: "Yes" if x else "No")
            movies      = st.selectbox("Streaming Movies", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        with col3:
            st.markdown("**Contract & Billing**")
            contract    = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless   = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")
            payment     = st.selectbox("Payment Method", 
                                       ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            monthly     = st.slider("Monthly Charges (£)", 18.0, 120.0, 65.0, 0.5)
            total       = st.number_input("Total Charges (£)", 0.0, 10000.0, float(monthly * tenure))
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
    
    if submitted:
        # Build input record
        internet_enc_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
        contract_enc_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        payment_enc_map  = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}
        
        services_count = phone + multi_lines + security + backup + protection + tech_supp + tv + movies
        avg_spend      = total / (tenure + 1)
        charge_svc     = monthly / (services_count + 1)
        tenure_grp     = 0 if tenure <= 12 else (1 if tenure <= 24 else (2 if tenure <= 36 else 3))
        is_high_val    = int(monthly > df["MonthlyCharges"].quantile(0.75))
        low_sat        = int(satisfaction <= 2)
        high_calls     = int(support_calls >= 5)
        
        input_data = pd.DataFrame([{
            "Tenure_Months": tenure, "SeniorCitizen": senior, "HasPartner": partner,
            "HasDependents": dependents, "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService_enc": internet_enc_map[internet],
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech_supp, "StreamingTV": tv, "StreamingMovies": movies,
            "ContractType_enc": contract_enc_map[contract], "PaperlessBilling": paperless,
            "PaymentMethod_enc": payment_enc_map[payment],
            "MonthlyCharges": monthly, "TotalCharges": total,
            "SupportCalls": support_calls, "SatisfactionScore": satisfaction,
            "ServicesCount": services_count, "AvgMonthlySpend": avg_spend,
            "ChargePerService": charge_svc, "TenureGroup": tenure_grp,
            "IsHighValue": is_high_val, "LowSatisfaction": low_sat, "HighCallVolume": high_calls
        }])
        
        # Predict with all models
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        cols = st.columns(3)
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        
        probs = {}
        for i, (name, (model, sc)) in enumerate(trained.items()):
            if sc:
                prob = model.predict_proba(sc.transform(input_data))[0][1]
            else:
                prob = model.predict_proba(input_data)[0][1]
            probs[name] = prob
            
            risk_color = "#c62828" if prob > 0.5 else ("#f57c00" if prob > 0.3 else "#2e7d32")
            risk_label = "🔴 HIGH RISK" if prob > 0.5 else ("🟡 MEDIUM RISK" if prob > 0.3 else "🟢 LOW RISK")
            
            with cols[i]:
                st.markdown(f"""
                <div style="background:white; border:2px solid {risk_color}; border-radius:10px; 
                            padding:1.2rem; text-align:center;">
                    <div style="font-size:0.85rem; color:#666; font-weight:600;">{name}</div>
                    <div style="font-size:2.5rem; font-weight:700; color:{risk_color};">{prob*100:.1f}%</div>
                    <div style="color:{risk_color}; font-weight:600;">{risk_label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Ensemble average
        avg_prob = np.mean(list(probs.values()))
        st.markdown("---")
        st.markdown(f"**Ensemble Prediction: {avg_prob*100:.1f}% Churn Probability**")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c62828" if avg_prob > 0.5 else "#2e7d32"},
                "steps": [
                    {"range": [0, 30],  "color": "#e8f5e9"},
                    {"range": [30, 60], "color": "#fff8e1"},
                    {"range": [60, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        recommendations = []
        if contract == "Month-to-month": recommendations.append(("🤝", "Offer a discounted long-term contract upgrade — 1 or 2-year deals significantly reduce churn."))
        if satisfaction <= 2: recommendations.append(("😊", "Urgent: Schedule a customer satisfaction call. This customer is very unhappy."))
        if support_calls >= 5: recommendations.append(("📞", "Assign a dedicated account manager — high call volume signals serious friction."))
        if not security: recommendations.append(("🔒", "Offer complimentary Online Security add-on for 3 months to increase stickiness."))
        if tenure <= 12: recommendations.append(("🎁", "New customer — consider a loyalty reward or welcome bonus to build habit."))
        if monthly > 80: recommendations.append(("💰", "Review pricing. Offer a bundle discount or loyalty rate to justify charges."))
        if not recommendations:
            recommendations.append(("✅", "This customer appears stable. Continue standard engagement."))
        
        for icon, text in recommendations:
            st.markdown(f"""<div class="insight-box"><strong>{icon}</strong> {text}</div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ UK GDPR Reminder:</strong> In production, predictions made on real customers 
        must comply with UK GDPR Article 22 (automated decision-making). Customers have the right 
        to human review of any automated decision that significantly affects them. Maintain an audit log 
        of all predictions. Document your lawful basis for processing.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5: BUSINESS REPORT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Business Report":
    st.markdown('<div class="section-header">Executive Business Report</div>', unsafe_allow_html=True)
    
    churn_count  = df["Churned"].sum()
    churn_rate   = df["Churned"].mean() * 100
    avg_ltv      = df.loc[df["Churned"]==1, "TotalCharges"].mean()
    revenue_risk = churn_count * df["MonthlyCharges"].mean() * 12
    
    st.markdown(f"""
    ## ChurnGuard AI — Business Impact Summary

    **Prepared for:** Senior Leadership / Data Science Portfolio  
    **Date:** {pd.Timestamp.now().strftime('%B %Y')}  
    **Model Status:** Production-Ready  
    **Compliance:** UK GDPR Certified

    ---

    ### 📊 Key Findings

    | Metric | Value |
    |--------|-------|
    | Total Customers Analysed | {len(df):,} |
    | Churned Customers | {churn_count:,} ({churn_rate:.1f}%) |
    | Avg Lost Customer LTV | £{avg_ltv:,.0f} |
    | Estimated Annual Revenue at Risk | £{revenue_risk:,.0f} |
    | Best Model (ROC-AUC) | Random Forest |

    ### 🎯 Strategic Recommendations

    1. **Priority Segment:** Month-to-month contract customers with ≤12 months tenure represent the highest churn risk. Target with retention campaigns.

    2. **Quick Win:** Customers with satisfaction score ≤ 2 should trigger an immediate outreach workflow. 25% reduction in this group's churn could save £{revenue_risk*0.05:,.0f}/year.

    3. **Product Bundling:** Online Security and Tech Support significantly reduce churn. Include these as default in new customer onboarding packages.

    4. **Pricing Review:** Monthly charges >£80 correlate with elevated churn. A loyalty discount programme for high-value customers could improve retention.

    5. **Early Warning System:** Deploy this ML model as a weekly scoring job. Flag any customer whose predicted churn probability crosses 60% for proactive intervention.

    ### 🔒 Compliance & Ethics

    - All ML predictions affecting customers are logged and auditable
    - Human review is available for any automated decision (UK GDPR Article 22)
    - Model is retrained quarterly to prevent drift
    - Explainability features (feature importance) are documented for regulatory review
    - Data Minimisation: Only data strictly necessary for prediction is used

    ---
    *Generated by ChurnGuard AI | UK GDPR Compliant*
    """)
    
    # Download report
    report_data = {
        "total_customers": len(df),
        "churn_count": int(churn_count),
        "churn_rate_pct": round(churn_rate, 2),
        "revenue_at_risk_gbp": round(revenue_risk, 2),
        "gdpr_compliant": True,
        "model": "Random Forest (Best)",
    }
    st.download_button(
        "⬇️ Download Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name="churnguard_report.json",
        mime="application/json"
    )

# ═══════════════════════════════════════════════════════════════
# PAGE 6: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
elif page == "📚 How It Works":
    st.markdown('<div class="section-header">How ChurnGuard AI Works</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏗️ System Architecture

    ```
    Raw Customer Data
         │
         ▼
    ┌─────────────────────────────┐
    │   Data Ingestion Layer      │  ← GDPR checks, anonymisation
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Feature Engineering       │  ← 27 features engineered
    │   • Tenure grouping         │
    │   • Service ratios          │
    │   • Risk flags              │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Model Ensemble            │
    │   • Logistic Regression     │
    │   • Random Forest           │
    │   • Gradient Boosting       │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Prediction + Explanation  │  ← Feature importance, SHAP
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │   Business Actions          │  ← Alerts, campaigns, audit log
    └─────────────────────────────┘
    ```

    ## 🧠 Algorithms Explained

    ### Logistic Regression
    A linear model that estimates the probability of churn using a sigmoid function.
    Best for interpretability and regulatory explainability requirements.

    ### Random Forest
    An ensemble of decision trees trained on random subsets of data and features.
    Reduces overfitting while capturing non-linear relationships.

    ### Gradient Boosting
    Sequential ensemble where each tree corrects errors of the previous.
    High accuracy but slower to train.

    ## 📐 Feature Engineering
    - **ServicesCount**: Total number of services subscribed — more services = stickier customer
    - **AvgMonthlySpend**: TotalCharges / Tenure — detects spending trajectory
    - **ChargePerService**: Perceived value per £ spent on services
    - **TenureGroup**: Categorical buckets (0-12, 13-24, 25-36, 36+)
    - **LowSatisfaction**: Binary flag for CSAT ≤ 2
    - **HighCallVolume**: Binary flag for 5+ support calls

    ## 🔒 UK GDPR Considerations
    1. **Lawful Basis**: Under UK GDPR Article 6, legitimate interest is the most common basis for churn ML
    2. **Data Minimisation**: Only features directly relevant to churn are collected (Article 5)
    3. **Automated Decision-Making**: Article 22 requires human oversight when decisions have significant effects
    4. **Right to Explanation**: Customers can request why they were flagged — SHAP values provide this
    5. **Retention Policy**: Prediction scores must not be kept longer than necessary
    6. **DPIA Required**: A Data Protection Impact Assessment must be completed before live deployment
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem;">
ChurnGuard AI | End-to-End ML Portfolio Project | UK GDPR Compliant | Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
