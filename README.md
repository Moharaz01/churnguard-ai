# 🔮 ChurnGuard AI — End-to-End Machine Learning Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly)](https://plotly.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4)](https://xgboost.readthedocs.io)
[![UK GDPR](https://img.shields.io/badge/UK%20GDPR-Fully%20Compliant-00A86B)](https://ico.org.uk)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

---

## 📌 What This Project Does — In Plain English

Imagine you run a telecoms company. Every month, some customers cancel their contracts and switch to a competitor. This is called **customer churn**, and it costs businesses enormous amounts of money.

**ChurnGuard AI** is a machine learning system that:
1. Looks at a customer's data (how long they've been with you, what services they use, how much they pay, etc.)
2. Predicts the probability that they will cancel in the near future
3. Tells the business exactly how much revenue is at risk
4. Suggests specific actions to retain at-risk customers

This is the kind of system that real telecoms companies (BT, Vodafone, O2, Virgin Media) use in production today.

---

## 🎯 Why This Project Matters for a Data Science Career

This project demonstrates the **complete ML lifecycle** — not just building a model, but the full professional workflow:

| Stage | What Was Done |
|-------|--------------|
| Business Problem | Defined churn as a binary classification task with revenue impact |
| Data Generation | Created GDPR-compliant synthetic dataset with realistic distributions |
| Exploratory Analysis | Identified churn drivers using statistical and visual analysis |
| Feature Engineering | Created 27 features including interaction terms and business metrics |
| Modelling | Trained and compared 3 algorithms with proper validation |
| Evaluation | Used business-relevant metrics, not just accuracy |
| Deployment | Built an interactive GUI deployable to the cloud |
| Compliance | Documented full UK GDPR compliance framework |

---

## 📊 Results at a Glance

| Model | Accuracy | AUC-ROC | F1-Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| Logistic Regression | 79.2% | 0.840 | 0.71 | 0.73 | 0.69 |
| **Random Forest** ⭐ | **83.4%** | **0.890** | **0.77** | **0.79** | **0.75** |
| Gradient Boosting | 81.8% | 0.870 | 0.74 | 0.77 | 0.72 |

**Business Impact Translated:** The Random Forest model correctly identifies 75% of customers who will churn. On a dataset of 5,000 customers with a 26% churn rate, this translates to identifying approximately £2.1 million of annual revenue at risk — enabling targeted retention campaigns.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CHURNGUARD AI PIPELINE                    │
└─────────────────────────────────────────────────────────────┘

    SYNTHETIC DATA GENERATION (GDPR Compliant)
    5,000 customer records, 18 raw features
    Statistical distributions based on real telecom patterns
              │
              ▼
    EXPLORATORY DATA ANALYSIS
    Churn rate by contract type, tenure, services
    Correlation matrix, distribution plots
              │
              ▼
    FEATURE ENGINEERING (27 features total)
    ┌─────────────────────────────────────┐
    │ • ServicesCount (additive)          │
    │ • AvgMonthlySpend (tenure-adjusted) │
    │ • TenureGroup (categorical)         │
    │ • LowSatisfaction flag              │
    │ • HighCallVolume flag               │
    │ • ChargePerService ratio            │
    │ • TenureUsageInteraction term       │
    └─────────────────────────────────────┘
              │
              ▼
    PREPROCESSING PIPELINE
    StandardScaler → LabelEncoder → Train/Test Split (80/20)
    StratifiedKFold Cross-Validation (k=5)
              │
              ▼
    MODEL TRAINING & COMPARISON
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │ Logistic         │ │ Random Forest    │ │ Gradient         │
    │ Regression       │ │ (Best ⭐)        │ │ Boosting         │
    │ AUC: 0.84        │ │ AUC: 0.89        │ │ AUC: 0.87        │
    └──────────────────┘ └──────────────────┘ └──────────────────┘
              │
              ▼
    PREDICTION + EXPLANATION
    Live prediction interface | Feature importance | Business impact
              │
              ▼
    BUSINESS REPORT
    Revenue at risk | Recommendations | Audit trail
```

---

## 🔑 Key Technical Decisions — With Explanations

### Why Random Forest Won

Random Forest performs best here because:
- The dataset has **non-linear relationships** (e.g., the effect of price on churn depends on tenure)
- Logistic Regression assumes a linear relationship and misses these interactions
- Gradient Boosting can overfit on this size of dataset; Random Forest's bagging is more stable
- Random Forest provides **feature importance scores** — essential for business explainability

### Why AUC-ROC Was the Primary Metric (Not Accuracy)

The dataset has a **class imbalance**: only 26% of customers churn. If the model simply predicted "no churn" for everyone, it would achieve 74% accuracy — but it would be completely useless.

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures the model's ability to *distinguish* between churners and non-churners across all decision thresholds, regardless of class imbalance.

### Why StratifiedKFold Cross-Validation

A single train/test split could give a misleadingly good or bad result depending on which customers ended up in each set by chance. StratifiedKFold:
- Splits the data into 5 folds
- Trains on 4 folds, validates on 1
- Rotates so every customer is in the validation set exactly once
- **Stratified** means each fold has the same 26% churn rate — no fold is accidentally easier or harder

### Why StandardScaler Is Fitted Only on Training Data

This is one of the most common mistakes in ML — and it matters for data integrity. If you scale using statistics calculated from the full dataset (including the test set), you are "leaking" information from the future into your model. The test set should represent truly unseen data. The scaler is fitted on training data only, then applied to test data using training statistics.

---

## 🔒 UK GDPR Compliance Documentation

### Data Used in This Project

| Data Type | Status | GDPR Justification |
|-----------|--------|-------------------|
| Customer records | ✅ Fully synthetic | No natural persons represented — GDPR does not apply |
| Demographic features | ✅ Statistically generated | No real names, addresses, or identifiers |
| Churn labels | ✅ Algorithmically derived | No real customer outcomes used |
| Model predictions | ✅ Demo only | Not used to make decisions about real people |

### If Deployed in Production (Real Customer Data)

When this system is deployed with real customer data, the following UK GDPR obligations apply:

**Article 5 — Data Processing Principles:**
- **Lawfulness:** Document the lawful basis (likely Article 6(1)(f) legitimate interests — preventing churn is a legitimate business interest that does not override customer rights)
- **Purpose Limitation:** Customer data collected for billing/service delivery cannot be repurposed for ML without a compatible purpose assessment
- **Data Minimisation:** Only collect the features genuinely needed for the churn model — not everything available
- **Accuracy:** Ensure training data reflects the current customer base — stale data leads to biased predictions
- **Storage Limitation:** Define how long prediction scores are retained; scores about individuals are personal data
- **Security:** Encrypt data at rest and in transit; control access to prediction scores

**Article 22 — Automated Decision-Making:**
If the model's output is used to automatically restrict services, change pricing, or take other significant actions without human review, Article 22 applies. Customers have the right to:
- Request human review of the automated decision
- Contest the decision
- Receive an explanation of the logic involved

**This system addresses Article 22 by:**
- Presenting predictions as decision-support tools, not autonomous decisions
- Including feature importance explanations for every prediction
- Recommending human intervention steps for at-risk customers (not automated actions)

**Article 35 — Data Protection Impact Assessment (DPIA):**
A DPIA is required before deployment because this system involves:
- Systematic, large-scale processing of personal data
- Automated profiling that may affect customers
- New technology (ML) being applied to existing customer data

**The ICO's DPIA template should be completed covering:**
1. Description of the processing
2. Necessity and proportionality
3. Risks to individuals
4. Measures to address risks

---

## 🚀 Quick Start — Run Locally in 5 Minutes

### Step 1: Check Python Version

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
```

You need Python 3.10 or higher. If you see Python 3.9 or lower, download the latest Python from [python.org](https://python.org).

### Step 2: Download This Project

```bash
# If you have Git installed:
git clone https://github.com/Moharaz01/churnguard-ai.git
cd churnguard-ai

# If you don't have Git, download the ZIP from GitHub and unzip it
# Then open a terminal in the unzipped folder
```

### Step 3: Create a Virtual Environment (Strongly Recommended)

A virtual environment keeps this project's packages separate from other Python projects on your computer, preventing version conflicts.

```bash
# Create the environment
python -m venv churnguard_env

# Activate it:
# On Windows:
churnguard_env\Scripts\activate

# On Mac or Linux:
source churnguard_env/bin/activate

# You'll see (churnguard_env) appear in your terminal — that means it worked
```

### Step 4: Install the Required Packages

```bash
pip install -r requirements_churnguard.txt
```

This will download and install all necessary libraries. It may take 2–5 minutes on the first run.

### Step 5: Run the Application

```bash
streamlit run churnguard_app.py
```

Your browser will automatically open to `http://localhost:8501` showing the ChurnGuard AI dashboard.

---

## ☁️ Deployment — Make It Live on the Internet

### Option A: Streamlit Community Cloud (Free — Recommended for Portfolio)

This is the easiest way to get your app live with a public URL.

**What you need:**
- A free GitHub account
- A free Streamlit account (sign up at share.streamlit.io with your GitHub account)

**Step-by-step:**

1. **Create a GitHub repository**
   - Go to github.com and click the `+` button → New repository
   - Name it `churnguard-ai`
   - Set it to **Public** (required for free Streamlit deployment)
   - Click Create repository

2. **Upload your files to GitHub**
   ```bash
   # In your project folder:
   git init
   git add churnguard_app.py requirements_churnguard.txt README.md .gitignore
   git commit -m "Initial commit: ChurnGuard AI churn prediction system"
   git branch -M main
   git remote add origin https://github.com/Moharaz01/churnguard-ai.git
   git push -u origin main
   ```

3. **Deploy on Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your GitHub repository: `churnguard-ai`
   - Set the **Main file path** to: `churnguard_app.py`
   - Click **Deploy**
   - Wait 2–5 minutes — Streamlit installs your packages automatically
   - This project is already live at: **[churnguard-moharaz.streamlit.app](https://churnguard-moharaz.streamlit.app)**

4. **Share the link in your CV, LinkedIn, and GitHub profile**

**Troubleshooting common issues:**
- *"Module not found" error:* Check that all package names in `requirements_churnguard.txt` are spelled correctly
- *App crashes on startup:* Click the "Logs" button in Streamlit Cloud to see the error message
- *App is very slow:* The first load downloads packages — subsequent loads are faster. Add a spinner to improve perceived performance.

---

### Option B: Docker (Professional Deployment)

Docker packages your app into a container that runs identically on any computer or server.

**Step 1: Create a Dockerfile in your project folder**

```dockerfile
# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer — faster rebuilds)
COPY requirements_churnguard.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_churnguard.txt

# Copy the rest of the application
COPY churnguard_app.py .

# Tell Docker which port Streamlit uses
EXPOSE 8501

# The command to run when the container starts
CMD ["streamlit", "run", "churnguard_app.py", 
     "--server.port=8501", 
     "--server.address=0.0.0.0",
     "--server.headless=true"]
```

**Step 2: Build and run the Docker container**

```bash
# Build the image (run this from your project folder)
docker build -t churnguard-ai .

# Run it locally
docker run -p 8501:8501 churnguard-ai

# Open http://localhost:8501 in your browser
```

**Step 3: Deploy to a cloud server (e.g., AWS, Azure, GCP)**

```bash
# Example: Deploy to AWS Elastic Container Registry + App Runner
# (Your cloud provider will have specific instructions for this)

# Push image to Docker Hub (free, public hosting)
docker tag churnguard-ai your-dockerhub-username/churnguard-ai:latest
docker push your-dockerhub-username/churnguard-ai:latest
```

---

### Option C: Hugging Face Spaces (Alternative Free Option)

Hugging Face Spaces supports Streamlit apps and is popular in the ML community.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space** → Select **Streamlit** as the SDK
3. Upload your files: `churnguard_app.py`, `requirements_churnguard.txt`
4. Rename `requirements_churnguard.txt` to `requirements.txt` (Hugging Face expects this name)
5. Your app will build and deploy automatically
6. URL format: `https://huggingface.co/spaces/[your-username]/churnguard-ai`

---

## 📁 File Structure

```
project1_churnguard_ml/
│
├── churnguard_app.py              ← Main Streamlit application
│   ├── Data generation (GDPR-compliant synthetic data)
│   ├── Exploratory Data Analysis pages
│   ├── Feature Engineering pipeline
│   ├── Model training & comparison
│   ├── Live prediction interface
│   ├── Business impact report
│   └── How It Works + GDPR page
│
├── requirements_churnguard.txt    ← All Python package dependencies
├── README.md                      ← This file
├── GDPR_ChurnGuard.md             ← Standalone GDPR compliance document
└── .gitignore                     ← Files excluded from Git version control
```

---

## 📦 Dependencies Explained

```
streamlit==1.35.0       ← The web application framework (creates the GUI)
pandas==2.2.0           ← Data manipulation and analysis
numpy==1.26.4           ← Numerical computing (arrays, maths)
scikit-learn==1.4.2     ← Machine learning algorithms (RF, LR, GBM, scaling)
plotly==5.22.0          ← Interactive charts and visualisations
xgboost==2.0.3          ← Alternative gradient boosting implementation
joblib==1.4.0           ← Saving and loading ML models efficiently
```

---

## 🧪 How to Understand and Explain This Project in an Interview

When asked "walk me through your ChurnGuard project", use this structure:

**1. Start with the business problem:**
> "The problem was customer churn in telecoms — customers cancelling contracts. This is expensive because acquiring a new customer costs 5–7 times more than retaining an existing one. The goal was to predict which customers were at risk so the business could intervene proactively."

**2. Explain your data approach:**
> "I used fully synthetic data for GDPR compliance — 5,000 customer records generated with realistic statistical distributions. No real personal data was processed."

**3. Explain your technical decisions:**
> "I chose Random Forest as the primary model because the relationship between features and churn is non-linear. I used AUC-ROC as my primary metric because the dataset was imbalanced — 74% non-churners — and accuracy would have been misleading. StratifiedKFold cross-validation ensured reliable performance estimates across folds."

**4. Quantify the result:**
> "The Random Forest achieved AUC 0.89 — meaning it correctly ranks a churner above a non-churner 89% of the time. Translated to business terms, this model could identify approximately £2.1 million of at-risk revenue annually."

**5. Mention compliance:**
> "I documented the full UK GDPR compliance framework including Article 22 automated decision-making provisions, the DPIA requirement, and the right to explanation — which is satisfied by the feature importance explanations built into the interface."

---

## 📜 Licence

MIT Licence — free to use, modify, and distribute with attribution.

---

## 📬 Contact

Built as part of a professional AI/ML portfolio targeting UK data science roles.

**Mrithik Das Raz** | [LinkedIn](https://linkedin.com/in/mdrmrithik01) | [GitHub](https://github.com/Moharaz01)
