# 🚀 SaaS Growth & Retention Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-streamlit-app-url-here.streamlit.app](https://saas-retention-analyzer.streamlit.app/))
*(Click here to view the live interactive dashboard)*

## 📖 Overview
The **SaaS Growth & Retention Analyzer** is a data-driven web application designed to uncover actionable product insights. In modern B2B/B2C SaaS platforms, algorithms are only as valuable as the growth and retention they drive. 

This project simulates a real-world product analytics workflow for a software creation platform. It ingests user behavior data and applies exploratory data analysis (EDA), statistical hypothesis testing, and machine learning to turn raw data into **winning product bets**.

## ✨ Key Features & Business Impact

### 1. Funnel Optimization (EDA)
* **What it does:** Visualizes the 5-step user onboarding journey using Plotly.
* **Business Value:** Identifies critical drop-off points. In this simulation, it reveals a massive bottleneck at the "Connect Database" step, allowing product teams to prioritize UI simplification where it matters most.

### 2. A/B Testing (Hypothesis Testing)
* **What it does:** Evaluates the success of a new onboarding UI ("Variant" vs "Control") using a Chi-Square statistical test.
* **Business Value:** Replaces guesswork with statistical rigor. Calculates precise conversion rates and P-values to confidently determine if a feature should be deployed to 100% of user traffic.

### 3. Behavioral Churn Modeling (Machine Learning)
* **What it does:** Trains a Decision Tree classifier (`scikit-learn`) on user engagement metrics to predict churn probability.
* **Business Value:** Extracts "Feature Importances" to discover the platform's "Aha! moment." If the model proves that *Feature B* drives retention, marketing teams can trigger targeted lifecycle emails to at-risk users.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend/Deployment:** Streamlit, Streamlit Community Cloud
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly, Matplotlib, Seaborn
* **Statistical Analysis & ML:** SciPy, Scikit-Learn

## 💻 How to Run Locally

If you prefer to run this dashboard on your local machine rather than viewing the live deployment:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mahirobot/saas-retention-analyzer.git](https://github.com/Mahirobot/saas-retention-analyzer.git)
   cd saas-retention-analyzer
