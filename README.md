# 🧠 Agent Performance Analysis, Prediction, and Monitoring System

## 📌 Project Overview

This project aims to analyze insurance agent performance, predict future NILL (zero sales) agents, and segment agents into performance categories for targeted interventions. The goal is to leverage data-driven insights to enhance agent productivity and overall business income. This project was developed by team DeepCell for the Data Storm 6.0 competition.

---

## 📊 Dataset

The analysis is based on a dataset including the following key metrics for each agent:

- `unique_customers_last_21_days`
- `unique_quotations_last_21_days`
- `unique_proposals_last_21_days`
- `new_policy_count`
- `ANBP_value` (Annualized New Business Premium)
- `net_income`
- `number_of_policy_holders`
- Agent tenure and sales history data.

---

## 🛠️ Project Components

The project is structured into several key components:

1.  **Exploratory Data Analysis (EDA)**: Understanding data distributions, correlations, and initial insights.
2.  **NILL Agent Prediction**: A predictive model to identify agents likely to have zero policy sales in the upcoming month.
3.  **Agent Performance Clustering & Monitoring**: Segmentation of agents into performance tiers (Low, Mid, High) using clustering techniques, tracking their transitions, and devising custom intervention strategies.
4.  **Interactive Dashboard**: A Streamlit application to visualize NILL agent predictions.

---

## 🔍 Key Analyses and Findings

### 1. Exploratory Data Analysis (EDA)

_(Detailed EDA can be found in `notebooks/EDA.ipynb` and `readme_EDA.md`)_

- **Distribution Analysis**: Many key features (`ANBP_value`, `net_income`, `number_of_policy_holders`) are positively skewed, indicating a long-tail distribution where a few agents significantly outperform others.
- **Correlation**: Strong positive correlation observed between `ANBP_value` and `net_income`.
- **Top Contributors**: Top agents in ANBP are often also top in net income, showing consistency.
- **Suggestions from EDA**:
  - Normalize skewed features (e.g., log transformation).
  - Feature engineering (e.g., conversion rates, average policy values).
  - Agent clustering for performance segmentation.
  - Enhanced visualizations (boxplots, violin plots).

### 2. NILL Agent Prediction

_(Implementation in `DeepCell.ipynb` - Part 1)_

- **Objective**: Predict if an agent will have zero new policy sales (`new_policy_count` = 0) in the next month.
- **Model**: An LGBMClassifier was trained for this binary classification task.
- **Features**: Utilized historical sales data, agent tenure, and engineered time-based features.
- **Outcome**: The model provides probabilities for an agent being "NILL" next month, enabling proactive measures. Predictions are available in `data/submission.csv` and visualized in the dashboard.

### 3. Agent Performance Clustering & Monitoring

_(Implementation in `DeepCell.ipynb` - Part 2, and `clustering.ipynb`)_

- **Objective**: Segment agents into distinct performance categories to understand characteristics and tailor improvement strategies.
- **Methodology**:
  - **KPI Selection**: Key Performance Indicators such as `new_policy_count`, `ANBP_value`, `net_income`, `unique_customers`, and `number_of_policy_holders` were used.
  - **Clustering Algorithm**: K-Means clustering was chosen after comparing several algorithms (KMeans, Agglomerative Clustering, DBSCAN, Gaussian Mixture), identifying three primary clusters: "Low," "Mid," and "High" performers.
- **Key Insights from Clustering**:
  - **Low Performers**: Often new agents; tend to stay in this category longer.
  - **Mid Performers**: Good customer base metrics but may struggle with high-value sales.
  - **High Performers**: Excel in `ANBP_value` and `net_income` but can be prone to dropping to lower tiers.
- **Performance Trends**:
  - Majority of agents tend to stay in the same performance level month-to-month.
  - A higher percentage of agents drop in performance than improve.
  - Most new agents start in the "Low" performance category.
- **Custom Intervention Strategies**: Developed tailored strategies for each performance segment (Low, Mid, High) focusing on onboarding, mentorship, skill-bridging, and retention.

---

## 🚀 Interactive Dashboard

_(Code in `dashboard/app.py`)_

A Streamlit dashboard has been developed to:

- Display the NILL agent predictions from the model.
- Show summary statistics of the predictions.
- Allow exploration of individual agent predictions.
- Provide a downloadable CSV of the results.

---

## 📁 File Structure

- `DeepCell.ipynb`: Main Jupyter notebook containing EDA, NILL agent prediction model, and agent performance clustering.
- `notebooks/EDA.ipynb`: Dedicated Jupyter notebook for detailed Exploratory Data Analysis.
- `clustering.ipynb`: Jupyter notebook focused on the agent clustering analysis.
- `dashboard/app.py`: Python script for the Streamlit dashboard.
- `data/`: Directory containing datasets (e.g., `train_storming_round.csv`, `test_storming_round.csv`, `submission.csv`).
- `requirements.txt`: List of Python dependencies for the project.
- `README.md`: This file, providing an overview of the project.
- `readme_EDA.md`: Detailed README focusing on the EDA phase.

---

## ⚙️ Setup and Usage

### 1. Prerequisites

- Python 3.8+

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### 3. Running the Notebooks

You can run the Jupyter notebooks (`DeepCell.ipynb`, `notebooks/EDA.ipynb`, `clustering.ipynb`) using Jupyter Lab or Jupyter Notebook.

### 4. Running the Dashboard

To start the Streamlit dashboard:

1.  Navigate to the `dashboard` directory:
    ```bash
    cd dashboard
    ```
2.  Ensure the prediction file `data/submission.csv` exists in the `data` directory relative to `app.py` (i.e., `../data/submission.csv` from the perspective of `app.py`, or adjust path in `app.py`). For the current `app.py`, it expects `data/submission.csv` in the same directory as `app.py` or a `data` subdirectory within `dashboard`. Assuming `data` is at the project root, the path in `app.py` might need to be `../data/submission.csv`.
    _The current `app.py` uses `pd.read_csv("data/submission.csv")`. If `app.py` is run from the `dashboard` directory, it will look for `dashboard/data/submission.csv`. Ensure your `submission.csv` is placed correctly or update the path in `app.py` to point to the root `data` folder (e.g., `pd.read_csv("../data/submission.csv")`)._

3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    The dashboard will open in your web browser.

---

## 🏁 Conclusion

This project provides a comprehensive framework for understanding and improving agent performance. By combining EDA, predictive modeling for NILL agents, and performance clustering, actionable insights are generated. The developed intervention strategies, based on data-driven segments, offer a pathway for targeted agent development, potentially leading to increased sales, higher net income, and better agent retention. The interactive dashboard further empowers stakeholders to utilize these predictions effectively.
This analysis sets the stage for further predictive modeling, optimization, and continuous performance monitoring.
