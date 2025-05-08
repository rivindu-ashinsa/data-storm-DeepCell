
# 🧠 Agent Performance & Net Income Analysis

## 📌 Project Objective

The main goal of this analysis is to understand the distribution and correlation between various performance metrics of insurance agents and their overall contribution to net income. We aim to uncover insights that can help identify high-performing agents and opportunities for strategic improvement.

---

## 📊 Dataset Overview

The dataset includes the following key metrics for each agent:
- `unique_customers_last_21_days`
- `unique_quotations_last_21_days`
- `unique_proposals_last_21_days`
- `new_policy_count`
- `ANBP_value` (Annualized New Business Premium)
- `net_income`
- `number_of_policy_holders`

---

## 🔍 Key Findings (By Rivindu)

### 1. Distribution Analysis
- Most features (like ANBP, net income, number of policy holders) are **positively skewed**.
- Few agents significantly outperform others, indicating a **long-tail distribution**.
- Histograms reveal exponential trends, suggesting a **Pareto principle** (80/20 rule) may apply.

### 2. ANBP vs. Net Income
- Scatter plots show a **strong positive correlation** between `ANBP_value` and `net_income`.
- Agents with high ANBP tend to generate more income.

### 3. Top Contributors
- Top 10 agents in terms of ANBP also appear in top 10 for net income.
- Indicates **consistency in high-performance metrics** across features.

---

## 🧠 Suggestions based on EDA

### ✅ 1. Normalize Skewed Features
- Apply log transformation to reduce outlier impact.
- Helpful for `ANBP_value`, `net_income`, `new_policy_count`, `number_of_policy_holders`.

### ✅ 2. Feature Engineering
- Create new metrics such as:
  - `conversion_rate = new_policy_count / unique_quotations_last_21_days`
  - `avg_policy_value = ANBP_value / new_policy_count`
  - `customer_to_policy_ratio = number_of_policy_holders / unique_customers_last_21_days`

### ✅ 3. Clustering Agents
- Apply clustering (e.g., KMeans) to segment agents into performance categories.

### ✅ 4. Trend & Time Analysis
- Extend the dataset with time series data (if available) to observe trends in performance.

### ✅ 5. Visualization Enhancements
- Use boxplots, violin plots, and log-scaled histograms for better visual storytelling.

### ✅ 6. Business Insight Reporting
- Recommend generating PDF reports or dashboards summarizing the KPIs for executives.

---

## 📁 Files Included

- `EDA.ipynb`: Full exploratory analysis and visualizations.
- `README.md`: Summary of findings and recommendations (this file).

---

## 📌 Conclusion

This project provides deep insights into agent performance using both basic and advanced data analysis. The suggestions offered pave the way for better decision-making through data transformation, segmentation, and reporting. This analysis sets the stage for further predictive modeling and optimization.
