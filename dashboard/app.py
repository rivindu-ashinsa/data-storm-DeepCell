import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/train_storming_round.csv")
    
    # Convert datetime columns
    date_columns = ['agent_join_month', 'first_policy_sold_month', 'year_month']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

df = load_data()

st.title("📊 Insurance Agent Performance Dashboard")

# Summary Metrics
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("New Policies", int(df["new_policy_count"].sum()))
col2.metric("Total Net Income", f"{df['net_income'].sum():,.2f}")
col3.metric("Avg ANBP", f"{df['ANBP_value'].mean():,.2f}")

# Policy Trend
st.subheader("📈 Policies Sold Over Time")
monthly_policy_sales = df.groupby(df['year_month'].dt.strftime('%Y-%m'))['new_policy_count'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(12, 4))
sns.barplot(x="year_month", y="new_policy_count", data=monthly_policy_sales, palette="Set2", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Top Agents Comparison
st.subheader("🏆 Top vs Worst 20 Agents (by Policies Sold)")
top_20 = df.groupby("agent_code")["new_policy_count"].sum().sort_values(ascending=False).head(20)
bottom_20 = df.groupby("agent_code")["new_policy_count"].sum().sort_values(ascending=False).tail(20)

comparison_df = pd.concat([top_20, bottom_20])
comparison_df = comparison_df.reset_index()
comparison_df["Category"] = ["Top 20"] * 20 + ["Worst 20"] * 20

fig2, ax2 = plt.subplots(figsize=(14, 5))
sns.barplot(data=comparison_df, x="agent_code", y="new_policy_count", hue="Category", palette="Set2", ax=ax2)
plt.xticks(rotation=90)
st.pyplot(fig2)

# Correlation Heatmap
st.subheader("📌 Correlation Heatmap")
corr_df = df.drop(columns=['row_id', 'agent_code', 'agent_join_month', 'first_policy_sold_month', 'year_month'])
corr = corr_df.corr()

fig3, ax3 = plt.subplots(figsize=(16, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# Agent Analysis
st.subheader("🔍 Agent-Level Exploration")
agent_id = st.selectbox("Select an Agent", df['agent_code'].unique())
agent_data = df[df['agent_code'] == agent_id]

st.write(f"Details for Agent Code: **{agent_id}**")
st.dataframe(agent_data)