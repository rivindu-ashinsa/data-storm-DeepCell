import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Predictions Dashboard", layout="wide")

# Example Binary Classification Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/submission.csv")
    df["classification"] = df["prediction"].apply(lambda x: "Follow Up" if x == 1 else "No Follow Up")
    df["recommendation"] = df["prediction"].apply(lambda x: "Yes" if x == 1 else "No")
    return pd.DataFrame(df)

df = load_data()

st.title("🔍 Machine Learning Prediction Dashboard")

# 🔹 Summary Section
st.subheader("📊 Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Follow Up", df[df["classification"] == "Follow Up"].shape[0])
col3.metric("Avg Prediction Score", f"{df['prediction'].mean():.2f}")

# 🔹 Classification Bar Chart
st.subheader("📋 Classification Distribution")
class_counts = df["classification"].value_counts().reset_index()
class_counts.columns = ["Classification", "Count"]

fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.barplot(data=class_counts, x="Classification", y="Count", palette="Set2", ax=ax1)
st.pyplot(fig1)

# 🔹 Recommendation Pie Chart
st.subheader("🧭 Recommendation Overview")
rec_counts = df["recommendation"].value_counts()

fig2, ax2 = plt.subplots()
ax2.pie(rec_counts, labels=rec_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"), startangle=90)
ax2.axis("equal")
st.pyplot(fig2)

# 🔹 Individual Prediction View
st.subheader("🔎 Explore Individual Prediction")
selected_id = st.selectbox("Select Agent ID", df["agent_id"])
selected_row = df[df["agent_id"] == selected_id]
st.write("### 📌 Agent Details")
st.table(selected_row)

# 🔹 Downloadable Data
st.subheader("⬇️ Download Results")
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "binary_classification_results.csv", "text/csv")
