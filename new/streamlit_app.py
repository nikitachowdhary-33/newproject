# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import random

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    page_icon="📰",
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.big-title { font-size:42px !important; font-weight:800; background:-webkit-linear-gradient(45deg,#ff4b4b,#1f77b4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.result-card { padding:1.5rem; border-radius:15px; box-shadow:0px 4px 15px rgba(0,0,0,0.08); margin-bottom:20px; }
.metric-prob { font-size:22px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<p class="big-title">🕵️‍♀️ Fake News Detector</p>', unsafe_allow_html=True)
st.caption("Paste news text or select an article. Predictions are randomized for demo.")

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Controls")
st.sidebar.info("Select an article from the dataset or paste custom text.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Made with ❤️ by Students**")

# ------------------ Load Dataset ------------------
DATA_PATH = os.path.join("data", "train.csv")
df = pd.read_csv(DATA_PATH)

# ------------------ Select News ------------------
st.subheader("📥 Select News from Dataset")
selected_title = st.selectbox("Choose an article title:", df["title"].tolist())
selected_row = df[df["title"] == selected_title].iloc[0]
text = selected_row["text"]

# Allow editing
text = st.text_area("News text (editable):", value=text, height=150)

# ------------------ Analyze Button ------------------
if st.button("🔎 Analyze Now", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        # Demo random prediction (replace with backend later)
        label = random.choice(["REAL", "FAKE"])
        probability_fake = random.uniform(0.0, 1.0)

        # Short summary (first 2 sentences)
        sents = text.strip().split(". ")
        summary = ". ".join(sents[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."

        # ------------------ Results ------------------
        st.subheader("📊 Results")
        col1, col2 = st.columns([2,1])

        with col1:
            if label == "FAKE":
                st.markdown('<div class="result-card" style="background:#ffe6e6;"><h3>🚨 Predicted: FAKE</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card" style="background:#e6ffe6;"><h3>✅ Predicted: REAL</h3></div>', unsafe_allow_html=True)

            st.progress(min(max(probability_fake, 0.0), 1.0))
            st.markdown(f"<p class='metric-prob'>Probability Fake: {probability_fake:.2%}</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 📝 Summary")
            st.info(summary if summary else "No summary available.")

        # ------------------ Tabs ------------------
        st.markdown("### 🔍 Details")
        tab1, tab2 = st.tabs(["Explanation", "Raw Output"])

        with tab1:
            st.write("Explanation will appear here when backend is connected.")

        with tab2:
            st.json({
                "title": selected_row["title"],
                "text": text,
                "label": label,
                "probability_fake": round(probability_fake,3),
                "summary": summary
            })