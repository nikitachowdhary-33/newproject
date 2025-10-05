# src/frontend/streamlit_app.py
import streamlit as st
import os
import runpy
from src.backend.predict import predict_news

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    page_icon="📰",
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px !important;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #ff4b4b, #1f77b4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-card {
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-prob {
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<p class="big-title">🕵️‍♀️ Fake News Detector</p>', unsafe_allow_html=True)
st.caption("A smart AI-powered tool to detect whether news looks **REAL or FAKE**.")

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Controls")
st.sidebar.info("Upload or paste text, then analyze. You can also retrain the model here.")
if st.sidebar.button("🔄 (Re)Train Model"):
    with st.spinner("Training model... please wait ⏳"):
        try:
            runpy.run_path(
                os.path.join(os.path.dirname(__file__), "..", "backend", "train_model.py"),
                run_name="__main__"
            )
            st.sidebar.success("✅ Training finished! Try analyzing again.")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Made with ❤️ by Students**")

# ------------------ Input ------------------
st.subheader("📥 Input")
text = st.text_area("Paste article text here (title + body):", height=200)

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "model.joblib")
VEC_PATH = os.path.join(BASE_DIR, "..", "..", "models", "vectorizer.joblib")

# ------------------ Main Action ------------------
if st.button("🔎 Analyze Now", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please paste some text first.")
    elif not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        st.error("❌ Model files not found. Please train the model first.")
    else:
        with st.spinner("Analyzing article with AI..."):
            try:
                result = predict_news(text)

                # Short summary
                sents = text.strip().split(". ")
                summary = ". ".join(sents[:2]).strip()
                if summary and not summary.endswith("."):
                    summary += "."

                # ------------------ Results Layout ------------------
                st.subheader("📊 Results")

                col1, col2 = st.columns([2, 1])

                with col1:
                    label = result.get("label")
                    prob = result.get("probability_fake")

                    # Result card
                    if label == "FAKE":
                        st.markdown(
                            '<div class="result-card" style="background:#ffe6e6;"><h3>🚨 Predicted: FAKE</h3></div>',
                            unsafe_allow_html=True
                        )
                    elif label == "REAL":
                        st.markdown(
                            '<div class="result-card" style="background:#e6ffe6;"><h3>✅ Predicted: REAL</h3></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info(f"Label: {label}")

                    if prob is not None:
                        st.progress(min(max(prob, 0.0), 1.0))
                        st.markdown(f"<p class='metric-prob'>Probability Fake: {prob:.2%}</p>", unsafe_allow_html=True)

                with col2:
                    st.markdown("### 📝 Summary")
                    st.info(summary if summary else "No summary available.")

                # ------------------ Tabs ------------------
                st.markdown("### 🔍 Details")
                tab1, tab2 = st.tabs(["Explanation", "Raw Output"])

                with tab1:
                    if result.get("explanation"):
                        for e in result["explanation"]:
                            st.write(f"- **{e['feature']}** — weight `{e['weight']:.4f}`")
                    else:
                        st.write("No explanation available.")

                with tab2:
                    st.json(result)

            except Exception as e:
                st.error(f"Prediction failed: {e}")