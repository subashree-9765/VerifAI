#!/usr/bin/env python3
"""Modern Streamlit UI for Fake News Detector"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

from detect_fake_news import classify_probability
from model_compat import load_pipeline as load_model_pipeline


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_path() -> Path:
    return project_root() / "outputs" / "pipeline.joblib"


def default_metrics_path() -> Path:
    return project_root() / "outputs" / "metrics.json"


@st.cache_resource
def load_pipeline(path: str):
    return load_model_pipeline(path)


def load_metrics(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_short_input(text: str):
    tokens = [token for token in text.strip().split() if token]
    return len(tokens) < 8 or len(text.strip()) < 50


def inject_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    .fake {
        background-color: rgba(255, 75, 75, 0.2);
        border: 2px solid red;
        color: red;
    }

    .real {
        background-color: rgba(50, 205, 50, 0.2);
        border: 2px solid limegreen;
        color: limegreen;
    }

    .uncertain {
        background-color: rgba(255, 165, 0, 0.2);
        border: 2px solid orange;
        color: orange;
    }

    .footer {
        text-align: center;
        color: gray;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=str(default_pipeline_path()))
    args, _ = parser.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    metrics_path = default_metrics_path()

    st.set_page_config(
        page_title="VerifAI",
        page_icon="📰",
        layout="wide"
    )

    inject_custom_css()

    st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-powered fake news style classification using TF-IDF + Logistic Regression</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("⚙ Model Info")

        metrics = load_metrics(metrics_path)
        if metrics:
            test = metrics.get("holdout_test", {})

            st.metric("Accuracy", f"{test.get('accuracy', 0):.3f}")
            st.metric("Macro F1", f"{test.get('macro_f1', 0):.3f}")
            st.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")

        # st.info("Educational project. Not for real-world fact verification.")

    if not pipeline_path.exists():
        st.error("Model not found. Run train_model.py first.")
        st.stop()

    pipeline = load_pipeline(str(pipeline_path))

    col1, col2 = st.columns([3, 1])

    with col1:
        text = st.text_area(
            "Paste your news article or headline",
            height=250,
            placeholder="Enter article content here..."
        )

    with col2:
        threshold = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01)
        uncertainty_margin = st.slider("Uncertainty", 0.00, 0.30, 0.10, 0.01)

    if st.button("🔍 Analyze News", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
            st.stop()

        prob_fake = float(pipeline.predict_proba([text])[0, 1])
        label = classify_probability(prob_fake, threshold, uncertainty_margin)

        if label == "FAKE":
            st.markdown(
                f'<div class="result-box fake">🚨 FAKE NEWS ({prob_fake:.1%})</div>',
                unsafe_allow_html=True
            )
            st.progress(prob_fake)

        elif label == "REAL":
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS ({1 - prob_fake:.1%})</div>',
                unsafe_allow_html=True
            )
            st.progress(1 - prob_fake)

        else:
            st.markdown(
                f'<div class="result-box uncertain">⚠ UNCERTAIN ({prob_fake:.1%})</div>',
                unsafe_allow_html=True
            )

        if is_short_input(text):
            st.warning("Input is too short. Try using a longer article for better results.")

        with st.expander("📊 Detailed Analysis"):
            st.write(f"Fake Probability: **{prob_fake:.2%}**")
            st.write(f"Threshold: **{threshold:.2f}**")
            st.write(f"Uncertainty Margin: **{uncertainty_margin:.2f}**")


    st.markdown("---")
    st.subheader("👨‍💻 Team Members")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Subashree**\n\nModel Training")

    with col2:
        st.info("**Monisha**\n\nProject development")

    with col3:
        st.info("**Keshani Karthik**\n\nUI Design")




if __name__ == "__main__":
    main()
