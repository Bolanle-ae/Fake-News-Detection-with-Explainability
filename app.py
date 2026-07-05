import re
import string
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #F5F3FF 0%, #EFF6FF 45%, #ECFEFF 100%);
    }

    .hero {
        background: linear-gradient(120deg, #7C3AED 0%, #EC4899 55%, #F59E0B 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-size: 2.1rem;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
    }

    .card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(124, 58, 237, 0.12);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 18px rgba(31, 41, 55, 0.06);
        margin-bottom: 1rem;
    }

    .badge-fake {
        display: inline-block;
        background: linear-gradient(120deg, #F43F5E, #F97316);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 0.6rem 1.4rem;
        border-radius: 999px;
        box-shadow: 0 6px 16px rgba(244, 63, 94, 0.35);
    }
    .badge-real {
        display: inline-block;
        background: linear-gradient(120deg, #10B981, #22D3EE);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 0.6rem 1.4rem;
        border-radius: 999px;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35);
    }

    .stButton>button {
        background: linear-gradient(120deg, #7C3AED, #EC4899);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        color: white;
        border: none;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EDE9FE 0%, #E0F2FE 100%);
    }

    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 0.85rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="Loading BERT model…")
def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def predict_proba(texts, tokenizer, model, device):
    import torch

    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>📰 Fake News Detection with Explainability</h1>
        <p>Powered by <b>BERT</b> for classification and <b>LIME</b> for word-level explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar settings ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_lime_features = st.slider("LIME: top features to highlight", 5, 20, 10)
    num_lime_samples = st.slider(
        "LIME: samples (higher = slower, more accurate)", 20, 200, 50, step=10
    )
    run_lime = st.checkbox("Generate LIME explanation", value=True)
    st.caption(
        "LIME runs multiple BERT forward passes — it may take 30–90 s on CPU."
    )
    st.divider()
    st.markdown(
        "**Model:** bert-base-uncased  \n"
        "**Explainability:** LIME  \n"
        "**Dataset:** Fake.csv / True.csv"
    )

# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📝 Enter an article")
title_input = st.text_input(
    "Article title", placeholder="Enter the news headline…"
)
text_input = st.text_area(
    "Article body",
    placeholder="Paste the full article text here…",
    height=200,
)
subject_input = st.text_input(
    "Subject / category (optional)", placeholder="e.g. politics, world news…"
)
analyse_clicked = st.button("🔍 Analyse article", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if analyse_clicked:
    raw = " ".join(filter(None, [title_input, text_input, subject_input]))

    if not raw.strip():
        st.warning("Please enter at least a title or article body.")
        st.stop()

    tokenizer, model, device = load_model()
    cleaned = clean_text(raw)
    CLASS_NAMES = ["Fake News", "Real News"]

    # ── Prediction ────────────────────────────────────────────────────────────
    with st.spinner("Running prediction…"):
        probs = predict_proba([cleaned], tokenizer, model, device)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Prediction")
    result_col, meter_col = st.columns([1, 2])

    with result_col:
        badge_class = "badge-fake" if pred_idx == 0 else "badge-real"
        st.markdown(
            f'<span class="{badge_class}">{pred_label}</span>', unsafe_allow_html=True
        )
        st.metric("Confidence", f"{confidence:.1%}")

    with meter_col:
        import pandas as pd

        chart_data = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": probs}
        ).set_index("Class")
        st.bar_chart(
            chart_data,
            height=180,
            color=["#F43F5E", "#10B981"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── LIME ──────────────────────────────────────────────────────────────────
    if run_lime:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 LIME Explanation")
        st.caption(
            "Words in **green** push the prediction toward Real News; "
            "words in **red** push toward Fake News."
        )

        with st.spinner("Running LIME (this may take a minute on CPU)…"):
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=CLASS_NAMES)

            def _predict(texts):
                return predict_proba(texts, tokenizer, model, device)

            exp = explainer.explain_instance(
                cleaned,
                _predict,
                num_features=num_lime_features,
                num_samples=num_lime_samples,
                labels=[pred_idx],
            )

        # Render LIME HTML inline
        html = exp.as_html()
        st.components.v1.html(html, height=500, scrolling=True)

        # Also show ranked word table
        st.markdown("##### Top influential words")
        word_weights = exp.as_list(label=pred_idx)
        df_words = pd.DataFrame(word_weights, columns=["Word", "Weight"])
        df_words["Direction"] = df_words["Weight"].apply(
            lambda w: "↑ Real News" if w > 0 else "↓ Fake News"
        )
        df_words["Weight"] = df_words["Weight"].abs()
        df_words = df_words.sort_values("Weight", ascending=False)
        st.dataframe(df_words, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">Model: bert-base-uncased (pretrained, not fine-tuned on this dataset) · '
    "Explainability: LIME · Dataset: Fake.csv / True.csv</p>",
    unsafe_allow_html=True,
)
