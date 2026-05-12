import streamlit as st
import torch
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


#  PAGE CONFIG  
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0f0c29 100%);
    color: #e8e4f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #c9c1e0 !important; }

/* ── Hero header ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #7c6fa0;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.6rem;
}
.card-content {
    font-size: 0.97rem;
    line-height: 1.75;
    color: #ddd6fe;
}

/* ── Metric chips ── */
.metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.25rem;
}
.metric-chip {
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.82rem;
    color: #c4b5fd;
    font-weight: 500;
}
.metric-chip span { color: #f0abfc; font-weight: 700; }

/* ── Method badge ── */
.badge-transformer {
    display: inline-block;
    background: linear-gradient(90deg,#6d28d9,#2563eb);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}
.badge-baseline {
    display: inline-block;
    background: linear-gradient(90deg,#0f766e,#0284c7);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
}

/* ── Inputs & buttons ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2d9f3 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.45);
}
.stSelectbox div, .stRadio div { color: #c9c1e0 !important; }
.stSlider > div { color: #c9c1e0; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Warning / info ── */
.stAlert { border-radius: 12px !important; }

/* ── Spinner text ── */
.stSpinner > div { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)



#  HELPER FUNCTIONS
def sentence_split(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [s.strip() for s in text.split(".") if len(s.strip()) > 5]


def chunk_text(text: str, tokenizer, max_tokens: int = 900) -> list[str]:
    sentences = sentence_split(text)
    chunks, current, count = [], [], 0
    for s in sentences:
        t = len(tokenizer.encode(s, add_special_tokens=False))
        if current and count + t > max_tokens:
            chunks.append(" ".join(current))
            current, count = [s], t
        else:
            current.append(s)
            count += t
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:4000]]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load model once and cache it across sessions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try loading from saved bundle first
    bundle_path = "features/advanced_model.pkl"
    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"].to(device).eval()
        source = "bundle"
    else:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
        source = MODEL_NAME

    return tokenizer, model, device, source


def transformer_summarize(text, tokenizer, model, device,
                           max_input_tokens=512, max_length=80,
                           min_length=20, num_beams=2) -> str:
    if not text.strip():
        return ""
    chunks = chunk_text(text, tokenizer, max_input_tokens)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk, return_tensors="pt",
            max_length=max_input_tokens, truncation=True
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_length=max_length, min_length=min_length,
                num_beams=num_beams, length_penalty=1.5,
                early_stopping=True, no_repeat_ngram_size=3,
            )
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(summaries)


def baseline_summarize(text: str, top_n: int = 3) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return ""
    try:
        tfidf = TfidfVectorizer().fit_transform(sentences)
        scores = tfidf.mean(axis=1).A1
        top_idx = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        )
        return " ".join(sentences[i] for i in top_idx)
    except Exception:
        return " ".join(sentences[:top_n])


def compression_pct(original: str, summary: str) -> str:
    o = max(len(original.split()), 1)
    s = len(summary.split())
    return f"{s / o * 100:.1f}%"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    method = st.radio(
        "Summarization method",
        ["🤖 Transformer + 📊 Baseline", "🤖 Transformer only", "📊 Baseline only"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Transformer Parameters**")
    max_length  = st.slider("Max output length (tokens)", 30, 200, 80, 5)
    min_length  = st.slider("Min output length (tokens)", 5,  80,  20, 5)
    num_beams   = st.slider("Beam search width",          1,  5,   2,  1)
    top_n       = st.slider("Baseline — top N sentences", 1,  6,   3,  1)

    st.markdown("---")
    st.markdown("**Model Info**")
    device_label = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU"
    st.info(f"Running on: **{device_label}**")

    st.markdown("---")
    st.caption("Built with Streamlit · DistilBART · TF-IDF")


# ─────────────────────────────────────────────
#  MAIN — HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">Intelligent Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">TRANSFORMER · TF-IDF BASELINE · SIDE-BY-SIDE COMPARISON</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### 📝 Input Article")

    input_mode = st.segmented_control(
        "Input mode", ["Type / Paste", "Upload .txt"], default="Type / Paste"
    )

    article_text = ""

    if input_mode == "Upload .txt":
        uploaded = st.file_uploader("Drop a .txt file here", type=["txt"], label_visibility="collapsed")
        if uploaded:
            article_text = uploaded.read().decode("utf-8", errors="ignore")
            st.success(f"✅ Loaded — {len(article_text.split()):,} words")
    else:
        article_text = st.text_area(
            "Paste your article",
            height=280,
            placeholder="Paste any news article, report, or long text here...",
            label_visibility="collapsed",
        )

    word_count = len(article_text.split()) if article_text.strip() else 0
    st.caption(f"Word count: **{word_count:,}**")

    if word_count > 0:
        st.progress(min(word_count / 2000, 1.0))

    run_btn = st.button("✨ Summarize Now", use_container_width=True)


# ─────────────────────────────────────────────
#  OUTPUT SECTION
# ─────────────────────────────────────────────
with col_output:
    st.markdown("#### 📄 Summary Output")

    if run_btn:
        if not article_text.strip():
            st.warning("⚠️ Please enter some text first.")
        elif word_count < 20:
            st.warning("⚠️ Text is too short to summarize (need at least 20 words).")
        else:
            run_transformer = "Transformer" in method
            run_baseline    = "Baseline"    in method

            adv_result  = ""
            base_result = ""

            # ── Load model ──────────────────────────
            with st.spinner("Loading model (first run may take ~30s)..."):
                try:
                    tokenizer, model, device, src = load_model()
                except Exception as e:
                    st.error(f"Model load failed: {e}")
                    st.stop()

            # ── Transformer inference ────────────────
            if run_transformer:
                with st.spinner("🤖 Transformer is thinking..."):
                    adv_result = transformer_summarize(
                        article_text, tokenizer, model, device,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                    )

            # ── Baseline inference ───────────────────
            if run_baseline:
                with st.spinner("📊 Computing TF-IDF scores..."):
                    base_result = baseline_summarize(article_text, top_n=top_n)

            # ── Display results ──────────────────────
            if adv_result:
                st.markdown('<span class="badge-transformer">🤖 TRANSFORMER (DistilBART)</span>', unsafe_allow_html=True)
                adv_comp = compression_pct(article_text, adv_result)
                st.markdown(f"""
                <div class="glass-card">
                  <div class="card-content">{adv_result}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-chip">Words <span>{len(adv_result.split())}</span></div>
                  <div class="metric-chip">Compression <span>{adv_comp}</span></div>
                  <div class="metric-chip">Beams <span>{num_beams}</span></div>
                </div>
                """, unsafe_allow_html=True)

            if adv_result and base_result:
                st.markdown("<br>", unsafe_allow_html=True)

            if base_result:
                st.markdown('<span class="badge-baseline">📊 BASELINE (TF-IDF Extractive)</span>', unsafe_allow_html=True)
                base_comp = compression_pct(article_text, base_result)
                st.markdown(f"""
                <div class="glass-card">
                  <div class="card-content">{base_result}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-chip">Words <span>{len(base_result.split())}</span></div>
                  <div class="metric-chip">Compression <span>{base_comp}</span></div>
                  <div class="metric-chip">Top-N <span>{top_n}</span></div>
                </div>
                """, unsafe_allow_html=True)

            # ── Download button ──────────────────────
            if adv_result or base_result:
                st.markdown("<br>", unsafe_allow_html=True)
                export_text = ""
                if adv_result:
                    export_text += f"=== TRANSFORMER SUMMARY ===\n{adv_result}\n\n"
                if base_result:
                    export_text += f"=== BASELINE SUMMARY ===\n{base_result}\n"
                st.download_button(
                    "⬇️ Download Summary (.txt)",
                    data=export_text,
                    file_name="summary_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 1.5rem; opacity:0.5;">
          <div style="font-size:2.5rem; margin-bottom:1rem;">✨</div>
          <div style="font-size:0.95rem; color:#9f8ec4;">
            Paste your article on the left<br>and hit <strong>Summarize Now</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  EXAMPLE ARTICLES EXPANDER
# ─────────────────────────────────────────────
with st.expander("💡 Try an example article"):
    st.markdown("""
**Copy and paste this into the input box:**

> The United Nations has warned that the global food crisis is worsening rapidly, 
with more than 800 million people going to bed hungry every night. 
Climate change, armed conflicts, and economic shocks are pushing food insecurity 
to record levels not seen in decades. The World Food Programme has called for 
urgent international action to prevent famine in several regions, particularly 
in sub-Saharan Africa and parts of South Asia. Experts say that without 
immediate funding and logistical support, the situation could deteriorate 
significantly over the next 12 months, affecting hundreds of millions more people.
""")
