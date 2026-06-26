import base64
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation, label

PARAMS_PATH = Path("params.yaml")
METRICS_PATH = Path("trainedmodels/metrics.json")
MODELS_DIR = Path("trainedmodels")
DEFAULT_THRESHOLD = 0.05

# ── Doctor SVG illustration ───────────────────────────────────────────────────
DOCTOR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 245">
  <circle cx="100" cy="95" r="82" fill="#e0eff5"/>
  <rect x="46" y="130" width="108" height="100" rx="14" fill="#ffffff" stroke="#dce8f0" stroke-width="1.5"/>
  <path d="M100 130 L74 154 L84 184 L100 165 L116 184 L126 154 Z" fill="#f0f4f8" stroke="#dce8f0" stroke-width="1"/>
  <line x1="100" y1="165" x2="100" y2="226" stroke="#dce8f0" stroke-width="1.5"/>
  <rect x="58" y="168" width="24" height="20" rx="3" fill="#f0f4f8" stroke="#dce8f0" stroke-width="1"/>
  <rect x="87" y="116" width="26" height="20" rx="5" fill="#f5cba7"/>
  <circle cx="100" cy="89" r="34" fill="#f5cba7"/>
  <path d="M66 87 Q66 53 100 53 Q134 53 134 87 Q126 70 100 68 Q74 70 66 87 Z" fill="#2c3e50"/>
  <ellipse cx="66" cy="91" rx="5.5" ry="7.5" fill="#f0b98c"/>
  <ellipse cx="134" cy="91" rx="5.5" ry="7.5" fill="#f0b98c"/>
  <ellipse cx="87" cy="87" rx="4.5" ry="5" fill="#2c3e50"/>
  <ellipse cx="113" cy="87" rx="4.5" ry="5" fill="#2c3e50"/>
  <circle cx="89" cy="85.5" r="1.5" fill="white"/>
  <circle cx="115" cy="85.5" r="1.5" fill="white"/>
  <path d="M82 80 Q87 77 92 80" stroke="#2c3e50" stroke-width="2" fill="none" stroke-linecap="round"/>
  <path d="M108 80 Q113 77 118 80" stroke="#2c3e50" stroke-width="2" fill="none" stroke-linecap="round"/>
  <path d="M89 101 Q100 110 111 101" stroke="#c0856b" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <path d="M80 148 Q67 159 67 177 Q67 196 82 196 Q98 196 98 177"
        stroke="#1a6b8a" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="80" cy="148" r="4.5" fill="#1a6b8a"/>
  <circle cx="80" cy="148" r="2" fill="#2196b0"/>
  <circle cx="98" cy="177" r="8" fill="#1a6b8a"/>
  <circle cx="98" cy="177" r="5" fill="#2196b0"/>
  <rect x="116" y="148" width="30" height="40" rx="4" fill="#e8f4f8" stroke="#90bcd4" stroke-width="1.5"/>
  <rect x="122" y="143" width="18" height="9" rx="3" fill="#90bcd4"/>
  <line x1="121" y1="161" x2="141" y2="161" stroke="#90bcd4" stroke-width="1.8"/>
  <line x1="121" y1="169" x2="141" y2="169" stroke="#90bcd4" stroke-width="1.8"/>
  <line x1="121" y1="177" x2="135" y2="177" stroke="#90bcd4" stroke-width="1.8"/>
</svg>"""

_svg_b64 = base64.b64encode(DOCTOR_SVG.encode()).decode()
DOCTOR_IMG_TAG = (
    f'<img src="data:image/svg+xml;base64,{_svg_b64}" '
    'style="width:100%;max-width:175px;display:block;margin:0 auto 0.4rem auto;" '
    'alt="AI Radiologist" />'
)

# ── Injected CSS ──────────────────────────────────────────────────────────────
CSS_BLOCK = """<style>
:root {
    --primary: #1a6b8a;
    --primary-light: #2196b0;
    --danger: #c0392b;
    --danger-light: #fdf0ef;
    --success: #1b7a4e;
    --success-light: #edfaf4;
    --bg: #f0f4f8;
    --card-bg: #ffffff;
    --text-muted: #5a6a7e;
}
.stApp {
    background-color: var(--bg);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
[data-testid="stHeader"] {
    background-color: var(--bg) !important;
}
[data-testid="stToolbar"] {
    display: none;
}
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #eaf4fb !important;
    border: 2px dashed #1a6b8a !important;
    border-radius: 14px !important;
    padding: 2rem 1.5rem !important;
    text-align: center !important;
    transition: background 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #d6eaf8 !important;
    border-color: #2196b0 !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: #2a6090 !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: #5a6a7e !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: #1a6b8a !important;
    color: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    position: relative !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Upload";
    color: white;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    font-weight: 600;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #2196b0 !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1100px;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0d3b53 0%, #1a6b8a 55%, #1b8a72 100%);
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] label {
    color: #e8f4f8 !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
    color: #a8d4e8 !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .streamlit-expanderContent {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 0 0 6px 6px !important;
}
.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,0.18);
    margin: 0.9rem 0;
}
.app-header {
    background: linear-gradient(90deg, #0d3b53, #1a6b8a);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.4rem;
}
.app-header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff !important;
    padding: 0 !important;
}
.app-header p {
    margin: 0.2rem 0 0 0;
    color: rgba(255,255,255,0.8);
    font-size: 0.86rem;
}
.med-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    box-shadow: 0 2px 12px rgba(26,107,138,0.09);
    border: 1px solid #dce8f0;
    margin-bottom: 1rem;
}
.result-tumor {
    background: var(--danger-light);
    border-left: 6px solid var(--danger);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.result-healthy {
    background: var(--success-light);
    border-left: 6px solid var(--success);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.result-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.score-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.15rem;
}
.score-value {
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.45rem;
}
.score-bar-container {
    background: #dce8f0;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.35rem 0 0.7rem 0;
}
.score-bar-fill-tumor {
    background: linear-gradient(90deg, #e07b70, #c0392b);
    height: 100%;
    border-radius: 999px;
}
.score-bar-fill-healthy {
    background: linear-gradient(90deg, #48c78e, #1b7a4e);
    height: 100%;
    border-radius: 999px;
}
.confidence-badge {
    display: inline-block;
    padding: 0.2em 0.75em;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.conf-high   { background: #d4edda; color: #155724; }
.conf-medium { background: #fff3cd; color: #856404; }
.conf-low    { background: #f8d7da; color: #721c24; }
.info-card {
    background: #eaf4fb;
    border: 1px solid #b8d8ec;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #1a4a6b;
    text-align: center;
    font-size: 0.95rem;
}
.scan-section-header {
    color: var(--primary);
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.7rem;
}
.scan-label {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
</style>"""


# ── ML helpers (unchanged) ────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model(str(MODELS_DIR / "encoder.keras"))
    generator = tf.keras.models.load_model(str(MODELS_DIR / "generator.keras"))
    return encoder, generator


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_threshold():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            data = json.load(f)
        return float(data["optimal_threshold"]), "from evaluation"
    return DEFAULT_THRESHOLD, "default"



def preprocess_image(uploaded_file, img_size: int) -> np.ndarray:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ..., np.newaxis]


def compute_anomaly_score(arr, encoder, generator):
    z = encoder.predict(arr, verbose=0)
    recon = generator.predict(z, verbose=0)
    mse = float(np.mean(np.square(arr - recon)))
    ssim_val = tf.image.ssim(
        tf.convert_to_tensor(arr),
        tf.convert_to_tensor(recon),
        max_val=1.0,
    ).numpy()[0]
    score = 0.7 * mse + 0.3 * float(1.0 - ssim_val)
    return score, recon


def build_error_map(arr, recon):
    error = np.square(arr[0, ..., 0] - recon[0, ..., 0])
    error = gaussian_filter(error, sigma=2)
    error = (error - error.min()) / (error.max() - error.min() + 1e-8)
    return error


def make_figure(img_array, title, cmap="gray"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def make_overlay_figure(original, error_map, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax.imshow(error_map, cmap="jet", alpha=0.6, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def make_boundary_figure(original, error_map, title):
    # Mark top 15% of error pixels, isolate largest region, then expand to cover full tumor
    thresh = np.percentile(error_map, 82)
    raw_mask = error_map > thresh

    # Keep only the largest connected component so scattered pixels are ignored
    labeled_arr, num_features = label(raw_mask)
    if num_features > 0:
        sizes = [int((labeled_arr == i).sum()) for i in range(1, num_features + 1)]
        largest_label = int(np.argmax(sizes)) + 1
        region = labeled_arr == largest_label
    else:
        region = raw_mask

    # Dilate to expand the seed region outward, then fill internal holes
    mask = binary_fill_holes(binary_dilation(region, iterations=5))

    overlay = np.zeros((*original.shape, 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.1, 0.1, 0.35]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def get_confidence_level(score: float, threshold: float) -> tuple:
    distance = abs(score - threshold) / (threshold + 1e-9)
    if distance > 0.5:
        return "HIGH", "conf-high"
    elif distance > 0.2:
        return "MEDIUM", "conf-medium"
    return "LOW", "conf-low"


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="NeuroScan AI — Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS_BLOCK, unsafe_allow_html=True)

    params = load_params()
    img_size = params["IMG_SIZE"]
    threshold, threshold_source = load_threshold()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(DOCTOR_IMG_TAG, unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;margin-bottom:0.8rem;">'
            '<div style="font-size:1.05rem;font-weight:700;color:#e8f4f8;">Dr. Sarah Mitchell</div>'
            '<div style="font-size:0.78rem;color:#a8d4e8;margin-top:0.15rem;">Neuroradiology Specialist</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        with st.expander("How to use", expanded=True):
            st.markdown(
                "**Step 1** — Click the upload button and select your brain MRI image\n\n"
                "**Step 2** — Wait a few seconds for the AI to analyze your scan\n\n"
                "**Step 3** — View your result: clear and easy to understand\n\n"
                "---\n"
                "⚠ **Important:** This tool is for early screening only. "
                "Always visit your doctor for an official medical diagnosis."
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.68rem;color:#6a9ab0;text-align:center;padding-bottom:0.5rem;">'
            'For research use only.<br>Results require clinical confirmation.</div>',
            unsafe_allow_html=True,
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<div style="font-size:2.4rem;line-height:1;">🧠</div>'
        '<div>'
        '<h1>NeuroScan — AI Brain Health Check</h1>'
        '<p>AI-assisted brain MRI analysis &nbsp;·&nbsp; Upload your scan to get instant results</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Upload card ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="med-card">'
        '<div style="color:var(--primary);font-size:1.15rem;font-weight:700;margin-bottom:0.4rem;text-align:center;">'
        '&#128247; Upload Your Brain MRI Scan</div>'
        '<div style="color:var(--text-muted);font-size:0.87rem;text-align:center;">'
        'Please upload a brain MRI image in JPG or PNG format. '
        'The AI will analyze it and tell you whether a tumor is present.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "🧠 Drop your brain MRI scan here or click Browse files",
        type=["jpg", "jpeg", "png"],
        help="Upload a grayscale brain MRI image in JPG or PNG format",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="info-card">'
            '<div style="font-size:1.5rem;margin-bottom:0.5rem;">🩺</div>'
            '<div style="font-weight:600;margin-bottom:0.3rem;">Ready to analyze your scan</div>'
            '<div style="font-size:0.88rem;color:#2a6090;">Upload your brain MRI image above to receive an instant AI-powered screening result.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Analyzing your MRI scan, please wait…"):
        encoder, generator = load_models()
        arr = preprocess_image(uploaded_file, img_size)
        score, recon = compute_anomaly_score(arr, encoder, generator)

    is_tumor = score >= threshold
    bar_pct = min(100, int((score / (threshold * 3)) * 100))
    conf_label, conf_class = get_confidence_level(score, threshold)

    # ── Result banner ─────────────────────────────────────────────────────────
    if is_tumor:
        result_html = (
            '<div class="result-tumor">'
            '<div class="result-title" style="color:var(--danger);">&#9888; Signs of Tumor Found</div>'
            '<div style="color:#7a2020;font-size:0.92rem;margin:0.4rem 0 0.8rem 0;">'
            'The AI has detected abnormal patterns in your MRI scan that may indicate a tumor. '
            'Please consult a neurologist or specialist as soon as possible.</div>'
            '<div class="score-label">AI Certainty Level</div>'
            f'<div class="score-bar-container">'
            f'<div class="score-bar-fill-tumor" style="width:{bar_pct}%;"></div>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-top:0.2rem;">'
            f'<span class="confidence-badge {conf_class}">{conf_label} Certainty</span>'
            f'</div>'
            '</div>'
        )
    else:
        result_html = (
            '<div class="result-healthy">'
            '<div class="result-title" style="color:var(--success);">&#10003; No Tumor Signs Found</div>'
            '<div style="color:#1a5c3a;font-size:0.92rem;margin:0.4rem 0 0.8rem 0;">'
            'Your MRI scan appears normal. No signs of a tumor were detected. '
            'Continue with regular health check-ups as advised by your doctor.</div>'
            '<div class="score-label">AI Certainty Level</div>'
            f'<div class="score-bar-container">'
            f'<div class="score-bar-fill-healthy" style="width:{bar_pct}%;"></div>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-top:0.2rem;">'
            f'<span class="confidence-badge {conf_class}">{conf_label} Certainty</span>'
            f'</div>'
            '</div>'
        )

    st.markdown(result_html, unsafe_allow_html=True)

    # ── Scan display ──────────────────────────────────────────────────────────
    orig = arr[0, ..., 0]

    st.markdown(
        '<div class="med-card">'
        '<div class="scan-section-header">&#128247; Your MRI Scan</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if is_tumor:
        error_map = build_error_map(arr, recon)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="scan-label">Original MRI</div>', unsafe_allow_html=True)
            st.pyplot(make_figure(orig, "MRI Scan"))
            plt.close("all")
        with col2:
            st.markdown('<div class="scan-label">Area of Concern</div>', unsafe_allow_html=True)
            st.pyplot(make_overlay_figure(orig, error_map, "Area of Concern"))
            plt.close("all")
    else:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown('<div class="scan-label">Your MRI Scan</div>', unsafe_allow_html=True)
            st.pyplot(make_figure(orig, "MRI Scan"))
            plt.close("all")

    st.markdown(
        '<div style="margin-top:1.2rem;padding:0.9rem 1.2rem;background:#f8f9fa;'
        'border-radius:8px;border:1px solid #dee2e6;font-size:0.82rem;color:#6c757d;text-align:center;">'
        '&#9432; This AI screening tool is not a substitute for professional medical advice. '
        'Always consult a qualified doctor or specialist for diagnosis and treatment.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
