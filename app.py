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
from scipy.ndimage import gaussian_filter

PARAMS_PATH = Path("params.yaml")
METRICS_PATH = Path("artifacts/model_evaluation/metrics.json")
MODELS_DIR = Path("trainedmodels")
DEFAULT_THRESHOLD = 0.05


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
    return DEFAULT_THRESHOLD, "default (run stage 04 for calibrated threshold)"


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
    ssim_err = float(1.0 - ssim_val)
    score = 0.7 * mse + 0.3 * ssim_err
    return score, recon


def build_error_map(arr, recon):
    error = np.square(arr[0, ..., 0] - recon[0, ..., 0])
    error = gaussian_filter(error, sigma=3)
    vmin, vmax = error.min(), error.max()
    error = (error - vmin) / (vmax - vmin + 1e-8)
    return error


def render_image_panel(title, img_array, cmap="gray"):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_array, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def render_overlay(original, error_map):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax.imshow(error_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title("Overlay", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def main():
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
    )

    params = load_params()
    img_size = params["IMG_SIZE"]
    threshold, threshold_source = load_threshold()

    with st.sidebar:
        st.header("Model Info")
        st.markdown(f"**Image size:** {img_size}×{img_size}")
        st.markdown(f"**Threshold:** `{threshold:.4f}` ({threshold_source})")
        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "A BiGAN model is trained on healthy brain MRIs only. "
            "At inference, the encoder maps the uploaded image to a latent code, "
            "and the generator reconstructs it. Healthy images reconstruct well; "
            "tumor regions cause high reconstruction error. "
            "The anomaly score combines 70% pixel MSE and 30% SSIM error."
        )

    st.title("🧠 Brain Tumor Detection — Anomaly Analysis")
    st.markdown("Upload a brain MRI scan to check for anomalies using unsupervised reconstruction-based detection.")

    if threshold_source == "default (run stage 04 for calibrated threshold)":
        st.warning("Stage 04 evaluation has not been run. Using default threshold — results may be less accurate.")

    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png"],
        help="Grayscale or color JPEG/PNG — will be converted to grayscale internally",
    )

    if uploaded_file is None:
        st.info("Upload an image above to start analysis.")
        return

    with st.spinner("Loading models and analyzing image…"):
        encoder, generator = load_models()
        arr = preprocess_image(uploaded_file, img_size)
        score, recon = compute_anomaly_score(arr, encoder, generator)

    is_tumor = score >= threshold

    st.divider()

    if is_tumor:
        st.error("⚠️ TUMOR DETECTED")
    else:
        st.success("✅ HEALTHY")

    col_score, col_thresh, col_status = st.columns(3)
    col_score.metric("Anomaly Score", f"{score:.4f}")
    col_thresh.metric("Threshold", f"{threshold:.4f}")
    col_status.metric("Status", "Abnormal" if is_tumor else "Normal")

    st.divider()

    orig = arr[0, ..., 0]
    rec = recon[0, ..., 0]

    if is_tumor:
        error_map = build_error_map(arr, recon)

        st.markdown("#### Reconstruction Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.pyplot(render_image_panel("Original", orig))
        with c2:
            st.pyplot(render_image_panel("Reconstruction", rec))
        with c3:
            fig, ax = plt.subplots(figsize=(3, 3))
            im = ax.imshow(error_map, cmap="jet", vmin=0, vmax=1)
            ax.set_title("Error Heatmap", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(pad=0.5)
            st.pyplot(fig)
            plt.close(fig)
        with c4:
            st.pyplot(render_overlay(orig, error_map))

        st.info(
            "Red/yellow regions show where the model's reconstruction error is highest — "
            "these areas deviate most from what a healthy brain MRI should look like."
        )
    else:
        st.markdown("#### Reconstruction Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(render_image_panel("Original", orig))
        with c2:
            st.pyplot(render_image_panel("Reconstruction", rec))
        st.info("The reconstruction closely matches the original — no significant anomaly detected.")


if __name__ == "__main__":
    main()
