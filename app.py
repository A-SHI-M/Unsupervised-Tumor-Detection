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
METRICS_PATH = Path("trainedmodels/metrics.json")
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
    error = gaussian_filter(error, sigma=3)
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
    ax.imshow(error_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
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

    st.title("Brain Tumor Detection")

    if threshold_source == "default":
        st.warning("Evaluation metrics not found — using default threshold (0.05).")

    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to start analysis.")
        return

    with st.spinner("Analyzing image…"):
        encoder, generator = load_models()
        arr = preprocess_image(uploaded_file, img_size)
        score, recon = compute_anomaly_score(arr, encoder, generator)

    is_tumor = score >= threshold

    if is_tumor:
        st.error("⚠️ TUMOR DETECTED")
    else:
        st.success("✅ HEALTHY")

    orig = arr[0, ..., 0]

    if is_tumor:
        error_map = build_error_map(arr, recon)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(make_figure(orig, "MRI Scan"))
            plt.close("all")
        with col2:
            st.pyplot(make_overlay_figure(orig, error_map, "Tumor Region"))
            plt.close("all")
    else:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.pyplot(make_figure(orig, "MRI Scan"))
            plt.close("all")


if __name__ == "__main__":
    main()
