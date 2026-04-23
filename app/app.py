import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = (224, 224)

CLASS_INFO = {
    'Glioma': {
        'color': '#FF4B4B',
        'emoji': '🔴',
        'desc': 'A tumor that starts in the glial cells of the brain or spine. Requires immediate medical attention.'
    },
    'Meningioma': {
        'color': '#FF8C00',
        'emoji': '🟠',
        'desc': 'A tumor that forms on membranes covering the brain and spinal cord. Usually benign but needs monitoring.'
    },
    'No Tumor': {
        'color': '#00C851',
        'emoji': '🟢',
        'desc': 'No tumor detected in the MRI scan. Brain appears normal.'
    },
    'Pituitary': {
        'color': '#AA00FF',
        'emoji': '🟣',
        'desc': 'A tumor in the pituitary gland at the base of the brain. Often treatable.'
    }
}


# ── Load model (cached) ───────────────────────────────────
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model


# ── Grad-CAM ──────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(img_array_orig, heatmap, alpha=0.4):
    img = np.uint8(img_array_orig[0] * 255)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap_res = cv2.resize(heatmap, IMG_SIZE)
    heatmap_col = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_col, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def is_valid_mri(uploaded_file):
    """Check if image looks like a brain MRI scan"""
    img = Image.open(uploaded_file).convert('L')
    img_array = np.array(img, dtype=np.float32)
    avg_pixel = img_array.mean()
    std_pixel = img_array.std()

    # MRI scans: mostly dark background with bright brain region
    # Invalid if: completely black, completely white, or no variation
    if avg_pixel < 5:
        return False, "Image is too dark to be a valid MRI scan."
    if avg_pixel > 240:
        return False, "Image is too bright to be a valid MRI scan."
    if std_pixel < 20:
        return False, "Image has no variation — not a valid MRI scan."

    return True, "OK"


# ── UI ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Brain Tumor Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a brain MRI scan — AI will classify it and show where it looks</div>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **Deep Learning** to detect brain tumors from MRI scans.

    **Model:** MobileNetV2 Transfer Learning  
    **Accuracy:** 89.31% test | 96.25% val  
    **Classes:**
    - 🔴 Glioma
    - 🟠 Meningioma  
    - 🟢 No Tumor
    - 🟣 Pituitary

    **How it works:**
    1. Upload an MRI image
    2. AI classifies the tumor type
    3. Grad-CAM shows WHERE the AI looks

    ⚠️ *For educational purposes only.*
    """)

    st.header("📊 Model Performance")
    st.metric("Test Accuracy", "89.31%")
    st.metric("Val Accuracy", "96.25%")
    st.metric("Training Images", "4,480")

# Main upload
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI scan in JPG or PNG format"
)

if uploaded_file:
    # ── MRI validity check ─────────────────────────────────
    valid, reason = is_valid_mri(uploaded_file)
    if not valid:
        st.error(f"⚠️ Invalid image: {reason}")
        st.info("Please upload a real brain MRI scan. Sample images are in `dataset/Testing/` folder.")
        st.stop()

    uploaded_file.seek(0)  # reset file pointer after check

    with st.spinner("🔍 Analyzing MRI scan..."):
        model = load_trained_model()
        img_array = preprocess_image(uploaded_file)

        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        pred_class = np.argmax(predictions[0])
        pred_name = CLASS_NAMES[pred_class]
        confidence = predictions[0][pred_class] * 100
        info = CLASS_INFO[pred_name]
        overlay_img = overlay_heatmap(img_array, heatmap)
        original_img = np.uint8(img_array[0] * 255)

    # ── Prediction banner ──────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="prediction-box" style="background: {info['color']}22; border: 2px solid {info['color']};">
        <h1 style="color: {info['color']}; margin:0">{info['emoji']} {pred_name}</h1>
        <h3 style="color: {info['color']}; margin:0">Confidence: {confidence:.1f}%</h3>
        <p style="color: #ccc; margin-top: 0.5rem">{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Images ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original MRI")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        st.image(overlay_img, use_container_width=True)
        st.caption("🔴 Red/Yellow = AI focus area | 🔵 Blue = less important")

    st.markdown("---")

    # ── Confidence chart ───────────────────────────────────
    st.subheader("📊 Confidence for All Classes")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#FF4B4B', '#FF8C00', '#00C851', '#AA00FF']
    bars = ax.barh(CLASS_NAMES, predictions[0] * 100, color=colors)

    for bar, val in zip(bars, predictions[0] * 100):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=11
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Model confidence per class')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.warning(
        "⚠️ This tool is for educational purposes only. Always consult a qualified medical professional for diagnosis.")

else:
    st.info("👆 Upload a brain MRI image above to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 What it does")
        st.markdown("Classifies brain MRI scans into 4 categories using AI")
    with col2:
        st.markdown("### 🔥 Grad-CAM")
        st.markdown("Shows exactly which part of the MRI the AI focuses on")
    with col3:
        st.markdown("### 📊 Confidence")
        st.markdown("Shows probability scores for all 4 tumor types")