import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os

# ── GPU setup ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Config ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
IMG_SIZE   = (224, 224)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_INFO = {
    'Glioma'     : 'Malignant — Requires immediate medical attention',
    'Meningioma' : 'Usually benign — Needs monitoring and follow-up',
    'No Tumor'   : 'No tumor detected — Brain appears normal',
    'Pituitary'  : 'Pituitary tumor — Often treatable with surgery or medication',
}

CLASS_SEVERITY = {
    'Glioma'     : 'high',
    'Meningioma' : 'medium',
    'No Tumor'   : 'none',
    'Pituitary'  : 'medium',
}

# ── Global model instance (loaded once at startup) ────────
_model = None


def load_model_once():
    """Called once at FastAPI startup via lifespan."""
    global _model
    if _model is None:
        print(f"📦 Loading model from: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
        print("✅ Model ready")
    return _model


def get_model():
    """Returns the already-loaded model. Raises if not loaded yet."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_once() first.")
    return _model


def preprocess_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → preprocessed numpy array.
    Uses rescale=1/255 (matching MobileNetV2 training pipeline).
    """
    img = Image.open(BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def predict_from_bytes(img_bytes: bytes) -> dict:
    """
    Main prediction function called by the /predict endpoint.

    Returns:
        {
            'class'      : str,
            'confidence' : float,
            'all_scores' : { class_name: float },
            'info'       : str,
            'severity'   : str,
        }
    """
    model     = get_model()
    img_array = preprocess_bytes(img_bytes)

    predictions = model.predict(img_array, verbose=0)
    pred_index  = int(np.argmax(predictions[0]))
    pred_class  = CLASS_NAMES[pred_index]
    confidence  = float(predictions[0][pred_index]) * 100

    all_scores = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        'class'      : pred_class,
        'confidence' : round(confidence, 2),
        'all_scores' : all_scores,
        'info'       : CLASS_INFO[pred_class],
        'severity'   : CLASS_SEVERITY[pred_class],
    }