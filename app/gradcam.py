import tensorflow as tf
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO

from predictor import get_model, preprocess_bytes, IMG_SIZE


# ── Find last Conv2D layer ────────────────────────────────
def get_last_conv_layer(model) -> str:
    """
    Walks model layers in reverse to find the last Conv2D.
    For MobileNetV2 this will be inside the base model sub-layers.
    """
    # First try top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # If model is nested (MobileNetV2 as a sub-model), dig inside
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name

    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str):
    """
    Identical logic to your 5_gradcam.py — just accepts numpy array directly.
    Returns (heatmap, predictions).
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index   = tf.argmax(predictions[0])
        class_score  = predictions[:, pred_index]

    grads        = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap_bytes(img_bytes: bytes, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Adapted from your overlay_heatmap() — works with raw bytes instead of file path.
    Returns (original_rgb, heatmap_rgb, overlay_rgb) as numpy arrays.
    """
    img     = Image.open(BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    img_rgb = np.array(img)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

    return img_rgb, heatmap_rgb, overlay


def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy RGB array → base64 PNG string for JSON response."""
    img     = Image.fromarray(img_array.astype(np.uint8))
    buffer  = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_gradcam(img_bytes: bytes) -> dict:
    """
    Main function called by /predict endpoint.
    Returns base64-encoded original, heatmap, and overlay images.
    """
    model           = get_model()
    img_array       = preprocess_bytes(img_bytes)
    last_conv_layer = get_last_conv_layer(model)

    heatmap, _ = make_gradcam_heatmap(img_array, model, last_conv_layer)
    original, heat, overlay = overlay_heatmap_bytes(img_bytes, heatmap)

    return {
        'original' : numpy_to_base64(original),
        'heatmap'  : numpy_to_base64(heat),
        'overlay'  : numpy_to_base64(overlay),
    }