import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── GPU setup ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
TEST_DIR    = os.path.join(BASE_DIR, "dataset", "Testing")
OUTPUT_DIR  = os.path.join(BASE_DIR, "gradcam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = (224, 224)

# ── Load model ────────────────────────────────────────────
print("📦 Loading best model...")
model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))
print("✅ Model loaded!")

# ── Find last conv layer ──────────────────────────────────
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"✅ Last conv layer: {layer.name}")
            return layer.name
    raise ValueError("No Conv2D layer found!")

last_conv_layer = get_last_conv_layer(model)

# ── Grad-CAM function ─────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Model that outputs conv layer + predictions
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

    # Gradients of class score w.r.t conv output
    grads       = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()

# ── Overlay heatmap on image ──────────────────────────────
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img     = cv2.imread(img_path)
    img     = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(img_rgb, 1-alpha, heatmap_rgb, alpha, 0)

    return img_rgb, heatmap_rgb, overlay

# ── Load and preprocess one image ─────────────────────────
def preprocess_image(img_path):
    img       = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Run Grad-CAM on sample images from each class ────────
print("\n🔥 Generating Grad-CAM heatmaps...\n")

fig, axes = plt.subplots(4, 3, figsize=(14, 18))
fig.suptitle('Grad-CAM — Where the AI looks to detect tumors',
             fontsize=15, fontweight='bold', y=1.01)

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_folder = os.path.join(TEST_DIR, class_name.lower().replace(" ", ""))

    # Handle "No Tumor" folder name
    if not os.path.exists(class_folder):
        class_folder = os.path.join(TEST_DIR, "notumor")

    # Pick first image from class
    images_in_class = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not images_in_class:
        continue

    img_path  = os.path.join(class_folder, images_in_class[0])
    img_array = preprocess_image(img_path)

    # Generate heatmap
    heatmap, predictions = make_gradcam_heatmap(img_array, model, last_conv_layer)
    pred_class      = np.argmax(predictions[0])
    confidence      = predictions[0][pred_class] * 100
    original, heat, overlay = overlay_heatmap(img_path, heatmap)

    # Plot row: original | heatmap | overlay
    ax_orig    = axes[class_idx][0]
    ax_heat    = axes[class_idx][1]
    ax_overlay = axes[class_idx][2]

    ax_orig.imshow(original)
    ax_orig.set_title(f'Original\n({class_name})', fontsize=10)
    ax_orig.axis('off')

    ax_heat.imshow(heat)
    ax_heat.set_title('Grad-CAM Heatmap', fontsize=10)
    ax_heat.axis('off')

    color  = 'green' if pred_class == class_idx else 'red'
    status = '✓ Correct' if pred_class == class_idx else f'✗ Predicted: {CLASS_NAMES[pred_class]}'
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(
        f'Overlay — {status}\nConfidence: {confidence:.1f}%',
        fontsize=10, color=color
    )
    ax_overlay.axis('off')

    print(f"  {class_name:12} → Predicted: {CLASS_NAMES[pred_class]:12} "
          f"| Confidence: {confidence:.1f}% | {'✓' if pred_class == class_idx else '✗'}")

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "gradcam_all_classes.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n📸 Grad-CAM saved to: {save_path}")
print("\n✅ Grad-CAM complete! The heatmap shows WHERE the AI looks to make predictions.")
print("   Red/Yellow areas = most important regions for the decision")
print("   Blue areas       = less important regions")