import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── GPU setup ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TEST_DIR    = os.path.join(BASE_DIR, "dataset", "Testing")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# ── Test data ─────────────────────────────────────────────
test_datagen   = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
true_labels = test_generator.classes

# ── Load both models ──────────────────────────────────────
print("\n📦 Loading models...")
cnn_model      = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))
transfer_model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))
print("✅ Both models loaded!")

# ── Predictions ───────────────────────────────────────────
print("\n🔍 Running predictions...")
test_generator.reset()
cnn_preds   = cnn_model.predict(test_generator, verbose=1)
cnn_labels  = np.argmax(cnn_preds, axis=1)

test_generator.reset()
tf_preds    = transfer_model.predict(test_generator, verbose=1)
tf_labels   = np.argmax(tf_preds, axis=1)

# ── Accuracy ──────────────────────────────────────────────
cnn_acc = np.mean(cnn_labels == true_labels) * 100
tf_acc  = np.mean(tf_labels  == true_labels) * 100

print(f"\n{'='*45}")
print(f"  📊 MODEL COMPARISON")
print(f"{'='*45}")
print(f"  Basic CNN Accuracy       : {cnn_acc:.2f}%")
print(f"  MobileNetV2 Accuracy     : {tf_acc:.2f}%")
print(f"  Winner                   : {'MobileNetV2 ✅' if tf_acc > cnn_acc else 'Basic CNN ✅'}")
print(f"{'='*45}")

# ── Classification reports ────────────────────────────────
print("\n📋 CNN Classification Report:")
print(classification_report(true_labels, cnn_labels, target_names=CLASS_NAMES))

print("\n📋 MobileNetV2 Classification Report:")
print(classification_report(true_labels, tf_labels, target_names=CLASS_NAMES))

# ── Confusion matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, preds, title in zip(
    axes,
    [cnn_labels, tf_labels],
    [f'Basic CNN ({cnn_acc:.1f}%)', f'MobileNetV2 ({tf_acc:.1f}%)']
):
    cm = confusion_matrix(true_labels, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(BASE_DIR, "confusion_matrices.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\n📸 Confusion matrices saved to: {save_path}")

# ── Bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
models  = ['Basic CNN', 'MobileNetV2\nTransfer Learning']
accs    = [cnn_acc, tf_acc]
colors  = ['#4C72B0', '#55A868']
bars    = ax.bar(models, accs, color=colors, width=0.4, edgecolor='white')

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f'{acc:.2f}%',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold'
    )

ax.set_ylim(0, 105)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path2 = os.path.join(BASE_DIR, "model_comparison.png")
plt.savefig(save_path2, dpi=150)
plt.show()
print(f"📸 Comparison chart saved to: {save_path2}")

print(f"\n🏆 Best model for deployment → best_model.keras")
print(f"   We will use this for Grad-CAM and the web app!")