import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shutil
import os

# ── GPU setup ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ Using GPU: {gpus[0].name}")

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TEST_DIR    = os.path.join(BASE_DIR, "dataset", "Testing")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# Final production model — prediction.py and the FastAPI backend
# will always load THIS file, so we just overwrite it with the winner
PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

# ── All candidate models ──────────────────────────────────
# Add or remove entries here as you train more models
CANDIDATES = [
    {
        'name'       : 'Basic CNN',
        'path'       : os.path.join(MODEL_DIR, 'cnn_model.h5'),
        'preprocess' : 'rescale',   # uses rescale=1./255
    },
    {
        'name'       : 'MobileNetV2 Finetuned',
        'path'       : os.path.join(MODEL_DIR, 'best_model_finetuned.keras'),
        'preprocess' : 'rescale',   # uses rescale=1./255
    },
    {
        'name'       : 'EfficientNetB0',
        'path'       : os.path.join(MODEL_DIR, 'efficientnet_phase1.keras'),
        'preprocess' : 'efficientnet',  # uses preprocess_input, no rescale
    },
]

# ── Helper: build test generator ──────────────────────────
def get_test_generator(preprocess_type):
    if preprocess_type == 'efficientnet':
        datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

# ── Evaluate all candidates ───────────────────────────────
print(f"\n{'='*55}")
print(f"  🔍 BRAINDETECT — MODEL SELECTOR")
print(f"{'='*55}\n")

results = []

for candidate in CANDIDATES:
    name = candidate['name']
    path = candidate['path']

    if not os.path.exists(path):
        print(f"⚠️  Skipping {name} — file not found: {path}")
        continue

    print(f"📦 Loading: {name}")
    try:
        model = load_model(path)
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}\n")
        continue

    test_gen = get_test_generator(candidate['preprocess'])
    test_gen.reset()

    loss, acc = model.evaluate(test_gen, verbose=1)

    # Per-class predictions for classification report
    test_gen.reset()
    preds      = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    true_labels = test_gen.classes

    print(f"   ✅ {name} Test Accuracy : {acc*100:.2f}%  |  Loss: {loss:.4f}\n")

    results.append({
        'name'        : name,
        'path'        : path,
        'preprocess'  : candidate['preprocess'],
        'accuracy'    : acc,
        'loss'        : loss,
        'preds'       : preds,
        'true_labels' : true_labels,
        'model'       : model,
    })

if not results:
    print("❌ No models could be evaluated. Check your model paths.")
    exit(1)

# ── Pick the winner ───────────────────────────────────────
results.sort(key=lambda x: x['accuracy'], reverse=True)
winner = results[0]

print(f"{'='*55}")
print(f"  📈 FINAL RESULTS")
print(f"{'='*55}")
for i, r in enumerate(results):
    medal = ['🥇', '🥈', '🥉'][i] if i < 3 else '  '
    print(f"  {medal} {r['name']:25} : {r['accuracy']*100:.2f}%  |  Loss: {r['loss']:.4f}")
print(f"{'='*55}")
print(f"\n🏆 Winner : {winner['name']} ({winner['accuracy']*100:.2f}%)")

# ── Save winner as production model ───────────────────────
print(f"\n💾 Saving winner as production model...")
print(f"   From : {winner['path']}")
print(f"   To   : {PRODUCTION_MODEL_PATH}")

# Save using keras native save so it always produces a clean .keras file
# regardless of whether winner was .h5 or .keras
winner['model'].save(PRODUCTION_MODEL_PATH)
print(f"✅ Production model saved → best_model.keras")
print(f"   prediction.py and FastAPI backend will automatically use this model.\n")

# Also write a metadata file so the backend knows which model won and
# which preprocessing it needs
meta_path = os.path.join(MODEL_DIR, "production_model_info.txt")
with open(meta_path, "w") as f:
    f.write(f"winner={winner['name']}\n")
    f.write(f"accuracy={winner['accuracy']:.6f}\n")
    f.write(f"loss={winner['loss']:.6f}\n")
    f.write(f"preprocess={winner['preprocess']}\n")
    f.write(f"model_path={PRODUCTION_MODEL_PATH}\n")
print(f"📄 Metadata saved → {meta_path}")

# ── Classification reports ────────────────────────────────
print(f"\n{'='*55}")
print(f"  📋 CLASSIFICATION REPORTS")
print(f"{'='*55}")
for r in results:
    print(f"\n{r['name']}:")
    print(classification_report(r['true_labels'], r['preds'], target_names=CLASS_NAMES))

# ── Confusion matrices ────────────────────────────────────
n       = len(results)
fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
if n == 1:
    axes = [axes]

for ax, r in zip(axes, results):
    cm = confusion_matrix(r['true_labels'], r['preds'])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    crown = ' 🏆' if r['name'] == winner['name'] else ''
    ax.set_title(f"{r['name']}{crown}\n{r['accuracy']*100:.2f}%",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('BrainDetect — Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "selector_confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"\n📸 Confusion matrices saved → {cm_path}")

# ── Bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
names   = [r['name'] for r in results]
accs    = [r['accuracy'] * 100 for r in results]
colors  = ['#2ecc71' if r['name'] == winner['name'] else '#4C72B0' for r in results]
bars    = ax.bar(names, accs, color=colors, width=0.4, edgecolor='white')

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{acc:.2f}%',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold'
    )

ax.set_ylim(0, 105)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('BrainDetect — Model Selection Results\n(green = selected for production)',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
chart_path = os.path.join(BASE_DIR, "selector_model_comparison.png")
plt.savefig(chart_path, dpi=150)
plt.show()
print(f"📸 Comparison chart saved → {chart_path}")

print(f"\n{'='*55}")
print(f"  ✅ MODEL SELECTION COMPLETE")
print(f"{'='*55}")
print(f"  Production model : {winner['name']}")
print(f"  Accuracy         : {winner['accuracy']*100:.2f}%")
print(f"  Saved to         : best_model.keras")
print(f"  Preprocessing    : {winner['preprocess']}")
print(f"{'='*55}")
print(f"\n🚀 Next step: FastAPI backend will load best_model.keras at startup")