import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# ── GPU setup ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ Using GPU: {gpus[0].name}")


# ── Config ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR   = os.path.join(BASE_DIR, "dataset", "Testing")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
FINETUNE_EPOCHS    = 15
UNFREEZE_LAYERS    = 30   # how many layers from the top of MobileNetV2 to unfreeze

FROZEN_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.keras")
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_finetuned.keras")

# ── Data generators (same as original, no leakage) ────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ── Load frozen model ─────────────────────────────────────
print(f"\n📂 Loading frozen model from: {FROZEN_MODEL_PATH}")
model = load_model(FROZEN_MODEL_PATH)
print("✅ Model loaded successfully")

# Baseline test accuracy before fine-tuning
print("\n📊 Baseline test accuracy (frozen model):")
base_loss, base_acc = model.evaluate(test_generator, verbose=1)
print(f"   Frozen Model Test Accuracy : {base_acc*100:.2f}%")
print(f"   Frozen Model Test Loss     : {base_loss:.4f}")

# ── Unfreeze top N layers of MobileNetV2 base ─────────────
# The MobileNetV2 base is model.layers[0] since it's a functional model
# We need to find the base model inside
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    # Fallback: treat all non-Dense layers as base
    print("⚠️  Could not find MobileNetV2 layer by name — unfreezing last 30 layers of full model")
    for layer in model.layers[-30:]:
        layer.trainable = True
else:
    total_layers = len(base_model.layers)
    print(f"\n🔓 Unfreezing top {UNFREEZE_LAYERS} of {total_layers} MobileNetV2 layers...")

    # Freeze all first, then selectively unfreeze
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

# Count trainable params
trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
total     = sum(np.prod(v.shape) for v in model.weights)
print(f"   Trainable parameters : {trainable:,} / {total:,}")

# ── Recompile with low LR ──────────────────────────────────
# Critical: use 10-100x smaller LR than original to avoid destroying pretrained weights
FINETUNE_LR = 1e-5
model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"\n✅ Model recompiled — learning rate: {FINETUNE_LR}")

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ── Fine-tune ──────────────────────────────────────────────
print(f"\n🚀 Fine-tuning top {UNFREEZE_LAYERS} layers for up to {FINETUNE_EPOCHS} epochs...\n")
history_fine = model.fit(
    train_generator,
    epochs=FINETUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# ── Evaluate fine-tuned model ──────────────────────────────
print("\n📊 Evaluating fine-tuned model on test set...")
fine_loss, fine_acc = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*55}")
print(f"  📈 RESULTS COMPARISON")
print(f"{'='*55}")
print(f"  Frozen MobileNetV2 Test Accuracy  : {base_acc*100:.2f}%")
print(f"  Fine-tuned MobileNetV2 Accuracy   : {fine_acc*100:.2f}%")
improvement = (fine_acc - base_acc) * 100
if improvement > 0:
    print(f"  Improvement                       : +{improvement:.2f}% ✅")
else:
    print(f"  Change                            : {improvement:.2f}% ⚠️  (fine-tuning hurt — use frozen model)")
print(f"{'='*55}")

if fine_acc >= base_acc:
    print(f"\n💾 Fine-tuned model saved to: {FINETUNED_MODEL_PATH}")
    print("   ✅ Update your app.py to load 'best_model_finetuned.keras'")
else:
    print(f"\n⚠️  Fine-tuned model did NOT improve — keep using: {FROZEN_MODEL_PATH}")
    print("   Try reducing UNFREEZE_LAYERS to 15 or lowering FINETUNE_LR to 5e-6")

# ── Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history_fine.history['accuracy'],     label='Train accuracy')
ax1.plot(history_fine.history['val_accuracy'], label='Val accuracy')
ax1.axhline(y=base_acc, color='gray', linestyle='--', label=f'Frozen baseline ({base_acc*100:.2f}%)')
ax1.set_title('Fine-tuning — Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history_fine.history['loss'],     label='Train loss')
ax2.plot(history_fine.history['val_loss'], label='Val loss')
ax2.set_title('Fine-tuning — Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, "finetune_curves.png")
plt.savefig(save_path)
plt.show()
print(f"📸 Fine-tuning curves saved to: {save_path}")