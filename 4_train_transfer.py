import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
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
else:
    print("⚠️  No GPU found, training on CPU")

# ── Config ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR   = os.path.join(BASE_DIR, "dataset", "Testing")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
PHASE1_EPOCHS    = 15
PHASE2_EPOCHS    = 20
UNFREEZE_LAYERS  = 30
PHASE1_LR        = 1e-3
PHASE2_LR        = 1e-5

PHASE1_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_phase1.keras")
PHASE2_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_finetuned.keras")

# ── Data generators ───────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

# ── Build model ───────────────────────────────────────────
print("\n🏗️  Building EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False   # frozen for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs, name='BrainDetect_EfficientNetB0')

total_params = sum(np.prod(v.shape) for v in model.weights)
print(f"   Total parameters : {total_params:,}")

# ── Phase 1 — Train head only (base frozen) ───────────────
print(f"\n🚀 Phase 1 — Training head for up to {PHASE1_EPOCHS} epochs (base frozen)...")

model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase1_callbacks = [
    ModelCheckpoint(
        PHASE1_MODEL_PATH,
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

history1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=val_generator,
    callbacks=phase1_callbacks
)

print("\n📊 Phase 1 test accuracy:")
test_generator.reset()
phase1_loss, phase1_acc = model.evaluate(test_generator, verbose=1)
print(f"   Phase 1 Test Accuracy : {phase1_acc*100:.2f}%")
print(f"   Phase 1 Test Loss     : {phase1_loss:.4f}")

# ── Phase 2 — Unfreeze top N layers, fine-tune ────────────
total_layers = len(base_model.layers)
print(f"\n🔓 Phase 2 — Unfreezing top {UNFREEZE_LAYERS} of {total_layers} EfficientNetB0 layers...")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"   Trainable parameters : {trainable:,} / {total_params:,}")

model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"✅ Model recompiled — learning rate: {PHASE2_LR}")

phase2_callbacks = [
    ModelCheckpoint(
        PHASE2_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

print(f"\n🚀 Fine-tuning top {UNFREEZE_LAYERS} layers for up to {PHASE2_EPOCHS} epochs...\n")
history2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    validation_data=val_generator,
    callbacks=phase2_callbacks
)

# ── Final evaluation ──────────────────────────────────────
print("\n📊 Evaluating fine-tuned EfficientNetB0 on test set...")
test_generator.reset()
fine_loss, fine_acc = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*55}")
print(f"  📈 RESULTS COMPARISON")
print(f"{'='*55}")
print(f"  Phase 1 (frozen) Test Accuracy    : {phase1_acc*100:.2f}%")
print(f"  Phase 2 (fine-tuned) Accuracy     : {fine_acc*100:.2f}%")
improvement = (fine_acc - phase1_acc) * 100
if improvement > 0:
    print(f"  Improvement                       : +{improvement:.2f}% ✅")
else:
    print(f"  Change                            : {improvement:.2f}% ⚠️  (use phase 1 model)")
print(f"{'='*55}")

best_acc  = max(phase1_acc, fine_acc)
best_path = PHASE2_MODEL_PATH if fine_acc >= phase1_acc else PHASE1_MODEL_PATH
print(f"\n💾 Best EfficientNetB0 model saved to : {best_path}")
print(f"   Best test accuracy                 : {best_acc*100:.2f}%")

# ── Save results for model selector ──────────────────────
results_path = os.path.join(MODEL_DIR, "efficientnet_results.txt")
with open(results_path, "w") as f:
    f.write(f"model=efficientnet_b0\n")
    f.write(f"test_accuracy={best_acc:.6f}\n")
    f.write(f"test_loss={fine_loss:.6f}\n")
    f.write(f"model_path={best_path}\n")
print(f"📄 Results saved to: {results_path}")

# ── Plot ───────────────────────────────────────────────────
epochs_p1 = len(history1.history['accuracy'])
epochs_p2 = len(history2.history['accuracy'])
total_x   = list(range(1, epochs_p1 + epochs_p2 + 1))

acc      = history1.history['accuracy']     + history2.history['accuracy']
val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss     = history1.history['loss']         + history2.history['loss']
val_loss = history1.history['val_loss']     + history2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EfficientNetB0 — Training History", fontsize=14, fontweight='bold')

ax1.plot(total_x, acc,     label='Train accuracy')
ax1.plot(total_x, val_acc, label='Val accuracy')
ax1.axvline(x=epochs_p1 + 0.5, color='gray', linestyle='--', label='Fine-tune start')
ax1.axhline(y=phase1_acc, color='orange', linestyle=':', label=f'Phase 1 baseline ({phase1_acc*100:.2f}%)')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(total_x, loss,     label='Train loss')
ax2.plot(total_x, val_loss, label='Val loss')
ax2.axvline(x=epochs_p1 + 0.5, color='gray', linestyle='--', label='Fine-tune start')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, "efficientnet_curves.png")
plt.savefig(save_path)
plt.show()
print(f"📸 Training curves saved to: {save_path}")