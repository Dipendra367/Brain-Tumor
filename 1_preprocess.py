import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Auto-detect project root ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR   = os.path.join(BASE_DIR, "dataset", "Testing")

print(f"📁 Project root : {BASE_DIR}")
print(f"📁 Training dir : {TRAIN_DIR}")
print(f"📁 Testing dir  : {TEST_DIR}")

# ── Sanity check ─────────────────────────────────────────
if not os.path.exists(TRAIN_DIR):
    print(f"\n❌ ERROR: Cannot find {TRAIN_DIR}")
    print("   Make sure dataset/Training/ exists in your project root.")
    exit()
else:
    print("✅ Dataset folders found!\n")

# ── Config ───────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# ── Augmentation for training ────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

# ── No augmentation for test ─────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)

# ── Load training data ────────────────────────────────────
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# ── Load validation data ──────────────────────────────────
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ── Load test data ────────────────────────────────────────
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ── Print summary ─────────────────────────────────────────
print("\n✅ Data loaded successfully!")
print(f"   Classes      : {train_generator.class_indices}")
print(f"   Train images : {train_generator.samples}")
print(f"   Val images   : {val_generator.samples}")
print(f"   Test images  : {test_generator.samples}")

# ── Visualize sample images ───────────────────────────────
images, labels = next(train_generator)
class_names    = list(train_generator.class_indices.keys())

plt.figure(figsize=(12, 6))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i])
    plt.title(class_names[np.argmax(labels[i])], fontsize=9)
    plt.axis('off')
plt.suptitle("Sample Training Images (augmented)", fontsize=13)
plt.tight_layout()

save_path = os.path.join(BASE_DIR, "sample_images.png")
plt.savefig(save_path)
plt.show()
print(f"📸 Sample images saved to: {save_path}")
