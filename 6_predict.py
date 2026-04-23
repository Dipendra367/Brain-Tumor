import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = (224, 224)

CLASS_INFO = {
    'Glioma'    : '🔴 Malignant — Requires immediate medical attention',
    'Meningioma': '🟠 Usually benign — Needs monitoring',
    'No Tumor'  : '🟢 No tumor detected — Brain appears normal',
    'Pituitary' : '🟣 Pituitary tumor — Often treatable'
}

# ── Load model ────────────────────────────────────────────
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!\n")

# ── Prediction function ───────────────────────────────────
def predict(img_path):
    """
    Input  : path to any brain MRI image
    Output : class name + confidence + all scores
    """
    # Load and preprocess
    img       = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_index  = np.argmax(predictions[0])
    pred_class  = CLASS_NAMES[pred_index]
    confidence  = predictions[0][pred_index] * 100

    # All class scores
    all_scores = {
        CLASS_NAMES[i]: f"{predictions[0][i]*100:.2f}%"
        for i in range(len(CLASS_NAMES))
    }

    return {
        'class'     : pred_class,
        'confidence': confidence,
        'all_scores': all_scores,
        'info'      : CLASS_INFO[pred_class]
    }

# ── Test on sample images ─────────────────────────────────
if __name__ == "__main__":
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "Testing")

    print("🔍 Running predictions on sample images...\n")
    print(f"{'='*55}")

    for class_name in CLASS_NAMES:
        # Find folder
        folder = os.path.join(TEST_DIR, class_name.lower().replace(" ", ""))
        if not os.path.exists(folder):
            folder = os.path.join(TEST_DIR, "notumor")

        # Get first image
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            continue

        img_path = os.path.join(folder, imgs[2])  # pick 3rd image
        result   = predict(img_path)

        print(f"📂 Actual class    : {class_name}")
        print(f"🧠 Predicted       : {result['class']}")
        print(f"📊 Confidence      : {result['confidence']:.2f}%")
        print(f"ℹ️  Info            : {result['info']}")
        print(f"📈 All scores      :")
        for cls, score in result['all_scores'].items():
            marker = " ◀ predicted" if cls == result['class'] else ""
            print(f"     {cls:12} : {score}{marker}")
        correct = '✅ CORRECT' if result['class'] == class_name else '❌ WRONG'
        print(f"🎯 Result          : {correct}")
        print(f"{'-'*55}")

    print(f"\n✅ Prediction function is ready!")
    print(f"   Use predict('path/to/image.jpg') in your app")