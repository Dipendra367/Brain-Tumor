import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TEST_DIR = 'dataset/Testing'

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(224,224),
    batch_size=32, class_mode='categorical', shuffle=False
)

models_to_check = {
    'MobileNet frozen'    : 'models/best_model.keras',
    'MobileNet finetuned' : 'models/best_model_finetuned.keras',
    'MobileNet 89%'       : 'models/best_model_89percent.keras',
    'CNN'                 : 'models/cnn_model.h5',
}

for name, path in models_to_check.items():
    print(f'\n🔍 Evaluating: {name}')
    try:
        model = tf.keras.models.load_model(path)
        test_gen.reset()
        loss, acc = model.evaluate(test_gen, verbose=0)
        print(f'   ✅ Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}')
    except Exception as e:
        print(f'   ❌ Failed to load: {e}')