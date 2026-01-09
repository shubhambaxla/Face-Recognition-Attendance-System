# Basic example: convert Keras embedding model and/or a simple Keras classifier to TFLite.
# Note: SVM classifier cannot be converted directly. For microcontrollers, prefer a small Keras classifier.
import tensorflow as tf, os
from tensorflow.keras.models import load_model

# Convert embedding model (this produces a tflite model for the feature extractor)
if os.path.exists('models/embedding_model.h5'):
    model = load_model('models/embedding_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # For smaller size, enable post-training quantization (dynamic)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open('models/embedding_model.tflite','wb').write(tflite_model)
    print('Saved models/embedding_model.tflite')
else:
    print('models/embedding_model.h5 not found. Run train_embeddings.py first.')

# If you trained a small Keras classifier, you can convert similarly.
