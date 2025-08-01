import tensorflow as tf

# --- 1. Load the trained Keras model ---
print("Loading Keras model...")
# Make sure to replace this path with the actual path to your model file
keras_model = tf.keras.models.load_model('path/to/your/violence_detection_model.h5')
print("Model loaded successfully.")

# --- 2. Convert to a Float32 TFLite model ---
# This is the most direct conversion and preserves the original model's precision.
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model_float = converter.convert()

# --- 3. Save the TFLite model ---
with open('violence_model_float.tflite', 'wb') as f:
    f.write(tflite_model_float)

print("Successfully saved Float32 TFLite model as violence_model_float.tflite")