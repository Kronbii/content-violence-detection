import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
ORIGINAL_MODEL_PATH = 'path/to/your/violence_detection_model.h5'
TFLITE_MODEL_PATH = 'violence_model_float.tflite'
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_test_data(num_samples=100):
    """
    IMPORTANT: Replace this function with your actual test data loading.
    The data should be preprocessed exactly as it was for training.
    """
    print(f"Generating {num_samples} dummy test images...")
    # Create float32 data in the [0, 1] range for this example
    X_test = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(num_samples,)) # Dummy labels
    return X_test, y_test

# --- Load Models ---
print("Loading original Keras model...")
original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Run Inference ---
X_test, y_test_labels = load_test_data()

# Original Model Predictions
print("\nRunning inference with original Keras model...")
original_pred_probs = original_model.predict(X_test)
original_predictions = np.argmax(original_pred_probs, axis=1)

# TFLite Model Predictions
print("Running inference with TFLite model...")
tflite_predictions = []
for image in X_test:
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    tflite_predictions.append(np.argmax(output_data[0]))

# --- Compare Performance ---
print("\n" + "="*50)
print("      Image Model Performance Comparison")
print("="*50)

print("\n--- Original Keras Model ---")
print(f"Accuracy: {accuracy_score(y_test_labels, original_predictions):.4f}")
print(classification_report(y_test_labels, original_predictions, target_names=['Non-Violence', 'Violence']))

print("\n--- Converted TFLite Model ---")
print(f"Accuracy: {accuracy_score(y_test_labels, tflite_predictions):.4f}")
print(classification_report(y_test_labels, tflite_predictions, target_names=['Non-Violence', 'Violence']))
print("="*50)