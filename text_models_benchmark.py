import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_ID = "unitary/toxic-bert"
TFLITE_MODEL_PATH = 'toxic_bert_float.tflite'

def load_test_data(tokenizer, num_samples=50):
    """
    IMPORTANT: Replace the dummy_texts list with your actual test sentences.
    """
    print(f"Tokenizing {num_samples} dummy sentences...")
    dummy_texts = [
        "This is a perfectly fine and friendly comment.",
        "You are a wonderful person and I appreciate you.",
        "I will find you and hurt you.", # Toxic example
        "Go away you horrible creature." # Toxic example
    ] * (num_samples // 4)
    
    # Preprocess text data
    inputs = tokenizer(dummy_texts, padding=True, truncation=True, return_tensors="np")
    # For this model, 0=non-toxic, 1=toxic. We'll create dummy labels.
    dummy_labels = ([0, 0, 1, 1] * (num_samples // 4))
    return inputs, np.array(dummy_labels)

# --- Load Models and Tokenizer ---
print("Loading tokenizer and original Hugging Face model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
tflite_input_details = interpreter.get_input_details()
tflite_output_details = interpreter.get_output_details()

# --- Run Inference ---
inputs, y_test_labels = load_test_data(tokenizer)

# Original Model Predictions
print("\nRunning inference with original Hugging Face model...")
original_outputs = original_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
original_predictions = np.argmax(original_outputs.logits, axis=1)

# TFLite Model Predictions
print("Running inference with TFLite model...")
# Note: The order of inputs for the TFLite model must match how it was converted.
# ONNX conversion typically places input_ids first, then attention_mask.
input_ids_index = tflite_input_details[0]['index']
attention_mask_index = tflite_input_details[1]['index']

tflite_predictions = []
for i in range(len(y_test_labels)):
    interpreter.set_tensor(input_ids_index, np.expand_dims(inputs['input_ids'][i], axis=0))
    interpreter.set_tensor(attention_mask_index, np.expand_dims(inputs['attention_mask'][i], axis=0))
    interpreter.invoke()
    output_data = interpreter.get_tensor(tflite_output_details[0]['index'])
    tflite_predictions.append(np.argmax(output_data[0]))

# --- Compare Performance ---
print("\n" + "="*50)
print("        Text Model Performance Comparison")
print("="*50)

print("\n--- Original Hugging Face Model ---")
print(f"Accuracy: {accuracy_score(y_test_labels, original_predictions):.4f}")
print(classification_report(y_test_labels, original_predictions, target_names=['Non-Toxic', 'Toxic']))

print("\n--- Converted TFLite Model ---")
print(f"Accuracy: {accuracy_score(y_test_labels, tflite_predictions):.4f}")
print(classification_report(y_test_labels, tflite_predictions, target_names=['Non-Toxic', 'Toxic']))
print("="*50)