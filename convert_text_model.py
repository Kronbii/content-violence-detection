# Part 1: Hugging Face to ONNX
from transformers.onnx import export, FeaturesManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# --- 1. Load Model and Tokenizer ---
model_id = "unitary/toxic-bert"
print(f"Loading model and tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# --- 2. Export to ONNX ---
feature = "sequence-classification"
onnx_path = "toxic-bert.onnx"
print(f"Exporting model to ONNX at {onnx_path}...")
onnx_config = FeaturesManager.get_config(model_id, feature)(model.config)
export(tokenizer=tokenizer, model=model, config=onnx_config, opset=13, output=onnx_path)
print("ONNX export complete.")

# --- 3. Convert ONNX to TensorFlow SavedModel ---
print("Converting ONNX model to TensorFlow SavedModel...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_model_path = "toxic-bert-tf"
tf_rep.export_graph(tf_model_path)
print(f"TensorFlow SavedModel created at {tf_model_path}")

# --- 4. Convert TensorFlow SavedModel to a Float32 TFLite model ---
print("Converting TensorFlow SavedModel to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model_float = converter.convert()

with open('toxic_bert_float.tflite', 'wb') as f:
    f.write(tflite_model_float)

print("Successfully saved Float32 TFLite model as toxic_bert_float.tflite")