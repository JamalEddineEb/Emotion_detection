import tensorflow as tf
import tf2onnx

# Load the Keras model from the .h5 file
model = tf.keras.models.load_model("model.h5")

# Define the ONNX file path
onnx_model_path = "model.onnx"

# Specify input signature for tf2onnx conversion
spec = (tf.TensorSpec(shape=model.input_shape, dtype=tf.float32, name="input"),)

# Convert the model to ONNX format
try:
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save the ONNX model to a file
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model successfully converted and saved to {onnx_model_path}")

except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Make sure you are using a compatible version of TensorFlow and tf2onnx.")
