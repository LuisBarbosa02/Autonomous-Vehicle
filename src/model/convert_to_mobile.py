# Import libraries
import tensorflow as tf

# Converter
converter = tf.lite.TFLiteConverter.from_saved_model("src/model/saved_model")

# TFLite model
tflite_model = converter.convert()

# Save model
with open("models/steering_model.tflite", "wb") as f:
    f.write(tflite_model)