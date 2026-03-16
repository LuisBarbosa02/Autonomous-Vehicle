# Import libraries
import tensorflow as tf
from ..config import MODEL_PATH

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Export model
model.export("src/model/saved_model")