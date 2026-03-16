# Import libraries
import numpy as np
import tensorflow as tf

from ..config import MODEL_PATH

class Perception:
    """
    Handles model loading and prediction
    """
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict_steering(self, frame):
        """
        Predict steering angle from the input frame
        """
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        return float(self.model(frame, training=False)[0][0])