# Import libraries
from tensorflow.keras import layers, Sequential, Input

# Model
def build_cnn_model():
    """
    Build a CNN model.
    :return: model
    """
    # Run layers one after the other
    model = Sequential([
        # Input images of any size
        Input(shape=(None, None, 3)),

        # Preprocessing
        layers.Resizing(66, 200),
        layers.Rescaling(1.0 / 255.0),

        # Convolution
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Flattening
        layers.Flatten(),

        # Fully connected
        layers.Dense(1164, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),

        # Steering angle
        layers.Dense(1)
    ])

    return model

# Run
if __name__ == '__main__':
    model = build_cnn_model()
    print(model.summary())