# Import libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mlflow

from .loader import load_labels
from .dataset import build_dataset
from ..config import TRAIN_DATASET_PATH, TRAIN_IMAGES_PATH, TEST_DATASET_PATH, TEST_IMAGES_PATH, VALIDATION_SPLIT, EPOCHS, LEARNING_RATE
from ..model.model import build_cnn_model
from .experiment import mlflow_experiment

# Train model
def train_model():
    # Get image name and steering
    image_paths, steering = load_labels(TRAIN_DATASET_PATH, TRAIN_IMAGES_PATH)

    # Separate training and validation datasets
    train_images, val_images, train_steer, val_steer = train_test_split(
        image_paths, steering, test_size=VALIDATION_SPLIT, random_state=42
    )

    # Build datasets
    train_dataset = build_dataset(train_images, train_steer, training=True)
    val_dataset = build_dataset(val_images, val_steer, training=False)

    # Load test data
    test_images, test_steer = load_labels(TEST_DATASET_PATH, TEST_IMAGES_PATH)
    test_dataset = build_dataset(test_images, test_steer, training=False)

    # Build model
    model = build_cnn_model()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    # Parameters
    params = {
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    }

    # Start MLflow run
    mlflow_experiment(params)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    # Fit model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(test_dataset)

    # Log test metrics
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_mae": test_mae
    })

    # Save model
    model.save("models/final_model.keras")

    # End MLflow run
    mlflow.end_run()

# Run
if __name__ == '__main__':
    train_model()