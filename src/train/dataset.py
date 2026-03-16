# Import libraries
import tensorflow as tf
from ..config import BATCH_SIZE

# Variables
AUTOTUNE = tf.data.AUTOTUNE

# Parse image
def parse_image(filename, steering):
    """
    Parse dataset to correctly import images.
    :param filename: Image file name
    :param steering: Steering value to be returned
    :return: image, steering
    """
    # Read image
    image = tf.io.read_file(filename)

    # Turn image readable to tensorflow
    image = tf.image.decode_jpeg(image, channels=3)

    # Change pixel's type
    image = tf.cast(image, tf.float32)

    return image, steering

# Build dataset
def build_dataset(image_paths, steering, training=True):
    """
    Build full training dataset
    :param image_paths: Path to images
    :param steering: Steering value
    :param training: If it is a training dataset
    :return: dataset
    """
    # Build dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, steering))

    # Shuffle dataset
    if training:
        dataset = dataset.shuffle(50000)

    # Parse dataset
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)

    # Create batches
    dataset = dataset.batch(BATCH_SIZE)

    # Prefetch
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset