# Import libraries
import pandas as pd
import os

# Load labels
def load_labels(csv_path, image_folder):
    """
    Load labels and get image paths
    :param csv_path: Path to dataset
    :return: image_paths, steering
    """
    # Load dataframe
    df = pd.read_csv(csv_path)

    # Get data
    image_paths = df.image.values
    image_paths = [os.path.join(image_folder, p) for p in image_paths]
    steering = df.steering.values.astype('float32')

    return image_paths, steering