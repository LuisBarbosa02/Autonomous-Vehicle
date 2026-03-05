# Import library
from ..config import WEATHER_CONFIG, FRAMES_PER_MODE
from .carla_client import connect_to_carla
from .collector import data_collector
from .augment_data import augment_data

import argparse
import os
import pandas as pd

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Choose dataset type: train or test"
    )
    return parser.parse_args()

# Main
def main():
    """
    Run data collector.
    """
    # Parse arguments
    args = parse_arguments()
    MODE = args.mode

    # Number of samples per weather condition
    FRAMES_PER_CONDITION = FRAMES_PER_MODE[MODE]

    # Paths
    DATASET_DIR = os.path.join('data', MODE)
    IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
    LABEL_FILE = os.path.join(DATASET_DIR, 'labels.csv')

    # Make image directory if non-existent
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Load data
    if os.path.exists(LABEL_FILE):
        dataframe = pd.read_csv(LABEL_FILE)
        start_index = len(dataframe)
    else:
        dataframe = None
        start_index = 0

    all_rows = []
    current_index = start_index

    # Load world
    world, traffic_manager, original_settings = connect_to_carla()

    # Run data collector
    try:
        for condition in WEATHER_CONFIG:
            rows, current_index = data_collector(
                world,
                traffic_manager,
                condition,
                IMAGE_DIR,
                current_index,
                FRAMES_PER_CONDITION
            )
            all_rows.extend(rows)

    finally:
        world.apply_settings(original_settings)
        print("\nCleaning up...")

    # Load all data into dataframe
    df = pd.DataFrame(
        all_rows,
        columns=["image", "steering", "timestamp", "condition"]
    )

    # Apply changes to training set
    if MODE == "train":    
        # Augmenting the data
        df = augment_data(df, IMAGE_DIR)

    # Combining data if previously existent
    if dataframe is not None:
        combined_df = pd.concat([dataframe, df], ignore_index=True)
    else:
        combined_df = df
    
    # Save the data
    combined_df.to_csv(LABEL_FILE, index=False)
    
    print("\nDataset collection complete.")
    print(f"Final dataset size: {len(combined_df)}")

# Run
if __name__ == '__main__':
    main()