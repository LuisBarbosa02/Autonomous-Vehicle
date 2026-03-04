# Import libraries
import os
import cv2
import pandas as pd

# Augment data
def augment_data(df, image_dir):
    """
    Augment dataset for better generalization.
    :param df: Dataframe containing data
    :param image_dir: Folder where images are saved
    :return: augmented_df
    """
    # Keep augmented rows
    augmented_rows = []

    # Augmenting data
    for _, row in df.iterrows():
        # Augment turns
        if abs(row.steering) > 0.2:
            # Get image
            img_path = os.path.join(image_dir, row.image)
            img = cv2.imread(img_path)

            # Flip image
            flipped_img = cv2.flip(img, 1)

            # Save new image
            new_name = "flip_" + row.image
            cv2.imwrite(os.path.join(image_dir, new_name), flipped_img)

            augmented_rows.append([
                new_name,
                -row.steering,
                row.timestamp,
                row.condition
            ])

    # Save conditions
    augmented_df = pd.DataFrame(
        augmented_rows,
        columns=["image", "steering", "timestamp", "condition"]
    )

    return pd.concat([df, augmented_df]).reset_index(drop=True)