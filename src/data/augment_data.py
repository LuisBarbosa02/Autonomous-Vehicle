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
        # Do not augment already augmented images
        if row.image.startswith("flip_"):
            continue
        
        # Augment turns
        if abs(row.steering) > 0.2:
            # Get image
            new_name = "flip_" + row.image
            img_path = os.path.join(image_dir, row.image)

            # Do not duplicate augmentation
            if os.path.exists(os.path.join(image_dir, new_name)):
                continue
            
            # Read image
            img = cv2.imread(img_path)

            # Safety check for image
            if img is None:
                continue

            # Flip image
            flipped_img = cv2.flip(img, 1)

            # Save new image
            cv2.imwrite(os.path.join(image_dir, new_name), flipped_img)

            # # Save conditions into list
            augmented_rows.append([
                new_name,
                -row.steering,
                row.timestamp,
                row.condition
            ])

    # Save conditions into dataframe
    augmented_df = pd.DataFrame(
        augmented_rows,
        columns=["image", "steering", "timestamp", "condition"]
    )

    return pd.concat([df, augmented_df]).reset_index(drop=True)