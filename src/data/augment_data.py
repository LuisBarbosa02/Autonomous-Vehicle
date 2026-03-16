# Import libraries
import os
import cv2
import pandas as pd
import numpy as np

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
    for row in df.itertuples():
        # Do not augment already augmented images
        if row.image.startswith(("flip_", "bright_", "shift_", "left_", "right_")):
            continue
        
        # Get image
        img_path = os.path.join(image_dir, row.image)
            
        # Read image
        img = cv2.imread(img_path)

        # Safety check for image
        if img is None:
            continue

        # Flip augmentation
        flip_name = "flip_" + row.image
        flip_path = os.path.join(image_dir, flip_name)
        if not os.path.exists(flip_path):
            flipped = cv2.flip(img, 1)
            cv2.imwrite(flip_path, flipped)

            augmented_rows.append([
                flip_name,
                -row.steering,
                row.timestamp,
                row.condition
            ])

        # Brightness augmentation
        bright_name = "bright_" + row.image
        bright_path = os.path.join(image_dir, bright_name)
        if not os.path.exists(bright_path):
            brightness = np.random.uniform(0.6, 1.4)
            bright = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            cv2.imwrite(bright_path, bright)

            augmented_rows.append([
                bright_name,
                row.steering,
                row.timestamp,
                row.condition
            ])

        # Small Recovery steering augmentation
        shift_name = "shift_" + row.image
        shift_path = os.path.join(image_dir, shift_name)
        if not os.path.exists(shift_path):
            shift = np.random.randint(-25, 26)

            M = np.float32([[1, 0, shift], [0, 1, 0]])
            translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
            steer_adjust = shift * 0.002
            
            cv2.imwrite(shift_path, translated)

            augmented_rows.append([
                shift_name,
                np.clip(row.steering + steer_adjust, -1, 1),
                row.timestamp,
                row.condition
            ])

        # Large recovery left steering augmentation
        left_name = "left_" + row.image
        left_path = os.path.join(image_dir, left_name)
        if not os.path.exists(left_path):
            left_shift = 60

            M_left = np.float32([[1, 0, left_shift], [0, 1, 0]])
    
            left_img = cv2.warpAffine(img, M_left, (img.shape[1], img.shape[0]))
            
            cv2.imwrite(left_path, left_img)

            augmented_rows.append([
                left_name,
                np.clip(row.steering + 0.20, -1, 1),
                row.timestamp,
                row.condition
            ])

        # Large recovery right steering augmentation
        right_name = "right_" + row.image
        right_path = os.path.join(image_dir, right_name)
        if not os.path.exists(right_path):
            right_shift = -60

            M_right = np.float32([[1, 0, right_shift], [0, 1, 0]])
    
            right_img = cv2.warpAffine(img, M_right, (img.shape[1], img.shape[0]))
            
            cv2.imwrite(right_path, right_img)

            augmented_rows.append([
                right_name,
                np.clip(row.steering - 0.20, -1, 1),
                row.timestamp,
                row.condition
            ])

    # Save conditions into dataframe
    augmented_df = pd.DataFrame(
        augmented_rows,
        columns=["image", "steering", "timestamp", "condition"]
    )

    return pd.concat([df, augmented_df]).reset_index(drop=True)