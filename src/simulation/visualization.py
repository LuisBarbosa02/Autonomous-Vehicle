# Import libraries
import cv2
import numpy as np
from ..config import PIXELS_PER_METER

# Transform image
def transform_image(frame, lateral_offset):
    """
    Simulate lateral vehicle deviation
    """
    h, w = frame.shape[:2]

    shift_pixels = int(lateral_offset * PIXELS_PER_METER)
    shift_pixels = max(min(shift_pixels, w - 1), -(w - 1))
    M = np.float32([[1, 0, shift_pixels],
                    [0, 1, 0]])

    shifted = cv2.warpAffine(frame, M, (w, h))
    return shifted

# Draw overlay
def draw_overlay(frame, steering, lateral, interventions):
    """
    Draw simulation info on the frame
    """
    cv2.putText(frame, f"Steering: {steering:.3f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"Interventions: {interventions}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, "Press I for intervention | Q to quit", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame