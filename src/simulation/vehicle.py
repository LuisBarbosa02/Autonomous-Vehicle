# Import libraries
import numpy as np
from ..config import SPEED, DT

# Vehicle state
class VehicleState:
    """
    Control the vehicle's state
    """
    def __init__(self):
        self.x = 0.0

    def update(self, steering):
        """
        Update the vehicle's state
        """
        self.x += SPEED * np.sin(steering) * DT