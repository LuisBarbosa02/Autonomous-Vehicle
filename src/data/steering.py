# Classify steering
def classify_steering(steer: float):
    """
    Classify the category of the steer.
    :param steer: steer value
    :return: str
    """
    # Steer absolute value
    steer = abs(steer)

    # Classify steer
    if steer < 0.05:
        return "near_zero"
    elif steer >= 0.05 and steer < 0.25:
        return "mild"
    else: # steer >= 0.25
        return "strong"