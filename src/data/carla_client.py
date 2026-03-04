# Import libraries
import carla
from ..config import HOST, PORT, FIXED_DELTA_SECONDS

# Connect to Carla
def connect_to_carla():
    """
    Connect and configure the Carla environment.
    :return: world, original_settings
    """
    # Load Carla environment
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    # Configure settings
    original_settings = world.get_settings()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    
    return world, original_settings