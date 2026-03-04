# Import libraries
import carla
from ..config import HOST, PORT, FIXED_DELTA_SECONDS, TOWN

# Connect to Carla
def connect_to_carla():
    """
    Connect and configure the Carla environment.
    :return: world, original_settings
    """
    # Load Carla environment
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.load_world(TOWN)

    # Configure settings
    original_settings = world.get_settings()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    # Adjusting traffic
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    
    return world, traffic_manager, original_settings