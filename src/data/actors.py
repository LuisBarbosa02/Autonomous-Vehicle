# Import libraries
import random
from ..config import IMAGE_WIDTH, IMAGE_HEIGHT
import carla

# Spawn vehicle
def spawn_vehicle(world):
    """
    Spawn a vehicle inside the Carla environment.
    :param world: Carla world
    :return: vehicle
    """
    # Get world blueprint
    blueprint_library = world.get_blueprint_library()
    
    # Get spawn points and shuffle it
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Vehicle blueprint
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

    # Spawn vehicle
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            return vehicle

    raise RuntimeError("Could not spawn vehicle")

# Attach camera
def attach_camera(world, vehicle):
    """
    Attach camera to the vehicle inside Carla.
    :param world: Carla world
    :param vehicle: Vehicle in which the camera will be attached to
    :return: camera
    """
    # Get world blueprint
    blueprint_library = world.get_blueprint_library()
    
    # Get and configure camera blueprint
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))

    # Position camera
    transform = carla.Transform(
        carla.Location(x=1.5, z=1.4), # Slighly forward and above the hood
        carla.Rotation(pitch=-5.0) # Slightly tilted downward
    )

    # Create camera
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera

# Attach collision sensor
def attach_collision_sensor(world, vehicle):
    """
    Attach collision sensor to the vehicle inside Carla.
    :param world: Carla world
    :param vehicle: Vehicle in which the collision sensor will be attached to
    :return: collision_sensor
    """
    # Get world blueprint
    blueprint_library = world.get_blueprint_library()

    # Get collision sensor blueprint
    collision_bp = blueprint_library.find("sensor.other.collision")

    # Create collision sensor
    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=vehicle
    )

    return collision_sensor