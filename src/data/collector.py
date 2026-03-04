# Import libraries
import carla
from queue import Queue, Empty
import numpy as np
import os
import cv2
from .actors import spawn_vehicle, attach_camera, attach_collision_sensor

# Data collector
def data_collector(world, traffic_manager, condition, image_dir, start_index, frames_per_condition):
    """
    Collect data deterministically in synchronous mode.
    :param world: Carla world
    :param traffic_manager: Manage traffic from Carla environment
    :param condition: Weather condition dictionary
    :param image_dir: Directory to save images
    :param start_index: Starting image index
    :param frames_per_condition: Number of frames to collect per weather condition
    :return: data_rows, current_index
    """
    print(f"\nCollecting condition: {condition['name']}")

    # Handle data
    collected_rows = []
    current_index = start_index

    # Collect data
    while len(collected_rows) < frames_per_condition:
        print(f"\nStarting attempt for {condition['name']}")

        # Apply weather
        weather = carla.WeatherParameters(
            sun_altitude_angle=condition['sun'],
            precipitation=condition['rain'],
            fog_density=0,
            cloudiness=50
        )
        world.set_weather(weather)

        # Spawn vehicle
        vehicle = spawn_vehicle(world)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        # Improve drive stability
        traffic_manager.vehicle_percentage_speed_difference(vehicle, 30.0) # Vehicle speed is x% less than max road speed
        traffic_manager.auto_lane_change(vehicle, False) # Disable lane changes
        traffic_manager.distance_to_leading_vehicle(vehicle, 5.0) # Set the distance from other vehicler to main vehicle
        
        # Stabilize autopilot
        for _ in range(5):
            world.tick()

        # Attach sensors to vehicle
        camera = attach_camera(world, vehicle)
        collision_sensor = attach_collision_sensor(world, vehicle)

        # Queue synchronous sensor readings
        image_queue = Queue()
        collision_queue = Queue()
        
        camera.listen(image_queue.put)
        collision_sensor.listen(collision_queue.put)
        world.tick() # Fully register sensors

        # Attempts
        attempt_rows = []
        collision_happened = False

        try:
            while len(attempt_rows) + len(collected_rows) < frames_per_condition:
                # Advance simulation by one step
                world.tick()

                # Check if image was taken
                try:
                    image = image_queue.get(timeout=2.0)
                except Empty:
                    print("Image timeout. Discarding attempt.")
                    collision_happened = True
                    break

                # Check collision
                if not collision_queue.empty():
                    print("Collision detected. Discarding attempt.")
                    collision_happened = True
                    break

                # Check if vehicle is still alive
                if vehicle is None or not vehicle.is_alive:
                    print("Vehicle destroyed unexpectedly.")
                    collision_happened = True
                    break

                # Convert raw image
                img = np.frombuffer(image.raw_data, dtype=np.uint8)
                img = img.reshape((image.height, image.width, 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Get steering value
                steering = vehicle.get_control().steer

                # Save image
                filename = f"img_{current_index:08d}.jpg"
                filepath = os.path.join(image_dir, filename)
            
                cv2.imwrite(filepath, img)

                # Save metadata
                attempt_rows.append([
                    filename,
                    steering,
                    image.timestamp,
                    condition["name"]
                ])

                # Increase current index
                current_index += 1

                # Show current frame processed
                total_now = len(attempt_rows) + len(collected_rows)
                print(f"{condition['name']} → {total_now}/{frames_per_condition}", end="\r")

        finally:
            # Stop sensors safely
            try:
                camera.stop()
            except:
                pass
            try:
                collision_sensor.stop()
            except:
                pass

            # Destroy actors safely
            try:
                camera.destroy()
            except:
                pass
            try:
                collision_sensor.destroy()
            except:
                pass
            try:
                vehicle.destroy()
            except:
                pass

            # Fully destroying
            world.tick()

        # Saving
        if not collision_happened:
            collected_rows.extend(attempt_rows)
            print(f"\nClean attempt kept ({len(attempt_rows)} frames).")
        else:
            # Remove partial saved images from disk
            for row in attempt_rows:
                try:
                    os.remove(os.path.join(image_dir, row[0]))
                except:
                    pass

            # Reset index rollback
            current_index -= len(attempt_rows)

    print(f"\nFinished condition: {condition['name']}")

    return collected_rows, current_index