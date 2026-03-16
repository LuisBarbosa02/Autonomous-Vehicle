# Import libraries
import cv2
import time

from ..config import VIDEO_PATH, DT
from .vehicle import VehicleState
from .perception import Perception
from .visualization import transform_image, draw_overlay

class Simulator:
    """
    Autonomous vehicle simulator
    """
    def __init__(self):
        self.vehicle = VehicleState()
        self.perception = Perception()
        self.interventions = 0
        self.start_time = None

    def reset_vehicle(self):
        """
        Reset vehicle's state
        """
        self.vehicle = VehicleState()

    def run(self):
        """
        Run simulation
        """
        cap = cv2.VideoCapture(VIDEO_PATH)
        self.start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            simulated_frame = transform_image(frame.copy(), self.vehicle.x)

            steering = self.perception.predict_steering(simulated_frame)

            self.vehicle.update(steering)

            display = draw_overlay(simulated_frame, steering, self.vehicle.x, self.interventions)

            cv2.imshow("Autonomous vehicle simulation", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('i'):
                self.interventions += 1
                print("Human intervention triggered!")
                self.reset_vehicle()
            elif key == ord('q'):
                break

            # Maintain fixed timestep
            elapsed = time.time() - self.start_time
            time.sleep(max(0, DT - elapsed % DT))

        cap.release()
        cv2.destroyAllWindows()

        # Calculate autonomy metric
        elapsed_time = time.time() - self.start_time
        autonomy = max(0, (1 - (self.interventions * 6) / elapsed_time) * 100)

        print("\nSimulation Results")
        print("------------------")
        print("Elapsed time:", round(elapsed_time,2))
        print("Interventions:", self.interventions)
        print("Autonomy:", round(autonomy,2), "%")

# Run simulator
if __name__ == "__main__":
    sim = Simulator()
    sim.run()