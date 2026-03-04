# Configurations
HOST = "172.23.48.1"
PORT = 2000

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

FPS = 10
FIXED_DELTA_SECONDS = 1.0 / FPS

FRAMES_PER_MODE = {
    "train": 3000,
    "test": 1000
}

WEATHER_CONFIG = [ # Environment conditions
    {"name": "day_clear", "sun": 45, "rain": 0},
    {"name": "day_rain", "sun": 45, "rain": 80},
    {"name": "night_clear", "sun": -30, "rain": 0},
    {"name": "night_rain", "sun": -30, "rain": 80},
]