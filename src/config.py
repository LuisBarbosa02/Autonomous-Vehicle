# Dataset configuration
HOST = "172.23.48.1"
PORT = 2000

TOWN = "Town05"

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

FPS = 10
FIXED_DELTA_SECONDS = 1.0 / FPS

FRAMES_PER_MODE = {
    "train": 8000,
    "test": 2000
}

WEATHER_CONFIG = [ # Environment conditions
    {"name": "day_clear", "sun": 45, "rain": 0},
    {"name": "day_rain", "sun": 45, "rain": 80},
    {"name": "night_clear", "sun": -30, "rain": 0},
    {"name": "night_rain", "sun": -30, "rain": 80},
]

# Training configuration

TRAIN_DATASET_PATH = "data/train/labels.csv"
TRAIN_IMAGES_PATH = "data/train/images"
TEST_DATASET_PATH = "data/test/labels.csv"
TEST_IMAGES_PATH = "data/test/images"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2