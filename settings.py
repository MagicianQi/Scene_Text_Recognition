# Redis
REDIS_HOST = ""
REDIS_PORT = ""
REDIS_PASSWORD = ""

# Model
EAST_API_URL = "http://text_detection:8503/v1/models/east:predict"
CRNN_API_URL = "http://text_recognition:8501/v1/models/crnn:predict"

# Image
MAX_IMAGE_SIZE = 960
SIDE_VERTEX_PIXEL_THRESHOLD = 0.9
TRUNC_THRESHOLD = 0.1
PIXEL_SIZE = 4
EPSILON = 1e-4

# test image config
FONT_FILE_PATH = "/opt/app/static/simhei.ttf"
SAVE_TEMP_IMAGE_PATH = "static/temporary/"
FONT_SIZE = 12
FONT_COLOR = (0, 0, 255)
LINE_COLOR = (255, 0, 0)
TEMP_IMAGE_RANDOM_NAME_LENGTH = 8

# Logs
ERROR_LOG_FILE_PATH = "/opt/logs/error.log"
RECORD_LOG_FILE_PATH = "/opt/logs/record.log"

# GIF
GIF_FRAME_INTERVAL = 5
GIF_MAX_FRAME = 10
