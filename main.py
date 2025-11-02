from test import ObjectTracking
from ultralytics import YOLO

INPUT_PATH = "assets/video/vehicle-counting.mp4"
OUTPUT_PATH = "assets/video/vehicle-counting-result.mp4"
LOG_PATH = "vehicle_count_log.txt"
if __name__ == "__main__":
    obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH, LOG_PATH)
    obj.process()