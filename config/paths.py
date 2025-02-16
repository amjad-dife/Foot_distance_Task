import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "data/input_videos")
OUTPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "data/output_videos")
OUTPUT_LOGS_DIR = os.path.join(BASE_DIR, "data/Logs")
# Model paths (if needed)
POSE_MODEL_PATH = os.path.join(BASE_DIR, "models/pose_estimation")
DEPTH_MODEL_PATH = os.path.join(BASE_DIR, "models/depth_estimation")