import cv2 # type: ignore
import os
from config.paths import INPUT_VIDEOS_DIR, OUTPUT_VIDEOS_DIR

def load_video(video_name):
    """
    Load a video from the input directory.
    Args:
        video_name: Name of the video file.
    Returns:
        cap: VideoCapture object.
    """
    video_path = os.path.join(INPUT_VIDEOS_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    return cap

def save_video(frames, video_name):
    """
    Save processed frames as a video in the output directory.
    Args:
        frames: List of frames to save.
        video_name: Name of the output video file.
    """
    if not frames:
        return

    # Get frame dimensions
    h, w, _ = frames[0].shape

    # Define output path
    output_path = os.path.join(OUTPUT_VIDEOS_DIR, video_name)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    out.release()