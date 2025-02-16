import cv2  # type: ignore
import numpy as np # type: ignore
from utils.pose_utils import extract_all_landmarks, extract_foot_keypoints
from utils.scale_down_utils import calculate_3d_distance
from utils.visualization import draw_foot_distance_for_scaling_approach
from utils.file_io import load_video, save_video

def process_video_scale_approach(video_name,average_height= 1.7,average_height_px= None):
    """
    Process a video to measure foot distance.
    Args:
        video_name: Name of the video file.
    """
    # Load video
    cap = load_video(video_name)
    frames = []
    video_distance = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract all landmarks
        body_landmarks = extract_all_landmarks(frame)
        
        # Calculate 3D distance between feet
        distance , info = calculate_3d_distance(body_landmarks, average_height,average_height_px)
        
        # print the distance
        # Extract foot keypoints
        left_ankle_2d, right_ankle_2d = extract_foot_keypoints(frame)

        if left_ankle_2d and right_ankle_2d:

            video_distance.append(distance)

        
            # Visualize foot distance
            frame = draw_foot_distance_for_scaling_approach(frame, left_ankle_2d, right_ankle_2d, distance,info)

        # Save frame
        frames.append(frame)

    # Save processed video
    save_video(frames, f"processed_by_Scaling_{video_name}")

    cap.release()
    return video_distance