import numpy as np # type: ignore
from utils.pose_utils import extract_foot_3D_keypoints
from utils.visualization import draw_foot_distance
from utils.file_io import load_video, save_video



def process_3D_video(video_name):
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

        # Extract foot keypoints
        left_ankle_3d, right_ankle_3d = extract_foot_3D_keypoints(frame)
        
        left_ankle_2d = left_ankle_3d[:2]
        right_ankle_2d = right_ankle_3d[:2]

        # Calculate 3D distance between feet
        distance = np.linalg.norm(np.array(left_ankle_3d) - np.array(right_ankle_3d))

        video_distance.append(distance)
        #print(f"z approach: distance:{distance}, left 2d: {left_ankle_2d}, right: {right_ankle_2d}")
        
        # Visualize foot distance
        frame = draw_foot_distance(frame, left_ankle_2d, right_ankle_2d, distance)

        # Save frame
        frames.append(frame)

    # Save processed video
    save_video(frames, f"processed_3D_{video_name}")

    cap.release()
    return video_distance