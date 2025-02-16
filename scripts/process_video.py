import numpy as np # type: ignore
from utils.pose_utils import extract_foot_keypoints
from utils.depth_utils import estimate_depth
from utils.visualization import draw_foot_distance
from utils.file_io import load_video, save_video

def process_video(video_name):
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
        left_ankle_2d, right_ankle_2d = extract_foot_keypoints(frame)
        
        #print(f"left_ankle_2D :{left_ankle_2d}, right_ankle_2D : {right_ankle_2d}")

        if left_ankle_2d and right_ankle_2d:
            # Estimate depth (placeholder for now) , the default is zero depth for that time 
            left_ankle_3d = estimate_depth(frame, [left_ankle_2d])[0]
            right_ankle_3d = estimate_depth(frame, [right_ankle_2d])[0]
            
            #print(f"left_ankle_3D :{left_ankle_3d}, right_ankle_3D : {right_ankle_3d}")

            # Calculate 3D distance between feet
            distance = np.linalg.norm(np.array(left_ankle_3d) - np.array(right_ankle_3d))
            video_distance.append(distance)

            # Visualize foot distance
            frame = draw_foot_distance(frame, left_ankle_2d, right_ankle_2d, distance)

        # Save frame
        frames.append(frame)

    # Save processed video
    save_video(frames, f"processed_{video_name}")

    cap.release() 
    return video_distance
