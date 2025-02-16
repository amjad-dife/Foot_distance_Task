import torch  # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils.pose_utils import extract_foot_keypoints
from utils.depth_utils import get_depth_map_using_depthAnyThing, estimate_depth_anyThing
from utils.visualization import draw_foot_distance, line_plot
from utils.file_io import load_video, save_video
from depth_anything.dpt import DepthAnything # type: ignore


def process_video_anything_approach(video_name, encoder='vits'):
    """
    Process a video to measure foot distance.
    Args:
        video_name: Name of the video file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    # Load the model 
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).to(device).eval()

    
    # Load video
    cap = load_video(video_name)
    frames = []
    video_distance = []
    left_ankle_3d_list = []
    right_ankle_3d_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"frame shape : {frame.shape}")

        # Extract foot keypoints
        left_ankle_2d, right_ankle_2d = extract_foot_keypoints(frame)
        
        print(f"left_ankle_2D :{left_ankle_2d}, right_ankle_2D : {right_ankle_2d}")

        if left_ankle_2d and right_ankle_2d:

            # get the depth_map of the entire frame 
            depth_map = get_depth_map_using_depthAnyThing(frame, depth_anything, device)

            # Estimate depth (placeholder for now)
            left_ankle_3d = estimate_depth_anyThing(depth_map, [left_ankle_2d], frame.shape)[0]
            right_ankle_3d = estimate_depth_anyThing(depth_map, [right_ankle_2d], frame.shape)[0]
            
            print(f"left_ankle_3D :{left_ankle_3d}, right_ankle_3D : {right_ankle_3d}")

            # Calculate 3D distance between feet
            distance = np.linalg.norm(np.array(left_ankle_3d) - np.array(right_ankle_3d))
            
            video_distance.append(distance)
            left_ankle_3d_list.append(left_ankle_3d)
            right_ankle_3d_list.append(right_ankle_3d)

            print(f"distance : {distance}")
            print("===============================================")

            # Visualize foot distance
            frame = draw_foot_distance(frame, left_ankle_2d, right_ankle_2d, distance)

        # Save frame
        frames.append(frame)

    # Save processed video
    save_video(frames, f"processed_with_depth_anything_{encoder}_{video_name}")

    # create and save a line plot that shows the distance accross the different frames of the video 
    line_plot(encoder,video_distance, video_name, left_ankle_3d_list, right_ankle_3d_list)

    cap.release() 
    return video_distance
