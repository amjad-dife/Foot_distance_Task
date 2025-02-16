import cv2 # type: ignore
import mediapipe as mp # type: ignore
import os
from config.constants import LEFT_ANKLE, RIGHT_ANKLE
from config.paths import OUTPUT_LOGS_DIR

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_foot_keypoints(frame):
    """
    Extract foot keypoints (ankles) from a frame using MediaPipe Pose.
    Args:
        frame: Input frame (numpy array).
    Returns:
        left_ankle_2d: 2D coordinates of the left ankle (x, y).
        right_ankle_2d: 2D coordinates of the right ankle (x, y).
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    
    
    log_frame_landmarks(frame,results.pose_landmarks.landmark)

    # Extract ankle keypoints from the results
    if results.pose_landmarks:
        # Get 2D coordinates of the ankles
        left_ankle = results.pose_landmarks.landmark[LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[RIGHT_ANKLE]

        # Convert normalized coordinates to pixel values
        h, w, _ = frame.shape
        left_ankle_2d = (int(left_ankle.x * w), int(left_ankle.y * h))
        right_ankle_2d = (int(right_ankle.x * w), int(right_ankle.y * h))

        return left_ankle_2d, right_ankle_2d
    else:
        return None, None

def extract_foot_3D_keypoints(frame):
    """
    Extract foot keypoints (ankles) from a frame using MediaPipe Pose.
    Args:
        frame: Input frame (numpy array).
    Returns:
        left_ankle_3d
        right_ankle_3d
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    
    # Extract ankle keypoints from the results
    if results.pose_landmarks:
        # Get 2D coordinates of the ankles
        left_ankle = results.pose_landmarks.landmark[LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[RIGHT_ANKLE]
        
        # Convert normalized coordinates to pixel values
        h, w, _ = frame.shape
        left_ankle_3d = (int(left_ankle.x * w), int(left_ankle.y * h), left_ankle.z) 
        right_ankle_3d = (int(right_ankle.x * w), int(right_ankle.y * h), right_ankle.z) 

        return left_ankle_3d, right_ankle_3d
    else:
        return None, None
    
def extract_all_landmarks(frame):
    """
    Extract all 33 landmarks from a frame and return them as a dictionary.
    Args:
        frame: Input frame (numpy array).
    Returns:
        landmarks_dict: Dictionary with all 33 landmarks and their 2D coordinates.
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Extract all keypoints from the results
    landmarks_dict = {}
    if results.pose_landmarks:
        keypoint_dict = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
            7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
            14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
            17: "left_pinky", 18: "right_pinky", 19: "left_index",
            20: "right_index", 21: "left_thumb", 22: "right_thumb",
            23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
            31: "left_foot_index", 32: "right_foot_index"
        }

        h, w, _ = frame.shape
        for idx, name in keypoint_dict.items():
            landmark = results.pose_landmarks.landmark[idx]
            landmarks_dict[name] = (int(landmark.x * w), int(landmark.y * h))

    return landmarks_dict

def log_frame_landmarks(frame,landmarks):
    """
    Log the 3D landmarks of a frame to a text file.
    Args:
        frame: Input frame (numpy array).
        landmarks: 3D landmarks of the frame.
    """
    # Define output path
    output_path = os.path.join(OUTPUT_LOGS_DIR, "landmarks.txt")
    keypoint_dict = {
    0: "nose",1: "left eye (inner)",2: "left eye",
    3: "left eye (outer)",4: "right eye (inner)",5: "right eye",
    6: "right eye (outer)",7: "left ear",8: "right ear",
    9: "mouth (left)",10: "mouth (right)",11: "left shoulder",
    12: "right shoulder",13: "left elbow",14: "right elbow",
    15: "left wrist",16: "right wrist",17: "left pinky",
    18: "right pinky",19: "left index",20: "right index",
    21: "left thumb",22: "right thumb",23: "left hip",
    24: "right hip",25: "left knee",26: "right knee",
    27: "left ankle",28: "right ankle",29: "left heel",
    30: "right heel",31: "left foot index",32: "right foot index"}

    h, w, _ = frame.shape 

    # Write landmarks to text file
    with open(output_path, "w") as f:
        f.write(f"{'idx':<3} {'keypoint_name':<20} {'landmark.x':<10} {'landmark.y':<10} {'landmark.z':<10}\n")
        for idx, landmark in enumerate(landmarks):
            # Write the landmark index and its x, y, z coordinates
            f.write(f"{idx:<3} {keypoint_dict[idx]:<20} {landmark.x:<10.5f} {landmark.y:<10.5f} {landmark.z:<10.5f}\n")
            f.write(f"{idx:<3} {keypoint_dict[idx]:<20} {int(landmark.x* w):<10.5f} {int(landmark.y* h):<10.5f}\n")
            f.write("\n")