import math

def estimate_height(landmarks):
    """
    Estimate the height of a person using the distance between the hips and shoulders.

    input:
        landmarks: A dictionary containing the 2D coordinates of body landmarks.
            Example: {'left_hip': (x1, y1), 'right_hip': (x2, y2), 'left_shoulder': (x3, y3), 'right_shoulder': (x4, y4)}

    return: The estimated height of the person.
    """
    left_hip = landmarks.get('left_hip')
    right_hip = landmarks.get('right_hip')
    left_shoulder = landmarks.get('left_shoulder')
    right_shoulder = landmarks.get('right_shoulder')

    if not left_hip or not right_hip or not left_shoulder or not right_shoulder:
        raise ValueError("Landmarks must include 'left_hip', 'right_hip', 'left_shoulder', and 'right_shoulder'.")

    # Calculate the average y-coordinate for hips and shoulders
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # Estimate the person's height as the distance between the average hip and shoulder y-coordinates
    person_height = abs(avg_hip_y - avg_shoulder_y)

    return person_height


def calculate_3d_distance(landmarks, average_height_in_meters=1.7, average_height_px=None):
    """Calculate 3D distance between ankles using scaling."""

    left_ankle = landmarks.get('left_ankle')
    right_ankle = landmarks.get('right_ankle')

    if not left_ankle or not right_ankle:
        raise ValueError("Landmarks must include 'left_ankle' and 'right_ankle'.")

    dx = right_ankle[0] - left_ankle[0]
    dy = right_ankle[1] - left_ankle[1]
    distance_2d_px = math.sqrt(dx**2 + dy**2)  # Distance in pixels

    person_height_px = estimate_height(landmarks)  # Height estimation in pixels


    if average_height_px is None:
        average_height_px= average_height_in_meters * 100  # Average height in pixels
        
    # Calculate the scale factor by comparing the estimated person height in pixels to the average height in pixels:
    if person_height_px == 0:
        person_height_px = 0.1
        #raise ValueError("Estimated person height in pixels is zero, cannot scale.")
    
    scale_factor = average_height_px / person_height_px  # Scale relative to average height in px

    distance_3d_px = distance_2d_px * scale_factor  # Scaled distance in pixels

    # Convert to meters only at the end:
    meters_per_pixel = average_height_in_meters / average_height_px  # Meters per pixel
    distance_3d_meters = distance_3d_px * meters_per_pixel

    info = {
        "person_height_px": person_height_px,
        "average_height_px": average_height_px,
        "distance_2d_px": distance_2d_px,
        "scale_factor": scale_factor,
        "distance_3d_meters": distance_3d_meters
    }

    return distance_3d_px, info