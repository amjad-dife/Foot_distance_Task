import cv2  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd

def draw_foot_distance(frame, left_ankle_2d, right_ankle_2d, distance):
    """
    Draw the foot distance on the frame.
    Args:
        frame: Input frame (numpy array).
        left_ankle_2d: 2D coordinates of the left ankle.
        right_ankle_2d: 2D coordinates of the right ankle.
        distance: Calculated foot distance.
    Returns:
        frame: Frame with visualization.
    """
    # Draw line between ankles
    cv2.line(frame, left_ankle_2d, right_ankle_2d, (0, 255, 0), 2)

    # Display distance
    cv2.putText(frame, f"Distance: {distance:.2f} Pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame 

def draw_foot_distance_for_scaling_approach(frame, left_ankle_2d, right_ankle_2d, distance, info):
    """
    Draw the foot distance on the frame.
    Args:
        frame: Input frame (numpy array).
        left_ankle_2d: 2D coordinates of the left ankle.
        right_ankle_2d: 2D coordinates of the right ankle.
        distance: Calculated foot distance.
    Returns:
        frame: Frame with visualization.
    """
    # Draw line between ankles
    cv2.line(frame, left_ankle_2d, right_ankle_2d, (0, 255, 0), 2)

   
    # Display additional information 
    cv2.putText(frame, f"Average Height in pxl: {info['average_height_px']:.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Person Height in pxl:  {info['person_height_px']:.2f} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
    cv2.putText(frame, f"Scale Factor:   {info['scale_factor']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0,255), 2)
    cv2.putText(frame, f"2D Distance in pxl:    {info['distance_2d_px']:.2f} px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"3D Distance in meters:    {info['distance_3d_meters']:.2f} meters", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return frame




def line_plot(encoder,video_distance, video_name, left_ankle_3d_list, right_ankle_3d_list):
    """
    Create a line plot that shows the distance across the different frames of the video.
    Args:
        video_distance: List of distance values.
        video_name: Name of the video file.
        left_ankle_3d_list: List of 3D coordinates for the left ankle.
        right_ankle_3d_list: List of 3D coordinates for the right ankle.
    """
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the distance
    sns.lineplot(data=video_distance, ax=ax1, label='Foot Distance')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Distance')
    ax1.set_title('Foot Distance and 3D Keypoints')

    # Create a second y-axis for the 3D keypoints
    ax2 = ax1.twinx()
    left_ankle_3d_array = np.array(left_ankle_3d_list)
    right_ankle_3d_array = np.array(right_ankle_3d_list)

    sns.lineplot(data=left_ankle_3d_array[:, 0], ax=ax2, label='Left Ankle X')
    sns.lineplot(data=left_ankle_3d_array[:, 1], ax=ax2, label='Left Ankle Y')
    sns.lineplot(data=left_ankle_3d_array[:, 2], ax=ax2, label='Left Ankle Z')

    sns.lineplot(data=right_ankle_3d_array[:, 0], ax=ax2, label='Right Ankle X')
    sns.lineplot(data=right_ankle_3d_array[:, 1], ax=ax2, label='Right Ankle Y')
    sns.lineplot(data=right_ankle_3d_array[:, 2], ax=ax2, label='Right Ankle Z')

    ax2.set_ylabel('3D Coordinates')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.savefig(f'foot_distance_{encoder}_{video_name}.png')


def plot_distance_comparison(distance_lists, approach_names, title="Distance Comparison", save_path="distance_comparison.png"):
    """
    Plots a line graph comparing distances from different approaches and saves the figure.

    Args:
        distance_lists: A list of lists, where each inner list contains the 
                        distances calculated by a specific approach.  
                        All inner lists should have the same length (number of frames).
        approach_names: A list of strings, where each string is the name of an 
                        approach.  Must be the same length as distance_lists.
        title: The title of the plot.
        save_path: The path (including filename) where the plot should be saved.

    Raises:
        ValueError: If the input lists are not of the same length or if the 
                    inner distance lists are not of the same length.
    """

    if len(distance_lists) != len(approach_names):
        raise ValueError("distance_lists and approach_names must have the same length.")

    num_frames = len(distance_lists[0]) if distance_lists else 0

    for distances in distance_lists:
        if len(distances) != num_frames:
            raise ValueError("All inner lists in distance_lists must have the same length.")

    data = []
    for i, distances in enumerate(distance_lists):
        for frame, distance in enumerate(distances):
            data.append({"Frame": frame, "Distance": distance, "Approach": approach_names[i]})

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Frame", y="Distance", hue="Approach", data=df)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend(title="Approach")
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)  # You can adjust dpi for image quality

    


