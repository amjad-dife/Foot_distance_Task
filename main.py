from scripts.process_video import process_video 
from scripts.process_video_scale_approach import process_video_scale_approach 
from scripts.process_video_mediapipe_z_approach import process_3D_video
from scripts.process_video_depth_anything_approach import process_video_anything_approach
from scripts.process_video_depth_midas_Approach import process_video_MiDas_approach
from utils.visualization import plot_distance_comparison 


if __name__ == "__main__":
    
    video_name = "Kirolos_video.mp4"  # Replace with your video name

    ### Baseline approach :  
    # The problem of magnitude of the distance between the feet is not accurate
    video_distance_baseline =process_video(video_name) #

    ### 1. scale by obj_hight/ obj_average hight 
    # The idea is to:
    #           - assume the average_height of an object in the real scean in pixel and in meters ,
    #           - get a scaling facter that represents to what extend the original height of the object got affected by how far it is than the camera 
    #  The main limitation of that approach is how to estimate the average_hight in meters and in pixels 
    video_distance_scale = process_video_scale_approach(video_name,average_height= 1.7,average_height_px= 100) 

    ### 2. depth estimation using Z value from mediapipe 
    # The idea is to get the (x,y,z) value from mediapipe pose estimation module 
    video_distance_Z_mediapipe = process_3D_video(video_name) 

    ### 3. depth estimation using MiDas model 
    video_distance_MiDas = process_video_MiDas_approach(video_name,encoder='MiDaS_small')

    ### 4. depth estimation using depth anything model with different encoder variations 
    # Process video using the depth_anything approach
    video_distance_vits = process_video_anything_approach(video_name, encoder='vits')  # small depth any thing model
    video_distance_vitb = process_video_anything_approach(video_name, encoder='vitb') #  base depth any thing model
    #video_distance_vitl =process_video_anything_approach(video_name, encoder='vitl') # large depth any thing model

    # compare the Different approaches 
    distance_lists = [video_distance_baseline,video_distance_scale,video_distance_Z_mediapipe,video_distance_MiDas,video_distance_vits,video_distance_vitb]
    approach_names = ["baseline","scaling approach","Mediapipe_Z approach","video_distance_MiDas","Vits depth_anything","Vitb depth_anything"]

    plot_distance_comparison(distance_lists, approach_names, title="Comparison of Different Distance Approaches", save_path="distance_comparison.png")
