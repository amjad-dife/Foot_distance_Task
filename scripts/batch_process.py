import os
from config.paths import INPUT_VIDEOS_DIR, OUTPUT_VIDEOS_DIR
from scripts.process_video import process_video

def batch_process():
    """
    Process all videos in the input directory and save results in the output directory.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    # Process each video one at a time
    for video_name in os.listdir(INPUT_VIDEOS_DIR):
        if video_name.endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: {video_name}")
            process_video(video_name)
            print(f"Finished processing: {video_name}")

    print("Batch processing complete!")

if __name__ == "__main__":
    batch_process()