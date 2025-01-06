import os
import cv2
import datetime

def save_keyframes(keyframes, video_path, base_output_folder="outputs/keyframes"):
    """
    Saves keyframes to a uniquely named directory based on the input video name.

    Args:
        keyframes (list): List of tuples (frame_index, frame_image) for detected keyframes.
        video_path (str): Path to the input video file.
        base_output_folder (str): Base directory where keyframe folders will be created.
    """
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a timestamped folder to ensure uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder = os.path.join(base_output_folder, f"{video_name}_{timestamp}")

    # Create the folder if it doesn't exist
    os.makedirs(unique_folder, exist_ok=True)

    for frame_index, frame_image in keyframes:
        filename = os.path.join(unique_folder, f"keyframe_{frame_index}.jpg")
        cv2.imwrite(filename, frame_image)
        print(f"Saved keyframe {frame_index} to {filename}")

    print(f"Keyframes saved to: {unique_folder}")



# def display_keyframes(keyframes):
#     """
#     Displays keyframes in a grid using matplotlib.

#     Args:
#         keyframes (list): List of tuples (frame_index, frame_image) for detected keyframes.
#     """
#     num_keyframes = len(keyframes)
#     plt.figure(figsize=(15, num_keyframes * 3))

#     for i, (frame_index, frame_image) in enumerate(keyframes):
#         plt.subplot(1, num_keyframes, i + 1)
#         # Convert BGR to RGB for displaying with matplotlib
#         frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
#         plt.imshow(frame_rgb)
#         plt.title(f"Frame {frame_index}")
#         plt.axis('off')

#     plt.show()
