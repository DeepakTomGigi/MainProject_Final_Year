import cv2
import os

def detect_keyframes(video_path, threshold=0.2, min_scene_length=10, skip_frames=5, downsample_ratio=0.5):
    """
    Detects keyframes in a video based on optical flow magnitude.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Optical flow magnitude threshold to detect significant motion.
        min_scene_length (int): Minimum time between consecutive keyframes in seconds.
        skip_frames (int): Number of frames to skip for efficiency.
        downsample_ratio (float): Factor by which to downsample frames for faster processing.

    Returns:
        list: A list of tuples (frame_index, frame_image) for detected keyframes.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    success, prev_frame = video.read()

    if not success:
        raise ValueError("Error reading the video file.")

    # Downsample the previous frame
    prev_frame = cv2.resize(prev_frame, None, fx=downsample_ratio, fy=downsample_ratio)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    keyframes = []
    curr_frame = 0
    last_keyframe = -min_scene_length * fps  # Ensure the first keyframe is picked
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        # Skip frames for computational efficiency
        for _ in range(skip_frames):
            success = video.grab()
            curr_frame += 1
            if not success:
                break

        success, curr_frame_img = video.retrieve()
        if not success:
            break

        # Downsample the current frame
        curr_frame_resized = cv2.resize(curr_frame_img, None, fx=downsample_ratio, fy=downsample_ratio)
        curr_frame_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)

        # Compute Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = mag.mean()

        # Add keyframe if motion exceeds threshold and scene length criteria
        if mean_magnitude > threshold and (curr_frame - last_keyframe) > min_scene_length * fps:
            keyframes.append((curr_frame, curr_frame_img))
            last_keyframe = curr_frame  # Update the last keyframe position

        prev_frame_gray = curr_frame_gray

    video.release()
    return keyframes



