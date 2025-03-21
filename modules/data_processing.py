import cv2
import os
import numpy as np

def detect_keyframes(video_path, motion_threshold=0.2, hist_threshold=50, min_scene_length=10, skip_frames=2, downsample_ratio=0.5):
    """
    Detects keyframes in a video based on optical flow magnitude and histogram analysis.

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

    prev_frame = cv2.resize(prev_frame, None, fx=downsample_ratio, fy=downsample_ratio)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_frame_gray], [0], None, [256], [0, 256])
    keyframes = []
    curr_frame = 0
    last_keyframe = -min_scene_length * fps

    while True:
        for _ in range(skip_frames):
            success = video.grab()
            curr_frame += 1
            if not success:
                break

        success, curr_frame_img = video.retrieve()
        if not success:
            break

        curr_frame_resized = cv2.resize(curr_frame_img, None, fx=downsample_ratio, fy=downsample_ratio)
        curr_frame_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)
        curr_hist = cv2.calcHist([curr_frame_gray], [0], None, [256], [0, 256])

        # Motion detection
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = mag.mean()

        # Histogram difference
        hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)

        # Keyframe condition
        if (mean_magnitude > motion_threshold or hist_diff > hist_threshold) and (curr_frame - last_keyframe) > min_scene_length * fps:
            keyframes.append((curr_frame, curr_frame_img))
            last_keyframe = curr_frame

        prev_frame_gray = curr_frame_gray
        prev_hist = curr_hist

    video.release()
    return keyframes
