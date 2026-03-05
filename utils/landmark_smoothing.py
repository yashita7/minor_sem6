import numpy as np


def smooth_landmarks(landmark_sequence, window_size=5):
    """
    Apply temporal moving average smoothing.
    landmark_sequence: list of frames (output from extract_landmarks)
    Returns smoothed sequence.
    """

    smoothed_data = []

    total_frames = len(landmark_sequence)

    for i in range(total_frames):

        start = max(0, i - window_size // 2)
        end = min(total_frames, i + window_size // 2 + 1)

        window_frames = landmark_sequence[start:end]

        smoothed_frame = {
            "frame": landmark_sequence[i]["frame"],
            "timestamp": landmark_sequence[i]["timestamp"],
            "pose": [],
            "left_hand": [],
            "right_hand": [],
            "face": []
        }

        # ----------- Helper Function -----------
        def average_landmarks(key):
            arr = np.array([f[key] for f in window_frames])
            return np.mean(arr, axis=0).tolist()

        smoothed_frame["pose"] = average_landmarks("pose")
        smoothed_frame["left_hand"] = average_landmarks("left_hand")
        smoothed_frame["right_hand"] = average_landmarks("right_hand")
        smoothed_frame["face"] = average_landmarks("face")

        smoothed_data.append(smoothed_frame)

    return smoothed_data
