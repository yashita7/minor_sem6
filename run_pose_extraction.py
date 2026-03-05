import os
import json

from utils.pose_extraction import extract_landmarks
from utils.landmark_smoothing import smooth_landmarks
from utils.feature_engineering import extract_feature_sequence


if __name__ == "__main__":

    # ==============================
    # Paths
    # ==============================
    video_path = "data/videos/sample_dance.mp4"
    raw_output_path = "data/output/raw_landmarks.json"
    smooth_output_path = "data/output/smoothed_landmarks.json"
    feature_output_path = "data/output/pose_features.json"

    target_fps = 30

    # ==============================
    # Check Video
    # ==============================
    if not os.path.exists(video_path):
        print(f"❌ Video not found at: {video_path}")
        exit()

    # ==============================
    # STEP 1: Extract Landmarks
    # ==============================
    print("🚀 Extracting raw landmarks...\n")

    raw_data = extract_landmarks(
        video_path=video_path,
        target_fps=target_fps,
        save_json_path=raw_output_path
    )

    # ==============================
    # STEP 2: Smooth Landmarks
    # ==============================
    print("\n🔵 Applying temporal smoothing...\n")

    smoothed_data = smooth_landmarks(raw_data, window_size=5)

    os.makedirs(os.path.dirname(smooth_output_path), exist_ok=True)

    with open(smooth_output_path, "w") as f:
        json.dump(smoothed_data, f)

    print("✅ Smoothed landmarks saved.")

    # ==============================
    # STEP 3: Extract Pose Features
    # ==============================
    print("\n🟣 Extracting pose angle features...\n")

    feature_data = extract_feature_sequence(smoothed_data)

    with open(feature_output_path, "w") as f:
        json.dump(feature_data, f)

    print("✅ Pose features saved.")

    # ==============================
    print("\n🎉 Pipeline Completed Successfully.")
