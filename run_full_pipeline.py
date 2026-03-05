from utils.pose_extraction import extract_landmarks

frames = extract_landmarks("data/videos/sample_dance.mp4")

print("Frames extracted:", len(frames))