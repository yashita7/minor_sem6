# from utils.inspect_hand_frame import inspect_frame

# if __name__ == "__main__":

#     json_path = "data/output/smoothed_landmarks.json"

#     # Change frame number here
#     inspect_frame(json_path, frame_number=100)

from utils.inspect_hand_frame import inspect_frame

if __name__ == "__main__":

    json_path = "data/output/smoothed_landmarks.json"

    inspect_frame(json_path, frame_number="sample_1")