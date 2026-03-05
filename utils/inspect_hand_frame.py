# import json
# import numpy as np


# # MediaPipe Hand Landmark Index Mapping
# FINGER_MAP = {
#     "thumb": [1, 2, 3, 4],
#     "index": [5, 6, 7, 8],
#     "middle": [9, 10, 11, 12],
#     "ring": [13, 14, 15, 16],
#     "pinky": [17, 18, 19, 20]
# }


# def extract_hand_positions(frame_data):
#     """
#     Extract structured finger positions from a single frame.
#     """

#     hands_info = {}

#     for hand_label in ["left_hand", "right_hand"]:

#         hand_landmarks = frame_data[hand_label]

#         hand_info = {}

#         for finger_name, indices in FINGER_MAP.items():
#             hand_info[finger_name] = [
#                 hand_landmarks[i] for i in indices
#             ]

#         hands_info[hand_label] = hand_info

#     return hands_info


# def inspect_frame(json_path, frame_number=0):
#     """
#     Inspect a particular frame from smoothed JSON.
#     """

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     if frame_number >= len(data):
#         print("❌ Frame number out of range.")
#         return

#     frame_data = data[frame_number]

#     print(f"\nInspecting Frame: {frame_data['frame']}")
#     print(f"Timestamp: {frame_data['timestamp']}s\n")

#     hand_positions = extract_hand_positions(frame_data)

#     for hand, fingers in hand_positions.items():
#         print(f"\n=== {hand.upper()} ===")

#         for finger, coords in fingers.items():
#             print(f"\n{finger.capitalize()} Finger:")
#             for idx, point in enumerate(coords):
#                 print(f"  Joint {idx+1}: {np.round(point, 4)}")


import json
import numpy as np


FINGER_MAP = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}


def is_hand_present(hand_landmarks):
    return any(any(coord != 0 for coord in point) for point in hand_landmarks)


def find_frame_with_hands(data):
    for frame_data in data:
        if is_hand_present(frame_data["left_hand"]) or \
           is_hand_present(frame_data["right_hand"]):
            return frame_data["frame"]
    return None


def inspect_frame(json_path, frame_number=None):

    with open(json_path, "r") as f:
        data = json.load(f)

    if frame_number is None:
        frame_number = find_frame_with_hands(data)

        if frame_number is None:
            print("❌ No hands detected in entire video.")
            return

        print(f"\nAuto-selected frame with hands: {frame_number}")

    frame_data = data[frame_number]

    print(f"\nInspecting Frame: {frame_data['frame']}")
    print(f"Timestamp: {frame_data['timestamp']}s")

    for hand_label in ["left_hand", "right_hand"]:

        print(f"\n=== {hand_label.upper()} ===")

        if not is_hand_present(frame_data[hand_label]):
            print("No hand detected.")
            continue

        hand_landmarks = frame_data[hand_label]

        for finger_name, indices in FINGER_MAP.items():

            print(f"\n{finger_name.capitalize()} Finger:")

            for idx, lm_index in enumerate(indices):
                point = hand_landmarks[lm_index]
                print(f"  Joint {idx+1}: {np.round(point, 4)}")
