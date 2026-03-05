# import os
# import cv2
# import json
# import numpy as np
# import mediapipe as mp


# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# mp_face = mp.solutions.face_mesh


# def extract_landmarks(video_path, target_fps=30, save_json_path=None):
#     """
#     Extract pose, hand, and face landmarks at fixed FPS.
#     Returns structured list of dictionaries.
#     """

#     cap = cv2.VideoCapture(video_path)

#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     if original_fps <= 0:
#         original_fps = 30

#     frame_interval = 1.0 / target_fps
#     next_capture_time = 0.0

#     print(f"Original FPS: {original_fps}")
#     print(f"Target FPS: {target_fps}")

#     results_data = []
#     frame_index = 0

#     with mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6
#     ) as pose_model, mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=2,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6
#     ) as hands_model, mp_face.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6
#     ) as face_model:

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             timestamp = frame_index / original_fps

#             # Time-based sampling (better than frame modulus)
#             if timestamp >= next_capture_time:

#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 pose_result = pose_model.process(rgb)
#                 hand_result = hands_model.process(rgb)
#                 face_result = face_model.process(rgb)

#                 frame_data = {
#                     "frame": int(frame_index),
#                     "timestamp": round(float(timestamp), 4),
#                     "pose": [],
#                     "left_hand": [],
#                     "right_hand": [],
#                     "face": []
#                 }

#                 # ---------------- POSE (33) ----------------
#                 if pose_result.pose_landmarks:
#                     frame_data["pose"] = [
#                         [lm.x, lm.y, lm.z, lm.visibility]
#                         for lm in pose_result.pose_landmarks.landmark
#                     ]
#                 else:
#                     frame_data["pose"] = [[0, 0, 0, 0]] * 33

#                 # ---------------- HANDS (21 each) ----------------
#                 if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
#                     for hand_lms, handedness in zip(
#                         hand_result.multi_hand_landmarks,
#                         hand_result.multi_handedness
#                     ):
#                         label = handedness.classification[0].label
#                         points = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]

#                         if label == "Left":
#                             frame_data["left_hand"] = points
#                         else:
#                             frame_data["right_hand"] = points

#                 if not frame_data["left_hand"]:
#                     frame_data["left_hand"] = [[0, 0, 0]] * 21

#                 if not frame_data["right_hand"]:
#                     frame_data["right_hand"] = [[0, 0, 0]] * 21

#                 # ---------------- FACE (468) ----------------
#                 if face_result.multi_face_landmarks:
#                     face_landmarks = face_result.multi_face_landmarks[0]
#                     frame_data["face"] = [
#                         [lm.x, lm.y, lm.z]
#                         for lm in face_landmarks.landmark
#                     ]
#                 else:
#                     frame_data["face"] = [[0, 0, 0]] * 468

#                 results_data.append(frame_data)

#                 print(f"Processed frame {frame_index} | Time {timestamp:.2f}s")

#                 next_capture_time += frame_interval

#             frame_index += 1

#     cap.release()

#     print(f"\nTotal extracted frames: {len(results_data)}")

#     # Save JSON if path provided
#     if save_json_path:
#         os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
#         with open(save_json_path, "w") as f:
#             json.dump(results_data, f)
#         print(f"Saved JSON to: {save_json_path}")

#     return results_data

import os
import cv2
import json
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def extract_landmarks(video_path, target_fps=30, save_json_path=None):

    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    frame_interval = 1.0 / target_fps
    next_capture_time = 0.0

    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")

    results_data = []
    frame_index = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose_model, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,  # lowered for mudras
        min_tracking_confidence=0.4
    ) as hands_model, mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_model:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / original_fps

            if timestamp >= next_capture_time:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pose_result = pose_model.process(rgb)
                hand_result = hands_model.process(rgb)
                face_result = face_model.process(rgb)

                frame_data = {
                    "frame": frame_index,
                    "timestamp": round(timestamp, 4),
                    "pose": [],
                    "left_hand": [],
                    "right_hand": [],
                    "face": []
                }

                # -------- POSE --------
                if pose_result.pose_landmarks:
                    frame_data["pose"] = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in pose_result.pose_landmarks.landmark
                    ]
                else:
                    frame_data["pose"] = [[0, 0, 0, 0]] * 33

                # -------- HANDS --------
                detected_hands = False

                if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                    detected_hands = True

                    for hand_lms, handedness in zip(
                        hand_result.multi_hand_landmarks,
                        hand_result.multi_handedness
                    ):
                        label = handedness.classification[0].label
                        points = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]

                        if label == "Left":
                            frame_data["left_hand"] = points
                        else:
                            frame_data["right_hand"] = points

                if not frame_data["left_hand"]:
                    frame_data["left_hand"] = [[0, 0, 0]] * 21

                if not frame_data["right_hand"]:
                    frame_data["right_hand"] = [[0, 0, 0]] * 21

                # -------- FACE --------
                if face_result.multi_face_landmarks:
                    face_landmarks = face_result.multi_face_landmarks[0]
                    frame_data["face"] = [
                        [lm.x, lm.y, lm.z]
                        for lm in face_landmarks.landmark
                    ]
                else:
                    frame_data["face"] = [[0, 0, 0]] * 468

                results_data.append(frame_data)

                print(
                    f"Frame {frame_index} | "
                    f"Hands Detected: {detected_hands}"
                )

                next_capture_time += frame_interval

            frame_index += 1

    cap.release()

    print(f"\nTotal extracted frames: {len(results_data)}")

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(results_data, f)

        print(f"Saved JSON to: {save_json_path}")

    return results_data

