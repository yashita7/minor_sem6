import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate angle between three points.
    a, b, c = [x, y, z]
    Angle at point b.
    """

    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_pose_features(frame):
    """
    Extract meaningful pose angles from a frame.
    """

    pose = frame["pose"]

    features = {}

    # Right elbow angle
    features["right_elbow"] = calculate_angle(
        pose[12],  # shoulder
        pose[14],  # elbow
        pose[16]   # wrist
    )

    # Left elbow angle
    features["left_elbow"] = calculate_angle(
        pose[11],
        pose[13],
        pose[15]
    )

    # Right knee angle
    features["right_knee"] = calculate_angle(
        pose[24],
        pose[26],
        pose[28]
    )

    # Left knee angle
    features["left_knee"] = calculate_angle(
        pose[23],
        pose[25],
        pose[27]
    )

    return features


def extract_feature_sequence(smoothed_data):
    """
    Convert entire smoothed sequence into feature vectors.
    """

    feature_sequence = []

    for frame in smoothed_data:
        features = extract_pose_features(frame)
        features["frame"] = frame["frame"]
        features["timestamp"] = frame["timestamp"]

        feature_sequence.append(features)

    return feature_sequence
