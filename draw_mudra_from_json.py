import json
import matplotlib.pyplot as plt

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def is_hand_present(hand):
    return any(any(coord != 0 for coord in point) for point in hand)


def draw_both_hands(left_hand, right_hand, title="Mudra"):
    plt.figure(figsize=(6,6))

    # Draw left hand
    if is_hand_present(left_hand):
        xs = [p[0] for p in left_hand]
        ys = [p[1] for p in left_hand]
        plt.scatter(xs, ys)

        for c in HAND_CONNECTIONS:
            plt.plot(
                [left_hand[c[0]][0], left_hand[c[1]][0]],
                [left_hand[c[0]][1], left_hand[c[1]][1]]
            )

    # Draw right hand
    if is_hand_present(right_hand):
        xs = [p[0] for p in right_hand]
        ys = [p[1] for p in right_hand]
        plt.scatter(xs, ys)

        for c in HAND_CONNECTIONS:
            plt.plot(
                [right_hand[c[0]][0], right_hand[c[1]][0]],
                [right_hand[c[0]][1], right_hand[c[1]][1]]
            )

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_and_draw(json_path, frame_number=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Auto select first frame with hands
    if frame_number is None:
        for frame in data:
            if is_hand_present(frame["left_hand"]) or \
               is_hand_present(frame["right_hand"]):
                frame_number = frame["frame"]
                break

    frame_data = data[frame_number]

    print("Drawing frame:", frame_data["frame"])

    draw_both_hands(
        frame_data["left_hand"],
        frame_data["right_hand"],
        title=f"Mudra - Frame {frame_number}"
    )


if __name__ == "__main__":
    json_path = "data/output/smoothed_landmarks.json"
    load_and_draw(json_path, frame_number=None)