import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_hand_skeleton(hand_landmarks):
    plt.figure(figsize=(6,6))

    xs = [lm[0] for lm in hand_landmarks]
    ys = [lm[1] for lm in hand_landmarks]

    plt.scatter(xs, ys)

    for connection in HAND_CONNECTIONS:
        x_coords = [
            hand_landmarks[connection[0]][0],
            hand_landmarks[connection[1]][0]
        ]
        y_coords = [
            hand_landmarks[connection[0]][1],
            hand_landmarks[connection[1]][1]
        ]
        plt.plot(x_coords, y_coords)

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title("Extracted Mudra Skeleton")
    plt.show()

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print("No hands detected.")
            return

        h, w, _ = image.shape

        # Create ONE figure
        plt.figure(figsize=(6,6))

        for hand_landmarks in results.multi_hand_landmarks:

            coords = []
            for lm in hand_landmarks.landmark:
                coords.append([lm.x * w, lm.y * h, lm.z])

            xs = [lm[0] for lm in coords]
            ys = [lm[1] for lm in coords]

            plt.scatter(xs, ys)

            for connection in HAND_CONNECTIONS:
                plt.plot(
                    [coords[connection[0]][0], coords[connection[1]][0]],
                    [coords[connection[0]][1], coords[connection[1]][1]]
                )

        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.title("Both Hands Mudra Skeleton")
        plt.show()

if __name__ == "__main__":
    process_image("C:\\Users\\Rishu\\OneDrive\\Desktop\\minor_sem_6\\frames\\sample_1.jpg")   # change to your image path