import cv2
import mediapipe as mp
import csv
from pathlib import Path

# Dataset location
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = DATASET_DIR / "alphabet_data_v2.csv"

# CSV columns
columns = ["label", "left_present", "right_present"]

for hand in ["left", "right"]:
    for i in range(21):
        columns.extend([
            f"{hand}_{i}_x",
            f"{hand}_{i}_y",
            f"{hand}_{i}_z"
        ])

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

sample_count = 0

print("ISL Alphabet Data Collection")
print("Show a sign and press its alphabet key (A-Z) to save.")
print("Press Q to quit.")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = {}

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                side = handedness.classification[0].label.lower()
                detected[side] = landmarks

                mp_draw.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            frame,
            "Show sign + press A-Z to save | Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples saved: {sample_count}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("ISL Alphabet Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            print("Key pressed:",key)

        if key == 27:
            break

        if ord("a") <= key <= ord("z"):
            if not detected:
                print("No hand detected. Try again.")
                continue

            left_values = [0.0] * 63
            right_values = [0.0] * 63

            for side, landmarks in detected.items():
                values = []

                for point in landmarks.landmark:
                    values.extend([point.x, point.y, point.z])

                if side == "left":
                    left_values = values
                elif side == "right":
                    right_values = values

            label = chr(key).upper()

            row = [
                label,
                int("left" in detected),
                int("right" in detected)
            ] + left_values + right_values

            with open(FILE_PATH, "a", newline="") as file:
                writer = csv.writer(file)

                if file.tell() == 0:
                    writer.writerow(columns)

                writer.writerow(row)
                print(f"Saved sample for {label}")
            sample_count += 1
            print(f"Saved {label} | Total samples: {sample_count}")

cap.release()
cv2.destroyAllWindows()

print(f"Collection completed. Dataset saved at: {FILE_PATH}")