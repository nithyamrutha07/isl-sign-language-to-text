import cv2
import mediapipe as mp
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "alphabet_random_forest.pkl"

# Load trained model
saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

print("Random Forest model loaded successfully!")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Mirror camera frame
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
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Prepare feature values
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

        features = {
            "left_present": int("left" in detected),
            "right_present": int("right" in detected)
        }

        for hand, values in [
            ("left", left_values),
            ("right", right_values)
        ]:
            for i in range(21):
                features[f"{hand}_{i}_x"] = values[i * 3]
                features[f"{hand}_{i}_y"] = values[i * 3 + 1]
                features[f"{hand}_{i}_z"] = values[i * 3 + 2]

        # Predict alphabet
        if detected:

            input_data = pd.DataFrame(
                [[features[col] for col in feature_columns]],
                columns=feature_columns
            )

            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "Show an ISL alphabet sign",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("ISL Alphabet Live Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

print("Live testing completed!")