import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("models\\asl_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            coords = []

            for point in lm:
                coords.append([point.x, point.y, point.z])

            coords = np.array(coords)

            wrist = coords[0]

            coords = coords - wrist

            max_value = np.max(np.abs(coords))

            if max_value != 0:
                coords = coords / max_value

            data = coords.flatten().reshape(1, -1)

            prediction = model.predict(data)[0]

            cv2.putText(
                frame,
                f"Letter: {prediction}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,255,0),
                3
            )

    cv2.imshow("ASL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()