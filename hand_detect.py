import cv2
import mediapipe as mp
import time

prev_gesture = None
gesture_start = 0
confirmed_action = ""

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()
else:
    print("✅ Webcam opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[hand_index].classification[0].label

            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) != 21:
                continue

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = []

            # Thumb detection
            if hand_label == "Right":
                fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
            else:
                fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)

            # Other fingers
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            for tip, pip in zip(tips, pips):
                fingers.append(1 if lm_list[tip][1] < lm_list[pip][1] else 0)

            total_fingers = fingers.count(1)

            # -------- Gesture Mapping --------
            action = ""

            if total_fingers == 1:
                action = "STOP"
            elif total_fingers == 2:
                action = "START"
            elif total_fingers == 3:
                action = "NEXT"
            elif total_fingers == 4:
                action = "PREVIOUS"
            elif total_fingers == 5:
                action = "WAIT"
            elif total_fingers == 0:
                action = "HELLO"

            # -------- Gesture Stability Logic --------
            if total_fingers != prev_gesture:
                prev_gesture = total_fingers
                gesture_start = time.time()

            gesture_duration = time.time() - gesture_start

            if gesture_duration > 0.7:
                confirmed_action = action

            # Wrist position
            cx, cy = lm_list[0]

            # Display info
            cv2.putText(frame, f'Fingers: {total_fingers}', (cx, cy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f'Action: {action}', (cx, cy + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.putText(frame, f'Triggered: {confirmed_action}', (cx, cy + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Debug finger states
            cv2.putText(frame, str(fingers), (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Hand Action Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()