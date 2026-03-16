import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

pinch_start = None
PINCH_THRESHOLD = 0.05
HOLD_TIME = 2.5


def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def calculate_angle(a, b, c):
    """
    Returns the angle ABC (in degrees)
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0]*bc[0] + ba[1]*bc[1]

    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    angle = math.degrees(math.acos(max(-1,min(1, dot/(mag_ba*mag_bc)))))

    return angle


while True:

    success, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            wrist = lm[0]

            fingers = []

            thumb_angle = calculate_angle(lm[2], lm[3], lm[4])

            if thumb_angle > 160:
                fingers.append(1)
            else:
                fingers.append(0)

            # finger tips
            tips = [8,12,16,20]
            mids = [6,10,14,18]

            for tip, mid in zip(tips, mids):

                tip_dist = distance(lm[tip], wrist)
                mid_dist = distance(lm[mid], wrist)

                if tip_dist > mid_dist:
                    fingers.append(1)
                else:
                    fingers.append(0)

            cv2.putText(frame,
                        f"Fingers: {fingers}",
                        (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

            # ----------------
            # Pinch Detection
            # ----------------

            thumb = lm[4]
            index = lm[20]

            pinch_dist = distance(thumb, index)

            if pinch_dist < PINCH_THRESHOLD:

                if pinch_start is None:
                    pinch_start = time.time()

                elapsed = time.time() - pinch_start

                cv2.putText(frame,
                            f"Pinch exit: {elapsed:.1f}s",
                            (20,90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

                if elapsed > HOLD_TIME:
                    print("Pinch exit")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            else:
                pinch_start = None

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()