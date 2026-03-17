import cv2
import mediapipe as mp
import time

from core.landmark_processor import process_landmarks
from core.predictor import predict
from core.smoothing import smooth_prediction
from core.word_builder import update_word

import core.word_builder as word_builder

from gestures.exit_gesture import check_exit

from handlers.gesture_handler import handle_one_hand, handle_two_hands

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

final_pred = ""

cursor_visible = True
last_blink = time.time()

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Cursor blinking
    if time.time() - last_blink > 0.5:
        cursor_visible = not cursor_visible
        last_blink = time.time()

    if results.multi_hand_landmarks:

        num_hands = len(results.multi_hand_landmarks)

        # -------------------------
        # ONE HAND → LETTER + DELETE
        # -------------------------
        if num_hands == 1:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            data = process_landmarks(hand)
            pred = predict(data)
            final_pred = smooth_prediction(pred)

            # Handles letter insertion + delete
            handle_one_hand(hand, frame, final_pred, update_word)

        # -------------------------
        # TWO HANDS → SPACE + EXIT
        # -------------------------
        elif num_hands == 2:

            left_hand = None
            right_hand = None

            for i, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label

                if label == "Left":
                    left_hand = results.multi_hand_landmarks[i]
                else:
                    right_hand = results.multi_hand_landmarks[i]

            if left_hand and right_hand:

                mp_draw.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

                # EXIT
                exit_flag, elapsed = check_exit(left_hand, right_hand, frame)

                if elapsed > 0:
                    cv2.putText(
                        frame,
                        f"EXIT: {elapsed:.1f}s",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                if exit_flag:
                    break

                # SPACE (handled cleanly)
                handle_two_hands(left_hand, right_hand)

    else:
        final_pred = ""  # prevents ghost letters

    # -------------------------
    # UI DISPLAY
    # -------------------------
    word = word_builder.word
    cursor_pos = word_builder.cursor_pos

    if cursor_visible:
        display_word = word[:cursor_pos] + "|" + word[cursor_pos:]
    else:
        display_word = word

    cv2.putText(
        frame,
        f"Letter: {final_pred}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Word: {display_word if display_word else '|'}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 0),
        3
    )

    cv2.imshow("ASL Trainer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()