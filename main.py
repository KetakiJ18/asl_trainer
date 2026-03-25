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
from core.word_game import generate_word, check_match

# -------------------------
# BUTTONS
# -------------------------
SUBMIT_BTN = (450, 50, 620, 120)
NEW_BTN = (450, 140, 620, 210)

hover_start = None
hover_target = None
HOVER_TIME = 3

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

# -------------------------
# STATE
# -------------------------
final_pred = ""
target_word = generate_word()
game_result = ""

cursor_visible = True
last_blink = time.time()

# -------------------------
# HOVER DETECTION
# -------------------------
def get_hovered_button(hand, w, h):
    index_tip = hand.landmark[8]
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)

    if SUBMIT_BTN[0] < x < SUBMIT_BTN[2] and SUBMIT_BTN[1] < y < SUBMIT_BTN[3]:
        return "submit", x, y
    elif NEW_BTN[0] < x < NEW_BTN[2] and NEW_BTN[1] < y < NEW_BTN[3]:
        return "new", x, y
    return None, x, y


# =========================
# MAIN LOOP
# =========================
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Cursor blink
    if time.time() - last_blink > 0.5:
        cursor_visible = not cursor_visible
        last_blink = time.time()

    hovering = None

    if results.multi_hand_landmarks:

        num_hands = len(results.multi_hand_landmarks)

        # -------------------------
        # ONE HAND
        # -------------------------
        if num_hands == 1:

            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            data = process_landmarks(hand)
            pred = predict(data)
            final_pred = smooth_prediction(pred)

            handle_one_hand(hand, frame, final_pred, update_word)

            h, w, _ = frame.shape
            hovering, cx, cy = get_hovered_button(hand, w, h)

            if hovering:

                if hover_target != hovering:
                    hover_start = time.time()
                    hover_target = hovering

                elapsed = time.time() - hover_start

                cv2.putText(
                    frame,
                    f"{hovering.upper()}: {elapsed:.1f}s",
                    (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                if elapsed > HOVER_TIME:

                    hover_start = None
                    hover_target = None

                    try:
                        if hovering == "submit":
                            user_word = word_builder.word

                            # FIXED crash here
                            if check_match(user_word):  
                                game_result = "✅ Correct!"
                            else:
                                game_result = "❌ Try Again"

                            word_builder.word = ""
                            word_builder.cursor_pos = 0

                        elif hovering == "new":
                            target_word = generate_word()
                            game_result = ""
                            word_builder.word = ""
                            word_builder.cursor_pos = 0

                    except Exception as e:
                        print("ERROR:", e)
                        game_result = "⚠ Error"

            else:
                hover_start = None
                hover_target = None

        # -------------------------
        # TWO HANDS
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

                exit_flag, elapsed = check_exit(left_hand, right_hand, frame)

                if elapsed > 0:
                    cv2.putText(
                        frame,
                        f"EXIT: {elapsed:.1f}s",
                        (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                if exit_flag:
                    break

                handle_two_hands(left_hand, right_hand)

    # -------------------------
    # UI
    # -------------------------
    word = word_builder.word
    cursor_pos = word_builder.cursor_pos

    display_word = (
        word[:cursor_pos] + "|" + word[cursor_pos:]
        if cursor_visible else word
    )

    cv2.putText(frame, f"Letter: {final_pred}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, f"Word: {display_word if display_word else '|'}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.putText(frame, f"Target: {target_word}",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame, f"Result: {game_result}",
                (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # -------------------------
    # DRAW BUTTONS
    # -------------------------
    def draw_button(frame, rect, text, active):
        color = (0, 200, 0) if active else (255, 0, 0)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(frame, text,
                    (rect[0] + 20, rect[1] + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    draw_button(frame, SUBMIT_BTN, "SUBMIT", hovering == "submit")
    draw_button(frame, NEW_BTN, "NEW WORD", hovering == "new")

    cv2.imshow("ASL Trainer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()