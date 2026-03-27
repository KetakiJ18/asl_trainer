import cv2
import mediapipe as mp
import time

# Core
from core.landmark_processor import process_landmarks
from core.predictor import predict
from core.smoothing import smooth_prediction
from core.word_builder import update_word
import core.word_builder as word_builder

# Modules
from ui.display import draw_text, draw_buttons
from ui.buttons import get_hovered_button
from ui.progress import draw_hover_progress
import game.game_manager as game

# Handlers
from handlers.gesture_handler import handle_one_hand, handle_two_hands
from gestures.exit_gesture import check_exit
from core.word_game import generate_word, check_match

# -------------------------
# INIT
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("ASL Trainer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Trainer", 1280, 720)

game.new_game(generate_word)

# -------------------------
# STATE
# -------------------------
final_pred = ""
hover_start = None
hover_target = None
HOVER_TIME = 1.5

cursor_visible = True
last_blink = time.time()

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
            hovering = get_hovered_button(hand, w, h)

            if hovering:

                if hover_target != hovering:
                    hover_start = time.time()
                    hover_target = hovering

                elapsed = time.time() - hover_start

                # fingertip
                index_tip = hand.landmark[8]
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)

                draw_hover_progress(frame, (cx, cy), elapsed, HOVER_TIME)

                if elapsed > HOVER_TIME:

                    hover_start = None
                    hover_target = None

                    if hovering == "submit":
                        game.submit_word(word_builder.word, check_match)
                        word_builder.word = ""
                        word_builder.cursor_pos = 0

                    elif hovering == "new":
                        game.new_game(generate_word)
                        word_builder.word = ""
                        word_builder.cursor_pos = 0

            else:
                hover_start = None
                hover_target = None

        # -------------------------
        # TWO HANDS (EXIT)
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

                exit_flag, exit_elapsed = check_exit(left_hand, right_hand, frame)

                # draw exit progress properly
                if exit_elapsed > 0:
                    h, w, _ = frame.shape

                    lx = int(left_hand.landmark[8].x * w)
                    ly = int(left_hand.landmark[8].y * h)

                    rx = int(right_hand.landmark[8].x * w)
                    ry = int(right_hand.landmark[8].y * h)

                    cx = (lx + rx) // 2
                    cy = (ly + ry) // 2 - 50

                    draw_hover_progress(frame, (cx, cy), exit_elapsed, 2)

                if exit_flag:
                    break

                handle_two_hands(left_hand, right_hand)

    # -------------------------
    # UI (DRAW LAST ALWAYS)
    # -------------------------
    draw_text(
        frame,
        final_pred,
        word_builder.word,
        word_builder.cursor_pos,
        game.target_word,
        game.game_result,
        cursor_visible
    )

    draw_buttons(frame, hovering)

    cv2.imshow("ASL Trainer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()