import cv2

def draw_text(frame, final_pred, word, cursor_pos, target, result, cursor_visible):

    # Background panel (clean UI)
    cv2.rectangle(frame, (10, 10), (500, 260), (30, 30, 30), -1)

    display_word = (
        word[:cursor_pos] + "|" + word[cursor_pos:]
        if cursor_visible else word
    )

    cv2.putText(frame, f"Letter: {final_pred}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(frame, f"Word: {display_word if display_word else '|'}",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Target: {target}",
                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.putText(frame, f"Result: {result}",
                (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 2)
    

def draw_buttons(frame, hovering):

    def draw_button(rect, text, active):
        x1, y1, x2, y2 = rect

        color = (0, 180, 0) if active else (50, 50, 200)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, text,
                    (x1 + 20, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    SUBMIT_BTN = (950, 80, 1200, 150)
    NEW_BTN = (950, 180, 1200, 250)

    draw_button(SUBMIT_BTN, "SUBMIT", hovering == "submit")
    draw_button(NEW_BTN, "NEW WORD", hovering == "new")