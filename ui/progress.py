import cv2
import math

def draw_hover_progress(frame, center, elapsed, total_time):

    if elapsed <= 0:
        return

    progress = min(elapsed / total_time, 1.0)
    angle = int(progress * 360)

    x, y = center

    cv2.circle(frame, (x, y), 25, (255, 255, 255), 2)

    cv2.ellipse(
        frame,
        (x, y),
        (25, 25),
        -90,              # start from top
        0,
        angle,
        (0, 255, 0),
        4
    )