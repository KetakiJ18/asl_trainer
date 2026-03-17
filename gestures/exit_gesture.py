import time
import numpy as np

exit_start = None
EXIT_HOLD_TIME = 1.2
EXIT_DISTANCE = 40

def check_exit(left_hand, right_hand, frame):

    global exit_start

    h, w, _ = frame.shape

    x1 = int(left_hand.landmark[8].x * w)
    y1 = int(left_hand.landmark[8].y * h)

    x2 = int(right_hand.landmark[8].x * w)
    y2 = int(right_hand.landmark[8].y * h)

    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if dist < EXIT_DISTANCE:
        if exit_start is None:
            exit_start = time.time()

        elapsed = time.time() - exit_start

        if elapsed > EXIT_HOLD_TIME:
            return True, elapsed

        return False, elapsed

    else:
        exit_start = None
        return False, 0