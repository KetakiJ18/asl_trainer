prev_x = None

def check_delete(hand):

    global prev_x

    x = hand.landmark[0].x  # wrist

    if prev_x is None:
        prev_x = x
        return False

    movement = prev_x - x  # left movement

    prev_x = x

    if movement > 0.15:
        return True

    return False