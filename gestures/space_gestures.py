def is_hand_open(hand):
    return hand.landmark[8].y < hand.landmark[6].y

def check_space(left_hand, right_hand):

    if is_hand_open(left_hand) and is_hand_open(right_hand):
        return True

    return False