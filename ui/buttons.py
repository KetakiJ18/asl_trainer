def get_hovered_button(hand, w, h):
    index_tip = hand.landmark[8]

    x = int(index_tip.x * w)
    y = int(index_tip.y * h)

    if 950 < x < 1200 and 80 < y < 150:
        return "submit"
    elif 950 < x < 1200 and 180 < y < 250:
        return "new"

    return None