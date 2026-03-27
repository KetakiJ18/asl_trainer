import time

current_letter = None
letter_start = None

LETTER_HOLD_TIME = 3

word = ""
cursor_pos = 0

def update_word(pred):

    global current_letter, letter_start, word, cursor_pos

    if pred == "":
        return word, None

    if pred != current_letter:
        current_letter = pred
        letter_start = time.time()
        return word, None

    else:
        elapsed = time.time() - letter_start

        if elapsed > LETTER_HOLD_TIME:
            word = word[:cursor_pos] + current_letter + " "+word[cursor_pos:]
            cursor_pos += 1
            letter_start = time.time()
            return word, current_letter

    return word, None