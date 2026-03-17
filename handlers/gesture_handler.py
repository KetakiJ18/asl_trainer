import time
import core.word_builder as word_builder
from gestures.delete_gesture import check_delete
from gestures.space_gestures import check_space

space_cooldown = 0
SPACE_DELAY = 1.5

def handle_one_hand(hand, frame, final_pred, update_word):

    word, _ = update_word(final_pred)

    # DELETE
    if check_delete(hand):
        if word_builder.cursor_pos > 0:
            word_builder.word = (
                word_builder.word[:word_builder.cursor_pos - 1] +
                word_builder.word[word_builder.cursor_pos:]
            )
            word_builder.cursor_pos -= 1

    return word


def handle_two_hands(left_hand, right_hand):

    global space_cooldown

    # SPACE
    if check_space(left_hand, right_hand):
        if time.time() - space_cooldown > SPACE_DELAY:

            word_builder.word = (
                word_builder.word[:word_builder.cursor_pos] +
                " " +
                word_builder.word[word_builder.cursor_pos:]
            )

            word_builder.cursor_pos += 1
            space_cooldown = time.time()