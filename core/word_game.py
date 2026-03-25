import random

WORDS = [
    "APPLE", "GRAPE", "MANGO", "BREAD", "CHAIR",
    "PLANT", "SNAKE", "TRAIN", "LIGHT", "SMILE"
]

current_target = random.choice(WORDS)


def generate_word():
    global current_target
    current_target = random.choice(WORDS)
    return current_target


def get_target():
    return current_target


def check_match(user_word):
    return user_word.strip() == current_target