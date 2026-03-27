target_word = ""
game_result = ""

def new_game(generate_word):
    global target_word, game_result
    target_word = generate_word()
    game_result = ""

def submit_word(user_word, check_match):
    global game_result

    try:
        if check_match(user_word):
            game_result = "Correct!"
        else:
            game_result = "Try Again"
    except:
        game_result = "Error"