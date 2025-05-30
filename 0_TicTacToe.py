
from random import randint
game_state = ["_" for i in range(0, 9)]


def print_board():
    print(game_state[0:3])
    print(game_state[3:6])
    print(game_state[6:9])


def update_board(position, symbol):
    # Task: Check if the position is valid (0-8)
    if i not in range(0,9):
        print("error not in range")
        return 1
    # Task: Check if the position is empty
    if game_state[position] != "_":
        print("error position not empty")
        return 1
    # otherwise return with a invalid message to the user
    game_state[position] = symbol
    return 0


def game_finished():
    # Write logic to decide if game is finished
    # and print the result ("win", "loose", "draw")
    winning_positions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # horizontal
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # vertical
        (0, 4, 8), (2, 4, 6)              # diagonal
    ]

    for a, b, c in winning_positions:
        if game_state[a] == game_state[b] == game_state[c] and game_state[a] != "_":
            if game_state[a] == "x":
                print("win")
            else:
                print("loose")
            return True

    if "_" not in game_state:
        print("draw")
        return True
    return False


if __name__ == "__main__":
    print("Welcome to TicTacToe!")
    print("You can put your 'x' at the following positions:")
    print('[0,1,2]\n[3,4,5]\n[6,7,8]')

    print("Current board:")
    print_board()
    while not game_finished():
        i = int(input("Where do you want to put your 'x'? (0-8)"))
        error = update_board(i, "x")
        while (error):
            print_board()
            i = int(input("Where do you want to put your 'x'? (0-8)"))
            error = update_board(i, "x")
        # Task: implement the opponents move
        # opponent does random move
        y = randint(0, 8)
        error = update_board(y, "o")
        while (error):
            y = randint(0, 8)
            error = update_board(y, "o")
        print_board()
