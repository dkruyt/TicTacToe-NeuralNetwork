import os

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

# Function to print the current state of the board
def print_board(board):
    symbols = {1: Fore.RED + 'X', -1: Fore.GREEN + 'O', 0: Style.RESET_ALL +' '}
    for i in range(3):
        print('\033[39m|'.join(symbols[board[i*3 + j]] for j in range(3)))
        if i < 2:
            print(Style.RESET_ALL + '-----')
    print(Style.RESET_ALL)
    print()