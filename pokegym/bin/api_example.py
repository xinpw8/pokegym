import sys
import os
from os.path import exists
import sys
from pyboy import PyBoy, WindowEvent

from ram_reader.red_ram_api import *
from ram_reader.red_ram_debug import *
import os

pyboy = PyBoy('../PokemonRed.gb')
pyboy.set_emulation_speed(5)  # Configurable emulation speed

def clear_screen():
        if os.name == 'nt':
                os.system('cls')
        else:
                os.system('clear')


def load_a_save_point():
        save_file = None
        save_file = "checkpoints_battles/charmander/pokemon_ai_58"
        if pyboy and save_file:
                with open(save_file, "rb") as f:
                        pyboy.load_state(f)


def check_to_save_game():
        if os.path.exists("save"):
                # Save to file
                file = "pokemon_ai"
                print(file)
                file_like_object = open(file, "wb")
                pyboy.save_state(file_like_object)
                os.remove("save")


load_a_save_point()

game = Game(pyboy)  # To use the Pokemon API, first instantiate a Game object passing in the PyBoy instance
save_itr, count, frame = 0, 0 ,0
while not pyboy.tick():
        frame += 1

        if frame < 24:
                continue
        frame = 0

        check_to_save_game()

        game.process_game_states()  # To use the Pokemon API, call the process_game_states() method once per tick cycle. Then you can access it's API methods.
        debug_str = get_debug_str(game) # Full of examples on how to call Pokemon Red API methods, get's debug or obs info from the game

        clear_screen()
        sys.stdout.write(f'\r{debug_str}')
        sys.stdout.flush()

pyboy.stop()
