import sys
import os
from os.path import exists
import sys
from pyboy import PyBoy, WindowEvent

from ram_reader.red_ram_api import *
from ram_reader.red_ram_debug import *
import os

pyboy = PyBoy('../PokemonRed.gb')
game = Game(pyboy)
frame = 0

pyboy.set_emulation_speed(5)  # Configurable emulation speed


cur_x = pyboy.get_memory_value(PLAYER_LOCATION_X)
cur_y = pyboy.get_memory_value(PLAYER_LOCATION_Y)
map = pyboy.get_memory_value(PLAYER_MAP)

wNorthConnectionStripSrc = pyboy.get_memory_value(0xD372)
wNorthConnectionStripDest = pyboy.get_memory_value(0xD374)
wNorthConnectionStripLength = pyboy.get_memory_value(0xD376)
wNorthConnectedMapWidth = pyboy.get_memory_value(0xD377)
wNorthConnectedMapYAlignment = pyboy.get_memory_value(0xD378)
wNorthConnectedMapXAlignment = pyboy.get_memory_value(0xD379)

wSouthhConnectionStripSrc = pyboy.get_memory_value(0xD37D)
wSouthConnectionStripDest = pyboy.get_memory_value(0xD37F)
wSouthConnectionStripLength = pyboy.get_memory_value(0xD381)
wSouthConnectedMapWidth = pyboy.get_memory_value(0xD382)
wSouthConnectedMapYAlignment = pyboy.get_memory_value(0xD383)
wSouthConnectedMapXAlignment = pyboy.get_memory_value(0xD384)

str0 = f'cur_x: {cur_x}, cur_y: {cur_y}, map: {map}'
str1 = (f' wNorthConnectionStripSrc: {wNorthConnectionStripSrc},'
        f' wNorthConnectionStripDest: {wNorthConnectionStripDest},'
        f' wNorthConnectionStripLength: {wNorthConnectionStripLength},'
        f' wNorthConnectedMapWidth: {wNorthConnectedMapWidth},'
        f' wNorthConnectedMapYAlignment: {wNorthConnectedMapYAlignment},'
        f' wNorthConnectedMapXAlignment: {wNorthConnectedMapXAlignment}')
str2 = (f' wSouthhConnectionStripSrc: {wSouthhConnectionStripSrc},'
        f' wSouthConnectionStripDest: {wSouthConnectionStripDest},'
        f' wSouthConnectionStripLength: {wSouthConnectionStripLength},'
        f' wSouthConnectedMapWidth: {wSouthConnectedMapWidth},'
        f' wSouthConnectedMapYAlignment: {wSouthConnectedMapYAlignment},'
        f' wSouthConnectedMapXAlignment: {wSouthConnectedMapXAlignment}')
#print(str0)
##print(str1)
#print(str2)
def clear_screen():
        if os.name == 'nt':
                os.system('cls')
        else:
                os.system('clear')
save_file = None
save_file = "checkpoints_battles/charmander/pokemon_ai_58"
if pyboy and save_file:
        with open(save_file, "rb") as f:
                pyboy.load_state(f)

save_itr = 0
count = 0
while not pyboy.tick():
        frame += 1

        if frame < 24:
                continue
        frame = 0

        print()
        print(f' y_pos: {game.ram_interface.read_memory(PLAYER_LOCATION_Y)}, x_pos: {game.ram_interface.read_memory(PLAYER_LOCATION_X)}')


        if os.path.exists("save"):
                # Save to file
                file = "pokemon_ai_" + str(save_itr)
                print(file)
                file_like_object = open(file, "wb")
                pyboy.save_state(file_like_object)
                save_itr += 1
                os.remove("save")

        game.process_game_states()

        #print()
        #print(game.map.get_centered_7x7_tiles())

        #clear_screen()
        sys.stdout.write(f'\r{get_debug_str(game)}')
        #sys.stdout.flush()



pyboy.stop()
