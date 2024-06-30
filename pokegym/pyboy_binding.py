from collections import defaultdict
from io import BytesIO
from pokegym.pyboy_singleton import get_pyboy_instance
from pyboy.utils import WindowEvent
from pokegym import ram_map
import numpy as np
import os
from pokegym.tg_constants import *
from pokegym import ram_map_leanke

from pokegym.constants import *
from pokegym.bin.ram_reader.red_ram_api import Game  # Importing the Game class
from pokegym.bin.ram_reader.red_memory_battle import *
from pokegym.bin.ram_reader.red_memory_env import *
from pokegym.bin.ram_reader.red_memory_items import *
from pokegym.bin.ram_reader.red_memory_map import *
from pokegym.bin.ram_reader.red_memory_menus import *
from pokegym.bin.ram_reader.red_memory_player import *
from pokegym.bin.ram_reader.red_ram_debug import *

import logging
# Configure logging
logging.basicConfig(
    filename="pyboy_logging.log",  # Name of the log file
    filemode="a",  # Append to the file
    format="%(message)s",  # Log format
    level=logging.INFO,  # Log level
)

# Initialize PyBoy
gb_path = 'pokemon_red.gb'  # Replace with the path to your ROM
symbols_path = os.path.join(os.path.dirname(__file__), '..', '..', 'PufferLib', 'pokemon_red.sym')
headless = False

pyboy = get_pyboy_instance(gb_path=gb_path, symbols_path=symbols_path, headless=headless)
game = Game(pyboy)  # Instantiate the Game class with PyBoy instance

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)
VALID_ACTIONS_STR = [1, 2, 3, 4, 5, 6, 7]
ACTION_FREQ = 24

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

def read_m(addr: str | int) -> int:
    if isinstance(addr, str):
        return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
    return pyboy.memory[addr]

def read_short(addr: str | int) -> int:
    if isinstance(addr, str):
        _, addr = pyboy.symbol_lookup(addr)
    data = pyboy.memory[addr : addr + 2]
    return int(data[0] << 8) + int(data[1])

def read_bit(addr: str | int, bit: int) -> bool:
    return bool(int(read_m(addr)) & (1 << bit))

def open_state_file(path):
    '''Load state file with BytesIO so we can cache it'''
    with open(path, 'rb') as f:
        initial_state = BytesIO(f.read())

    return initial_state

def read_event_bits():
    _, addr = pyboy.symbol_lookup("wEventFlags")
    return pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

def get_badges():
    return read_m("wObtainedBadges").bit_count()

def read_party():
    _, addr = pyboy.symbol_lookup("wPartySpecies")
    party_length = pyboy.memory[pyboy.symbol_lookup("wPartyCount")[1]]
    return pyboy.memory[addr : addr + party_length]

def load_pyboy_state(state):
    '''Reset state stream and load it into PyBoy'''
    state.seek(0)
    pyboy.load_state(state)

def hook_register(pyboy, bank, addr, callback, context):
    try:
        pyboy.hook_register(bank, addr, callback, context)
    except ValueError:
        pass
    
def check_if_party_has_hm(hm_id: int) -> bool:
    return game.does_party_have_hm(hm_id)

def teach_hm(tmhm: int, pp: int, pokemon_species_ids):
    logging.info(f'pyboy_step_handler.py: teach_hm() executing')
    party_size = read_m("wPartyCount")
    for i in range(party_size):
        _, species_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
        poke = pyboy.memory[species_addr]
        if poke in pokemon_species_ids:
            for slot in range(4):
                if read_m(f"wPartyMon{i+1}Moves") not in {0xF, 0x13, 0x39, 0x46, 0x94}:
                    _, move_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
                    _, pp_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
                    pyboy.memory[move_addr + slot] = tmhm
                    pyboy.memory[pp_addr + slot] = pp
                    break


def run_action_on_emulator(self, action):
    self.action_hist[action] += 1
    # press button then release after some steps
    # TODO: Add video saving logic

    # print(f'got_hms:\n hm01: {self.events.get_event("EVENT_GOT_HM01")}\n hm03: {self.events.get_event("EVENT_GOT_HM03")}\n hm04: {self.events.get_event("EVENT_GOT_HM04")}\n')
    # print(f'party has hm:\n cut: {self.check_if_party_has_hm(0x0F)}\n surf: {self.check_if_party_has_hm(0x39)}\n strength: {self.check_if_party_has_hm(0x46)}\n')
    # print(f'party badge count: {self.bit_count(self.pyboy.memory[0xD356])}')
    # badge_address = 0xD356
    # badge_memory = self.pyboy.memory[badge_address]

    # for i in range(8):
    #     badge_status = read_bit(badge_address, i)
    #     print(f'Badge {i+1} status: {badge_status}')
        
        
    if not self.disable_ai_actions:
        self.pyboy.send_input(VALID_ACTIONS[action])
        self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
    self.pyboy.tick(self.action_freq, render=True)

    if self.events.get_event("EVENT_GOT_HM01"):  # 0xD803, 0 CUT
        if self.auto_teach_cut and not self.check_if_party_has_hm(0x0F):
            self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
        if self.auto_use_cut:
            # set badge 2 (Misty - CascadeBadge) if not obtained or can't use Cut
            if read_bit(0xD356, 0) == 0:
                self.set_badge(1)
            self.cut_if_next()

    if self.events.get_event("EVENT_GOT_HM03"):  # 0xD857, 0 SURF
        if self.auto_teach_surf and not self.check_if_party_has_hm(0x39):
            self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
        if self.auto_use_surf:
            # set badge 5 (Koga - SoulBadge) if not obtained or can't use Surf
            if read_bit(0xD356, 4) == 0:
                self.set_badge(5)
            self.surf_if_attempt(VALID_ACTIONS[action])

    if self.events.get_event("EVENT_GOT_HM04"):  # 0xD78E, 0 STRENGTH
        if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
            self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
        if self.auto_solve_strength_puzzles:
            # set badge 4 (Erika - RainbowBadge) if not obtained or can't use Strength
            if read_bit(0xD356, 3) == 0:
                self.set_badge(4)
            self.solve_missable_strength_puzzle()
            self.solve_switch_strength_puzzle()

    if self.events.get_event("EVENT_GOT_HM02"): # 0xD7E0, 6 FLY
        # if self.auto_teach_fly and not self.check_if_party_has_hm(0x02):
        #     self.teach_hm(TmHmMoves.FLY.value, 15, FLY_SPECIES_IDS)
            # # set badge 3 (Lt. Surge - ThunderBadge) if not obtained or can't use Fly
            # if read_bit(0xD356, 3) == 0:
            #     self.set_badge(1)
        pass
    
    if 'Poke Flute' in self.api.items.get_bag_item_ids() and self.auto_pokeflute:
        self.use_pokeflute()

    if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0 and self.skip_rocket_hideout_bool:
        self.skip_rocket_hideout()

    if self.skip_silph_co_bool and int(self.read_bit(0xD76C, 0)) != 0:  # has poke flute
        self.skip_silph_co()

    if (
        self.skip_safari_zone_bool
        and self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]] == 7
    ):
        self.skip_safari_zone()


def get_game_coords():
    logging.info(f'pyboy_step_handler.py: get_game_coords() executing')
    return (read_m(0xD362), read_m(0xD361), read_m(0xD35E))

def read_m(addr: str | int) -> int:
    if isinstance(addr, str):
        return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
    return pyboy.memory[addr]


# from collections import defaultdict
# from io import BytesIO
# from pokegym.pyboy_singleton import get_pyboy_instance
# from pyboy.utils import WindowEvent
# from pokegym import ram_map
# import numpy as np
# import os
# from pokegym.tg_constants import *
# from pokegym import ram_map_leanke

# from pokegym.constants import *
# from pokegym.bin.ram_reader.red_ram_api import Game  # Importing the Game class
# from pokegym.bin.ram_reader.red_memory_battle import *
# from pokegym.bin.ram_reader.red_memory_env import *
# from pokegym.bin.ram_reader.red_memory_items import *
# from pokegym.bin.ram_reader.red_memory_map import *
# from pokegym.bin.ram_reader.red_memory_menus import *
# from pokegym.bin.ram_reader.red_memory_player import *
# from pokegym.bin.ram_reader.red_ram_debug import *

# import logging
# # Configure logging
# logging.basicConfig(
#     filename="pyboy_logging.log",  # Name of the log file
#     filemode="a",  # Append to the file
#     format="%(message)s",  # Log format
#     level=logging.INFO,  # Log level
# )

# # Initialize PyBoy
# gb_path = 'pokemon_red.gb'  # Replace with the path to your ROM
# symbols_path = os.path.join(os.path.dirname(__file__), '..', '..', 'PufferLib', 'pokemon_red.sym')

# pyboy = get_pyboy_instance(gb_path, window='null', symbols=symbols_path)
# game = Game(pyboy)  # Instantiate the Game class with PyBoy instance

# EVENT_FLAGS_START = 0xD747
# EVENTS_FLAGS_LENGTH = 320
# MUSEUM_TICKET = (0xD754, 0)
# VALID_ACTIONS_STR = [1, 2, 3, 4, 5, 6, 7]
# ACTION_FREQ = 24

# VALID_ACTIONS = [
#     WindowEvent.PRESS_ARROW_DOWN,
#     WindowEvent.PRESS_ARROW_LEFT,
#     WindowEvent.PRESS_ARROW_RIGHT,
#     WindowEvent.PRESS_ARROW_UP,
#     WindowEvent.PRESS_BUTTON_A,
#     WindowEvent.PRESS_BUTTON_B,
#     WindowEvent.PRESS_BUTTON_START,
# ]

# VALID_RELEASE_ACTIONS = [
#     WindowEvent.RELEASE_ARROW_DOWN,
#     WindowEvent.RELEASE_ARROW_LEFT,
#     WindowEvent.RELEASE_ARROW_RIGHT,
#     WindowEvent.RELEASE_ARROW_UP,
#     WindowEvent.RELEASE_BUTTON_A,
#     WindowEvent.RELEASE_BUTTON_B,
#     WindowEvent.RELEASE_BUTTON_START,
# ]

# VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

# def read_m(addr: str | int) -> int:
#     if isinstance(addr, str):
#         return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
#     return pyboy.memory[addr]

# def read_short(addr: str | int) -> int:
#     if isinstance(addr, str):
#         _, addr = pyboy.symbol_lookup(addr)
#     data = pyboy.memory[addr : addr + 2]
#     return int(data[0] << 8) + int(data[1])

# def read_bit(addr: str | int, bit: int) -> bool:
#     return bool(int(read_m(addr)) & (1 << bit))

# def open_state_file(path):
#     '''Load state file with BytesIO so we can cache it'''
#     with open(path, 'rb') as f:
#         initial_state = BytesIO(f.read())

#     return initial_state

# def read_event_bits():
#     _, addr = pyboy.symbol_lookup("wEventFlags")
#     return pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

# def get_badges():
#     return read_m("wObtainedBadges").bit_count()

# def read_party():
#     _, addr = pyboy.symbol_lookup("wPartySpecies")
#     party_length = pyboy.memory[pyboy.symbol_lookup("wPartyCount")[1]]
#     return pyboy.memory[addr : addr + party_length]

# def make_env(gb_path, headless=False, quiet=False, **kwargs):
#     base_dir = os.path.dirname(__file__)
#     symbols_path = os.path.join(base_dir, '..', '..', 'PufferLib', 'pokemon_red.sym')
    
#     # Valid keyword arguments for PyBoy
#     pyboy_kwargs = {
#         "window": "null" if headless else "SDL2",
#         "symbols": symbols_path,
#     }

#     game = PyBoy(
#         gb_path,
#         **pyboy_kwargs,
#     )

#     screen = game.screen

#     if not headless:
#         game.set_emulation_speed(0) # 6

#     return game, screen

# def load_pyboy_state(state):
#     '''Reset state stream and load it into PyBoy'''
#     state.seek(0)
#     pyboy.load_state(state)

# def hook_register(pyboy, bank, addr, callback, context):
#     try:
#         pyboy.hook_register(bank, addr, callback, context)
#     except ValueError:
#         pass
    
# def check_if_party_has_hm(hm_id: int) -> bool:
#     return game.does_party_have_hm(hm_id)

# def teach_hm(tmhm: int, pp: int, pokemon_species_ids):
#     logging.info(f'pyboy_step_handler.py: teach_hm() executing')
#     party_size = read_m("wPartyCount")
#     for i in range(party_size):
#         _, species_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
#         poke = pyboy.memory[species_addr]
#         if poke in pokemon_species_ids:
#             for slot in range(4):
#                 if read_m(f"wPartyMon{i+1}Moves") not in {0xF, 0x13, 0x39, 0x46, 0x94}:
#                     _, move_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
#                     _, pp_addr = pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
#                     pyboy.memory[move_addr + slot] = tmhm
#                     pyboy.memory[pp_addr + slot] = pp
#                     break

# def run_action_on_emulator(self, action):
#     self.action_hist[action] += 1
#     if not self.disable_ai_actions:
#         self.pyboy.send_input(VALID_ACTIONS[action])
#         self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
#     self.pyboy.tick(self.action_freq, render=True)

#     if self.read_bit(0xD803, 0):
#         if self.auto_teach_cut and not self.check_if_party_has_hm(0x0F):
#             self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
#         if self.auto_use_cut:
#             self.cut_if_next()

#     if self.read_bit(0xD78E, 0):
#         if self.auto_teach_surf and not self.check_if_party_has_hm(0x39):
#             self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
#         if self.auto_use_surf:
#             self.surf_if_attempt(VALID_ACTIONS[action])

#     if self.read_bit(0xD857, 0):
#         if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
#             self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
#         if self.auto_solve_strength_puzzles:
#             self.solve_missable_strength_puzzle()
#             self.solve_switch_strength_puzzle()

#     if 'Poke Flute' in self.api.items.get_bag_item_ids() and self.auto_pokeflute:
#         self.use_pokeflute()

#     if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0 and self.skip_rocket_hideout_bool:
#         self.skip_rocket_hideout()

#     if self.skip_silph_co_bool and int(self.read_bit(0xD76C, 0)) != 0:  # has poke flute
#         self.skip_silph_co()
    
#     if self.skip_safari_zone_bool:
#         self.skip_safari_zone()

# def get_game_coords():
#     logging.info(f'pyboy_step_handler.py: get_game_coords() executing')
#     return (read_m(0xD362), read_m(0xD361), read_m(0xD35E))

# def read_m(addr: str | int) -> int:
#     if isinstance(addr, str):
#         return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
#     return pyboy.memory[addr]

# def read_bit(addr: str | int, bit: int) -> bool:
#     return bool(int(read_m(addr)) & (1 << bit))
