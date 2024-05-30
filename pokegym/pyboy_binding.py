from pdb import set_trace as T
from io import BytesIO

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pokegym import ram_map
from pokegym.ram_addresses import RamAddress as RAM
import numpy as np

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

def read_m(pyboy, addr: str | int) -> int:
    if isinstance(addr, str):
        return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
    return pyboy.memory[addr]

def read_short(pyboy, addr: str | int) -> int:
    if isinstance(addr, str):
        _, addr = pyboy.symbol_lookup(addr)
    data = pyboy.memory[addr : addr + 2]
    return int(data[0] << 8) + int(data[1])

def read_bit(pyboy, addr: str | int, bit: int) -> bool:
    return bool(int(read_m(pyboy, addr)) & (1 << bit))

def read_event_bits(pyboy):
    _, addr = pyboy.symbol_lookup("wEventFlags")
    return pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

def get_badges(pyboy):
    return read_m(pyboy, "wObtainedBadges").bit_count()

def read_party(pyboy):
    _, addr = pyboy.symbol_lookup("wPartySpecies")
    party_length = pyboy.memory[pyboy.symbol_lookup("wPartyCount")[1]]
    return pyboy.memory[addr : addr + party_length]

def make_env(gb_path, headless=True, quiet=False, **kwargs):
    game = PyBoy(
        gb_path,
        window="null" if headless else "SDL2",
        **kwargs,
    )

    screen = game.screen

    if not headless:
        game.set_emulation_speed(6)

    return game, screen

def open_state_file(path):
    '''Load state file with BytesIO so we can cache it'''
    with open(path, 'rb') as f:
        initial_state = BytesIO(f.read())

    return initial_state

def load_pyboy_state(pyboy, state):
    '''Reset state stream and load it into PyBoy'''
    state.seek(0)
    pyboy.load_state(state)

def hook_register(pyboy, bank, addr, callback, context):
    pyboy.hook_register(bank, addr, callback, context)

def check_if_party_has_cut(pyboy) -> bool:
    def read_m_range(pyboy, start_addr: int, length: int) -> list:
        return [pyboy.memory[start_addr + i] for i in range(length)]
    party_size = read_m(pyboy, "wPartyCount")
    for i in range(party_size):
        _, addr = pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
        moves = read_m_range(pyboy, addr, 4)
        if 15 in moves:
            return True
    return False

def cut_if_next(pyboy):
    # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
    in_erika_gym = pyboy.memory[pyboy.symbol_lookup("wCurMapTileset")[1]] == 7
    in_overworld = pyboy.memory[pyboy.symbol_lookup("wCurMapTileset")[1]] == 0
    if in_erika_gym or in_overworld:
        wTileMap = pyboy.symbol_lookup("wTileMap")[1]
        tileMap = pyboy.memory[wTileMap : wTileMap + 20 * 18]
        tileMap = np.array(tileMap, dtype=np.uint8)
        tileMap = np.reshape(tileMap, (18, 20))
        y, x = 8, 8
        up, down, left, right = (
            tileMap[y - 2 : y, x : x + 2],  # up
            tileMap[y + 2 : y + 4, x : x + 2],  # down
            tileMap[y : y + 2, x - 2 : x],  # left
            tileMap[y : y + 2, x + 2 : x + 4],  # right
        )

        # Gym trees apparently get the same tile map as outside bushes
        # GYM = 7
        if (in_overworld and 0x3D in up) or (in_erika_gym and 0x50 in up):
            pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_UP, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
        elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
            pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
        elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
            pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
        elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
            pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
        else:
            return

        # open start menu
        pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
        pyboy.tick(ACTION_FREQ, render=True)
        # scroll to pokemon
        # 1 is the item index for pokemon
        while pyboy.memory[pyboy.symbol_lookup("wCurrentMenuItem")[1]] != 1:
            pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
        pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
        pyboy.tick(ACTION_FREQ, render=True)

        # find pokemon with cut
        while True:
            pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            pyboy.tick(ACTION_FREQ, render=True)
            party_mon = pyboy.memory[pyboy.symbol_lookup("wCurrentMenuItem")[1]]
            _, addr = pyboy.symbol_lookup(f"wPartyMon{party_mon+1}Moves")
            if 15 in pyboy.memory[addr : addr + 4]:
                break

        # press a bunch of times
        for _ in range(5):
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            pyboy.tick(4 * ACTION_FREQ, render=True)

def run_action_on_emulator(pyboy, action):
    if not read_bit(pyboy, "wd730", 6):
        # if not instant text speed, then set it to instant
        txt_value = read_m(pyboy, "wd730")
        pyboy.memory[pyboy.symbol_lookup("wd730")[1]] = ram_map.set_bit(txt_value, 6)
    # press button then release after some steps
    pyboy.send_input(VALID_ACTIONS[action])
    pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
    pyboy.tick(ACTION_FREQ, render=True)

    # TODO: Add support for video recording
    # if save_video and fast_video:
    #     add_video_frame()
    if check_if_party_has_cut(pyboy):
        cut_if_next(pyboy)
