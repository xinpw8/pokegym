from pdb import set_trace as T
from io import BytesIO

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pokegym import ram_map
from pokegym.ram_addresses import RamAddress as RAM

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

# def run_action_on_emulator(pyboy, screen, action, headless=True, fast_video=True, frame_skip=24):
#     '''Sends actions to PyBoy'''
#     if not read_bit(pyboy, "wd730", 6):
#         # if not instant text speed, then set it to instant
#         txt_value = read_m(pyboy, "wd730")
#         pyboy.memory[pyboy.symbol_lookup("wd730")[1]] = ram_map.set_bit(txt_value, 6)

#     press, release = action.PRESS, action.RELEASE
#     pyboy.send_input(press)

#     # if headless or fast_video:
#     #     pyboy._rendering(False)

#     frames = []
#     for i in range(frame_skip):
#         if i == 8:  # Release button after 8 frames
#             pyboy.send_input(release)
#         if not fast_video:  # Save every frame
#             frames.append(screen.screen_ndarray())
#         pyboy.tick(1, True)

#     if fast_video:  # Save only the last frame
#         frames.append(screen.screen_ndarray())

#     return frames
