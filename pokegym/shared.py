# shared.py
from pyboy.utils import WindowEvent
import numpy as np

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

ACTION_FREQ = 24

def addr_to_opcodes_list(addr: int)->list:
    return np.array([addr], dtype=np.uint16).view(np.uint8).tolist()

# Shared function implementations
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
