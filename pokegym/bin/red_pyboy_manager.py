import sys
import numpy as np

from pyboy import PyBoy, WindowEvent
from red_env_constants import *


def pyboy_init_actions(extra_buttons):
    valid_actions = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
    ]

    if extra_buttons:
        valid_actions.extend([
            WindowEvent.PRESS_BUTTON_START,
            #WindowEvent.PASS
        ])

    return valid_actions


def pyboy_term_actions(action):
    match action:
        case WindowEvent.PRESS_ARROW_DOWN:
            return WindowEvent.RELEASE_ARROW_DOWN
        case WindowEvent.PRESS_ARROW_UP:
            return WindowEvent.RELEASE_ARROW_UP
        case WindowEvent.PRESS_ARROW_LEFT:
            return WindowEvent.RELEASE_ARROW_LEFT
        case WindowEvent.PRESS_ARROW_RIGHT:
            return WindowEvent.RELEASE_ARROW_RIGHT
        case WindowEvent.PRESS_BUTTON_A:
            return WindowEvent.RELEASE_BUTTON_A
        case WindowEvent.PRESS_BUTTON_B:
            return WindowEvent.RELEASE_BUTTON_B
        case WindowEvent.PRESS_BUTTON_START:
            return WindowEvent.RELEASE_BUTTON_START
        case _:
            return WindowEvent.PASS


class PyBoyManager:
    def __init__(self, env):
        if env.debug:
            print('**** PyBoyManager ****')

        self.env = env
        self.pyboy = None
        self.valid_actions = pyboy_init_actions(self.env.extra_buttons)
        self.action = None
        self.action_history = np.zeros((OBSERVATION_MEMORY_SIZE,), dtype=np.uint8)
        self.move_accepted = True
        
        self.setup_pyboy()

    def setup_pyboy(self):
        window_type = 'dummy' if self.env.headless else 'SDL2'
        self.pyboy = PyBoy(
            self.env.rom_location,
            debugging=False,
            disable_input=False,
            window_type=window_type,
            hide_window='--quiet' in sys.argv,
        )

        if not self.env.headless:
            self.pyboy.set_emulation_speed(PYBOY_RUN_SPEED)  # Configurable emulation speed

        self.reload_game()

    def reload_game(self):
        self._load_save_file(self.env.init_state)

    def _load_save_file(self, save_file):
        if self.pyboy and save_file:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)

    def get_memory_value(self, addr):
        return self.pyboy.get_memory_value(addr)

    def _read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.get_memory_value(addr))[-bit - 1] == '1'
    
    def _update_action_obs(self, input):
        self.action_history = np.roll(self.action_history, 1)
        self.action_history[0] = input

    def a_button_selected(self):
        if self.action == WindowEvent.PRESS_BUTTON_A:
            return True
        
        return False

    def run_dpad_cmd(self, action, termination_action):
        if not self.env.save_video and self.env.headless:
            self.pyboy._rendering(False)

        # press button then release after some steps
        self.pyboy.send_input(action)

        frames = 23

        # AI sent a dpad cmd during a chat interaction, which is allowed but just unproductive. Don't burn more
        # resources than needed to run the cmd.
        # TODO: Magic num fix when adding RAM constants commit
        # if self.get_memory_value(0x8800) != 0:
        #    frames = 23  # TODO: Need button_cmd handling before lowering this

        # Frames for animation vary, xy move ~22, wall collision ~13 & zone reload ~66. Wasted frames are wasted
        # training cycles, frames/tick is expensive. Also, try to prefect OBS output image with completed frame cycle.
        count, animation_started = 0, False
        for i in range(frames):
            count += 1
            self.pyboy.tick()

            # TODO: Magic num fix when adding RAM constants commit
            moving_animation = self.get_memory_value(0xC108) != 0 or self.get_memory_value(0xC107) != 0

            if animation_started and moving_animation == 0:
                continue

            # Release the key once the animation starts, thus it should only be possible to advance 1 position.
            if moving_animation > 0:
                animation_started = True
                self.pyboy.send_input(termination_action)

        #if self.env.debug:
        #    print(f'dpad wait frames: {count}')

        # We never saw movement anim so we never sent term, send it now
        if not animation_started:
            self.pyboy.send_input(termination_action)

        self.pyboy._rendering(True)
        self.pyboy.tick()

        #if not (termination_action == WindowEvent.RELEASE_BUTTON_B or termination_action == WindowEvent.RELEASE_BUTTON_A):
        #    self.env.support.save_screenshot()

    def run_action_on_emulator(self, input):
        self.action = self.valid_actions[input]
        termination_action = pyboy_term_actions(self.action)

        # TODO: This was a bug to start with using action WindowsEvent Enum over input const int. The transformation of 0-6 action to 1-7 in
        # a jumbled order though causes a 2x exploration increase. It'd be good to figure out another way to introduce the noise, but for now leaving this as is.
        self._update_action_obs(input)

        if not self.env.game.allow_menu_selection(self.action):
            self.move_accepted = False
            return

        if self.env.debug:
            print(f'\n\naction: {WindowEvent(self.action).__str__()}')
            print(self.action_history)

        # Pass counts as running a cmd for terms of assigning reward
        if termination_action == WindowEvent.PASS:
            print(f'ignoring command')
            return True
        
        #for i in range(24):
        #    self.pyboy.tick()

        self.run_dpad_cmd(self.action, termination_action)
        self.move_accepted = True
