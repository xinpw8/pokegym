from pathlib import Path

import sys
current_dir = Path(__file__).parent
pufferlib_dir = current_dir.parent.parent
if str(pufferlib_dir) not in sys.path:
    sys.path.append(str(pufferlib_dir))

from pdb import set_trace as T
import types
import uuid
from gymnasium import Env, spaces
import numpy as np
from skimage.transform import resize

from collections import defaultdict, deque
import io, os
import random
from pyboy.utils import WindowEvent

import matplotlib.pyplot as plt
from pathlib import Path
import mediapy as media

from pokegym.pyboy_binding import (
    VALID_ACTIONS,
    VALID_RELEASE_ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
    hook_register,
)
from . import ram_map, game_map, data, ram_map_leanke
import subprocess
import multiprocessing
import time
import random

from pokegym.constants import *
from pokegym.bin.ram_reader.red_ram_api import *
from pokegym.bin.ram_reader.red_memory_battle import *
from pokegym.bin.ram_reader.red_memory_env import *
from pokegym.bin.ram_reader.red_memory_items import *
from pokegym.bin.ram_reader.red_memory_map import *
from pokegym.bin.ram_reader.red_memory_menus import *
from pokegym.bin.ram_reader.red_memory_player import *
from pokegym.bin.ram_reader.red_ram_debug import *
from enum import IntEnum
from multiprocessing import Manager
from .ram_addresses import RamAddress as RAM
from stream_agent_wrapper import StreamWrapper
import sys
current_dir = Path(__file__).parent
pufferlib_dir = current_dir.parent.parent
if str(pufferlib_dir) not in sys.path:
    sys.path.append(str(pufferlib_dir))
import dill as pickle
import datetime
import dill
import re
from datetime import datetime
import json


CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
STATE_PATH = __file__.rstrip("environment.py") + "current_state/"
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
def get_random_state():
    state_files = [f for f in os.listdir(STATE_PATH) if f.endswith(".state")]
    if not state_files:
        raise FileNotFoundError("No State files found in the specified directory.")
    return random.choice(state_files)
state_file = get_random_state()
randstate = os.path.join(STATE_PATH, state_file)

# List of tree positions in pixel coordinates (x, y)
TREE_POSITIONS_PIXELS = [
    (3184, 3584, 6), # celadon gym 4
    (3375, 3391, 6), # celadon right
    (2528, 3616, 134), # gym 4 middle
    (2480, 3568, 134), # gym 4 left
    (2560, 3584, 134), # gym 4 right
    (1104, 2944, 13), # below pewter 1
    (1264, 3136, 13), # below pewter 2
    (1216, 3616, 13), # below pewter 3
    (1216, 3744, 13), # below pewter 4
    (1216, 3872, 13), # below pewter 5
    (1088, 4000, 1), # old man viridian city
    (992, 4288, 1),  # viridian city left
    (3984, 4512, 5), # (249, 282) to vermilion city gym
    (4640, 1392, 36), # near bill's house 
    (4464, 2176, 20), # cerulean to rock tunnel
    (5488, 2336, 21), # outside rock tunnel 1
    (5488, 2368, 21), # outside rock tunnel 2
    (5488, 2400, 21), # outside rock tunnel 3
    (5488, 2432, 21)  # outside rock tunnel 4
]
# Convert pixel coordinates to grid coordinates and then to global coordinates
TREE_POSITIONS_GRID_GLOBAL = [
    (y//16, x//16) for x, y, map_n in TREE_POSITIONS_PIXELS
]

PLAY_STATE_PATH = __file__.rstrip("environment.py") + "just_died_mt_moon.state" # "outside_mt_moon_hp.state" # "Bulbasaur.state" # "celadon_city_cut_test.state"
EXPERIMENTAL_PATH = __file__.rstrip("environment.py") + "celadon_city_cut_test.state"
STATE_PATH = __file__.rstrip("environment.py") + "current_state/"
# print(f'TREE_POSOTIONS_CONVERTED = {TREE_POSITIONS_GRID_GLOBAL}')
MAPS_WITH_TREES = set(map_n for _, _, map_n in TREE_POSITIONS_PIXELS)
TREE_COUNT_PER_MAP = {6: 2, 134: 3, 13: 5, 1: 2, 5: 1, 36: 1, 20: 1, 21: 4}

# Testing environment w/ no AI
import sdl2 
import sdl2.ext

def map_sdl2_key_to_pyboy(key):
    """Maps SDL2 keys to PyBoy WindowEvent."""
    mapping = {
        sdl2.SDLK_DOWN: WindowEvent.PRESS_ARROW_DOWN,
        sdl2.SDLK_UP: WindowEvent.PRESS_ARROW_UP,
        sdl2.SDLK_LEFT: WindowEvent.PRESS_ARROW_LEFT,
        sdl2.SDLK_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
        sdl2.SDLK_z: WindowEvent.PRESS_BUTTON_A,  # Assuming 'Z' is 'A' button
        sdl2.SDLK_x: WindowEvent.PRESS_BUTTON_B,  # Assuming 'X' is 'B' button
        sdl2.SDLK_RETURN: WindowEvent.PRESS_BUTTON_START,
        sdl2.SDLK_SPACE: WindowEvent.PRESS_BUTTON_SELECT
    }
    return mapping.get(key)

def map_sdl2_key_release_to_pyboy(key):
    """Maps SDL2 key releases to PyBoy WindowEvent."""
    mapping = {
        sdl2.SDLK_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
        sdl2.SDLK_UP: WindowEvent.RELEASE_ARROW_UP,
        sdl2.SDLK_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
        sdl2.SDLK_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        sdl2.SDLK_z: WindowEvent.RELEASE_BUTTON_A,  # Assuming 'Z' is 'A' button
        sdl2.SDLK_x: WindowEvent.RELEASE_BUTTON_B,  # Assuming 'X' is 'B' button
        sdl2.SDLK_RETURN: WindowEvent.RELEASE_BUTTON_START,
        sdl2.SDLK_SPACE: WindowEvent.RELEASE_BUTTON_SELECT
    }
    return mapping.get(key)

def play():
    env = Environment(
        rom_path="pokemon_red.gb",
        state_path=EXPERIMENTAL_PATH,
        headless=False,
        sound=False,
        sound_emulated=False,
        verbose=True,
    )

    env = StreamWrapper(env, stream_metadata={"user": "localtesty |BET|\n"})
    env.reset()
    env.game.set_emulation_speed(0)
    sdl2.ext.init()

    # Create a mapping from WindowEvent to action index
    window_event_to_action = {
        WindowEvent.PRESS_ARROW_DOWN: 0,
        WindowEvent.PRESS_ARROW_LEFT: 1,
        WindowEvent.PRESS_ARROW_RIGHT: 2,
        WindowEvent.PRESS_ARROW_UP: 3,
        WindowEvent.PRESS_BUTTON_A: 4,
        WindowEvent.PRESS_BUTTON_B: 5,
        WindowEvent.PRESS_BUTTON_START: 6,
        WindowEvent.PRESS_BUTTON_SELECT: 7,
        # Add more mappings if necessary
    }

    # Main emulation loop
    running = True
    while running:
        env.game.tick()  # Advance the emulator state
        env.render()  # Render the game state

        events = sdl2.ext.get_events()
        if not events:
            continue

        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
            elif event.type == sdl2.SDL_KEYDOWN:
                pyboy_event = map_sdl2_key_to_pyboy(event.key.keysym.sym)
                if pyboy_event:
                    env.game.send_input(pyboy_event)  # Send mapped input to PyBoy
                    action_index = window_event_to_action.get(pyboy_event)
                    if action_index is not None:
                        observation, reward, done, _, info = env.step(action_index)
                        print(f"new Reward: {reward}\n")  # Print the reward each step
                        if done:
                            print(f"Game over: {done}")
                            running = False
                            break
            elif event.type == sdl2.SDL_KEYUP:
                pyboy_event = map_sdl2_key_release_to_pyboy(event.key.keysym.sym)
                if pyboy_event:
                    env.game.send_input(pyboy_event)  # Send mapped input to PyBoy
                    action_index = window_event_to_action.get(pyboy_event)
                    if action_index is not None:
                        observation, reward, done, _, info = env.step(action_index)
                        print(f"new Reward: {reward}\n")  # Print the reward each step
                        if done:
                            print(f"Game over: {done}")
                            running = False
                            break

    sdl2.ext.quit()
    env.close()  # Close the environment                
    
# Initialize a Manager for shared dict for completion % tracking across all envs
manager = Manager()
# shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data
shared_data = manager.dict()

class Base:
    # Shared counter among processes
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    
    # Initialize a shared integer with a lock for atomic updates
    shared_length = multiprocessing.Value('i', 0)  # 'i' for integer
    lock = multiprocessing.Lock()  # Lock to synchronize access


    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None, # None
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        # Increment counter atomically to get unique sequential identifier
        with Base.counter_lock:
            env_id = Base.counter.value
            Base.counter.value += 1
            
        print(f'env_id {env_id} created.')
            
        self.shared_data = shared_data
        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = STATE_PATH + "Bulbasaur.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        self.screenshot_counter = 0
        
        file_path = 'experiments/running_experiment.txt'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write('default_exp_name')
        # Logging initializations
        with open("experiments/running_experiment.txt", "r") as file:
            exp_name = file.read()
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        self.env_id = env_id
        self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        self.s_path.mkdir(parents=True, exist_ok=True)
        
        self.save_state_dir = self.s_path / "save_states"
        self.save_state_dir.mkdir(exist_ok=True)
        
        self.video_path = Path(f'./videos')
        self.video_path.mkdir(parents=True, exist_ok=True)
        # self.csv_path = Path(f'./csv')
        # self.csv_path.mkdir(parents=True, exist_ok=True)
        self.reset_count = 0
        self.explore_hidden_obj_weight = 1
        self.pokemon_center_save_states = []
        self.pokecenters = [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 147, 182]
        
        # BET ADDED nimixx api
        # Import this class for api
        self.api = Game(self.game)
        
        R, C = self.screen.ndarray.shape[:2]
        self.obs_size = (R // 2, C // 2) # 72, 80, 3

        if self.use_screen_memory:
            # self.screen_memory = defaultdict(
            #     lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            # )
            # self.obs_size += (5,)
            self.obs_size += (4,)
        else:
            self.obs_size += (3,)
        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=self.obs_size
        )
        self.action_space = spaces.Discrete(len(VALID_ACTIONS))
        self.register_hooks()
        
        # BET ADDED
        # BET ADDED TREE INSTANTIATIONS
        self.min_distances = {}  # Key: Tree position, Value: minimum distance reached
        self.rewarded_distances = {}  # Key: Tree position, Value: set of rewarded distances
        self.used_cut_on_map_n = 0
        self.seen_map_dict = {}
        # Define the specific coordinate ranges to encourage exploration around trees
        self.celadon_tree_attraction_coords = {
            'inside_barrier': [((28, 34), (2, 17)), ((33, 34), (18, 36))],
            'tree_coord': [(32, 35)],
            'outside_box': [((30, 31), (33, 37))]
        }

        self.celadon_reset_done = False
        self.vermilion_reset_done = False
        self.seen_routes_9_and_10_and_rock_tunnel = False
        
        # BET ADDED STATE INSTANTIATIONS
        self.past_events_string = self.all_events_string
        self._all_events_string = ''
        self.time = 0
        self.special_exploration_scale = 0
        self.elite_4_lost = False
        self.elite_4_early_done = False
        self.elite_4_started_step = None
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.has_lemonade_in_bag = False
        self.has_fresh_water_in_bag = False
        self.has_soda_pop_in_bag = False
        self.has_silph_scope_in_bag = False
        self.has_lift_key_in_bag = False
        self.has_pokedoll_in_bag = False
        self.has_bicycle_in_bag = False
        self.gym_info = GYM_INFO
        self.hideout_seen_coords = {(16, 12, 201), (17, 9, 201), (24, 19, 201), (15, 12, 201), (23, 13, 201), (18, 10, 201), (22, 15, 201), (25, 19, 201), (20, 19, 201), (14, 12, 201), (22, 13, 201), (25, 15, 201), (21, 19, 201), (22, 19, 201), (13, 11, 201), (16, 9, 201), (25, 14, 201), (13, 13, 201), (25, 18, 201), (16, 11, 201), (23, 19, 201), (18, 9, 201), (25, 16, 201), (18, 11, 201), (22, 14, 201), (19, 19, 201), (18, 19, 202), (25, 13, 201), (13, 10, 201), (24, 13, 201), (13, 12, 201), (25, 17, 201), (16, 10, 201), (13, 14, 201)}
        self.cut_coords = {}
        self.cut_tiles = 0
        self.completed_milestones = []
        self.current_time = datetime.now()
        
        # BET ADDED INITIALIZATION OF ALL REWARDS INSTANTIATIONS
        self.event_reward = 0
        self.bill_capt_reward = 0
        self.seen_pokemon_reward = 0
        self.caught_pokemon_reward = 0
        self.moves_obtained_reward = 0
        self.bill_reward = 0
        self.hm_reward = 0
        self.level_reward = 0
        self.death_reward = 0
        self.badges_reward = 0
        self.healing_reward = 0
        self.exploration_reward = 0
        self.cut_rew = 0
        self.that_guy_reward = 0
        self.cut_coords_reward = 0
        self.cut_tiles_reward = 0
        self.tree_distance_reward = 0
        self.dojo_reward = 0
        self.hideout_reward = 0
        self.has_lemonade_in_bag_reward = 0
        self.has_fresh_water_in_bag_reward = 0
        self.has_soda_pop_in_bag_reward = 0
        self.has_silph_scope_in_bag_reward = 0
        self.has_lift_key_in_bag_reward = 0
        self.has_pokedoll_in_bag_reward = 0
        self.has_bicycle_in_bag_reward = 0
        self.dojo_events_reward = 0
        self.silph_co_events_reward = 0
        self.hideout_events_reward = 0
        self.poke_tower_events_reward = 0
        self.lock_1_use_reward = 0
        self.gym3_events_reward = 0
        self.gym4_events_reward = 0
        self.gym5_events_reward = 0
        self.gym6_events_reward = 0
        self.gym7_events_reward = 0
        self.reward_scale = 4
        self.level_reward_badge_scale = 0
        self.bill_state = 0
        # self.badges = 0
        self.len_respawn_reward = 0
        self.final_reward = 0
        self.reward = 0
        self.rocket_hideout_maps = [199, 200, 201, 202, 203]
        self.poketower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.silph_co_maps = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.pokemon_tower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.vermilion_city_gym_map = [92]
        self.advanced_gym_maps = [92, 134, 157, 166, 178] # Vermilion, Celadon, Fuchsia, Cinnabar, Saffron
        self.routes_9_and_10_and_rock_tunnel = [20, 21, 82, 232]
        self.bonus_exploration_reward_maps = self.rocket_hideout_maps + self.poketower_maps + self.silph_co_maps + self.vermilion_city_gym_map + self.advanced_gym_maps

        
        self.rocket_hideout_reward_shape = [5, 6, 7, 8, 10]
        self.vermilion_city_and_gym_reward_shape = [10]
        self.cut = 0
        self.max_multiplier_cap = 5  # Cap for the dynamic reward multiplier
        self.exploration_reward_cap = 2000  # Cap for the total exploration reward
        self.initial_multiplier_value = 1  # Starting value for the multiplier
                
        # Dynamic progress detection / rewarding
        self.last_progress_step = 0  # Step at which the last progression was made
        self.current_step = 0  # Current step in the training process
        self.progression_detected = False
        self.bonus_reward_increment = 0.02  # Initial bonus reward increment
        self.max_bonus_reward_increment = 0.1  # Maximum bonus reward increment
        self.bonus_reward_growth_rate = 0.01  # How much the bonus increases per step without progression
        
        # More dynamic rewards stuff
        self.seen_coords_specific_maps = set()
        self.steps_since_last_new_location = 0
        self.step_threshold = 20480
        self.recent_frames = []
        self.agent_stats = {}
        
        self.used_cut_reward = 0

        # Milestone completion variables (adjust as desired)
        self.milestone_keys = ["badge_1", "mt_moon_completion", "badge_2", 
                       "bill_completion", "rubbed_captains_back", 
                       "taught_cut", "used_cut_on_good_tree"]
        self.milestone_threshold_values = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3] # len == len(self.milestone_threshold_dict)
        self.milestone_threshold_dict = ({key: value} for key, value in zip(self.milestone_keys, self.milestone_threshold_values))
        self.overall_progress = {}
        self.actions_required = {}
        self.selected_files = {}

        # ## BET FIXED WINDOW INIT ##
        self.bet_fixed_window_init()

        # BET added 5/6/24
        self.rubbed_captains_back = 0
        self.rubbed_captains_back_reward = 0

    def register_hooks(self):
        hook_register(self.game, None, "DisplayStartMenu", self.start_menu_hook, None)
        hook_register(self.game, None, "RedisplayStartMenu", self.start_menu_hook, None)
        hook_register(self.game, None, "StartMenu_Item", self.item_menu_hook, None)
        hook_register(self.game, None, "StartMenu_Pokemon", self.pokemon_menu_hook, None)
        hook_register(self.game, None, "StartMenu_Pokemon.choseStats", self.chose_stats_hook, None)
        hook_register(self.game, None, "StartMenu_Item.choseItem", self.chose_item_hook, None)
        hook_register(self.game, None, "DisplayTextID.spriteHandling", self.sprite_hook, None)
        hook_register(self.game, 
            None, "CheckForHiddenObject.foundMatchingObject", self.hidden_object_hook, None
        )
        hook_register(self.game, None, "HandleBlackOut", self.blackout_hook, None)
        hook_register(self.game, None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        # hook_register(self.game, None, "UsedCut.nothingToCut", self.cut_hook, context=True)
        # hook_register(self.game, None, "UsedCut.canCut", self.cut_hook, context=False)
    
    def bet_fixed_window(self, glob_r, glob_c):
        """Create a centered window around the player's position that tracks explored coords across all maps"""
        half_window_height = 72 // 2
        half_window_width = 80 // 2        
        start_y = max(0, glob_r - half_window_height)
        start_x = max(0, glob_c - half_window_width)
        end_y = min(444, glob_r + half_window_height)
        end_x = min(436, glob_c + half_window_width)
        viewport = self.global_map[start_y:end_y, start_x:end_x]
        if viewport.shape != (72, 80):
            viewport = resize(viewport, (72, 80), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)        
        return viewport

    def load_and_apply_region_data(self):
        MAP_PATH = __file__.rstrip('environment.py') + 'map_data.json'
        map_data = json.load(open(MAP_PATH, 'r'))['regions']
        for region in map_data:
            region_id = int(region["id"])
            if region_id == -1:
                continue
            x_start, y_start = region["coordinates"]
            width, height = region["tileSize"]
            self.global_map[y_start:y_start + height, x_start:x_start + width] = 0  # Make regions visible

    def bet_fixed_window_init(self):
        # self.fig, self.ax = plt.subplots()
        # plt.ion()  # Turn on interactive mode
        self.global_map = np.full((444, 436, 1), -1, dtype=np.uint8)  # Initialize transparent background
        self.image = None
        self.bet_seen_coords = set()
        self.load_and_apply_region_data() 
                
    # def update_seen_coords(self):
    #     for (y, x) in self.bet_seen_coords:
    #         if 0 <= y < 444 and 0 <= x < 436:
    #             self.global_map[y, x] = 255  # Mark visited locations white
    
    def update_seen_coords(self):
        mask = np.zeros((444, 436), dtype=bool)        
        valid_coords = [(y, x) for (y, x) in self.bet_seen_coords if 0 <= y < 444 and 0 <= x < 436]
        ys, xs = zip(*valid_coords)
        mask[ys, xs] = True
        self.global_map[mask] = 255  # Mark visited locations white

    
    def save_screenshot(self, event, map_n):
        self.screenshot_counter += 1
        ss_dir = Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{self.screenshot_counter}_{event}_{map_n}.jpeg'),
            self.screen.screen_ndarray())  # (144, 160, 3)

    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        current_map_n = ram_map.map_n
        self.pokemon_center_save_states.append(state)
        # self.initial_states.append(state)
        
    def load_pokemon_center_state(self):
        return self.pokemon_center_save_states[len(self.pokemon_center_save_states) -1]
    
    def load_last_state(self):
        return self.initial_states[len(self.initial_states) - 1]
    
    def load_first_state(self):
        return self.initial_states[0]
    
    def load_random_state(self):
        rand_idx = random.randint(0, len(self.initial_states) - 1)
        return self.initial_states[rand_idx]

    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}

    def get_fixed_window(self, arr, y, x, window_size):
        height, width, _ = arr.shape
        h_w, w_w = window_size[0], window_size[1]
        h_w, w_w = window_size[0] // 2, window_size[1] // 2

        y_min = max(0, y - h_w)
        y_max = min(height, y + h_w + (window_size[0] % 2))
        x_min = max(0, x - w_w)
        x_max = min(width, x + w_w + (window_size[1] % 2))

        window = arr[y_min:y_max, x_min:x_max]

        pad_top = h_w - (y - y_min)
        pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
        pad_left = w_w - (x - x_min)
        pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)

        return np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    def render(self): # , bet_window):
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.game)
            
            # # Update tile map
            # mmap = self.screen_memory[map_n]
            # if 0 <= r <= 254 and 0 <= c <= 254:
            #     mmap[r, c] = 255

            self.update_seen_coords()
            glob_r, glob_c = game_map.local_to_global(r, c, map_n)
            bet_window = self.bet_fixed_window(glob_r, glob_c) if self.badges >= 3 else self.get_fixed_window(self.global_map, glob_r, glob_c, self.observation_space.shape)
            # Downsamples the screen and retrieves a fixed window from mmap,
            # then concatenates along the 3rd-dimensional axis (image channel)
            return np.concatenate(
                (
                    self.screen.ndarray[::2, ::2, :3],
                    bet_window,
                    # self.get_fixed_window(mmap, r, c, self.observation_space.shape),
                ),
                axis=2,
            )
        else:
            return self.screen.screen_ndarray()[::2, ::2]

    # def render(self):
    #     if self.use_screen_memory:
    #         r, c, map_n = ram_map.position(self.game)
    #         # Update tile map
    #         mmap = self.screen_memory[map_n]
    #         if 0 <= r <= 254 and 0 <= c <= 254:
    #             mmap[r, c] = 255

    #         # Downsamples the screen and retrieves a fixed window from mmap,
    #         # then concatenates along the 3rd-dimensional axis (image channel)
    #         return np.concatenate(
    #             (
    #                 self.screen.screen_ndarray()[::2, ::2],
    #                 self.get_fixed_window(mmap, r, c, self.observation_space.shape),
    #             ),
    #             axis=2,
    #         )
    #     else:
    #         return self.screen.screen_ndarray()[::2, ::2]

    # BET ADDED TREE OBSERVATIONS
    # into Vermilion gym tree reward reset
    # PyBoy hooks
    def hidden_object_hook(self, *args, **kwargs):
        hidden_object_addr = ram_map.symbol_lookup(self.game, "wHiddenObjectIndex")[1]
        hidden_object_id = ram_map.read_m(self.game, hidden_object_addr)

        map_id_addr = ram_map.symbol_lookup(self.game, "wCurMap")[1]
        map_id = ram_map.read_m(self.game, map_id_addr)

        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1


    def sprite_hook(self, *args, **kwargs):
        sprite_id_addr = ram_map.symbol_lookup(self.game, "hSpriteIndexOrTextID")[1]
        sprite_id = ram_map.read_m(self.game, sprite_id_addr)
        
        map_id_addr = ram_map.symbol_lookup(self.game, "wCurMap")[1]
        map_id = ram_map.read_m(self.game, map_id_addr)
        
        self.seen_npcs[(map_id, sprite_id)] = 1

    def start_menu_hook(self, *args, **kwargs):
        if ram_map.read_m(self.game, "wIsInBattle") == 0:
            self.seen_start_menu = 1

    def item_menu_hook(self, *args, **kwargs):
        if ram_map.read_m(self.game, "wIsInBattle") == 0:
            self.seen_bag_menu = 1

    def pokemon_menu_hook(self, *args, **kwargs):
        if ram_map.read_m(self.game, "wIsInBattle") == 0:
            self.seen_pokemon_menu = 1

    def chose_stats_hook(self, *args, **kwargs):
        if ram_map.read_m(self.game, "wIsInBattle") == 0:
            self.seen_stats_menu = 1

    def chose_item_hook(self, *args, **kwargs):
        if ram_map.read_m(self.game, "wIsInBattle") == 0:
            self.seen_action_bag_menu = 1

    def blackout_hook(self, *args, **kwargs):
        self.blackout_count += 1

    def blackout_update_hook(self, *args, **kwargs):
        self.blackout_check = ram_map.read_m(self.game, "wLastBlackoutMap")
    
    def reset_rewarded_distances_for_vermilion(self):
        if self.badges >= 3:
            if not hasattr(self, 'vermilion_reset_done'):
                self.rewarded_distances[(3984//16, 4512//16, 5)] = set()
                self.vermilion_reset_done = True
        else:
            # Reset twice per episode, and on reset
            reset_interval = 20480 // 3
            reset_times = [reset_interval + 1, reset_interval * 2 + 1]
            if self.time in reset_times or self.time == 0:
                self.rewarded_distances[(3984//16, 4512//16, 5)] = set()
    
    # Cerulean to Rock Tunnel tree reward reset
    # Reset reward only if Gym 3 has been completed.
    def reset_rewarded_distances_for_cerulean(self):
        if self.badges >= 3:
            # Reset twice per episode, and on reset
            reset_interval = 20480 // 3
            reset_times = [reset_interval + 1, reset_interval * 2 + 1]
            if not self.seen_routes_9_and_10_and_rock_tunnel:
                if self.time in reset_times or self.time == 0:
                    self.rewarded_distances[(4464//16, 2176//16, 20)] = set()
            
    def reset_rewarded_distances_for_celadon(self):
        events_status_gym4 = ram_map_leanke.monitor_gym4_events(self.game)
        if events_status_gym4['four'] != 0:
            if not hasattr(self, 'celadon_reset_done'):
                self.rewarded_distances[(3184//16, 3584//16, 6)] = set()
                self.celadon_reset_done = True
        else:
            # Reset thrice per episode, and on reset
            reset_interval = 20480 // 4
            reset_times = [reset_interval, reset_interval * 2, reset_interval * 3]
            if self.time in reset_times or self.time == 0:
                self.rewarded_distances[(3184//16, 3584//16, 6)] = set()
    
    def visited_maps(self):
        _, _, map_n = ram_map.position(self.game)
        if map_n in self.routes_9_and_10_and_rock_tunnel:
            self.seen_routes_9_and_10_and_rock_tunnel = True
    
    def detect_and_reward_trees(self, player_grid_pos, map_n, vision_range=5):
        if map_n not in MAPS_WITH_TREES:
            # print(f"\nNo trees to interact with in map {map_n}.")
            return 0.0
        # Correct the coordinate order: player_x corresponds to glob_c, and player_y to glob_r
        player_x, player_y = player_grid_pos  # Use the correct order based on your description
        # print(f"\nPlayer Grid Position: (X: {player_x}, Y: {player_y})")
        # print(f"Vision Range: {vision_range}")
        # print(f"Trees in map {map_n}:")
        tree_counter = 0  # For numbering trees
        total_reward = 0.0
        tree_counter = 0
        for y, x, m in TREE_POSITIONS_PIXELS:
            if m == map_n:
                tree_counter += 1
                tree_x, tree_y = x // 16, y // 16
                # Adjusting print statement to reflect corrected tree position
                corrected_tree_y = tree_y if not (tree_x == 212 and tree_y == 210) else 211
                # print(f"  Tree #{tree_counter} Grid Position: (X: {tree_x}, Y: {corrected_tree_y})")
                distance = abs(player_x - tree_x) + abs(player_y - corrected_tree_y)
                # print(f"  Distance to Tree #{tree_counter}: {distance}")
                if distance <= vision_range:
                    reward = 1 / max(distance, 1)  # Prevent division by zero; assume at least distance of 1
                    total_reward += reward
                    # print(f"  Reward for Tree #{tree_counter}: {reward}\n")
                else:
                    pass# print(f"  Tree #{tree_counter} is outside vision range.\n")
        # print(f"Total reward from trees: {total_reward}\n")
        return total_reward
        
    def compute_tree_reward(self, player_pos, trees_positions, map_n, N=3, p=2, q=1):
        if map_n not in MAPS_WITH_TREES:
            # print(f"No cuttable trees in map {map_n}.")
            return 0.0        
        trees_per_current_map_n = TREE_COUNT_PER_MAP[map_n]
        if self.used_cut_on_map_n >= trees_per_current_map_n:
            return 0.0
        if not hasattr(self, 'min_distances'):
            self.min_distances = {}
        if not hasattr(self, 'rewarded_distances'):
            self.rewarded_distances = {}
        total_reward = 0
        nearest_trees_features = self.trees_features(player_pos, trees_positions, N)        
        for i in range(N):
            if i < len(nearest_trees_features):  # Ensure there are enough features
                distance = nearest_trees_features[i]            
            tree_key = (trees_positions[i][0], trees_positions[i][1], map_n)
            if tree_key not in self.min_distances:
                self.min_distances[tree_key] = float('inf')            
            if distance < self.min_distances[tree_key] and distance not in self.rewarded_distances.get(tree_key, set()):
                self.min_distances[tree_key] = distance
                if tree_key not in self.rewarded_distances:
                    self.rewarded_distances[tree_key] = set()
                self.rewarded_distances[tree_key].add(distance)                
                # Adjust reward computation
                if distance == 1:  # Maximal reward for being directly adjacent
                    distance_reward = 1
                else:
                    distance_reward = 1 / (distance ** p)                
                priority_reward = 1 / ((i+1) ** q)
                total_reward += distance_reward * priority_reward
        return total_reward
        
    def calculate_distance(self, player_pos, tree_pos):
        """Calculate the Manhattan distance from player to a tree."""
        dy, dx = np.abs(np.array(tree_pos) - np.array(player_pos))
        distance = dy + dx  # Manhattan distance for grid movement
        return distance
    
    # def calculate_distance_and_angle(self, player_pos, tree_pos):
    #     """Recalculate the Euclidean distance and angle from player to a tree."""
    #     # Ensure the player_pos and tree_pos are in (y, x) format
    #     dy, dx = np.array(tree_pos) - np.array(player_pos)
    #     distance = np.sqrt(dy**2 + dx**2)
    #     angle = np.arctan2(dy, dx)  # Angle in radians, considering dy first due to (y, x) ordering
    #     return distance, angle

    def trees_features(self, player_pos, trees_positions, N=3):
        # Calculate distances to all trees
        distances = [self.calculate_distance(player_pos, pos) for pos in trees_positions]
        # Sort by distance and select the nearest N
        nearest_trees = sorted(distances)[:N]        
        # Create a flat list of distances for the nearest N trees
        features = []
        for distance in nearest_trees:
            features.append(distance)
        # Pad with zeros if fewer than N trees are available
        if len(nearest_trees) < N:
            features.extend([0] * (N - len(features)))            
        return features

    # def save_fixed_window(window, file_path="fixed_window.png"):
    #     # Check if window is grayscale (2D array)
    #     if window.ndim == 2:
    #         # Convert grayscale to RGB
    #         window_rgb = gray2rgb(window)
    #     elif window.ndim == 3 and window.shape[2] == 1:
    #         # Convert single channel to RGB
    #         window_rgb = np.repeat(window, 3, axis=2)
    #     else:
    #         # Assume window is already RGB or RGBA
    #         window_rgb = window
    #     # Save the RGB image
    #     plt.imsave(file_path, window_rgb)
    #     print(f"Fixed window image saved to {file_path}")

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, action, self.headless)
        return self.render(), 0, False, False, {}

    def video(self):
        video = self.screen.screen_ndarray()
        return video

    def close(self):
        self.game.stop(False)

    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = {}
    
    # BET ADDED
    @property
    def all_events_string(self):
        if not hasattr(self, '_all_events_string'):
            self._all_events_string = ''  # Default fallback
            return self._all_events_string
        else:
            # cache all events string to improve performance
            if not self._all_events_string:
                event_flags_start = 0xD747
                event_flags_end = 0xD886
                result = ''
                for i in range(event_flags_start, event_flags_end):
                    result += bin(ram_map.mem_val(self.game, i))[2:].zfill(8)  # .zfill(8)
                self._all_events_string = result
            return self._all_events_string    

class Environment(Base):
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None, # none
        headless=True,
        save_video=False,
        quiet=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(rom_path, state_path, headless, save_video, quiet, **kwargs)
        self.counts_map = np.zeros((444, 436))
        self.death_count = 0
        self.verbose = verbose
        self.screenshot_counter = 0
        self.include_conditions = []
        self.seen_maps_difference = set()
        self.current_maps = []
        self.talk_to_npc_reward = 0
        self.talk_to_npc_count = {}
        self.already_got_npc_reward = set()
        self.ss_anne_state = False
        self.explore_npc_weight = 1
        self.is_dead = False
        self.last_map = -1
        self.log = True
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.talk_to_npc_reward = 0
        self.talk_to_npc_count = {}
        self.already_got_npc_reward = set()
        self.ss_anne_state = False
        self.seen_npcs = {}
        self.explore_npc_weight = 1
        self.last_map = -1
        self.init_hidden_obj_mem()
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        self.log = False
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.visited_pokecenter_list = []
        self._all_events_string = ''
        self.used_cut_coords_set = set()
        self.rewarded_coords = set()
        self.rewarded_position = (0, 0)
        # self.seen_coords = set() ## moved from reset
        self.state_loaded_instead_of_resetting_in_game = 0
        self.badge_count = 0
        
        # BET ADDED
        self.saved_states_dict = {}
        self.seen_maps_no_reward = set()
        self.seen_coords_no_reward = set()
        self.seen_map_dict = {}
        self.is_warping = False
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self._all_events_string = ''
        self.total_healing_rew = 0
        self.last_health = 1
        
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203]
        self.saffron_city = [10, 70, 76, 178, 180, 182]
        self.fighting_dojo = [177]
        self.vermilion_gym = [92]
        self.exploration_reward = 0
        self.celadon_tree_reward = 0
        self.celadon_tree_reward_counter = 0
        self.ctr_bonus = 0
        
        # Standard amount for exploration reward per new tile explored
        self.exploration_reward_increment = 0.002
        # Bonus amount for exploration reward per new tile explored
        self.bonus_fixed_expl_reward_increment = 0.005
        self.bonus_dynamic_reward = 0
        self.bonus_dynamic_reward_increment = 0.0003
        self.bonus_dynamic_reward_multiplier = 0.0001
        
        # key: map_n, value: bonus reward amount per coord explored
        self.fixed_bonus_expl_maps = {199: self.bonus_fixed_expl_reward_increment,
                                     200: self.bonus_fixed_expl_reward_increment,
                                     201: self.bonus_fixed_expl_reward_increment,
                                     202: self.bonus_fixed_expl_reward_increment,
                                     203: self.bonus_fixed_expl_reward_increment,
                                     92: self.bonus_fixed_expl_reward_increment}
        self.dynamic_bonus_expl_maps = set()
        self.dynamic_bonus_expl_seen_coords = set()
        self.shaped_exploration_reward = 0
        self.lock_1_use_counter = 0
        self.last_lock_1_use_counter = 0
        self.last_can_mem_val = -1
        self.can_reward = 0
        self.respawn = set()
        
        # Exploration reward capper
        self.healthy_increases = [23, 30, 33, 37, 42, 43]
        self.ema = 0  # Initialize the EMA at 0 for the exploration reward
        self.last_time = 0  # To calculate intervals
        self.smoothing = 2 / (5000000 + 1)  # Assuming a period of 5 million steps for smoothing
        self.average_healthy_slope = self.calculate_average_slope(self.healthy_increases)
        self.unhealthy_slope_threshold = 1.25 * self.average_healthy_slope
        # For calculating projected healthy increase, we need a reference point in time and reward
        self.reference_time = 0
        self.reference_reward = self.healthy_increases[0]

        # #for reseting at 7 resets - leave commented out
        # self.prev_map_n = None
        # self.max_events = 0
        # self.max_level_sum = 0
        self.max_opponent_level = 0
        # self.seen_coords = set()
        # self.seen_maps = set()
        # self.total_healing = 0
        # self.last_hp = 1.0
        # self.last_party_size = 1
        # self.hm_count = 0
        # self.cut = 0
        self.used_cut = 0
        # self.cut_coords = {}
        # self.cut_tiles = {} # set([])
        # self.cut_state = deque(maxlen=3)
        # self.seen_start_menu = 0
        # self.seen_pokemon_menu = 0
        # self.seen_stats_menu = 0
        # self.seen_bag_menu = 0
        # self.seen_cancel_bag_menu = 0
        # self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        # self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        # self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        
    def read_hp_fraction(self):
        hp_sum = sum([ram_map.read_hp(self.game, add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([ram_map.read_hp(self.game, add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        if max_hp_sum:
            return hp_sum / max_hp_sum
        else:
            return 0
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            # fixed catching pokemon might treated as healing
            # fixed leveling count as healing, min heal amount is 4%
            heal_amount = cur_health - self.last_health
            if self.last_num_poke == self.read_num_poke() and self.last_health > 0 and heal_amount > 0.04:
                if heal_amount > (0.60 / self.read_num_poke()):
                    # changed to static heal reward
                    # 1 pokemon from 0 to 100% hp is 0.167 with 6 pokemon
                    # so 0.1 total heal is around 60% hp
                    print(f' healed: {heal_amount:.2f}')
                    self.total_healing_rew += 0.1
                # if heal_amount > 0.5:
                #     print(f' healed: {heal_amount:.2f}')
                #     # self.save_screenshot('healing')
                # self.total_healing_rew += heal_amount * 1
            elif self.last_health <= 0:
                    self.d
    
    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = ram_map.read_m(self.game,  i + 0xD2F7)
            seen_mem = ram_map.read_m(self.game,  i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   
    
    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if ram_map.read_m(self.game,  i) != 0:
                for j in range(4):
                    move_id = ram_map.read_m(self.game,  i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
                        if move_id == 15:
                            self.cut = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(ram_map.read_m(self.game,  0xda80)):
            offset = i*box_struct_length + 0xda96
            if ram_map.read_m(self.game,  offset) != 0:
                for j in range(4):
                    move_id = ram_map.read_m(self.game,  offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1
                        
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = ram_map.read_m(self.game,  first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids
    
    def poke_count_hms(self):
        pokemon_info = ram_map.pokemon_l(self.game)
        pokes_hm_counts = {
            'Cut': 0,
            'Flash': 0,
            'Fly': 0,
            'Surf': 0,
            'Strength': 0,
        }
        for pokemon in pokemon_info:
            moves = pokemon['moves']
            pokes_hm_counts['Cut'] += 'Cut' in moves
            pokes_hm_counts['Flash'] += 'Flash' in moves
            pokes_hm_counts['Fly'] += 'Fly' in moves
            pokes_hm_counts['Surf'] += 'Surf' in moves
            pokes_hm_counts['Strength'] += 'Strength' in moves
        return pokes_hm_counts
    
    def get_hm_rewards(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        total_hm_cnt = 0
        for hm_id in hm_ids:
            if hm_id in items:
                total_hm_cnt += 1
        return total_hm_cnt * 1
            
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.video())

    def get_game_coords(self):
        return (ram_map.mem_val(self.game, 0xD362), ram_map.mem_val(self.game, 0xD361), ram_map.mem_val(self.game, 0xD35E))
    
    def check_if_in_start_menu(self) -> bool:
        return (
            ram_map.mem_val(self.game, 0xD057) == 0
            and ram_map.mem_val(self.game, 0xCF13) == 0
            and ram_map.mem_val(self.game, 0xFF8C) == 6
            and ram_map.mem_val(self.game, 0xCF94) == 0
        )

    def check_if_in_pokemon_menu(self) -> bool:
        return (
            ram_map.mem_val(self.game, 0xD057) == 0
            and ram_map.mem_val(self.game, 0xCF13) == 0
            and ram_map.mem_val(self.game, 0xFF8C) == 6
            and ram_map.mem_val(self.game, 0xCF94) == 2
        )

    def check_if_in_stats_menu(self) -> bool:
        return (
            ram_map.mem_val(self.game, 0xD057) == 0
            and ram_map.mem_val(self.game, 0xCF13) == 0)
            
    def update_heat_map(self, r, c, current_map):
        '''
        Updates the heat map based on the agent's current position.

        Args:
            r (int): global y coordinate of the agent's position.
            c (int): global x coordinate of the agent's position.
            current_map (int): ID of the current map (map_n)

        Updates the counts_map to track the frequency of visits to each position on the map.
        '''
        # Convert local position to global position
        try:
            glob_r, glob_c = game_map.local_to_global(r, c, current_map)
        except IndexError:
            print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
            glob_r = 0
            glob_c = 0

        # Update heat map based on current map
        if self.last_map == current_map or self.last_map == -1:
            # Increment count for current global position
                try:
                    self.counts_map[glob_r, glob_c] += 1
                except:
                    pass
        else:
            # Reset count for current global position if it's a new map for warp artifacts
            self.counts_map[(glob_r, glob_c)] = -1

        # Update last_map for the next iteration
        self.last_map = current_map

    def check_if_in_bag_menu(self) -> bool:
        return (
            ram_map.mem_val(self.game, 0xD057) == 0
            and ram_map.mem_val(self.game, 0xCF13) == 0
            # and ram_map.mem_val(self.game, 0xFF8C) == 6 # only sometimes
            and ram_map.mem_val(self.game, 0xCF94) == 3
        )

    def check_if_cancel_bag_menu(self, action) -> bool:
        return (
            action == WindowEvent.PRESS_BUTTON_A
            and ram_map.mem_val(self.game, 0xD057) == 0
            and ram_map.mem_val(self.game, 0xCF13) == 0
            # and ram_map.mem_val(self.game, 0xFF8C) == 6
            and ram_map.mem_val(self.game, 0xCF94) == 3
            and ram_map.mem_val(self.game, 0xD31D) == ram_map.mem_val(self.game, 0xCC36) + ram_map.mem_val(self.game, 0xCC26)
        )
        
    def update_reward(self, new_position):
        """
        Update and determine if the new_position should be rewarded based on every 10 steps
        taken within the same map considering the Manhattan distance.

        :param new_position: Tuple (glob_r, glob_c, map_n) representing the new global position and map identifier.
        """
        should_reward = False
        new_glob_r, new_glob_c, new_map_n = new_position

        # Check if the new position should be rewarded compared to existing positions on the same map
        for rewarded_position in self.rewarded_coords:
            rewarded_glob_r, rewarded_glob_c, rewarded_map_n = rewarded_position
            if new_map_n == rewarded_map_n:
                distance = abs(rewarded_glob_r - new_glob_r) + abs(rewarded_glob_c - new_glob_c)
                if distance >= 10:
                    should_reward = True
                    break

        if should_reward:
            self.rewarded_coords.add(new_position)
            
    def check_bag_for_silph_scope(self):
        if 0x4A in self.get_items_in_bag():
            if 0x48 in self.get_items_in_bag():
                self.have_silph_scope = True
                return self.have_silph_scope
            
    def current_coords(self):
        return self.last_10_coords[0]
    
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    def update_seen_map_dict(self):
        # if self.get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.current_map_id() - 1
        x, y = self.current_coords()
        if cur_map_id not in self.seen_map_dict:
            self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
        # do not update if is warping
        if not self.is_warping:
            if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
                self.stuck_cnt += 1
                # print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
                if self.stuck_cnt > 50:
                    print(f'stucked for > 50 steps, force ES')
                    self.early_done = True
                    self.stuck_cnt = 0
                # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
            else:
                self.stuck_cnt = 0
                self.seen_map_dict[cur_map_id][y, x] = self.time

    def get_badges(self):
        badge_count = ram_map.bit_count(ram_map.mem_val(self.game, 0xD356))
        # return badge_count
        if badge_count < 8 or self.elite_4_lost or self.elite_4_early_done:
            return badge_count
        else:
            # LORELEIS D863, bit 1
            # BRUNOS D864, bit 1
            # AGATHAS D865, bit 1
            # LANCES D866, bit 1
            # CHAMPION D867, bit 1
            elite_four_event_addr_bits = [
                [0xD863, 1],  # LORELEIS
                [0xD864, 1],  # BRUNOS
                [0xD865, 1],  # AGATHAS
                [0xD866, 1],  # LANCES
                [0xD867, 1],  # CHAMPION
            ]
            elite_4_extra_badges = 0
            for addr_bit in elite_four_event_addr_bits:
                if ram_map.read_bit(self.game, addr_bit[0], addr_bit[1]):
                    elite_4_extra_badges += 1
            return 8 + elite_4_extra_badges

    def get_levels_sum(self):
        poke_levels = [max(ram_map.mem_val(self.game, a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def read_event_bits(self):
        return [
            int(bit)
            for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
            for bit in f"{ram_map.read_bit(self.game, i):08b}"
        ]
    
    def read_num_poke(self):
        return ram_map.mem_val(self.game, 0xD163)
    
    def update_num_poke(self):
        self.last_num_poke = self.read_num_poke()
    
    def get_max_n_levels_sum(self, n, max_level):
        num_poke = self.read_num_poke()
        poke_level_addresses = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        poke_levels = [max(min(ram_map.mem_val(self.game, a), max_level) - 2, 0) for a in poke_level_addresses[:num_poke]]
        return max(sum(sorted(poke_levels)[-n:]) - 4, 0)
    
    @property
    def is_in_elite_4(self):
        return self.current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]
    
    def get_levels_reward(self):
        if not self.level_reward_badge_scale:
            level_sum = self.get_levels_sum()
            self.max_level_rew = max(self.max_level_rew, level_sum)
        else:
            badge_count = min(self.get_badges(), 8)
            gym_next = self.gym_info[badge_count]
            gym_num_poke = gym_next['num_poke']
            gym_max_level = gym_next['max_level'] * self.level_reward_badge_scale
            self.level_reward = self.get_max_n_levels_sum(gym_num_poke, gym_max_level)  # changed, level reward for all 6 pokemon
            if badge_count >= 7 and self.level_reward > self.max_level_rew and not self.is_in_elite_4:
                level_diff = self.level_reward - self.max_level_rew
                if level_diff > 6 and self.party_level_post == 0:
                    # self.party_level_post = 0
                    pass
                else:
                    self.party_level_post += level_diff
            self.max_level_rew = max(self.max_level_rew, self.level_reward)
        return ((self.max_level_rew - self.party_level_post) * 0.5) + (self.party_level_post * 2.0)
        # return self.max_level_rew * 0.5  # 11/11-3 changed: from 0.5 to 1.0
    
    def get_special_key_items_reward(self):
        items = self.get_items_in_bag()
        special_cnt = 0
        # SPECIAL_KEY_ITEM_IDS
        for item_id in SPECIAL_KEY_ITEM_IDS:
            if item_id in items:
                special_cnt += 1
        return special_cnt * 1.0
    
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    ram_map.bit_count(ram_map.read_m(self.game,  i))
                    for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(ram_map.read_bit(self.game, *ram_map.MUSEUM_TICKET_ADDR)),
            0,
        )
        
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
    
    def update_max_op_level(self):
        #opponent_level = ram_map.mem_val(self.game, 0xCFE8) - 5 # base level
        opponent_level = max([ram_map.mem_val(self.game, a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.1  # 0.1

    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        
        last_pokecenter = ram_map.mem_val(self.game, 0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        if last_pokecenter not in self.pokecenter_ids:
            print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
            return -1
        else:
            return self.pokecenter_ids.index(last_pokecenter)   
    
    def get_special_rewards(self):
        rewards = 0
        rewards += len(self.hideout_elevator_maps) * 2.0
        bag_items = self.get_items_in_bag()
        if 0x2B in bag_items:
            # 6.0 full mansion rewards + 1.0 extra key items rewards
            rewards += 7.0
        return rewards
    
    def get_hm_usable_reward(self):
        total = 0
        if self.can_use_cut:
            total += 1
        if self.can_use_surf:
            total += 1
        return total * 2.0
    
    def get_special_key_items_reward(self):
        items = self.get_items_in_bag()
        special_cnt = 0
        # SPECIAL_KEY_ITEM_IDS
        for item_id in SPECIAL_KEY_ITEM_IDS:
            if item_id in items:
                special_cnt += 1
        return special_cnt * 1.0
    
    def get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [ram_map.mem_val(self.game, first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves
    
    def get_hm_move_reward(self):
        all_moves = self.get_party_moves()
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        hm_move_count = 0
        for hm_move in hm_moves:
            if hm_move in all_moves:
                hm_move_count += 1
        return hm_move_count * 1.5
    
    def update_visited_pokecenter_list(self):
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)

    def get_early_done_reward(self):
        return self.elite_4_early_done * -0.3
    
    def get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        return len(self.visited_pokecenter_list) * 2     
    
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = ram_map.mem_val(self.game, 0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(ram_map.mem_val(self.game, i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(ram_map.mem_val(self.game, i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = ram_map.mem_val(self.game, 0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = ram_map.mem_val(self.game, 0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        last_event_rew = self.max_event_rew
        self.max_event_rew = self.update_max_event_rew()
        state_scores = {
            'event': self.max_event_rew,  
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            'level': self.get_levels_reward(), 
            # 'heal': self.total_healing_rew,
            'op_lvl': self.update_max_op_level(),
            # 'dead': -self.get_dead_reward(),
            'badge': self.get_badges_reward(),  # 5
            #'op_poke':self.max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': self.reward_scale * seen_poke_count * 400,
            # 'explore': self.get_knn_reward(last_event_rew),
            'visited_pokecenter': self.get_visited_pokecenter_reward(),
            'hm': self.get_hm_rewards(),
            # 'hm_move': self.get_hm_move_reward(),  # removed this for now
            'hm_usable': self.get_hm_usable_reward(),
            'trees_cut': self.get_used_cut_coords_reward(),
            'early_done': self.get_early_done_reward(),  # removed
            'special_key_items': self.get_special_key_items_reward(),
            'special': self.get_special_rewards(),
            'heal': self.total_healing_rew,
        }

        # multiply by reward scale
        state_scores = {k: v * self.reward_scale for k, v in state_scores.items()}
        
        return state_scores
    
    # BET ADDING A BUNCH OF STUFF
    def minor_patch_victory_road(self):
        address_bits = [
            # victory road
            [0xD7EE, 0],
            [0xD7EE, 7],
            [0xD813, 0],
            [0xD813, 6],
            [0xD869, 7],
        ]
        for ab in address_bits:
            event_value = ram_map.mem_val(self.game, ab[0])
            ram_map.write_mem(self.game, ab[0], ram_map.set_bit(event_value, ab[1]))
    
    def update_last_10_map_ids(self):
        current_modified_map_id = ram_map.mem_val(self.game, 0xD35E) + 1
        # check if current_modified_map_id is in last_10_map_ids
        if current_modified_map_id == self.last_10_map_ids[0][0]:
            return
        else:
            # if self.last_10_map_ids[0][0] != 0:
            #     print(f'map changed from {MAP_ID_REF[self.last_10_map_ids[0][0] - 1]} to {MAP_ID_REF[current_modified_map_id - 1]} at step {self.step_count}')
            self.last_10_map_ids = np.roll(self.last_10_map_ids, 1, axis=0)
            self.last_10_map_ids[0] = [current_modified_map_id, self.step_count]
            map_id = current_modified_map_id - 1
            if map_id in [0x6C, 0xC2, 0xC6, 0x22]:
                self.minor_patch_victory_road()
            # elif map_id == 0x09:
            if map_id not in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                if self.last_10_map_ids[1][0] - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                    # lost in elite 4
                    self.elite_4_lost = True
                    self.elite_4_started_step = None
            if map_id == 0xF5:
                # elite four first room
                # reset elite 4 lost flag
                if self.elite_4_lost:
                    self.elite_4_lost = False
                if self.elite_4_started_step is None:
                    self.elite_4_started_step = self.step_count
    
    def get_event_rewarded_by_address(self, address, bit):
        # read from rewarded_events_string
        event_flags_start = 0xD747
        event_pos = address - event_flags_start
        # bit is reversed
        # string_pos = event_pos * 8 + bit
        string_pos = event_pos * 8 + (7 - bit)
        return self.rewarded_events_string[string_pos] == '1'
    
    def init_caches(self):
        # for cached properties
        self._all_events_string = ''
        self._battle_type = None
        self._cur_seen_map = None
        self._minimap_warp_obs = None
        self._is_warping = None
        self._items_in_bag = None
        self._minimap_obs = None
        self._minimap_sprite = None
        self._bottom_left_screen_tiles = None
        self._num_mon_in_box = None
    
    def update_cut_badge(self):
        if not self._cut_badge:
            # print(f"Attempting to read bit from addr: {RAM.wObtainedBadges.value}, which is type: {type(RAM.wObtainedBadges.value)}")
            self._cut_badge = ram_map.read_bit(self.game, RAM.wObtainedBadges.value, 1) == 1

    def update_surf_badge(self):
        if not self._cut_badge:
            return
        if not self._surf_badge:
            self._surf_badge = ram_map.read_bit(self.game, RAM.wObtainedBadges.value, 4) == 1   

    def update_last_10_coords(self):
        current_coord = np.array([ram_map.mem_val(self.game, 0xD362), ram_map.mem_val(self.game, 0xD361)])
        # check if current_coord is in last_10_coords
        if (current_coord == self.last_10_coords[0]).all():
            return
        else:
            self.last_10_coords = np.roll(self.last_10_coords, 1, axis=0)
            self.last_10_coords[0] = current_coord
    
    @property
    def can_use_surf(self):
        if not self._can_use_surf:
            if self._surf_badge:
                if not self._have_hm03:
                    self._have_hm03 = 0xC6 in self.get_items_in_bag()
                if self._have_hm03:
                    self._can_use_surf = True
        return self._can_use_surf
    
    @property
    def can_use_cut(self):
        # return ram_map.mem_val(self.game, 0xD2E2) == 1
        # check badge, store last badge count, if changed, check if can use cut, bit 1, save permanently
        if not self._can_use_cut:
            if self._cut_badge:
                if not self._have_hm01:
                    self._have_hm01 = 0xc4 in self.get_items_in_bag()
                if self._have_hm01:
                    self._can_use_cut = True
            # self._can_use_cut = self._cut_badge is True and 0xc4 in self.get_items_in_bag()
        return self._can_use_cut
    
    @property
    def have_silph_scope(self):
        if self.can_use_cut and not self._have_silph_scope:
            self._have_silph_scope = 0x48 in self.get_items_in_bag()
        return self._have_silph_scope
    
    @property
    def can_use_flute(self):
        if self.can_use_cut and not self._have_pokeflute:
            self._have_pokeflute = 0x49 in self.get_items_in_bag()
        return self._have_pokeflute
    
    def get_base_event_flags(self):
        # event patches
        # 1. triggered EVENT_FOUND_ROCKET_HIDEOUT 
        # event_value = ram_map.mem_val(self.game, 0xD77E)  # bit 1
        # ram_map.write_mem(self.game, 0xD77E, ram_map.set_bit(event_value, 1))
        # 2. triggered EVENT_GOT_TM13 , fresh_water trade
        event_value = ram_map.mem_val(self.game, 0xD778)  # bit 4
        ram_map.write_mem(self.game, 0xD778, ram_map.set_bit(event_value, 4))
        # address_bits = [
        #     # seafoam islands
        #     [0xD7E8, 6],
        #     [0xD7E8, 7],
        #     [0xD87F, 0],
        #     [0xD87F, 1],
        #     [0xD880, 0],
        #     [0xD880, 1],
        #     [0xD881, 0],
        #     [0xD881, 1],
        #     # victory road
        #     [0xD7EE, 0],
        #     [0xD7EE, 7],
        #     [0xD813, 0],
        #     [0xD813, 6],
        #     [0xD869, 7],
        # ]
        # for ab in address_bits:
        #     event_value = ram_map.mem_val(self.game, ab[0])
        #     ram_map.write_mem(self.game, ab[0], ram_map.set_bit(event_value, ab[1]))

        n_ignored_events = 0
        for event_id in IGNORED_EVENT_IDS:
            if self.all_events_string[event_id] == '1':
                n_ignored_events += 1
        return max(
            self.all_events_string.count('1')
            - n_ignored_events,
        0,
    )
    
    def print_dict_items_and_sum(self, d):
        total_sum = 0  # Initialize sum for the current dictionary level
        
        for key, value in list(d.items()):
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_filtered_dict, sum_nested = self.print_dict_items_and_sum(value)
                if nested_filtered_dict:  # Only include and print if the nested dictionary is not empty
                    print(f"{key}:")
                    d[key] = nested_filtered_dict  # Update the current dictionary with the filtered nested dictionary
                    total_sum += sum_nested  # Add the sum from the nested dictionary to the total sum
                else:
                    del d[key]  # Remove the key if the nested dictionary is empty after filtering
            elif isinstance(value, (int, float)) and value > 0:
                # Include and print the key-value pair if the value is a number greater than 0
                print(f"{key}: {value}")
                total_sum += value  # Add numeric value to total sum
            else:
                del d[key]  # Remove the key if the value is not a positive number

        return d, total_sum  # Return the filtered dictionary and total sum of values

    def write_hp_for_first_pokemon(self, new_hp, new_max_hp):
        """Write new HP value for the first party Pokémon."""
        # HP address for the first party Pokémon
        hp_addr = ram_map.HP_ADDR[0]
        max_hp_addr = ram_map.MAX_HP_ADDR[0]        
        # Break down the new_hp value into two bytes
        hp_high = new_hp // 256  # Get the high byte
        hp_low = new_hp % 256    # Get the low byte
        max_hp_high = new_max_hp // 256  # Get the high byte
        max_hp_low = new_max_hp % 256    # Get the low byte        
        # Write the high byte and low byte to the corresponding memory addresses
        ram_map.write_mem(self.game, hp_addr, hp_high)
        ram_map.write_mem(self.game, hp_addr + 1, hp_low)
        ram_map.write_mem(self.game, max_hp_addr, max_hp_high)
        ram_map.write_mem(self.game, max_hp_addr + 1, max_hp_low)
        # print(f"Set Max HP for the first party Pokémon to {new_max_hp}")
        # print(f"Set HP for the first party Pokémon to {new_hp}")
    
    def update_party_hp_to_max(self):
        """
        Update the HP of all party Pokémon to match their Max HP.
        """
        for i in range(len(ram_map.CHP)):
            # Read Max HP
            max_hp = ram_map.read_uint16(self.game, ram_map.MAX_HP_ADDR[i])            
            # Calculate high and low bytes for Max HP to set as current HP
            hp_high = max_hp // 256
            hp_low = max_hp % 256
            # Update current HP to match Max HP
            ram_map.write_mem(self.game, ram_map.CHP[i], hp_high)
            ram_map.write_mem(self.game, ram_map.CHP[i] + 1, hp_low)
            # print(f"Updated Pokémon {i+1}: HP set to Max HP of {max_hp}.")
                
    def restore_party_move_pp(self):
        """
        Restores the PP of all moves for the party Pokémon based on moves_dict data.
        """
        for i in range(len(ram_map.MOVE1)):  # Assuming same length for MOVE1 to MOVE4
            moves_ids = [ram_map.mem_val(self.game, move_addr) for move_addr in [ram_map.MOVE1[i], ram_map.MOVE2[i], ram_map.MOVE3[i], ram_map.MOVE4[i]]]
            
            for j, move_id in enumerate(moves_ids):
                if move_id in ram_map.moves_dict:
                    # Fetch the move's max PP
                    max_pp = ram_map.moves_dict[move_id]['PP']
                    
                    # Determine the corresponding PP address based on the move slot
                    pp_addr = [ram_map.MOVE1PP[i], ram_map.MOVE2PP[i], ram_map.MOVE3PP[i], ram_map.MOVE4PP[i]][j]
                    
                    # Restore the move's PP
                    ram_map.write_mem(self.game, pp_addr, max_pp)
                    # print(f"Restored PP for {ram_map.moves_dict[move_id]['Move']} to {max_pp}.")
                else:
                    pass
                    # print(f"Move ID {move_id} not found in moves_dict.")

    def count_gym_3_lock_1_use(self):
        gym_3_event_dict = ram_map_leanke.monitor_gym3_events(self.game)
        if (lock_1_status := gym_3_event_dict['lock_one']) == 2 and self.last_lock_1_use_counter == 0:
            self.lock_1_use_counter += 1
            self.last_lock_1_use_counter = lock_1_status
        if gym_3_event_dict['lock_one'] == 0:
            self.last_lock_1_use_counter = 0

    def get_lock_1_use_reward(self):
        self.lock_1_use_reward = self.lock_1_use_counter / 2 # * 10

    def get_can_reward(self):
        mem_val = ram_map.trash_can_memory(self.game)
        if mem_val == self.last_can_mem_val:
            self.can_reward = 0
        else:
            try:
                self.can_reward = mem_val / mem_val
                self.last_can_mem_val = mem_val
            except:
                self.can_reward = 0

    def get_all_events_reward(self):
        if self.all_events_string != self.past_events_string:
            first_i = -1
            for i in range(len(self.all_events_string)):
                if self.all_events_string[i] == '1' and self.rewarded_events_string[i] == '0' and i not in IGNORED_EVENT_IDS:
                    self.rewarded_events_string = self.rewarded_events_string[:i] + '1' + self.rewarded_events_string[i+1:]
                    if first_i == -1:
                        first_i = i
            if first_i != -1:
                # update past event ids
                self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
                self.last_10_event_ids[0] = [first_i, self.time]
        return self.rewarded_events_string.count('1') - self.base_event_flags
            # # elite 4 stage
            # elite_four_event_addr_bits = [
            #     [0xD863, 0],  # EVENT START
            #     [0xD863, 1],  # LORELEIS
            #     [0xD863, 6],  # LORELEIS AUTO WALK
            #     [0xD864, 1],  # BRUNOS
            #     [0xD864, 6],  # BRUNOS AUTO WALK
            #     [0xD865, 1],  # AGATHAS
            #     [0xD865, 6],  # AGATHAS AUTO WALK
            #     [0xD866, 1],  # LANCES
            #     [0xD866, 6],  # LANCES AUTO WALK
            # ]
            # ignored_elite_four_events = 0
            # for ab in elite_four_event_addr_bits:
            #     if self.get_event_rewarded_by_address(ab[0], ab[1]):
            #         ignored_elite_four_events += 1
            # return self.rewarded_events_string.count('1') - self.base_event_flags - ignored_elite_four_events
    
    def calculate_event_rewards(self, events_dict, base_reward, reward_increment, reward_multiplier):
        """
        Calculate total rewards for events in a dictionary.

        :param events_dict: Dictionary containing event completion status with associated points.
        :param base_reward: The starting reward for the first event.
        :param reward_increment: How much to increase the reward for each subsequent event.
        :param reward_multiplier: Multiplier to adjust rewards' significance.
        :return: Total reward calculated for all events.
        """
        total_reward = 0
        current_reward = base_reward
        assert isinstance(events_dict, dict), f"Expected dict, got {type(events_dict)}\nvariable={events_dict}"

        for event, points in events_dict.items():
            if points > 0:  # Assuming positive points indicate completion or achievement
                total_reward += current_reward * points * reward_multiplier
                current_reward += reward_increment
        return total_reward
    
    def calculate_event_rewards_detailed(self, events, base_reward, reward_increment, reward_multiplier):
        # This function calculates rewards for each event and returns them as a dictionary
        # Example return format:
        # {'event1': 10, 'event2': 11, 'event3': 12, ...}
        detailed_rewards = {}
        for event_name, event_value in events.items():
            if event_value > 0:
                detailed_rewards[event_name] = base_reward + (event_value * reward_increment * reward_multiplier)
            else:
                detailed_rewards[event_name] = (event_value * reward_increment * reward_multiplier)
        return detailed_rewards
    
    def update_map_id_to_furthest_visited(self):
        # Define the ordered list of map IDs from earliest to latest
        map_ids_ordered = [1, 2, 15, 3, 5, 21, 4, 6, 10, 7, 8, 9]        
        # Obtain the current map ID (map_n) of the player
        _, _, map_n = ram_map.position(self.game)        
        # Check if the current map ID is in the list of specified map IDs
        if map_n in map_ids_ordered:
            # Find the index of the current map ID in the list
            current_index = map_ids_ordered.index(map_n)            
            # Select the furthest (latest) visited map ID from the list
            # This is done by slicing the list up to the current index + 1
            # and selecting the last element, ensuring we prioritize later IDs
            furthest_visited_map_id = map_ids_ordered[:current_index + 1][-1]            
            # Update the map ID to the furthest visited
            ram_map.write_mem(self.game, 0xd719, furthest_visited_map_id)
            # print(f"env_id {self.env_id}: Updated map ID to the furthest visited: {furthest_visited_map_id}")

    def in_coord_range(self, coord, ranges):
        """Utility function to check if a coordinate is within a given range."""
        r, c = coord
        if isinstance(ranges[0], tuple):  # Check if range is a tuple of ranges
            return any(r >= range_[0] and r <= range_[1] for range_ in ranges[0]) and \
                any(c >= range_[0] and c <= range_[1] for range_ in ranges[1])
        return r == ranges[0] and c == ranges[1]

    def check_bag_items(self, current_bag_items):
        if 'Lemonade' in current_bag_items:
            self.has_lemonade_in_bag = True
            self.has_lemonade_in_bag_reward = 20
        if 'Fresh Water' in current_bag_items:
            self.has_fresh_water_in_bag = True
            self.has_fresh_water_in_bag_reward = 20
        if 'Soda Pop' in current_bag_items:
            self.has_soda_pop_in_bag = True
            self.has_soda_pop_in_bag_reward = 20
        if 'Silph Scope' in current_bag_items:
            self.has_silph_scope_in_bag = True
            self.has_silph_scope_in_bag_reward = 20
        if 'Lift Key' in current_bag_items:
            self.has_lift_key_in_bag = True
            self.has_lift_key_in_bag_reward = 20
        if 'Poke Doll' in current_bag_items:
            self.has_pokedoll_in_bag = True
            self.has_pokedoll_in_bag_reward = 20
        if 'Bicycle' in current_bag_items:
            self.has_bicycle_in_bag = True
            self.has_bicycle_in_bag_reward = 20
        
    def is_new_map(self, r, c, map_n):
        self.update_heat_map(r, c, map_n)
        if map_n != self.prev_map_n:
            self.used_cut_on_map_n = 0
            self.prev_map_n = map_n
            if map_n not in self.seen_maps:
                self.seen_maps.add(map_n)
            # self.save_state()
        
    def get_level_reward(self):
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 50: # 30
            self.level_reward = 1 * self.max_level_sum
        else:
            self.level_reward = 50 + (self.max_level_sum - 50) / 4 # 30
        
    def get_healing_and_death_reward(self):
        party_size, party_levels = ram_map.party(self.game)
        hp = ram_map.read_hp_fraction(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size
        if hp_delta > 0.2 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.death_count_per_episode += 1
            self.is_dead = True
        elif hp > 0.01:  # TODO: Check if this matters
            self.is_dead = False
        self.last_hp = hp
        self.last_party_size = party_size
        self.death_reward = 0
        self.healing_reward = self.total_healing   
    
    @property
    def badges(self):
        return ram_map.read_m("wObtainedBadges").bit_count()
    
    def get_badges_reward(self):
        badge_bonus = self.badges if self.badges >= 2 else 1
        badge_2_bonus = 2 if self.badges == 2 else 1
        self.badges_reward = 10 * self.badges * badge_bonus * badge_2_bonus
        self.badges_reward = 10 * self.badges  # 5 BET


    # def tree_distance_reward_increase(self):
    #     r, c, map_n = ram_map.position(self.game)
    #     if self.cut == 1:
    #         # if r == 0x1F and c == 0x23 and map_n == 0x06:
    #         if (0x1F, 0x23, 0x06) not in self.seen_coords: # hasn't cut Gym 4 tree yet
    #                 self.tree_distance_reward += 0.01
    #                 self.tdri_6_true = True
    #         if r == 0x11 and c == 0x0F and map_n == 0x05:
    #             if (r, c, map_n) not in self.seen_coords:
    #                 self.expl += 5
    #         if r == 0x12 and c == 0x0E and map_n == 0x05:
    #             if (r, c, map_n) not in self.seen_coords:
    #                 self.expl += 5
    
    def get_bill_reward(self):
        self.bill_state = ram_map.saved_bill(self.game)
        self.bill_reward = 2.5 * self.bill_state if self.badges >= 2 else 0 # 5
        
    def get_bill_capt_reward(self):
        self.bill_capt_reward = ram_map.bill_capt(self.game) if self.badges >= 2 else 0

    def get_rubbed_captains_back_reward(self):
        self.rubbed_captains_back = int(ram_map.read_bit(self.game, 0xD803, 1))
        self.rubbed_captains_back_reward = 5 * self.rubbed_captains_back
    
    def got_bill_but_not_badge_2(self):
        if self.badges >= 1:
            if self.bill_state and not self.badges >= 2:
                return 1
            else:
                return 0
        else:
            return 0
    
    @property
    def taught_cut_reward(self):
        if self.cut != 0:
            self._taught_cut_reward = 10
            return self._taught_cut_reward
    
    def get_hm_reward(self):
        hm_count = ram_map.get_hm_count(self.game)
        if hm_count >= 1 and self.hm_count == 0:
            # self.save_state()
            self.hm_count = 1
        self.hm_reward = hm_count * 20 # 10

    def get_cut_reward(self):
        r, c, map_n = ram_map.position(self.game)
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        if self.used_cut_this_reset == 1: # only reward first cut per reset
            if map_n in [5, 6] and self.badges < 3:
                # don't reward a cut if agent already inside gym 3 enclosure
                if glob_r > 282: # global y coord of vermilion tree; is inside the cut enclosure
                    self.cut_reward = 0
                else:
                    self.cut_reward = self.used_cut * 10
                
            elif map_n == 2 and self.badges > 2:
                self.cut_reward = self.used_cut * 10  # 10 works - 2 might be better, though
            else:
                self.cut_reward = 0
        else:
            self.cut_reward = 0

    def get_tree_distance_reward(self, r, c, map_n):
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        if glob_r < 113:
            pass
        elif self.cut < 1:               
            # tree_distance_reward = tree_distance_reward / 10
            # self.tree_distance_reward = tree_distance_reward
            self.tree_distance_reward = 0
        else:
            self.tree_distance_reward = self.detect_and_reward_trees((glob_r, glob_c), map_n, vision_range=5)       
            if self.badges >= 3:
                if map_n in [5]:
                    self.tree_distance_reward = 0
            if self.badges >= 4:
                if map_n in [6]:
                    self.tree_distance_reward = 0

    def get_party_size(self):
        party_size, _ = ram_map.party(self.game)
        return party_size
    
    def get_party_levels(self):
        _, levels = ram_map.party(self.game)
        return levels
    
    def get_money(self):
        money = ram_map.money(self.game)
        return money

    def get_opponent_level_reward(self):
        max_opponent_level = max(ram_map.opponent(self.game))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        self.opponent_level_reward = 0.006 * self.max_opponent_level  # previously disabled BET
        return self.opponent_level_reward

    def get_events(self):
        events = ram_map.events(self.game)
        return events
    
    def get_event_reward(self):
        events = ram_map.events(self.game)
        self.max_events = max(self.max_events, events)
        self.event_reward = self.max_events

    def get_dojo_reward(self):
        self.dojo_reward = ram_map_leanke.dojo(self.game)
        self.defeated_fighting_dojo = 1 * int(ram_map.read_bit(self.game, 0xD7B1, 0))
        self.got_hitmonlee = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 6))
        self.got_hitmonchan = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 7))

    def get_hideout_reward(self):
        self.hideout_reward = ram_map_leanke.hideout(self.game)

    def get_silph_co_events_reward(self):
        self.silph_co_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_silph_co_events(self.game),
            base_reward=10, reward_increment=10, reward_multiplier=2)

    def get_dojo_events_reward(self):
        self.dojo_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_dojo_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=3)

    def get_hideout_events_reward(self):
        self.hideout_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_hideout_events(self.game),
            base_reward=10, reward_increment=10, reward_multiplier=3)

    def get_poke_tower_events_reward(self):
        self.poke_tower_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_poke_tower_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)

    def get_gym_events_reward(self):
        self.gym3_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym3_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=5) # BET added increased
        self.gym4_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym4_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=5) # BET added increased
        self.gym5_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym5_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        self.gym6_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym6_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        self.gym7_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym7_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)

    def calculate_menu_rewards(self):
        start_menu = self.seen_start_menu * 0.01
        pokemon_menu = self.seen_pokemon_menu * 0.1
        stats_menu = self.seen_stats_menu * 0.1
        bag_menu = self.seen_bag_menu * 0.1
        self.that_guy_reward = (start_menu + pokemon_menu + stats_menu + bag_menu)
    
    def calculate_cut_coords_and_tiles_rewards(self):
        self.cut_coords_reward = sum(self.cut_coords.values()) * 1.0
        self.cut_tiles_reward = len(self.cut_tiles) * 1.0
        
    def get_used_cut_coords_reward(self):
        return len(self.used_cut_coords_dict) * 0.2
    
    def calculate_seen_caught_pokemon_moves(self):   
        self.seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        self.caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
        self.moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)
    
    def subtract_previous_reward_v1(self, last_reward, reward):
        if self.last_reward is None:
            updated_reward = 0
            updated_last_reward = 0
        else:
            updated_reward = reward - self.last_reward
            updated_last_reward = reward
        return updated_reward, updated_last_reward
    
    def cut_check(self, action):
        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if ram_map.mem_val(self.game, 0xD057) == 0: # is_in_battle if 1
            if self.cut == 1:
                player_direction = ram_map.read_m(self.game,  0xC109)
                x, y, map_id = self.get_game_coords()  # x, y, map_id
                if player_direction == 0:  # down
                    coords = (x, y + 1, map_id)
                if player_direction == 4:
                    coords = (x, y - 1, map_id)
                if player_direction == 8:
                    coords = (x - 1, y, map_id)
                if player_direction == 0xC:
                    coords = (x + 1, y, map_id)
                self.cut_state.append(
                    (
                        ram_map.read_m(self.game,  0xCFC6),
                        ram_map.read_m(self.game,  0xCFCB),
                        ram_map.read_m(self.game,  0xCD6A),
                        ram_map.read_m(self.game,  0xD367),
                        ram_map.read_m(self.game,  0xD125),
                        ram_map.read_m(self.game,  0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 0 # 5 # 10 reward value for cutting a tree successfully
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                if int(ram_map.read_bit(self.game, 0xD803, 0)):
                    if self.check_if_in_start_menu():
                        self.seen_start_menu = 1
                    if self.check_if_in_pokemon_menu():
                        self.seen_pokemon_menu = 1
                    if self.check_if_in_stats_menu():
                        self.seen_stats_menu = 1
                    if self.check_if_in_bag_menu():
                        self.seen_bag_menu = 1
                    if self.check_if_cancel_bag_menu(action):
                        self.seen_cancel_bag_menu = 1

    def BET_cut_check(self):
        # Cut check 2 - BET ADDED: used cut on tree
        # r, c, map_n = ram_map.position(self.game)
        if ram_map.used_cut(self.game) == 61:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write
            self.used_cut += 1
            self.used_cut_this_reset += 1
            self.used_cut_on_map_n += 1

    def celadon_gym_4_tree_reward(self, action):
        (r, c, map_n) = ram_map.position(self.game)
        current_location = (r, c, map_n)
        if current_location == (31, 35, 6):
            if 20 > self.celadon_tree_reward_counter >= 1:
                if action == 0 or action == 6:
                    self.ctr_bonus += 0.05
            self.celadon_tree_reward = 3 + self.ctr_bonus
            self.celadon_tree_reward_counter += 1

    def calculate_average_slope(self, increases):
        slopes = [(increases[i] - increases[i - 1]) for i in range(1, len(increases))]
        return sum(slopes) / len(slopes)
    
    def update_ema(self, reward, current_time):
        if self.last_time == 0:  # First update
            self.ema = reward
        else:
            elapsed_time = current_time - self.last_time
            interval_smoothing = 2 / (elapsed_time + 1)
            self.ema = (reward * interval_smoothing) + (self.ema * (1 - interval_smoothing))
        self.last_time = current_time
        return self.ema
    
    def get_adjusted_reward(self, reward, current_time):
        ema_reward = self.update_ema(reward, current_time)
        elapsed_time_since_reference = current_time - self.reference_time
        # Calculate the projected healthy reward based on average healthy slope
        projected_healthy_reward = self.reference_reward + (self.average_healthy_slope * elapsed_time_since_reference)
        slope_from_reference = (ema_reward - self.reference_reward) / max(elapsed_time_since_reference, 1)
        
        if slope_from_reference > self.unhealthy_slope_threshold:
            # If growth is unhealthy, cap the reward at the projected healthy level
            return min(ema_reward, projected_healthy_reward)
        else:
            # Update reference if growth is healthy
            self.reference_time = current_time
            self.reference_reward = ema_reward
            return ema_reward
    
    def get_exploration_reward(self, map_n):
        r, c, map_n = ram_map.position(self.game)
        if self.steps_since_last_new_location >= self.step_threshold:
            self.bonus_dynamic_reward_multiplier += self.bonus_dynamic_reward_increment
        self.bonus_dynamic_reward = self.bonus_dynamic_reward_multiplier * len(self.dynamic_bonus_expl_seen_coords)    
        if map_n in self.poketower_maps and int(ram_map.read_bit(self.game, 0xD838, 7)) == 0:
            rew = 0
        elif map_n in self.bonus_exploration_reward_maps:
            rew = (0.03 * len(self.seen_coords)) # if self.used_cut < 1 else 0.13 * len(self.seen_coords)
        else:
            rew = (0.02 * len(self.seen_coords)) # if self.used_cut < 1 else 0.1 * len(self.seen_coords)
        self.exploration_reward = rew # + self.bonus_dynamic_reward
        self.exploration_reward += self.shaped_exploration_reward
        self.seen_coords.add((r, c, map_n))
        self.exploration_reward = self.get_adjusted_reward(self.exploration_reward, self.time)
    
    def tentative_new_get_exploration_reward(self):
        r, c, map_n = ram_map.position(self.game)
        if self.steps_since_last_new_location >= self.step_threshold:
            self.bonus_dynamic_reward_multiplier += self.bonus_dynamic_reward_increment
        self.bonus_dynamic_reward = self.bonus_dynamic_reward_multiplier * len(self.dynamic_bonus_expl_seen_coords)    
        if map_n in self.poketower_maps and int(ram_map.read_bit(self.game, 0xD838, 7)) == 0:
            rew = 0
        elif map_n in self.bonus_exploration_reward_maps:
            self.hideout_progress = ram_map_leanke.monitor_hideout_events(self.game)
            self.pokemon_tower_progress = ram_map_leanke.monitor_poke_tower_events(self.game)
            self.silph_progress = ram_map_leanke.monitor_silph_co_events(self.game)
            
            # Objectives on bonus map COMPLETED: disincentivize exploration of: Gym 3 \ Gym 4 \ Rocket Hideout \ Pokemon Tower \ Silph Co
            if (map_n == 92 and self.badges >= 3) or \
                (map_n == 134 and self.badges >= 4) or \
                    (map_n in self.rocket_hideout_maps and self.hideout_progress['beat_rocket_hideout_giovanni'] != 0) or \
                        (map_n in self.pokemon_tower_maps and self.pokemon_tower_progress['rescued_mr_fuji'] != 0) or \
                            (map_n in self.silph_co_maps and self.silph_progress["beat_silph_co_giovanni"] != 0):
                rew = (0.01 * len(self.seen_coords))
            
            # Objectives on bonus map NOT complete: incentivize exploration of: Gym 3 \ Gym 4 \ Rocket Hideout \ Pokemon Tower \ Silph Co
            elif (map_n == 92 and self.badges < 3) or \
                (map_n == 134 and self.badges < 4) or \
                    (map_n in self.rocket_hideout_maps and self.hideout_progress['beat_rocket_hideout_giovanni'] == 0) or \
                        (map_n in self.pokemon_tower_maps and self.pokemon_tower_progress['rescued_mr_fuji'] == 0) or \
                            (map_n in self.silph_co_maps and self.silph_progress["beat_silph_co_giovanni"] == 0):
                rew = (0.03 * len(self.seen_coords))

            # Shouldn't trigger, but it's there in case I missed some states
            else:
                rew = (0.02 * len(self.seen_coords))
        
        else:
            rew = (0.02 * len(self.seen_coords))
            
        self.exploration_reward = rew # + self.bonus_dynamic_reward
        self.exploration_reward += self.shaped_exploration_reward
        self.seen_coords.add((r, c, map_n))
        self.exploration_reward = self.get_adjusted_reward(self.exploration_reward, self.time)
    
    def get_location_shaped_reward(self):
        r, c, map_n = ram_map.position(self.game)
        current_location = (r, c, map_n)
        # print(f'cur_loc={current_location}')
        if map_n in self.fixed_bonus_expl_maps: # map_n eligible for a fixed bonus reward?
            bonus_fixed_increment = self.fixed_bonus_expl_maps[map_n] # fixed additional reward from dict
            self.shaped_exploration_reward = bonus_fixed_increment
            if current_location not in self.dynamic_bonus_expl_seen_coords: # start progress monitoring on eligible map_n 
                # print(f'current_location={current_location}')
                # print(f'self.dynamic_bonus_expl_seen_coords={self.dynamic_bonus_expl_seen_coords}')
                self.dynamic_bonus_expl_seen_coords.add(current_location) # add coords if unseen
                self.steps_since_last_new_location = 0
            else:
                self.steps_since_last_new_location += 1
                # print(f'incremented self.steps_since_last_new_location by 1 ({self.steps_since_last_new_location})')
        else:
            self.shaped_exploration_reward = 0
        
    def get_respawn_reward(self):
        center = ram_map.read_m(self.game, 0xD719)
        self.respawn.add(center)
        self.len_respawn_reward = len(self.respawn)
        return self.len_respawn_reward # * 5
    
    # # For testing
    # def compute_and_print_rewards(event_reward, bill_capt_rew, seen_pokemon_reward, caught_pokemon_reward, moves_obtained_reward, bill_reward, hm_reward, level_reward, death_reward, badges_reward, healing_reward, exploration_reward, cut_rew, that_guy_reward, cut_coords_reward, cut_tiles_reward, tree_distance_reward, dojo_reward, hideout_reward, lemonade_in_bag_reward, silph_scope_in_bag_reward, lift_key_in_bag_reward, pokedoll_in_bag_reward, bicycle_in_bag_reward, special_location_rewards, can_reward, reward_scale):
    #     reward_components = {
    #         "event_reward": event_reward,
    #         "bill_capt_rew": bill_capt_rew,
    #         "seen_pokemon_reward": seen_pokemon_reward,
    #         "caught_pokemon_reward": caught_pokemon_reward,
    #         "moves_obtained_reward": moves_obtained_reward,
    #         "bill_reward": bill_reward,
    #         "hm_reward": hm_reward,
    #         "level_reward": level_reward,
    #         "death_reward": death_reward,
    #         "badges_reward": badges_reward,
    #         "healing_reward": healing_reward,
    #         "exploration_reward": exploration_reward,
    #         "cut_rew": cut_rew,
    #         "that_guy": that_guy_reward / 2,
    #         "cut_coords": cut_coords_reward,
    #         "cut_tiles": cut_tiles_reward,
    #         "tree_distance_reward": tree_distance_reward * 0.6,
    #         "dojo_reward": dojo_reward * 5,
    #         "hideout_reward": hideout_reward * 5,
    #         "lemonade_in_bag_reward": lemonade_in_bag_reward,
    #         "silph_scope_in_bag_reward": silph_scope_in_bag_reward,
    #         "lift_key_in_bag_reward": lift_key_in_bag_reward,
    #         "pokedoll_in_bag_reward": pokedoll_in_bag_reward,
    #         "bicycle_in_bag_reward": bicycle_in_bag_reward,
    #         "special_location_rewards": special_location_rewards,
    #         "can_reward": can_reward,
    #     }
    #     total_reward = 0
    #     # Print each reward component and its value
    #     for component, value in reward_components.items():
    #         if isinstance(value, (int, float)):
    #             # print(f"{component}: {value}")
    #             total_reward += value
    #     # Apply the reward scale
    #     total_reward *= reward_scale
    #     # print(f"\n\nTotal Reward: {total_reward}")
    #     return total_reward

    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        # BET ADDED
        # # Logic for loading envs based off % milestone achievement
        # # Decision to load a state based on "files_to_load"
        # self.assess_milestone_completion_percentage()
        # # Only 1 env should execute the below:
        # if self.env_id == 0:
        #     # Updates shared dict (self.shared_data) with milestones dict for each env
        #     self.assess_milestone_completion_percentage()
        #     # Returns average completion percentage for each milestone across all envs
        #     self.overall_progress = self.compute_overall_progress(self.shared_data)
        #     # Compares self.overall_progress and self.milestone_threshold dicts
        #     self.actions_required = self.assess_progress_for_action(self.overall_progress, self.milestone_threshold_dict)
        #     # Assesses states for loading using saved state files/dirs
        #     self.sort_and_assess_envs_for_loading(self.shared_data, self.actions_required, 96)
                    
        # # Only loads a state if env was determined to need loading
        # self.load_state_from_tuple()
                
        self.init_caches()
        assert len(self.all_events_string) == 2552, f'len(self.all_events_string): {len(self.all_events_string)}'
        self.rewarded_events_string = '0' * 2552
        self.base_event_flags = self.get_base_event_flags()
        
        if self.reset_count == 0:
            load_pyboy_state(self.game, self.load_first_state())

        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        # if self.use_screen_memory:
        #     self.screen_memory = defaultdict(
        #         lambda: np.zeros((255, 255, 1), dtype=np.uint8)
        #     )

        self.reset_count += 1
        self.time = 0
        
        self.max_episode_steps = max_episode_steps if self.badges <3 else 40960
        
        
        self.reward_scale = reward_scale
        self.last_reward = None

        self.prev_map_n = None
        self.init_hidden_obj_mem()
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.death_count_per_episode = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.hm_count = 0
        self.cut = 0
        self.used_cut = 0 # don't reset, for tracking
        self.cut_coords = {}
        self.cut_tiles = {}
        self.cut_state = deque(maxlen=3)
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_cancel_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        self.town = 1
        
        self.seen_coords_no_reward = set()
        self._all_events_string = ''
        self.agent_stats = []
        self.base_explore = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.party_level_base = 0
        self.party_level_post = 0
        self.last_num_mon_in_box = 0
        self.death_count = 0
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
        self.step_count = 0
        self.past_rewards = np.zeros(10240, dtype=np.float32)
        self.rewarded_events_string = '0' * 2552
        self.seen_map_dict = {}
        self._last_item_count = 0
        self._is_box_mon_higher_level = False
        self.secret_switch_states = {}
        self.hideout_elevator_maps = []
        self.use_mart_count = 0
        self.use_pc_swap_count = 0
        self.total_reward = 0
        self.rewarded_coords = set()
        self.museum_punishment = deque(maxlen=10)
        # self.rewarded_distances = {}
        
        # BET ADDED A BUNCH
        self._cut_badge = False
        self._have_hm01 = False
        self._can_use_cut = False
        self._surf_badge = False
        self._have_hm03 = False
        self._can_use_surf = False
        self._have_pokeflute = False
        self._have_silph_scope = False
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()

        self.used_cut_coords_dict = {}
        
        # BET ADDED TESTING TODAY
        self.shaped_hideout_reward = 0
        self.shaped_map_reward = 0
        
        self.l_seen_coords = set()
        self.expl = 0
        self.celadon_tree_reward = 0
        self.celadon_tree_reward_counter = 0
        self.bonus_dynamic_reward_multiplier = self.initial_multiplier_value  # Reset the dynamic bonus multiplier
        self.steps_since_last_new_location = 0  # Reset step counter for new locations
        self.exploration_reward = 0
        
        # ADDED 5/4/24
        self.can_reward = 0
        self.last_can_mem_val = 0
        self.bet_fixed_window_init()
        self.used_cut_this_reset = 0 # how many tree cuts were performed this reset
        
        # ADDED 5/5/24
        self._taught_cut_reward = 0
        
        # ADDED 5/6/24
        self.rubbed_captains_back = 0
        self.rubbed_captains_back_reward = 0
        self.blackout_count = 0

        return self.render(), {}

    def step(self, action, fast_video=True):
        
        run_action_on_emulator(self.game, action)
        self.time += 1
        
        if self.save_video:
            self.add_video_frame()
            
        # if self.time % 10 == 0:
        #     self.save_all_states_v3()
            
        # Get local coordinate position and map_n
        r, c, map_n = ram_map.position(self.game)
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        self.bet_seen_coords.add((glob_r, glob_c))


        # if map_n == 15:
        #     self.completed_milestones.append(15)
        # Celadon tree reward
        # print(f'action={action}')
        self.celadon_gym_4_tree_reward(action)
        
        # BET added 5/4/24
        self.reset_rewarded_distances_for_celadon()
        self.reset_rewarded_distances_for_cerulean()
        self.reset_rewarded_distances_for_vermilion()
        
        # Call nimixx api
        self.api.process_game_states() 
        current_bag_items = self.api.items.get_bag_item_ids()
        self.update_cut_badge()
        self.update_surf_badge()
        # self.update_last_10_map_ids() # not used currently
        # self.update_last_10_coords() # not used currently
        # self.update_seen_map_dict() # not used currently
       
        # BET ADDED COOL NEW FUNCTIONS SECTION
        # Standard rewards
        self.is_new_map(r, c, map_n)
        self.check_bag_items(current_bag_items)
        self.get_location_shaped_reward()         # Calculate the dynamic bonus reward for this step
        # self.get_exploration_reward(map_n)
        self.tentative_new_get_exploration_reward()
        self.get_level_reward()
        self.get_healing_and_death_reward()
        self.get_badges_reward()
        self.get_bill_reward()
        self.get_bill_capt_reward()
        self.get_hm_reward()
        self.get_cut_reward()
        self.get_tree_distance_reward(r, c, map_n)
        self.visited_maps() 
        self.get_respawn_reward()
        self.get_money()
        self.get_opponent_level_reward()
        self.get_event_reward()
        
        # Special location rewards
        self.get_dojo_reward()
        self.get_hideout_reward()
        self.get_silph_co_events_reward()
        self.get_dojo_events_reward()
        self.get_hideout_events_reward()
        self.get_poke_tower_events_reward()
        self.get_gym_events_reward()
        # Gym 3 for cans that have no switch (to encourage can checking)
        self.get_can_reward()
        self.get_lock_1_use_reward()

        # Cut check
        self.BET_cut_check()
        # self.cut_check(action)

        # Other functions
        self.update_pokedex()
        self.update_moves_obtained()         
        self.count_gym_3_lock_1_use()    
        
        # Calculate some rewards
        self.calculate_menu_rewards()
        # self.calculate_cut_coords_and_tiles_rewards()
        self.calculate_seen_caught_pokemon_moves()
        
        # BET added 5/6/24
        self.get_rubbed_captains_back_reward()

        # # Share data for milestone completion % assessment
        # self.assess_milestone_completion_percentage()
        
        
        # Final reward calculation
        self.calculate_reward()
        reward, self.last_reward = self.subtract_previous_reward_v1(self.last_reward, self.reward)

        info = {}
        done = self.time >= self.max_episode_steps

        if self.save_video and done:
            self.full_frame_writer.close()
        
        # if done:
        #     self.save_state()
        if done or self.time % 5000 == 0:   # 10000
            info = {
                "pokemon_exploration_map": self.counts_map,
                "stats": {
                    "step": self.time,
                    "x": c,
                    "y": r,
                    "map": map_n,
                    # "map_location": self.get_map_location(map_n),
                    # "max_map_progress": self.max_map_progress,
                    "pcount": int(ram_map.read_m(self.game, 0xD163)),
                    "levels": self.get_party_levels(),
                    "levels_sum": sum(self.get_party_levels()),
                    # "ptypes": self.read_party(),
                    # "hp": self.read_hp_fraction(),
                    "coord": np.sum(self.counts_map),  # np.sum(self.seen_global_coords),
                    # "map_id": np.sum(self.seen_map_ids),
                    # "npc": sum(self.seen_npcs.values()),
                    # "hidden_obj": sum(self.seen_hidden_objs.values()),
                    "deaths": self.death_count,
                    "deaths_per_episode": self.death_count_per_episode,
                    "badges": float(self.badges),
                    "self.badge_count": self.badge_count,
                    "badge_1": float(self.badges >= 1),
                    "badge_2": float(self.badges >= 2),
                    "badge_3": float(self.badges >= 3),
                    "badge_4": float(self.badges >= 4),
                    "badge_5": float(self.badges >= 5),
                    "badge_6": float(self.badges >= 6),
                    "events": len(self.past_events_string),
                    # "action_hist": self.action_hist,
                    # "caught_pokemon": int(sum(self.caught_pokemon)),
                    # "seen_pokemon": int(sum(self.seen_pokemon)),
                    # "moves_obtained": int(sum(self.moves_obtained)),
                    "opponent_level": self.max_opponent_level,
                    "met_bill": int(ram_map.read_bit(self.game, 0xD7F1, 0)),
                    "used_cell_separator_on_bill": int(ram_map.read_bit(self.game, 0xD7F2, 3)),
                    "ss_ticket": int(ram_map.read_bit(self.game, 0xD7F2, 4)),
                    "met_bill_2": int(ram_map.read_bit(self.game, 0xD7F2, 5)),
                    "bill_said_use_cell_separator": int(ram_map.read_bit(self.game, 0xD7F2, 6)),
                    "left_bills_house_after_helping": int(ram_map.read_bit(self.game, 0xD7F2, 7)),
                    "got_hm01": int(ram_map.read_bit(self.game, 0xD803, 0)),
                    "rubbed_captains_back": int(ram_map.read_bit(self.game, 0xD803, 1)),
                    # "taught_cut": int(self.check_if_party_has_cut()),
                    "cut_coords": sum(self.cut_coords.values()),
                    'pcount': int(ram_map.mem_val(self.game, 0xD163)), 
                    # 'visited_pokecenterr': self.get_visited_pokecenter_reward(),
                    # 'rewards': int(self.total_reward) if self.total_reward is not None else 0,
                    "maps_explored": len(self.seen_maps),
                    "party_size": self.get_party_size(),
                    "highest_pokemon_level": max(self.get_party_levels()),
                    "total_party_level": sum(self.get_party_levels()),
                    "deaths": self.death_count,
                    # "ss_anne_obtained": ss_anne_obtained,
                    "event": self.get_events(),
                    "money": self.get_money(),
                    "seen_npcs_count": len(self.seen_npcs),
                    "seen_pokemon": np.sum(self.seen_pokemon),
                    "caught_pokemon": np.sum(self.caught_pokemon),
                    "moves_obtained": np.sum(self.moves_obtained),
                    "hidden_obj_count": len(self.seen_hidden_objs),
                    "bill_saved": self.bill_state,
                    "hm_count": self.hm_count,
                    "cut_taught": self.cut,
                    "maps_explored": np.sum(self.seen_maps),
                    "party_size": self.get_party_size(),
                    "bill_capt": (self.bill_capt_reward/5),
                    'cut_coords': self.cut_coords,
                    'cut_tiles': self.cut_tiles,
                    'bag_menu': self.seen_bag_menu,
                    'stats_menu': self.seen_stats_menu,
                    'pokemon_menu': self.seen_pokemon_menu,
                    'start_menu': self.seen_start_menu,
                    'used_cut': self.used_cut,
                    'state_loaded_instead_of_resetting_in_game': self.state_loaded_instead_of_resetting_in_game,
                    'defeated_fighting_dojo': self.defeated_fighting_dojo,
                    'got_hitmonlee': self.got_hitmonlee,
                    'got_hitmonchan': self.got_hitmonchan,  
                    'self.bonus_dynamic_reward_multiplier': self.bonus_dynamic_reward_multiplier,
                    'self.bonus_dynamic_reward_increment': self.bonus_dynamic_reward_increment,
                    'len(self.dynamic_bonus_expl_seen_coords)': len(self.dynamic_bonus_expl_seen_coords),
                    'len(self.seen_coords)': len(self.seen_coords),
                    'got_bill_but_not_badge_2': self.got_bill_but_not_badge_2(),
                    
                                    
                    "silph_co_events_aggregate": {
                        **ram_map_leanke.monitor_silph_co_events(self.game),
                    },
                    "dojo_events_aggregate": {
                        **ram_map_leanke.monitor_dojo_events(self.game),  
                    },
                    "hideout_events_aggregate": {
                        **ram_map_leanke.monitor_hideout_events(self.game),
                    },
                    "poke_tower_events_aggregate": {
                        **ram_map_leanke.monitor_poke_tower_events(self.game),
                    },
                    "gym_events": {
                            "gym_3_events": { 
                        **ram_map_leanke.monitor_gym3_events(self.game),
                        "gym_3_lock_1_use_count": self.lock_1_use_counter,
                        "interacted_with_a_wrong_can": self.can_reward,
                        }, 
                            "gym_4_events": {  
                        **ram_map_leanke.monitor_gym4_events(self.game),  
                        }, 
                            "gym_5_events": {  
                        **ram_map_leanke.monitor_gym5_events(self.game),
                        },  
                            "gym_6_events": {  
                        **ram_map_leanke.monitor_gym6_events(self.game), 
                        }, 
                            "gym_7_events": {  
                        **ram_map_leanke.monitor_gym7_events(self.game), 
                        },
                    },
                },
                "reward": {
                    "delta": self.reward,
                    "event": self.event_reward,
                    "level": self.level_reward,
                    "opponent_level": self.get_opponent_level_reward(),
                    "death": self.death_reward,
                    "badges": self.badges_reward,
                    "bill_saved_reward": self.bill_reward,
                    "bill_capt": self.bill_capt_reward,
                    "hm_count_reward": self.hm_reward,
                    # "ss_anne_done_reward": ss_anne_state_reward,
                    "healing": self.healing_reward,
                    "exploration": self.exploration_reward,
                    # "explore_npcs_reward": explore_npcs_reward,
                    "seen_pokemon_reward": self.seen_pokemon_reward,
                    "caught_pokemon_reward": self.caught_pokemon_reward,
                    "moves_obtained_reward": self.moves_obtained_reward,
                    # "hidden_obj_count_reward": explore_hidden_objs_reward,
                    # "used_cut_reward": self.cut_reward,
                    # "cut_coords_reward": self.cut_coords_reward,
                    # "cut_tiles_reward": self.cut_tiles_reward,
                    # "used_cut_on_tree": used_cut_on_tree_rew,
                    "tree_distance_reward": self.tree_distance_reward,
                    "dojo_reward_old": self.dojo_reward,
                    # "hideout_reward": self.hideout_reward,
                    "has_lemonade_in_bag_reward": self.has_lemonade_in_bag_reward,
                    "has_fresh_water_in_bag_reward": self.has_fresh_water_in_bag_reward,
                    "has_soda_pop_in_bag_reward": self.has_soda_pop_in_bag_reward,
                    "has_silph_scope_in_bag_reward": self.has_silph_scope_in_bag_reward,
                    "has_lift_key_in_bag_reward": self.has_lift_key_in_bag_reward,
                    "has_pokedoll_in_bag_reward": self.has_pokedoll_in_bag_reward,
                    "has_bicycle_in_bag_reward": self.has_bicycle_in_bag_reward,
                    "respawn_reward": self.len_respawn_reward,
                    "celadon_tree_reward": self.celadon_tree_reward,
                    "self.bonus_dynamic_reward": self.bonus_dynamic_reward,
                    "self.shaped_exploration_reward": self.shaped_exploration_reward,
                    "taught_cut_reward": self.taught_cut_reward,
                    "self.rubbed_captains_back_reward": self.rubbed_captains_back_reward,
                    
                    "special_location_rewards": {
                        "detailed_rewards_silph_co": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_silph_co_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                        "detailed_rewards_dojo": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_dojo_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                        "detailed_rewards_hideout": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_hideout_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                        "detailed_rewards_poke_tower": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_poke_tower_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                        "detailed_rewards_gyms": {
                            "gym_3_detailed_rewards": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_gym3_events(self.game), 
                                base_reward=1.6, reward_increment=2, reward_multiplier=1), # 10
                            "gym_3_lock_1_use_reward": self.lock_1_use_reward,
                            "interacted_with_a_wrong_can_reward": self.can_reward,
                        },
                            "gym_4_detailed_rewards": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_gym4_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                            "gym_5_detailed_rewards": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_gym5_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                            "gym_6_detailed_rewards": {
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_gym6_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                            "gym_7_detailed_rewards": { 
                            **self.calculate_event_rewards_detailed(
                                ram_map_leanke.monitor_gym7_events(self.game), 
                                base_reward=10, reward_increment=2, reward_multiplier=1),
                        },
                        },
                        },
                },

                }
            
                # "pokemon_exploration_map": self.counts_map, # self.explore_map, #  self.counts_map,
                
            # d, total_sum = self.print_dict_items_and_sum(info['reward'])
            # print(f'\ntotal_sum={total_sum}\n')
            
        return self.render(), reward, done, done, info
    
    def calculate_reward(self):
        self.reward = self.reward_scale * (
            + self.event_reward
            + self.bill_capt_reward
            + self.seen_pokemon_reward
            + self.caught_pokemon_reward
            + self.moves_obtained_reward
            + self.bill_reward
            + self.hm_reward
            + self.level_reward
            + self.death_reward
            + self.badges_reward
            + self.healing_reward
            + self.exploration_reward
            + self.cut_reward * 0 # cut script
            + self.cut_coords_reward * 0 # cut script
            + self.cut_tiles_reward * 0 # cut script
            + self.used_cut_reward * 0 # cut script
            + self.that_guy_reward / 2
            + self.tree_distance_reward * 0.6 # 1.2 # * 0.6
            + self.dojo_reward * 5
            + self.hideout_reward * 5
            + self.has_lemonade_in_bag_reward
            + self.has_fresh_water_in_bag_reward
            + self.has_soda_pop_in_bag_reward
            + self.has_silph_scope_in_bag_reward
            + self.has_lift_key_in_bag_reward
            + self.has_pokedoll_in_bag_reward
            + self.has_bicycle_in_bag_reward
            + (self.dojo_events_reward + self.silph_co_events_reward +
            self.hideout_events_reward + self.poke_tower_events_reward +
            self.gym4_events_reward +
            self.gym5_events_reward + self.gym6_events_reward +
            self.gym7_events_reward)
            + (self.gym4_events_reward +
            self.gym5_events_reward + self.gym6_events_reward +
            self.gym7_events_reward)
            + self.gym3_events_reward * 2 if self.badges < 3 else 0
            + self.can_reward if self.badges < 3 else 0
            + self.len_respawn_reward
            + self.lock_1_use_reward if self.badges < 3 else 0
            + self.celadon_tree_reward
        )
    
        # special_location_reward = (self.dojo_events_reward + self.silph_co_events_reward + 
        #        self.hideout_events_reward + self.poke_tower_events_reward + 
        #        self.gym3_events_reward + self.gym4_events_reward + 
        #        self.gym5_events_reward + self.gym6_events_reward + 
        #        self.gym7_events_reward)

        # self.compute_and_print_rewards(self.event_reward, self.bill_capt_reward, self.seen_pokemon_reward, self.caught_pokemon_reward, self.moves_obtained_reward, self.bill_reward, self.hm_reward, self.level_reward, self.death_reward, self.badges_reward, self.healing_reward, self.exploration_reward, self.cut_reward, self.that_guy_reward, self.cut_coords_reward, self.cut_tiles_reward, self.tree_distance_reward, self.dojo_reward, self.hideout_reward, self.has_lemonade_in_bag_reward, self.has_silph_scope_in_bag_reward, self.has_lift_key_in_bag_reward, self.has_pokedoll_in_bag_reward, self.has_bicycle_in_bag_reward, special_location_reward, self.can_reward)
