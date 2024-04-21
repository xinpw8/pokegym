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
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
# from . import ram_map, game_map, data, ram_map_leanke
from pokegym import ram_map, game_map, data, ram_map_leanke

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
from multiprocessing import Manager, Lock, Value
from pokegym.ram_addresses import RamAddress as RAM
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
import warnings
import json

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*env*')



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

# List of tree positions in pixel coordinates
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
    (3984, 4512, 5), # to vermilion city gym
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
GYM3PATH = __file__.rstrip("environment.py") + "at_surge_pre_badge.state"
STATE_PATH = __file__.rstrip("environment.py") + "current_state/"
EXPERIMENTAL_PATH = __file__.rstrip("environment.py") + "cerulean_misty_beaten.state"
# print(f'TREE_POSOTIONS_CONVERTED = {TREE_POSITIONS_GRID_GLOBAL}')
MAPS_WITH_TREES = set(map_n for _, _, map_n in TREE_POSITIONS_PIXELS)
TREE_COUNT_PER_MAP = {6: 2, 134: 3, 13: 5, 1: 2, 5: 1, 36: 1, 20: 1, 21: 4}

# Testing environment w/ no AI
# pokegym.play from pufferlib folder
def play():
    """Creates an environment and plays it"""
    env = Environment(
        rom_path="pokemon_red.gb",
        state_path=EXPERIMENTAL_PATH,
        headless=False,
        disable_input=False,
        sound=False,
        sound_emulated=False,
        verbose=True,
    )

    # Update pokemap visualizer text here
    env = StreamWrapper(env, stream_metadata={"user": "boucybet |BET| test\n"})
    env.reset()
    env.game.set_emulation_speed(0)

    # Display available actions
    print("Available actions:")
    for idx, action in enumerate(ACTIONS):
        print(f"{idx}: {action}")

    # Create a mapping from WindowEvent to action index
    window_event_to_action = {
        "PRESS_ARROW_DOWN": 0,
        "PRESS_ARROW_LEFT": 1,
        "PRESS_ARROW_RIGHT": 2,
        "PRESS_ARROW_UP": 3,
        "PRESS_BUTTON_A": 4,
        "PRESS_BUTTON_B": 5,
        "PRESS_BUTTON_START": 6,
        "PRESS_BUTTON_SELECT": 7,
        # Add more mappings if necessary
    }

    while True:
        # Get input from pyboy's get_input method
        input_events = env.game.get_input()
        env.game.tick()
        env.render()
        if len(input_events) == 0:
            continue
                
        for event in input_events:
            event_str = str(event)
            if event_str in window_event_to_action:
                action_index = window_event_to_action[event_str]
                observation, reward, done, _, info = env.step(
                    action_index, # fast_video=False
                )
                
                # Check for game over
                if done:
                    print(f"{done}")
                    break

                # Additional game logic or information display can go here
                # print(f"new Reward: {reward}\n")
                
import logging
import sys
import threading
    
# Initialize a Manager for shared dict for completion % tracking across all envs
manager = Manager()
# shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data
shared_data = manager.dict()

class Base:
    # Shared counter and lock for creating unique environment IDs
    counter = Value('i', 0)  # Shared counter initialized to 0
    counter_lock = Lock()
    shared_data = multiprocessing.Manager().dict()
    shared_lock = Lock()  # Lock for accessing shared_data
    
    # Initialize a multiprocessing manager for shared resources across processes
    manager = multiprocessing.Manager()
    
    # Shared set for tracking visited maps and a lock for process-safe operations
    shared_set = manager.dict()  # Using dict to simulate a set in Manager
    set_lock = manager.Lock()  # Lock for accessing the shared set

    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None, # None
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        with Base.counter_lock:
            self.env_id = Base.counter.value
            Base.counter.value += 1
            
        print(f'env_id {self.env_id} created.')
            
        self.shared_data = shared_data
        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = EXPERIMENTAL_PATH if self.env_id != 1 else GYM3PATH # STATE_PATH + "Bulbasaur.state"
        # TESTING BET ADDED
        # if self.env_id == 1:
        #     state_path = STATE_PATH + "cut2.state"
            
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        self.screenshot_counter = 0
        self.step_threshold = 20480
        self.rocket_hideout_maps = [199, 200, 201, 202, 203]
        self.poketower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.silph_co_maps = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.routes_9_and_10_and_rock_tunnel = [20, 21, 82, 232]
        self.vermilion_city_gym_map = [92]
        self.bonus_exploration_reward_maps = self.rocket_hideout_maps + self.poketower_maps + self.silph_co_maps + self.vermilion_city_gym_map
        self.leave_location_maps = self.bonus_exploration_reward_maps + [54] + [65] + [92] + [134] + [88]
        
        self.rocket_hideout_reward_shape = [5, 6, 7, 8, 10]
        self.vermilion_city_and_gym_reward_shape = [10]
        self.cut = 0
        self.max_multiplier_cap = 5  # Cap for the dynamic reward multiplier
        self.exploration_reward_cap = 2000  # Cap for the total exploration reward
        self.initial_multiplier_value = 1  # Starting value for the multiplier
        self.seen_coords = set() # self.seen_coords.add((r, c, map_n))
        
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
        self.milestones = {}
        self.completed_milestones = []
        self.current_time = datetime.now()      
        
        # More dynamic rewards stuff
        self.seen_coords_specific_maps = set()
        self.steps_since_last_new_location = 0
        self.recent_frames = []
        self.agent_stats = {}
        self.has_lemonade_in_bag_reward = 0
        
        # Initialization of dir structure
        # New way of getting exp name created by clean_pufferl    
        self.experiments_base_dir = Path.cwd() / 'experiments'
        # print(f'environment.py: experiments_base_dir {self.experiments_base_dir} assigned!')
        # List all subdirectories in the experiments base directory
        experiment_dirs = [d for d in self.experiments_base_dir.iterdir() if d.is_dir()]
        # print(f'environment.py: subdirectories of {self.experiments_base_dir}: {experiment_dirs}')
        # Sort directories by their creation time, latest first
        sorted_experiment_dirs = sorted(experiment_dirs, key=os.path.getmtime, reverse=True)
        # print(f'environment.py: sorting the subdirectories...\n\nSorted by creation time, latest first: {sorted_experiment_dirs}')
        # Take the most recently created directory
        if sorted_experiment_dirs:
            latest_experiment_dir = sorted_experiment_dirs[0]
            # print(f"Most recently created experiment directory: {latest_experiment_dir}")
        else:
            # print("No experiment directories found.")
            latest_experiment_dir = None        
        if latest_experiment_dir:
            # Construct the path to running_experiment.txt within the required_resources directory
            running_experiment_file_path = latest_experiment_dir / "required_resources" / "running_experiment.txt"
            # print(f'Looking for file running_experiment.txt in {running_experiment_file_path}')
            # Check if the file exists and read the experiment name
            if running_experiment_file_path.exists():
                with running_experiment_file_path.open('r') as file:
                    exp_name = file.read().strip()
                # print(f"Experiment name read from file: {exp_name}")
            else:
                # print("running_experiment.txt not found in the latest experiment directory.")
                exp_name = "pokegym"
        else:
            exp_name = None

        self.percent_complete_file_path = latest_experiment_dir / "required_resources" / "percent_complete.txt"
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        self.s_path.mkdir(parents=True, exist_ok=True)
        self.save_state_dir = self.s_path / "save_states"
        self.save_state_dir.mkdir(exist_ok=True)
        self.log_file_dir = self.s_path / "log_files"
        self.log_file_dir.mkdir(exist_ok=True)
        # Define the path to the log file
        self.log_file_path = self.log_file_dir / f"env_{self.env_id}.log"
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file_path.touch(exist_ok=True)
        
        
        # Set up logging
        self.logger = logging.getLogger(f'')
        self.logger.setLevel(logging.INFO)  # Set the base level to debug
        
        # Create file handler which logs even debug messages
        self.fh = logging.FileHandler(self.log_file_path, mode='w')
        self.fh.setLevel(logging.INFO)  # Set the file handler's level

        # Create formatter and add it to the handler
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)

        # Add the handler to the logger
        self.logger.addHandler(self.fh)
        self.logger.propagate = False

        # Example of a log message
        self.logger.info(f"Environment {self.env_id}'s logging is set up.")      

        self.video_path = Path(f'./videos') if self.save_video else None
        if self.video_path is not None:
            self.video_path.mkdir(parents=True, exist_ok=True)
        # self.csv_path = Path(f'./csv')
        # self.csv_path.mkdir(parents=True, exist_ok=True)
        self.reset_count = 0
        self.explore_hidden_obj_weight = 1
        self.pokemon_center_save_states = []
        self.pokecenters = [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 147, 182]
        self.used_cut_on_map_n = 0
        
        # BET ADDED nimixx api
        # Import this class for api
        self.api = Game(self.game)
        
        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2) # 72, 80, 3

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )
            self.obs_size += (4,)
        else:
            self.obs_size += (3,)
        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=self.obs_size
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        
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
        self.badges = 0
        self.len_respawn_reward = 0
        self.final_reward = 0
        self.reward = 0
        self.early_map_list = [0, 1, 12, 51, 2, 54, 59, 14, 59, 60, 61]   
        self.bill_and_ss_anne_maps = [88, 101]
        self.cerulean_gym_map = [65]
        self.advanced_gym_maps = [92, 134, 157, 166, 178] # Vermilion, Celadon, Fuchsia, Cinnabar, Saffron
        self.rocket_hideout_b4f_and_lift_maps = [202, 203]
        self.pokemon_tower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.silph_co_maps = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.rocket_hideout_maps = [199, 200, 201, 202, 203]
        self.poketower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.vermilion_city_gym_map = [92]
    
        self.bonus_exploration_reward_maps = self.rocket_hideout_maps + self.silph_co_maps + self.vermilion_city_gym_map + self.silph_co_maps + self.rocket_hideout_b4f_and_lift_maps + self.vermilion_city_gym_map
        
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
        self.used_states = set()        
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
        
        # State loading
        self.badge_0_states = None
        self.badge_1_states = None
        self.badge_2_states = None        
        self.badge_3_states = None        
        self.badge_4_states = None        
        self.badge_5_states = None        
        self.badge_6_states = None        
        self.badge_7_states = None        
        self.badge_8_states = None        
                
        self.badge_1_states_saved = False
        self.badge_2_states_saved = False
        self.badge_3_states_saved = False
        self.badge_4_states_saved = False
        self.badge_5_states_saved = False
        self.badge_6_states_saved = False
        self.badge_7_states_saved = False
        self.badge_8_states_saved = False
        self.leave_location_active = False
        self.map_40_states_saved = False
        self.map_0_states_saved = False
        self.map_6_states_saved = False
        self.flag = False
        self.badge_completion_percentage = 0
        self.silph_co_completed = False
        self.pokemon_tower_completed = False
        self.rocket_hideout_completed = False
        self.bill_completed = False
        self.badge_1_completed = False
        self.badge_2_completed = False        
        self.badge_3_completed = False
        self.badge_4_completed = False       
        self.completion_list = [self.silph_co_completed,
                                self.pokemon_tower_completed,
                                self.rocket_hideout_completed,
                                self.bill_completed,
                                self.badge_1_completed,
                                self.badge_2_completed,        
                                self.badge_3_completed,
                                self.badge_4_completed]     
        self.last_badge = 0    
        self.completion_data = {}
        
        
        
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

    def render(self):
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.game)
            # Update tile map
            mmap = self.screen_memory[map_n]
            if 0 <= r <= 254 and 0 <= c <= 254:
                mmap[r, c] = 255

            # Downsamples the screen and retrieves a fixed window from mmap,
            # then concatenates along the 3rd-dimensional axis (image channel)
            return np.concatenate(
                (
                    self.screen.screen_ndarray()[::2, ::2],
                    self.get_fixed_window(mmap, r, c, self.observation_space.shape),
                ),
                axis=2,
            )
        else:
            return self.screen.screen_ndarray()[::2, ::2]

    # BET ADDED TREE OBSERVATIONS
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
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}
        
    def video(self):
        video = self.screen.screen_ndarray()
        return video

    def close(self):
        self.game.stop(False)

    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = set()
    
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
        
    def save_state2(self, milestone):
        # Ensure the directory for save states is properly initialized and exists
        datetime_str = "timeless" # datetime.now().strftime("%Y%m%d_%H%M%S")
        state_dir = self.save_state_dir / f'env_id_{self.env_id}'
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file_path = state_dir / f"env_id_{self.env_id}_{datetime_str}_state.pkl"
        pyboy_state_file_path = state_dir / f"env_id_{self.env_id}_{datetime_str}_state.state"

        # Collect and serialize the environment's state
        state = {}
        for key, value in vars(self).items():
            try:
                dill.dumps(value)  # Test if the value is serializable
                state[key] = value
                logging.info(f"Successfully serialized: {key}")
            except Exception as e:
                logging.warning(f"Could not serialize {key}: {e}")

        # Attempt to save the serialized state to a file
        try:
            with open(state_file_path, 'wb') as f:
                dill.dump(state, f)
            logging.info(f"Environment state saved to {state_file_path}")
        except Exception as e:
            logging.error(f"Failed to save environment state: {e}")

        # Attempt to save the PyBoy emulator state to a file
        try:
            with open(pyboy_state_file_path, 'wb') as f:
                self.game.save_state(f)
            logging.info(f"PyBoy state saved to {pyboy_state_file_path}")
        except Exception as e:
            logging.error(f"Failed to save PyBoy state: {e}")

        print(f"Env {self.env_id} state saved for milestone {milestone} at {state_file_path} and {pyboy_state_file_path}")


class Environment(Base):
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
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
        self.seen_npcs = set()
        self.explore_npc_weight = 1
        self.is_dead = False
        self.last_map = -1
        self.log = True
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.talk_to_npc_reward = 0
        self.talk_to_npc_count = {}
        self.already_got_npc_reward = set()
        self.ss_anne_state = False
        self.seen_npcs = set()
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
        self.bonus_dynamic_reward_increment = 0.003 # 0.0003
        self.bonus_dynamic_reward_multiplier = 0.001 # 0.0001
        
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
        self.quick_loader_active = False
        
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
        self.badge_2_checkpoint = 0
        self.saved_bill_checkpoint = 0    
        self.silph_co_events_status = []
        
        
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
            caught_mem = self.game.get_memory_value(i + 0xD2F7)
            seen_mem = self.game.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   
    
    def update_moves_obtained(self):
        try:
            # Scan party and update self.cut if cut is obtained
            for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
                if self.game.get_memory_value(i) != 0:
                    for j in range(4):
                        move_id = self.game.get_memory_value(i + j + 8)
                        if move_id != 0:
                            if move_id != 0:
                                self.moves_obtained[move_id] = 1
                            if move_id == 15:
                                self.cut = 1
            # Scan current box (since the box doesn't auto increment in pokemon red)
            num_moves = 4
            box_struct_length = 25 * num_moves * 2
            for i in range(self.game.get_memory_value(0xda80)):
                offset = i*box_struct_length + 0xda96
                if self.game.get_memory_value(offset) != 0:
                    for j in range(4):
                        move_id = self.game.get_memory_value(offset + j + 8)
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
        except Exception as e:
            print(f'ENV_{self.env_id}: update_moves_obtained() failed!! Here is why: {e}')
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.game.get_memory_value(first_item + i)
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
            
    def check_bag_for_silph_scope(self):
        if 0x4A in self.get_items_in_bag():
            if 0x48 in self.get_items_in_bag():
                self.have_silph_scope = True
                return self.have_silph_scope

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
    
    @property
    def is_in_elite_4(self):
        return self.current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]
    
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    ram_map.bit_count(self.game.get_memory_value(i))
                    for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(ram_map.read_bit(self.game, *ram_map.MUSEUM_TICKET_ADDR)),
            0,
        )

    def get_badges_reward(self):
        num_badges = self.get_badges()
        # if num_badges < 3:
        #     return num_badges * 5
        # elif num_badges > 2:
        #     return 10 + ((num_badges - 2) * 10)  # reduced from 20 to 10
        if num_badges < 9:
            return num_badges * 5
        elif num_badges < 13:  # env19v2 PPO23
            return 40  # + ((num_badges - 8) * 1)
        else:
            return 40 + 10
        # return num_badges * 5  # env18v4

    def get_used_cut_coords_reward(self):
        return len(self.used_cut_coords_dict) * 0.2
    
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

    def get_early_done_reward(self):
        return self.elite_4_early_done * -0.3
    
    def get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        return len(self.visited_pokecenter_list) * 2     
    
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
        self.lock_1_use_reward = self.lock_1_use_counter * 10

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
    
    # Pokecenter stuff (not used)
    # Pokecenter stuff (not used)
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
            self.has_lemonade_in_bag_reward = 10
        if 'Fresh Water' in current_bag_items:
            self.has_fresh_water_in_bag = True
            self.has_fresh_water_in_bag_reward = 10
        if 'Soda Pop' in current_bag_items:
            self.has_soda_pop_in_bag = True
            self.has_soda_pop_in_bag_reward = 10
        if 'Silph Scope' in current_bag_items:
            self.has_silph_scope_in_bag = True
            self.has_silph_scope_in_bag_reward = 20
        if 'Lift Key' in current_bag_items:
            self.has_lift_key_in_bag = True
            self.has_lift_key_in_bag_reward = 20 # 10
        if 'Poke Doll' in current_bag_items:
            self.has_pokedoll_in_bag = True
            self.has_pokedoll_in_bag_reward = 3 # 10
        if 'Bicycle' in current_bag_items:
            self.has_bicycle_in_bag = True
            self.has_bicycle_in_bag_reward = 1 # 2
        
    def is_new_map(self, r, c, map_n):
        self.update_heat_map(r, c, map_n)
        if map_n != self.prev_map_n:
            self.used_cut_on_map_n = 0
            self.prev_map_n = map_n
            # self.add_map(map_n)
            if map_n not in self.seen_maps:
                # self.save_all_states_v3()
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
        hp = ram_map.hp(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size
        if hp_delta > 0.2 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.death_count_per_episode += 1
            self.is_dead = True
        elif hp > 0.01:
            self.is_dead = False
        self.last_hp = hp
        self.last_party_size = party_size
        self.death_reward = 0
        self.healing_reward = self.total_healing   
        
    def get_badges_for_swarm(self):
        self.badges = ram_map.badges(self.game)
        badge_thresholds = [1, 2, 3, 4, 5, 6, 7, 8]
        saved_flags = [
            'badge_1_states_saved', 'badge_2_states_saved', 'badge_3_states_saved',
            'badge_4_states_saved', 'badge_5_states_saved', 'badge_6_states_saved',
            'badge_7_states_saved', 'badge_8_states_saved'
        ]
        for threshold, flag in zip(badge_thresholds, saved_flags):
            if 0 < self.badges <= threshold and not getattr(self, flag):
                setattr(self, flag, True)
                saved_state = self.save_all_states_v3()
                setattr(self, f'badge_{threshold}_states', saved_state)
                
    # Returns True if a new badge was gotten in the current step
    def check_for_new_badge(self):
        self.badges = ram_map.badges(self.game)
        if self.badges > self.last_badge and self.reset_count > 3:
            try:
                print(f'ENV_{self.env_id}, check_for_new_badge: Triggered new badge condition!! self.badges={self.badges}, self.last_badge={self.last_badge}')
                logging.critical(f'ENV_{self.env_id}, check_for_new_badge: Triggered new badge condition!! self.badges={self.badges}, self.last_badge={self.last_badge}')
                saved_state = self.save_all_states_v3()
                setattr(self, f'badge_{self.badges}_states', saved_state)
                try:
                    delattr(self, f'badge_{self.last_badge}_states')
                except AttributeError as e:
                    print(f"ENV_{self.env_id}, check_for_new_badge: Attempted to delete non-existent attribute for badge {self.last_badge}: {e}")
                    logging.critical(f"ENV_{self.env_id}, check_for_new_badge: Attempted to delete non-existent attribute for badge {self.last_badge}: {e}")
                next_badge = self.badges
                self.last_badge = next_badge
            except Exception as e:
                print(f"ENV_{self.env_id}, check_for_new_badge: FAILED TO SAVE STATE {self.badges}, {self.last_badge}: {e}")

            return True
        else:
            return False
        
    # For testing purposes BET ADDED
    def get_map_for_swarm(self):
        r, c, map_n = ram_map.position(self.game)
        if map_n == 6:
            map_thresholds = [6]
            saved_flags = [
                'map_6_states_saved'
            ]
            for threshold, flag in zip(map_thresholds, saved_flags):
                if map_n == threshold and not getattr(self, flag):
                    setattr(self, flag, True)
                    print(f'ENV_{self.env_id} flag={flag}')
                    saved_state = self.save_all_states_v3()
                    setattr(self, f'badge_{threshold}_states', saved_state)

    def get_badges_reward(self):
        self.badges = ram_map.badges(self.game)
        self.badges_reward = 10 * self.badges  # 5 BET

    def get_bill_reward(self):
        self.bill_state = ram_map.saved_bill(self.game)
        self.bill_reward = 5 * self.bill_state
        
    def get_bill_capt_reward(self):
        self.bill_capt_reward = ram_map.bill_capt(self.game)

    def get_hm_reward(self):
        hm_count = ram_map.get_hm_count(self.game)
        if hm_count >= 1 and self.hm_count == 0:
            # self.save_state()
            self.hm_count = 1
        self.hm_reward = hm_count * 10

    def get_cut_reward(self):
        self.cut_reward = self.cut * 10 # 8  # 10 works - 2 might be better, though

    def get_tree_distance_reward(self, r, c, map_n):
        glob_r, glob_c = game_map.local_to_global(r, c, map_n) 
        tree_distance_reward = self.detect_and_reward_trees((glob_r, glob_c), map_n, vision_range=5)
        if self.cut < 1:
            tree_distance_reward = tree_distance_reward / 10
        self.tree_distance_reward = tree_distance_reward

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
            base_reward=10, reward_increment=2, reward_multiplier=1)
        self.gym4_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym4_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
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
                player_direction = self.game.get_memory_value(0xC109)
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
                        self.game.get_memory_value(0xCFC6),
                        self.game.get_memory_value(0xCFCB),
                        self.game.get_memory_value(0xCD6A),
                        self.game.get_memory_value(0xD367),
                        self.game.get_memory_value(0xD125),
                        self.game.get_memory_value(0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 5 # 10 reward value for cutting a tree successfully
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

    def calculate_average_slope(self, increases):
        slopes = [(increases[i] - increases[i - 1]) for i in range(1, len(increases))]
        return sum(slopes) / len(slopes)
    
    # Gives slight negative rewards to envs that have completed the conditionals    
    # Currently a work-in-progress    
    # Exploration reward increments for certain maps when progress is not made.
    # This makes agents not want to leave those maps.
    # Once the progress is made, exploration reward has to be higher elsewhere
    # to incentivize leaving.    
    def leave_location_after_quest_complete(self):
        r, c, map_n = ram_map.position(self.game)
        completed = False  # Default to not completed
        # Check specific quests based on location and update milestone completion status
        # After loading swarm, should be True
        # Also True if any previously completed (to prevent returning to map in future)
        if any(self.completion_list):
            # Gym 1
            if map_n == 54 and self.badges >= 1 and self.completion_data["badge_1"] >= 90:  # Assuming the criteria is at least 90%
                completed = True
                self.badge_1_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 54 (actual={map_n}) and self.badges >= 1 (actual={self.badges} and self.completion_data["badge_1"] >= 90 (actual={self.completion_data["badge_1"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 54 (actual={map_n}) and self.badges >= 1 (actual={self.badges} and self.completion_data["badge_1"] >= 90 (actual={self.completion_data["badge_1"]}) True)')

            # Gym 2
            elif map_n == 65 and self.badges >= 2 and self.completion_data["badge_2"] >= 80:
                completed = True
                self.badge_2_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 65 (actual={map_n}) and self.badges >= 2 (actual={self.badges} and self.completion_data["badge_2"] >= 80 (actual={self.completion_data["badge_2"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 65 (actual={map_n}) and self.badges >= 2 (actual={self.badges} and self.completion_data["badge_2"] >= 80 (actual={self.completion_data["badge_2"]}) True)')

            # Gym3
            elif map_n == 92 and self.badges >= 3 and self.completion_data["badge_3"] >= 15:
                completed = True
                self.badge_3_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 92 (actual={map_n}) and self.badges >= 3 (actual={self.badges} and self.completion_data["badge_3"] >= 15 (actual={self.completion_data["badge_3"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 92 (actual={map_n}) and self.badges >= 3 (actual={self.badges} and self.completion_data["badge_3"] >= 15 (actual={self.completion_data["badge_3"]}) True)')

            # Gym 4
            elif map_n == 134 and self.badges >= 4 and self.completion_data["badge_4"] >= 3:
                completed = True
                self.badge_4_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 134 (actual={map_n}) and self.badges >= 4 (actual={self.badges} and self.completion_data["badge_4"] >= 3 (actual={self.completion_data["badge_4"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 134 (actual={map_n}) and self.badges >= 4 (actual={self.badges} and self.completion_data["badge_4"] >= 3 (actual={self.completion_data["badge_4"]}) True)')

            # Bill's House
            elif map_n == 88 and ram_map.read_bit(self.game, 0xD7F2, 7) == 1 and self.completion_data["bill_saved"] >= 70:  # Bill's quest completed
                completed = True
                self.bill_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 88 (actual={map_n}) and ram_map.read_bit(self.game, 0xD7F2, 7) == 1 (actual={ram_map.read_bit(self.game, 0xD7F2, 7)} and self.completion_data["saved_bill"] >= 70 (actual={self.completion_data["saved_bill"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n == 88 (actual={map_n}) and ram_map.read_bit(self.game, 0xD7F2, 7) == 1 (actual={ram_map.read_bit(self.game, 0xD7F2, 7)} and self.completion_data["saved_bill"] >= 70 (actual={self.completion_data["saved_bill"]}) True)')

            # Rocket Hideout
            elif map_n in self.rocket_hideout_maps and self.completion_data["beat_rocket_hideout_giovanni"] >= 40:
                self.hideout_progress = ram_map_leanke.monitor_hideout_events(self.game)
                if self.hideout_progress['beat_rocket_hideout_giovanni'] != 0:
                    completed = True
                    self.rocket_hideout_completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.rocket_hideout_maps (actual={map_n in self.rocket_hideout_maps}) and self.completion_data["beat_rocket_hideout_giovanni"] >= 40 (actual={self.completion_data["beat_rocket_hideout_giovanni"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.rocket_hideout_maps (actual={map_n in self.rocket_hideout_maps}) and self.completion_data["beat_rocket_hideout_giovanni"] >= 40 (actual={self.completion_data["beat_rocket_hideout_giovanni"]}) True)')
                    
            # Pokemon Tower
            elif map_n in self.pokemon_tower_maps and self.completion_data["rescued_mr_fuji"] >= 20:
                self.pokemon_tower_progress = ram_map_leanke.monitor_poke_tower_events(self.game)
                if self.pokemon_tower_progress['rescued_mr_fuji'] != 0 or self.has_silph_scope_in_bag == True:
                    completed = True
                    self.pokemon_tower_completed = True              
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.pokemon_tower_maps (actual={map_n in self.pokemon_tower_maps}) and self.completion_data["rescued_mr_fuji"] >= 20 (actual={self.completion_data["rescued_mr_fuji"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.pokemon_tower_maps (actual={map_n in self.pokemon_tower_maps}) and self.completion_data["rescued_mr_fuji"] >= 20 (actual={self.completion_data["rescued_mr_fuji"]}) True)')
      
            elif map_n in self.silph_co_maps and self.completion_data["beat_silph_co_giovanni"] >= 10:
                self.silph_progress = ram_map_leanke.monitor_silph_co_events(self.game)
                if self.silph_progress["beat_silph_co_giovanni"] != 0:
                    completed = True
                print(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.silph_co_maps (actual={map_n in self.silph_co_maps}) and self.completion_data["beat_silph_co_giovanni"] >= 10 (actual={self.completion_data["beat_silph_co_giovanni"]}) True)')
                logging.critical(f'ENV_{self.env_id} leave_location_after_quest_complete if map_n in self.silph_co_maps (actual={map_n in self.silph_co_maps}) and self.completion_data["beat_silph_co_giovanni"] >= 10 (actual={self.completion_data["beat_silph_co_giovanni"]}) True)')
            # Adjust exploration rewards based on the completion status
            if completed:
                # Reset exploration reward to slightly negative to incentivize leaving the area
                self.exploration_reward = -0.01
                if not self.alert_printed_this_reset:
                    print(f"Quest completed at {map_n}, negative exploration reward for env_{self.env_id} applied.")
                    logging.warning(f"Quest completed at {map_n}, negative exploration reward for env_{self.env_id} applied.")
                    self.alert_printed_this_reset = True
        # Ridiculously high rewards will keep envs on the map for the purposes of grouping
        elif map_n in self.leave_location_maps:
            self.exploration_reward += 1 # 0.5
            if not self.alert_printed_this_reset:
                logging.warning(f"leave_location_after_quest_complete primary elif: {map_n}, env_{self.env_id}")
                print(f"leave_location_after_quest_complete primary elif: {map_n}, env_{self.env_id}")
                self.alert_printed_this_reset = True
        else:
            pass
        return self.exploration_reward

    # TO-DO: global progress manager
    def get_exploration_reward(self, map_n):
        r, c, map_n = ram_map.position(self.game)
        # Must have Cut and be on a bonusable map (Gym 3, Gym 4, Hideout, Pokemon Tower, Silph Co)
        # Increments reward positively if no progress is made after 20480 steps
        if self.cut != 0 and map_n in self.bonus_exploration_reward_maps:
            if self.steps_since_last_new_location >= self.step_threshold:
                self.bonus_dynamic_reward_multiplier += self.bonus_dynamic_reward_increment
            self.bonus_dynamic_reward = self.bonus_dynamic_reward_multiplier * len(self.dynamic_bonus_expl_seen_coords)         
        self.silph_progress = ram_map_leanke.monitor_silph_co_events(self.game)
        self.hideout_progress = ram_map_leanke.monitor_hideout_events(self.game)
        # Poketower - only reward if Rocket Hideout is complete
        if map_n in self.poketower_maps:
            if self.hideout_progress['beat_rocket_hideout_giovanni'] == 0:
                reward = 0
            else:
                reward = (0.05 * len(self.seen_coords)) # if self.used_cut < 1 else 0.13 * len(self.seen_coords)
        # Vermilion, Celadon, Fuchsia, Cinnabar, Saffron gym maps - special rewards for these gyms
        elif map_n in self.advanced_gym_maps:
            reward = (0.15 * len(self.seen_coords))
        elif map_n in self.bonus_exploration_reward_maps:
            reward = (0.05 * len(self.seen_coords)) # if self.used_cut < 1 else 0.13 * len(self.seen_coords)
        # Default reward if nowhere special
        else:
            reward = (0.02 * len(self.seen_coords)) if self.used_cut < 1 else 0.1 * len(self.seen_coords)
        # Sum exploration reward computed here
        self.seen_coords.add((r, c, map_n))
        self.shaped_exploration_reward = self.get_location_shaped_reward()
        if self.shaped_exploration_reward != None:
            self.exploration_reward = reward + self.bonus_dynamic_reward + self.shaped_exploration_reward
        else:
            self.exploration_reward = reward + self.bonus_dynamic_reward
        # Rewards smoothing/control to prevent runaway rewards. Currently using leave_map fn to control.
        # self.exploration_reward = self.get_adjusted_reward(self.exploration_reward, self.time)

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
        center = self.game.get_memory_value(0xD719)
        self.respawn.add(center)
        self.len_respawn_reward = len(self.respawn)
        return self.len_respawn_reward # * 5
    
    # For testing
    def compute_and_print_rewards(event_reward, bill_capt_rew, seen_pokemon_reward, caught_pokemon_reward, moves_obtained_reward, bill_reward, hm_reward, level_reward, death_reward, badges_reward, healing_reward, exploration_reward, cut_rew, that_guy_reward, cut_coords_reward, cut_tiles_reward, tree_distance_reward, dojo_reward, hideout_reward, lemonade_in_bag_reward, silph_scope_in_bag_reward, lift_key_in_bag_reward, pokedoll_in_bag_reward, bicycle_in_bag_reward, special_location_rewards, can_reward, reward_scale):
        reward_components = {
            "event_reward": event_reward,
            "bill_capt_rew": bill_capt_rew,
            "seen_pokemon_reward": seen_pokemon_reward,
            "caught_pokemon_reward": caught_pokemon_reward,
            "moves_obtained_reward": moves_obtained_reward,
            "bill_reward": bill_reward,
            "hm_reward": hm_reward,
            "level_reward": level_reward,
            "death_reward": death_reward,
            "badges_reward": badges_reward,
            "healing_reward": healing_reward,
            "exploration_reward": exploration_reward,
            "cut_rew": cut_rew,
            "that_guy": that_guy_reward / 2,
            "cut_coords": cut_coords_reward,
            "cut_tiles": cut_tiles_reward,
            "tree_distance_reward": tree_distance_reward * 0.6,
            "dojo_reward": dojo_reward * 5,
            "hideout_reward": hideout_reward * 5,
            "lemonade_in_bag_reward": lemonade_in_bag_reward,
            "silph_scope_in_bag_reward": silph_scope_in_bag_reward,
            "lift_key_in_bag_reward": lift_key_in_bag_reward,
            "pokedoll_in_bag_reward": pokedoll_in_bag_reward,
            "bicycle_in_bag_reward": bicycle_in_bag_reward,
            "special_location_rewards": special_location_rewards,
            "can_reward": can_reward,
        }
        total_reward = 0
        # Print each reward component and its value
        for component, value in reward_components.items():
            if isinstance(value, (int, float)):
                # print(f"{component}: {value}")
                total_reward += value
        # Apply the reward scale
        total_reward *= reward_scale
        # print(f"\n\nTotal Reward: {total_reward}")
        return total_reward
    
    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})
    
    # Serializes instance vars() and saves pyboy state file
    def save_all_states_v3(self, is_failed=False):
        state = io.BytesIO()
        pkl = io.BytesIO()
        try:
            self.game.save_state(state)
            state.seek(0)
            logging.info(f"ENV_{self.env_id}: Saved PyBoy state in-memory")
        except Exception as e:
            print(f"ENV_{self.env_id}: Failed to save PyBoy state in-memory. Error: {e}")
        picklable_state = {}
        for key, value in vars(self).items():
            try:
                dill.dumps(value)
                picklable_state[key] = value
            except Exception as e:
                logging.warning(f"ENV_{self.env_id}: Failed to pickle {key}: {e}")
        try:
            dill.dump(picklable_state, pkl)
            pkl.seek(0)
            logging.info(f"ENV_{self.env_id}: Saved PICKLE instance state in-memory")
        except Exception as e:
            print(f"ENV_{self.env_id}: Failed to save instance state in-memory. Error: {e}")
        return {"state": {"pyboy": state.read(), "pkl": pkl.read()}}

    def load_state_from_tuple_simple2(self, pkl_memory, state_memory):
        try:
            pkl_memory.seek(0)
            state = dill.load(pkl_memory)
            for key, value in state.items():
                setattr(self, key, value)
            logging.info(f"ENV_{self.env_id}, load_state_from_tuple_simple2: Loaded pkl from memory!!")
        except Exception as e:
            logging.error(f"ENV_{self.env_id}, load_state_from_tuple_simple2: Could not load the pickle because: {e}") 
            print(f"ENV_{self.env_id}, load_state_from_tuple_simple2: Could not load the pickle because: {e}")        
        try:
            state_memory.seek(0)
            self.game.load_state(state_memory)
            logging.info(f"ENV_{self.env_id}: Loaded pyboy state from memory!!")
        except Exception as e:
            logging.error(f"ENV_{self.env_id}, load_state_from_tuple_simple2: Failed to load PyBoy state from memory: {e}")
            print(f"ENV_{self.env_id}, load_state_from_tuple_simple2: Failed to load PyBoy state from memory: {e}")

        # self.reset_count = 0
        self.step_count = 0
        # self.leave_location_active = True
        self.reset_count += 1
        
    def write_badge(self):
        event_value = ram_map.mem_val(self.game, ram_map.BADGE_1_ADDR)  # bit 4
        ram_map.write_mem(self.game, ram_map.BADGE_1_ADDR, ram_map.set_bit(event_value, 3))
        ram_map.write_mem(self.game, RAM.wObtainedBadges.value, 1)
    
    def update_badge_status(self, badge_number):
        """
        Set the badge bit for the given badge number.
        Badge numbers are zero-indexed.
        """
        if badge_number < 0 or badge_number > 7:
            raise ValueError("Badge number must be between 0 and 7.")

        badge_addr = RAM.wObtainedBadges.value  # Assuming this is predefined in your constants
        current_value = self.game.get_memory_value(badge_addr)
        new_value = ram_map.set_bit(current_value, badge_number)
        ram_map.write_mem(self.game, badge_addr, new_value)
        logging.info(f"Badge {badge_number + 1} status updated in memory #1.")
        badge_addr = ram_map.BADGE_1_ADDR
        current_value = self.game.get_memory_value(badge_addr)
        new_value = ram_map.set_bit(current_value, badge_number)
        ram_map.write_mem(self.game, badge_addr, new_value)
        logging.info(f"Badge {badge_number + 1} status updated in memory #2.")

    def load_completion_data(self):
        try:
            with open(self.percent_complete_file_path, 'r') as file:
                data = json.load(file)
            return {key: float(value) for key, value in data.items()}
        except Exception as e:
            print(f'ENV_{self.env_id} load_completion_data: could not read json file d/t {e}!')
            logging.critical(f'ENV_{self.env_id} load_completion_data: could not read json file d/t {e}!')

        
    def reset(self, seed=None, options=None, state=None, max_episode_steps=5120, reward_scale=4):
        options = options or {}
        if options == None:
            print(f'ENV_{self.env_id} reset(): options=None')
            logging.critical(f'ENV_{self.env_id} reset(): options=None')
        else:
            print(f'ENV_{self.env_id} reset(): options.keys()={options.keys()}')
            logging.critical(f'ENV_{self.env_id} reset(): options.keys()={options.keys()}')
            
        if state == None:
            print(f'ENV_{self.env_id} reset(): state=None')
            logging.critical(f'ENV_{self.env_id} reset(): state=None')
        else:
            print(f'ENV_{self.env_id} reset(): state.keys()={state}')
            logging.critical(f'ENV_{self.env_id} reset(): state.keys()={state}')
            
            
        print(f'RESET COUNT={self.reset_count}')
        logging.info(f'RESET COUNT={self.reset_count}')
        
        
        if self.reset_count == 0:
            print(f'ENV_{self.env_id}, reset(): Loading PyBoy file self.load_first_state()')
            logging.info(f'ENV_{self.env_id}, reset(): Loading PyBoy file self.load_first_state()')
            load_pyboy_state(self.game, self.load_first_state())
            
        elif options.get("state", None) is not None: # and options.get("pkl", None) is not None:
        # Load the state from memory if provided through options
            logging.info(f'reset(): RESET number {self.reset_count}, env_{self.env_id}: Attempting to load  self.load_state_from_tuple_simple2(io.BytesIO(options["pkl"]), io.BytesIO(options["state"]))')
            print(f'reset(): RESET number {self.reset_count}, env_{self.env_id}: Attempting to load  self.load_state_from_tuple_simple2(io.BytesIO(options["pkl"]), io.BytesIO(options["state"]))')
           
            self.load_state_from_tuple_simple2(io.BytesIO(options["state"][1]), io.BytesIO(options["state"][0]))
            # Always works; here as a backup
            self.game.load_state(io.BytesIO(options["state"][0]))
            
        elif options.get("state", None) is None:
            pass
            logging.info(f'RESET {self.reset_count}, env_{self.env_id}: options.get("state", None) is None!')          
            print(f'RESET {self.reset_count}, env_{self.env_id}: options.get("state", None) is None!')          
       
        try:
            opts = options.get("state", None)
            self.badge_completion_percentage = opts[2]
            logging.info(f'reset(): RESET number {self.reset_count}, env_{self.env_id}, badge_completion_percentage from clean_pufferl: {self.badge_completion_percentage}')
            print(f'reset(): RESET number {self.reset_count}, env_{self.env_id}, badge_completion_percentage from clean_pufferl: {self.badge_completion_percentage}')
        except:
            logging.info(f'reset(): RESET number {self.reset_count}, env_{self.env_id}, FAILED TO OBTAIN badge_completion_percentage from clean_pufferl')
            pass
        
        # OLD REGULAR 
        # elif options.get("state", None) is not None:
        #     self.game.load_state(io.BytesIO(options["state"]))
        #     self.reset_count += 1

        self.init_caches()
        assert len(self.all_events_string) == 2552, f'len(self.all_events_string): {len(self.all_events_string)}'
        self.rewarded_events_string = '0' * 2552
        self.base_event_flags = self.get_base_event_flags()

        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )

        self.reset_count += 1
        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.last_reward = None

        self.prev_map_n = None
        self.init_hidden_obj_mem()
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set() # self.seen_coords.add((r, c, map_n))
        self.seen_maps = set()
        self.death_count_per_episode = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.hm_count = 0
        self.cut = 0
        # self.used_cut = 0 # don't reset, for tracking
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
        self.rewarded_distances = {}
        
        # BET ADDED
        self._cut_badge = False
        self._have_hm01 = False
        self._can_use_cut = False
        self._surf_badge = False
        self._have_hm03 = False
        self._can_use_surf = False
        self._have_pokeflute = False
        self._have_silph_scope = False
        self.used_cut_coords_dict = {}
        
        # BET ADDED
        self.shaped_hideout_reward = 0
        self.shaped_map_reward = 0
        
        self.l_seen_coords = set()
        self.expl = 0
        self.celadon_tree_reward = 0
        self.celadon_tree_reward_counter = 0
        self.initial_multiplier_value = 1
        self.bonus_dynamic_reward_multiplier = self.initial_multiplier_value  # Reset the dynamic bonus multiplier
        self.steps_since_last_new_location = 0  # Reset step counter for new locations
        self.initial_multiplier_value = 1
        self.exploration_reward = 0        
        self.pokemon_tower_progress = {}
        self.hideout_progress = {}
        self.silph_progress = {}
        self.alert_printed_this_reset = False
        
        # r, c, map_n = ram_map.position(self.game)
        # print(f'env_id={self.env_id}, reset#={self.reset_count}, map_n={map_n}')
        
        # Pass an empty dict if no specific state dict is provided
        if state is None:
            saved_states_dict = {} # self.save_all_states_v3()
        else:
            saved_states_dict = state or {}

        # Check for saved states from badge achievements (saved in step())
        saved_states_keys = [key for key in vars(self) if key.startswith('badge_') and key.endswith('_state')]
        if saved_states_keys:
            print(f'ENV_{self.env_id}: saved_states_keys=True')
            logging.warning(f'reset(): saved_states_keys=True! env_{self.env_id}, reset number {self.reset_count}')
            saved_states_dict = getattr(self, saved_states_keys[0])
            # delattr(self, saved_states_keys[0])  # Remove the attribute after accessing
            return self.render(), saved_states_dict
        else:
            logging.warning(f'PROBLEM LOOKING FOR saved_states_keys in env_{self.env_id}, reset number {self.reset_count}')
                
        # self.first = False
        # state = io.BytesIO()
        # self.game.save_state(state)
        # state.seek(0)
        
        if self.reset_count % 1 == 0 and self.env_id == 1:
            logging.critical(f'triggered save_all_states_v3 env_{self.env_id}, reset_count={self.reset_count}')
            print(f'triggered save_all_states_v3 env_{self.env_id}, reset_count={self.reset_count}')
            saved_states_dict = self.save_all_states_v3()
        
        return self.render(), saved_states_dict # {"state": {"pyboy": state.read(), "pkl": pkl.read()}}
        # return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1

        # if self.env_id == 1:
        #     self.badges = ram_map.badges(self.game)
        #     print(f'env_id=1 ...BEFORE WRITE MEM badges={self.badges}')
        #     logging.critical(f'env_id=1 ...BEFORE WRITE MEM badges={self.badges}')
        #     self.update_badge_status(7)
        #     self.badges = ram_map.badges(self.game)
        #     print(f'env_id=1 ...AFTER WRITE MEM  badges={self.badges}')
        #     logging.critical(f'env_id=1 ...AFTER WRITE MEM  badges={self.badges}')

        if self.time == 0 or self.time % 100 == 0:
            self.completion_data = self.load_completion_data()
            try:
                print(f'self.completion_data (json contents from clean_pufferl): {self.completion_data}')
            except Exception as e:
                print(f'ENV_{self.env_id} step() LINE 2342: could not print completion_data d/t {e}')
            finally:
                if self.completion_data is None:
                    self.completion_data = {
                                            "badge_1": "0.000000",
                                            "badge_2": "0.000000",
                                            "badge_3": "0.000000",
                                            "badge_4": "0.000000",
                                            "badge_5": "0.000000",
                                            "badge_6": "0.000000",
                                            "badge_7": "0.000000",
                                            "badge_8": "0.000000",
                                            "bill_saved": "0.000000",
                                            "beat_rocket_hideout_giovanni": "0.000000",
                                            "rescued_mr_fuji": "0.000000",
                                            "beat_silph_co_giovanni": "0.000000"
                                        }
        
        if self.save_video:
            self.add_video_frame()
            
        # Get local coordinate position and map_n
        r, c, map_n = ram_map.position(self.game)
        # Checks for new badge; calls states save if True
        self.check_for_new_badge()  
        # self.get_badges_for_swarm()
        # self.get_map_for_swarm() # for testing
        self.celadon_gym_4_tree_reward(action)
        
        # Call nimixx api
        self.api.process_game_states()
        current_bag_items = self.api.items.get_bag_item_ids()
        self.update_cut_badge()
        self.update_surf_badge()
       
        # BET ADDED COOL NEW FUNCTIONS SECTION
        # Standard rewards
        self.is_new_map(r, c, map_n)
        self.check_bag_items(current_bag_items)
        self.get_exploration_reward(map_n)
        # if self.leave_location_active:
        #     # logging.info(f'env_{self.env_id}: self.leave_location_active = True!')
        self.leave_location_after_quest_complete()
        self.get_level_reward()
        self.get_healing_and_death_reward()
        self.get_badges_reward()
        self.get_bill_reward()
        self.get_bill_capt_reward()
        self.get_hm_reward()
        self.get_cut_reward()
        
        # Only encourage tree-checking after HM01 is taught
        if self.cut != 0:
            self.get_tree_distance_reward(r, c, map_n)   
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
        self.cut_check(action)

        # Other functions
        self.update_pokedex()
        self.update_moves_obtained()         
        self.count_gym_3_lock_1_use()    
        
        # Calculate some rewards
        # Menu rewards only if hm01 is in bag
        # if self.hm_count != 0:
        self.calculate_menu_rewards()
        self.calculate_cut_coords_and_tiles_rewards()
        self.calculate_seen_caught_pokemon_moves()   
        
        # Final reward calculation
        self.calculate_reward()
        reward, self.last_reward = self.subtract_previous_reward_v1(self.last_reward, self.reward)

        info = {}
        done = self.time >= self.max_episode_steps

        if self.save_video and done:
            self.full_frame_writer.close()
        
        # info.update({"stats": {"seen_pokemon": np.sum(self.seen_pokemon)}})
        # info.update({"stats": {"event": self.get_events()}})
        self.hideout_progress = ram_map_leanke.monitor_hideout_events(self.game)
        self.pokemon_tower_progress = ram_map_leanke.monitor_poke_tower_events(self.game)
        self.silph_progress = ram_map_leanke.monitor_silph_co_events(self.game)

        if 'stats' not in info:
            info['stats'] = {}
        
        info['stats'].update({"badges": float(self.badges)})
        info['stats'].update({'beat_rocket_hideout_giovanni': self.hideout_progress['beat_rocket_hideout_giovanni']})
        info['stats'].update({'rescued_mr_fuji': self.pokemon_tower_progress['rescued_mr_fuji']})
        info['stats'].update({'beat_silph_co_giovanni': self.silph_progress['beat_silph_co_giovanni']})

        if done or self.time % 2000 == 0 or self.time == 0: # 10000
            info = {
                "pokemon_exploration_map": self.counts_map,
                "stats": {
                    "step": self.time,
                    "x": c,
                    "y": r,
                    "map": map_n,
                    # "map_location": self.get_map_location(map_n),
                    # "max_map_progress": self.max_map_progress,
                    "pcount": int(self.game.get_memory_value(0xD163)),
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
                    "pokemon_exploration_map": self.counts_map,
                    "seen_npcs_count": len(self.seen_npcs),
                    "seen_pokemon": np.sum(self.seen_pokemon),
                    "caught_pokemon": np.sum(self.caught_pokemon),
                    "moves_obtained": np.sum(self.moves_obtained),
                    "hidden_obj_count": len(self.seen_hidden_objs),
                    "bill_saved": self.bill_state,
                    "hm_count": self.hm_count,
                    "cut_taught": self.cut,
                    "badge_1": float(self.badges >= 1),
                    "badge_2": float(self.badges >= 2),
                    "badge_3": float(self.badges >= 3),
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
                    'self.milestones': self.milestones,   
                    'beat_rocket_hideout_giovanni': self.hideout_progress['beat_rocket_hideout_giovanni'],
                    'rescued_mr_fuji': self.pokemon_tower_progress['rescued_mr_fuji'],    
                    'beat_silph_co_giovanni': self.silph_progress['beat_silph_co_giovanni'], 
                                             
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
                    "used_cut_reward": self.cut_reward,
                    "cut_coords_reward": self.cut_coords_reward,
                    "cut_tiles_reward": self.cut_tiles_reward,
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
                                base_reward=10, reward_increment=2, reward_multiplier=1),
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
            + self.cut_reward
            + self.that_guy_reward / 2
            + self.cut_coords_reward
            + self.cut_tiles_reward
            + self.tree_distance_reward * 0.6
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
            self.gym3_events_reward + self.gym4_events_reward +
            self.gym5_events_reward + self.gym6_events_reward +
            self.gym7_events_reward)
            + (self.gym3_events_reward + self.gym4_events_reward +
            self.gym5_events_reward + self.gym6_events_reward +
            self.gym7_events_reward)
            + self.can_reward
            + self.len_respawn_reward
            + self.lock_1_use_reward
            + self.celadon_tree_reward
            + self.used_cut_reward
        )
    

        
        ###### USEFUL STUFF BUT COMMENTED OUT B/C NOT CURRENTLY USING
    #     self.last_achievements = {}
    #     self.current_achievements = {}
    #     self.already_saved_state = set()
    #     self.problematic_vars = []
    #     self.path_to_pyboy_state = None
    #     self.path_to_pkl = None
    #     self.step_counter = 0
    #     self.step_count = 0
    #     ###############################################################
    #     # New today BET ADDED
    #     # self.cut = False  # Placeholder for cut HM status
    #     self.used_cut = 0  # Placeholder for usage of cut   
    #     # Combine all map IDs into a single list for easier management
    #     self.map_ids = [
    #         0, 1, 12, 51, 2, 54, 59, 14, 60, 61, 88, 101, 65, 92, 134, 157, 166, 178,
    #         199, 200, 201, 202, 203, 142, 143, 144, 145, 146, 147, 148,
    #         181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236
    #     ]
    #     self.global_visited_map = set()
    #     # self.initialize_shared_data(self.env_id)
    # # def initialize_map_completion_status(self):
    # #     """ Initialize the completion status for each map """
    # #     return {f"map_{i}_completion": False for i in range(1, 10)}  # Example maps
        
    # def initialize_milestones(self):
    #     self.milestones = {
    #         "badge_1": self.is_badge_1_achieved(),
    #         "mt_moon_completion": self.is_mt_moon_completed(),
    #         "badge_2": self.is_badge_2_achieved(),
    #         "bill_completion": self.is_bill_completed(),
    #         "rubbed_captains_back": self.is_captains_back_rubbed(),
    #         "taught_cut": self.is_cut_taught(),
    #         "used_cut_on_good_tree": self.is_cut_used_on_good_tree(),
    #         "map_completion_status": {f"map_{map_id}_completion": self.check_map_completion(map_id) for map_id in self.map_ids}
    #     }
    #     print(f'509 initialize_milestones(): self.milestones {self.milestones} for env_id {self.env_id} initialized!!')

    # def is_badge_1_achieved(self):
    #     return self.badges >= 1    
    
    # def is_badge_2_achieved(self):
    #     return self.badges >= 2

    # def is_badge_3_achieved(self):
    #     return self.badges >= 3
    
    # def is_badge_4_achieved(self):
    #     return self.badges >= 4

    # def is_mt_moon_completed(self):
    #     r, c, map_n = ram_map.position(self.game)
    #     return map_n == 4

    # def is_bill_completed(self):
    #     self.saved_bill_checkpoint = ram_map.saved_bill(self.game)
    #     return self.saved_bill_checkpoint
    
    # def is_captains_back_rubbed(self):
    #     return int(ram_map.read_bit(self.game, 0xD803, 1))

    # def is_cut_taught(self):
    #     return self.cut

    # def is_cut_used_on_good_tree(self):
    #     return self.cut_coords

    # def check_map_completion(self, map_id):
    #     r, c, map_n = ram_map.position(self.game)
    #     return map_n == map_id

    # def check_milestone2(self, milestone):
    #     # Dispatch to the appropriate method based on the milestone name
    #     if milestone in self.milestones:
    #         return self.milestones[milestone]
    #     elif milestone.startswith("map_") and milestone.endswith("_completion"):
    #         map_id = int(milestone.split("_")[1])
    #         complete = self.check_map_completion(map_id)
    #         if complete:
    #             print(f'self.check_map_completion returned (True) {complete}')
    #         else:
    #             print(f'self.check_map_completion returned (False) {complete}')
    #     else:
    #         # Default case, if no specific check is implemented for the milestone
    #         print(f"556 check_milestone2 No specific check implemented for milestone {milestone}....")
    #         return False

    # def initialize_shared_data(self, env_id):
    #     with Base.shared_lock:  # Use the shared lock to synchronize access
    #         if env_id not in Base.shared_data:
    #             Base.shared_data[env_id] = {
    #                 'milestones': self.initialize_milestones(),
    #                 'files_to_load': None
    #             }
    #         print(f"Initial shared data for env_id {env_id}: {Base.shared_data[env_id]}")
    #         self.milestones = Base.shared_data[env_id]
    #         # print(f"\nInitial shared data for env_id {env_id}: {Base.shared_data[env_id]}\n")
    #         print(f"\nInitial shared data for self.milestones for {env_id}: {self.milestones}\n")  
            
    # def sync_shared_data(self):
    #     # All envs
    #     # Obtain the .pkl and .state file paths explicitly from the envs' local self.milestones dict
    #     if 'files_to_load' in self.milestones and self.milestones['files_to_load']:
    #         self.path_to_pkl = self.milestones['files_to_load'][0]
    #         print(f'self.path_to_pkl = self.milestones["files_to_load"][0]: {self.path_to_pkl}')
    #         self.path_to_pyboy_state = self.milestones['files_to_load'][1]
    #         print(f'self.path_to_pyboy_state = self.milestones["files_to_load"][0]: {self.path_to_pyboy_state}')
    #     else:
    #         print("No files are set to load.")

    #     # Copy the global shared dict
    #     local_copy = deepcopy(Base.shared_data[self.env_id])
    #     # Update the copy with current milestones and state paths, if any
    #     local_copy['milestones'].update(self.milestones)
    #     local_copy['files_to_load'] = (self.path_to_pkl, self.path_to_pyboy_state)
    #     # Re-assign the modified local copy back to the shared dictionary
    #     Base.shared_data[self.env_id] = local_copy
    #     # Example of flattened keys
    #     # Base.shared_data[(self.env_id, 'badge_1')] = True
    #     # Base.shared_data[(self.env_id, 'files_to_load')] = ('path_to_pkl', 'path_to_state')

    # def save_all_states_v3_debug(self):
    #     datetime_str = "timeless" # datetime.now().strftime("%Y%m%d_%H%M%S")
    #     self.state_dir = self.save_state_dir / f'debug_{datetime_str}'
    #     self.state_dir.mkdir(parents=True, exist_ok=True)
    #     self.state_file_path = self.state_dir / "debug_state.pkl"
    #     self.pyboy_state_file_path = self.state_dir / "debug_pyboy_state.state"
        
    #     state = {key: value for key, value in vars(self).items()}
    #     try:
    #         with open(self.state_file_path, 'wb') as f:
    #             dill.dump(state, f)
    #         print(f"Debug: State successfully saved to {self.state_file_path}")
    #     except Exception as e:
    #         print(f"Debug: Failed to save state. Error: {e}")

    #     try:
    #         with open(self.pyboy_state_file_path, 'wb') as f:
    #             self.game.save_state(f)
    #         print(f"Debug: PyBoy state successfully saved to {self.pyboy_state_file_path}")
    #     except Exception as e:
    #         print(f"Debug: Failed to save PyBoy state. Error: {e}")

    # # Serializes instance vars() and saves pyboy state file
    # def save_all_states_v3(self, is_failed=False):
    #     r, c, map_n = ram_map.position(self.game)
    #     # Make sure self.save_state_dir is properly initialized in Base class init
    #     datetime_str = "timeless" # datetime.now().strftime("%Y%m%d_%H%M%S")
    #     self.state_dir = self.save_state_dir / Path(f'event_reward_{self.event_reward}') / f'env_id_{self.env_id}'
    #     self.state_dir.mkdir(parents=True, exist_ok=True)
    #     self.state_file_path = self.state_dir / f"env_id_{self.env_id}_{datetime_str}_map_n_{map_n}_state.pkl"
    #     self.pyboy_state_file_path = self.state_dir / f"env_id_{self.env_id}_{datetime_str}_map_n_{map_n}_state.state"
    #     state = {}
    #     for key, value in vars(self).items():
    #         try:
    #             dill.dumps(value)
    #             state[key] = value
    #             # logging.info(f"Pickled: {key}")
    #         except Exception as e:
    #             pass
    #             # logging.warning(f"Failed to pickle {key}: {e}")        
    #     # Save the state dictionary that only includes picklable items
    #     try:
    #         with open(self.state_file_path, 'wb') as f:
    #             dill.dump(state, f)
    #         logging.info(f"Saved state to {self.state_file_path}")
    #     except Exception as e:
    #         print(f"Failed to save state. Error: {e}")
    #         state = {key: value for key, value in vars(self).items() if not key in self.problematic_vars}
    #         try:
    #             with open(self.state_file_path, 'wb') as f:
    #                 dill.dump(state, f)
    #             print(f"Saved state to {self.state_file_path}")
    #         except Exception as e:
    #             print(f"Failed to save state. Error: {e}")        
    #     try:
    #         with open(self.pyboy_state_file_path, 'wb') as f:
    #             self.game.save_state(f)
    #         logging.info(f"Saved PyBoy state to {self.pyboy_state_file_path}")
    #     except Exception as e:
    #         print(f'{self.pyboy_state_file_path} failed to save pyboy state. Error: {e}')
    #         print(f'{self.pyboy_state_file_path} failed to save pyboy state. Error: {e}')    

    # def update_milestones(self):
    #     milestone_achieved = False  # Flag to track if any milestone has been achieved        
    #     # Initialize a dictionary to track which milestones have been saved if it doesn't exist
    #     if not hasattr(self, 'milestones_saved'):
    #         self.milestones_saved = {}
    #     # Iterate over all milestones
    #     print(f'632 self.milestones.saved()={self.milestones_saved.items()}')
    #     print(f'633 self.milestones.items()={self.milestones.items()}')
    #     for key, func in self.milestones.items():
    #         if callable(func):
    #             new_status = func()
    #             # Check if the milestone has just been achieved and hasn't been saved yet
    #             if new_status == 1 and self.milestones[key] != new_status and key not in self.milestones_saved:
    #                 milestone_achieved = True
    #                 self.milestones_saved[key] = True
    #                 print(f"ENV_ID {self.env_id}: Milestone {key} achieved.")
    #             self.milestones[key] = new_status
    #         else:
    #             if key == "map_completion_status":
    #                 print(f'else\n')
    #                 print(f'645 self.milestones["map_completion_status"]={self.milestones["map_completion_status"]}')
    #                 # Update each map completion status
    #                 for map_key in self.milestones[key]:
    #                     # print(f'ENV_ID {self.env_id}: map_key={map_key}')
    #                     map_id = int(map_key.split('_')[1])
    #                     new_status = (1, None) if any(coord[2] == map_id for coord in self.seen_coords) else (0, None)
    #                     if new_status == (1, None) and self.milestones[key][map_key] != new_status and map_key not in self.milestones_saved:
    #                         milestone_achieved = True
    #                         self.milestones_saved[map_key] = True
    #                         print(f"ENV_ID {self.env_id}: {map_key} achieved.")
    #                         with open(self.log_file_path, "w") as f:
    #                             f.write(f"ENV_ID {self.env_id}: {map_key}")                                
    #                     self.milestones[key][map_key] = new_status
    #                     print(f'ENV_ID {self.env_id}: self.milestones[key][map_key]: {self.milestones[key][map_key]}, key: {key}, map_key: {map_key}')
    #     # If any milestone has been achieved and it's the first time being saved, trigger saving all states
    #     if milestone_achieved:
    #         self.save_all_states_v3_debug()
    #         print(f"ENV_ID {self.env_id}: State saved due to milestone completion.")    

    # def log_milestones(self):
    #     # Log current milestone data to file
    #     for milestone, status in self.milestones.items():
    #         pass
    #         # print(f"log_milestones: milestone = {milestone}: status = {status}")    
        
    # def log_shared_data(self):
    #     """
    #     Logs the current state of shared_data for this environment.
    #     """
    #     with self.lock:  # Ensure thread-safe access to shared data
    #         # Retrieve this environment's data from the shared dictionary
    #         env_data = Base.shared_data.get(self.env_id, {})
    #         # Log each milestone and its status
    #         for milestone, value in env_data.items():
    #             pass
    #             # print(f"\nSHARED_DICT for env_id {self.env_id}: Milestone {milestone}: Status {value}\n")
     
    # def assess_and_trigger_state_loads(self):
    #     num_envs = len(Base.shared_data)  # Get number of envs
    #     milestone_completion_counts = {key: 0 for key in self.milestones.keys()}
    #     threshold_dict = dict(zip(self.milestones.keys(), self.milestone_threshold_values))
    #     # Aggregate completion counts
    #     for env_id, milestones in Base.shared_data.items():
    #         for key, value in milestones.items():
    #             # Handle different types of milestone data
    #             if isinstance(value, tuple):
    #                 completed, _ = value
    #             elif isinstance(value, (int, float)):  # Handle simple numeric milestones
    #                 completed = value
    #             else:
    #                 continue  # Skip unsupported types
    #             if completed:
    #                 milestone_completion_counts[key] += 1

    #     # Calculate completion percentages and assess against thresholds
    #     for key, count in milestone_completion_counts.items():
    #         completion_percentage = count / num_envs
    #         if completion_percentage >= threshold_dict.get(key, self.default_load_threshold):  # Provide a default threshold
    #             # Calculate the number of environments that need loading
    #             envs_to_load = num_envs - int(completion_percentage * num_envs)
    #             self.trigger_state_loading(key, envs_to_load)
     
    # def trigger_state_loading(self, milestone_key, num_envs_to_load):
    #     # Assuming the function to get the best state is defined and available
    #     # This example simply logs the action; replace with actual loading logic
    #     # print(f"ENV_ID: {self.env_id}: Triggering state loading for {num_envs_to_load} environments due to milestone {milestone_key}")
    #     best_state_paths = self.get_best_state_paths()
    #     self.assign_states_to_envs(best_state_paths, num_envs_to_load)

    # def get_best_state_paths(self):
    #     # Placeholder function to retrieve the path of the best state
    #     # Replace with actual logic to determine the best state
    #     # Blank right now
    #     return ("path_to_pkl_file.pkl", "path_to_pyboy_state.state")

    # def assign_states_to_envs(self, state_paths, num_envs_to_load):
    #     # Assign state files to environments that need them
    #     env_ids_to_load = random.sample(range(1, self.num_envs), num_envs_to_load)  # Excluding env_0
    #     for env_id in env_ids_to_load:
    #         with self.lock:
    #             Base.shared_data[env_id]['files_to_load'] = state_paths 
                
    # def sort_and_assess_envs_for_loading_v2(self, actions_required, num_envs):
    #     worst_envs, best_env_files = [], []
    #     for milestone, action_percentage in actions_required.items():
    #         envs_completion_times = [
    #             (env_id, data.get(milestone, [None, None])[1], data.get("files_to_load"))
    #             for env_id, data in Base.shared_data.items()
    #             if milestone in data and isinstance(data[milestone], list)
    #         ]
    #         sorted_envs = sorted(envs_completion_times, key=lambda x: x[1] or float('inf'))
    #         split_index = int(len(sorted_envs) * (1 - action_percentage))
    #         best_env_files.extend(
    #             files_to_load for _, _, files_to_load in sorted_envs[:split_index] if files_to_load
    #         )
    #         worst_envs.extend(env_id for env_id, _, _ in sorted_envs[split_index:])
    #     worst_envs = list(dict.fromkeys(worst_envs))  # Deduplicate
    #     for env_id in worst_envs:
    #         if best_env_files:
    #             selected_files = random.choice(best_env_files)
    #             with self.lock:
    #                 Base.shared_data[env_id]["files_to_load"] = selected_files
    #         else:
    #             print("No best environment files available for loading.")
    
    # def load_state_from_tuple_simple2(self, pkl_file, state_file):
    #     try:
    #         # Load state from the .pkl file
    #         with open(pkl_file, 'rb') as f:
    #             state = pickle.load(f)
    #             for key, value in state.items():
    #                 setattr(self, key, value)
    #         self.logger.info(f"Loaded state from {pkl_file}.")            
    #         # Load PyBoy state from the .state file
    #         with open(state_file, 'rb') as f:
    #             self.game.load_state(f)
    #         self.logger.info(f"Loaded PyBoy state from {state_file}.")            
    #     except Exception as e:
    #         self.logger.error(f"Failed to load states from {pkl_file} and {state_file}: {e}")
    #     # Reset or initialize as necessary post-loading
    #     self.reset_count = 0
    #     self.step_count = 0
    #     self.reset_count += 1         
        
    # def load_state_from_tuple_simple(self, pickled_file_path, pyboy_state_file_path):
    #     print(f"Selected .pkl for loading: {pickled_file_path}")
    #     print(f"Selected .state for loading: {pyboy_state_file_path}")            
    #     # Load state from the .pkl file
    #     try:
    #         with open(pickled_file_path, 'rb') as f:
    #             state = pickle.load(f)
    #             for key, value in state.items():
    #                 setattr(self, key, value)
    #         print(f"Environment state at {pickled_file_path} loaded successfully.")
    #     except Exception as e:
    #         print(f"Failed to load environment state. Error: {e}")            
    #     # Load PyBoy state if the .state file exists
    #     try:
    #         with open(pyboy_state_file_path, 'rb') as f:
    #             self.game.load_state(f)  # Ensure this is the correct method to load your PyBoy state
    #         print(f"PyBoy state at {pyboy_state_file_path} loaded successfully.")
    #     except Exception as e:
    #         print(f"Failed to load PyBoy state. Error: {e}")
    #     # Reset or initialize as necessary post-loading
    #     self.reset_count = 0
    #     self.step_count = 0
    #     self.reset_count += 1         
                
    # def load_state_from_tuple(self):
    #     # Retrieve the tuple of file paths for the current environment
    #     with Base.lock:
    #         files_to_load = Base.shared_data[self.env_id].get("files_to_load", None)
    #     if not files_to_load:
    #         # print("No files specified for loading.")
    #         return
    #     if files_to_load != None:
    #         # Unpack the tuple into its components
    #         pickled_file_path, pyboy_state_file_path = files_to_load
    #         print(f"Selected .pkl for loading: {pickled_file_path}")
    #         print(f"Selected .state for loading: {pyboy_state_file_path}")            
    #         # Load state from the .pkl file
    #         try:
    #             with open(pickled_file_path, 'rb') as f:
    #                 state = pickle.load(f)
    #                 for key, value in state.items():
    #                     setattr(self, key, value)
    #             print(f"Environment state at {pickled_file_path} loaded successfully.")
    #         except Exception as e:
    #             print(f"Failed to load environment state. Error: {e}")            
    #         # Load PyBoy state if the .state file exists
    #         try:
    #             with open(pyboy_state_file_path, 'rb') as f:
    #                 self.game.load_state(f)  # Ensure this is the correct method to load your PyBoy state
    #             print(f"PyBoy state at {pyboy_state_file_path} loaded successfully.")
    #         except Exception as e:
    #             print(f"Failed to load PyBoy state. Error: {e}")
    #         # Reset or initialize as necessary post-loading
    #         self.reset_count = 0
    #         self.step_count = 0
    #         self.reset_count += 1            
                
    # All envs
    # Simple state-loading fn
    # Gets 
    # def load_state_from_tuple_gpt(self):
    #     print("Checking shared data before loading state:", Base.shared_data.get(self.env_id, {}))
    #     files_to_load = Base.shared_data[self.env_id].get("files_to_load")
    #     if files_to_load != None:
    #         print(f"Attempting to load from {files_to_load}...")
    #         with Base.counter_lock:
    #             files_to_load = Base.shared_data[self.env_id].get("files_to_load")
    #         if files_to_load:
    #             pickled_file_path, pyboy_state_file_path = files_to_load
    #             print(f"Ready to load .pkl from: {pickled_file_path}")
    #             print(f"Ready to load .state from: {pyboy_state_file_path}")
    #             # Try to load the pickle file
    #             try:
    #                 with open(pickled_file_path, 'rb') as f:
    #                     state = pickle.load(f)
    #                     for key, value in state.items():
    #                         setattr(self, key, value)
    #                 print(f"Environment state loaded from {pickled_file_path}.")
    #             except Exception as e:
    #                 print(f"Exception encountered while loading .pkl file from {[pickled_file_path]}: {e}")
    #             # Try to load the PyBoy state file
    #             try:
    #                 with open(pyboy_state_file_path, 'rb') as f:
    #                     self.game.load_state(f)
    #                 print(f"PyBoy state loaded from {pyboy_state_file_path}.")
    #             except Exception as e:
    #                 print(f"Exception encountered while loading .state file: {e}")
    #         else:
    #             print("No state files specified or found for loading.")

    # def assign_files_to_load(self, pkl_path, state_path):
    #     if pkl_path and state_path:
    #         self.milestones['files_to_load'] = (pkl_path, state_path)
    #         print(f"\nAssigned files to load for env_id {self.env_id}: {self.milestones.get('files_to_load')}\n")
    #     else:
    #         print("Invalid file paths provided.")

    # def search_and_assign_files(self):
    #     file_dict = {}
    #     for root, dirs, files in os.walk(self.exp_path):
    #         for file in files:
    #             if file.endswith('.pkl') or file.endswith('.state'):
    #                 # self.logger.info(f'search1 HIT {file}')
    #                 path = os.path.join(root, file)
    #                 base_name = os.path.splitext(path)[0]
    #                 file_type = 'pkl' if file.endswith('.pkl') else 'state'
    #                 if base_name not in file_dict:
    #                     file_dict[base_name] = {}
    #                 file_dict[base_name][file_type] = path
    #     current_length = len(file_dict)
    #     self.logger.info(f'current_length={current_length}; last_len={self.length_last_assigned_file_dict}')
    #     if current_length != self.length_last_assigned_file_dict:
    #         for base_name, paths in file_dict.items():
    #             # print(f'{file_dict}')
    #             pkl_path = paths.get('pkl')
    #             state_path = paths.get('state')
    #             if (pkl_path, state_path) not in self.already_assigned_states:
    #                 if pkl_path and state_path:
    #                     self.already_assigned_states.add((pkl_path, state_path))
    #                     self.assign_files_to_load(pkl_path, state_path)
    #                 else:
    #                     self.logger.info(f"Unpaired file found: {pkl_path if pkl_path else state_path}")
    #         if not file_dict:
    #             self.logger.info("No new files found.")
    #     self.length_last_assigned_file_dict = current_length
    
    # def search_and_assign_files4(self):
    #     import os
    #     search_directory = self.experiments_base_dir
    #     new_files = {}        
    #     for file in os.listdir(search_directory):
    #         if file.endswith(".pkl") or file.endswith(".state"):
    #             base_name = file.split('_state')[0]
    #             file_path = os.path.join(search_directory, file)
    #             if base_name not in new_files:
    #                 new_files[base_name] = []
    #             new_files[base_name].append(file_path)
    #     for base_name, files in new_files.items():
    #         if len(files) == 2:
    #             pkl_file = next((f for f in files if f.endswith(".pkl")), None)
    #             state_file = next((f for f in files if f.endswith(".state")), None)
    #             if pkl_file and state_file:
    #                 self.logger.info(f"Detected new file pair: {pkl_file}, {state_file}")
    #                 if self.simple_loader:
    #                     self.load_state_from_tuple_simple2(pkl_file, state_file)
    #                 else:
    #                     Base.shared_data[self.env_id]['files_to_load'] = (pkl_file, state_file)
    #         else:
    #             self.logger.warning(f"Unmatched or incomplete file set for {base_name} detected.")
    #     if not new_files:
    #         self.logger.info("No new files found in the search directory.")
        
    # def search_and_assign_files3(self):
    #     import os
    #     search_directory = self.experiments_base_dir
    #     print(f'search_and_assign_files2 searching dir {search_directory} for new files...')
    #     new_files = {}        
    #     # Collect all relevant files and organize them by base name
    #     for file in os.listdir(search_directory):
    #         if file.endswith(".pkl") or file.endswith(".state"):
    #             base_name = file.split('_state')[0]  # Extracts the part of the filename before '_state'
    #             file_path = os.path.join(search_directory, file)
    #             if base_name not in new_files:
    #                 new_files[base_name] = []
    #             new_files[base_name].append(file_path)
    #             print(f'search_and_assign_files2 file {file} categorized under {base_name}. Path: {file_path}')
    #     # If new files are found, attempt to find matched pairs
    #     for base_name, files in new_files.items():
    #         if len(files) == 2:
    #             # Ensure exactly one .pkl and one .state file per pair
    #             pkl_file = next((f for f in files if f.endswith(".pkl")), None)
    #             state_file = next((f for f in files if f.endswith(".state")), None)
    #             if pkl_file and state_file:
    #                 if self.simple_loader:
    #                     self.load_state_from_tuple_simple2(pkl_file, state_file)
    #                 else:
    #                     Base.shared_data[self.env_id]['files_to_load'] = (pkl_file, state_file)
    #                     print(f"Env_{self.env_id} found new files and updated shared data with {pkl_file} and {state_file}.")
    #         else:
    #             print(f"Unmatched or incomplete file set for {base_name}.")
    #     else:
    #         pass
    #         # print("No new files found in the search directory.")

    # @staticmethod
    # def add_map(map_n):
    #     """Safely add a new map to the shared set."""
    #     with Base.set_lock:
    #         Base.shared_set[map_n] = True
    #         print(f'map_n={map_n} added to Base.shared_set={Base.shared_set}')

    # @staticmethod
    # def check_map(map_n):
    #     """Check if the map is already in the shared set."""
    #     with Base.set_lock:
    #         return Base.shared_set.get(map_n, False)
    #################################################################################################
        
##################################################################
##################################################################
#### NEW NEW

    # def sync_shared_data2(self):
    #     # Synchronize local milestones with those in shared data
    #     with Base.shared_lock:
    #         for key, shared_value in Base.shared_data[self.env_id].items():
    #             if key in self.milestones:
    #                 self.milestones[key] = shared_value
    #         # print(f"Env {self.env_id} synced with shared data.")

    # def search_and_assign_files2(self):
    #     import os
    #     search_directory = self.experiments_base_dir
    #     print(f'search_and_assign_files2 searching dir {search_directory} for new files...')
    #     new_files = []
    #     for file in os.listdir(search_directory):
    #         if file.endswith(".pkl") or file.endswith(".state"):
    #             print(f'search_and_assign_files2 file {file} ends with .pkl or .state!')
    #             new_files.append(os.path.join(search_directory, file))
    #             print(f'search_and_assign_files2 the path of that file {file} is {new_files[0]}')

        # # If new files are found, randomly select a pair (simulation of new file detection)
        # if new_files:
        #     print(f'search_and_assign_files2 if new_files=True; new_files={new_files}')
        #     pkl_file = next((f for f in new_files if f.endswith(".pkl")), None)
        #     state_file = next((f for f in new_files if f.endswith(".state")), None)
        #     if pkl_file and state_file:
        #         if self.simple_loader == True:
        #             self.load_state_from_tuple_simple()
        #         else:
        #             Base.shared_data[self.env_id]['files_to_load'] = (pkl_file, state_file)
        #             print(f"Env_{self.env_id} found new files and updated shared data.")

    # def load_state_from_tuple_gpt2(self):
    #     files_to_load = Base.shared_data.get(self.env_id, {}).get('files_to_load')
    #     if files_to_load:
    #         pkl_path, state_path = files_to_load
    #         # Load state from pkl
    #         import pickle
    #         try:
    #             with open(pkl_path, 'rb') as f:
    #                 state_dict = pickle.load(f)
    #             # Assume state_dict contains keys that match attributes to restore
    #             for key, value in state_dict.items():
    #                 setattr(self, key, value)
    #             # print(f"Env {self.env_id} loaded state from {pkl_path}.")
    #         except Exception as e:
    #             # print(f"Failed to load pkl file: {e}")

    #             # Load PyBoy state if applicable
    #             try:
    #                 self.game.load_state(state_path)
    #                 # print(f"Env {self.env_id} loaded PyBoy state from {state_path}.")
    #             except Exception as e:
    #                 pass
    #                 # print(f"Failed to load PyBoy state file: {e}")

    #             # Clear the load path to prevent re-loading in subsequent cycles
    #             with Base.shared_lock:
    #                 Base.shared_data[self.env_id]['files_to_load'] = None
    #         else:
    #             pass
    #             # print(f'load_state_from_tuple_gpt2: No files_to_load for any envs...')
                
    # def check_milestone2(self, milestone):
    #     result = False
    #     if milestone == "badge_1":
    #         result = self.is_badge_1_achieved()
    #     elif milestone.startswith("map_") and milestone.endswith("_completion"):
    #         map_id = int(milestone.split("_")[1])
    #         complete = self.check_map_completion(map_id)
    #         if complete:
    #             pass
    #             # print(f'self.check_map_completion returned (True) {complete}')
    #         else:
    #             pass
    #             # print(f'self.check_map_completion returned (False) {complete}')
    #     # print(f"Checking {milestone}: {result}")
    #     return result
        
    # def update_milestones2(self):
    #     milestone_achieved = False
    #     # print(f'1148 v2 self.milestones.items()={self.milestones.items()}')
    #     # print(f'1148 v2 self.milestones["milestones"]={self.milestones["milestones"]}')
    #     for milestone, achieved in self.milestones.items():
    #         # print(f'1155 update_milestones2 self.check_milestone2(milestone) for milestone {milestone} returned {self.check_milestone2(milestone)}. achieved={achieved}')
    #         if self.check_milestone2(milestone) and not achieved:
    #             # print(f'1157 update_milestones2 conditional True')
    #             self.milestones[milestone] = True
    #             milestone_achieved = True
    #             # print(f"1158 Env {self.env_id} achieved milestone {milestone}.")
    #             self.save_state_if_first2(milestone)
    #     return milestone_achieved
    
    # def update_milestones3(self):
    #     milestone_achieved = False
    #     for key in self.milestones.keys():
    #         achieved = self.check_milestone2(key)
    #         if achieved and not self.milestones[key]:
    #             self.milestones[key] = True
    #             milestone_achieved = True
    #             # print(f"Milestone {key} achieved.")
    #             self.save_state_if_first2(key)
    #     return milestone_achieved

    # def save_state_if_first2(self, milestone):
    #     with Base.shared_lock:
    #         # Check global condition
    #         if 'milestone_saved' not in Base.shared_data or not Base.shared_data['milestone_saved']:
    #             Base.shared_data['milestone_saved'] = True
    #             self.save_state2(milestone)
    #             print(f"Env {self.env_id} saved state for milestone {milestone}.")
                
    # def check_milestone2(self, milestone):
    #     """ Dispatch to the appropriate method based on the milestone name """
    #     # Check for specific milestones based on predefined conditions
    #     if milestone == "badge_1":
    #         return self.is_badge_1_achieved()
    #     elif milestone == "mt_moon_completion":
    #         return False # self.is_mt_moon_completed()
    #     elif milestone == "badge_2":
    #         return self.is_badge_2_achieved()
    #     elif milestone == "bill_completion":
    #         return self.is_bill_completed()
    #     elif milestone == "captains_back_rubbed":
    #         return self.is_captains_back_rubbed()
    #     elif milestone == "cut_taught":
    #         return self.is_cut_taught()
    #     elif milestone == "used_cut_on_good_tree":
    #         return self.is_cut_used_on_good_tree()
    #     else:
    #         # Attempt to dynamically handle map completion milestones
    #         return self.check_dynamic_map_completion(milestone)

    # def check_dynamic_map_completion(self, milestone):
    #     if milestone.startswith("map_") and milestone.endswith("_completion"):
    #         map_id = int(milestone.split("_")[1])
    #         # print(f'1190 check_dynamic_map_completion map_id={map_id}')
    #         return self.check_map_completion(map_id)
    #     else:
    #         # print(f"No specific check implemented for milestone {milestone}")
    #         return False    