from pathlib import Path
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
from . import ram_map, game_map, data, ram_map_leanke
import subprocess
import multiprocessing
import time
import random
from multiprocessing import Manager
from pokegym import data

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
import math
import datetime
import pickle
import json
import copy
import random
import uuid
from math import floor, sqrt
from typing import Union
from .constants import POKECENTER_TO_INDEX_DICT
import pandas as pd
import csv

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
# print(f'TREE_POSOTIONS_CONVERTED = {TREE_POSITIONS_GRID_GLOBAL}')
MAPS_WITH_TREES = set(map_n for _, _, map_n in TREE_POSITIONS_PIXELS)
TREE_COUNT_PER_MAP = {6: 2, 134: 3, 13: 5, 1: 2, 5: 1, 36: 1, 20: 1, 21: 4}

class Base:
    # Shared counter among processes
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    
    # Initialize a shared integer with a lock for atomic updates
    shared_length = multiprocessing.Value('i', 0)  # 'i' for integer
    lock = multiprocessing.Lock()  # Lock to synchronize access
    
    # Initialize a Manager for shared BytesIO object
    manager = Manager()
    shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data
    
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        # Increment counter atomically to get unique sequential identifier
        with Base.counter_lock:
            env_id = Base.counter.value
            Base.counter.value += 1
            
        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = STATE_PATH + "Bulbasaur.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        self.screenshot_counter = 0
        
        file_path = Path('/experiments/required_resources/running_experiment.txt')
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write('default_exp_name')
        # Logging initializations
        with open("/experiments/required_resources/running_experiment.txt", "r") as file:
            exp_name = file.read()
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        self.env_id = env_id
        self.s_path = Path('/experiments') # Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        self.video_path = Path(f'./videos')
        self.video_path.mkdir(parents=True, exist_ok=True)
        # self.csv_path = Path(f'./csv')
        # self.csv_path.mkdir(parents=True, exist_ok=True)
        self.reset_count = 0
        self.explore_hidden_obj_weight = 1
        self.pokemon_center_save_states = []
        self.pokecenters = [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 147, 182]
        self.used_cut_on_map_n = 0
        
        # BET ADDED nimixx api
        self.api = Game(self.game) # import this class for api BET
        
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
        # BET ADDED TREE 
        self.min_distances = {}  # Key: Tree position, Value: minimum distance reached
        self.rewarded_distances = {}  # Key: Tree position, Value: set of rewarded distances
        self.used_cut_on_map_n = 0
        self.seen_map_dict = {}
        
        # BET ADDED MORE
        self.past_events_string = self.all_events_string
        self._all_events_string = ''
        self.time = 0
        self.level_reward_badge_scale = 0
        self.special_exploration_scale = 0
        self.elite_4_lost = False
        self.elite_4_early_done = False
        self.elite_4_started_step = None
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.reward_scale = 1
        self.has_fresh_water_in_bag = False
        self.has_lemonade_in_bag = False
        self.has_silph_scope_in_bag = False
        self.has_lift_key_in_bag = False
        self.has_pokedoll_in_bag = False
        self.has_bicycle_in_bag = False
        self.has_fresh_water_in_bag_reward = 0
        self.has_lemonade_in_bag_reward = 0
        self.has_silph_scope_in_bag_reward = 0
        self.has_lift_key_in_bag_reward = 0
        self.has_pokedoll_in_bag_reward = 0
        self.has_bicycle_in_bag_reward = 0
        self.gym_info = GYM_INFO
        self.save_state_dir = Path('/experiments')
        self.current_level = 0
        self.early_stopping_min_reward = 2.0
        self.print_rewards = False # print sb4-style rewards printout in terminal
        self.save_final_state = True
        self.total_reward = 0
        self.last_event_rew = 0
        
    # Turns a method into a memoized property - property-access cached data cheaply!
    # May use this later but probably not.
    def memoized_property(func):
        property_name = f"_{func.__name__}_memoized"  
        @property
        def wrapper(self):
            if not hasattr(self, property_name):
                setattr(self, property_name, func(self))
            return getattr(self, property_name)
        return wrapper    
    
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
        # self.seen_coords_bet = set() ## moved from reset
        self.state_loaded_instead_of_resetting_in_game = 0
        self.badge_count = 0
        self.prev_badge_count = 0
        
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
                    self.died_count += 1
    
    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.game.get_memory_value(i + 0xD2F7)
            seen_mem = self.game.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   
    
    def update_moves_obtained(self):
        # Scan party
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
                    
    def init_map_mem(self):
        self.seen_coords = {}
        self.perm_seen_coords = {}
        self.special_seen_coords_count = 0
    
    def update_seen_coords(self):
        x_pos, y_pos = self.current_coords()
        map_n = self.current_map_id - 1
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.special_exploration_scale and map_n in SPECIAL_MAP_IDS and coord_string not in self.perm_seen_coords:
            # self.seen_coords[coord_string] = self.step_count
            self.special_seen_coords_count += 1
        self.seen_coords[coord_string] = self.step_count
        self.perm_seen_coords[coord_string] = self.step_count
    
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
    
    @property
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    def update_seen_map_dict(self):
        # if self.get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.current_map_id - 1
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
        self.badge_count = badge_count
        if self.badge_count > self.prev_badge_count:
            self.save_all_states()
            self.prev_badge_count = self.badge_count
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

    def check_if_level_completed(self):
        self.level_completed = False
        self.level_completed = self.scripted_level_manager()
    
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

    def update_num_mon_in_box(self):
        self.last_num_mon_in_box = self.num_mon_in_box
    
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
            level_reward = self.get_max_n_levels_sum(gym_num_poke, gym_max_level)  # changed, level reward for all 6 pokemon
            if badge_count >= 7 and level_reward > self.max_level_rew and not self.is_in_elite_4:
                level_diff = level_reward - self.max_level_rew
                if level_diff > 6 and self.party_level_post == 0:
                    # self.party_level_post = 0
                    pass
                else:
                    self.party_level_post += level_diff
            self.max_level_rew = max(self.max_level_rew, level_reward)
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
                    ram_map.bit_count(self.game.get_memory_value(i))
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
        self.last_event_rew = last_event_rew
        self.max_event_rew = self.update_max_event_rew()
        state_scores = {
            'event': self.max_event_rew,  
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            'level': self.get_levels_reward(), 
            # 'heal': self.total_healing_rew,
            'op_lvl': self.update_max_op_level(),
            'dead': -self.get_dead_reward(),
            'badge': self.get_badges_reward(),  # 5
            #'op_poke':self.max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': self.reward_scale * seen_poke_count * 400,
            'explore': self.exploration_reward,
            'visited_pokecenter': self.get_visited_pokecenter_reward(),
            'hm': self.get_hm_rewards(),
            # 'hm_move': self.get_hm_move_reward(),  # removed this for now
            'hm_usable': self.get_hm_usable_reward(),
            'trees_cut': self.get_used_cut_coords_reward(),
            'early_done': self.get_early_done_reward(),  # removed
            'special_key_items': self.get_special_key_items_reward(),
            'special': self.get_special_rewards(),
            'heal': self.total_healing_rew,
            # 'level_completed': self.get_level_completed_reward(),
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
    
    @property
    def num_mon_in_box(self):
        if self._num_mon_in_box is None:
            self._num_mon_in_box = ram_map.mem_val(self.game, 0xda80)
        return self._num_mon_in_box
    
    def get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i] and arr2[i] == '1':
                return i
        return -1
    
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
    
    def calculate_event_rewards(self, events_dict, base_reward=10, reward_increment=1, reward_multiplier=1):
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
            detailed_rewards[event_name] = base_reward + (event_value * reward_increment * reward_multiplier)
        return detailed_rewards
    
    def get_dead_reward(self):
        # money_weight = np.clip(self.read_money() / 100_000.0, 0.1, 1.0)
        # return -money_weight * self.died_count
        return 0.1 * self.died_count  # modified from 0.1 to 0.3 after 400k+ steps

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            if self.step_count % 5 == 0:
                prog_string = f's: {self.step_count:7d} env: {self.current_level:1}:{self.env_id:2}'
                # if self.enable_stage_manager:
                #     prog_string += f' stage: {self.stage_manager.stage:2d}'
                for key, val in self.progress_reward.items():
                    if key in ['level', 'explore', 'event', 'dead']:
                        prog_string += f' {key}: {val:6.2f}'
                    elif key in ['level_completed', 'early_done']:
                        continue
                    else:
                        prog_string += f' {key[:10]}: {val:5.2f}'
                prog_string += f' sum: {self.total_reward:5.2f}'
                print(f'\r{prog_string}', end='', flush=True)
        
        # if self.step_count % 1000 == 0:
        #     try:
        #         plt.imsave(
        #             self.s_path / Path(f'curframe_{self.env_id}.jpeg'), 
        #             self.render(reduce_res=False))
        #     except:
        #         pass

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                try:
                    # plt.imsave(
                    #     fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    #     rearrange(obs_memory['image'], 'c h w -> h w c'))
                    plt.imsave(
                        fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                        self.render(reduce_res=False))
                except Exception as e:
                    print(f'error saving final state: {e}')
                # if self.save_state_dir:
                #     self.save_all_states()

        if self.save_video and done:
            self.full_frame_writer.close()
            # modify video name to include final reward (self.total_reward) as prefix
            new_name = f'r{self.total_reward:.4f}_env{self.env_id}_{self.reset_count}.mp4'
            new_path = self.full_frame_write_full_path.parent / Path(new_name)
            self.full_frame_write_full_path.rename(new_path)
            # self.model_frame_writer.close()
        
        if self.save_state_dir:
            if done:
                if self.level_completed:
                    self.save_all_states()
                # elif not self.early_done:
                #     # do not save early done at all, useless info
                #     self.save_all_states(is_failed=True)
                self.record_statistic()
            elif self.level_completed and self.level_manager_eval_mode:
                self.save_all_states()
                self.record_statistic()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.env_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.env_id}.csv.gz'), compression='gzip', mode='a')
    
    def record_statistic(self):
        if self.save_state_dir:
            stats_path = self.save_state_dir / Path('stats')
            stats_path.mkdir(exist_ok=True)
            with open(stats_path / Path(f'level_{self.current_level}.txt'), 'a') as f:
                # append S for success and F for failure
                if self.level_completed:
                    f.write(f'S')
                elif self.early_done:
                    f.write(f'F')
                else:
                    f.write(f'F')


    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0): # 20480
        """Resets the game. Seeding is NOT supported"""
        # BET ADDED
        self.init_map_mem()
        self.init_caches()
        assert len(self.all_events_string) == 2552, f'len(self.all_events_string): {len(self.all_events_string)}'
        self.rewarded_events_string = '0' * 2552
        self.base_event_flags = self.get_base_event_flags()
        
        if self.reset_count == 0:
            print(f'reset count=0')
            load_pyboy_state(self.game, self.load_first_state())
            self.save_all_states()
        
        if self.reset_count % 2 == 0:
            self.load_most_recent_state()
            
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
        self.seen_coords_bet = set()
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
        self.last_health = 1
        self.died_count = 1
        self.last_num_poke = 1
        self.last_num_mon_in_box = 0
        self.visited_pokecenter_list = []
        
        self.used_cut_coords_dict = {}
     

        return self.render(), {}
    
    def get_highest_reward_state_dir_based_on_reset_count(self, dirs_given, weightage=0.01):
        '''
        path_given is all_state_dirs
        all_state_dirs is self.start_from_state_dir.glob('*')
        state folders name as such '{self.total_reward:5.2f}_{session_id}_{self.instance_id}_{self.reset_count}'
        return the state folder with highest total_reward divided by reset_count.
        '''
        if not dirs_given:
            return None
        if weightage <= 0:
            print(f'weightage should be greater than 0, weightage: {weightage}')
            weightage = 0.01
        dirs_given = list(dirs_given)
        # get highest total_reward divided by reset_count
        return max(dirs_given, key=lambda x: float(x.name.split('_')[0]) / (float(x.name.split('_')[-1]) + weightage))
    
    def save_all_states(self):
        # Ensure the hardcoded save path is a Path object
        save_path = Path(HARDCODED_SAVE_PATH)
        # Create the directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Use a timestamp to differentiate the states
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Construct the full path for the new state directory
        state_dir = save_path / f"state_{datetime_str}_{self.total_reward:5.2f}"
        # Create the state directory
        state_dir.mkdir(exist_ok=True)
        
        # Save the state data
        state = {key: getattr(self, key) for key in STATES_TO_SAVE_LOAD}
        with open(state_dir / 'state.pkl', 'wb') as f:
            pickle.dump(state, f)
        # If using a game emulator that has a separate state file, save it as well
        with open(state_dir / 'game_state.state', 'wb') as f:
            self.game.save_state(f)

        print(f"State saved at {state_dir}")
        
    # def save_all_states(self, is_failed=False):
    #     # STATES_TO_SAVE_LOAD = ['recent_frames', 'agent_stats', 'base_explore', 'max_opponent_level',
    #     # 'max_event_rew', 'max_level_rew', 'last_health', 'last_num_poke', 'last_num_mon_in_box', 'total_healing_rew', 
    #     # 'died_count', 'prev_knn_rew', 'visited_pokecenter_list', 'last_10_map_ids', 'last_10_coords', 'past_events_string',
    #     # 'last_10_event_ids', 'early_done', 'step_count', 'past_rewards', 'base_event_flags', 'rewarded_events_string', 
    #     # 'seen_map_dict', '_cut_badge', '_have_hm01', '_can_use_cut', '_surf_badge', '_have_hm03', '_can_use_surf',
    #     # '_have_pokeflute', '_have_silph_scope', 'used_cut_coords_dict', '_last_item_count', '_is_box_mon_higher_level', 
    #     # 'hideout_elevator_maps', 'use_mart_count', 'use_pc_swap_count']
    #     # pyboy state file, 
    #     # state pkl file, 
    #     if not self.save_state_dir:
    #         return
    #     self.save_state_dir.mkdir(exist_ok=True)
    #     print(f'made self.save_state_dir at {self.save_state_dir}.')
    #     # state_dir naming, state_dir/{current_level}/{datetime}_{step_count}_{total_reward:5.2f}/ .state | .pkl
    #     if not is_failed:
    #         level_increment = 1
    #         # if self.level_completed_skip_type == 1:
    #         #     # special case
    #         #     level_increment = 2
    #         state_dir = self.save_state_dir / Path(f'level_{self.current_level + level_increment}')  # + 1 for next level
    #     else:
    #         # create failed folder
    #         state_dir = self.save_state_dir / Path(f'failed')
    #         state_dir.mkdir(exist_ok=True)
    #         state_dir = self.save_state_dir / Path(f'failed/level_{self.current_level}')
    #     state_dir.mkdir(exist_ok=True)
    #     datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     state_dir = state_dir / Path(f'{datetime_str}_{self.step_count}_{self.total_reward:5.2f}')
    #     state_dir.mkdir(exist_ok=True)
    #     # state pkl file all the required variables defined in self.reset()
    #     # recent_frames, agent_stats, base_explore, max_opponent_level, max_event_rew, max_level_rew, last_health, last_num_poke, last_num_mon_in_box, total_healing_rew, died_count, prev_knn_rew, visited_pokecenter_list, last_10_map_ids, last_10_coords, past_events_string, last_10_event_ids, early_done, step_count, past_rewards, base_event_flags, rewarded_events_string, seen_map_dict, _cut_badge, _have_hm01, _can_use_cut, _surf_badge, _have_hm03, _can_use_surf, _have_pokeflute, _have_silph_scope, used_cut_coords_dict, _last_item_count, _is_box_mon_higher_level, hideout_elevator_maps, use_mart_count, use_pc_swap_count
    #     with open(state_dir / Path('state.pkl'), 'wb') as f:
    #         state = {key: getattr(self, key) for key in STATES_TO_SAVE_LOAD}
    #         pickle.dump(state, f)
    #     # pyboy state file
    #     with open(state_dir / Path('state.state'), 'wb') as f:
    #         self.game.save_state(f)
    #         print(f'state.state saved at {state_dir / Path("state.state")}')

    # def load_state(self, state_dir):
    #     # STATES_TO_SAVE_LOAD
    #     save_path = HARDCODED_SAVE_PATH
    #     # List all state directories, sorted by modification time, newest first
    #     state_dir = sorted(save_path.glob('state_*'), key=lambda x: x.stat().st_mtime, reverse=True)

    #     if not state_dir:
    #         print(f"LINE1604 Failure in load_state function: No saved states found.")
            
    #     try:
    #         with open(state_dir / Path('state.state'), 'rb') as f:
    #             self.game.load_state(f)
    #     except:
    #         pass
                
    #         print(f'state.state loaded from {state_dir / Path("state.state")}!')


    def load_most_recent_state(self):
        save_path = Path(HARDCODED_SAVE_PATH)
        # List all state directories, sorted by modification time, newest first
        all_state_dirs = sorted(save_path.glob('state_*'), key=lambda x: x.stat().st_mtime, reverse=True)
        print(f'all_state_dirs: {all_state_dirs}')
        
        if not all_state_dirs:
            print("No saved states found.")
            return

        # Select the most recent state directory
        most_recent_state_dir = all_state_dirs[0]
        print(f"Loading state from {most_recent_state_dir}")

        # Assuming the filenames 'state.state' and 'state.pkl' are consistent
        game_state_path = most_recent_state_dir / 'state.state'
        data_state_path = most_recent_state_dir / 'state.pkl'

        # Load the game state
        if game_state_path.exists():
            with open(game_state_path, 'rb') as f:
                self.game.load_state(f)
                print(f"Game state loaded from {game_state_path}")

        # Load the data state
        if data_state_path.exists():
            with open(data_state_path, 'rb') as f:
                state = pickle.load(f)
                # Apply the loaded state to the environment
                for key, value in state.items():
                    setattr(self, key, value)
                print(f"Data state loaded from {data_state_path}")

            
        # with open(state_dir / Path('state.pkl'), 'rb') as f:
        #     state = pickle.load(f)
            if 'party_level_base' not in state:
                state['party_level_base'] = 0
            if 'party_level_post' not in state:
                state['party_level_post'] = 0
            if 'secret_switch_states' not in state:
                state['secret_switch_states'] = {}
            for key in STATES_TO_SAVE_LOAD:
                # if key == 'secret_switch_states' and key not in state:
                #     self.secret_switch_states = {}
                # else:
                setattr(self, key, state[key])
        self.reset_count = 0
        # self.step_count = 0
        self.early_done = False
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        # self.past_rewards = np.zeros(10240, dtype=np.float32)
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.past_rewards[0] = self.total_reward - self.progress_reward['heal'] - self.get_dead_reward() # - self.get_knn_reward_exclusion() 
        # set all past reward to current total reward, so that the agent will not be penalized for the first step
        self.past_rewards[1:] = self.past_rewards[0] - (self.early_stopping_min_reward * self.reward_scale)
        self.reset_count += 1

    def scripted_level_manager(self):
        # current_level = 0
        selected_level = LEVELS[self.current_level]
        # if self.current_level == 2:
        #     if self.stage_manager.stage > 7:
        #         if self.get_badges() >= 5:
        #             self.level_completed_skip_type = 1
        #             return True
        if 'badge' in selected_level:
            if self.get_badges() < selected_level['badge']:
                # not enough badge
                return False
        if 'event' in selected_level:
            if selected_level['event'] == 'CHAMPION':
                # D867, bit 1
                if not ram_map.read_bit(self.game, 0xD867, 1):
                    # not beaten champion yet
                    return False
        if 'last_pokecenter' in selected_level:
            found = False
            for pokecenter in selected_level['last_pokecenter']:
                if POKECENTER_TO_INDEX_DICT[pokecenter] == self.get_last_pokecenter_id():
                    found = True
                    break
            if not found:
                # not in the last pokecenter
                return False
        # if reached here, means all conditions met
        return True
        # if not self.level_manager_eval_mode:
        #     # if training mode, then return True
        #     return True
        # else:
        #     # if eval mode, increase current_level
        #     self.current_level += 1
        #     return True

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1
        
        if self.save_video:
            self.add_video_frame()
        
        # Exploration reward
        r, c, map_n = ram_map.position(self.game)
        self.seen_coords_bet.add((r, c, map_n))
        
        # Call nimixx api
        self.api.process_game_states() 
        current_bag_items = self.api.items.get_bag_item_ids()
        self.update_cut_badge()
        self.update_surf_badge()
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        
        # print(f'\ngame states: {self.api.process_game_states()},\n bag_items: {current_bag_items},\n {self.update_pokedex()}\n')    

        if 'Fresh Water' in current_bag_items:
            self.has_fresh_water_in_bag = True
            self.has_fresh_water_in_bag_reward = 20
        if 'Lemonade' in current_bag_items:
            self.has_lemonade_in_bag = True
            self.has_lemonade_in_bag_reward = 20
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
        
        # BET: increase exploration after cutting at least 1 tree to encourage exploration vs cut perseveration
        exploration_reward = 0.02 * len(self.seen_coords_bet) if self.used_cut < 1 else 0.1 * len(self.seen_coords_bet) # 0.2 doesn't work (too high??)
        self.exploration_reward = exploration_reward
        
        self.update_heat_map(r, c, map_n)
        if map_n != self.prev_map_n:
            self.used_cut_on_map_n = 0
            self.prev_map_n = map_n
            if map_n not in self.seen_maps:
                self.seen_maps.add(map_n)
            # self.save_state()

        # Level reward
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 50: # 30
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 50 + (self.max_level_sum - 50) / 4 # 30
            
        # Healing and death rewards
        hp = ram_map.hp(self.game)
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
        death_reward = 0
        healing_reward = self.total_healing

        # Badge reward
        badges = ram_map.badges(self.game)
        badges_reward = 10 * badges # 5 BET

        # Save Bill
        bill_state = ram_map.saved_bill(self.game)
        bill_reward = 5 * bill_state
        
        # HM reward
        hm_count = ram_map.get_hm_count(self.game)
        
        # Save state on obtaining hm
        if hm_count >= 1 and self.hm_count == 0:
            # self.save_state()
            self.hm_count = 1
        hm_reward = hm_count * 10
        cut_rew = self.cut * 8 # 10 works - 2 might be better, though 
        
        # BET ADDED TREE REWARDS
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)

        tree_distance_reward = self.detect_and_reward_trees((glob_r, glob_c), map_n, vision_range=5)
        # tree_distance_reward = self.compute_tree_reward((glob_r, glob_c), TREE_POSITIONS_GRID_GLOBAL, map_n) * cut_rew

        # Money 
        money = ram_map.money(self.game)
        
        # Opponent level reward
        max_opponent_level = max(ram_map.opponent(self.game))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = 0.006 * self.max_opponent_level # previously disabled BET
        
        # Event rewards
        events = ram_map.events(self.game)
        self.max_events = max(self.max_events, events)
        event_reward = self.max_events

        # Dojo reward
        dojo_reward = ram_map_leanke.dojo(self.game)
        defeated_fighting_dojo = -1 * int(ram_map.read_bit(self.game, 0xD7B1, 0))
        got_hitmonlee = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 6))
        got_hitmonchan = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 7))
        
        # # Hideout reward
        hideout_reward = ram_map_leanke.hideout(self.game)
        
        # SilphCo rewards
        silph_co_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_silph_co_events(self.game), 
            base_reward=10, reward_increment=1, reward_multiplier=1)
        
        # Dojo rewards
        dojo_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_dojo_events(self.game), 
            base_reward=10, reward_increment=1, reward_multiplier=1)
        
        # Hideout rewards
        hideout_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_hideout_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)

        # Poketower rewards
        poke_tower_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_poke_tower_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)

        # Gym rewards
        gym3_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym3_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)
        gym4_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym4_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)
        gym5_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym5_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)
        gym6_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym6_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)
        gym7_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym7_events(self.game),
            base_reward=10, reward_increment=1, reward_multiplier=1)

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
                    self.cut_coords[coords] = 10 # 10
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

        # Misc
        self.update_pokedex()
        self.update_moves_obtained()
        self.get_badges()
        # For save/load
        self.update_seen_coords()    
        self.update_heal_reward()
        self.update_num_poke()
        self.update_num_mon_in_box()
        self.check_if_level_completed()
        

        bill_capt_rew = ram_map.bill_capt(self.game)
        
        # Cut check 2 - BET ADDED: used cut on tree
        if ram_map.used_cut(self.game) == 61:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write
            if (map_n, r, c) in self.used_cut_coords_set:
                pass
            else:
                self.used_cut += 1

        used_cut_on_tree_rew = 0 # should be 0 to prevent cut abuse
        start_menu = self.seen_start_menu * 0.01
        pokemon_menu = self.seen_pokemon_menu * 0.1
        stats_menu = self.seen_stats_menu * 0.1
        bag_menu = self.seen_bag_menu * 0.1
        cut_coords = sum(self.cut_coords.values()) * 1.0
        cut_tiles = len(self.cut_tiles) * 1.0
        that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu)
    
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
        moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)

        reward = self.reward_scale * (
            event_reward
            + bill_capt_rew
            + seen_pokemon_reward
            + caught_pokemon_reward
            + moves_obtained_reward
            + bill_reward
            + hm_reward
            + level_reward
            + death_reward
            + badges_reward
            + healing_reward
            + exploration_reward 
            + cut_rew
            + that_guy / 2 # reward for cutting an actual tree (but not erika's trees)
            + cut_coords # reward for cutting anything at all
            + cut_tiles # reward for cutting a cut tile, e.g. a patch of grass
            + tree_distance_reward * 0.6 # 1 is too high # 0.25 # 0.5
            + dojo_reward * 5
            + hideout_reward * 5
            + self.has_lemonade_in_bag_reward
            + self.has_fresh_water_in_bag_reward
            + self.has_silph_scope_in_bag_reward
            + self.has_lift_key_in_bag_reward
            + self.has_pokedoll_in_bag_reward
            + self.has_bicycle_in_bag_reward
            + (dojo_events_reward + silph_co_events_reward + 
               hideout_events_reward + poke_tower_events_reward + 
               gym3_events_reward + gym4_events_reward + 
               gym5_events_reward + gym6_events_reward + 
               gym7_events_reward)
            + (gym3_events_reward + gym4_events_reward +
               gym5_events_reward + gym6_events_reward +
               gym7_events_reward)
        )

        # Subtract previous reward
        # TODO: Don't record large cumulative rewards in the first place
        if self.last_reward is None:
            reward = 0
            self.last_reward = 0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward

        info = {}
        done = self.time >= self.max_episode_steps


        if self.save_video and done:
            self.full_frame_writer.close()
        
        # if done:
        #     self.save_state()
        if done or self.time % 10000 == 0:   
            levels = [self.game.get_memory_value(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]       
            info = {
                "pokemon_exploration_map": self.counts_map, # self.explore_map, #  self.counts_map, 
                "stats": {
                    "step": self.time,
                    "x": c,
                    "y": r,
                    "map": map_n,
                    # "map_location": self.get_map_location(map_n),
                    # "max_map_progress": self.max_map_progress,
                    "pcount": int(self.game.get_memory_value(0xD163)),
                    "levels": levels,
                    "levels_sum": sum(levels),
                    # "ptypes": self.read_party(),
                    # "hp": self.read_hp_fraction(),
                    "coord": np.sum(self.counts_map),  # np.sum(self.seen_global_coords),
                    # "map_id": np.sum(self.seen_map_ids),
                    # "npc": sum(self.seen_npcs.values()),
                    # "hidden_obj": sum(self.seen_hidden_objs.values()),
                    "deaths": self.death_count,
                    "deaths_per_episode": self.death_count_per_episode,
                    "badges": float(badges),
                    "self.badge_count": self.badge_count,
                    "badge_1": float(badges >= 1),
                    "badge_2": float(badges >= 2),
                    "badge_3": float(badges >= 3),
                    "badge_4": float(badges >= 4),
                    "badge_5": float(badges >= 5),
                    "badge_6": float(badges >= 6),
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
                    # "cut_coords": sum(self.cut_coords.values()),
                    'pcount': int(ram_map.mem_val(self.game, 0xD163)), 
                    # 'visited_pokecenterr': self.get_visited_pokecenter_reward(),
                    # 'rewards': int(self.total_reward) if self.total_reward is not None else 0,
                    "maps_explored": len(self.seen_maps),
                    "party_size": party_size,
                    "highest_pokemon_level": max(party_levels),
                    "total_party_level": sum(party_levels),
                    "deaths": self.death_count,
                    # "ss_anne_obtained": ss_anne_obtained,
                    "event": events,
                    "money": money,
                    "pokemon_exploration_map": self.counts_map,
                    "seen_npcs_count": len(self.seen_npcs),
                    "seen_pokemon": np.sum(self.seen_pokemon),
                    "caught_pokemon": np.sum(self.caught_pokemon),
                    "moves_obtained": np.sum(self.moves_obtained),
                    "hidden_obj_count": len(self.seen_hidden_objs),
                    "bill_saved": bill_state,
                    "hm_count": hm_count,
                    "cut_taught": self.cut,
                    "badge_1": float(badges >= 1),
                    "badge_2": float(badges >= 2),
                    "badge_3": float(badges >= 3),
                    "maps_explored": np.sum(self.seen_maps),
                    "party_size": party_size,
                    "bill_capt": (bill_capt_rew/5),
                    'cut_coords': cut_coords,
                    'cut_tiles': cut_tiles,
                    'bag_menu': bag_menu,
                    'stats_menu': stats_menu,
                    'pokemon_menu': pokemon_menu,
                    'start_menu': start_menu,
                    'used_cut': self.used_cut,
                    'state_loaded_instead_of_resetting_in_game': self.state_loaded_instead_of_resetting_in_game,
                    'defeated_fighting_dojo': defeated_fighting_dojo,
                    'got_hitmonlee': got_hitmonlee,
                    'got_hitmonchan': got_hitmonchan,
                },
                "reward": {
                    "delta": reward,
                    "event": event_reward,
                    "level": level_reward,
                    "opponent_level": opponent_level_reward,
                    "death": death_reward,
                    "badges": badges_reward,
                    "bill_saved_reward": bill_reward,
                    "hm_count_reward": hm_reward,
                    # "ss_anne_done_reward": ss_anne_state_reward,
                    "healing": healing_reward,
                    "exploration": exploration_reward,
                    # "explore_npcs_reward": explore_npcs_reward,
                    "seen_pokemon_reward": seen_pokemon_reward,
                    "caught_pokemon_reward": caught_pokemon_reward,
                    "moves_obtained_reward": moves_obtained_reward,
                    # "hidden_obj_count_reward": explore_hidden_objs_reward,
                    "used_cut_reward": cut_rew,
                    # "used_cut_on_tree": used_cut_on_tree_rew,
                    "tree_distance_reward": tree_distance_reward,
                    "dojo_reward_old": dojo_reward,
                    "hideout_reward": hideout_reward,
                    "has_fresh_water_in_bag_reward": self.has_fresh_water_in_bag_reward,
                    "has_lemonade_in_bag_reward": self.has_lemonade_in_bag_reward,
                    "has_silph_scope_in_bag_reward": self.has_silph_scope_in_bag_reward,
                    "has_lift_key_in_bag_reward": self.has_lift_key_in_bag_reward,
                    "has_pokedoll_in_bag_reward": self.has_pokedoll_in_bag_reward,
                    "has_bicycle_in_bag_reward": self.has_bicycle_in_bag_reward,
                },
                "detailed_rewards_silph_co": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_silph_co_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                "detailed_rewards_dojo": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_dojo_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                "detailed_rewards_hideout": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_hideout_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                "detailed_rewards_poke_tower": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_poke_tower_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                "detailed_rewards_gyms": {
                    "gym_3_detailed_rewards": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_gym3_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1), 
                },
                    "gym_4_detailed_rewards": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_gym4_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                    "gym_5_detailed_rewards": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_gym5_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                    "gym_6_detailed_rewards": {
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_gym6_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                    "gym_7_detailed_rewards": { 
                    **self.calculate_event_rewards_detailed(
                        ram_map_leanke.monitor_gym7_events(self.game), 
                        base_reward=10, reward_increment=1, reward_multiplier=1),
                },
                },
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
                # "pokemon_exploration_map": self.counts_map, # self.explore_map, #  self.counts_map, 
            }
        
        return self.render(), reward, done, done, info