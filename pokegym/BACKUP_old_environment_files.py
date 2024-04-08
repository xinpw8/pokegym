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

import sys
current_dir = Path(__file__).parent
pufferlib_dir = current_dir.parent.parent
if str(pufferlib_dir) not in sys.path:
    sys.path.append(str(pufferlib_dir))

from stream_agent_wrapper import StreamWrapper



CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
# def get_random_state():
#     state_files = [f for f in os.listdir(STATE_PATH) if f.endswith(".state")]
#     if not state_files:
#         raise FileNotFoundError("No State files found in the specified directory.")
#     return random.choice(state_files)
# state_file = get_random_state()
# randstate = os.path.join(STATE_PATH, state_file)

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


PLAY_STATE_PATH = __file__.rstrip("environment.py") + "just_died_mt_moon.state" # "outside_mt_moon_hp.state" # "Bulbasaur.state" # "celadon_city_cut_test.state"
EXPERIMENTAL_PATH = __file__.rstrip("environment.py") + "lock_1_gym_3.state"
STATE_PATH = __file__.rstrip("environment.py") + "current_state/"

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

    env = StreamWrapper(env, stream_metadata={"user": "localtesty |BET|\n"})

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
                print(f"new Reward: {reward}\n")
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
            
        # self.state_file = get_random_state()
        # self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = STATE_PATH + "lock_1_gym_3.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
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
        self.video_path = Path(f'./videos')
        self.video_path.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(f'./csv')
        self.csv_path.mkdir(parents=True, exist_ok=True)
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
        self.has_lemonade_in_bag = False
        self.has_silph_scope_in_bag = False
        self.has_lift_key_in_bag = False
        self.has_pokedoll_in_bag = False
        self.has_bicycle_in_bag = False
        self.has_lemonade_in_bag_reward = 0
        self.has_silph_scope_in_bag_reward = 0
        self.has_lift_key_in_bag_reward = 0
        self.has_pokedoll_in_bag_reward = 0
        self.has_bicycle_in_bag_reward = 0
        self.gym_info = GYM_INFO
        # Define the specific coordinate ranges to encourage exploration around trees
        self.celadon_tree_attraction_coords = {
            'inside_barrier': [((28, 34), (2, 17)), ((33, 34), (18, 36))],
            'tree_coord': [(32, 35)],
            'outside_box': [((30, 31), (33, 37))]
        }
        self.hideout_seen_coords = {(16, 12, 201), (17, 9, 201), (24, 19, 201), (15, 12, 201), (23, 13, 201), (18, 10, 201), (22, 15, 201), (25, 19, 201), (20, 19, 201), (14, 12, 201), (22, 13, 201), (25, 15, 201), (21, 19, 201), (22, 19, 201), (13, 11, 201), (16, 9, 201), (25, 14, 201), (13, 13, 201), (25, 18, 201), (16, 11, 201), (23, 19, 201), (18, 9, 201), (25, 16, 201), (18, 11, 201), (22, 14, 201), (19, 19, 201), (18, 19, 202), (25, 13, 201), (13, 10, 201), (24, 13, 201), (13, 12, 201), (25, 17, 201), (16, 10, 201), (13, 14, 201)}

        
    
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
        
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203]
        self.saffron_city = [10, 70, 76, 178, 180, 182]
        self.fighting_dojo = [177]
        self.vermilion_gym = [92]
        self.exploration_reward = 0
        self.lock_1_use_counter = 0
        self.last_lock_1_use_counter = 0
        self.last_can_mem_val = -1
        self.can_reward = 0

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
            if event_value > 0:
                detailed_rewards[event_name] = base_reward + (event_value * reward_increment * reward_multiplier)
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

    def print_dict_items(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{key}:")
                self.print_dict_items(value)
            else:
                print(f"{key}: {value}")
                
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
            
    def get_trash_can_memory_reward(self):
        mem_val = ram_map.trash_can_memory(self.game)
        if mem_val == self.last_can_mem_val:
            reward = 0
        else:
            try:
                reward = mem_val / mem_val
                self.last_can_mem_val = mem_val
            except:
                reward = 0
        return reward

    def reset(self, seed=None, options=None, max_episode_steps=20, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        # BET ADDED
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
        self.seen_coords_bet = set()

        self.used_cut_coords_dict = {}
        self.write_hp_for_first_pokemon(300,300)
     

        return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1
        
        # Made every env perseverate on bill tree???
        # self.update_map_id_to_furthest_visited()
        self.restore_party_move_pp()
        self.write_hp_for_first_pokemon(300,300)

        # print(f'self.get_trash_can_memory_reward()={self.can_reward}')
        
        if self.save_video:
            self.add_video_frame()
        
        # Exploration reward
        r, c, map_n = ram_map.position(self.game)
        self.seen_coords.add((r, c, map_n))
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
        
        self.update_heat_map(r, c, map_n)
        if map_n != self.prev_map_n:
            self.used_cut_on_map_n = 0
            self.prev_map_n = map_n
            if map_n not in self.seen_maps:
                self.seen_maps.add(map_n)
            # self.save_state()

        # BET: increase exploration after cutting at least 1 tree to encourage exploration vs cut perseveration
        exploration_reward = 0.02 * len(self.seen_coords_bet) if self.used_cut < 1 else 0.1 * len(self.seen_coords_bet) # 0.2 doesn't work (too high??)
        self.exploration_reward = exploration_reward
        
        special_maps = self.pokehideout + self.saffron_city + [self.fighting_dojo] + [self.vermilion_gym]
        if map_n in special_maps:
            self.exploration_reward = exploration_reward * 2
        
            
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
        self.count_gym_3_lock_1_use()
        lock_1_use_reward = self.lock_1_use_counter * 10
            
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
        defeated_fighting_dojo = 1 * int(ram_map.read_bit(self.game, 0xD7B1, 0))
        got_hitmonlee = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 6))
        got_hitmonchan = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 7))
        
        # Hideout reward
        hideout_reward = ram_map_leanke.hideout(self.game)
        
        # SilphCo rewards
        silph_co_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_silph_co_events(self.game), 
            base_reward=10, reward_increment=10, reward_multiplier=2)
        
        # Dojo rewards
        dojo_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_dojo_events(self.game), 
            base_reward=10, reward_increment=2, reward_multiplier=3)
        
        # Hideout rewards
        hideout_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_hideout_events(self.game),
            base_reward=10, reward_increment=10, reward_multiplier=3)

        # Poketower rewards
        poke_tower_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_poke_tower_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)

        # Gym rewards
        gym3_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym3_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        gym4_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym4_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        gym5_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym5_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        gym6_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym6_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)
        gym7_events_reward = self.calculate_event_rewards(
            ram_map_leanke.monitor_gym7_events(self.game),
            base_reward=10, reward_increment=2, reward_multiplier=1)

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
            + self.exploration_reward 
            + cut_rew
            + that_guy / 2 # reward for cutting an actual tree (but not erika's trees)
            + cut_coords # reward for cutting anything at all
            + cut_tiles # reward for cutting a cut tile, e.g. a patch of grass
            + tree_distance_reward * 0.6 # 1 is too high # 0.25 # 0.5
            + dojo_reward * 5
            + hideout_reward * 5 # woo! extra hideout rewards!!
            + self.has_lemonade_in_bag_reward
            + self.has_silph_scope_in_bag_reward
            + self.has_lift_key_in_bag_reward
            + self.has_pokedoll_in_bag_reward
            + self.has_bicycle_in_bag_reward
            + (dojo_events_reward + silph_co_events_reward + 
               hideout_events_reward + poke_tower_events_reward + 
               gym3_events_reward + gym4_events_reward + 
               gym5_events_reward + gym6_events_reward + 
               gym7_events_reward)
            + self.can_reward
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
        if done or self.time % 1 == 0:   
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
                    "exploration": self.exploration_reward,
                    # "explore_npcs_reward": explore_npcs_reward,
                    "seen_pokemon_reward": seen_pokemon_reward,
                    "caught_pokemon_reward": caught_pokemon_reward,
                    "moves_obtained_reward": moves_obtained_reward,
                    # "hidden_obj_count_reward": explore_hidden_objs_reward,
                    "used_cut_reward": cut_rew,
                    # "used_cut_on_tree": used_cut_on_tree_rew,
                    "tree_distance_reward": tree_distance_reward,
                    "dojo_reward_old": dojo_reward,
                    # "hideout_reward": hideout_reward,
                    "has_lemonade_in_bag_reward": self.has_lemonade_in_bag_reward,
                    "has_silph_scope_in_bag_reward": self.has_silph_scope_in_bag_reward,
                    "has_lift_key_in_bag_reward": self.has_lift_key_in_bag_reward,
                    "has_pokedoll_in_bag_reward": self.has_pokedoll_in_bag_reward,
                    "has_bicycle_in_bag_reward": self.has_bicycle_in_bag_reward,
                },
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
                    "gym_3_lock_1_use_reward": lock_1_use_reward,
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
                # "pokemon_exploration_map": self.counts_map, # self.explore_map, #  self.counts_map, 
            }
            
            # self.print_dict_items(info)
        
        return self.render(), reward, done, done, info


# BET ADDED commented whole thing for testing
# import tracemalloc
# # Suppress annoying warnings
# import warnings
# warnings.filterwarnings("ignore")
# from pathlib import Path
# from pdb import set_trace as T
# # import types
# import uuid
# from gymnasium import Env, spaces
# import numpy as np
# from skimage.transform import resize
# from collections import defaultdict
# import io, os
# # import random
# # import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# import mediapy as media
# import subprocess
# from pokegym import data
# from pokegym.bin.ram_reader.red_ram_api import *
# import random
# from pokegym.constants import *

# from torch.profiler import profile, ProfilerActivity, schedule
# import time

            
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import gray2rgb

# from pokegym.pyboy_binding import (
#     ACTIONS,
#     make_env,
#     open_state_file,
#     load_pyboy_state,
#     run_action_on_emulator,
# )
# from pokegym import ram_map, game_map
# import multiprocessing
# from pokegym.bin.ram_reader.red_memory_battle import *
# from pokegym.bin.ram_reader.red_memory_env import *
# from pokegym.bin.ram_reader.red_memory_items import *
# from pokegym.bin.ram_reader.red_memory_map import *
# from pokegym.bin.ram_reader.red_memory_menus import *
# from pokegym.bin.ram_reader.red_memory_player import *
# from pokegym.bin.ram_reader.red_ram_debug import *
# from enum import IntEnum
# from multiprocessing import Manager
# from .ram_addresses import RamAddress as RAM

# from . import ram_map, game_map, data, ram_map_leanke
# import subprocess
# import time
# import random
# from multiprocessing import Manager
# from pokegym import data
# from collections import defaultdict, deque

# import sys
# # Calculate the path to the directory containing stream_agent_wrapper.py
# current_dir = Path(__file__).parent
# pufferlib_dir = current_dir.parent.parent  # Adjust based on the actual structure

# # Add pufferlib directory to sys.path
# if str(pufferlib_dir) not in sys.path:
#     sys.path.append(str(pufferlib_dir))

# # Now you can import stream_agent_wrapper
# from stream_agent_wrapper import StreamWrapper

# from pokegym.pyboy_binding import Left, Down, Right

# CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
# CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
# CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]

# # List of tree positions in pixel coordinates
# TREE_POSITIONS_PIXELS = [
#     (3184, 3584, 6), # celadon gym 4 (y, x, map_n)
#     (3375, 3392, 6), # celadon right
#     (2528, 3616, 134), # gym 4 middle
#     (2480, 3568, 134), # gym 4 left
#     (2560, 3584, 134), # gym 4 right
#     (1104, 2944, 13), # below pewter 1
#     (1264, 3136, 13), # below pewter 2
#     (1216, 3616, 13), # below pewter 3
#     (1216, 3744, 13), # below pewter 4
#     (1216, 3872, 13), # below pewter 5
#     (1088, 4000, 1), # old man viridian city
#     (992, 4288, 1),  # viridian city left
#     (3984, 4512, 5), # to vermilion city gym
#     (4640, 1392, 36), # near bill's house 
#     (4464, 2176, 20), # cerulean to rock tunnel
#     (5488, 2336, 21), # outside rock tunnel 1
#     (5488, 2368, 21), # outside rock tunnel 2
#     (5488, 2400, 21), # outside rock tunnel 3
#     (5488, 2432, 21)  # outside rock tunnel 4
# ]
# # Convert pixel coordinates to grid coordinates and then to global coordinates
# TREE_POSITIONS_GRID_GLOBAL = [
#     (y//16, x//16) for x, y, map_n in TREE_POSITIONS_PIXELS
# ]
# # print(f'TREE_POSOTIONS_CONVERTED = {TREE_POSITIONS_GRID_GLOBAL}')
# MAPS_WITH_TREES = set(map_n for _, _, map_n in TREE_POSITIONS_PIXELS)
# TREE_COUNT_PER_MAP = {6: 2, 134: 3, 13: 5, 1: 2, 5: 1, 36: 1, 20: 1, 21: 4}

# PLAY_STATE_PATH = __file__.rstrip("environment.py") + "just_died_mt_moon.state" # "outside_mt_moon_hp.state" # "Bulbasaur.state" # "celadon_city_cut_test.state"
# EXPERIMENTAL_PATH = __file__.rstrip("environment.py") + "celadon_city_cut_test.state"

# # Testing environment w/ no AI
# # pokegym.play from pufferlib folder
# def play():
#     """Creates an environment and plays it"""
#     env = Environment(
#         rom_path="pokemon_red.gb",
#         state_path=EXPERIMENTAL_PATH,
#         headless=False,
#         disable_input=False,
#         sound=False,
#         sound_emulated=False,
#         verbose=True,
#     )

#     env = StreamWrapper(env, stream_metadata={"user": "localtesty |BET|\n"})

#     env.reset()
#     env.game.set_emulation_speed(0)

#     # Display available actions
#     print("Available actions:")
#     for idx, action in enumerate(ACTIONS):
#         print(f"{idx}: {action}")

#     # Create a mapping from WindowEvent to action index
#     window_event_to_action = {
#         "PRESS_ARROW_DOWN": 0,
#         "PRESS_ARROW_LEFT": 1,
#         "PRESS_ARROW_RIGHT": 2,
#         "PRESS_ARROW_UP": 3,
#         "PRESS_BUTTON_A": 4,
#         "PRESS_BUTTON_B": 5,
#         "PRESS_BUTTON_START": 6,
#         "PRESS_BUTTON_SELECT": 7,
#         # Add more mappings if necessary
#     }

#     while True:
#         # Get input from pyboy's get_input method
#         input_events = env.game.get_input()
#         env.game.tick()
#         env.render()
#         if len(input_events) == 0:
#             continue
                
#         for event in input_events:
#             event_str = str(event)
#             if event_str in window_event_to_action:
#                 action_index = window_event_to_action[event_str]
#                 observation, reward, done, _, info = env.step(
#                     action_index, # fast_video=False
#                 )
                
#                 # Check for game over
#                 if done:
#                     print(f"{done}")
#                     break

#                 # Additional game logic or information display can go here
#                 print(f"new Reward: {reward}\n")

# class Base:
#     # Shared counter among processes
#     counter_lock = multiprocessing.Lock()
#     counter = multiprocessing.Value('i', 0)
    
#     # Initialize a shared integer with a lock for atomic updates
#     shared_length = multiprocessing.Value('i', 0)  # 'i' for integer
#     lock = multiprocessing.Lock()  # Lock to synchronize access
    
#     # Initialize a Manager for shared BytesIO object
#     manager = Manager()
#     shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data

#     def __init__(
#         self,
#         rom_path="pokemon_red.gb",
#         state_path=EXPERIMENTAL_PATH,
#         headless=True,
#         save_video=False,
#         quiet=False,
#         **kwargs,
#     ):
#         # Increment counter atomically to get unique sequential identifier
#         with Base.counter_lock:
#             env_id = Base.counter.value
#             Base.counter.value += 1
        
#         """Creates a PokemonRed environment"""
#         # Change state_path if you want to load off a different state file to start
#         if state_path is None:
#             state_path = __file__.rstrip("environment.py") + "Bulbasaur.state"
#             # state_path = __file__.rstrip("environment.py") + "Bulbasaur_fast_text_no_battle_animations_fixed_battle.state"
#         # Make the environment
#         self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
#         self.initial_states = [open_state_file(state_path)]
#         self.always_starting_state = [open_state_file(state_path)]
#         self.experimental_state = [open_state_file(EXPERIMENTAL_PATH)]
#         self.save_video = save_video
#         self.headless = headless
#         self.use_screen_memory = True
#         self.screenshot_counter = 0
#         self.step_states = []
#         self.map_n_100_steps = 40
#         self.cat = 0
#         # self.counts_array = np.zeros([256,50,50], dtype=np.uint8)
#         # counts_array_update(arr, map_n, r, c):
#         #     self.counts_array[map_n, r, c] += 1
        
#         # BET nimixx api
#         self.api = Game(self.game) # import this class for api BET
        
#         # Logging initializations
#         with open("/puffertank/0.7/pufferlib/experiments/running_experiment.txt", "r") as file:
#         # with open("experiments/running_experiment.txt", "r") as file:
#         # with open("experiments/test_exp.txt", "r") as file: # for testing video writing BET
#             exp_name = file.read()
#         self.exp_path = Path(f'experiments/{str(exp_name)}')
#         # self.env_id = Path(f'session_{str(uuid.uuid4())[:8]}')
#         self.env_id = env_id
#         self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        
#         # Manually create running_experiment.txt at pufferlib/experiments/running_experiment.txt
#         # Set logging frequency in steps and log_file_aggregator.py path here.
#         # Logging makes a file pokemon_party_log.txt in each environment folder at
#         # pufferlib/experiments/2w31qioa/sessions/{session_uuid8}/pokemon_party_log.txt
#         self.log = True
#         self.stepwise_csv_logging = False
#         self.log_on_reset = True
#         self.log_frequency = 500 # Frequency to log, in steps, if self.log=True and self.log_on_reset=False
#         self.aggregate_frequency = 600
#         self.aggregate_file_path = 'log_file_aggregator.py'
        
#         self.reset_count = 0
#         self.explore_hidden_obj_weight = 1
#         self.initial_wall_time = time.time()
#         self.seen_maps = set()
        
#         # BET ADDED TREE 
#         self.min_distances = {}  # Key: Tree position, Value: minimum distance reached
#         self.rewarded_distances = {}  # Key: Tree position, Value: set of rewarded distances
#         self.used_cut_on_map_n = 0
#         self.seen_map_dict = {}
        
#         # BET ADDED MORE
#         self.past_events_string = self.all_events_string
#         self._all_events_string = ''
#         self.time = 0
#         self.level_reward_badge_scale = 0
#         self.special_exploration_scale = 0
#         self.elite_4_lost = False
#         self.elite_4_early_done = False
#         self.elite_4_started_step = None
#         self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
#         self.reward_scale = 1
#         self.has_lemonade_in_bag = False
#         self.has_silph_scope_in_bag = False
#         self.has_lift_key_in_bag = False
#         self.has_pokedoll_in_bag = False
#         self.has_bicycle_in_bag = False
#         self.has_lemonade_in_bag_reward = 0
#         self.has_silph_scope_in_bag_reward = 0
#         self.has_lift_key_in_bag_reward = 0
#         self.has_pokedoll_in_bag_reward = 0
#         self.has_bicycle_in_bag_reward = 0
#         self.reset_count = 0
#         self.explore_hidden_obj_weight = 1
#         self.pokemon_center_save_states = []
#         self.pokecenters = [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 147, 182]
#         self.used_cut_on_map_n = 0
#         self.save_video = save_video
#         self.headless = headless
#         self.mem_padding = 2
#         self.memory_shape = 80
#         self.use_screen_memory = True
#         self.screenshot_counter = 0
#         self.gym_info = GYM_INFO
#         self.hideout_seen_coords = {(16, 12, 201), (17, 9, 201), (24, 19, 201), (15, 12, 201), (23, 13, 201), (18, 10, 201), (22, 15, 201), (25, 19, 201), (20, 19, 201), (14, 12, 201), (22, 13, 201), (25, 15, 201), (21, 19, 201), (22, 19, 201), (13, 11, 201), (16, 9, 201), (25, 14, 201), (13, 13, 201), (25, 18, 201), (16, 11, 201), (23, 19, 201), (18, 9, 201), (25, 16, 201), (18, 11, 201), (22, 14, 201), (19, 19, 201), (18, 19, 202), (25, 13, 201), (13, 10, 201), (24, 13, 201), (13, 12, 201), (25, 17, 201), (16, 10, 201), (13, 14, 201)}

#         R, C = self.screen.raw_screen_buffer_dims()
#         self.obs_size = (R // 2, C // 2)
#         # self.obs_size = (R // 2, C // 2)

#         if self.use_screen_memory:
#             self.screen_memory = defaultdict(
#                 lambda: np.zeros((255, 255, 1), dtype=np.uint8)
#             )
#             self.obs_size += (4,)
#         else:
#             self.obs_size += (3,)
#         self.observation_space = spaces.Box(
#             low=0, high=255, dtype=np.uint8, shape=self.obs_size
#         )
#         self.action_space = spaces.Discrete(len(ACTIONS))

#     def update_shared_len(self):
#         with self.lock:
#             if len(self.seen_maps) > 5 and (len(self.seen_maps) + 1) > self.shared_length.value:
#                 self.shared_length.value = len(self.seen_maps)
                
#                 # Save the selected game state as a shared BytesIO object
#                 if len(self.initial_states) > 1:
#                     new_state = self.initial_states[-2]
#                 else:
#                     new_state = self.always_starting_state
#                 new_state.seek(0)  # Make sure we're reading from the beginning
#                 self.shared_bytes_io_data[0] = new_state.getvalue()  # Serialize and store
                
#                 print(f"Env {self.env_id}: Updated shared length to {self.shared_length.value} and state.")
    
#     def load_interrupt(self):
#         with self.lock:
#             if len(self.seen_maps) > 5 and self.shared_length.value - len(self.seen_maps) > 3:
#                 with self.lock:
#                     self.reset() # self.load_shared_state()
                
#     def load_shared_state(self):
#         shared_state_data = self.shared_bytes_io_data[0]
#         if shared_state_data:
#             shared_state = io.BytesIO(shared_state_data)
#             load_pyboy_state(self.game, shared_state)

#     def init_hidden_obj_mem(self):
#         self.seen_hidden_objs = set()
    
#     def save_screenshot(self, event, map_n):
#         self.screenshot_counter += 1
#         ss_dir = Path('screenshots')
#         ss_dir.mkdir(exist_ok=True)
#         plt.imsave(
#             # ss_dir / Path(f'ss_{x}_y_{y}_steps_{steps}_{comment}.jpeg'),
#             ss_dir / Path(f'{self.screenshot_counter}_{event}_{map_n}.jpeg'),
#             self.screen.screen_ndarray())  # (144, 160, 3)

#     def save_state_step(self):
#         state = io.BytesIO()
#         state.seek(0)
#         return self.game.save_state(state)
    
#     def save_state(self):
#         state = io.BytesIO()
#         state.seek(0)
#         self.game.save_state(state)
#         self.initial_states.append(state)
#         r, c, map_n = ram_map.position(self.game)
#         print(f'saved a state: steps: {self.time} | map_n: {map_n} | env_id: {self.env_id}\n')
    
#     def load_last_state(self):
#         r, c, map_n = ram_map.position(self.game)
#         print(f'loaded a state: steps: {self.time} | map_n: {map_n} | env_id: {self.env_id}\n')
#         return self.initial_states[-1]
    
#     def load_first_state(self):
#         return self.always_starting_state[0]
    
#     def load_experimental_state(self):
#         return self.experimental_state[0]

#     def reset(self, seed=None, options=None):
#         """Resets the game. Seeding is NOT supported"""
#         return self.screen.screen_ndarray(), {}

#     def detect_and_reward_trees(self, player_grid_pos, map_n, vision_range=5):
#         if map_n not in MAPS_WITH_TREES:
#             # print(f"\nNo trees to interact with in map {map_n}.")
#             return 0.0

#         # Correct the coordinate order: player_x corresponds to glob_c, and player_y to glob_r
#         player_x, player_y = player_grid_pos  # Use the correct order based on your description

#         # print(f"\nPlayer Grid Position: (X: {player_x}, Y: {player_y})")
#         # print(f"Vision Range: {vision_range}")
#         # print(f"Trees in map {map_n}:")

#         tree_counter = 0  # For numbering trees
#         total_reward = 0.0
#         tree_counter = 0
#         for y, x, m in TREE_POSITIONS_PIXELS:
#             if m == map_n:
#                 tree_counter += 1
#                 tree_x, tree_y = x // 16, y // 16

#                 # Adjusting print statement to reflect corrected tree position
#                 corrected_tree_y = tree_y if not (tree_x == 212 and tree_y == 210) else 211
#                 # print(f"  Tree #{tree_counter} Grid Position: (X: {tree_x}, Y: {corrected_tree_y})")

#                 distance = abs(player_x - tree_x) + abs(player_y - corrected_tree_y)
#                 # print(f"  Distance to Tree #{tree_counter}: {distance}")

#                 if distance <= vision_range:
#                     reward = 1 / max(distance, 1)  # Prevent division by zero; assume at least distance of 1
#                     total_reward += reward
#                     # print(f"  Reward for Tree #{tree_counter}: {reward}\n")
#                 else:
#                     pass
#                     # print(f"  Tree #{tree_counter} is outside vision range.\n")

#         # print(f"Total reward from trees: {total_reward}\n")
#         return total_reward
        
#     def compute_tree_reward(self, player_pos, trees_positions, map_n, N=3, p=2, q=1):
#         if map_n not in MAPS_WITH_TREES:
#             # print(f"No cuttable trees in map {map_n}.")
#             return 0.0
        
#         trees_per_current_map_n = TREE_COUNT_PER_MAP[map_n]
#         if self.used_cut_on_map_n >= trees_per_current_map_n:
#             return 0.0

#         if not hasattr(self, 'min_distances'):
#             self.min_distances = {}
#         if not hasattr(self, 'rewarded_distances'):
#             self.rewarded_distances = {}

#         total_reward = 0
#         nearest_trees_features = self.trees_features(player_pos, trees_positions, N)
        
#         for i in range(N):
#             if i < len(nearest_trees_features):  # Ensure there are enough features
#                 distance = nearest_trees_features[i]
            
#             tree_key = (trees_positions[i][0], trees_positions[i][1], map_n)
#             if tree_key not in self.min_distances:
#                 self.min_distances[tree_key] = float('inf')
            
#             if distance < self.min_distances[tree_key] and distance not in self.rewarded_distances.get(tree_key, set()):
#                 self.min_distances[tree_key] = distance
#                 if tree_key not in self.rewarded_distances:
#                     self.rewarded_distances[tree_key] = set()
#                 self.rewarded_distances[tree_key].add(distance)
                
#                 # Adjust reward computation
#                 if distance == 1:  # Maximal reward for being directly adjacent
#                     distance_reward = 1
#                 else:
#                     distance_reward = 1 / (distance ** p)
                
#                 priority_reward = 1 / ((i+1) ** q)
#                 total_reward += distance_reward * priority_reward

#         return total_reward
        
#     def calculate_distance(self, player_pos, tree_pos):
#         """Calculate the Manhattan distance from player to a tree."""
#         dy, dx = np.abs(np.array(tree_pos) - np.array(player_pos))
#         distance = dy + dx  # Manhattan distance for grid movement
#         return distance
    
#     # def calculate_distance_and_angle(self, player_pos, tree_pos):
#     #     """Recalculate the Euclidean distance and angle from player to a tree."""
#     #     # Ensure the player_pos and tree_pos are in (y, x) format
#     #     dy, dx = np.array(tree_pos) - np.array(player_pos)
#     #     distance = np.sqrt(dy**2 + dx**2)
#     #     angle = np.arctan2(dy, dx)  # Angle in radians, considering dy first due to (y, x) ordering
#     #     return distance, angle

#     def trees_features(self, player_pos, trees_positions, N=3):
#         # Calculate distances to all trees
#         distances = [self.calculate_distance(player_pos, pos) for pos in trees_positions]

#         # Sort by distance and select the nearest N
#         nearest_trees = sorted(distances)[:N]
        
#         # Create a flat list of distances for the nearest N trees
#         features = []
#         for distance in nearest_trees:
#             features.append(distance)
            
#         # Pad with zeros if fewer than N trees are available
#         if len(nearest_trees) < N:
#             features.extend([0] * (N - len(features)))
            
#         return features

#     def save_fixed_window(window, file_path="fixed_window.png"):
#         # Check if window is grayscale (2D array)
#         if window.ndim == 2:
#             # Convert grayscale to RGB
#             window_rgb = gray2rgb(window)
#         elif window.ndim == 3 and window.shape[2] == 1:
#             # Convert single channel to RGB
#             window_rgb = np.repeat(window, 3, axis=2)
#         else:
#             # Assume window is already RGB or RGBA
#             window_rgb = window
#         # Save the RGB image
#         plt.imsave(file_path, window_rgb)
#         print(f"Fixed window image saved to {file_path}")

#     # Helps AI explore. Gives reduced view of PyBoy emulator window, centered on player.
#     def get_fixed_window(self, arr, y, x, window_size):
#         height, width, _ = arr.shape
#         h_w, w_w = window_size[0], window_size[1]
#         h_w, w_w = window_size[0] // 2, window_size[1] // 2
#         y_min = max(0, y - h_w)
#         y_max = min(height, y + h_w + (window_size[0] % 2))
#         x_min = max(0, x - w_w)
#         x_max = min(width, x + w_w + (window_size[1] % 2))
#         window = arr[y_min:y_max, x_min:x_max]
#         pad_top = h_w - (y - y_min)
#         pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
#         pad_left = w_w - (x - x_min)
#         pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)
#         return np.pad(
#             window,
#             ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
#             mode="constant",
#         )

#     def render(self):
#         if self.use_screen_memory:
#             r, c, map_n = ram_map.position(self.game)
#             # Update tile map
#             mmap = self.screen_memory[map_n]
#             if 0 <= r <= 254 and 0 <= c <= 254:
#                 mmap[r, c] = 255
                 
#             # Downsamples the screen and retrieves a fixed window from mmap,
#             # then concatenates along the 3rd-dimensional axis (image channel)
#             return np.concatenate(
#                 (
#                     self.screen.screen_ndarray()[::2, ::2],
#                     self.get_fixed_window(mmap, r, c, self.observation_space.shape),
#                 ),
#                 axis=2,
#             )   
#         else:
#             return self.screen.screen_ndarray()[::2, ::2]

#     def step(self, action):
#         run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
#         return self.render(), 0, False, False, {}
        
#     def video(self):
#         video = self.screen.screen_ndarray()
#         return video
    
#     def custom_video(self):
#         custom_vid = self.get_fixed_window()

#     def close(self):
#         self.game.stop(False)
    
#     # BET ADDED
#     @property
#     def all_events_string(self):
#         if not hasattr(self, '_all_events_string'):
#             self._all_events_string = ''  # Default fallback
#             return self._all_events_string
#         else:
#             # cache all events string to improve performance
#             if not self._all_events_string:
#                 event_flags_start = 0xD747
#                 event_flags_end = 0xD886
#                 result = ''
#                 for i in range(event_flags_start, event_flags_end):
#                     result += bin(ram_map.mem_val(self.game, i))[2:].zfill(8)  # .zfill(8)
#                 self._all_events_string = result
#             return self._all_events_string

# class Environment(Base):
#     def __init__(
#         self,
#         rom_path="pokemon_red.gb",
#         state_path=None,
#         headless=True,
#         save_video=False,
#         quiet=False,
#         verbose=True,
#         **kwargs,
#     ):
#         super().__init__(rom_path, state_path, headless, save_video, quiet, **kwargs)
#         # self.menus = Menus(self) # should actually be self.api.menus (Game owns everything)
#         self.last_menu_state = 'None'
#         self.menus_rewards = 0
#         self.sel_cancel = 0
#         self.start_sel = 0
#         self.none_start = 0
#         self.pk_cancel_menu = 0
#         self.pk_menu = 0
#         self.cut_nothing = 0
#         self.different_menu = 'None'

#         self.counts_map = np.zeros((444, 436))
#         self.verbose = True # verbose
#         self.screenshot_counter = 0
#         self.include_conditions = []
#         self.seen_maps_difference = set() # Vestigial - kept for consistency
#         self.seen_maps_times = set()
#         self.current_maps = []
#         self.exclude_map_n = {37, 38, 39, 43, 52, 53, 55, 57} # No rewards for pointless building exploration
#         # self.exclude_map_n_moon = {0, 1, 2, 12, 13, 14, 15, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 193, 68}
#         self.is_dead = False
#         self.talk_to_npc_reward = 0
#         self.talk_to_npc_count = {}
#         self.already_got_npc_reward = set()
#         self.ss_anne_state = False
#         self.seen_npcs = set()
#         self.explore_npc_weight = 1
#         self.last_map = -1
#         self.init_hidden_obj_mem()
#         self.seen_pokemon = np.zeros(152, dtype=np.uint8)
#         self.caught_pokemon = np.zeros(152, dtype=np.uint8)
#         self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
#         self.past_mt_moon = False
#         self.used_cut = 0
#         self.done_counter = 0
#         # self.map_n_reward = 0 
#         self.max_increments = 100 #BET experimental
#         self.got_hm01 = 0

#         # BET ADDED
#         self.saved_states_dict = {}
#         self.seen_maps_no_reward = set()
#         self.seen_coords_no_reward = set()
#         self.seen_map_dict = {}
#         self.is_warping = False
#         self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
#         self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
#         self._all_events_string = ''
        
#         self.got_hm01 = 0
#         self.rubbed_captains_back = 0
#         self.ss_anne_left = 0
#         self.walked_past_guard_after_ss_anne_left = 0
#         self.started_walking_out_of_dock = 0
#         self.walked_out_of_dock = 0
        
#         self.poke_has_cut = 0
#         self.poke_has_flash = 0
#         self.poke_has_fly = 0
#         self.poke_has_surf = 0
#         self.poke_has_strength = 0
#         self.bill_reward = 0
#         self.hm_reward = 0 
#         self.got_hm01_reward = 0
#         self.rubbed_captains_back_reward = 0
#         self.ss_anne_state_reward = 0
#         self.ss_anne_left_reward = 0
#         self.walked_past_guard_after_ss_anne_left_reward = 0
#         self.started_walking_out_of_dock_reward = 0
#         self.explore_npcs_reward = 0
#         self.seen_pokemon_reward = 0
#         self.caught_pokemon_reward = 0
#         self.moves_obtained_reward = 0
#         self.explore_hidden_objs_reward = 0
#         self.poke_has_cut_reward = 0
#         self.poke_has_flash_reward = 0
#         self.poke_has_fly_reward = 0
#         self.poke_has_surf_reward = 0
#         self.poke_has_strength_reward = 0
#         self.used_cut_reward = 0
#         self.walked_out_of_dock_reward = 0
#         self.badges = 0
#         self.badges_reward = 0
#         self.badges_rew = 0
#         self.items_in_bag = 0
#         self.hm_count = 0
#         self.bill_state = 0
#         self.bill_reward = 0
#         self.base_event_flags = sum(
#             ram_map.bit_count(self.game.get_memory_value(i))
#             for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
#         )
#         self.max_event_rew = 0
#         # EXTRA INIT BELOW CUZ LAZY THIS IS JUST TESTING FILE
#         self.counts_map = np.zeros((444, 436))
#         self.death_count = 0
#         self.verbose = verbose
#         self.screenshot_counter = 0
#         self.include_conditions = []
#         self.seen_maps_difference = set()
#         self.current_maps = []
#         self.talk_to_npc_reward = 0
#         self.talk_to_npc_count = {}
#         self.already_got_npc_reward = set()
#         self.ss_anne_state = False
#         self.seen_npcs = set()
#         self.explore_npc_weight = 1
#         self.is_dead = False
#         self.last_map = -1
#         self.log = True
#         self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         self.talk_to_npc_reward = 0
#         self.talk_to_npc_count = {}
#         self.already_got_npc_reward = set()
#         self.ss_anne_state = False
#         self.seen_npcs = set()
#         self.explore_npc_weight = 1
#         self.last_map = -1
#         self.init_hidden_obj_mem()
#         self.seen_pokemon = np.zeros(152, dtype=np.uint8)
#         self.caught_pokemon = np.zeros(152, dtype=np.uint8)
#         self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
#         self.log = False
#         self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
#         self.visited_pokecenter_list = []
#         self._all_events_string = ''
#         self.used_cut_coords_set = set()
#         self.rewarded_coords = set()
#         self.rewarded_position = (0, 0)
#         # self.seen_coords = set() ## moved from reset
#         self.state_loaded_instead_of_resetting_in_game = 0
#         self.badge_count = 0
        
#         # BET ADDED
#         self.saved_states_dict = {}
#         self.seen_maps_no_reward = set()
#         self.seen_coords_no_reward = set()
#         self.seen_map_dict = {}
#         self.is_warping = False
#         self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
#         self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
#         self._all_events_string = ''
#         self.total_healing_rew = 0
#         self.last_health = 1
        
#         self.poketower = [142, 143, 144, 145, 146, 147, 148]
#         self.pokehideout = [199, 200, 201, 202, 203]
#         self.saffron_city = [10, 70, 76, 178, 180, 182]
#         self.fighting_dojo = [177]
#         self.vermilion_gym = [92]
#         self.exploration_reward = 0

#         # #for reseting at 7 resets - leave commented out
#         # self.prev_map_n = None
#         # self.max_events = 0
#         # self.max_level_sum = 0
#         self.max_opponent_level = 0
#         # self.seen_coords = set()
#         # self.seen_maps = set()
#         # self.total_healing = 0
#         # self.last_hp = 1.0
#         # self.last_party_size = 1
#         # self.hm_count = 0
#         # self.cut = 0
#         self.used_cut = 0
#         # self.cut_coords = {}
#         # self.cut_tiles = {} # set([])
#         # self.cut_state = deque(maxlen=3)
#         # self.seen_start_menu = 0
#         # self.seen_pokemon_menu = 0
#         # self.seen_stats_menu = 0
#         # self.seen_bag_menu = 0
#         # self.seen_cancel_bag_menu = 0
#         # self.seen_pokemon = np.zeros(152, dtype=np.uint8)
#         # self.caught_pokemon = np.zeros(152, dtype=np.uint8)
#         # self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)        
        
        
#     def read_hp_fraction(self):
#         hp_sum = sum([ram_map.read_hp(self.game, add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
#         max_hp_sum = sum([ram_map.read_hp(self.game, add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
#         if max_hp_sum:
#             return hp_sum / max_hp_sum
#         else:
#             return 0
    
#     def update_heal_reward(self):
#         cur_health = self.read_hp_fraction()
#         if cur_health > self.last_health:
#             # fixed catching pokemon might treated as healing
#             # fixed leveling count as healing, min heal amount is 4%
#             heal_amount = cur_health - self.last_health
#             if self.last_num_poke == self.read_num_poke() and self.last_health > 0 and heal_amount > 0.04:
#                 if heal_amount > (0.60 / self.read_num_poke()):
#                     # changed to static heal reward
#                     # 1 pokemon from 0 to 100% hp is 0.167 with 6 pokemon
#                     # so 0.1 total heal is around 60% hp
#                     print(f' healed: {heal_amount:.2f}')
#                     self.total_healing_rew += 0.1
#                 # if heal_amount > 0.5:
#                 #     print(f' healed: {heal_amount:.2f}')
#                 #     # self.save_screenshot('healing')
#                 # self.total_healing_rew += heal_amount * 1
#             elif self.last_health <= 0:
#                     self.d
    
#     def get_game_coords(self):
#         return (self.game.get_memory_value(0xD362), self.game.get_memory_value(0xD361), self.game.get_memory_value(0xD35E))
    
#     def init_map_mem(self):
#         # Maybe I should preallocate a giant matrix for all map ids
#         # All map ids have the same size, right?
#         self.seen_coords_tg = {}
#         # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
#         self.seen_map_ids_tg = np.zeros(256)
        
#     def update_seen_coords(self):
#         x_pos, y_pos, map_n = self.get_game_coords()
#         self.seen_coords_tg[(x_pos, y_pos, map_n)] = 1
#         # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
#         self.seen_map_ids_tg[map_n] = 1
    
#     def get_explore_map(self):
#         explore_map = np.zeros((444, 436))
#         for (x, y, map_n), v in self.seen_coords_tg.items():
#             gy, gx = game_map.local_to_global(y, x, map_n)
#             if gy >= explore_map.shape[0] or gx >= explore_map.shape[1]:
#                 print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
#             else:
#                 explore_map[gy, gx] = v

#         return explore_map
    
#     def update_pokedex(self):
#         for i in range(0xD30A - 0xD2F7):
#             caught_mem = self.game.get_memory_value(i + 0xD2F7)
#             seen_mem = self.game.get_memory_value(i + 0xD30A)
#             for j in range(8):
#                 self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
#                 self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   
    
#     def update_moves_obtained(self):
#         # Scan party
#         for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
#             if self.game.get_memory_value(i) != 0:
#                 for j in range(4):
#                     move_id = self.game.get_memory_value(i + j + 8)
#                     if move_id != 0:
#                         if move_id != 0:
#                             self.moves_obtained[move_id] = 1
#         # Scan current box (since the box doesn't auto increment in pokemon red)
#         num_moves = 4
#         box_struct_length = 25 * num_moves * 2
#         for i in range(self.game.get_memory_value(0xda80)):
#             offset = i*box_struct_length + 0xda96
#             if self.game.get_memory_value(offset) != 0:
#                 for j in range(4):
#                     move_id = self.game.get_memory_value(offset + j + 8)
#                     if move_id != 0:
#                         self.moves_obtained[move_id] = 1
    
#     def get_items_in_bag(self, one_indexed=0):
#         first_item = 0xD31E
#         # total 20 items
#         # item1, quantity1, item2, quantity2, ...
#         item_ids = []
#         for i in range(0, 20, 2):
#             item_id = self.game.get_memory_value(first_item + i)
#             if item_id == 0 or item_id == 0xff:
#                 break
#             item_ids.append(item_id + one_indexed)
#         return item_ids
    
#     def poke_count_hms(self):
#         pokemon_info = ram_map.pokemon_l(self.game)
#         pokes_hm_counts = {
#             'Cut': 0,
#             'Flash': 0,
#             'Fly': 0,
#             'Surf': 0,
#             'Strength': 0,
#         }
#         for pokemon in pokemon_info:
#             moves = pokemon['moves']
#             pokes_hm_counts['Cut'] += 'Cut' in moves
#             pokes_hm_counts['Flash'] += 'Flash' in moves
#             pokes_hm_counts['Fly'] += 'Fly' in moves
#             pokes_hm_counts['Surf'] += 'Surf' in moves
#             pokes_hm_counts['Strength'] += 'Strength' in moves
#         return pokes_hm_counts
    
#     def write_to_log(self):
#         pokemon_info = ram_map.pokemon_l(self.game)
#         bag_items = self.api.items.get_bag_item_ids()
#         session_path = self.s_path
#         base_dir = self.exp_path
#         base_dir.mkdir(parents=True, exist_ok=True)
#         session_path.mkdir(parents=True, exist_ok=True)
#         # Writing Pokémon info to session log
#         with open(session_path / self.full_name_log, 'w') as f:
#             for pokemon in pokemon_info:
#                 f.write(f"Slot: {pokemon['slot']}\n")
#                 f.write(f"Name: {pokemon['name']}\n")
#                 f.write(f"Level: {pokemon['level']}\n")
#                 f.write(f"Moves: {', '.join(pokemon['moves'])}\n")
#                 f.write("\n")  # Add a newline between Pokémon
#                 # print(f'WROTE POKEMON LOG TO {session_path}/{self.full_name_log}')
#             f.write("Bag Items:\n")
#             for i, item in enumerate(bag_items,1):
#                 f.write(f"{item}\n")
#         # Writing visited locations and times to log
#         with open(session_path / self.full_name_checkpoint_log, 'w') as f:
#             for location, time_visited in self.seen_maps_times:
#                 f.write(f"Location ID: {location}\n")
#                 f.write(f"Time Visited: {time_visited}\n")
#                 f.write("\n")
#                 # print(f'WROTE CHECKPOINT LOG TO {session_path}/{self.full_name_checkpoint_log}')
    
#     def env_info_to_csv(self, env_id, reset, x, y, map_n, csv_file_path):
#         df = pd.DataFrame([[env_id, reset, x, y, map_n]])
#         df.to_csv(csv_file_path, mode='a', header=not csv_file_path.exists(), index=False)
        
#     def write_env_info_to_csv(self):
#         x, y, map_n = ram_map.position(self.game)
#         base_dir = self.exp_path
#         reset = self.reset_count
#         env_id = self.env_id
#         csv_file_path = base_dir / "steps_map.csv"

#         self.env_info_to_csv(env_id, reset, x, y, map_n, csv_file_path)
    
#     def get_hm_rewards(self):
#         hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
#         items = self.get_items_in_bag()
#         total_hm_cnt = 0
#         for hm_id in hm_ids:
#             if hm_id in items:
#                 total_hm_cnt += 1
#         return total_hm_cnt * 1

#     def add_video_frame(self):
#         self.full_frame_writer.add_image(self.video())
        
#     def add_custom_frame(self):
#         r, c, map_n = ram_map.position(self.game)
#         mmap = self.screen_memory[map_n]
#         fixed_window = self.get_fixed_window(mmap, r, c, self.observation_space.shape)
#         fixed_window = np.squeeze(fixed_window, axis=2)
#         self.full_frame_writer.add_image(fixed_window)
             
#     def update_heat_map(self, r, c, current_map):
#         '''
#         Updates the heat map based on the agent's current position.
#         Args:
#             r (int): global y coordinate of the agent's position.
#             c (int): global x coordinate of the agent's position.
#             current_map (int): ID of the current map (map_n)
#         Updates the counts_map to track the frequency of visits to each position on the map.
#         '''
#         # Convert local position to global position
#         try:
#             glob_r, glob_c = game_map.local_to_global(r, c, current_map)
#         except IndexError:
#             print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
#             glob_r = 0
#             glob_c = 0

#         # Update heat map based on current map
#         if self.last_map == current_map or self.last_map == -1:

#         # Increment count for current global position
#             try:
#                 self.counts_map[glob_r, glob_c] += 1
#             except:
#                 pass
#         else:
#             # Reset count for current global position if it's a new map for warp artifacts
#             self.counts_map[(glob_r, glob_c)] = -1

#     # print(f'counts_map={self.counts_map}, shape={np.shape(self.counts_map)}, size={np.size(self.counts_map)}, sum={np.sum(self.counts_map)}')

#         # Update last_map for the next iteration
#         self.last_map = current_map

#     # def pass_states(self):
#     #     num_envs = 64
#     #     step_length = 1310720 / num_envs
#     #     if self.reset_count == step_length:
#     #        state = self.save_state_step()
#     #        return state
    
#     # def find_neighboring_npc(self, npc_bank, npc_id, player_direction, player_x, player_y) -> int:
#     #     npc_y = ram_map.npc_y(self.game, npc_id, npc_bank)
#     #     npc_x = ram_map.npc_x(self.game, npc_id, npc_bank)
#     #     if (
#     #         (player_direction == 0 and npc_x == player_x and npc_y > player_y) or
#     #         (player_direction == 4 and npc_x == player_x and npc_y < player_y) or
#     #         (player_direction == 8 and npc_y == player_y and npc_x < player_x) or
#     #         (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
#     #     ):
#     #         # Manhattan distance
#     #         return abs(npc_y - player_y) + abs(npc_x - player_x)
#     #     return 1000
    
#     def menu_rewards(self):
#         # print(f'self.last_menu_state={self.last_menu_state}')
#         if self.got_hm01 > 0:
#             menu_state = self.api.game_state.name

#             start_menu_pk = 'START_MENU_POKEMON'
#             select_pk_menu = 'SELECT_POKEMON_'
#             pokecenter_cancel_menu = 'POKECENTER_CANCEL'
#             menu_reward_val = 0
#             if menu_state == 'EXPLORING':
#                 self.last_menu_state = 'EXPLORING'
#                 return menu_reward_val
#             if menu_state != self.last_menu_state:
#                 self.different_menu = True
#             # Reward cutting (trying to use Cut) even if cutting nothing
#             # Menu state stays the same for 'Cut failed' dialogue vs changing to SELECT_POKEMON_
#             if menu_state == 'POKECENTER_CANCEL' and self.different_menu == True:
#                 self.cut_nothing += 1
#                 menu_reward_val += 0.0005 / (self.cut_nothing ** 2)     
            
#             if start_menu_pk in menu_state:
#                 # print(f'{start_menu_pk} in menu_state=True')
#                 if self.last_menu_state == 'None' or self.last_menu_state == 'EXPLORING':
#                     self.none_start += 1
#                     menu_reward_val += 0.000055 / (self.none_start ** 2)
#                 self.last_menu_state = start_menu_pk
                
#             if select_pk_menu in menu_state and pokecenter_cancel_menu not in self.last_menu_state:
#                 # print(f'{select_pk_menu} in menu_state=True')
#                 self.pk_menu += 1
#                 if self.last_menu_state == start_menu_pk:
#                     self.start_sel += 1
#                     menu_reward_val += 0.000055 / (self.start_sel ** 2)
#                 self.last_menu_state = select_pk_menu
#                 menu_reward_val += 0.000055 / (self.pk_menu ** 2)   
                
#             if pokecenter_cancel_menu in menu_state and pokecenter_cancel_menu not in self.last_menu_state:
#                 # print(f'{pokecenter_cancel_menu} in menu_state:=True')
#                 self.pk_cancel_menu += 1
#                 if self.last_menu_state == select_pk_menu:
#                     self.sel_cancel += 1
#                     menu_reward_val += 0.000055 / (self.sel_cancel ** 2)
#                 self.last_menu_state = pokecenter_cancel_menu
#                 menu_reward_val += 0.000055 / (self.pk_cancel_menu ** 2)

#         else:
#             menu_reward_val = 0
#         return menu_reward_val

#     def current_coords(self):
#         return self.last_10_coords[0]
    
#     def current_map_id(self):
#         return self.last_10_map_ids[0, 0]
    
#     def update_seen_map_dict(self):
#         # if self.get_minimap_warp_obs()[4, 4] != 0:
#         #     return
#         cur_map_id = self.current_map_id() - 1
#         x, y = self.current_coords()
#         if cur_map_id not in self.seen_map_dict:
#             self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
#         # # do not update if is warping
#         if not self.is_warping:
#             if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
#                 self.stuck_cnt += 1
#                 print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
#                 if self.stuck_cnt > 50:
#                     print(f'stucked for > 50 steps, force ES')
#                     self.early_done = True
#                     self.stuck_cnt = 0
#                 # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
#             else:
#                 self.stuck_cnt = 0
#                 self.seen_map_dict[cur_map_id][y, x] = self.time

#     def poke_count_hms(self):
#         pokemon_info = ram_map.pokemon_l(self.game)
#         pokes_hm_counts = {
#             'Cut': 0,
#             'Flash': 0,
#             'Fly': 0,
#             'Surf': 0,
#             'Strength': 0,
#         }
#         for pokemon in pokemon_info:
#             moves = pokemon['moves']
#             pokes_hm_counts['Cut'] += 'Cut' in moves
#             pokes_hm_counts['Flash'] += 'Flash' in moves
#             pokes_hm_counts['Fly'] += 'Fly' in moves
#             pokes_hm_counts['Surf'] += 'Surf' in moves
#             pokes_hm_counts['Strength'] += 'Strength' in moves
#         return pokes_hm_counts
    
#     def get_hm_rewards(self):
#         hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
#         items = self.get_items_in_bag()
#         total_hm_cnt = 0
#         for hm_id in hm_ids:
#             if hm_id in items:
#                 total_hm_cnt += 1
#         return total_hm_cnt * 1
            
#     def add_video_frame(self):
#         self.full_frame_writer.add_image(self.video())

#     def get_game_coords(self):
#         return (ram_map.mem_val(self.game, 0xD362), ram_map.mem_val(self.game, 0xD361), ram_map.mem_val(self.game, 0xD35E))
    
#     def check_if_in_start_menu(self) -> bool:
#         return (
#             ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             and ram_map.mem_val(self.game, 0xFF8C) == 6
#             and ram_map.mem_val(self.game, 0xCF94) == 0
#         )

#     def check_if_in_pokemon_menu(self) -> bool:
#         return (
#             ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             and ram_map.mem_val(self.game, 0xFF8C) == 6
#             and ram_map.mem_val(self.game, 0xCF94) == 2
#         )

#     def check_if_in_stats_menu(self) -> bool:
#         return (
#             ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0)
            
#     def update_heat_map(self, r, c, current_map):
#         '''
#         Updates the heat map based on the agent's current position.

#         Args:
#             r (int): global y coordinate of the agent's position.
#             c (int): global x coordinate of the agent's position.
#             current_map (int): ID of the current map (map_n)

#         Updates the counts_map to track the frequency of visits to each position on the map.
#         '''
#         # Convert local position to global position
#         try:
#             glob_r, glob_c = game_map.local_to_global(r, c, current_map)
#         except IndexError:
#             print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
#             glob_r = 0
#             glob_c = 0

#         # Update heat map based on current map
#         if self.last_map == current_map or self.last_map == -1:
#             # Increment count for current global position
#                 try:
#                     self.counts_map[glob_r, glob_c] += 1
#                 except:
#                     pass
#         else:
#             # Reset count for current global position if it's a new map for warp artifacts
#             self.counts_map[(glob_r, glob_c)] = -1

#         # Update last_map for the next iteration
#         self.last_map = current_map

#     def check_if_in_bag_menu(self) -> bool:
#         return (
#             ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             # and ram_map.mem_val(self.game, 0xFF8C) == 6 # only sometimes
#             and ram_map.mem_val(self.game, 0xCF94) == 3
#         )

#     def check_if_cancel_bag_menu(self, action) -> bool:
#         return (
#             action == WindowEvent.PRESS_BUTTON_A
#             and ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             # and ram_map.mem_val(self.game, 0xFF8C) == 6
#             and ram_map.mem_val(self.game, 0xCF94) == 3
#             and ram_map.mem_val(self.game, 0xD31D) == ram_map.mem_val(self.game, 0xCC36) + ram_map.mem_val(self.game, 0xCC26)
#         )
        
#     def update_reward(self, new_position):
#         """
#         Update and determine if the new_position should be rewarded based on every 10 steps
#         taken within the same map considering the Manhattan distance.

#         :param new_position: Tuple (glob_r, glob_c, map_n) representing the new global position and map identifier.
#         """
#         should_reward = False
#         new_glob_r, new_glob_c, new_map_n = new_position

#         # Check if the new position should be rewarded compared to existing positions on the same map
#         for rewarded_position in self.rewarded_coords:
#             rewarded_glob_r, rewarded_glob_c, rewarded_map_n = rewarded_position
#             if new_map_n == rewarded_map_n:
#                 distance = abs(rewarded_glob_r - new_glob_r) + abs(rewarded_glob_c - new_glob_c)
#                 if distance >= 10:
#                     should_reward = True
#                     break

#         if should_reward:
#             self.rewarded_coords.add(new_position)
            
#     def check_bag_for_silph_scope(self):
#         if 0x4A in self.get_items_in_bag():
#             if 0x48 in self.get_items_in_bag():
#                 self.have_silph_scope = True
#                 return self.have_silph_scope
            
#     def current_coords(self):
#         return self.last_10_coords[0]
    
#     def current_map_id(self):
#         return self.last_10_map_ids[0, 0]
    
#     def update_seen_map_dict(self):
#         # if self.get_minimap_warp_obs()[4, 4] != 0:
#         #     return
#         cur_map_id = self.current_map_id() - 1
#         x, y = self.current_coords()
#         if cur_map_id not in self.seen_map_dict:
#             self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
#         # do not update if is warping
#         if not self.is_warping:
#             if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
#                 self.stuck_cnt += 1
#                 # print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
#                 if self.stuck_cnt > 50:
#                     print(f'stucked for > 50 steps, force ES')
#                     self.early_done = True
#                     self.stuck_cnt = 0
#                 # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
#             else:
#                 self.stuck_cnt = 0
#                 self.seen_map_dict[cur_map_id][y, x] = self.time

#     def get_badges(self):
#         badge_count = ram_map.bit_count(ram_map.mem_val(self.game, 0xD356))
#         # return badge_count
#         if badge_count < 8 or self.elite_4_lost or self.elite_4_early_done:
#             return badge_count
#         else:
#             # LORELEIS D863, bit 1
#             # BRUNOS D864, bit 1
#             # AGATHAS D865, bit 1
#             # LANCES D866, bit 1
#             # CHAMPION D867, bit 1
#             elite_four_event_addr_bits = [
#                 [0xD863, 1],  # LORELEIS
#                 [0xD864, 1],  # BRUNOS
#                 [0xD865, 1],  # AGATHAS
#                 [0xD866, 1],  # LANCES
#                 [0xD867, 1],  # CHAMPION
#             ]
#             elite_4_extra_badges = 0
#             for addr_bit in elite_four_event_addr_bits:
#                 if ram_map.read_bit(self.game, addr_bit[0], addr_bit[1]):
#                     elite_4_extra_badges += 1
#             return 8 + elite_4_extra_badges

#     def get_levels_sum(self):
#         poke_levels = [max(ram_map.mem_val(self.game, a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
#         return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
#     def read_event_bits(self):
#         return [
#             int(bit)
#             for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
#             for bit in f"{ram_map.read_bit(self.game, i):08b}"
#         ]
    
#     def read_num_poke(self):
#         return ram_map.mem_val(self.game, 0xD163)
    
#     def update_num_poke(self):
#         self.last_num_poke = self.read_num_poke()
    
#     def get_max_n_levels_sum(self, n, max_level):
#         num_poke = self.read_num_poke()
#         poke_level_addresses = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
#         poke_levels = [max(min(ram_map.mem_val(self.game, a), max_level) - 2, 0) for a in poke_level_addresses[:num_poke]]
#         return max(sum(sorted(poke_levels)[-n:]) - 4, 0)
    
#     @property
#     def is_in_elite_4(self):
#         return self.current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]
    
#     def get_levels_reward(self):
#         if not self.level_reward_badge_scale:
#             level_sum = self.get_levels_sum()
#             self.max_level_rew = max(self.max_level_rew, level_sum)
#         else:
#             badge_count = min(self.get_badges(), 8)
#             gym_next = self.gym_info[badge_count]
#             gym_num_poke = gym_next['num_poke']
#             gym_max_level = gym_next['max_level'] * self.level_reward_badge_scale
#             level_reward = self.get_max_n_levels_sum(gym_num_poke, gym_max_level)  # changed, level reward for all 6 pokemon
#             if badge_count >= 7 and level_reward > self.max_level_rew and not self.is_in_elite_4:
#                 level_diff = level_reward - self.max_level_rew
#                 if level_diff > 6 and self.party_level_post == 0:
#                     # self.party_level_post = 0
#                     pass
#                 else:
#                     self.party_level_post += level_diff
#             self.max_level_rew = max(self.max_level_rew, level_reward)
#         return ((self.max_level_rew - self.party_level_post) * 0.5) + (self.party_level_post * 2.0)
#         # return self.max_level_rew * 0.5  # 11/11-3 changed: from 0.5 to 1.0
    
#     def get_special_key_items_reward(self):
#         items = self.get_items_in_bag()
#         special_cnt = 0
#         # SPECIAL_KEY_ITEM_IDS
#         for item_id in SPECIAL_KEY_ITEM_IDS:
#             if item_id in items:
#                 special_cnt += 1
#         return special_cnt * 1.0
    
#     def get_all_events_reward(self):
#         # adds up all event flags, exclude museum ticket
#         return max(
#             sum(
#                 [
#                     ram_map.bit_count(self.game.get_memory_value(i))
#                     for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
#                 ]
#             )
#             - self.base_event_flags
#             - int(ram_map.read_bit(self.game, *ram_map.MUSEUM_TICKET_ADDR)),
#             0,
#         )
        
#     def update_max_event_rew(self):
#         cur_rew = self.get_all_events_reward()
#         self.max_event_rew = max(cur_rew, self.max_event_rew)
#         return self.max_event_rew
    
#     def update_max_op_level(self):
#         #opponent_level = ram_map.mem_val(self.game, 0xCFE8) - 5 # base level
#         opponent_level = max([ram_map.mem_val(self.game, a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
#         #if opponent_level >= 7:
#         #    self.save_screenshot('highlevelop')
#         self.max_opponent_level = max(self.max_opponent_level, opponent_level)
#         return self.max_opponent_level * 0.1  # 0.1
    
#     def get_badges_reward(self):
#         num_badges = self.get_badges()
#         # if num_badges < 3:
#         #     return num_badges * 5
#         # elif num_badges > 2:
#         #     return 10 + ((num_badges - 2) * 10)  # reduced from 20 to 10
#         if num_badges < 9:
#             return num_badges * 5
#         elif num_badges < 13:  # env19v2 PPO23
#             return 40  # + ((num_badges - 8) * 1)
#         else:
#             return 40 + 10
#         # return num_badges * 5  # env18v4

#     def get_last_pokecenter_list(self):
#         pc_list = [0, ] * len(self.pokecenter_ids)
#         last_pokecenter_id = self.get_last_pokecenter_id()
#         if last_pokecenter_id != -1:
#             pc_list[last_pokecenter_id] = 1
#         return pc_list
    
#     def get_last_pokecenter_id(self):
        
#         last_pokecenter = ram_map.mem_val(self.game, 0xD719)
#         # will throw error if last_pokecenter not in pokecenter_ids, intended
#         if last_pokecenter == 0:
#             # no pokecenter visited yet
#             return -1
#         if last_pokecenter not in self.pokecenter_ids:
#             print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
#             return -1
#         else:
#             return self.pokecenter_ids.index(last_pokecenter)   
    
#     def get_special_rewards(self):
#         rewards = 0
#         rewards += len(self.hideout_elevator_maps) * 2.0
#         bag_items = self.get_items_in_bag()
#         if 0x2B in bag_items:
#             # 6.0 full mansion rewards + 1.0 extra key items rewards
#             rewards += 7.0
#         return rewards
    
#     def get_hm_usable_reward(self):
#         total = 0
#         if self.can_use_cut:
#             total += 1
#         if self.can_use_surf:
#             total += 1
#         return total * 2.0
    
#     def get_special_key_items_reward(self):
#         items = self.get_items_in_bag()
#         special_cnt = 0
#         # SPECIAL_KEY_ITEM_IDS
#         for item_id in SPECIAL_KEY_ITEM_IDS:
#             if item_id in items:
#                 special_cnt += 1
#         return special_cnt * 1.0
    
#     def get_used_cut_coords_reward(self):
#         return len(self.used_cut_coords_dict) * 0.2
    
#     def get_party_moves(self):
#         # first pokemon moves at D173
#         # 4 moves per pokemon
#         # next pokemon moves is 44 bytes away
#         first_move = 0xD173
#         moves = []
#         for i in range(0, 44*6, 44):
#             # 4 moves per pokemon
#             move = [ram_map.mem_val(self.game, first_move + i + j) for j in range(4)]
#             moves.extend(move)
#         return moves
    
#     def get_hm_move_reward(self):
#         all_moves = self.get_party_moves()
#         hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
#         hm_move_count = 0
#         for hm_move in hm_moves:
#             if hm_move in all_moves:
#                 hm_move_count += 1
#         return hm_move_count * 1.5
    
#     def update_visited_pokecenter_list(self):
#         last_pokecenter_id = self.get_last_pokecenter_id()
#         if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
#             self.visited_pokecenter_list.append(last_pokecenter_id)

#     def get_early_done_reward(self):
#         return self.elite_4_early_done * -0.3
    
#     def get_visited_pokecenter_reward(self):
#         # reward for first time healed in pokecenter
#         return len(self.visited_pokecenter_list) * 2     
    
#     def get_game_state_reward(self, print_stats=False):
#         # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
#         # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
#         '''
#         num_poke = ram_map.mem_val(self.game, 0xD163)
#         poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
#         #money = self.read_money() - 975 # subtract starting money
#         seen_poke_count = sum([self.bit_count(ram_map.mem_val(self.game, i)) for i in range(0xD30A, 0xD31D)])
#         all_events_score = sum([self.bit_count(ram_map.mem_val(self.game, i)) for i in range(0xD747, 0xD886)])
#         oak_parcel = self.read_bit(0xD74E, 1) 
#         oak_pokedex = self.read_bit(0xD74B, 5)
#         opponent_level = ram_map.mem_val(self.game, 0xCFF3)
#         self.max_opponent_level = max(self.max_opponent_level, opponent_level)
#         enemy_poke_count = ram_map.mem_val(self.game, 0xD89C)
#         self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
#         if print_stats:
#             print(f'num_poke : {num_poke}')
#             print(f'poke_levels : {poke_levels}')
#             print(f'poke_xps : {poke_xps}')
#             #print(f'money: {money}')
#             print(f'seen_poke_count : {seen_poke_count}')
#             print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
#         '''
#         last_event_rew = self.max_event_rew
#         self.max_event_rew = self.update_max_event_rew()
#         state_scores = {
#             'event': self.max_event_rew,  
#             #'party_xp': self.reward_scale*0.1*sum(poke_xps),
#             'level': self.get_levels_reward(), 
#             # 'heal': self.total_healing_rew,
#             'op_lvl': self.update_max_op_level(),
#             # 'dead': -self.get_dead_reward(),
#             'badge': self.get_badges_reward(),  # 5
#             #'op_poke':self.max_opponent_poke * 800,
#             #'money': money * 3,
#             #'seen_poke': self.reward_scale * seen_poke_count * 400,
#             # 'explore': self.get_knn_reward(last_event_rew),
#             'visited_pokecenter': self.get_visited_pokecenter_reward(),
#             'hm': self.get_hm_rewards(),
#             # 'hm_move': self.get_hm_move_reward(),  # removed this for now
#             'hm_usable': self.get_hm_usable_reward(),
#             'trees_cut': self.get_used_cut_coords_reward(),
#             'early_done': self.get_early_done_reward(),  # removed
#             'special_key_items': self.get_special_key_items_reward(),
#             'special': self.get_special_rewards(),
#             'heal': self.total_healing_rew,
#         }

#         # multiply by reward scale
#         state_scores = {k: v * self.reward_scale for k, v in state_scores.items()}
        
#         return state_scores
    
#     # BET ADDING A BUNCH OF STUFF
#     def minor_patch_victory_road(self):
#         address_bits = [
#             # victory road
#             [0xD7EE, 0],
#             [0xD7EE, 7],
#             [0xD813, 0],
#             [0xD813, 6],
#             [0xD869, 7],
#         ]
#         for ab in address_bits:
#             event_value = ram_map.mem_val(self.game, ab[0])
#             ram_map.write_mem(self.game, ab[0], ram_map.set_bit(event_value, ab[1]))
    
#     def update_last_10_map_ids(self):
#         current_modified_map_id = ram_map.mem_val(self.game, 0xD35E) + 1
#         # check if current_modified_map_id is in last_10_map_ids
#         if current_modified_map_id == self.last_10_map_ids[0][0]:
#             return
#         else:
#             # if self.last_10_map_ids[0][0] != 0:
#             #     print(f'map changed from {MAP_ID_REF[self.last_10_map_ids[0][0] - 1]} to {MAP_ID_REF[current_modified_map_id - 1]} at step {self.step_count}')
#             self.last_10_map_ids = np.roll(self.last_10_map_ids, 1, axis=0)
#             self.last_10_map_ids[0] = [current_modified_map_id, self.step_count]
#             map_id = current_modified_map_id - 1
#             if map_id in [0x6C, 0xC2, 0xC6, 0x22]:
#                 self.minor_patch_victory_road()
#             # elif map_id == 0x09:
#             if map_id not in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
#                 if self.last_10_map_ids[1][0] - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
#                     # lost in elite 4
#                     self.elite_4_lost = True
#                     self.elite_4_started_step = None
#             if map_id == 0xF5:
#                 # elite four first room
#                 # reset elite 4 lost flag
#                 if self.elite_4_lost:
#                     self.elite_4_lost = False
#                 if self.elite_4_started_step is None:
#                     self.elite_4_started_step = self.step_count
    
#     def get_event_rewarded_by_address(self, address, bit):
#         # read from rewarded_events_string
#         event_flags_start = 0xD747
#         event_pos = address - event_flags_start
#         # bit is reversed
#         # string_pos = event_pos * 8 + bit
#         string_pos = event_pos * 8 + (7 - bit)
#         return self.rewarded_events_string[string_pos] == '1'
    
#     def init_caches(self):
#         # for cached properties
#         self._all_events_string = ''
#         self._battle_type = None
#         self._cur_seen_map = None
#         self._minimap_warp_obs = None
#         self._is_warping = None
#         self._items_in_bag = None
#         self._minimap_obs = None
#         self._minimap_sprite = None
#         self._bottom_left_screen_tiles = None
#         self._num_mon_in_box = None
    
#     def update_cut_badge(self):
#         if not self._cut_badge:
#             # print(f"Attempting to read bit from addr: {RAM.wObtainedBadges.value}, which is type: {type(RAM.wObtainedBadges.value)}")
#             self._cut_badge = ram_map.read_bit(self.game, RAM.wObtainedBadges.value, 1) == 1

#     def update_surf_badge(self):
#         if not self._cut_badge:
#             return
#         if not self._surf_badge:
#             self._surf_badge = ram_map.read_bit(self.game, RAM.wObtainedBadges.value, 4) == 1   

#     def update_last_10_coords(self):
#         current_coord = np.array([ram_map.mem_val(self.game, 0xD362), ram_map.mem_val(self.game, 0xD361)])
#         # check if current_coord is in last_10_coords
#         if (current_coord == self.last_10_coords[0]).all():
#             return
#         else:
#             self.last_10_coords = np.roll(self.last_10_coords, 1, axis=0)
#             self.last_10_coords[0] = current_coord
    
#     @property
#     def can_use_surf(self):
#         if not self._can_use_surf:
#             if self._surf_badge:
#                 if not self._have_hm03:
#                     self._have_hm03 = 0xC6 in self.get_items_in_bag()
#                 if self._have_hm03:
#                     self._can_use_surf = True
#         return self._can_use_surf
    
#     @property
#     def can_use_cut(self):
#         # return ram_map.mem_val(self.game, 0xD2E2) == 1
#         # check badge, store last badge count, if changed, check if can use cut, bit 1, save permanently
#         if not self._can_use_cut:
#             if self._cut_badge:
#                 if not self._have_hm01:
#                     self._have_hm01 = 0xc4 in self.get_items_in_bag()
#                 if self._have_hm01:
#                     self._can_use_cut = True
#             # self._can_use_cut = self._cut_badge is True and 0xc4 in self.get_items_in_bag()
#         return self._can_use_cut
    
#     @property
#     def have_silph_scope(self):
#         if self.can_use_cut and not self._have_silph_scope:
#             self._have_silph_scope = 0x48 in self.get_items_in_bag()
#         return self._have_silph_scope
    
#     @property
#     def can_use_flute(self):
#         if self.can_use_cut and not self._have_pokeflute:
#             self._have_pokeflute = 0x49 in self.get_items_in_bag()
#         return self._have_pokeflute
    
#     def get_base_event_flags(self):
#         # event patches
#         # 1. triggered EVENT_FOUND_ROCKET_HIDEOUT 
#         # event_value = ram_map.mem_val(self.game, 0xD77E)  # bit 1
#         # ram_map.write_mem(self.game, 0xD77E, ram_map.set_bit(event_value, 1))
#         # 2. triggered EVENT_GOT_TM13 , fresh_water trade
#         event_value = ram_map.mem_val(self.game, 0xD778)  # bit 4
#         ram_map.write_mem(self.game, 0xD778, ram_map.set_bit(event_value, 4))
#         # address_bits = [
#         #     # seafoam islands
#         #     [0xD7E8, 6],
#         #     [0xD7E8, 7],
#         #     [0xD87F, 0],
#         #     [0xD87F, 1],
#         #     [0xD880, 0],
#         #     [0xD880, 1],
#         #     [0xD881, 0],
#         #     [0xD881, 1],
#         #     # victory road
#         #     [0xD7EE, 0],
#         #     [0xD7EE, 7],
#         #     [0xD813, 0],
#         #     [0xD813, 6],
#         #     [0xD869, 7],
#         # ]
#         # for ab in address_bits:
#         #     event_value = ram_map.mem_val(self.game, ab[0])
#         #     ram_map.write_mem(self.game, ab[0], ram_map.set_bit(event_value, ab[1]))

#         n_ignored_events = 0
#         for event_id in IGNORED_EVENT_IDS:
#             if self.all_events_string[event_id] == '1':
#                 n_ignored_events += 1
#         return max(
#             self.all_events_string.count('1')
#             - n_ignored_events,
#         0,
#     )
    
#     def get_all_events_reward(self):
#         if self.all_events_string != self.past_events_string:
#             first_i = -1
#             for i in range(len(self.all_events_string)):
#                 if self.all_events_string[i] == '1' and self.rewarded_events_string[i] == '0' and i not in IGNORED_EVENT_IDS:
#                     self.rewarded_events_string = self.rewarded_events_string[:i] + '1' + self.rewarded_events_string[i+1:]
#                     if first_i == -1:
#                         first_i = i
#             if first_i != -1:
#                 # update past event ids
#                 self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
#                 self.last_10_event_ids[0] = [first_i, self.time]
#         return self.rewarded_events_string.count('1') - self.base_event_flags
#             # # elite 4 stage
#             # elite_four_event_addr_bits = [
#             #     [0xD863, 0],  # EVENT START
#             #     [0xD863, 1],  # LORELEIS
#             #     [0xD863, 6],  # LORELEIS AUTO WALK
#             #     [0xD864, 1],  # BRUNOS
#             #     [0xD864, 6],  # BRUNOS AUTO WALK
#             #     [0xD865, 1],  # AGATHAS
#             #     [0xD865, 6],  # AGATHAS AUTO WALK
#             #     [0xD866, 1],  # LANCES
#             #     [0xD866, 6],  # LANCES AUTO WALK
#             # ]
#             # ignored_elite_four_events = 0
#             # for ab in elite_four_event_addr_bits:
#             #     if self.get_event_rewarded_by_address(ab[0], ab[1]):
#             #         ignored_elite_four_events += 1
#             # return self.rewarded_events_string.count('1') - self.base_event_flags - ignored_elite_four_events
    
#     def calculate_event_rewards(self, events_dict, base_reward, reward_increment, reward_multiplier):
#         """
#         Calculate total rewards for events in a dictionary.

#         :param events_dict: Dictionary containing event completion status with associated points.
#         :param base_reward: The starting reward for the first event.
#         :param reward_increment: How much to increase the reward for each subsequent event.
#         :param reward_multiplier: Multiplier to adjust rewards' significance.
#         :return: Total reward calculated for all events.
#         """
#         total_reward = 0
#         current_reward = base_reward
#         assert isinstance(events_dict, dict), f"Expected dict, got {type(events_dict)}\nvariable={events_dict}"

#         for event, points in events_dict.items():
#             if points > 0:  # Assuming positive points indicate completion or achievement
#                 total_reward += current_reward * points * reward_multiplier
#                 current_reward += reward_increment
#         return total_reward
    
#     def calculate_event_rewards_detailed(self, events, base_reward, reward_increment, reward_multiplier):
#         # This function calculates rewards for each event and returns them as a dictionary
#         # Example return format:
#         # {'event1': 10, 'event2': 11, 'event3': 12, ...}
#         detailed_rewards = {}
#         for event_name, event_value in events.items():
#             if event_value > 0:
#                 detailed_rewards[event_name] = base_reward + (event_value * reward_increment * reward_multiplier)
#             else:
#                 detailed_rewards[event_name] = (event_value * reward_increment * reward_multiplier)
#         return detailed_rewards
    
#     def update_map_id_to_furthest_visited(self):
#         # Define the ordered list of map IDs from earliest to latest
#         map_ids_ordered = [1, 2, 15, 3, 5, 21, 4, 6, 10, 7, 8, 9]        
#         # Obtain the current map ID (map_n) of the player
#         _, _, map_n = ram_map.position(self.game)        
#         # Check if the current map ID is in the list of specified map IDs
#         if map_n in map_ids_ordered:
#             # Find the index of the current map ID in the list
#             current_index = map_ids_ordered.index(map_n)            
#             # Select the furthest (latest) visited map ID from the list
#             # This is done by slicing the list up to the current index + 1
#             # and selecting the last element, ensuring we prioritize later IDs
#             furthest_visited_map_id = map_ids_ordered[:current_index + 1][-1]            
#             # Update the map ID to the furthest visited
#             ram_map.write_mem(self.game, 0xd719, furthest_visited_map_id)
#             # print(f"env_id {self.env_id}: Updated map ID to the furthest visited: {furthest_visited_map_id}")
#     def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0): # 40960 # 20480 # 2560
#         """Resets the game. Seeding is NOT supported"""
#         # BET ADDED
#         self.init_caches()
#         assert len(self.all_events_string) == 2552, f'len(self.all_events_string): {len(self.all_events_string)}'
#         self.rewarded_events_string = '0' * 2552
#         self.base_event_flags = self.get_base_event_flags()
#         # self.agent_stats = []
#         self.base_explore = 0
#         self.max_opponent_level = 0
#         self.max_event_rew = 0
#         self.max_level_rew = 0
#         self.party_level_base = 0
#         self.party_level_post = 0
#         self.last_health = 1
#         self.last_num_poke = 1
#         self.last_num_mon_in_box = 0
#         self.total_healing_rew = 0
#         self.died_count = 0
#         self.prev_knn_rew = 0
#         self.visited_pokecenter_list = []
#         self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
#         self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
#         self.past_events_string = ''
#         self.last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
#         self.early_done = False
#         self.step_count = 0
#         self.past_rewards = np.zeros(10240, dtype=np.float32)
#         self.base_event_flags = self.get_base_event_flags()
#         self.seen_map_dict = {}
#         self.update_last_10_map_ids()
#         self.update_last_10_coords()
#         self.update_seen_map_dict()
#         self._cut_badge = False
#         self._have_hm01 = False
#         self._can_use_cut = False
#         self._surf_badge = False
#         self._have_hm03 = False
#         self._can_use_surf = False
#         self._have_pokeflute = False
#         self._have_silph_scope = False
#         self.used_cut_coords_dict = {}
#         self._last_item_count = 0
#         self._is_box_mon_higher_level = False
#         self.secret_switch_states = {}
#         self.hideout_elevator_maps = []
#         self.use_mart_count = 0
#         self.use_pc_swap_count = 0
#         self.progress_reward = self.get_game_state_reward()
#         self.total_reward = sum([val for _, val in self.progress_reward.items()])
#         self.reset_count += 1
#         self.steps_after_death = 0
#         self.current_pp_list = []
#         self.reset_count += 1
#         self.time = 0
#         self.max_episode_steps = max_episode_steps
#         self.reward_scale = reward_scale
#         self.last_reward = None

#         self.prev_map_n = None
#         self.init_hidden_obj_mem()
#         self.max_events = 0
#         self.max_level_sum = 0
#         self.max_opponent_level = 0
#         self.seen_coords = set()
#         self.seen_maps = set()
#         self.death_count_per_episode = 0
#         self.total_healing = 0
#         self.last_hp = 1.0
#         self.last_party_size = 1
#         self.hm_count = 0
#         self.cut = 0
#         self.used_cut = 0 # don't reset, for tracking
#         self.cut_coords = {}
#         self.cut_tiles = {}
#         self.cut_state = deque(maxlen=3)
#         self.seen_start_menu = 0
#         self.seen_pokemon_menu = 0
#         self.seen_stats_menu = 0
#         self.seen_bag_menu = 0
#         self.seen_cancel_bag_menu = 0
#         self.seen_pokemon = np.zeros(152, dtype=np.uint8)
#         self.caught_pokemon = np.zeros(152, dtype=np.uint8)
#         self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        
#         self.seen_coords_no_reward = set()
#         self._all_events_string = ''
#         self.agent_stats = []
#         self.base_explore = 0
#         self.max_event_rew = 0
#         self.max_level_rew = 0
#         self.party_level_base = 0
#         self.party_level_post = 0
#         self.last_num_mon_in_box = 0
#         self.death_count = 0
#         self.visited_pokecenter_list = []
#         self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
#         self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
#         self.past_events_string = ''
#         self.last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
#         self.step_count = 0
#         self.past_rewards = np.zeros(10240, dtype=np.float32)
#         self.rewarded_events_string = '0' * 2552
#         self.seen_map_dict = {}
#         self._last_item_count = 0
#         self._is_box_mon_higher_level = False
#         self.secret_switch_states = {}
#         self.hideout_elevator_maps = []
#         self.use_mart_count = 0
#         self.use_pc_swap_count = 0
#         self.total_reward = 0
#         self.rewarded_coords = set()
#         self.museum_punishment = deque(maxlen=10)
#         self.rewarded_distances = {} 
        
#         # BET ADDED A BUNCH
#         self._cut_badge = False
#         self._have_hm01 = False
#         self._can_use_cut = False
#         self._surf_badge = False
#         self._have_hm03 = False
#         self._can_use_surf = False
#         self._have_pokeflute = False
#         self._have_silph_scope = False
#         self.update_last_10_map_ids()
#         self.update_last_10_coords()
#         self.update_seen_map_dict()

#         self.used_cut_coords_dict = {}

#         roll = random.uniform(0, 1)
                        
#         if roll <= 0.01:
#             load_pyboy_state(self.game, self.load_first_state()) # load the first save state every 1% of the time
#         else:
#             # load_pyboy_state(self.game, self.load_last_state()) # load the last save state
#             with self.lock:
#                 self.load_shared_state()
                
#         load_pyboy_state(self.game, open_state_file(EXPERIMENTAL_PATH))
        
#         if self.save_video:
#             base_dir = self.s_path
#             base_dir.mkdir(parents=True, exist_ok=True)
#             full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
#             self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=30) # (144, 160)
#             self.full_frame_writer.__enter__()
      
#         self.full_name_log = Path(f'pokemon_party_log').with_suffix('.txt')
#         self.full_name_checkpoint_log = Path(f'checkpoint_log_{self.env_id}').with_suffix('.txt')
#         self.write_to_log()
#             # Aggregate the data in each env log file. Default location of file: pufferlib/log_file_aggregator.py
#         try:
#             subprocess.run(['python', self.aggregate_file_path], check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Error running log_file_aggregator.py: {e}")

#         if self.use_screen_memory:
#             self.screen_memory = defaultdict(
#                 lambda: np.zeros((255, 255, 1), dtype=np.uint8)
#             )

#         # BET CHANGED BET ADDED
#         self.time = 0
        
#         self.max_episode_steps = max_episode_steps
#         self.reward_scale = reward_scale
#         self.prev_map_n = None
#         self.init_hidden_obj_mem()
#         self.max_events = 0
#         self.max_level_sum = 0
#         self.max_opponent_level = 0
#         # self.update_shared_len()
#         self.seen_coords = set()
#         # self.seen_maps = set()
#         self.map_n_reward = 0  # BET experimental 2/7/24
#         self.death_count = 0
#         self.total_healing = 0
#         self.last_hp = 1.0
#         self.last_party_size = 1
#         self.last_reward = None
#         self.b_seen_coords = {}

#         self.reset_count += 1
#         self.initial_states = self.always_starting_state
#         self.init_map_mem()
#         self.explore_hidden_objs_reward = 0
#         self.exploration_reward = 0
        
#         # BET ADDED A BUNCH
#         self._cut_badge = False
#         self._have_hm01 = False
#         self._can_use_cut = False
#         self._surf_badge = False
#         self._have_hm03 = False
#         self._can_use_surf = False
#         self._have_pokeflute = False
#         self._have_silph_scope = False
#         self.update_last_10_map_ids()
#         self.update_last_10_coords()
#         self.update_seen_map_dict()

#         self.used_cut_coords_dict = {}
        
#         # BET TESTING HP
#         self.write_hp_for_first_pokemon(1,20)
#         self.alternate_trigger = False
#         self.just_died = False
#         self.god_mode = False
#         self.else_clause_triggered = False
#         self.action_sequence = deque()
    
#         return self.render(), {}
    
#     def update_map_id_to_furthest_visited(self):
#         # Define the ordered list of map IDs from earliest to latest
#         map_ids_ordered = [1, 2, 15, 3, 5, 21, 4, 6, 10, 7, 8, 9]        
#         # Obtain the current map ID (map_n) of the player
#         _, _, map_n = ram_map.position(self.game)        
#         # Check if the current map ID is in the list of specified map IDs
#         if map_n in map_ids_ordered:
#             # Find the index of the current map ID in the list
#             current_index = map_ids_ordered.index(map_n)            
#             # Select the furthest (latest) visited map ID from the list
#             # This is done by slicing the list up to the current index + 1
#             # and selecting the last element, ensuring we prioritize later IDs
#             furthest_visited_map_id = map_ids_ordered[:current_index + 1][-1]            
#             # Update the map ID to the furthest visited
#             ram_map.write_mem(self.game, 0xd719, furthest_visited_map_id)
#             # print(f"env_id {self.env_id}: Updated map ID to the furthest visited: {furthest_visited_map_id}")

#     def print_ram_value(self, ram_value):
#         ram_map.mem_val(self.game, ram_value)
#         print(ram_value)    

#     def write_hp_for_first_pokemon(self, new_hp, new_max_hp):
#         """Write new HP value for the first party Pokémon."""
#         # HP address for the first party Pokémon
#         hp_addr = ram_map.HP_ADDR[0]
#         max_hp_addr = ram_map.MAX_HP_ADDR[0]        
#         # Break down the new_hp value into two bytes
#         hp_high = new_hp // 256  # Get the high byte
#         hp_low = new_hp % 256    # Get the low byte
#         max_hp_high = new_max_hp // 256  # Get the high byte
#         max_hp_low = new_max_hp % 256    # Get the low byte        
#         # Write the high byte and low byte to the corresponding memory addresses
#         ram_map.write_mem(self.game, hp_addr, hp_high)
#         ram_map.write_mem(self.game, hp_addr + 1, hp_low)
#         ram_map.write_mem(self.game, max_hp_addr, max_hp_high)
#         ram_map.write_mem(self.game, max_hp_addr + 1, max_hp_low)
#         # print(f"Set Max HP for the first party Pokémon to {new_max_hp}")
#         # print(f"Set HP for the first party Pokémon to {new_hp}")
    
#     def update_party_hp_to_max(self):
#         """
#         Update the HP of all party Pokémon to match their Max HP.
#         """
#         for i in range(len(ram_map.CHP)):
#             # Read Max HP
#             max_hp = ram_map.read_uint16(self.game, ram_map.MAX_HP_ADDR[i])            
#             # Calculate high and low bytes for Max HP to set as current HP
#             hp_high = max_hp // 256
#             hp_low = max_hp % 256
#             # Update current HP to match Max HP
#             ram_map.write_mem(self.game, ram_map.CHP[i], hp_high)
#             ram_map.write_mem(self.game, ram_map.CHP[i] + 1, hp_low)
#             # print(f"Updated Pokémon {i+1}: HP set to Max HP of {max_hp}.")
                
#     def restore_party_move_pp(self):
#         """
#         Restores the PP of all moves for the party Pokémon based on moves_dict data.
#         """
#         for i in range(len(ram_map.MOVE1)):  # Assuming same length for MOVE1 to MOVE4
#             moves_ids = [ram_map.mem_val(self.game, move_addr) for move_addr in [ram_map.MOVE1[i], ram_map.MOVE2[i], ram_map.MOVE3[i], ram_map.MOVE4[i]]]
            
#             for j, move_id in enumerate(moves_ids):
#                 if move_id in ram_map.moves_dict:
#                     # Fetch the move's max PP
#                     max_pp = ram_map.moves_dict[move_id]['PP']
                    
#                     # Determine the corresponding PP address based on the move slot
#                     pp_addr = [ram_map.MOVE1PP[i], ram_map.MOVE2PP[i], ram_map.MOVE3PP[i], ram_map.MOVE4PP[i]][j]
                    
#                     # Restore the move's PP
#                     ram_map.write_mem(self.game, pp_addr, max_pp)
#                     # print(f"Restored PP for {ram_map.moves_dict[move_id]['Move']} to {max_pp}.")
#                 else:
#                     pass
#                     # print(f"Move ID {move_id} not found in moves_dict.")


#     def in_coord_range(self, coord, ranges):
#         """Utility function to check if a coordinate is within a given range."""
#         r, c = coord
#         if isinstance(ranges[0], tuple):  # Check if range is a tuple of ranges
#             return any(r >= range_[0] and r <= range_[1] for range_ in ranges[0]) and \
#                 any(c >= range_[0] and c <= range_[1] for range_ in ranges[1])
#         return r == ranges[0] and c == ranges[1]

#     # def set_d732_bit_6_to_true(self):
#     #     """
#     #     Set bit 6 of the value at memory address 0xD732 to true (1).
#     #     """
#     #     addr = 0xD732  # Memory address
#     #     bit_to_set = 6  # Bit position to set to 1 (remember, counting starts at 0)
#     #     # Read the current value at address 0xD732
#     #     current_value = ram_map.mem_val(self.game, addr)
#     #     # Set bit 6 to 1 using bitwise OR operation
#     #     new_value = ram_map.set_bit(current_value, bit_to_set)
#     #     # Write the updated value back to the same address
#     #     ram_map.write_mem(self.game, addr, new_value)

#     #     print(f"Bit 6 at address {hex(addr)} set to 1.")
#     #     wLastBlackoutMap_d719 = ram_map.mem_val(self.game, 0xd719)
#     #     wLastBlackoutMap_d732_bit_6 = ram_map.read_bit(self.game, 0xd719, 6)
#     #     print(f'\nd719={wLastBlackoutMap_d719}\nd732_bit_6={wLastBlackoutMap_d732_bit_6}')
    
#     def step(self, action, fast_video=True):
                
#         # Exploration reward
#         r, c, map_n = ram_map.position(self.game)
#         self.seen_coords.add((r, c, map_n))        

#         # # Move past the spinny tiles. Can solve spinny tile problem later, eh?
#         # if map_n == 201 and ((r, c) == (13, 14) or (r, c) == (13, 11)):
        
#         #     r_pos, c_pos = ram_map.relocate(self.game, 22, 15)
#         #     print(f'new coords: ({r_pos},{c_pos})')

#         # # Check for specific conditions and execute sequence
#         # if map_n == 201 and ((r, c) == (13, 14) or (r, c) == (13, 11)):
#         #     # Define the sequence of actions to execute, mapped to the indexes of ACTIONS
#         #     sequence = ['L', 'L', 'D', 'D', 'D', 'L', 'L', 'L', 'D', 'D', 'R', 'R']
#         #     print(f'sequence={sequence}')
#         #     action_map = {'L': 1, 'D': 0, 'R': 2}  # Map the letters to the indexes of ACTIONS
#         #     print(f'action_map={action_map}')
#         #     # Clear any existing sequence and populate the action_sequence with the new sequence
#         #     self.action_sequence.clear()
#         #     self.action_sequence.extend(action_map[s] for s in sequence)

#         # # Check if there's an action sequence to execute
#         # if self.action_sequence:
#         #     # Pop the next action from the sequence
#         #     action_to_execute = self.action_sequence.popleft()
#         # else:
#         #     # Use the provided action if no sequence to execute
#         #     action_to_execute = action

#         # # Print the action to be executed
#         # print(f"Executing action: {ACTIONS[action_to_execute].__name__}")

#         # # Execute the action on the emulator
#         # run_action_on_emulator(self.game, self.screen, ACTIONS[action_to_execute],
#         #                        self.headless, fast_video=fast_video)
        
        
#         run_action_on_emulator(self.game, self.screen, ACTIONS[action],
#             self.headless, fast_video=fast_video)
        
        
#         self.time += 1 # 1

#         if self.save_video:
#             self.add_video_frame()

#         # Call nimixx api
#         self.api.process_game_states() 
#         current_bag_items = self.api.items.get_bag_item_ids()
#         self.update_cut_badge()
#         self.update_surf_badge()
#         self.update_last_10_map_ids()
#         self.update_last_10_coords()
#         self.update_seen_map_dict()
        
#         # thatguy code
#         self.update_seen_coords()
        
#         # BET TESTING SECTION
        
#         # ram_map.HP_ADDR
#         # MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
#         # low_hp = map(self.print_ram_value, ram_map.HP_ADDR)
#         # high_hp = map(self.print_ram_value, MAX_HP_ADDR)
#         # for l in low_hp:
#         #     print(f'low_hp={l}')
#         # for h in high_hp:
#         #     print(f'high_hp={h}')
        
        
#         # BET TESTING SECTION
#         self.update_map_id_to_furthest_visited()
#         wLastBlackoutMap_d719 = ram_map.mem_val(self.game, 0xd719)
#         # wLastBlackoutMap_d732_bit_6 = ram_map.read_bit(self.game, 0xd719, 6)
#         # print(f'\nd719={wLastBlackoutMap_d719}\n')
        
#         # if self.time == 10:
#         #     self.write_hp_for_first_pokemon(3, 5)
        
         
#         # New map_n logic
#         r, c, map_n = ram_map.position(self.game)
#             # Convert local position to global position
#         try:
#             glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#         except IndexError:
#             print(f'IndexError: index {glob_r} or {glob_c} is out of bounds for axis 0 with size 444.')
#             glob_r = 0
#             glob_c = 0
        
#         # BET ADDED CUT TREE OBSERVATION
#         glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#         # tree_distance_reward = self.compute_tree_reward((glob_r, glob_c), TREE_POSITIONS_GRID_GLOBAL, map_n)
#         tree_distance_reward = self.detect_and_reward_trees((glob_r, glob_c), map_n, vision_range=5)
#         # features = self.trees_features((glob_r, glob_c), TREE_POSITIONS_GRID_GLOBAL)
#         # print(f'features={features}')
#         # self.compute_tree_reward(features)
#         # print(f'\n\ntree_reward={tree_distance_reward}\n\n')
        
        
#         self.api.process_game_states()
#         # print(f'self.api.process_game_states()={self.api.process_game_states()}\n')
#         game_state = self.api.game_state.name
#         # print(f'self.api.game_state.name(): {game_state}\n')
#         # print(f'self.api.get_game_state(): {self.api.get_game_state()}\n')
        

        
#         self.api.items.get_bag_item_ids()
#         # print(f'self.api.items.get_bag_item_ids()={self.api.items.get_bag_item_ids()}\n')
#         # print(f'self.api.game_state={self.api.game_state.name}\n')
        
#         bag_item_ids = self.api.items.get_bag_item_ids()
#         bag_item_counts = self.api.items.get_bag_item_quantities()
#         pc_item_counts = self.api.items.get_pc_item_quantities()
#         # print(f'bag_iteM: {bag_item_ids}, {bag_item_counts}, {pc_item_counts}')
        
#         # Call nimixx api
#         self.api.process_game_states()
#         current_bag_items = self.api.items.get_bag_item_ids()
#         # print(f'\ngame states: {self.api.process_game_states()},\n bag_items: {current_bag_items},\n {self.update_pokedex()}\n')    

#         if 'Lemonade' in current_bag_items:
#             self.has_lemonade_in_bag = True
#             self.has_lemonade_in_bag_reward = 20
#         if 'Silph Scope' in current_bag_items:
#             self.has_silph_scope_in_bag = True
#             self.has_silph_scope_in_bag_reward = 20
#         if 'Lift Key' in current_bag_items:
#             self.has_lift_key_in_bag = True
#             self.has_lift_key_in_bag_reward = 20
#         if 'Poke Doll' in current_bag_items:
#             self.has_pokedoll_in_bag = True
#             self.has_pokedoll_in_bag_reward = 20
#         if 'Bicycle' in current_bag_items:
#             self.has_bicycle_in_bag = True
#             self.has_bicycle_in_bag_reward = 20
        
#         self.update_heat_map(r, c, map_n)
#         if map_n != self.prev_map_n:
#             self.used_cut_on_map_n = 0
#             self.prev_map_n = map_n
#             if map_n not in self.seen_maps:
#                 self.seen_maps.add(map_n)
#             # self.save_state()          
            
            
#         self.update_pokedex()
#         self.update_moves_obtained()
        
#         # if self.time % (self.max_episode_steps / 2) == 0:
#         #     self.load_interrupt()
        
#         self.seen_coords.add((r, c, map_n))
#         coord_string = f"x:{c} y:{r} m:{map_n}"
#         self.b_seen_coords[coord_string] = self.time
        
#         if map_n in self.poketower: # = [142, 143, 144, 145, 146, 147, 148]
#             print(f'map_n {map_n} in self.poketower!')
#         if map_n in self.pokehideout: # = [199, 200, 201, 202, 203]
#             print(f'map_n {map_n} in self.pokehideout!')
#         if map_n in self.saffron_city: # = [10, 70, 76, 178, 180, 182]
#             print(f'map_n {map_n} in self.saffron_city!')
#         if map_n in self.fighting_dojo: # = [177]
#             print(f'map_n {map_n} in self.fighting_dojo!')
#         if map_n in self.vermilion_gym: # = [92]
#             print(f'map_n {map_n} in self.vermilion_gym!')
        
#         # Define the specific coordinate ranges to encourage exploration around trees
#         self.celadon_tree_attraction_coords = {
#             'inside_barrier': [((28, 34), (2, 17)), ((33, 34), (18, 36))],
#             'tree_coord': [(32, 35)],
#             'outside_box': [((30, 31), (33, 37))]
#         }

#         # For either (r,c) combinations below, on map_n == 201, lock out pyboy input and
#         # override-input the next actions as: LLDDDLLLDDRR
#         # where L is left, D is down, R is right
        
#         # map_n == 201
#         # r == 13
#         # c == 14
        
#         # r == 13
#         # c == 10
        
#         # # Specific tiles to reward only for map 201 rocket hideout level 3
#         # (22,13), (22,14), (21,14), (21,15), (20,15), (19,15), # big reward
#         # (9, (13 >= 20)), ((9 >= 11), 13), (11, 12), ((11 >= 13), 10), # bigger reward
#         # (13, (12 >= 14)),  # bigger reward
#         # ((14 >= 16), 12), # bigger reward
#         # (16, (9 >= 11)), # bigger reward
#         # ((17 >= 18), 9), # bigger reward
#         # (18, (10 >= 11)), # biggest reward!
#         # ((23 >= 25), 13), # big reward
#         # ((19 >= 26), 19), # big reward
#         # ((25 >= 26), (13 >= 19)) # big reward
        

#         # Define groups of maps as lists for easier checking
#         special_maps = self.pokehideout + self.saffron_city + [self.fighting_dojo] + [self.vermilion_gym]
#         # Check if the game state indicates a special condition
#         if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 1:
#             # Determine the base exploration reward based on the map
#             if map_n in special_maps:
#                 base_reward = 0.2
#             else:
#                 base_reward = 0.02 if self.used_cut < 1 else 0.15
#             self.exploration_reward = base_reward * len(self.seen_coords)
#             # Specific rewards for map 201 (Rocket Hideout Level 3)
#             if map_n == 201:
#                 # Process each seen coordinate
#                 for coord in self.seen_coords:
#                     r, c = coord[:2]  # Assuming seen_coords stores tuples like (row, column, map_n)
#                     # Direct coordinate matches with specific rewards
#                     direct_coords = [(22, 13), (22, 14), (21, 14), (21, 15), (20, 15), (19, 15), (11, 12), (18, 10)]
#                     big_reward_coords = [(13, 12), (13, 13), (13, 14), (14, 12), (15, 12), (16, 9), (16, 10), (17, 9), (23, 13), (24, 13), (25, 13)]
#                     biggest_reward_coord = (18, 11)                    
#                     # Checking and applying rewards
#                     if (r, c) in direct_coords:
#                         self.exploration_reward += 0.3
#                     if (r, c) in big_reward_coords:
#                         self.exploration_reward += 0.5
#                     if (r, c) == biggest_reward_coord:
#                         self.exploration_reward += 1.0                    
#                     # Ranges with specific rewards
#                     range_checks = [
#                         (lambda r, c: r == 9 and 13 <= c <= 20, 0.5),
#                         (lambda r, c: 9 <= r <= 11 and c == 13, 0.5),
#                         (lambda r, c: 11 <= r <= 13 and c == 10, 0.5),
#                         (lambda r, c: 14 <= r <= 16 and c == 12, 0.5),
#                         (lambda r, c: 19 <= r <= 26 and c == 19, 0.3),
#                         (lambda r, c: 25 <= r <= 26 and 13 <= c <= 19, 0.3),
#                     ]
#                     # Apply rewards for ranges
#                     for check, reward in range_checks:
#                         if check(r, c):
#                             self.exploration_reward += reward
#         elif map_n == 202:
#             # Adjust the exploration reward for map 202 specifically, if necessary
#             self.exploration_reward += 0.5
#                     # Extra reward for exploring specific coordinates around trees
#         elif map_n == 6:
#             for coord in self.seen_coords:
#                 if any(self.in_coord_range(coord, ranges) for ranges in self.celadon_tree_attraction_coords.values()):
#                     self.exploration_reward += 0.45  # Additional reward for each coordinate in the specified ranges
#                     # print(f'BONUS reward for coord ({r},{c}) on {map_n}')
#         else:
#             # Default exploration reward logic for other cases
#             self.exploration_reward = 0.02 * len(self.seen_coords) if self.used_cut < 1 else 0.15 * len(self.seen_coords)



# # Note on the implementation:
# # - Rewards are accumulated for each seen coordinate matching the conditions.
# # - The use of tuples for specific coordinates and range checks allows for straightforward extension and adjustment.
# # - This implementation assumes `self.seen_coords` contains tuples of (r, c) representing visited grid locations.

                
#         # OLD, BEFORE ROCKET HIDEOUT REWARDS
#         # # Integration into the existing exploration reward logic
#         # print(f'int(ram_map.read_bit(self.game, 0xD81B, 7)) = {int(ram_map.read_bit(self.game, 0xD81B, 7)) }')
#         # if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0:
#         #     # Initial exploration reward based on map
#         #     if map_n in self.poketower + self.pokehideout + self.saffron_city + self.fighting_dojo + self.vermilion_gym:
#         #         base_reward = 0.2
#         #         print(f'base_reward increased on {map_n}')
#         #     else:
#         #         base_reward = 0.02 if self.used_cut < 1 else 0.15  # Adjust base reward based on tree-cutting
#         #         print(f'base_reward on {map_n}')
#         #     # Calculate the base exploration reward
#         #     self.exploration_reward = base_reward * len(self.seen_coords)
#         #     # Extra reward for exploring specific coordinates around trees
#         #     if map_n == 6:
#         #         for coord in self.seen_coords:
#         #             if any(self.in_coord_range(coord, ranges) for ranges in self.celadon_tree_attraction_coords.values()):
#         #                 self.exploration_reward += 0.15  # Additional reward for each coordinate in the specified ranges
#         #                 print(f'BONUS reward for coord ({r},{c}) on {map_n}')
#         # else:
#         #     # If not in the specific case, use the default logic
#         #     self.exploration_reward = 0.02 * len(self.seen_coords) if self.used_cut < 1 else 0.15 * len(self.seen_coords)
#         #     print(f'else clause for self.exploration_reward, on {map_n}')

#         # # BET ADDED Actual exploration reward
#         # if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0:
#         #     if map_n in self.poketower:
#         #         self.exploration_reward = 0
#         #     elif map_n in self.pokehideout:
#         #         self.exploration_reward = (0.2 * len(self.seen_coords))
#         #     elif map_n in self.saffron_city:
#         #         self.exploration_reward = (0.2 * len(self.seen_coords))
#         #     elif map_n in self.fighting_dojo:
#         #         self.exploration_reward = (0.2 * len(self.seen_coords))
#         #     elif map_n in self.vermilion_gym:
#         #         self.exploration_reward = (0.2 * len(self.seen_coords))
#         #     else:
#         #         # BET: increase exploration after cutting at least 1 tree to encourage exploration vs cut perseveration
#         #         self.exploration_reward = 0.02 * len(self.seen_coords) if self.used_cut < 1 else 0.15 * len(self.seen_coords) # 0.2 doesn't work (too high??)
#         # else:
#         #     # BET: increase exploration after cutting at least 1 tree to encourage exploration vs cut perseveration
#         #     self.exploration_reward = 0.02 * len(self.seen_coords) if self.used_cut < 1 else 0.15 * len(self.seen_coords) # 0.2 doesn't work (too high??)
        
#         # # Coords in Celadon to get to tree to cut to get to gym 4; (r, c) on local map_n 6
#         # celadon_go_to_tree_coords = [(31,35), (32,35), (33,35), (34,35), (34,34), (34,33), (34,32), (34,31),\
#         #     (34,30), (34,29), (34,28), (34,32), (34,31), (34,33), (34,32), (34,31),]
        
#         # # Inside cut tree barrier
#         # ((28 >= 34), (2 >= 17))
#         # ((33 >= 34), (18 >= 36))
        
#         # # Coord of the tree itself
#         # (32, 35)
        
#         # # Box outside cut tree to attract attention
#         # ((30 >= 31), (33 >= 37))
  
#         if map_n != self.prev_map_n:
#             self.used_cut_on_map_n = 0
#             if map_n not in self.seen_maps_no_reward and map_n not in self.seen_maps:
#                 if map_n not in self.exclude_map_n:
#                     pass
#                     # self.save_state()
#                 try:
#                     i = self.saved_states_dict[f'{map_n}'] # number of states saved on this map_n
#                 except KeyError:
#                     self.saved_states_dict[f'{map_n}'] = 0
#                     i = 0
#                 self.saved_states_dict[f'{map_n}'] = i + 1 # increment number
#                 # print(f'state saved\nmap_n, saved_states_dict, {map_n, self.saved_states_dict}\nseen_maps_no_reward: {self.seen_maps_no_reward}')
#             self.seen_maps_no_reward.add(map_n) # add map_n to unrewardable maps set
            
#             self.prev_map_n = map_n
            
#             # Logic for time-to-checkpoint logging
#             if map_n not in self.seen_maps:
#                 first_elements = [tup[0] for tup in self.seen_maps_times]
#                 if map_n not in first_elements:
#                     self.seen_maps_times.add((map_n, (time.time() - self.initial_wall_time)))
#                 # self.full_name_checkpoint_log = Path(f'checkpoint_log_{self.env_id}').with_suffix('.txt')
#                 # self.write_to_log()
#                 self.seen_maps.add(map_n)
#                 self.talk_to_npc_count[map_n] = 0  # Initialize NPC talk count for this new map
#                 # self.save_state() # Default save state location. Moving elsewhere for testing...
        
#         # Level reward
#         # Tapers after 30 to prevent overleveling
#         party_size, party_levels = ram_map.party(self.game)
#         self.party_size = party_size
#         self.party_levels = party_levels
#         self.max_level_sum = max(self.max_level_sum, sum(party_levels)) if self.max_level_sum or party_levels else 0
#         if self.max_level_sum < 30:
#             level_reward = 1 * self.max_level_sum
#         else:
#             level_reward = 30 + (self.max_level_sum - 30) / 4

#         # Healing and death rewards
#         hp = ram_map.hp(self.game)
#         hp_delta = hp - self.last_hp
#         party_size_constant = party_size == self.last_party_size

#         # if hp < 0.05 and game_state != 'BATTLE_ANIMATION' and game_state != 'BATTLE_TEXT':
#         #     self.save_state()
        
#         # Only reward if not reviving at pokecenter
#         if hp_delta > 0 and party_size_constant and not self.is_dead:
#             self.total_healing += hp_delta
        
#         # if self.time % 10 == 0:
#         #     load_pyboy_state(self.game, self.load_experimental_state())
#         # if self.time % 20 == 0:
#         #     load_pyboy_state(self.game, self.load_last_state())
        
#         # print(f'top of hp fn and self.is_dead={self.is_dead}')
#         if hp > 0.01: # loading save state here causes infinite loop
#             self.is_dead = False
#             self.steps_after_death = 0
#             self.alternate_trigger = False
#             # print(f'POKEMON IS ALIVE!!!!! self.is_dead is {self.is_dead}')    
#         # Capture the step of death
#         elif hp <= 0 and self.last_hp > 0:
#             self.restore_party_move_pp()
#             if self.just_died == True:
#                 self.death_count += 1
#                 # BET ADDED TESTING
#                 # self.update_party_hp_to_max()
#                 # self.save_state()
#                 self.is_dead = True
#                 self.steps_after_death = 0
#         elif hp <= 0:
#             self.restore_party_move_pp()
#             if self.alternate_trigger == False: # loading saved state here makes bulba swappable after death
#                 self.death_count += 1
#                 self.steps_after_death = 0
#                 # self.update_party_hp_to_max()
#                 # self.save_state()
#                 self.is_dead = True
#                 # load_pyboy_state(self.game, self.load_last_state())
#             self.alternate_trigger = True
#             # print(f'ALTERNATE DEATH TRIGGER ACTIVATED!!')
#         else:
#             self.else_clause_triggered = True
#             self.death_count += 1
#             self.steps_after_death = 0

#         # if self.time == 10:
#         #     self.reset()
#         # if self.time % 50 == 0:
#         #     self.save_state()        
#         # # Increment the step counter if is_dead is True
#         # if self.is_dead:
#         #     self.steps_after_death += 1
#         #     # Only load the last state after 5 steps
#         #     if self.steps_after_death >= 2:
#         #         print(f'LOAD CONDITION MET!!!!!!!!!!!!!')
#         #         self.load_last_state()
#         #         # Optionally, reset is_dead and the counter here, depending on your desired behavior
#         #         self.is_dead = False
#         #         self.steps_after_death = 0

#         # Update last known values for next iteration
#         self.last_hp = hp
#         self.last_party_size = party_size

#         # Set rewards
#         healing_reward = self.total_healing
#         death_reward = 0

#         # Opponent level reward
#         max_opponent_level = max(ram_map.opponent(self.game))
#         self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
#         opponent_level_reward = 0

#         # Badge reward
#         self.badges = ram_map.badges(self.game)
#         if 1 > self.badges > 0:
#             self.seen_maps_times.add(('Badge 1', (time.time() - self.initial_wall_time)))
#         elif 2 > self.badges > 0:
#             self.seen_maps_times.add(('Badge 2', (time.time() - self.initial_wall_time)))

#         self.badges_reward = 5 * self.badges # 5

#         # Save Bill
#         self.bill_state = ram_map.saved_bill(self.game)
#         self.bill_reward = 10 * self.bill_state
        
#         # SS Anne appeared
#         # Vestigial function that seems to work
#         ss_anne_state = ram_map.ss_anne_appeared(self.game)
#         if ss_anne_state:
#             ss_anne_state_reward = 5
#         else:
#             ss_anne_state_reward = 0
#         self.ss_anne_state_reward = ss_anne_state_reward
        
#         # Event rewards
#         events = ram_map.events(self.game)
#         self.max_events = max(self.max_events, events)
#         event_reward = self.max_events

#         # Dojo reward
#         dojo_reward = ram_map_leanke.dojo(self.game)
#         defeated_fighting_dojo = 1 * int(ram_map.read_bit(self.game, 0xD7B1, 0))
#         got_hitmonlee = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 6))
#         got_hitmonchan = 3 * int(ram_map.read_bit(self.game, 0xD7B1, 7))
        
#         # Hideout reward
#         hideout_reward = ram_map_leanke.hideout(self.game)
        
#         # SilphCo rewards
#         silph_co_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_silph_co_events(self.game), 
#             base_reward=10, reward_increment=10, reward_multiplier=2)
        
#         # Dojo rewards
#         dojo_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_dojo_events(self.game), 
#             base_reward=10, reward_increment=2, reward_multiplier=3)
        
#         # Hideout rewards
#         hideout_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_hideout_events(self.game),
#             base_reward=10, reward_increment=10, reward_multiplier=3)

#         # Poketower rewards
#         poke_tower_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_poke_tower_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)

#         # Gym rewards
#         gym3_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_gym3_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)
#         gym4_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_gym4_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)
#         gym5_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_gym5_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)
#         gym6_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_gym6_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)
#         gym7_events_reward = self.calculate_event_rewards(
#             ram_map_leanke.monitor_gym7_events(self.game),
#             base_reward=10, reward_increment=2, reward_multiplier=1)
        
#         # Used Cut
#         if ram_map.used_cut(self.game) == 61:
#             self.used_cut += 1
#             self.used_cut_on_map_n += 1
#             # print(f'used cut! (on a tree)')
#             # print(f'ram_map.used_cut(self.game) == 61: {ram_map.used_cut(self.game)}')
#             ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write
            

#         # Cut check
#         # 0xCFC6 - wTileInFrontOfPlayer
#         # 0xCFCB - wUpdateSpritesEnabled
#         if ram_map.mem_val(self.game, 0xD057) == 0: # is_in_battle if 1
#             if self.cut == 1:
#                 player_direction = self.game.get_memory_value(0xC109)
#                 x, y, map_id = self.get_game_coords()  # x, y, map_id
#                 if player_direction == 0:  # down
#                     coords = (x, y + 1, map_id)
#                 if player_direction == 4:
#                     coords = (x, y - 1, map_id)
#                 if player_direction == 8:
#                     coords = (x - 1, y, map_id)
#                 if player_direction == 0xC:
#                     coords = (x + 1, y, map_id)
#                 self.cut_state.append(
#                     (
#                         self.game.get_memory_value(0xCFC6),
#                         self.game.get_memory_value(0xCFCB),
#                         self.game.get_memory_value(0xCD6A),
#                         self.game.get_memory_value(0xD367),
#                         self.game.get_memory_value(0xD125),
#                         self.game.get_memory_value(0xCD3D),
#                     )
#                 )
#                 if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
#                     self.cut_coords[coords] = 10 # 10
#                     self.cut_tiles[self.cut_state[-1][0]] = 1
#                 elif self.cut_state == CUT_GRASS_SEQ:
#                     self.cut_coords[coords] = 0.001
#                     self.cut_tiles[self.cut_state[-1][0]] = 1
#                 elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
#                     self.cut_coords[coords] = 0.001
#                     self.cut_tiles[self.cut_state[-1][0]] = 1

#                 if int(ram_map.read_bit(self.game, 0xD803, 0)):
#                     if self.check_if_in_start_menu():
#                         self.seen_start_menu = 1
#                     if self.check_if_in_pokemon_menu():
#                         self.seen_pokemon_menu = 1
#                     if self.check_if_in_stats_menu():
#                         self.seen_stats_menu = 1
#                     if self.check_if_in_bag_menu():
#                         self.seen_bag_menu = 1
#                     if self.check_if_cancel_bag_menu(action):
#                         self.seen_cancel_bag_menu = 1    
        
#         # SS Anne rewards
#         # Experimental
#         got_hm01_reward = 5 if ram_map.got_hm01(self.game) else 0
#         rubbed_captains_back_reward = 5 if ram_map.rubbed_captains_back(self.game) else 0
#         ss_anne_left_reward = 5 if ram_map.ss_anne_left(self.game) else 0
#         walked_past_guard_after_ss_anne_left_reward = 5 if ram_map.walked_past_guard_after_ss_anne_left(self.game) else 0
#         started_walking_out_of_dock_reward = 5 if ram_map.started_walking_out_of_dock(self.game) else 0
#         walked_out_of_dock_reward = 5 if ram_map.walked_out_of_dock(self.game) else 0

#         # Badge reward
#         badges = ram_map.badges(self.game)
#         badges_reward = 10 * badges # 5 BET

#         # Save Bill
#         bill_state = ram_map.saved_bill(self.game)
#         bill_reward = 5 * bill_state
        
        
#         # HM reward
#         hm_count = ram_map.get_hm_count(self.game)
        
#         # Save state on obtaining hm
#         if hm_count >= 1 and self.hm_count == 0:
#             # self.save_state()
#             self.hm_count = 1
#         hm_reward = hm_count * 10
#         cut_rew = self.cut * 8 # 10 works - 2 might be better, though 
        
#         # HM reward
#         hm_reward = self.get_hm_rewards()
#         self.hm_reward = hm_count * 5

#         # SS Anne flags
#         # Experimental
#         got_hm01 = int(bool(got_hm01_reward))
#         self.rubbed_captains_back = int(bool(rubbed_captains_back_reward))
#         self.ss_anne_left = int(bool(ss_anne_left_reward))
#         self.walked_past_guard_after_ss_anne_left = int(bool(walked_past_guard_after_ss_anne_left_reward))
#         self.started_walking_out_of_dock = int(bool(started_walking_out_of_dock_reward))
#         self.walked_out_of_dock = int(bool(walked_out_of_dock_reward))
        
#         # got_hm01 flag to enable cut menu conditioning
#         self.got_hm01 = got_hm01
#         self.got_hm01_reward = self.got_hm01 * 5

#         # Event reward
#         events = ram_map.events(self.game)
#         self.events = events
#         self.max_events = max(self.max_events, events)
#         event_reward = self.max_events
        
#         # Event reward thatguy
#         event_reward_2 = self.update_max_event_rew()

#         money = ram_map.money(self.game)
#         self.money = money
        
#         # # Explore NPCs
#         # # Known to not actually work correctly. Counts first sign on each map as NPC. Treats NPCs as hidden obj and vice versa.
#         # # Intentionally left this way because it works better, i.e. proper NPC/hidden obj. rewarding/ignoring signs gets
#         # # worse results.
#         #         # check if the font is loaded
#         # if ram_map.mem_val(self.game, 0xCFC4):
#         #     # check if we are talking to a hidden object:
#         #     if ram_map.mem_val(self.game, 0xCD3D) == 0x0 and ram_map.mem_val(self.game, 0xCD3E) == 0x0:
#         #         # add hidden object to seen hidden objects
#         #         self.seen_hidden_objs.add((ram_map.mem_val(self.game, 0xD35E), ram_map.mem_val(self.game, 0xCD3F)))
#         #     else:
#         #         # check if we are talking to someone
#         #         # if ram_map.if_font_is_loaded(self.game):
#         #             # get information for player
#         #         player_direction = ram_map.player_direction(self.game)
#         #         player_y = ram_map.player_y(self.game)
#         #         player_x = ram_map.player_x(self.game)
#         #         # get the npc who is closest to the player and facing them
#         #         # we go through all npcs because there are npcs like
#         #         # nurse joy who can be across a desk and still talk to you
#         #         mindex = (0, 0)
#         #         minv = 1000
#         #         for npc_bank in range(1):
#         #             for npc_id in range(1, ram_map.sprites(self.game) + 15):
#         #                 npc_dist = self.find_neighboring_npc(npc_bank, npc_id, player_direction, player_x, player_y)
                        
#         #                 self.find_neighboring_npc, 0, 0, player_direction, player_x, player_y

#         #                 if npc_dist < minv:
#         #                     mindex = (npc_bank, npc_id)
#         #                     minv = npc_dist        
                
#         #         self.find_neighboring_npc, mindex[0], mindex[1], player_direction, player_x, player_y
                
#         #         self.seen_npcs.add((ram_map.map_n(self.game), mindex[0], mindex[1]))

#         explore_npcs_reward = self.reward_scale * self.explore_npc_weight * len(self.seen_npcs) * 0.00015
#         seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon) * 0.00010
#         caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon) * 0.00010
#         moves_obtained_reward = self.reward_scale * sum(self.moves_obtained) * 0.00010
#         explore_hidden_objs_reward = self.reward_scale * self.explore_hidden_obj_weight * len(self.seen_hidden_objs) * 0.00015
#         used_cut_reward = self.used_cut * 100

#         # Misc
#         self.update_pokedex()
#         self.update_moves_obtained()

#         bill_capt_rew = ram_map.bill_capt(self.game)
        
#         # Cut check 2 - BET ADDED: used cut on tree
#         if ram_map.used_cut(self.game) == 61:
#             ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write
#             if (map_n, r, c) in self.used_cut_coords_set:
#                 pass
#             else:
#                 self.used_cut += 1

#         used_cut_on_tree_rew = 0 # should be 0 to prevent cut abuse
#         start_menu = self.seen_start_menu * 0.01
#         pokemon_menu = self.seen_pokemon_menu * 0.1
#         stats_menu = self.seen_stats_menu * 0.1
#         bag_menu = self.seen_bag_menu * 0.1
#         cut_coords = sum(self.cut_coords.values()) * 1.0
#         cut_tiles = len(self.cut_tiles) * 1.0
#         that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu)
    
#         seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
#         caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
#         moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)

#         reward = self.reward_scale * (
#             event_reward
#             + bill_capt_rew
#             + seen_pokemon_reward
#             + caught_pokemon_reward
#             + moves_obtained_reward
#             + bill_reward
#             + hm_reward
#             + level_reward
#             + death_reward
#             + badges_reward
#             + healing_reward
#             + self.exploration_reward 
#             + cut_rew
#             + that_guy / 2 # reward for cutting an actual tree (but not erika's trees)
#             + cut_coords # reward for cutting anything at all
#             + cut_tiles # reward for cutting a cut tile, e.g. a patch of grass
#             + tree_distance_reward * 0.6 # 1 is too high # 0.25 # 0.5
#             + dojo_reward * 5
#             + hideout_reward * 5 # woo! extra hideout rewards!!
#             + self.has_lemonade_in_bag_reward
#             + self.has_silph_scope_in_bag_reward
#             + self.has_lift_key_in_bag_reward
#             + self.has_pokedoll_in_bag_reward
#             + self.has_bicycle_in_bag_reward
#             + (dojo_events_reward + silph_co_events_reward + 
#                hideout_events_reward + poke_tower_events_reward + 
#                gym3_events_reward + gym4_events_reward + 
#                gym5_events_reward + gym6_events_reward + 
#                gym7_events_reward)
#             + (gym3_events_reward + gym4_events_reward +
#                gym5_events_reward + gym6_events_reward +
#                gym7_events_reward)
#         )
        
#         self.explore_hidden_objs_reward = explore_hidden_objs_reward
#         self.explore_npcs_reward = explore_npcs_reward
#         self.event_rew = event_reward
#         self.level_rew = level_reward
#         self.healing_rew = healing_reward
#         self.rew = reward
#         self.got_hm01 = got_hm01
#         self.hm_reward = hm_reward 
#         self.got_hm01_reward = got_hm01_reward
#         self.rubbed_captains_back_reward = rubbed_captains_back_reward
#         self.ss_anne_state_reward = ss_anne_state_reward
#         self.ss_anne_left_reward = ss_anne_left_reward
#         self.walked_past_guard_after_ss_anne_left_reward = walked_past_guard_after_ss_anne_left_reward
#         self.started_walking_out_of_dock_reward = started_walking_out_of_dock_reward
#         self.explore_npcs_reward = explore_npcs_reward
#         self.seen_pokemon_reward = seen_pokemon_reward
#         self.caught_pokemon_reward = caught_pokemon_reward
#         self.moves_obtained_reward = moves_obtained_reward
#         self.explore_hidden_objs_reward = explore_hidden_objs_reward
#         self.used_cut_reward = used_cut_reward
#         self.walked_out_of_dock_reward = walked_out_of_dock_reward
#         self.hm_count = hm_count
        
#         # Subtract previous reward
#         # TODO: Don't record large cumulative rewards in the first place
#         if self.last_reward is None:
#             reward = 0
#             self.last_reward = 0
#         else:
#             nxt_reward = reward
#             reward -= self.last_reward
#             self.last_reward = nxt_reward

#         info = {}
#         # TODO: Make log frequency a configuration parameter
#         if self.time % (self.max_episode_steps // 2) == 0:
#             info = self.agent_stats()
        
#         # BET ADDED: ENABLE FOR VISUALIZATION OF OBSERVATIONS    
#         # if self.save_video:
#         #     self.add_custom_frame()
            
#         done = self.time >= self.max_episode_steps
        
#         if self.save_video and done:
#             self.full_frame_writer.close()
            

#         if self.verbose:
#             r, c, map_n = ram_map.position(self.game)
#             glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#             print(
                
#                 f'\nsteps: {self.time}\n',
#                 f'r: {r}\n',
#                 f'c: {c}\n',
#                 f'map_n: {map_n}\n',
#                 f'r * 16 = {r * 16}\n',
#                 f'c * 16 = {c * 16}\n',
#                 f'glob_r: {glob_r}\n',
#                 f'glob_c: {glob_c}\n',
#                 # f'get_all_events_reward: {self.get_all_events_reward()}\n',
#                 # f'Lemonade in item bag?: {self.has_lemonade_in_bag}\n',
#                 # f'Silph Scope in item bag?: {self.has_silph_scope_in_bag}\n', 
#                 # f'Lift Key in item bag?: {self.has_lift_key_in_bag}\n',
#                 # f'Poke Doll in item bag?: {self.has_pokedoll_in_bag}\n',
#                 # f'Bicycle in item bag?: {self.has_bicycle_in_bag}\n',
#                 # f'self.is_dead: {self.is_dead}\n',
#                 # f'alternate_trigger: {self.alternate_trigger}\n',
#                 # f'death_count: {self.death_count}\n',
#                 # f'self.just_died: {self.just_died}\n',
#                 # f'else_clause_triggered: {self.else_clause_triggered}\n',
#                 # f'self.steps_after_death: {self.steps_after_death}\n',
#                 # f'self.god_mode: {self.god_mode}\n',
#                 # f'hp: {hp}\n',

#                 # f'exploration reward: {exploration_reward}',
#                 # f'level_Reward: {level_reward}',
#                 # f'healing: {healing_reward}',
#                 # f'death: {death_reward}',
#                 # f'op_level: {opponent_level_reward}',
#                 # f'badges reward: {self.badges_reward}',
#                 # f'event reward: {event_reward}',
#                 # f'money: {money}',
#                 # f'ai reward: {reward}',
#                 # f'Info: {info}',
#             )

#         return self.render(), reward, done, done, info

#     def agent_stats(self):
#         return {
#             "reward": {
#                 "delta": self.rew,
#                 # "event": self.event_rew,
#                 # "event_2": self.update_max_event_rew,
#                 # "level": self.level_rew,
#                 # "badges": self.badges_rew,
#                 # "bill_saved_reward": self.bill_reward,
#                 # "hm_count_reward": self.hm_reward,
#                 # "got_hm01_reward": self.got_hm01_reward,
#                 # "rubbed_captains_back_reward": self.rubbed_captains_back_reward,
#                 # "ss_anne_state_reward": self.ss_anne_state_reward,
#                 # "ss_anne_left_reward": self.ss_anne_left_reward,
#                 # "walked_past_guard_after_ss_anne_left_reward": self.walked_past_guard_after_ss_anne_left_reward,
#                 # "started_walking_out_of_dock_reward": self.started_walking_out_of_dock_reward,
#                 # "walked_out_of_dock_reward": self.walked_out_of_dock_reward, 
#                 # "exploration": self.exploration_reward,
#                 # "explore_npcs_reward": self.explore_npcs_reward,
#                 # "seen_pokemon_reward": self.seen_pokemon_reward,
#                 # "caught_pokemon_reward": self.caught_pokemon_reward,
#                 # "moves_obtained_reward": self.moves_obtained_reward,
#                 # "hidden_obj_count_reward": self.explore_hidden_objs_reward,
#                 # "poke_has_cut_reward": self.poke_has_cut_reward,
#                 # "poke_has_flash_reward": self.poke_has_flash_reward,
#                 # "poke_has_fly_reward": self.poke_has_fly_reward,
#                 # "poke_has_surf_reward": self.poke_has_surf_reward,
#                 # "poke_has_strength_reward": self.poke_has_strength_reward,
#                 # "used_cut_reward": self.used_cut_reward,
#                 # "menus_reward": self.menus_rewards,
#                 # "healing_reward": self.healing_rew,
#                 # "new_map_n_reward": self.map_n_reward,
#             },
#             "stats": {
#             "maps_explored": len(self.seen_maps),
#             # "party_size": self.last_party_size,
#             # "highest_pokemon_level": max(self.party_levels, default=0),
#             # "total_party_level": sum(self.party_levels),
#             # "deaths": self.death_count,
#             # "bill_saved": self.bill_state,
#             # "hm_count": self.hm_count,
#             # "got_hm01": self.got_hm01,
#             # "rubbed_captains_back": self.rubbed_captains_back,
#             # "ss_anne_left": self.ss_anne_left,
#             # "ss_anne_state": self.ss_anne_state,
#             # "walked_past_guard_after_ss_anne_left": self.walked_past_guard_after_ss_anne_left,
#             # "started_walking_out_of_dock": self.started_walking_out_of_dock,
#             # "walked_out_of_dock": self.walked_out_of_dock,
#             # "badge_1": float(self.badges >= 1),
#             # "badge_2": float(self.badges >= 2),
#             # "event": self.events,
#             # "money": self.money,
#             # "seen_npcs_count": len(self.seen_npcs),
#             # "seen_pokemon": sum(self.seen_pokemon),
#             # "caught_pokemon": sum(self.caught_pokemon),
#             # "moves_obtained": sum(self.moves_obtained),
#             # "hidden_obj_count": len(self.seen_hidden_objs),
#             # "poke_has_cut": self.poke_has_cut,
#             # "poke_has_flash": self.poke_has_flash,
#             # "poke_has_fly": self.poke_has_fly,
#             # "poke_has_surf": self.poke_has_surf,
#             # "poke_has_strength": self.poke_has_strength,
#             # "used_cut": self.used_cut,
#             # "cut_nothing": self.cut_nothing,
#             # "total_healing": self.total_healing,
#             # "checkpoints": self.seen_maps_times,
#             # "saved_states_dict": list(self.saved_states_dict.items()),
#             },
#             # "pokemon_exploration_map": self.get_explore_map(), 
#             # "200_step_pyboy_save_state": self.pass_states(),
#             # "logging": logging,
#             # "env_uuid": self.env_id,
#         }
        
#     # Only reward exploration for the below coordinates
#     # Default: path through Mt. Moon, then whole map rewardable.
#     # Reward if True
#     def rewardable_coords(self, glob_c, glob_r, map_n):
#         if map_n in self.seen_maps_no_reward:
#             return False
#         else:
#             return True # reward EVERYTHING
#         # r, c, map_n = ram_map.position(self.game)
#         # if map_n == 15 or map_n == 3:
#         #     self.past_mt_moon = True
#         # # Whole map included; excluded if in self.exclude_map_n
#         # if self.past_mt_moon == True and map_n not in self.exclude_map_n:
#         #     self.include_conditions = [(0 <= glob_c <= 436) and (0 <= glob_r <= 444)]
#         # else:
#         #     if map_n not in self.exclude_map_n:
#         #         # Path through Mt. Moon
#         #         self.include_conditions = [(80 >= glob_c >= 72) and (294 < glob_r <= 320),
#         #         (69 < glob_c < 74) and (313 >= glob_r >= 295),
#         #         (73 >= glob_c >= 72) and (220 <= glob_r <= 330),
#         #         (75 >= glob_c >= 74) and (310 >= glob_r <= 319),
#         #         (81 >= glob_c >= 73) and (294 < glob_r <= 313),
#         #         (73 <= glob_c <= 81) and (294 < glob_r <= 308),
#         #         (80 >= glob_c >= 74) and (330 >= glob_r >= 284),
#         #         (90 >= glob_c >= 89) and (336 >= glob_r >= 328),
#         #         # Viridian Pokemon Center
#         #         (282 >= glob_r >= 277) and glob_c == 98,
#         #         # Pewter Pokemon Center
#         #         (173 <= glob_r <= 178) and glob_c == 42,
#         #         # Route 4 Pokemon Center
#         #         (131 <= glob_r <= 136) and glob_c == 132,
#         #         (75 <= glob_c <= 76) and (271 < glob_r < 273),
#         #         (82 >= glob_c >= 74) and (284 <= glob_r <= 302),
#         #         (74 <= glob_c <= 76) and (284 >= glob_r >= 277),
#         #         (76 >= glob_c >= 70) and (266 <= glob_r <= 277),
#         #         (76 <= glob_c <= 78) and (274 >= glob_r >= 272),
#         #         (74 >= glob_c >= 71) and (218 <= glob_r <= 266),
#         #         (71 >= glob_c >= 67) and (218 <= glob_r <= 235),
#         #         (106 >= glob_c >= 103) and (228 <= glob_r <= 244),
#         #         (116 >= glob_c >= 106) and (228 <= glob_r <= 232),
#         #         (116 >= glob_c >= 113) and (196 <= glob_r <= 232),
#         #         (113 >= glob_c >= 89) and (208 >= glob_r >= 196),
#         #         (97 >= glob_c >= 89) and (188 <= glob_r <= 214),
#         #         (102 >= glob_c >= 97) and (189 <= glob_r <= 196),
#         #         (89 <= glob_c <= 91) and (188 >= glob_r >= 181),
#         #         (74 >= glob_c >= 67) and (164 <= glob_r <= 184),
#         #         (68 >= glob_c >= 67) and (186 >= glob_r >= 184),
#         #         (64 <= glob_c <= 71) and (151 <= glob_r <= 159),
#         #         (71 <= glob_c <= 73) and (151 <= glob_r <= 156),
#         #         (73 <= glob_c <= 74) and (151 <= glob_r <= 164),
#         #         (103 <= glob_c <= 74) and (157 <= glob_r <= 156),
#         #         (80 <= glob_c <= 111) and (155 <= glob_r <= 156),
#         #         (111 <= glob_c <= 99) and (155 <= glob_r <= 150),
#         #         (111 <= glob_c <= 154) and (150 <= glob_r <= 153),
#         #         (138 <= glob_c <= 154) and (153 <= glob_r <= 160),
#         #         (153 <= glob_c <= 154) and (153 <= glob_r <= 154),
#         #         (143 <= glob_c <= 144) and (153 <= glob_r <= 154),
#         #         (154 <= glob_c <= 158) and (134 <= glob_r <= 145),
#         #         (152 <= glob_c <= 156) and (145 <= glob_r <= 150),
#         #         (42 <= glob_c <= 43) and (173 <= glob_r <= 178),
#         #         (158 <= glob_c <= 163) and (134 <= glob_r <= 135),
#         #         (161 <= glob_c <= 163) and (114 <= glob_r <= 128),
#         #         (163 <= glob_c <= 169) and (114 <= glob_r <= 115),
#         #         (114 <= glob_c <= 169) and (167 <= glob_r <= 102),
#         #         (169 <= glob_c <= 179) and (102 <= glob_r <= 103),
#         #         (178 <= glob_c <= 179) and (102 <= glob_r <= 95),
#         #         (178 <= glob_c <= 163) and (95 <= glob_r <= 96),
#         #         (164 <= glob_c <= 163) and (110 <= glob_r <= 96),
#         #         (163 <= glob_c <= 151) and (110 <= glob_r <= 109),
#         #         (151 <= glob_c <= 154) and (101 <= glob_r <= 109),
#         #         (151 <= glob_c <= 152) and (101 <= glob_r <= 97),
#         #         (153 <= glob_c <= 154) and (97 <= glob_r <= 101),
#         #         (151 <= glob_c <= 154) and (97 <= glob_r <= 98),
#         #         (152 <= glob_c <= 155) and (69 <= glob_r <= 81),
#         #         (155 <= glob_c <= 169) and (80 <= glob_r <= 81),
#         #         (168 <= glob_c <= 184) and (39 <= glob_r <= 43),
#         #         (183 <= glob_c <= 178) and (43 <= glob_r <= 51),
#         #         (179 <= glob_c <= 183) and (48 <= glob_r <= 59),
#         #         (179 <= glob_c <= 158) and (59 <= glob_r <= 57),
#         #         (158 <= glob_c <= 161) and (57 <= glob_r <= 30),
#         #         (158 <= glob_c <= 150) and (30 <= glob_r <= 31),
#         #         (153 <= glob_c <= 150) and (34 <= glob_r <= 31),
#         #         (168 <= glob_c <= 254) and (134 <= glob_r <= 140),
#         #         (282 >= glob_r >= 277) and (436 >= glob_c >= 0), # Include Viridian Pokecenter everywhere
#         #         (173 <= glob_r <= 178) and (436 >= glob_c >= 0), # Include Pewter Pokecenter everywhere
#         #         (131 <= glob_r <= 136) and (436 >= glob_c >= 0), # Include Route 4 Pokecenter everywhere
#         #         (137 <= glob_c <= 197) and (82 <= glob_r <= 142), # Mt Moon Route 3
#         #         (137 <= glob_c <= 187) and (53 <= glob_r <= 103), # Mt Moon B1F
#         #         (137 <= glob_c <= 197) and (16 <= glob_r <= 66), # Mt Moon B2F
#         #         (137 <= glob_c <= 436) and (82 <= glob_r <= 444),  # Most of the rest of map after Mt Moon
#         #     ]
#         #         return any(self.include_conditions)
#         #     else:
#         #         return False



# Extra fns
        
        # # Exploration reward
        # # r, c, map_n = ram_map.position(self.game)
        # if self.cut == 1:
        #     # r=31 c=35 map_n=6 (outside celadon cut tree)
        #     if r == 0x1F and c == 0x23 and map_n == 0x06:
        #         print(f'\n1) r=r == 0x1F and c == 0x23 and map_n == 0x06: {r}, {c}, {map_n}\n')
        #         if (r, c, map_n) not in self.l_seen_coords:
        #             self.expl += 5
        #     # r=17 c=15 map_n=5 (outside vermilion cut tree)
        #     if r == 0x11 and c == 0x0F and map_n == 0x05:
        #         print(f'\n2) r == 0x11 and c == 0x0F and map_n == 0x05: {r}, {c}, {map_n}\n')
        #         if (r, c, map_n) not in self.l_seen_coords:
        #             self.expl += 5
        #     # r=18 c=14 map_n=5 (outside vermilion cut tree)
        #     if r == 0x12 and c == 0x0E and map_n == 0x05:
        #         print(f'\n3)r == 0x12 and c == 0x0E and map_n == 0x05: {r}, {c}, {map_n}\n')
        #         if (r, c, map_n) not in self.l_seen_coords:
        #             self.expl += 5
        # self.l_seen_coords.add((r, c, map_n))
        # l_exploration_reward = (0.02 * len(self.l_seen_coords)) + self.expl
        # print(f'\nl_exploration_reward={l_exploration_reward}\n')
        # print(f'\nself.expl={self.expl}\n')