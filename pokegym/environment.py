import tracemalloc

from pathlib import Path
from pdb import set_trace as T
# import types
import uuid
from gymnasium import Env, spaces
import numpy as np
from skimage.transform import resize
from collections import defaultdict
import io, os
# import random
# import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mediapy as media
import subprocess
from pokegym import data
from pokegym.bin.ram_reader.red_ram_api import *
import random
from pokegym.constants import *

from torch.profiler import profile, ProfilerActivity, schedule
import time

from pokegym.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
from pokegym import ram_map, game_map
import multiprocessing
from pokegym.bin.ram_reader.red_memory_battle import *
from pokegym.bin.ram_reader.red_memory_env import *
from pokegym.bin.ram_reader.red_memory_items import *
from pokegym.bin.ram_reader.red_memory_map import *
from pokegym.bin.ram_reader.red_memory_menus import *
from pokegym.bin.ram_reader.red_memory_player import *
from pokegym.bin.ram_reader.red_ram_debug import *
from enum import IntEnum

# def get_time(fn, *args, **kwargs):
#     start_time = time.time()
#     result = fn(*args, **kwargs)
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f'Time taken by {fn.__name__}: {total_time} seconds')
#     return result
# total_times = {}

# def get_time(fn, *args, **kwargs):
#         start_time = time.time()

#         result = fn(*args, **kwargs)

#         end_time = time.time()
#         total_time = end_time - start_time

#         # Update the total time for this function
#         if fn.__name__ not in total_times:
#             total_times[fn.__name__] = total_time
#         else:
#             total_times[fn.__name__] += total_time

#         print(f'Time taken by {fn.__name__}: {total_time} seconds')

#         return result

# def average_times(n):
#     # Calculate and print average times
#     with open('ztimes.txt', 'w') as f:
#         for fn_name, total_time in total_times.items():
#             average_time = total_time / n
#             print(f'Average time for {fn_name}: {average_time} seconds')
#             f.write(f'Average time for {fn_name}: {average_time} seconds\n')

# Testing environment w/ no AI
# pokegym.play from pufferlib folder
def play():
    """Creates an environment and plays it"""
    env = Environment(
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=False,
        disable_input=False,
        sound=False,
        sound_emulated=False,
        verbose=False,
    )

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
        
        # for event in input_events:
        #     event_str = str(event)
        #     if event_str in window_event_to_action:
        #         action_index = window_event_to_action[event_str]
        #         if action_index == 0 and (random.randint(1, 10) > 8):
        #             observation, reward, done, _, info = env.step(
        #                 action_index, fast_video=False
        #             )
        #         else:
        #             action_index = 3
        #             observation, reward, done, _, info = env.step(
        #                 action_index, fast_video=False
        #             )
                
        for event in input_events:
            event_str = str(event)
            if event_str in window_event_to_action:
                action_index = window_event_to_action[event_str]
                observation, reward, done, _, info = env.step(
                    action_index, fast_video=False
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
        
        """Creates a PokemonRed environment"""
        # Change state_path if you want to load off a different state file to start
        if state_path is None:
            # state_path = __file__.rstrip("environment.py") + "Bulbasaur.state"
            state_path = __file__.rstrip("environment.py") + "Bulbasaur_fast_text_no_battle_animations_fixed_battle.state"
        # Make the environment
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.always_starting_state = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.use_screen_memory = True
        self.screenshot_counter = 0
        self.step_states = []
        # self.counts_array = np.zeros([256,50,50], dtype=np.uint8)
        # counts_array_update(arr, map_n, r, c):
        #     self.counts_array[map_n, r, c] += 1
        
        # BET nimixx api
        self.api = Game(self.game) # import this class for api BET
        
        # Logging initializations
        with open("experiments/running_experiment.txt", "r") as file:
        # with open("experiments/test_exp.txt", "r") as file: # for testing video writing BET
            exp_name = file.read()
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        # self.env_id = Path(f'session_{str(uuid.uuid4())[:8]}')
        self.env_id = env_id
        self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        
        # Manually create running_experiment.txt at pufferlib/experiments/running_experiment.txt
        # Set logging frequency in steps and log_file_aggregator.py path here.
        # Logging makes a file pokemon_party_log.txt in each environment folder at
        # pufferlib/experiments/2w31qioa/sessions/{session_uuid8}/pokemon_party_log.txt
        self.log = True
        self.stepwise_csv_logging = False
        self.log_on_reset = True
        self.log_frequency = 500 # Frequency to log, in steps, if self.log=True and self.log_on_reset=False
        self.aggregate_frequency = 600
        self.aggregate_file_path = 'log_file_aggregator.py'
        
        self.reset_count = 0
        self.explore_hidden_obj_weight = 1
        self.initial_wall_time = time.time()

        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2)

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
            
    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = set()
    
    def save_screenshot(self, event, map_n):
        self.screenshot_counter += 1
        ss_dir = Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            # ss_dir / Path(f'ss_{x}_y_{y}_steps_{steps}_{comment}.jpeg'),
            ss_dir / Path(f'{self.screenshot_counter}_{event}_{map_n}.jpeg'),
            self.screen.screen_ndarray())  # (144, 160, 3)

    def save_state_step(self):
        state = io.BytesIO()
        state.seek(0)
        return self.game.save_state(state)
    
    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states.append(state)
    
    def load_last_state(self):
        return self.initial_states[-1]
    
    def load_first_state(self):
        return self.always_starting_state[0]

    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}

    # Helps AI explore. Gives reduced view of PyBoy emulator window, centered on player.
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

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}
        
    def video(self):
        video = self.screen.screen_ndarray()
        return video

    def close(self):
        self.game.stop(False)

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
        # self.menus = Menus(self) # should actually be self.api.menus (Game owns everything)
        self.last_menu_state = 'None'
        self.menus_rewards = 0
        self.sel_cancel = 0
        self.start_sel = 0
        self.none_start = 0
        self.pk_cancel_menu = 0
        self.pk_menu = 0
        self.cut_nothing = 0
        self.different_menu = 'None'

        self.counts_map = np.zeros((444, 436))
        self.verbose = verbose
        self.screenshot_counter = 0
        self.include_conditions = []
        self.seen_maps_difference = set() # Vestigial - kept for consistency
        # self.seen_maps = 0
        self.seen_maps_times = set()
        self.current_maps = []
        self.exclude_map_n = {37, 38, 39, 43, 52, 53, 55, 57} # No rewards for pointless building exploration
        # self.exclude_map_n_moon = {0, 1, 2, 12, 13, 14, 15, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 193, 68}
        self.is_dead = False
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
        self.past_mt_moon = False
        self.used_cut = 0
        self.done_counter = 0
        # self.map_n_reward = 0 
        self.max_increments = 100 #BET experimental
        self.got_hm01 = 0
        self.seen_maps = set()
        self.saved_states_dict = {}
        self.seen_maps_no_reward = set()
        self.seen_coords_no_reward = set()
        self.seen_map_dict = {}
        self.is_warping = False
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        
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
    
    def write_to_log(self):
        pokemon_info = ram_map.pokemon_l(self.game)
        bag_items = self.api.items.get_bag_item_ids()
        session_path = self.s_path
        base_dir = self.exp_path
        base_dir.mkdir(parents=True, exist_ok=True)
        session_path.mkdir(parents=True, exist_ok=True)
        # Writing Pokémon info to session log
        with open(session_path / self.full_name_log, 'w') as f:
            for pokemon in pokemon_info:
                f.write(f"Slot: {pokemon['slot']}\n")
                f.write(f"Name: {pokemon['name']}\n")
                f.write(f"Level: {pokemon['level']}\n")
                f.write(f"Moves: {', '.join(pokemon['moves'])}\n")
                f.write("\n")  # Add a newline between Pokémon
                # print(f'WROTE POKEMON LOG TO {session_path}/{self.full_name_log}')
            f.write("Bag Items:\n")
            for i, item in enumerate(bag_items,1):
                f.write(f"{item}\n")
        # Writing visited locations and times to log
        with open(session_path / self.full_name_checkpoint_log, 'w') as f:
            for location, time_visited in self.seen_maps_times:
                f.write(f"Location ID: {location}\n")
                f.write(f"Time Visited: {time_visited}\n")
                f.write("\n")
                # print(f'WROTE CHECKPOINT LOG TO {session_path}/{self.full_name_checkpoint_log}')
    
    def env_info_to_csv(self, env_id, reset, x, y, map_n, csv_file_path):
        df = pd.DataFrame([[env_id, reset, x, y, map_n]])
        df.to_csv(csv_file_path, mode='a', header=not csv_file_path.exists(), index=False)
        
    def write_env_info_to_csv(self):
        x, y, map_n = ram_map.position(self.game)
        base_dir = self.exp_path
        reset = self.reset_count
        env_id = self.env_id
        csv_file_path = base_dir / "steps_map.csv"

        self.env_info_to_csv(env_id, reset, x, y, map_n, csv_file_path)
    
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

    # print(f'counts_map={self.counts_map}, shape={np.shape(self.counts_map)}, size={np.size(self.counts_map)}, sum={np.sum(self.counts_map)}')

        # Update last_map for the next iteration
        self.last_map = current_map

    # def pass_states(self):
    #     num_envs = 64
    #     step_length = 1310720 / num_envs
    #     if self.reset_count == step_length:
    #        state = self.save_state_step()
    #        return state
    
    def find_neighboring_npc(self, npc_bank, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = ram_map.npc_y(self.game, npc_id, npc_bank)
        npc_x = ram_map.npc_x(self.game, npc_id, npc_bank)
        if (
            (player_direction == 0 and npc_x == player_x and npc_y > player_y) or
            (player_direction == 4 and npc_x == player_x and npc_y < player_y) or
            (player_direction == 8 and npc_y == player_y and npc_x < player_x) or
            (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
        ):
            # Manhattan distance
            return abs(npc_y - player_y) + abs(npc_x - player_x)
        return 1000
    
    def menu_rewards(self):
        # print(f'self.last_menu_state={self.last_menu_state}')
        if self.got_hm01 > 0:
            menu_state = self.api.game_state.name

            start_menu_pk = 'START_MENU_POKEMON'
            select_pk_menu = 'SELECT_POKEMON_'
            pokecenter_cancel_menu = 'POKECENTER_CANCEL'
            menu_reward_val = 0
            if menu_state == 'EXPLORING':
                self.last_menu_state = 'EXPLORING'
                return menu_reward_val
            if menu_state != self.last_menu_state:
                self.different_menu = True
            # Reward cutting (trying to use Cut) even if cutting nothing
            # Menu state stays the same for 'Cut failed' dialogue vs changing to SELECT_POKEMON_
            if menu_state == 'POKECENTER_CANCEL' and self.different_menu == True:
                self.cut_nothing += 1
                menu_reward_val += 0.0005 / (self.cut_nothing ** 2)     
            
            if start_menu_pk in menu_state:
                # print(f'{start_menu_pk} in menu_state=True')
                if self.last_menu_state == 'None' or self.last_menu_state == 'EXPLORING':
                    self.none_start += 1
                    menu_reward_val += 0.000055 / (self.none_start ** 2)
                self.last_menu_state = start_menu_pk
                
            if select_pk_menu in menu_state and pokecenter_cancel_menu not in self.last_menu_state:
                # print(f'{select_pk_menu} in menu_state=True')
                self.pk_menu += 1
                if self.last_menu_state == start_menu_pk:
                    self.start_sel += 1
                    menu_reward_val += 0.000055 / (self.start_sel ** 2)
                self.last_menu_state = select_pk_menu
                menu_reward_val += 0.000055 / (self.pk_menu ** 2)   
                
            if pokecenter_cancel_menu in menu_state and pokecenter_cancel_menu not in self.last_menu_state:
                # print(f'{pokecenter_cancel_menu} in menu_state:=True')
                self.pk_cancel_menu += 1
                if self.last_menu_state == select_pk_menu:
                    self.sel_cancel += 1
                    menu_reward_val += 0.000055 / (self.sel_cancel ** 2)
                self.last_menu_state = pokecenter_cancel_menu
                menu_reward_val += 0.000055 / (self.pk_cancel_menu ** 2)

        else:
            menu_reward_val = 0
        return menu_reward_val

    def current_coords(self):
        return self.last_10_coords[0]
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    
    def update_seen_map_dict(self):
        # if self.get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.current_map_id - 1
        x, y = self.current_coords
        if cur_map_id not in self.seen_map_dict:
            self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
        # # do not update if is warping
        if not self.is_warping:
            if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
                self.stuck_cnt += 1
                print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
                if self.stuck_cnt > 50:
                    print(f'stucked for > 50 steps, force ES')
                    self.early_done = True
                    self.stuck_cnt = 0
                # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
            else:
                self.stuck_cnt = 0
                self.seen_map_dict[cur_map_id][y, x] = self.time

    # Only reward exploration for the below coordinates
    # Default: path through Mt. Moon, then whole map rewardable.
    # Reward if True
    def rewardable_coords(self, glob_c, glob_r, map_n):
        if map_n in self.seen_maps_no_reward:
            return False
        else:
            return True # reward EVERYTHING
        # r, c, map_n = ram_map.position(self.game)
        # if map_n == 15 or map_n == 3:
        #     self.past_mt_moon = True
        # # Whole map included; excluded if in self.exclude_map_n
        # if self.past_mt_moon == True and map_n not in self.exclude_map_n:
        #     self.include_conditions = [(0 <= glob_c <= 436) and (0 <= glob_r <= 444)]
        # else:
        #     if map_n not in self.exclude_map_n:
        #         # Path through Mt. Moon
        #         self.include_conditions = [(80 >= glob_c >= 72) and (294 < glob_r <= 320),
        #         (69 < glob_c < 74) and (313 >= glob_r >= 295),
        #         (73 >= glob_c >= 72) and (220 <= glob_r <= 330),
        #         (75 >= glob_c >= 74) and (310 >= glob_r <= 319),
        #         (81 >= glob_c >= 73) and (294 < glob_r <= 313),
        #         (73 <= glob_c <= 81) and (294 < glob_r <= 308),
        #         (80 >= glob_c >= 74) and (330 >= glob_r >= 284),
        #         (90 >= glob_c >= 89) and (336 >= glob_r >= 328),
        #         # Viridian Pokemon Center
        #         (282 >= glob_r >= 277) and glob_c == 98,
        #         # Pewter Pokemon Center
        #         (173 <= glob_r <= 178) and glob_c == 42,
        #         # Route 4 Pokemon Center
        #         (131 <= glob_r <= 136) and glob_c == 132,
        #         (75 <= glob_c <= 76) and (271 < glob_r < 273),
        #         (82 >= glob_c >= 74) and (284 <= glob_r <= 302),
        #         (74 <= glob_c <= 76) and (284 >= glob_r >= 277),
        #         (76 >= glob_c >= 70) and (266 <= glob_r <= 277),
        #         (76 <= glob_c <= 78) and (274 >= glob_r >= 272),
        #         (74 >= glob_c >= 71) and (218 <= glob_r <= 266),
        #         (71 >= glob_c >= 67) and (218 <= glob_r <= 235),
        #         (106 >= glob_c >= 103) and (228 <= glob_r <= 244),
        #         (116 >= glob_c >= 106) and (228 <= glob_r <= 232),
        #         (116 >= glob_c >= 113) and (196 <= glob_r <= 232),
        #         (113 >= glob_c >= 89) and (208 >= glob_r >= 196),
        #         (97 >= glob_c >= 89) and (188 <= glob_r <= 214),
        #         (102 >= glob_c >= 97) and (189 <= glob_r <= 196),
        #         (89 <= glob_c <= 91) and (188 >= glob_r >= 181),
        #         (74 >= glob_c >= 67) and (164 <= glob_r <= 184),
        #         (68 >= glob_c >= 67) and (186 >= glob_r >= 184),
        #         (64 <= glob_c <= 71) and (151 <= glob_r <= 159),
        #         (71 <= glob_c <= 73) and (151 <= glob_r <= 156),
        #         (73 <= glob_c <= 74) and (151 <= glob_r <= 164),
        #         (103 <= glob_c <= 74) and (157 <= glob_r <= 156),
        #         (80 <= glob_c <= 111) and (155 <= glob_r <= 156),
        #         (111 <= glob_c <= 99) and (155 <= glob_r <= 150),
        #         (111 <= glob_c <= 154) and (150 <= glob_r <= 153),
        #         (138 <= glob_c <= 154) and (153 <= glob_r <= 160),
        #         (153 <= glob_c <= 154) and (153 <= glob_r <= 154),
        #         (143 <= glob_c <= 144) and (153 <= glob_r <= 154),
        #         (154 <= glob_c <= 158) and (134 <= glob_r <= 145),
        #         (152 <= glob_c <= 156) and (145 <= glob_r <= 150),
        #         (42 <= glob_c <= 43) and (173 <= glob_r <= 178),
        #         (158 <= glob_c <= 163) and (134 <= glob_r <= 135),
        #         (161 <= glob_c <= 163) and (114 <= glob_r <= 128),
        #         (163 <= glob_c <= 169) and (114 <= glob_r <= 115),
        #         (114 <= glob_c <= 169) and (167 <= glob_r <= 102),
        #         (169 <= glob_c <= 179) and (102 <= glob_r <= 103),
        #         (178 <= glob_c <= 179) and (102 <= glob_r <= 95),
        #         (178 <= glob_c <= 163) and (95 <= glob_r <= 96),
        #         (164 <= glob_c <= 163) and (110 <= glob_r <= 96),
        #         (163 <= glob_c <= 151) and (110 <= glob_r <= 109),
        #         (151 <= glob_c <= 154) and (101 <= glob_r <= 109),
        #         (151 <= glob_c <= 152) and (101 <= glob_r <= 97),
        #         (153 <= glob_c <= 154) and (97 <= glob_r <= 101),
        #         (151 <= glob_c <= 154) and (97 <= glob_r <= 98),
        #         (152 <= glob_c <= 155) and (69 <= glob_r <= 81),
        #         (155 <= glob_c <= 169) and (80 <= glob_r <= 81),
        #         (168 <= glob_c <= 184) and (39 <= glob_r <= 43),
        #         (183 <= glob_c <= 178) and (43 <= glob_r <= 51),
        #         (179 <= glob_c <= 183) and (48 <= glob_r <= 59),
        #         (179 <= glob_c <= 158) and (59 <= glob_r <= 57),
        #         (158 <= glob_c <= 161) and (57 <= glob_r <= 30),
        #         (158 <= glob_c <= 150) and (30 <= glob_r <= 31),
        #         (153 <= glob_c <= 150) and (34 <= glob_r <= 31),
        #         (168 <= glob_c <= 254) and (134 <= glob_r <= 140),
        #         (282 >= glob_r >= 277) and (436 >= glob_c >= 0), # Include Viridian Pokecenter everywhere
        #         (173 <= glob_r <= 178) and (436 >= glob_c >= 0), # Include Pewter Pokecenter everywhere
        #         (131 <= glob_r <= 136) and (436 >= glob_c >= 0), # Include Route 4 Pokecenter everywhere
        #         (137 <= glob_c <= 197) and (82 <= glob_r <= 142), # Mt Moon Route 3
        #         (137 <= glob_c <= 187) and (53 <= glob_r <= 103), # Mt Moon B1F
        #         (137 <= glob_c <= 197) and (16 <= glob_r <= 66), # Mt Moon B2F
        #         (137 <= glob_c <= 436) and (82 <= glob_r <= 444),  # Most of the rest of map after Mt Moon
        #     ]
        #         return any(self.include_conditions)
        #     else:
        #         return False

    def reset(self, seed=None, options=None, max_episode_steps=2560, reward_scale=4.0): # 20480 # 2560
        """Resets the game. Seeding is NOT supported"""
        roll = random.uniform(0, 1)
        if roll <= 0.05:
            load_pyboy_state(self.game, self.load_first_state()) # load the first save state every 5% of the time
        else:
            load_pyboy_state(self.game, self.load_last_state()) # load the last save state
        
        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=30)
            self.full_frame_writer.__enter__()
      
        self.full_name_log = Path(f'pokemon_party_log').with_suffix('.txt')
        self.full_name_checkpoint_log = Path(f'checkpoint_log_{self.env_id}').with_suffix('.txt')
        self.write_to_log()
            # Aggregate the data in each env log file. Default location of file: pufferlib/log_file_aggregator.py
        try:
            subprocess.run(['python', self.aggregate_file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running log_file_aggregator.py: {e}")

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.prev_map_n = None
        self.init_hidden_obj_mem()
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.death_count = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.last_reward = None
        self.b_seen_coords = {}

        self.reset_count += 1
        self.initial_states = self.always_starting_state
        return self.render(), {}

    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=schedule(wait=1, warmup=0, active=2, repeat=1),
    #         with_modules=True,
    #         with_stack=True,
    #     ) as prof:
    

    def step(self, action, fast_video=True):
        run_action_on_emulator(
            self.game,
            self.screen,
            ACTIONS[action],
            self.headless,
            fast_video=fast_video,
        )
        # step_timer_start = time.time()
        self.time += 1 # 1
        # print(f'self.time={self.time}')

        # s_process_game_states_time = time.time()
        # Call nimixx api
        self.api.process_game_states()
        # print(f'self.api.process_game_states()={self.api.process_game_states()}')
        # game_state = self.api.game_state.name
        # print(f'self.api.game_state.name(): {game_state}')
        # print(f'self.api.get_game_state(): {self.api.get_game_state()}')
        
        self.api.items.get_bag_item_ids()
        # print(f'self.api.items.get_bag_item_ids()={self.api.items.get_bag_item_ids()}')
        # print(f'self.api.game_state={self.api.game_state.name}')
        # 'START_MENU_POKEMON'
        # 'SELECT_POKEMON_'
        # 'POKECENTER_CANCEL'
        # e_process_game_states_time = time.time()
        # process_game_states_time = s_process_game_states_time - e_process_game_states_time
        
        # Reward menuing that can later be used to access Cut
        # s669 = time.time()
        self.menus_rewards = self.menu_rewards()
        # s671 = time.time()
        # menu_rewards_time = s671 - s669
        
        # print(f'manual times: process_game_states_time={process_game_states_time}, menu_rewards_time={menu_rewards_time}\n')
        
        # self.time += 1
        if self.save_video:
            self.add_video_frame()
        
        # Exploration reward     
        r, c, map_n = ram_map.position(self.game)
            # Convert local position to global position
        try:
            glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        except IndexError:
            print(f'IndexError: index {glob_r} or {glob_c} is out of bounds for axis 0 with size 444.')
            glob_r = 0
            glob_c = 0
        
        # # Only reward for specified coordinates, not all coordinates seen
        # if self.rewardable_coords(glob_c, glob_r, map_n):
        #     self.seen_coords.add((r, c, map_n))
        # else:
        #     self.seen_coords_no_reward.add((glob_c, glob_r, map_n))
        
        self.seen_coords.add((r, c, map_n))
        coord_string = f"x:{c} y:{r} m:{map_n}"
        self.b_seen_coords[coord_string] = self.time
        exploration_reward = 0.01 * len(self.b_seen_coords)
        # Exploration reward
        # exploration_reward = 0.01 * len(self.seen_coords) # BET: default 0.01
            
        # New map_n logic
        if map_n != self.prev_map_n:
            if map_n not in self.seen_maps_no_reward and map_n not in self.seen_maps:
                self.save_state()
                try:
                    i = self.saved_states_dict[f'{map_n}'] # number of states saved on this map_n
                except KeyError:
                    self.saved_states_dict[f'{map_n}'] = 0
                    i = 0
                self.saved_states_dict[f'{map_n}'] = i + 1 # increment number
                # print(f'state saved\nmap_n, saved_states_dict, {map_n, self.saved_states_dict}\nseen_maps_no_reward: {self.seen_maps_no_reward}')
            self.seen_maps_no_reward.add(map_n) # add map_n to unrewardable maps set
            
            self.prev_map_n = map_n
            
            if map_n not in self.seen_maps:
                first_elements = [tup[0] for tup in self.seen_maps_times]
                if map_n not in first_elements:
                    self.seen_maps_times.add((map_n, (time.time() - self.initial_wall_time)))
                self.full_name_checkpoint_log = Path(f'checkpoint_log_{self.env_id}').with_suffix('.txt')
                self.write_to_log()
                self.seen_maps.add(map_n)
                self.talk_to_npc_count[map_n] = 0  # Initialize NPC talk count for this new map
                # self.save_state() # Default save state location. Moving elsewhere for testing...
        
        # # BET testing 2/5/24 (not great results)
        # if self.time % 22000 == 0:
        #     self.save_state()
            
        self.update_pokedex()
        self.update_moves_obtained()

        # exploration_reward = 0.01 * sum(self.counts_map) / self.max_increments
        # print(f'new exp rew: {exploration_reward}')

        # print(f'old exp rew: {exploration_reward}')
        # np.sum(self.seen_coords_array)
        self.update_heat_map(r, c, map_n)
        
        # Level reward
        # Tapers after 30 to prevent overleveling
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels)) if self.max_level_sum or party_levels else 0
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
            
        # Healing and death rewards
        hp = ram_map.hp(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size

        # Only reward if not reviving at pokecenter
        if hp_delta > 0 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta

        # Dead if hp is zero
        # Don't rely on these flags for triggers...
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.is_dead = True
        elif hp > 0.01:  # TODO: Check if this matters
            self.is_dead = False
                
        # Writes data to a csv (and text log) every step. Gets big but surprisingly doesn't diminish SPS too much.
        # Don't log if party hp<0
        if self.stepwise_csv_logging:
            self.write_env_info_to_csv()

        if self.log and self.time % (self.max_episode_steps * 0.9) == 0:
            # print(f'self.time % 4000 == 0 TRUE')
            self.full_name_log = Path(f'pokemon_party_log').with_suffix('.txt')
            self.write_to_log()
    
        # Aggregate the data in each env log file. Default script location: pufferlib/log_file_aggregator.py
        if self.time % (self.max_episode_steps * 0.96) == 0:
            # print(f'selt.time % 5000 == 0 TRUE')
            try:
                subprocess.run(['python', self.aggregate_file_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running log_file_aggregator.py: {e}")
    
        # Update last known values for next iteration
        # Intentionally rewards 0 yet remains in rewards.
        self.last_hp = hp
        self.last_party_size = party_size
        death_reward = 0
        
        # Set rewards
        healing_reward = self.total_healing

        # Opponent level reward
        # Intentionally rewards 0 yet remains in rewards.
        opponent_levels = ram_map.opponent(self.game)
        max_opponent_level = max(opponent_levels) if opponent_levels else 0
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = 0 # 0.2 * self.max_opponent_level

        # Badge reward
        badges = ram_map.badges(self.game)
        if 1 > badges > 0:
            self.seen_maps_times.add(('Badge 1', (time.time() - self.initial_wall_time)))
        elif 2 > badges > 0:
            self.seen_maps_times.add(('Badge 2', (time.time() - self.initial_wall_time)))

        badges_reward = 5 * badges # 5

        # Save Bill
        bill_state = ram_map.saved_bill(self.game)
        bill_reward = 10 * bill_state
        
        # SS Anne appeared
        # Vestigial function that seems to work
        ss_anne_state = ram_map.ss_anne_appeared(self.game)
        if ss_anne_state:
            ss_anne_state_reward = 5
        else:
            ss_anne_state_reward = 0
        
        # Has HMs reward
        # Returns number of party pokemon with each HM
        poke_counts = self.poke_count_hms()
        poke_has_cut = poke_counts['Cut']
        poke_has_flash = poke_counts['Flash']
        poke_has_fly = poke_counts['Fly']
        poke_has_surf = poke_counts['Surf']
        poke_has_strength = poke_counts['Strength']
        
        # HM count
        hm_count = sum(poke_counts.values())
        
        # Bag items
        items_in_bag = self.get_items_in_bag()
        # print(f'items_in_bag: {items_in_bag}')

        # Rewards based on the number of each HM
        poke_has_cut_reward = poke_has_cut * 20
        poke_has_flash_reward = poke_has_flash
        poke_has_fly_reward = poke_has_fly
        poke_has_surf_reward = poke_has_surf
        poke_has_strength_reward = poke_has_strength
        
        # Used Cut
        if ram_map.used_cut(self.game) == 61:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write
            self.used_cut += 1
        
        # SS Anne rewards
        # Experimental
        got_hm01_reward = 5 if ram_map.got_hm01(self.game) else 0
        rubbed_captains_back_reward = 5 if ram_map.rubbed_captains_back(self.game) else 0
        ss_anne_left_reward = 5 if ram_map.ss_anne_left(self.game) else 0
        walked_past_guard_after_ss_anne_left_reward = 5 if ram_map.walked_past_guard_after_ss_anne_left(self.game) else 0
        started_walking_out_of_dock_reward = 5 if ram_map.started_walking_out_of_dock(self.game) else 0
        walked_out_of_dock_reward = 5 if ram_map.walked_out_of_dock(self.game) else 0

        # HM reward
        hm_reward = self.get_hm_rewards()
        hm_reward = hm_count * 5

        # SS Anne flags
        # Experimental
        got_hm01 = int(bool(got_hm01_reward))
        rubbed_captains_back = int(bool(rubbed_captains_back_reward))
        ss_anne_left = int(bool(ss_anne_left_reward))
        walked_past_guard_after_ss_anne_left = int(bool(walked_past_guard_after_ss_anne_left_reward))
        started_walking_out_of_dock = int(bool(started_walking_out_of_dock_reward))
        walked_out_of_dock = int(bool(walked_out_of_dock_reward))
        
        # got_hm01 flag to enable cut menu conditioning
        self.got_hm01 = got_hm01

        # Event reward
        events = ram_map.events(self.game)
        
        # Save on Event
        self.max_events = max(self.max_events, events) if self.max_events or events else 0
        event_reward = self.max_events

        money = ram_map.money(self.game)
        
        # Explore NPCs
        # Known to not actually work correctly. Counts first sign on each map as NPC. Treats NPCs as hidden obj and vice versa.
        # Intentionally left this way because it works better, i.e. proper NPC/hidden obj. rewarding/ignoring signs gets
        # worse results.
                # check if the font is loaded
        if ram_map.mem_val(self.game, 0xCFC4):
            # check if we are talking to a hidden object:
            if ram_map.mem_val(self.game, 0xCD3D) == 0x0 and ram_map.mem_val(self.game, 0xCD3E) == 0x0:
                # add hidden object to seen hidden objects
                self.seen_hidden_objs.add((ram_map.mem_val(self.game, 0xD35E), ram_map.mem_val(self.game, 0xCD3F)))
            else:
                # check if we are talking to someone
                # if ram_map.if_font_is_loaded(self.game):
                    # get information for player
                player_direction = ram_map.player_direction(self.game)
                player_y = ram_map.player_y(self.game)
                player_x = ram_map.player_x(self.game)
                # get the npc who is closest to the player and facing them
                # we go through all npcs because there are npcs like
                # nurse joy who can be across a desk and still talk to you
                mindex = (0, 0)
                minv = 1000
                for npc_bank in range(1):
                    for npc_id in range(1, ram_map.sprites(self.game) + 15):
                        npc_dist = self.find_neighboring_npc(npc_bank, npc_id, player_direction, player_x, player_y)
                        
                        self.find_neighboring_npc, 0, 0, player_direction, player_x, player_y

                        if npc_dist < minv:
                            mindex = (npc_bank, npc_id)
                            minv = npc_dist        
                
                self.find_neighboring_npc, mindex[0], mindex[1], player_direction, player_x, player_y
                
                self.seen_npcs.add((ram_map.map_n(self.game), mindex[0], mindex[1]))

        explore_npcs_reward = self.reward_scale * self.explore_npc_weight * len(self.seen_npcs) * 0.00015
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon) * 0.00010
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon) * 0.00010
        moves_obtained_reward = self.reward_scale * sum(self.moves_obtained) * 0.00010
        explore_hidden_objs_reward = self.reward_scale * self.explore_hidden_obj_weight * len(self.seen_hidden_objs) * 0.00015
        used_cut_reward = self.used_cut * 100

        # BET experimental Zero rewards for testing
        # self.menus_rewards = 0
        seen_pokemon_reward = 0
        
        reward = self.reward_scale * (
            + event_reward
            + ss_anne_state_reward
            + explore_npcs_reward # Doesn't reset on reset but maybe should?
            + seen_pokemon_reward # 0
            + caught_pokemon_reward
            + moves_obtained_reward
            + explore_hidden_objs_reward # Resets on reset
            + bill_reward
            + hm_reward
            + level_reward
            + opponent_level_reward # 0
            + death_reward  # 0 # Resets on reset
            + badges_reward
            + healing_reward # Resets each step
            + exploration_reward # Resets on reset
            + got_hm01_reward
            + rubbed_captains_back_reward
            + ss_anne_left_reward
            + walked_past_guard_after_ss_anne_left_reward
            + started_walking_out_of_dock_reward
            + walked_out_of_dock_reward
            + poke_has_cut_reward
            + used_cut_reward
            + poke_has_flash_reward
            + poke_has_fly_reward
            + poke_has_surf_reward
            + poke_has_strength_reward   
            + self.menus_rewards
            # + self.map_n_reward         
        )

        # info = {}
        # info = {
        #         "reward": {
        #             "delta": reward,
        #             "event": event_reward,
        #             "level": level_reward,
        #             "opponent_level": opponent_level_reward,
        #             "death": death_reward,
        #             "badges": badges_reward,
        #             "bill_saved_reward": bill_reward,
        #             "hm_count_reward": hm_reward,
        #             "got_hm01_reward": got_hm01_reward,
        #             "rubbed_captains_back_reward": rubbed_captains_back_reward,
        #             "ss_anne_state_reward": ss_anne_state_reward,
        #             "ss_anne_left_reward": ss_anne_left_reward,
        #             "walked_past_guard_after_ss_anne_left_reward": walked_past_guard_after_ss_anne_left_reward,
        #             "started_walking_out_of_dock_reward": started_walking_out_of_dock_reward,
        #             "walked_out_of_dock_reward": walked_out_of_dock_reward, 
        #             "exploration": exploration_reward,
        #             "explore_npcs_reward": explore_npcs_reward,
        #             "seen_pokemon_reward": seen_pokemon_reward,
        #             "caught_pokemon_reward": caught_pokemon_reward,
        #             "moves_obtained_reward": moves_obtained_reward,
        #             "hidden_obj_count_reward": explore_hidden_objs_reward,
        #             "poke_has_cut_reward": poke_has_cut_reward,
        #             "poke_has_flash_reward": poke_has_flash_reward,
        #             "poke_has_fly_reward": poke_has_fly_reward,
        #             "poke_has_surf_reward": poke_has_surf_reward,
        #             "poke_has_strength_reward": poke_has_strength_reward,
        #             "used_cut_reward": used_cut_reward,
        #         },
        #         "maps_explored": len(self.seen_maps),
        #         "party_size": party_size,
        #         "highest_pokemon_level": max(party_levels, default=0),
        #         "total_party_level": sum(party_levels),
        #         "deaths": self.death_count,
        #         "bill_saved": bill_state,
        #         "hm_count": hm_count,
        #         "got_hm01": got_hm01,
        #         "rubbed_captains_back": rubbed_captains_back,
        #         "ss_anne_left": ss_anne_left,
        #         "ss_anne_state": ss_anne_state,
        #         "walked_past_guard_after_ss_anne_left": walked_past_guard_after_ss_anne_left,
        #         "started_walking_out_of_dock": started_walking_out_of_dock,
        #         "walked_out_of_dock": walked_out_of_dock,
        #         "badge_1": float(badges >= 1),
        #         "badge_2": float(badges >= 2),
        #         "event": events,
        #         "money": money,
        #         "pokemon_exploration_map": self.counts_map,
        #         "seen_npcs_count": len(self.seen_npcs),
        #         "seen_pokemon": sum(self.seen_pokemon),
        #         "caught_pokemon": sum(self.caught_pokemon),
        #         "moves_obtained": sum(self.moves_obtained),
        #         "hidden_obj_count": len(self.seen_hidden_objs),
        #         "poke_has_cut": poke_has_cut,
        #         "poke_has_flash": poke_has_flash,
        #         "poke_has_fly": poke_has_fly,
        #         "poke_has_surf": poke_has_surf,
        #         "poke_has_strength": poke_has_strength,
        #         "used_cut": self.used_cut,
        #         "menus_reward": self.menus_rewards,
        #         "none->start": self.none_start,
        #         "start->sel": self.start_sel,
        #         "sel->cancel": self.sel_cancel,
        #         "cut_nothing": self.cut_nothing,
        #         # "logging": logging,
        #         # "env_uuid": self.env_id,
        #     }

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
            
        # get_time(self.update_pokedex)
        # get_time(self.update_moves_obtained)
        # get_time(self.update_heat_map, r, c, map_n)
        # get_time(self.write_env_info_to_csv)
        # get_time(self.write_to_log)
        # get_time(subprocess.run, ['python', self.aggregate_file_path], check=True)
        # get_time(self.poke_count_hms)
        # get_time(ram_map.write_mem, self.game, 0xCD4D, 00)
        # get_time(ram_map.got_hm01, self.game)
        # get_time(ram_map.rubbed_captains_back, self.game)
        # get_time(ram_map.ss_anne_left, self.game)
        # get_time(ram_map.walked_past_guard_after_ss_anne_left, self.game)
        # get_time(ram_map.started_walking_out_of_dock, self.game)
        # get_time(ram_map.walked_out_of_dock, self.game)
        # get_time(self.get_hm_rewards)
        # get_time(ram_map.mem_val, self.game, 0xCFC4)
        # get_time(ram_map.mem_val, self.game, 0xCD3D)
        # get_time(ram_map.mem_val, self.game, 0xCD3E)
        # get_time(ram_map.mem_val, self.game, 0xD35E)
        # get_time(ram_map.mem_val, self.game, 0xCD3F)
        # get_time(ram_map.if_font_is_loaded, self.game)
        # get_time(ram_map.player_direction, self.game)
        # get_time(ram_map.player_y, self.game)
        # get_time(ram_map.player_x, self.game)
        # get_time(self.poke_count_hms)
        # get_time(ram_map.used_cut, self.game)
        # get_time(ram_map.write_mem, self.game, 0xCD4D, 00)
        # get_time(ram_map.got_hm01, self.game)
        # get_time(ram_map.rubbed_captains_back, self.game)
        # get_time(ram_map.ss_anne_left, self.game)
        # get_time(ram_map.walked_past_guard_after_ss_anne_left, self.game)
        # get_time(ram_map.started_walking_out_of_dock, self.game)
        # get_time(ram_map.walked_out_of_dock, self.game)
        # get_time(ram_map.ss_anne_appeared, self.game)
        # get_time(ram_map.saved_bill, self.game)
        # get_time(sum, poke_counts.values())
        # get_time(ram_map.badges, self.game)
        # get_time(ram_map.opponent, self.game)
        # get_time(ram_map.events, self.game)
        # get_time(ram_map.money, self.game)
        # get_time(subprocess.run, ['python', self.aggregate_file_path], check=True)
        
        # average_times(self.time)
        
        # with open('ztimes.txt', 'w') as f:
        #     f.write(f'{average_times(self.time)}')
        
        if done or (self.time == self.max_episode_steps):
            info = {
                "reward": {
                    "delta": reward,
                    "event": event_reward,
                    "level": level_reward,
                    "opponent_level": opponent_level_reward,
                    "death": death_reward,
                    "badges": badges_reward,
                    "bill_saved_reward": bill_reward,
                    "hm_count_reward": hm_reward,
                    "got_hm01_reward": got_hm01_reward,
                    "rubbed_captains_back_reward": rubbed_captains_back_reward,
                    "ss_anne_state_reward": ss_anne_state_reward,
                    "ss_anne_left_reward": ss_anne_left_reward,
                    "walked_past_guard_after_ss_anne_left_reward": walked_past_guard_after_ss_anne_left_reward,
                    "started_walking_out_of_dock_reward": started_walking_out_of_dock_reward,
                    "walked_out_of_dock_reward": walked_out_of_dock_reward, 
                    "exploration": exploration_reward,
                    "explore_npcs_reward": explore_npcs_reward,
                    "seen_pokemon_reward": seen_pokemon_reward,
                    "caught_pokemon_reward": caught_pokemon_reward,
                    "moves_obtained_reward": moves_obtained_reward,
                    "hidden_obj_count_reward": explore_hidden_objs_reward,
                    "poke_has_cut_reward": poke_has_cut_reward,
                    "poke_has_flash_reward": poke_has_flash_reward,
                    "poke_has_fly_reward": poke_has_fly_reward,
                    "poke_has_surf_reward": poke_has_surf_reward,
                    "poke_has_strength_reward": poke_has_strength_reward,
                    "used_cut_reward": used_cut_reward,
                    "menus_reward": self.menus_rewards,
                    "healing_reward": healing_reward,
                    # "new_map_n_reward": self.map_n_reward,
                },
                "maps_explored": len(self.seen_maps),
                "party_size": party_size,
                "highest_pokemon_level": max(party_levels, default=0),
                "total_party_level": sum(party_levels),
                "deaths": self.death_count,
                "bill_saved": bill_state,
                "hm_count": hm_count,
                "got_hm01": got_hm01,
                "rubbed_captains_back": rubbed_captains_back,
                "ss_anne_left": ss_anne_left,
                "ss_anne_state": ss_anne_state,
                "walked_past_guard_after_ss_anne_left": walked_past_guard_after_ss_anne_left,
                "started_walking_out_of_dock": started_walking_out_of_dock,
                "walked_out_of_dock": walked_out_of_dock,
                "badge_1": float(badges >= 1),
                "badge_2": float(badges >= 2),
                "event": events,
                "money": money,
                "pokemon_exploration_map": self.counts_map,
                "seen_npcs_count": len(self.seen_npcs),
                "seen_pokemon": sum(self.seen_pokemon),
                "caught_pokemon": sum(self.caught_pokemon),
                "moves_obtained": sum(self.moves_obtained),
                "hidden_obj_count": len(self.seen_hidden_objs),
                "poke_has_cut": poke_has_cut,
                "poke_has_flash": poke_has_flash,
                "poke_has_fly": poke_has_fly,
                "poke_has_surf": poke_has_surf,
                "poke_has_strength": poke_has_strength,
                "used_cut": self.used_cut,
                "cut_nothing": self.cut_nothing,
                "total_healing": self.total_healing,
                "checkpoints": self.seen_maps_times,
                "saved_states_dict": list(self.saved_states_dict.items()),
                
                # "200_step_pyboy_save_state": self.pass_states(),
                # "logging": logging,
                # "env_uuid": self.env_id,
            }

        if self.verbose:            
            print(
                # f'number of signs: {ram_map.signs(self.game)}, number of sprites: {ram_map.sprites(self.game)}\n',
                f"steps: {self.time}\n",
                # f"seen_npcs #: {len(self.seen_npcs)}\n",
                # f"seen_npcs set: {self.seen_npcs}\n",
                # f"is_in_battle: {ram_map.is_in_battle(self.game)}",
                f"exploration reward: {exploration_reward}\n",
                # f"explore_npcs reward: {explore_npcs_reward}\n",
                # f"level_Reward: {level_reward}\n",
                # f"healing: {healing_reward}\n",
                # f"death: {death_reward}\n",
                # f"op_level: {opponent_level_reward}\n",
                # f"badges reward: {badges_reward}\n",
                # f"event reward: {event_reward}\n",
                # f"money: {money}\n",
                f"ai reward: {reward}\n",
                f"map_n: {map_n}\n",
                f"bag_items: {items_in_bag}\n",
                f"Info: {info}\n",
            )
        # step_timer_end = time.time()
        # step_time = step_timer_end - step_timer_start
        # print(f'step_time={step_time}')
        # with open('ztimes.txt', 'w') as f:
        #     f.write(f'Total Step Time: {step_time}')
        return self.render(), reward, done, done, info
    #     prof.step()
    # prof.export_chrome_trace("environment_trade.json")
    # raise