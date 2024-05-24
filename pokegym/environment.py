from pdb import set_trace as T
from pathlib import Path
import uuid
from gymnasium import spaces
import numpy as np
import json
from skimage.transform import resize
from collections import deque
import io, os
import random
import matplotlib.pyplot as plt
import mediapy as media

from pokegym import ram_map, game_map, data
from pokegym.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)

STATE_PATH = __file__.rstrip("environment.py") + "States/"
GLITCH = __file__.rstrip("environment.py") + "glitch/"
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

class Base:
    def __init__(self,rom_path="pokemon_red.gb", state_path=None,
                 headless=True, save_video=False, quiet=False, **kwargs,):
        
        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        state_path = STATE_PATH + "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=True, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.env_id = Path(f'{str(uuid.uuid4())[:4]}')
        self.reset_count = 0
        self.bet_seen_coords = set()

        ## Bool ##
        self.use_screen_memory = True

        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2) # 72, 80, 3

        if self.use_screen_memory:
            self.global_map = np.full((444, 436, 1), -1, dtype=np.uint8)
            self.obs_size += (4,)
        else:
            self.obs_size += (3,)
        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=self.obs_size
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
    
    ## Gym Environment ##
    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}
    
    def render(self):
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.game)
            glob_r, glob_c = game_map.local_to_global(r, c, map_n)
            combined_obs = np.concatenate(
                (
                    self.screen.screen_ndarray()[::2, ::2],
                    self.bet_fixed_window(glob_r, glob_c),
                ),
                axis=2,
            )
            return combined_obs
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

    ## Save and Load State ##
    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states.append(state)

    def glitch_state(self):
        saved = open(f"{GLITCH}glitch_{self.reset_count}_{self.env_id}.state", "wb")
        self.game.save_state(saved)
        party = data.logs(self.game)
        with open(f"{GLITCH}log_{self.reset_count}_{self.env_id}.txt", 'w') as log:
            log.write(party)

    def load_last_state(self):
        return self.initial_states[len(self.initial_states) - 1]

    def load_first_state(self):
        return self.initial_states[0]

    def load_random_state(self):
        rand_idx = random.randint(0, len(self.initial_states) - 1)
        return self.initial_states[rand_idx]
    
    ## BET FIXED WINDOW ##
    def load_and_apply_region_data(self):
        # Load map data from the JSON file
        MAP_PATH = __file__.rstrip('environment.py') + 'map_data.json'
        map_data = json.load(open(MAP_PATH, 'r'))['regions']

        for region in map_data:
            region_id = int(region["id"])
            if region_id == -1:
                continue  # Skip region with ID -1
            x_start, y_start = region["coordinates"]
            width, height = region["tileSize"]
            # Assuming regions need to be visually distinct from the black background
            self.global_map[y_start:y_start + height, x_start:x_start + width] = 0  # Example value to make regions visible

    def update_bet_seen_coords(self):
        for (y, x) in self.bet_seen_coords:
            if 0 <= y < 444 and 0 <= x < 436:
                self.global_map[y, x] = 255  # Mark visited locations white

    def bet_fixed_window(self, glob_r, glob_c):
        h_w, w_w = self.observation_space.shape[0], self.observation_space.shape[1]
        half_window_height = h_w // 2  # 72 // 2
        half_window_width = w_w // 2  # 80 // 2
        start_y = max(0, glob_r - half_window_height)
        start_x = max(0, glob_c - half_window_width)
        end_y = min(444, glob_r + half_window_height)
        end_x = min(436, glob_c + half_window_width)
        viewport = self.global_map[start_y:end_y, start_x:end_x] # print global_map  [(0, 436), (0, 444), (y,x) ]
        if viewport.shape != (h_w, w_w):
            viewport = resize(viewport, (h_w, w_w), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
        return viewport

class Environment(Base):
    def __init__(self,rom_path="pokemon_red.gb",state_path=None,headless=True,save_video=False,quiet=False,verbose=False,**kwargs,):

        super().__init__(rom_path, state_path, headless, save_video, quiet, **kwargs)
        self.counts_map = np.zeros((444, 436))
        self.death_count = 0
        self.verbose = verbose
        self.include_conditions = []
        self.seen_maps_difference = set()
        self.current_maps = []
        self.is_dead = False
        self.last_map = -1
        self.log = True
        self.used_cut = 0
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203]
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        load_pyboy_state(self.game, self.load_last_state())

        # ## BET FIXED WINDOW INIT ##
        self.load_and_apply_region_data()

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
    
    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        self.reset_count += 1

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

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.last_reward = None

        self.prev_map_n = None
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.hm_count = 0
        self.cut = 0
        self.cut_coords = {}
        self.cut_tiles = {} # set([])
        self.cut_state = deque(maxlen=3)
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_cancel_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = {} # np.zeros(255, dtype=np.uint8)
        self.town = 1
        self.gymthree = 0
        self.gymfour = 0

        # BET fixed window
        self.bet_seen_coords = set()
        self.counts_map = np.zeros((444, 436))
        return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1

        if self.save_video:
            self.add_video_frame()

        # Exploration
        r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        self.seen_coords.add((r, c, map_n))
        # Exploration reward
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        self.bet_seen_coords.add((glob_r, glob_c))
        self.update_bet_seen_coords()
        self.update_heat_map(r, c, map_n)

        if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0: # pre hideout
            if map_n in self.poketower:
                exploration_reward = 0
            elif map_n in self.pokehideout:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD81B, 7)) == 1: # hideout done poketower not done
            if map_n in self.poketower:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD76C, 0)) == 0 and int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 1: # tower done no flute
            if map_n == 149:
                exploration_reward = (0.03 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten pre silphco
            if map_n in self.silphco:
                exploration_reward = (0.03 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 1 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten post silphco
            if map_n in self.silphco:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        else:
            exploration_reward = (0.02 * len(self.seen_coords))

        if map_n == 92:
            self.gymthree = 1
        if map_n == 134:
            self.gymfour = 1

        # Level reward
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 15 + (self.max_level_sum - 15) / 4

        # Healing and death rewards
        hp = ram_map.hp(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size
        if hp_delta > 0.5 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.is_dead = True
        elif hp > 0.01:  # TODO: Check if this matters
            self.is_dead = False
        self.last_hp = hp
        self.last_party_size = party_size
        death_reward = 0 # -0.08 * self.death_count  # -0.05
        healing_reward = self.total_healing

        # HM reward
        hm_count = ram_map.get_hm_count(self.game)
        if hm_count >= 1 and self.hm_count == 0:
            self.hm_count = 1

        if ram_map.mem_val(self.game, 0xD057) == 0: # is_in_battle if 1
            if self.cut == 1:
                player_direction = self.game.get_memory_value(0xC109)
                y, x, map_id = ram_map.position(self.game) # this is [y, x, z]  # x, y, map_id
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
                    self.cut_coords[coords] = 5 # from 14
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                if int(ram_map.read_bit(self.game, 0xD803, 0)):
                    if ram_map.check_if_in_start_menu(self.game):
                        self.seen_start_menu = 1
                    if ram_map.check_if_in_pokemon_menu(self.game):
                        self.seen_pokemon_menu = 1
                    if ram_map.check_if_in_stats_menu(self.game):
                        self.seen_stats_menu = 1
                    if ram_map.check_if_in_bag_menu(self.game):
                        self.seen_bag_menu = 1

        if ram_map.used_cut(self.game) == 61:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write resets tile check
            self.used_cut += 1

        # Misc
        badges = ram_map.badges(self.game)
        self.update_pokedex()
        self.update_moves_obtained()

        silph = ram_map.silph_co(self.game)
        rock_tunnel = ram_map.rock_tunnel(self.game)
        ssanne = ram_map.ssanne(self.game)
        mtmoon = ram_map.mtmoon(self.game)
        routes = ram_map.routes(self.game)
        misc = ram_map.misc(self.game)
        snorlax = ram_map.snorlax(self.game)
        hmtm = ram_map.hmtm(self.game)
        bill = ram_map.bill(self.game)
        oak = ram_map.oak(self.game)
        towns = ram_map.towns(self.game)
        lab = ram_map.lab(self.game)
        mansion = ram_map.mansion(self.game)
        safari = ram_map.safari(self.game)
        dojo = ram_map.dojo(self.game)
        hideout = ram_map.hideout(self.game)
        tower = ram_map.poke_tower(self.game)
        gym1 = ram_map.gym1(self.game)
        gym2 = ram_map.gym2(self.game)
        gym3 = ram_map.gym3(self.game)
        gym4 = ram_map.gym4(self.game)
        gym5 = ram_map.gym5(self.game)
        gym6 = ram_map.gym6(self.game)
        gym7 = ram_map.gym7(self.game)
        gym8 = ram_map.gym8(self.game)
        rival = ram_map.rival(self.game)

        cut_rew = self.cut * 10
        event_reward = sum([silph, rock_tunnel, ssanne, mtmoon, routes, misc, snorlax, hmtm, bill, oak, towns, lab, mansion, safari, dojo, hideout, tower, gym1, gym2, gym3, gym4, gym5, gym6, gym7, gym8, rival])
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
        moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)
        used_cut_rew = self.used_cut * 0.1
        cut_coords = sum(self.cut_coords.values()) * 1.0
        cut_tiles = len(self.cut_tiles) * 1.0
        start_menu = self.seen_start_menu * 0.01
        pokemon_menu = self.seen_pokemon_menu * 0.1
        stats_menu = self.seen_stats_menu * 0.1
        bag_menu = self.seen_bag_menu * 0.1
        that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu ) / 2

        reward = self.reward_scale * (
            + level_reward
            + healing_reward
            + exploration_reward
            + cut_rew
            + event_reward
            + seen_pokemon_reward
            + caught_pokemon_reward
            + moves_obtained_reward
            + used_cut_rew
            + cut_coords
            + cut_tiles
            + that_guy
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
        if done:
            poke = self.game.get_memory_value(0xD16B)
            level = self.game.get_memory_value(0xD18C)
            info = {
                "Events": {
                    "silph": silph,
                    "rock_tunnel": rock_tunnel,
                    "ssanne": ssanne,
                    "mtmoon": mtmoon,
                    "routes": routes,
                    "misc": misc,
                    "snorlax": snorlax,
                    "hmtm": hmtm,
                    "bill": bill,
                    "oak": oak,
                    "towns": towns,
                    "lab": lab,
                    "mansion": mansion,
                    "safari": safari,
                    "dojo": dojo,
                    "hideout": hideout,
                    "tower": tower,
                    "gym1": gym1,
                    "gym2": gym2,
                    "gym3": gym3,
                    "gym4": gym4,
                    # "gym5": gym5,
                    # "gym6": gym6,
                    # "gym7": gym7,
                    # "gym8": gym8,
                    "rival": rival,
                },
                "BET": {
                    "Reward_Delta": reward,
                    "Seen_Poke": seen_pokemon_reward,
                    "Caught_Poke": caught_pokemon_reward,
                    "Moves_Obtain": moves_obtained_reward,
                    # "Get_HM": hm_reward,
                    "Level": level_reward,
                    "Death": death_reward,
                    "Healing": healing_reward,
                    "Exploration": exploration_reward,
                    "Taught_Cut": cut_rew,
                    "Menuing": that_guy,
                    "Used_Cut": used_cut_rew,
                    "Cut_Coords": cut_coords,
                    "Cut_Tiles": cut_tiles,
                },
                "hm_count": hm_count,
                "cut_taught": self.cut,
                "badge_1": float(badges >= 1),
                "badge_2": float(badges >= 2),
                "badge_3": float(badges >= 3),
                "badge_4": float(badges >= 4),
                "badge_5": float(badges >= 5),
                "badge_6": float(badges >= 6),
                "badge_7": float(badges >= 7),
                "badge_8": float(badges >= 8),
                "badges": float(badges),
                "maps_explored": np.sum(self.seen_maps),
                "party_size": party_size,
                "moves_obtained": sum(self.moves_obtained),
                "deaths": self.death_count,
                'cut_coords': cut_coords,
                'cut_tiles': cut_tiles,
                'bag_menu': bag_menu,
                'stats_menu': stats_menu,
                'pokemon_menu': pokemon_menu,
                'start_menu': start_menu,
                'used_cut': self.used_cut,
                'gym_three': self.gymthree,
                'gym_four': self.gymfour,
            }
        return self.render(), reward, done, done, info
    
    ## BET FIXED WINDOW ##
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.video())

    def ensure_visualization_setup(self):
          # Initialize transparent background
        self.load_and_apply_region_data()

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

# from pathlib import Path
# from pdb import set_trace as T
# import types
# import uuid
# from gymnasium import Env, spaces
# import numpy as np

# from collections import defaultdict, deque
# import io, os
# import random
# from pyboy.utils import WindowEvent

# import matplotlib.pyplot as plt
# from pathlib import Path
# import mediapy as media
# from pokegym import game_map

# from pokegym import ram_map
# from pokegym.pyboy_binding import (
#     ACTIONS,
#     make_env,
#     open_state_file,
#     load_pyboy_state,
#     run_action_on_emulator,
# )
# from pokegym import data
# import json
# from skimage.transform import resize


# STATE_PATH = __file__.rstrip("environment.py") + "States/"
# GLITCH = __file__.rstrip("environment.py") + "glitch/"
# CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
# CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
# CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
# def get_random_state():
#     state_files = [f for f in os.listdir(STATE_PATH) if f.endswith(".state")]
#     if not state_files:
#         raise FileNotFoundError("No State files found in the specified directory.")
#     return random.choice(state_files)
# state_file = get_random_state()
# randstate = os.path.join(STATE_PATH, state_file)

# class Base:
#     def __init__(
#         self,
#         rom_path="pokemon_red.gb",
#         state_path=None,
#         headless=True,
#         save_video=False,
#         quiet=False,
#         **kwargs,
#     ):
#         self.state_file = get_random_state()
#         self.randstate = os.path.join(STATE_PATH, self.state_file)
#         """Creates a PokemonRed environment"""
#         if state_path is None:
#             state_path = STATE_PATH + "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
#                 # Make the environment
#         self.game, self.screen = make_env(rom_path, headless, quiet, save_video=True, **kwargs)
#         self.initial_states = [open_state_file(state_path)]
#         self.save_video = save_video
#         self.headless = headless
#         self.mem_padding = 2
#         self.memory_shape = 80
#         self.use_screen_memory = True
#         self.screenshot_counter = 0
#         self.env_id = Path(f'{str(uuid.uuid4())[:4]}')
#         self.reset_count = 0               
#         self.explore_hidden_obj_weight = 1
#         self.bet_seen_coords = set()
#         self.global_map = np.full((444, 436), 0, dtype=np.uint8)  # Initialize with background
#         self.image = None
#         self.bet_seen_coords = set()
#         self.step_counter = 0

#         R, C = self.screen.raw_screen_buffer_dims()
#         self.obs_size = (R // 2, C // 2) # 72, 80, 3

#         if self.use_screen_memory:
#             # self.screen_memory = defaultdict(
#             #     lambda: np.zeros((255, 255, 1), dtype=np.uint8)
#             # )
#             self.obs_size += (4,)
#         else:
#             self.obs_size += (3,)
#         self.observation_space = spaces.Box(
#             low=0, high=255, dtype=np.uint8, shape=self.obs_size
#         )
#         self.action_space = spaces.Discrete(len(ACTIONS))
    
        
#     def load_and_apply_region_data(self):
#         # Load map data from the JSON file
#         MAP_PATH = __file__.rstrip('environment.py') + 'map_data.json'
#         map_data = json.load(open(MAP_PATH, 'r'))['regions']

#         for region in map_data:
#             region_id = int(region["id"])
#             if region_id == -1:
#                 continue  # Skip region with ID -1
#             x_start, y_start = region["coordinates"]
#             width, height = region["tileSize"]
#             # Assuming regions need to be visually distinct from the black background
#             self.global_map[y_start:y_start + height, x_start:x_start + width] = 0  # Example value to make regions visible            
    
#     def update_bet_seen_coords(self):
#         for (y, x) in self.bet_seen_coords:
#             if 0 <= y < 444 and 0 <= x < 436:
#                 self.global_map[y, x] = 255  # Mark visited locations white   
    
#     # def save_screenshot(self, event, map_n):
#     def save_screenshot(self):
#         self.screenshot_counter += 1
#         ss_dir = Path('screenshots')
#         ss_dir.mkdir(exist_ok=True)
#         r, c, map_n = ram_map.position(self.game)
#         glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        
#         bet_window = self.bet_fixed_window(glob_r, glob_c)
        
#         # Save the grayscale image using plt.imsave with cmap='gray'
#         plt.imsave(
#             ss_dir / Path(f'{self.screenshot_counter}.jpeg'),
#             bet_window,
#             cmap='gray'
#         )
#             # self.screen.screen_ndarray())  # (144, 160, 3)

#     def save_state(self):
#         state = io.BytesIO()
#         state.seek(0)
#         self.game.save_state(state)
#         self.initial_states.append(state)

#     def glitch_state(self):
#         saved = open(f"{GLITCH}glitch_{self.reset_count}_{self.env_id}.state", "wb")
#         self.game.save_state(saved)
#         party = data.logs(self.game)
#         with open(f"{GLITCH}log_{self.reset_count}_{self.env_id}.txt", 'w') as log:
#             log.write(party)
    
#     def load_last_state(self):
#         return self.initial_states[len(self.initial_states) - 1]
    
#     def load_first_state(self):
#         return self.initial_states[0]
    
#     def load_random_state(self):
#         rand_idx = random.randint(0, len(self.initial_states) - 1)
#         return self.initial_states[rand_idx]

#     def reset(self, seed=None, options=None):
#         """Resets the game. Seeding is NOT supported"""
#         return self.screen.screen_ndarray(), {}

#     # def get_fixed_window(self, arr, y, x, window_size):
#     #     height, width, _ = arr.shape
#     #     h_w, w_w = window_size[0], window_size[1]
#     #     h_w, w_w = window_size[0] // 2, window_size[1] // 2

#     #     y_min = max(0, y - h_w)
#     #     y_max = min(height, y + h_w + (window_size[0] % 2))
#     #     x_min = max(0, x - w_w)
#     #     x_max = min(width, x + w_w + (window_size[1] % 2))

#     #     window = arr[y_min:y_max, x_min:x_max]

#     #     pad_top = h_w - (y - y_min)
#     #     pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
#     #     pad_left = w_w - (x - x_min)
#     #     pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)

#     #     return np.pad(
#     #         window,
#     #         ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
#     #         mode="constant",
#     #     )


#     def bet_fixed_window(self, glob_r, glob_c):
#         h_w, w_w = self.observation_space.shape[0], self.observation_space.shape[1]
#         half_window_height = h_w // 2  # 72 // 2
#         half_window_width = w_w // 2  # 80 // 2        
#         start_y = max(0, glob_r - half_window_height)
#         start_x = max(0, glob_c - half_window_width)
#         end_y = min(444, glob_r + half_window_height)
#         end_x = min(436, glob_c + half_window_width)
#         viewport = self.global_map[start_y:end_y, start_x:end_x]
#         if viewport.shape != (h_w, w_w):
#             viewport = resize(viewport, (h_w, w_w), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
#         return viewport

#     def render(self):
#         if self.use_screen_memory:
#             r, c, map_n = ram_map.position(self.game)
#             glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#             bet_window = self.bet_fixed_window(glob_r, glob_c)
#             # self.save_step_visualization(bet_window)  # Save visualization (for testing)
#             combined_obs = np.concatenate(
#                 (
#                     self.screen.screen_ndarray()[::2, ::2],
#                     bet_window,
#                 ),
#                 axis=2,
#             )
#             return combined_obs
#         else:
#             return self.screen.screen_ndarray()[::2, ::2]

#     def save_step_visualization(self, bet_window):
#         """Saves the visualization of bet_window."""
#         self.step_counter += 1
#         if self.step_counter % 100 == 0:  # Save every 100 steps
#             plt.imshow(bet_window, cmap='gray')
#             plt.title(f'bet_window Step {self.step_counter}')
#             plt.savefig(f'bet_window_step_{self.step_counter}.png')
#             plt.close()

#     def step(self, action):
#         run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
#         return self.render(), 0, False, False, {}
        
#     def video(self):
#         video = self.screen.screen_ndarray()
#         return video

#     def close(self):
#         self.game.stop(False)

# class Environment(Base):
#     def __init__(self,rom_path="pokemon_red.gb",state_path=None,headless=True,save_video=False,quiet=False,verbose=False,**kwargs,):

#         super().__init__(rom_path, state_path, headless, save_video, quiet, **kwargs)
#         self.counts_map = np.zeros((444, 436))
#         self.death_count = 0
#         self.verbose = verbose
#         self.include_conditions = []
#         self.seen_maps_difference = set()
#         self.current_maps = []
#         self.is_dead = False
#         self.last_map = -1
#         self.log = True
#         self.used_cut = 0
#         # self.seen_coords = set()
#         self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         self.poketower = [142, 143, 144, 145, 146, 147, 148]
#         self.pokehideout = [199, 200, 201, 202, 203]
#         self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
#         # ## BET FIXED WINDOW INIT ##
#         self.global_map = np.full((444, 436), 0, dtype=np.uint8)  # Initialize with background
#         self.image = None
#         self.seen_coords = set()
#         self.load_and_apply_region_data()
#         load_pyboy_state(self.game, self.load_last_state())
    
#     def ensure_visualization_setup(self):
#         # self.fig, self.ax = plt.subplots()  # Ensures that self.fig and self.ax are initialized immediately
#         # plt.ion()  # Turn on interactive mode
#         self.global_map = np.full((444, 436, 1), -1, dtype=np.uint8)  # Initialize transparent background
#         # self.downscaled_map = np.full((77, 80), -1, dtype=np.uint8) 
#         self.pooled_map = np.full((140, 160, 1), 0, dtype=np.uint8)
#         self.image = None
#         self.seen_coords = set()
#         self.load_and_apply_region_data()     
    
#     def update_pokedex(self):
#         for i in range(0xD30A - 0xD2F7):
#             caught_mem = self.game.get_memory_value(i + 0xD2F7)
#             seen_mem = self.game.get_memory_value(i + 0xD30A)
#             for j in range(8):
#                 self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
#                 self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  

#     def town_state(self):
#         state = io.BytesIO()
#         state.seek(0)
#         self.game.save_state(state)
#         self.initial_states.append(state)
#         return 
    
#     def update_moves_obtained(self):
#         # Scan party
#         for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
#             if self.game.get_memory_value(i) != 0:
#                 for j in range(4):
#                     move_id = self.game.get_memory_value(i + j + 8)
#                     if move_id != 0:
#                         if move_id != 0:
#                             self.moves_obtained[move_id] = 1
#                         if move_id == 15:
#                             self.cut = 1
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
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             and ram_map.mem_val(self.game, 0xFF8C) == 6
#             and ram_map.mem_val(self.game, 0xCF94) == 1
#         )

#     def check_if_in_bag_menu(self) -> bool:
#         return (
#             ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             # and newram_map.mem_val(self.game, 0xFF8C) == 6 # only sometimes
#             and ram_map.mem_val(self.game, 0xCF94) == 3
#         )

#     def check_if_cancel_bag_menu(self, action) -> bool:
#         return (
#             action == WindowEvent.PRESS_BUTTON_A
#             and ram_map.mem_val(self.game, 0xD057) == 0
#             and ram_map.mem_val(self.game, 0xCF13) == 0
#             # and newram_map.mem_val(self.game, 0xFF8C) == 6
#             and ram_map.mem_val(self.game, 0xCF94) == 3
#             and ram_map.mem_val(self.game, 0xD31D) == ram_map.mem_val(self.game, 0xCC36) + ram_map.mem_val(self.game, 0xCC26)
#         )
        
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

#     def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
#         """Resets the game. Seeding is NOT supported"""
#         self.reset_count += 1
        
#         if self.save_video:
#             base_dir = self.s_path
#             base_dir.mkdir(parents=True, exist_ok=True)
#             full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
#             self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
#             self.full_frame_writer.__enter__()

#         # if self.use_screen_memory:
#         #     self.screen_memory = defaultdict(
#         #         lambda: np.zeros((255, 255, 1), dtype=np.uint8)
#         #     )

#         self.time = 0
#         self.max_episode_steps = max_episode_steps
#         self.reward_scale = reward_scale
#         self.last_reward = None

#         self.prev_map_n = None
#         self.max_events = 0
#         self.max_level_sum = 0
#         self.max_opponent_level = 0
#         self.seen_coords = set()
#         self.seen_maps = set()
#         self.total_healing = 0
#         self.last_hp = 1.0
#         self.last_party_size = 1
#         self.hm_count = 0
#         self.cut = 0
#         self.cut_coords = {}
#         self.cut_tiles = {} # set([])
#         self.cut_state = deque(maxlen=3)
#         self.seen_start_menu = 0
#         self.seen_pokemon_menu = 0
#         self.seen_stats_menu = 0
#         self.seen_bag_menu = 0
#         self.seen_cancel_bag_menu = 0
#         self.seen_pokemon = np.zeros(152, dtype=np.uint8)
#         self.caught_pokemon = np.zeros(152, dtype=np.uint8)
#         self.moves_obtained = {} # np.zeros(255, dtype=np.uint8)
#         self.town = 1
#         self.gymthree = 0
#         self.gymfour = 0

#         # BET fixed window
#         self.ensure_visualization_setup()
#         self.bet_seen_coords = set()
#         r, c, map_n = ram_map.position(self.game)
#         glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#         self.bet_seen_coords.add((glob_r, glob_c))
#         self.update_bet_seen_coords()

#         return self.render(), {}

#     def step(self, action, fast_video=True):
#         run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
#         self.time += 1

#         if self.save_video:
#             self.add_video_frame()
        
#         # Exploration
#         r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
#         # Exploration reward
#         glob_r, glob_c = game_map.local_to_global(r, c, map_n)
#         self.bet_seen_coords.add((glob_r, glob_c))
#         self.update_bet_seen_coords()
#         self.update_heat_map(r, c, map_n)
        
#         # # vis
#         # if self.time % 100 == 0:
#         #     self.save_screenshot()        
        
#         if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0: # pre hideout
#             if map_n in self.poketower:
#                 exploration_reward = 0
#             elif map_n in self.pokehideout:
#                 exploration_reward = (0.03 * len(self.seen_coords))
#             else:
#                 exploration_reward = (0.02 * len(self.seen_coords))
#         elif int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD81B, 7)) == 1: # hideout done poketower not done
#             if map_n in self.poketower:
#                 exploration_reward = (0.03 * len(self.seen_coords))
#             else:
#                 exploration_reward = (0.02 * len(self.seen_coords))
#         elif int(ram_map.read_bit(self.game, 0xD76C, 0)) == 0 and int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 1: # tower done no flute
#             if map_n == 149:
#                 exploration_reward = (0.03 * len(self.seen_coords))
#             elif map_n in self.poketower:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             elif map_n in self.pokehideout:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             else:
#                 exploration_reward = (0.02 * len(self.seen_coords))
#         elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten pre silphco
#             if map_n in self.silphco:
#                 exploration_reward = (0.03 * len(self.seen_coords))
#             elif map_n in self.poketower:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             elif map_n in self.pokehideout:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             else:
#                 exploration_reward = (0.02 * len(self.seen_coords))
#         elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 1 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten post silphco
#             if map_n in self.silphco:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             elif map_n in self.poketower:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             elif map_n in self.pokehideout:
#                 exploration_reward = (0.01 * len(self.seen_coords))
#             else:
#                 exploration_reward = (0.02 * len(self.seen_coords))
#         else:
#             exploration_reward = (0.02 * len(self.seen_coords))

#         if map_n == 92:
#             self.gymthree = 1
#         if map_n == 134:
#             self.gymfour = 1

#         # Level reward
#         party_size, party_levels = ram_map.party(self.game)
#         self.max_level_sum = max(self.max_level_sum, sum(party_levels))
#         if self.max_level_sum < 15:
#             level_reward = 1 * self.max_level_sum
#         else:
#             level_reward = 15 + (self.max_level_sum - 15) / 4
            
#         # Healing and death rewards
#         hp = ram_map.hp(self.game)
#         hp_delta = hp - self.last_hp
#         party_size_constant = party_size == self.last_party_size
#         if hp_delta > 0.5 and party_size_constant and not self.is_dead:
#             self.total_healing += hp_delta
#         if hp <= 0 and self.last_hp > 0:
#             self.death_count += 1
#             self.is_dead = True
#         elif hp > 0.01:  # TODO: Check if this matters
#             self.is_dead = False
#         self.last_hp = hp
#         self.last_party_size = party_size
#         death_reward = 0 # -0.08 * self.death_count  # -0.05
#         healing_reward = self.total_healing
        
#         # HM reward
#         hm_count = ram_map.get_hm_count(self.game)
#         if hm_count >= 1 and self.hm_count == 0:
#             self.hm_count = 1
#         # hm_reward = hm_count * 10
        

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
#                     self.cut_coords[coords] = 5 # from 14
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

#         if ram_map.used_cut(self.game) == 61:
#             ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write resets tile check
#             self.used_cut += 1

#         # Misc
#         badges = ram_map.badges(self.game)
#         self.update_pokedex()
#         self.update_moves_obtained()
        
#         silph = ram_map.silph_co(self.game)
#         rock_tunnel = ram_map.rock_tunnel(self.game)
#         ssanne = ram_map.ssanne(self.game)
#         mtmoon = ram_map.mtmoon(self.game)
#         routes = ram_map.routes(self.game)
#         misc = ram_map.misc(self.game)
#         snorlax = ram_map.snorlax(self.game)
#         hmtm = ram_map.hmtm(self.game)
#         bill = ram_map.bill(self.game)
#         oak = ram_map.oak(self.game)
#         towns = ram_map.towns(self.game)
#         lab = ram_map.lab(self.game)
#         mansion = ram_map.mansion(self.game)
#         safari = ram_map.safari(self.game)
#         dojo = ram_map.dojo(self.game)
#         hideout = ram_map.hideout(self.game)
#         tower = ram_map.poke_tower(self.game)
#         gym1 = ram_map.gym1(self.game)
#         gym2 = ram_map.gym2(self.game)
#         gym3 = ram_map.gym3(self.game)
#         gym4 = ram_map.gym4(self.game)
#         gym5 = ram_map.gym5(self.game)
#         gym6 = ram_map.gym6(self.game)
#         gym7 = ram_map.gym7(self.game)
#         gym8 = ram_map.gym8(self.game)
#         rival = ram_map.rival(self.game)

#         cut_rew = self.cut * 10    
#         event_reward = sum([silph, rock_tunnel, ssanne, mtmoon, routes, misc, snorlax, hmtm, bill, oak, towns, lab, mansion, safari, dojo, hideout, tower, gym1, gym2, gym3, gym4, gym5, gym6, gym7, gym8, rival])
#         seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
#         caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
#         moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)
#         used_cut_rew = self.used_cut * 0.1
#         cut_coords = sum(self.cut_coords.values()) * 1.0
#         cut_tiles = len(self.cut_tiles) * 1.0
#         start_menu = self.seen_start_menu * 0.01
#         pokemon_menu = self.seen_pokemon_menu * 0.1
#         stats_menu = self.seen_stats_menu * 0.1
#         bag_menu = self.seen_bag_menu * 0.1
#         that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu ) / 2
    
#         reward = self.reward_scale * (
#             + level_reward
#             + healing_reward
#             + exploration_reward 
#             + cut_rew
#             + event_reward     
#             + seen_pokemon_reward
#             + caught_pokemon_reward
#             + moves_obtained_reward
#             + used_cut_rew
#             + cut_coords 
#             + cut_tiles
#             + that_guy
#         )

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
#         done = self.time >= self.max_episode_steps
#         if self.save_video and done:
#             self.full_frame_writer.close()
#         if done:
#             poke = self.game.get_memory_value(0xD16B)
#             level = self.game.get_memory_value(0xD18C)
#             info = {
#                 "Events": {
#                     "silph": silph,
#                     "rock_tunnel": rock_tunnel,
#                     "ssanne": ssanne,
#                     "mtmoon": mtmoon,
#                     "routes": routes,
#                     "misc": misc,
#                     "snorlax": snorlax,
#                     "hmtm": hmtm,
#                     "bill": bill,
#                     "oak": oak,
#                     "towns": towns,
#                     "lab": lab,
#                     "mansion": mansion,
#                     "safari": safari,
#                     "dojo": dojo,
#                     "hideout": hideout,
#                     "tower": tower,
#                     "gym1": gym1,
#                     "gym2": gym2,
#                     "gym3": gym3,
#                     "gym4": gym4,
#                     # "gym5": gym5,
#                     # "gym6": gym6,
#                     # "gym7": gym7,
#                     # "gym8": gym8,
#                     "rival": rival,
#                 },
#                 "BET": {
#                     "Reward_Delta": reward,
#                     "Seen_Poke": seen_pokemon_reward,
#                     "Caught_Poke": caught_pokemon_reward,
#                     "Moves_Obtain": moves_obtained_reward,
#                     # "Get_HM": hm_reward,
#                     "Level": level_reward,
#                     "Death": death_reward,
#                     "Healing": healing_reward,
#                     "Exploration": exploration_reward,
#                     "Taught_Cut": cut_rew,
#                     "Menuing": that_guy,
#                     "Used_Cut": used_cut_rew,
#                     "Cut_Coords": cut_coords,
#                     "Cut_Tiles": cut_tiles,
#                 },
#                 "hm_count": hm_count,
#                 "cut_taught": self.cut,
#                 "badge_1": float(badges >= 1),
#                 "badge_2": float(badges >= 2),
#                 "badge_3": float(badges >= 3),
#                 "badge_4": float(badges >= 4),
#                 "badge_5": float(badges >= 5),
#                 "badge_6": float(badges >= 6),
#                 "badge_7": float(badges >= 7),
#                 "badge_8": float(badges >= 8),
#                 "badges": float(badges),
#                 "maps_explored": np.sum(self.seen_maps),
#                 "party_size": party_size,
#                 "moves_obtained": sum(self.moves_obtained),
#                 "deaths": self.death_count,
#                 'cut_coords': cut_coords,
#                 'cut_tiles': cut_tiles,
#                 'bag_menu': bag_menu,
#                 'stats_menu': stats_menu,
#                 'pokemon_menu': pokemon_menu,
#                 'start_menu': start_menu,
#                 'used_cut': self.used_cut,
#                 'gym_three': self.gymthree,
#                 'gym_four': self.gymfour,
#                 'pokemon_exploration_map': self.counts_map,
#             }
        
#         return self.render(), reward, done, done, info