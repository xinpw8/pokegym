import numpy as np
from collections import deque

# Assuming these constants are defined in red_env_constants
from red_env_constants import *
from red_gym_obs_tester import RedGymObsTester


class RedGymMap:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymMap ****')

        self.env = env
        self.x_pos_org, self.y_pos_org, self.n_map_org = None, None, None
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = 0  # TODO: Inc/dec to 6
        self.moved_location = False  # indicates if the player moved 1 or more spot
        self.discovered_location = False # indicates if the player is in previously unvisited location
        self.location_history = deque()
        self.steps_discovered = 0
        self.collisions = 0

        self.screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.float32)
        self.visited = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.walkable = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, BITS_PER_BYTE), dtype=np.float32)  # x,y,map stacked, 7 bits as all val's are < 128

        self.tester = RedGymObsTester(self)


    def _update_tile_obs(self):
        # The screen tiles don't have sprites/npc's with them
        self.screen[0:7, 0:7] = self.env.game.map.get_centered_7x7_tiles()
        self.walkable[0:7, 0:7] = self.env.game.map.get_centered_step_count_7x7_screen()

    def _update_visited_obs(self, x_pos_new, y_pos_new, n_map_new):
        callback = lambda x, y, pos: self._update_matrix_visited(x, y, pos)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)

        # DO NOT set cur pos as visited on the obs until the next turn, it REALLY helps the AI
        # ie.. self.visited[3][3] = 0 (this is intentional)

    def _update_npc_and_norm_obs(self, x_pos_new, y_pos_new, n_map_new):
        sprites = self.env.game.map.get_npc_location_dict(n_map_new)

        callback = lambda x, y, pos: self._update_matrix_npc_and_normalize(x, y, pos, sprites)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)


    def _update_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        try:
            x_pos_binary = format(x_pos_new, f'0{BITS_PER_BYTE}b')
            y_pos_binary = format(y_pos_new, f'0{BITS_PER_BYTE}b')
            m_pos_binary = format(n_map_new, f'0{BITS_PER_BYTE}b')

            self.coordinates = np.roll(self.coordinates, 1, axis=0)
        
            # appends the x,y, pos binary form to the bottom of the screen and visited matrix's
            for i, bit in enumerate(x_pos_binary):
                self.coordinates[0][i] = bit

            for i, bit in enumerate(y_pos_binary):
                self.coordinates[1][i] = bit

            for i, bit in enumerate(m_pos_binary):
                self.coordinates[2][i] = bit
        except Exception as e:
            print(f"An error occurred: {e}")
            self.env.support.save_and_print_info(False, True, True)
            self.env.support.save_debug_string("An error occurred: {e}")
            assert(True)

    def _traverse_matrix(self, x_pos_new, y_pos_new, n_map_new, callback):
        center_x = center_y = SCREEN_VIEW_SIZE // 2

        for y in range(SCREEN_VIEW_SIZE):
            for x in range(SCREEN_VIEW_SIZE):
                center_x = center_y = SCREEN_VIEW_SIZE // 2
                x_offset = x - center_x
                y_offset = y - center_y
                current_pos = x_pos_new + x_offset, y_pos_new + y_offset, n_map_new

                callback(x, y, current_pos)


    def _update_matrix_visited(self, x, y, pos):
        if pos in self.visited_pos:
            self.visited[y][x] = 0
        else:
            self.visited[y][x] = 1


    def _update_matrix_npc_and_normalize(self, x, y, pos, sprites):
        if pos in sprites:
            self.screen[y][x] = self.env.memory.byte_to_float_norm[sprites[pos]] + 0.1
        else:
            self.screen[y][x] = self.env.memory.byte_to_float_norm[int(self.screen[y][x])]

    def _clear_map_obs(self):
        self.screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.float32)
        self.visited = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.walkable = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, BITS_PER_BYTE), dtype=np.float32)

    def save_post_action_pos(self):
        x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()
        self.moved_location = not (self.x_pos_org == x_pos_new and
                                   self.y_pos_org == y_pos_new and
                                   self.n_map_org == n_map_new)

        if self.moved_location:
            # Bug check: AI is only allowed to move 0 or 1 spots per turn, new maps change x,y ref pos so don't count.
            # When the game goes to a new map, it changes m first, then y,x will update on the next turn, still some corner cases like fly, blackout, bike
            if self.new_map:
                self.x_pos_org, self.y_pos_org, self.n_map_org = x_pos_new, y_pos_new, n_map_new
                self.new_map -= 1
            elif n_map_new == self.n_map_org:
                if not (abs(self.x_pos_org - x_pos_new) + abs(self.y_pos_org - y_pos_new) <= 1):
                    self.update_map_stats()

                    debug_str = ""
                    #while len(self.location_history):
                    #    debug_str += self.location_history.popleft()
                    # self.env.support.save_debug_string(debug_str)
                    # assert False
            else:
                self.new_map = 6

            if (x_pos_new, y_pos_new, n_map_new) in self.visited_pos:
                self.discovered_location = True


    def save_pre_action_pos(self):
        self.x_pos_org, self.y_pos_org, self.n_map_org = self.env.game.map.get_current_location()
        self.discovered_location = False

        if len(self.visited_pos_order) > MAX_STEP_MEMORY:
            del_key = self.visited_pos_order.popleft()
            del self.visited_pos[del_key]

        current_pos = (self.x_pos_org, self.y_pos_org, self.n_map_org)
        if current_pos not in self.visited_pos:
            self.visited_pos[current_pos] = self.env.step_count
            self.visited_pos_order.append(current_pos)


    def update_map_stats(self):
        new_x_pos, new_y_pos, new_map_n = self.env.game.map.get_current_location()

        debug_str = f"Moved: {self.moved_location} \n"
        if self.new_map:
            debug_str = f"\nNew Map!\n"
        debug_str += f"Start location: {self.x_pos_org, self.y_pos_org, self.n_map_org} \n"
        debug_str += f"New location: {new_x_pos, new_y_pos, new_map_n} \n"
        debug_str += f"\n"
        debug_str += f"{self.screen}"
        debug_str += f"\n"
        debug_str += f"{self.visited}"
        debug_str += f"\n"
        debug_str += f"{self.walkable}"

        if len(self.location_history) > 10:
            self.location_history.popleft()
        self.location_history.append(debug_str)

    def get_exploration_reward(self):
        x_pos, y_pos, map_n = self.env.game.map.get_current_location()
        if not self.moved_location:
            if (not (self.env.gameboy.action_history[0] == 5 or self.env.gameboy.action_history[0] == 6) and 
                self.env.game.get_game_state() ==  self.env.game.GameState.EXPLORING and self.new_map == False):
                self.collisions += 1
        
            return 0
        elif (x_pos, y_pos, map_n) in self.visited_pos:
            return 0.01
        else:
            self.steps_discovered += 1
            return 1        

    def update_map_obs(self):
        if self.env.game.battle.in_battle:
            self._clear_map_obs()  # Don't show the map while in battle b/c human can't see map when in battle
        else:
            x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()

            # Order matters here
            self._update_tile_obs()
            self._update_visited_obs(x_pos_new, y_pos_new, n_map_new)
            self._update_npc_and_norm_obs(x_pos_new, y_pos_new, n_map_new)  # Overwrites screen with npc's locations
            self._update_pos_obs(x_pos_new, y_pos_new, n_map_new)

        self.update_map_stats()

