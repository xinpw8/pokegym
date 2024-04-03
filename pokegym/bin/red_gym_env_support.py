import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from red_gym_map import RedGymMap
from red_env_constants import *

from ram_reader.red_ram_debug import *


def calc_byte_float_norm(size):
    bytes_norm = []
    for i in range(BYTE_SIZE):
        bytes_norm.append(math.floor((i / size) * 10 ** 4) / 10 ** 4) # normalize lookup 0-255 to 4-digit float

    return bytes_norm


class RedGymGlobalMemory:
    def __init__(self):
        self.byte_to_float_norm = calc_byte_float_norm(4096.0)


class RedGymEnvSupport:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymEnvSupport ****')

        self.env = env
        random.seed(self.env.thread_id)  # + self.env.reset_count

    def choose_random_game_load(self):

        # pick a random starter string bulbasaur, charmander, squirtle
        starters = [
            #'checkpoints_battles/bulbasaur' + '/pokemon_ai_' + str(random.randint(1, 24)),
            #'checkpoints_battles/charmander' + '/pokemon_ai_' + str(random.randint(32, 58)),
            #'checkpoints_battles/squirtle' + '/pokemon_ai_' + str(random.randint(58, 84)),
            #'checkpoints_battles/mt_moon' + '/pokemon_ai_' + str(random.randint(0, 17)),
            "checkpoints_pallet/pokemon_ai_" + str(random.randint(1, 22)),
            #"checkpoints_bill/pokemon_ai_" + str(random.randint(0, 134)),
            ]
        save_file = random.choice(starters)

        return save_file


    def save_screenshot(self, image=None):
        x_pos, y_pos, map_n = self.env.map.get_current_location()

        if image is None:
            image = self.env.screen.render(reduce_res=False)

        ss_dir = self.env.s_path / Path(f'screenshots/{self.env.instance_id}')
        ss_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{self.env.step_count}_{map_n}_{x_pos}_{y_pos}.jpeg'),
            image)

    def check_if_done(self):
        return self.env.step_count >= self.env.max_steps

    def save_and_print_info(self, done, save_debug = False, print_locations = False):
        if self.env.print_rewards:
            prog_string = self._construct_progress_string()
            if save_debug:
                game_debug = get_debug_str(self.env.game)
                if print_locations:
                    while len(self.env.map.location_history):
                        game_debug += self.env.map.location_history.popleft()

                self.save_debug_string(game_debug)
            elif self.env.debug:
                # os.system('clear')
                game_debug = get_debug_str(self.env.game)
                print(f'\r\n\naction: {WindowEvent(self.env.gameboy.action_history[-1]).__str__()}\n'
                      f'Move Allowed(REAL): {self.env.gameboy.move_accepted}\n'
                      f'{self.env.map.location_history[-1]}\n\n'
                      f'{game_debug}\n\n'
                      f'{prog_string}', end='', flush=True)
            else:
                print(f'\r{prog_string}', end='', flush=True)

        if self.env.print_rewards and done:
            self._print_final_rewards()

        if self.env.save_video and done:
            self._close_video_writers()

        if done:
            self._save_run_data()

    def save_debug_string(self, output_str):
        debug_path = self.env.s_path / 'debug'
        debug_path.mkdir(exist_ok=True)

        # Construct the full file path
        file_path = debug_path / f'thread_{self.env.thread_id}_step_{self.env.step_count}.txt'

        # Write the output string to the file
        with open(file_path, 'w') as file:
            file.write(output_str)

    def normalize_np_array(self, np_array, lookup=True, size=256.0):
        if lookup:
            np_array = np.vectorize(lambda x: self.env.memory.byte_to_float_norm[int(x)])(np_array)
        else:
            np_array = np.vectorize(lambda x: int(x) / size)(np_array)

        return np_array

    def _save_current_frame(self):
        plt.imsave(
            self.env.s_path / Path(f'curframe_{self.env.instance_id}.jpeg'),
            self.env.screen.render(reduce_res=False))

    def _close_video_writers(self):
        self.env.full_frame_writer.close()
        self.env.model_frame_writer.close()

    def _construct_progress_string(self):
        prog_string = f'step: {self.env.step_count:6d}'
        for key, val in self.env.agent_stats[-1].items():
            prog_string += f' {key}: {val:5.3f}'
        prog_string += f' decay: {self.env.battle.get_battle_decay():5.3f}'
        return prog_string

    def _print_final_rewards(self):
        print('', flush=True)

        if self.env.save_final_state:
            fs_path = self.env.s_path / 'final_states'
            fs_path.mkdir(exist_ok=True)
            plt.imsave(
                fs_path / Path(f'frame_r{self.env.total_reward:.4f}_{self.env.reset_count}_small.jpeg'),
                self.env.screen.render())
            plt.imsave(
                fs_path / Path(f'frame_r{self.env.total_reward:.4f}_{self.env.reset_count}_full.jpeg'),
                self.env.screen.render(reduce_res=False))

    def _save_run_data(self):
        stats_path = self.env.s_path / 'agent_stats'
        stats_path.mkdir(exist_ok=True)
        pd.DataFrame(self.env.agent_stats).to_csv(
            stats_path / Path(f'agent_stats_{self.env.instance_id}.csv.gz'), compression='gzip', mode='a')


