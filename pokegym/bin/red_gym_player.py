import numpy as np
from red_env_constants import *
#from ram_reader.red_memory_map import *


class RedGymPlayer:
    def __init__(self, env):
        self.env = env
        if env.debug:
            print('**** RedGymPlayer ****')
        
        self.current_badges = 0

    def get_badge_reward(self):
        badges = self.env.game.player.get_badges()
        if badges > self.current_badges:
            self.current_badges = badges
            return 1000
            
        return 0
    

    def obs_player_pokemon(self):
        return np.array(self.env.game.player.get_player_lineup_pokemon(), dtype=np.uint8)
    
    def obs_player_levels(self):
        return np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_levels()), dtype=np.float32)
    
    def obs_player_types(self):
        return np.array(self.env.game.player.get_player_lineup_types(), dtype=np.uint8).reshape(12, )
    
    def obs_player_health(self):
        return np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_health(), False, 705), dtype=np.float32)
    
    def obs_player_moves(self):
        return np.array(self.env.game.player.get_player_lineup_moves(), dtype=np.uint8).reshape(24, )
    
    def obs_player_xp(self):
        return np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_xp(), False, 250000), dtype=np.float32)
    
    def obs_player_pp(self):
        return np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_pp()), dtype=np.float32)
    
    def obs_player_stats(self):
        return np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_stats()), dtype=np.float32)
    
    def obs_player_status(self):
        # https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_I)#Status_conditions
        # First 3 bits unused
        status = self.env.game.player.get_player_lineup_status()
        status_array = np.array(status, dtype=np.uint8)

        binary_status = np.zeros(30, dtype=np.uint8)  # 6 pokemon * 5 status bits
        for i, status in enumerate(status_array):
            binary_status[i*5:(i+1)*5] = np.unpackbits(status)[3:8]

        return binary_status

    def obs_total_badges(self):
        badges = self.env.game.player.get_badges()
        badges_array = np.array(badges, dtype=np.uint8)
        binary_badges = np.unpackbits(badges_array)[0:8]
        return binary_badges.astype(np.uint8)
    

