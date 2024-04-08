import numpy as np
from red_env_constants import *
from ram_reader.red_memory_items import *
from ram_reader.red_memory_battle import BattleTypes


class RedGymPlayer:
    def __init__(self, env):
        self.env = env
        if env.debug:
            print('**** RedGymPlayer ****')
        
        self.current_badges = 0
        self.bag_items = {}
        self.bank_items = {}
        self.money = 0
        self.died = 0
        self.dead = False


    def _inc_died(self):
        if self.env.game.battle.get_battle_type() == BattleTypes.DIED or self.env.game.player.is_player_dead():
            if not self.dead:
                self.died += 1
                self.dead = True
                return
        else:
            self.dead = False

    def _lookup_player_items(self, item_ids, item_counts):
        items = {}
        for i in range(len(item_ids)):
            if item_counts[i] == 0:
                break

            items[item_ids[i]] = item_counts[i]

        return items
    
    def _get_player_money(self):
        return self.env.game.player.get_player_money()
    
    def save_post_action_player(self):
        self._inc_died()
    
    def get_item_reward(self):
        bag_item_ids = self.env.game.items.get_bag_item_ids()
        bag_item_counts = self.env.game.items.get_bag_item_quantities()
        pc_item_counts = self.env.game.items.get_pc_item_quantities()

        # Prevent reward by shuffling items around in bank & bag
        cur_total_items = sum(bag_item_counts) + sum(pc_item_counts)
        prev_total_items = sum(self.bag_items.values()) + sum(self.bank_items.values())
        if cur_total_items == prev_total_items:
            return 0

        # Reward for gaining items, ignore using/selling items here
        item_norm, item_delta, item_key = 0, 0, 0
        for i in range(len(bag_item_ids)):
            item_key = bag_item_ids[i]
            
            item_delta = np.int32(bag_item_counts[i]) - np.int32(self.bag_items.get(item_key, 0))
            item_norm = abs((item_delta * ITEM_COSTS.get(item_key, 0)) / 100)

            if item_norm != 0:
                break

        # Don't reward selling items
        cur_money = self._get_player_money()
        money_delta = cur_money - self.money
        if money_delta > 0:
            return 0
            
        # item_norm could be pos bought item or neg sold item but both are good rewards, using should always win over buy/sell loops b/c selling halves money value
        return 25 * item_norm
    

    def get_badge_reward(self):
        badges = self.env.game.player.get_badges()
        if badges > self.current_badges:
            self.current_badges = badges
            return 1000
            
        return 0
    
    def save_pre_action_player(self):
        self.bag_items = self._lookup_player_items(self.env.game.items.get_bag_item_ids(), self.env.game.items.get_bag_item_quantities())
        self.bank_items = self._lookup_player_items(self.env.game.items.get_pc_item_ids(), self.env.game.items.get_pc_item_quantities())
        self.money = self._get_player_money()

    def obs_player_pokemon(self):
        pokemon_array = np.array(self.env.game.player.get_player_lineup_pokemon(), dtype=np.uint8)
        return np.pad(pokemon_array, ((0, 6 - len(pokemon_array))), mode='constant')
    
    def obs_player_levels(self):
        levels_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_levels()), dtype=np.float32)
        padded_levels = np.pad(levels_array, (0, 6 - len(levels_array)), mode='constant')
        return padded_levels
    
    def obs_player_types(self):
        types_array = np.array(self.env.game.player.get_player_lineup_types(), dtype=np.uint8).flatten()
        padded_types = np.pad(types_array, (0, 12 - len(types_array)), constant_values=0)
        return padded_types
    
    def obs_player_health(self):
        health_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_health(), False, 705), dtype=np.float32).flatten()
        padded_health = np.pad(health_array, (0, 12 - len(health_array)), mode='constant')
        return padded_health
    
    def obs_player_moves(self):
        moves_array = np.array(self.env.game.player.get_player_lineup_moves(), dtype=np.uint8).flatten()
        padded_moves = np.pad(moves_array, (0, 24 - len(moves_array)), constant_values=0)
        return padded_moves
    
    def obs_player_xp(self):
        xp_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_xp(), False, 250000), dtype=np.float32)
        padded_xp = np.pad(xp_array, (0, 6 - len(xp_array)), mode='constant')
        return padded_xp
    
    def obs_player_pp(self):
        pp_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_pp()), dtype=np.float32).flatten()
        padded_pp = np.pad(pp_array, (0, 24 - len(pp_array)), mode='constant')
        return padded_pp
    
    def obs_player_stats(self):
        stats_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_stats()), dtype=np.float32).flatten()
        padded_stats = np.pad(stats_array, (0, 24 - len(stats_array)), mode='constant')
        return padded_stats
    
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
    
    def obs_bag_ids(self):
        bag_item_ids = self.env.game.items.get_bag_item_ids()
        padded_ids = np.pad(bag_item_ids, (0, 20 - len(bag_item_ids)), constant_values=0)
        return np.array(padded_ids, dtype=np.uint8)
    
    def obs_bag_quantities(self):
        return self.env.support.normalize_np_array(self.env.game.items.get_bag_item_quantities())
    
    def obs_total_money(self):
        return self.env.support.normalize_np_array(np.array([self.money], dtype=np.float32), False, 200000)