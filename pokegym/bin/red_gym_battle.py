# Assuming these constants are defined in red_env_constants
from math import floor

import numpy as np
from red_env_constants import *
from ram_reader.red_memory_battle import *
from ram_reader.red_memory_menus import BATTLE_MENU_STATES, RedRamMenuValues



class BattleTurn:
    def __init__(self):
        self.menus_visited = {}  # Reward for visiting a menu for the first time each battle turn, dec static reward

class BattleMemory:
    def __init__(self):
        # Start of turn values
        self.pre_player_pokemon = 0
        self.pre_enemy_pokemon = 0
        self.pre_player_modifiers_sum = 0
        self.pre_enemy_modifiers_sum = 0
        self.pre_player_hp = 0
        self.pre_enemy_hp = 0
        self.pre_player_status = 0
        self.pre_enemy_status = 0
        self.pre_type_hint = 0
        self.battle_turn = BattleTurn()


class RedGymBattle:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymBattle ****')

        self.env = env
        self.wild_pokemon_killed = 0
        self.trainer_pokemon_killed = 0
        self.gym_pokemon_killed = 0
        self.current_battle_action_cnt = 0
        self.total_battle_action_cnt = 0
        self.total_battle_turns = 0 
        self.total_battles = 0
        self.battle_has_started = False
        self.battle_won = False
        self.total_party_hp_lost = 0
        self.total_enemy_hp_lost = 0
        self.last_party_head_hp = 0
        self.last_enemy_head_hp = 0
        self.battle_memory = None  # Don't use the space unless in battle


    LEVEL_DELTA_DECAY = {
        0 : 0.9,
        1 : 0.75,
        2 : 0.55,
        3 : 0.35,
        4 : 0.15,
    }


    def _clear_battle_stats(self):
            self.last_party_head_hp = 0
            self.last_enemy_head_hp = 0
            self.current_battle_action_cnt = 0
            self.battle_has_started = False
            self.battle_memory = None

    def _calc_battle_type_stats(self):
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == BattleTypes.WILD_BATTLE:
            self.wild_pokemon_killed += 1
        elif battle_type == BattleTypes.TRAINER_BATTLE:
            self.trainer_pokemon_killed += 1
        # TODO: Need to ID Gym Battle
        #elif battle_type == int(BattleTypes.GYM_BATTLE):
        #    self.gym_pokemon_killed += 1
        elif battle_type == BattleTypes.DIED:
            pass
        else:
            print(f'Unknown battle type: {battle_type}')

    def _inc_move_count(self):
        if not self.env.game.battle.battle_done:
            self.current_battle_action_cnt += 1
            self.total_battle_action_cnt += 1

    def _inc_battle_counter(self):
        if not self.battle_has_started:
            self.total_battles += 1
            self.battle_has_started = True

    def _inc_hp_lost_vs_taken(self):
        if not self.env.game.battle.in_battle:
            return
        
        _, party_head_hp = self.env.game.battle.get_player_party_head_hp()
        _, enemy_head_hp = self.env.game.battle.get_enemy_party_head_hp()

        if self.last_party_head_hp == 0:
            self.last_party_head_hp = party_head_hp
        if self.last_enemy_head_hp == 0:
            self.last_enemy_head_hp = enemy_head_hp

        if party_head_hp < self.last_party_head_hp:
            self.total_party_hp_lost += (self.last_party_head_hp - party_head_hp)
            self.last_party_head_hp = party_head_hp

        if enemy_head_hp < self.last_enemy_head_hp:
            self.total_enemy_hp_lost += (self.last_enemy_head_hp - enemy_head_hp)
            self.last_enemy_head_hp = enemy_head_hp

    def _calc_level_decay(self, avg_enemy_level, avg_player_lvl):
        POKEMON_BATTLE_LEVEL_FLOOR = 1
        level_delta = avg_player_lvl - avg_enemy_level
        if level_delta < POKEMON_BATTLE_LEVEL_FLOOR:
            return 0

        return min(level_delta, len(self.LEVEL_DELTA_DECAY))
        
    def _calc_avg_pokemon_level(self, pokemon):        
        avg_level, size = 0, 0
        for i, level in enumerate(pokemon):
            if level == 0:
                break

            avg_level += level
            size = i
        
        return avg_level / (size + 1)
    
    def _get_battle_turn_stats(self):
        player = self.env.game.battle.get_player_party_head_modifiers()
        enemy = self.env.game.battle.get_enemy_party_head_modifiers()

        player_hp_total, player_hp_avail = self.env.game.battle.get_player_party_head_hp()
        enemy_hp_total, enemy_hp_avail = self.env.game.battle.get_enemy_party_head_hp()
        return {
            'player_pokemon' : self.env.game.battle.get_player_head_index(),
            'enemy_pokemon' : self.env.game.battle.get_enemy_party_head_pokemon(),
            'player_effects_sum' : sum(player[1:]),
            'enemy_effects_sum' : sum(enemy[3:]),
            'player_hp_total' : player_hp_total,
            'player_hp_avail' : player_hp_avail,
            'enemy_hp_total' : enemy_hp_total,
            'enemy_hp_avail' : enemy_hp_avail,
            'player_status' : self.env.game.battle.get_player_party_head_status(),
            'enemy_status' : self.env.game.battle.get_enemy_party_head_status(),
            'type_hint' : self.env.game.battle.get_battle_type_hint(),
        }

    def _update_pre_battle_memory(self):
        turn_stats = self._get_battle_turn_stats()
        self.battle_memory.pre_player_pokemon = turn_stats['player_pokemon']
        self.battle_memory.pre_enemy_pokemon = turn_stats['enemy_pokemon']
        self.battle_memory.pre_player_modifiers_sum = turn_stats['player_effects_sum']
        self.battle_memory.pre_enemy_modifiers_sum = turn_stats['enemy_effects_sum']
        self.battle_memory.pre_player_hp = turn_stats['player_hp_avail']
        self.battle_memory.pre_enemy_hp = turn_stats['enemy_hp_avail']
        self.battle_memory.pre_player_status = turn_stats['player_status']
        self.battle_memory.pre_enemy_status = turn_stats['enemy_status']
        self.battle_memory.pre_type_hint = turn_stats['type_hint']

    def _update_menu_selected(self):
        if ((self.env.gameboy.a_button_selected() and self.env.game.game_state == self.env.game.GameState.BATTLE_TEXT) or
             self.env.game.game_state == self.env.game.GameState.BATTLE_ANIMATION):
            return
        
        selection_count = self.battle_memory.battle_turn.menus_visited.get(self.env.game.game_state.value, 0)
        self.battle_memory.battle_turn.menus_visited[self.env.game.game_state.value] = selection_count + 1
    
    def get_battle_decay(self):
        avg_enemy_level = self._calc_avg_pokemon_level(self.env.game.battle.get_enemy_lineup_levels())
        avg_player_lvl = self._calc_avg_pokemon_level(self.env.game.player.get_player_lineup_levels())
        decay = self._calc_level_decay(avg_enemy_level, avg_player_lvl)

        return self.LEVEL_DELTA_DECAY.get(decay, 0.001)
    
    def save_pre_action_battle(self):
        if not self.env.game.battle.in_battle:
            return
        
        # Handles starting mid battle on loads
        if self.battle_memory == None:
            self.battle_memory = BattleMemory()
        
        self._update_pre_battle_memory()

    def save_post_action_battle(self):
        if not self.env.game.battle.in_battle:
            self._clear_battle_stats()
            return
        
        if self.battle_memory == None:
            self.battle_memory = BattleMemory()
        
        # IN BATTLE: Falls through
            
        if self.env.game.battle.new_turn:
            self.battle_memory.battle_turn = BattleTurn()

        self.battle_won = self.env.game.battle.win_battle()  # allows single occurrence won flag per battle, when enemy mon's hp all -> 0
        if self.battle_won:
            self.env.game.battle.battle_done = True   # TODO: The API handles setting this, back this out

        self._inc_move_count()
        self._inc_battle_counter()
        self._inc_hp_lost_vs_taken()
        self._update_menu_selected()

        # cal this way instead of w/ inc_move_count() b/c of long post battle text, which can count as still in battle
        if not self.battle_won:
            return
        
        # Won Battle falls though, to update total battle's stat's. This calc can only happen once per battle b/c of battle_won flag's design
        self._calc_battle_type_stats()
        self.total_battle_turns += self.env.game.battle.turns_in_current_battle

    def get_battle_win_reward(self):
        if not self.env.game.battle.in_battle:
            return 0
        elif not self.battle_won:
            return 0.1
        
        # Won Battle falls though
        BATTLE_MOVE_CEILING = 350
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == BattleTypes.WILD_BATTLE:
            multiplier = max(.1, -0.1 * self.env.reset_count + 1)  # 1 for resets less than 5, 1 to .1 until 10 resets, and 0.1 after 10 resets
            return (max(0, (BATTLE_MOVE_CEILING - self.current_battle_action_cnt) * self.get_battle_decay())) * multiplier
        elif battle_type == BattleTypes.TRAINER_BATTLE:
            multiplier = max(.20, -0.005 * self.env.reset_count + 1)  # 1 for resets less than 5, 1 to .1 until 10 resets, and 0.1 after 10 resets
            pokemon_fought = self.env.game.battle.get_enemy_party_count()
            return (500 * pokemon_fought + (max(0, (BATTLE_MOVE_CEILING * pokemon_fought) - self.current_battle_action_cnt))) * multiplier
        # TODO: Need to ID Gym Battle
        #elif battle_type == BattleTypes.GYM_BATTLE):
        #    return 600
        elif battle_type == BattleTypes.DIED:
            return 0
        
        self.env.support.save_and_print_info(False, True, True)
        assert(False), "Unknown battle type"
    
    def _pp_select_reward(self):
        pp_1, pp_2, pp_3, pp_4 = self.env.game.battle.get_player_party_head_pp()
        match self.env.game.game_state:
            case RedRamMenuValues.BATTLE_MOVE_1:
                return int(pp_1 == 0)
            case RedRamMenuValues.BATTLE_MOVE_2:
                return int(pp_2 == 0)
            case RedRamMenuValues.BATTLE_MOVE_3:
                return int(pp_3 == 0)
            case RedRamMenuValues.BATTLE_MOVE_4:
                return int(pp_4 == 0)

        return 0
    
    def _menu_selection_punish(self):
        selection_count = self.battle_memory.battle_turn.menus_visited.get(self.env.game.game_state.value, 0)
        if selection_count == 1:
            return 0  # Don't reward new menu discovery or AI will farm menu hovering
        
        # TODO: Run in trainer battle not working, need to fix, no neg
        return max(-0.001 * pow(selection_count, 2), -0.15)
            
    def _get_battle_action_reward(self):
        if not self.env.gameboy.a_button_selected():
            return 0
        
        action_reward = 0
        action_reward += self._pp_select_reward() * -0.1

        return action_reward
    
    def _get_battle_hint_reward(self, turn_stats):
        player_pokemon_switch = (turn_stats['player_pokemon'] != self.battle_memory.pre_player_pokemon)
        enemy_pokemon_switch = (turn_stats['enemy_pokemon'] != self.battle_memory.pre_enemy_pokemon)
        type_hint_delta = turn_stats['type_hint'] - self.battle_memory.pre_type_hint  # pos good, neg bad

        if player_pokemon_switch or enemy_pokemon_switch:
            if type_hint_delta > 0:
                return 4
            elif type_hint_delta < 0:  # Discourage bad switches and switch cycling for point farming
                return -0.1
            
        return 0
        
    def _get_battle_stats_reward(self, turn_stats):
        # Can't have a stat inc/dec reward on the 1st turn b/c nothings happened yet
        if self.total_battle_turns == 0:
            return 0
        
        player_modifiers_delta = turn_stats['player_effects_sum'] - self.battle_memory.pre_player_modifiers_sum  # pos good, neg bad
        enemy_modifiers_delta = turn_stats['enemy_effects_sum'] - self.battle_memory.pre_enemy_modifiers_sum  # pos bad, neg good
        player_hp_delta = turn_stats['player_hp_avail'] - self.battle_memory.pre_player_hp  # pos good, neg bad
        enemy_hp_delta = turn_stats['enemy_hp_avail'] - self.battle_memory.pre_enemy_hp # pos bad, neg good

        reward = 0

        if player_modifiers_delta > 0:
            reward += 3
        if enemy_modifiers_delta < 0:
            reward += 3
        if player_hp_delta > 0:
            reward += 6 * max((player_hp_delta / turn_stats['player_hp_total']), 0.375)
        if enemy_hp_delta < 0:
            reward += 6 * max((abs(enemy_hp_delta) / turn_stats['enemy_hp_total']), 0.375) * turn_stats['type_hint']
        if turn_stats['player_status'] == 0 and self.battle_memory.pre_player_status != 0:
            reward += 5
        if turn_stats['enemy_status'] != 0 and self.battle_memory.pre_enemy_status == 0:
            reward += 5

        return reward
    
    def get_battle_action_reward(self):
        if not self.env.game.battle.in_battle:
            return 0

        turn_stats = self._get_battle_turn_stats()

        selection_reward = self._menu_selection_punish()
        #(f'Menu Selection Reward: {selection_reward}')
        #if reward < 0:
        #    return reward  # No decay for bad menu selections

        #reward += self._get_battle_action_reward()
        #print(f'Action Reward: {self._get_battle_action_reward()}')
        #hit_reward = self._get_battle_hint_reward(turn_stats)
        #print(f'Hint Reward: {hit_reward}')

        #print(f'hint: {turn_stats["type_hint"]}')

        stats_reward = self._get_battle_stats_reward(turn_stats)
        #print(f'Stats Reward: {stats_reward}')

        return selection_reward + (stats_reward * self.get_battle_decay())

    def get_avg_battle_action_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_action_cnt / self.total_battles

    def get_avg_battle_turn_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_turns / self.total_battles

    def get_kill_to_death(self):
        died = self.env.player.died + 1

        return (self.wild_pokemon_killed + self.trainer_pokemon_killed + self.gym_pokemon_killed) / died
    
    def get_damage_done_vs_taken(self):
        if self.total_party_hp_lost == 0:
            return 0
        return self.total_enemy_hp_lost / self.total_party_hp_lost
    
    def obs_in_battle(self):
        return np.array([self.env.game.battle.in_battle], dtype=np.uint8)
    
    def obs_battle_type(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((4, ), dtype=np.uint8)
        
        battle_type = np.array(self.env.game.battle.get_battle_type(), dtype=np.uint8)
        binary_status = np.unpackbits(battle_type)[4:]

        return binary_status.astype(np.uint8)

    def obs_enemies_left(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.float32)

        return np.array([self.env.game.battle.get_battles_pokemon_left()], dtype=np.float32)

    def obs_player_head_index(self):
        if not self.env.game.battle.in_battle:  # TODO: What if mon fainted? Should show next avail mon in party
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_player_head_index()], dtype=np.uint8)
    
    def obs_player_head_pokemon(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_player_head_pokemon()], dtype=np.uint8)

    def obs_player_modifiers(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((6, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_player_party_head_modifiers(), dtype=np.float32))

    def obs_enemy_head(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_enemy_party_head_pokemon()], dtype=np.uint8)

    def obs_enemy_level(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array([self.env.game.battle.get_enemy_party_head_level()], dtype=np.float32) * 2)

    def obs_enemy_hp(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_enemy_party_head_hp(), dtype=np.float32), False, 705.0)

    def obs_enemy_types(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.uint8)

        return np.array(self.env.game.battle.get_enemy_party_head_types(), dtype=np.uint8)

    def obs_enemy_modifiers(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((6, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_enemy_party_head_modifiers(), dtype=np.float32))

    def obs_enemy_status(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((5, ), dtype=np.uint8)

        # https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_I)#Status_conditions
        # First 3 bits unused
        status = self.env.game.battle.get_enemy_party_head_status()
        status_array = np.array(status, dtype=np.uint8)
        binary_status = np.unpackbits(status_array)[3:8]
        return binary_status.astype(np.uint8)
    
    def obs_battle_moves_selected(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.uint8)

        return np.array(self.env.game.battle.get_battle_turn_moves(), dtype=np.uint8)

    def obs_type_hint(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((4, ), dtype=np.uint8)
                
        hint = np.array(self.env.game.battle.get_battle_type_hint(), dtype=np.uint8)
        binary_status = np.unpackbits(hint)[4:]

        return binary_status.astype(np.uint8)
