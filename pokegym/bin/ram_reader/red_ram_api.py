from enum import IntEnum
import numpy as np
from pyboy import PyBoy, WindowEvent

from .red_memory_battle import *
from .red_memory_env import *
from .red_memory_items import *
from .red_memory_map import *
from .red_memory_menus import *
from .red_memory_player import *



class PyBoyRAMInterface:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read_memory(self, address):
        return self.pyboy.get_memory_value(address)

    def write_memory(self, address, value):
        return self.pyboy.set_memory_value(address, value)
        

class Game:
    def __init__(self, pyboy):
        self.ram_interface = PyBoyRAMInterface(pyboy)

        self.world = World(self)
        self.battle = Battle(self)
        self.items = Items(self)
        self.map = Map(self)
        self.menus = Menus(self)
        self.player = Player(self)

        self.game_state = self.GameState.GAME_STATE_UNKNOWN

        self.process_game_states()

    class GameState(IntEnum):
        FILTERED_INPUT = 0
        IN_BATTLE = 1
        BATTLE_ANIMATION = 2
        # catch mon
        TALKING = 3
        EXPLORING = 4
        ON_PC = 5
        POKE_CENTER = 6
        MART = 7
        GYM = 8
        START_MENU = 9
        GAME_MENU = 10
        BATTLE_TEXT = 11
        FOLLOWING_NPC = 12
        GAME_STATE_UNKNOWN = 115

    
    # Order of precedence is important here, we want to check for battle first, then menus
    def process_game_states(self):
        ORDERED_GAME_STATES = [
            self.menus.get_pre_battle_menu_state,  # For menu's that could be in both battle and non-battle states
            self.battle.get_battle_state,
            self.player.is_following_npc,
            self.menus.get_menu_state,
            # TODO: Locations (mart, gym, pokecenter, etc.)
        ]

        for game_state in ORDERED_GAME_STATES:
            self.game_state = game_state()
            if self.game_state != self.GameState.GAME_STATE_UNKNOWN:
                return self.game_state
        
        self.game_state = self.GameState.EXPLORING
    
    def get_game_state(self):
        return np.array([self.game_state], dtype=np.uint8)
        

    def allow_menu_selection(self, input):
        FILTERED_INPUTS = {
            RedRamMenuValues.START_MENU_POKEDEX: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_SELF: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_SAVE: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_OPTION: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_QUIT: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.MENU_SELECT_STATS: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.BATTLE_SELECT_STATS: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.PC_OAK: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CONFIRM_STATS: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CHANGE_BOX: {WindowEvent.PRESS_BUTTON_A},
        }

        filtered_keys = FILTERED_INPUTS.get(self.game_state, None)
        if filtered_keys is None or input not in filtered_keys:
            return True

        return False


class World:
    def __init__(self, env):
        self.env = env
    
    def get_game_milestones(self):
        return np.array([self.env.ram_interface.read_memory(item) for item in GAME_MILESTONES], dtype=np.uint8)
    
    def get_playing_audio_track(self):
        return self.env.ram_interface.read_memory(AUDIO_CURRENT_TRACK_NO_DELAY)
    
    def get_pokemart_options(self):
        mart = np.zeros((POKEMART_AVAIL_SIZE,), dtype=np.uint8)
        for i in range(POKEMART_AVAIL_SIZE):
            item = self.env.ram_interface.read_memory(POKEMART_ITEMS + i)
            if item == 0xFF:
                break

            mart[i] = item

        return mart
        
    # TODO: Need item costs, 0xcf8f wItemPrices isn't valid: http://www.psypokes.com/rby/shopping.php

    def get_pokecenter_id(self):
        return self.env.ram_interface.read_memory(POKECENTER_VISITED)


class Battle:
    def __init__(self, env):
        self.env = env
        self.in_battle = False
        self.turns_in_current_battle = 1
        self.new_turn = False
        self.last_turn_count = 0
        self.battle_done = False

    def _in_battle_state(self):
        if self.env.game_state in BATTLE_MENU_STATES or self.env.game_state == self.env.GameState.BATTLE_TEXT:
            return True
        return False
    
    def _loaded_pokemon_address(self):
        party_index = self.env.ram_interface.read_memory(PLAYER_LOADED_POKEMON)
        return party_index * PARTY_OFFSET
    
    def _get_battle_menu_overwrites(self, game_state):
        # These are nasty's in the game where the reg's don't follow the same pattern as the other menu's, so we have to override them.
        # All these overwrites are based off the face we KNOW we're in battle, thus what menu's are/aren't possible.
        if game_state == RedRamMenuValues.PC_LOGOFF:
            game_state = RedRamMenuValues.MENU_YES
        elif game_state == RedRamMenuValues.MENU_SELECT_STATS:  # Corner-case, during battle the sub-menu's for switch/stats are reversed
            game_state = RedRamMenuValues.BATTLE_SELECT_SWITCH
        elif game_state == RedRamMenuValues.MENU_SELECT_SWITCH:
            game_state = RedRamMenuValues.BATTLE_SELECT_STATS

        if (game_state == RedRamMenuValues.MENU_YES or game_state == RedRamMenuValues.MENU_NO):
            text_dst_pointer = self.env.ram_interface.read_memory(TEXT_DST_POINTER)
            if text_dst_pointer == 0xF0 and game_state == RedRamMenuValues.MENU_YES:
                return RedRamMenuValues.NAME_POKEMON_YES
            elif text_dst_pointer == 0xF0 and game_state == RedRamMenuValues.MENU_NO:
                return RedRamMenuValues.NAME_POKEMON_NO
            elif text_dst_pointer == 0xED and game_state == RedRamMenuValues.MENU_YES:
                return RedRamMenuValues.SWITCH_POKEMON_YES
            elif text_dst_pointer == 0xED and game_state == RedRamMenuValues.MENU_NO:
                return RedRamMenuValues.SWITCH_POKEMON_NO

        if (game_state == RedRamMenuValues.MENU_YES or game_state == RedRamMenuValues.MENU_NO or
            game_state == RedRamMenuValues.BATTLE_SELECT_SWITCH or game_state == RedRamMenuValues.BATTLE_SELECT_STATS):
            return game_state
        
        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def _get_battle_menu_state(self, battle_type):
        cursor_location, state = self.env.menus.get_item_menu_context()
        game_state = TEXT_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamMenuValues.UNKNOWN_MENU)

        game_state = self._get_battle_menu_overwrites(game_state)
        if game_state != self.env.GameState.GAME_STATE_UNKNOWN:
            return game_state
    
        if cursor_location == RedRamMenuKeys.MENU_CLEAR or not battle_type:
            return self.env.GameState.BATTLE_ANIMATION
        
        # Very tricky to figure this one out, there is no clear ID for battle text but we can infer it from a combo of other reg's. Battle text pause get's it 50% of the time
        # but there is a delay sometimes which give false positive on ID'ing menu's. Text box id work's the rest of the time but it shares a common value with pokemon menu so
        # it alone also can't be used but the UNKNOWN_D730 reg in battle is always 0x40 when in the pokemon menu, letting us rule out pokemon menu in battle.
        if ((self.env.ram_interface.read_memory(TEXT_BOX_ID) == 0x01 and self.env.ram_interface.read_memory(UNKNOWN_D730) != 0x40) or
             self.env.ram_interface.read_memory(BATTLE_TEXT_PAUSE_FLAG) == 0x00):
            return self.env.GameState.BATTLE_TEXT

        if state != RedRamMenuValues.UNKNOWN_MENU:
            if self.env.menus._get_menu_item_state(cursor_location) != RedRamSubMenuValues.UNKNOWN_MENU:
                item_number = self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_1) + self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_2) + 1
                state = TEXT_MENU_ITEM_LOCATIONS.get(item_number, RedRamMenuValues.ITEM_RANGE_ERROR)

            return state

        return self.env.GameState.GAME_STATE_UNKNOWN

    def get_battle_state(self):
        battle_type = self.get_battle_type()
        in_pre_battle = self.is_in_pre_battle()

        if not (battle_type or in_pre_battle):
            self.turns_in_current_battle = 1
            self.last_turn_count = 0
            self.in_battle = False
            self.battle_done = False
            self.new_turn = False
            return self.env.GameState.GAME_STATE_UNKNOWN
        
        self.in_battle = True

        turns_in_current_battle = self.env.ram_interface.read_memory(TURNS_IN_CURRENT_BATTLE)
        if turns_in_current_battle != self.last_turn_count:
            self.turns_in_current_battle += 1 
            self.last_turn_count = turns_in_current_battle
            self.new_turn = True
        else:
            self.new_turn = False

        return self._get_battle_menu_state(battle_type)
    
    
    def win_battle(self):
        # You can only win once per battle, so don't call w/o being ready to process a win otherwise you'll lose capturing it for the battle cycle
        if (self.in_battle == False or self.battle_done == True or self.get_battle_type() == 0 or
            self.get_battles_pokemon_left() != 0 or self.env.ram_interface.read_memory(TURNS_IN_CURRENT_BATTLE) == 0):
            return False
        
        self.battle_done = True
        return True
    
    def get_battle_type(self):
        battle_type = self.env.ram_interface.read_memory(BATTLE_TYPE)
        if battle_type == 255:
            battle_type = BattleTypes.DIED  # Died in battle, reassigned to 4 to save bits as 4-255 unused
        return battle_type
    
    def is_in_pre_battle(self):
        return self.env.ram_interface.read_memory(CURRENT_OPPONENT)
    
    def get_special_battle_type(self):
        return self.env.ram_interface.read_memory(SPECIAL_BATTLE_TYPE)
    
    def get_player_head_index(self):
        return self.env.ram_interface.read_memory(PLAYER_LOADED_POKEMON)
    
    def get_player_head_pokemon(self):
        offset = self._loaded_pokemon_address()
        return Pokemon(self.env).get_pokemon(offset)

    def get_player_party_head_hp(self):
        offset = self._loaded_pokemon_address()
        return Pokemon(self.env).get_pokemon_health(offset)
    
    def get_player_party_head_status(self):
        offset = self._loaded_pokemon_address()
        return Pokemon(self.env).get_pokemon_status(offset)
    
    def get_player_party_head_pp(self):
        offset = self._loaded_pokemon_address()
        return Pokemon(self.env).get_pokemon_pp_avail(offset)
    
    def get_player_party_head_modifiers(self):
        if not self.get_battle_type():
            return 0, 0, 0, 0, 0, 0

        attack_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_ATTACK_MODIFIER)
        defense_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_DEFENSE_MODIFIER)
        speed_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPEED_MODIFIER)
        accuracy_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_ACCURACY_MODIFIER)
        special_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPECIAL_MODIFIER)
        evasion_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPECIAL_MODIFIER)

        return attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod
    
    def get_player_head_modifiers_dict(self):
        attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod = self.get_player_party_head_modifiers()

        return {
            'attack_mod': attack_mod,
            'defense_mod': defense_mod,
            'speed_mod': speed_mod,
            'accuracy_mod': accuracy_mod,
            'special_mod': special_mod,
            'evasion_mod': evasion_mod
        }
    
    def get_enemy_party_count(self):
        return self.env.ram_interface.read_memory(ENEMY_PARTY_COUNT)
    
    def get_enemy_party_head_pokemon(self):
        return self.env.ram_interface.read_memory(ENEMYS_POKEMON)
    
    def get_enemy_party_head_types(self):
        return self.env.ram_interface.read_memory(ENEMYS_POKEMON_TYPES[0]), self.env.ram_interface.read_memory(ENEMYS_POKEMON_TYPES[1])
    
    def get_enemy_party_head_hp(self):
        hp_total = (self.env.ram_interface.read_memory(ENEMYS_POKEMON_MAX_HP[0]) << 8) + self.env.ram_interface.read_memory(ENEMYS_POKEMON_MAX_HP[1])
        hp_avail = (self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[0]) << 8) + self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[1])

        return hp_total, hp_avail
    
    def get_enemy_party_head_level(self):
        return self.env.ram_interface.read_memory(ENEMYS_POKEMON_LEVEL)
    
    def get_enemy_party_head_status(self):
        return self.env.ram_interface.read_memory(ENEMYS_POKEMON_STATUS)
    
    def get_enemy_party_head_modifiers(self):
        attack_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_ATTACK_MODIFIER)
        defense_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_DEFENSE_MODIFIER)
        speed_mod =  self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPEED_MODIFIER)
        accuracy_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_ACCURACY_MODIFIER)
        special_mod =  self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPECIAL_MODIFIER)
        evasion_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPECIAL_MODIFIER)

        return attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod

    def get_enemy_fighting_pokemon_dict(self):
        hp_total, hp_avail = self.get_enemy_party_head_hp()
        attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod = self.get_enemy_party_head_modifiers()
        type_1, type_2 = self.get_enemy_party_head_types()

        return {
            'party_count': self.get_enemy_party_count(),
            'pokemon': self.get_enemy_party_head_pokemon(),
            'level': self.get_enemy_party_head_level(),
            'hp_total': hp_total,
            'hp_avail': hp_avail,
            'type_1': type_1,
            'type_2': type_2,
            'status': self.get_enemy_party_head_status(),
            'attack_mod':  attack_mod,
            'defense_mod': defense_mod,
            'speed_mod': speed_mod,
            'accuracy_mod': accuracy_mod,
            'special_mod': special_mod,
            'evasion_mod': evasion_mod
        }

    def get_battle_turn_moves(self):
        player_selected_move = self.env.ram_interface.read_memory(PLAYER_SELECTED_MOVE)
        enemy_selected_move = self.env.ram_interface.read_memory(ENEMY_SELECTED_MOVE)

        return player_selected_move, enemy_selected_move                                                         
    
    def get_battles_pokemon_left(self):
        alive_pokemon = 0

        if not self.in_battle:
            return 0 

        # Wild mons only have 1 pokemon alive and their status is in diff reg's
        if self.get_battle_type() == BattleTypes.WILD_BATTLE:
            return int(self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[0]) != 0 or self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[1]) != 0)
        
        for i in range(POKEMON_MAX_COUNT):
            if (self.env.ram_interface.read_memory(ENEMY_TRAINER_POKEMON_HP[0] + ENEMY_TRAINER_POKEMON_HP_OFFSET * i) != 0 or
                self.env.ram_interface.read_memory(ENEMY_TRAINER_POKEMON_HP[1] + ENEMY_TRAINER_POKEMON_HP_OFFSET * i) != 0):
                alive_pokemon += 1 

        return alive_pokemon
    
    def get_battle_type_hint(self): 
        if not self.get_battle_type():
            return 0

        pokemon = self.env.ram_interface.read_memory(PLAYER_LOADED_POKEMON)
        player_type_1 = self.env.ram_interface.read_memory(POKEMON_1_TYPES[0] + pokemon * PARTY_OFFSET)
        player_type_2 = self.env.ram_interface.read_memory(POKEMON_1_TYPES[1] + pokemon * PARTY_OFFSET)
        enemy_type_1 = self.env.ram_interface.read_memory(ENEMYS_POKEMON_TYPES[0])
        enemy_type_2 = self.env.ram_interface.read_memory(ENEMYS_POKEMON_TYPES[1])

        return (max(POKEMON_MATCH_TYPES.get((player_type_1, enemy_type_1), 1), POKEMON_MATCH_TYPES.get((player_type_1, enemy_type_2), 1)) *
                max(POKEMON_MATCH_TYPES.get((player_type_2, enemy_type_1), 1), POKEMON_MATCH_TYPES.get((player_type_2, enemy_type_2), 1)))
    
    def get_enemy_lineup_levels(self):
        # Wild Pokemon, only ever one
        if self.get_battle_type() == BattleTypes.WILD_BATTLE:
            return [self.env.ram_interface.read_memory(ENEMYS_POKEMON_LEVEL)]

        lineup_levels = []
        for party_index in range(POKEMON_PARTY_SIZE):
            offset = party_index * ENEMYS_POKEMON_OFFSET
            level = self.env.ram_interface.read_memory(ENEMYS_POKEMON_INDEX_LEVEL + offset)
            if level:
                lineup_levels.append(level)
            else:
                break

        return lineup_levels


class Items:
    def __init__(self, env):
        self.env = env

    def _get_items_in_range(self, size, index, offset):
        items = [None] * size
        for i in range(size):
            item_val = self.env.ram_interface.read_memory(index + i * offset)
            items[i] = 0 if item_val == 0xFF else item_val  # Modern parsing, we don't need termination byte so strip it out
        return items
    
    def get_bag_item_count(self):
        return self.env.ram_interface.read_memory(BAG_TOTAL_ITEMS)

    def get_bag_item_ids(self):
        return np.array(self._get_items_in_range(BAG_SIZE, BAG_ITEMS_INDEX, ITEMS_OFFSET))

    def get_bag_item_quantities(self):
        return np.array([self.env.ram_interface.read_memory(BAG_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(BAG_SIZE)], dtype=np.uint8)

    def get_pc_item_ids(self):
        return np.array(self._get_items_in_range(STORAGE_SIZE, PC_ITEMS_INDEX, ITEMS_OFFSET))
    
    def get_pc_item_quantities(self):
        return np.array([self.env.ram_interface.read_memory(PC_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(STORAGE_SIZE)], dtype=np.uint8)
    
    def get_pc_pokemon_count(self):
        return self.env.ram_interface.read_memory(BOX_POKEMON_COUNT)
    
    def get_pc_pokemon_stored(self):
        return np.array([(self.env.ram_interface.read_memory(BOX_POKEMON_1 + i * BOX_OFFSET), self.env.ram_interface.read_memory(BOX_POKEMON_1_LEVEL + i * BOX_OFFSET)) for i in range(BOX_SIZE)], dtype=np.uint8)

    def get_item_quantity(self):
        # TODO: need to map sub menu state for buy/sell count
        if self.env.game_state != RedRamMenuValues.ITEM_QUANTITY:
            return np.array([0], dtype=np.float32)
        
        return np.array([self.env.ram_interface.read_memory(ITEM_SELECTION_QUANTITY)], dtype=np.float32)
                

class Map:
    def __init__(self, env):
        self.env = env

    def get_current_map(self):
        return self.env.ram_interface.read_memory(PLAYER_MAP)

    def get_current_location(self):
        return self.env.ram_interface.read_memory(PLAYER_LOCATION_X), self.env.ram_interface.read_memory(PLAYER_LOCATION_Y), self.get_current_map()
    
    def get_centered_7x7_tiles(self):
        # Starting addresses for each row
        starting_addresses = [TILE_COL_1_ROW_1, TILE_COL_1_ROW_2, TILE_COL_1_ROW_3, TILE_COL_1_ROW_4,
                               TILE_COL_1_ROW_5, TILE_COL_1_ROW_6, TILE_COL_1_ROW_7]
        screen_size = len(starting_addresses)
        increment_per_column = 2

        screen = np.zeros((screen_size, screen_size), dtype=np.uint8)
        for row, start_addr in enumerate(starting_addresses):
            for col in range(screen_size):
                address = start_addr + col * increment_per_column
                
                screen[row][col] = self.env.ram_interface.read_memory(address)

        return screen
    
    def get_centered_step_count_7x7_screen(self):
        collision_ptr_1, collision_ptr_2 = self.env.ram_interface.read_memory(TILE_COLLISION_PTR_1), self.env.ram_interface.read_memory(TILE_COLLISION_PTR_2)
        screen = self.get_centered_7x7_tiles()
        for row in range(screen.shape[0]):
            for col in range(screen.shape[1]):
                if screen[row][col] in WALKABLE_TILES.get((collision_ptr_1, collision_ptr_2), []):
                    screen[row][col] = 0
                else:
                    screen[row][col] = 1

        return screen
    

    def get_npc_location_dict(self, skip_moving_npc=True):
        # Moderate testing show's NPC's are never on screen during map transitions
        sprites = {}
        for i, sprite_addr in enumerate(SPRITE_STARTING_ADDRESSES):
            on_screen = self.env.ram_interface.read_memory(sprite_addr + 0x0002)

            if on_screen == 0xFF:
                continue

            # Moving sprites can cause complexity, use at discretion
            #can_move = self.env.ram_interface.read_memory(sprite_addr + 0x0106)
            #if skip_moving_npc and can_move != 0xFF:
            #    continue
            
            picture_id = self.env.ram_interface.read_memory(sprite_addr)
            x_pos = self.env.ram_interface.read_memory(sprite_addr + 0x0105) - 4  # topmost 2x2 tile has value 4), thus the offset
            y_pos = self.env.ram_interface.read_memory(sprite_addr + 0x0104) - 4  # topmost 2x2 tile has value 4), thus the offset
            # facing = self.env.ram_interface.read_memory(sprite_addr + 0x0009)

            sprites[(x_pos, y_pos, self.get_current_map())] = picture_id
            
        return sprites


class Menus:
    def __init__(self, env):
        self.env = env

    def get_item_menu_context(self):
        cursor_location = (self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_LOCATION[0]),
                    self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_LOCATION[1]))
        return cursor_location, TEXT_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamMenuValues.UNKNOWN_MENU)
    
    def get_pre_battle_menu_state(self):
        text_on_screen = self.env.ram_interface.read_memory(TEXT_FONT_ON_LOADED)
        if not text_on_screen:
            return self.env.GameState.GAME_STATE_UNKNOWN
                
        cursor_location, state = self.get_item_menu_context()
        text_dst_ptr = self.env.ram_interface.read_memory(TEXT_DST_POINTER)
        if state == RedRamMenuValues.MENU_YES or state == RedRamMenuValues.MENU_NO:
            if text_dst_ptr == 0xF2 and state == RedRamMenuValues.MENU_YES:
                return RedRamMenuValues.OVERWRITE_MOVE_YES
            elif text_dst_ptr == 0xF2 and state == RedRamMenuValues.MENU_NO:
                return RedRamMenuValues.OVERWRITE_MOVE_NO
            elif text_dst_ptr == 0xB9 and state == RedRamMenuValues.MENU_YES:
                return RedRamMenuValues.ABANDON_MOVE_YES
            elif text_dst_ptr == 0xB9 and state == RedRamMenuValues.MENU_NO:
                return RedRamMenuValues.ABANDON_MOVE_NO
            elif text_dst_ptr == 0xEE or text_dst_ptr == 0xF0:  # would otherwise be default y/n on a text screen
                return self.env.GameState.TALKING
        elif cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_N and text_dst_ptr == 0xB9:  # Overwrite one off shared menu
            return RedRamMenuValues.OVERWRITE_MOVE_1
        elif (cursor_location == RedRamMenuKeys.OVERWRITE_MOVE_2 or
               cursor_location == RedRamMenuKeys.OVERWRITE_MOVE_3 or
                 cursor_location == RedRamMenuKeys.OVERWRITE_MOVE_4) and text_dst_ptr == 0xB9:
            return state
            
        return self.env.GameState.GAME_STATE_UNKNOWN


    def get_menu_state(self):
        text_on_screen = self.env.ram_interface.read_memory(TEXT_FONT_ON_LOADED)
        if text_on_screen:
            cursor_location, state = self.get_item_menu_context()

            # when text is on screen but menu reg's are clear, we can't be in a menu
            if cursor_location == RedRamMenuKeys.MENU_CLEAR:
                return self.env.GameState.TALKING
            
            # In a sub-box that requires fetching count of menu pos, such as mart items
            sub_state = self._get_menu_item_state(cursor_location)
            if sub_state != RedRamSubMenuValues.UNKNOWN_MENU:
                return sub_state

            # check the bigger of the two submenu's, they have the same val's, to see if we are in a submenu
            sub_state = self._get_sub_menu_state(cursor_location)
            if sub_state != RedRamSubMenuValues.UNKNOWN_MENU:
                return sub_state

            return state
        else:
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[0], 0x00)
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[1], 0x00)
            for i in range(POKEMART_AVAIL_SIZE):
                self.env.ram_interface.write_memory(POKEMART_ITEMS + i, 0x00)

        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def _get_sub_menu_state(self, cursor_location):
        if PC_POKE_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU) == RedRamSubMenuValues.UNKNOWN_MENU:
            return RedRamSubMenuValues.UNKNOWN_MENU

        # Peek at screen memory to detect submenu's which have hard coded menu renderings w/ diff's between them. Reverse engineered.
        pc_menu_screen_peek = self.env.ram_interface.read_memory(PC_SUB_MENU_SCREEN_PEEK)

        # pokemon pc sub menu
        if pc_menu_screen_peek == 0x91:
            if cursor_location != RedRamSubMenuKeys.SUB_MENU_6:  # menu 6 is the same for deposit and withdraw so we have to normalize it
                return PC_POKE_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU)
            else:
                pc_menu_screen_peek = self.env.ram_interface.read_memory(PC_SUB_MENU_DEPO_WITH_SCREEN_PEEK)
                return RedRamSubMenuValues.PC_SOMEONE_CONFIRM_WITHDRAW if pc_menu_screen_peek == 0x91 else RedRamSubMenuValues.PC_SOMEONE_CONFIRM_DEPOSIT
            
        # item pc sub menu
        elif pc_menu_screen_peek == 0x93:
            return PC_ITEM_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU)
        
        return RedRamSubMenuValues.UNKNOWN_MENU

    def _get_menu_item_state(self, cursor_location):
        if cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_1 or cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_2 or cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_N:
            item_number = self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_1) + self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_2) + 1
            if self.env.ram_interface.read_memory(ITEM_COUNT_SCREEN_PEAK) == 0x7E:  # 0x7E is the middle pokeball icon on screen, unique to the 3 sub menu pop out
                return RedRamMenuValues.ITEM_QUANTITY
            
            # self.env.ram_interface.write_memory(ITEM_SELECTION_QUANTITY, 0x00)
            
            return TEXT_MENU_ITEM_LOCATIONS.get(item_number, RedRamMenuValues.ITEM_RANGE_ERROR)
        
        return RedRamSubMenuValues.UNKNOWN_MENU
    

class Pokemon:
    def __init__(self, env):
        self.env = env
    
    def get_pokemon(self, offset):
        return self.env.ram_interface.read_memory(POKEMON_1 + offset)
    
    def get_pokemon_level(self, offset):
        return self.env.ram_interface.read_memory(POKEMON_1_LEVEL_ACTUAL + offset)
    
    def get_pokemon_type(self, offset):
        type_1 = self.env.ram_interface.read_memory(POKEMON_1_TYPES[0] + offset)
        type_2 = self.env.ram_interface.read_memory(POKEMON_1_TYPES[1] + offset)

        return type_1, type_2
    
    def get_pokemon_health(self, offset):
        hp_total = (self.env.ram_interface.read_memory(POKEMON_1_MAX_HP[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_MAX_HP[1] + offset)
        hp_avail = (self.env.ram_interface.read_memory(POKEMON_1_CURRENT_HP[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_CURRENT_HP[1] + offset)

        return hp_total, hp_avail
    
    def get_pokemon_xp(self, offset):
        xp = ((self.env.ram_interface.read_memory(POKEMON_1_EXPERIENCE[0] + offset) << 16) +
              (self.env.ram_interface.read_memory(POKEMON_1_EXPERIENCE[1] + offset) << 8) +
               self.env.ram_interface.read_memory(POKEMON_1_EXPERIENCE[2] + offset))

        return xp
    
    def get_pokemon_moves(self, offset):
        move_1 = self.env.ram_interface.read_memory(POKEMON_1_MOVES[0]+ offset)
        move_2 = self.env.ram_interface.read_memory(POKEMON_1_MOVES[1]+ offset)
        move_3 = self.env.ram_interface.read_memory(POKEMON_1_MOVES[2]+ offset)
        move_4 = self.env.ram_interface.read_memory(POKEMON_1_MOVES[3]+ offset)

        return move_1, move_2, move_3, move_4
    
    def get_pokemon_pp_avail(self, offset):
        pp_1 = self.env.ram_interface.read_memory(POKEMON_1_PP_MOVES[0]+ offset)
        pp_2 = self.env.ram_interface.read_memory(POKEMON_1_PP_MOVES[1]+ offset)
        pp_3 = self.env.ram_interface.read_memory(POKEMON_1_PP_MOVES[2]+ offset)
        pp_4 = self.env.ram_interface.read_memory(POKEMON_1_PP_MOVES[3]+ offset)

        return pp_1, pp_2, pp_3, pp_4
    
    def get_pokemon_stats(self, offset):
        attack = (self.env.ram_interface.read_memory(POKEMON_1_ATTACK[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_ATTACK[1] + offset)
        defense = (self.env.ram_interface.read_memory(POKEMON_1_DEFENSE[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_DEFENSE[1] + offset)
        speed = (self.env.ram_interface.read_memory(POKEMON_1_SPEED[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_SPEED[1] + offset)
        special = (self.env.ram_interface.read_memory(POKEMON_1_SPECIAL[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_SPECIAL[1] + offset)

        return attack, defense, speed, special
    
    def get_pokemon_status(self, offset):
        return self.env.ram_interface.read_memory(POKEMON_1_STATUS + offset)

    def get_pokemon_data_dict(self, party_index=0):
        offset = party_index * PARTY_OFFSET
        pokemon = self.get_pokemon(offset)
        level = self.get_pokemon_level(offset)
        type_1, type_2 = self.get_pokemon_type(offset)
        hp_total, hp_avail = self.get_pokemon_health(offset)
        xp = self.get_pokemon_xp(offset)
        move_1, move_2, move_3, move_4 = self.get_pokemon_moves(offset)
        pp_1, pp_2, pp_3, pp_4 = self.get_pokemon_pp_avail(offset)
        attack, defense, speed, special = self.get_pokemon_stats(offset)
        health_status = self.get_pokemon_status(offset)

        # http://www.psypokes.com/rby/maxstats.php
        return {
            'pokemon': pokemon,
            'level': level,
            'type_1': type_1,
            'type_2': type_2,
            'hp_total': hp_total,  # HP Max is 703
            'hp_avail': hp_avail,
            'xp': xp,
            'move_1': move_1,
            'move_2': move_2,
            'move_3': move_3,
            'move_4': move_4,
            'pp_1': pp_1,
            'pp_2': pp_2,
            'pp_3': pp_3,
            'pp_4': pp_4,
            'attack': attack,  # Max is 366
            'defense': defense,  # Max is 458
            'speed': speed,  # Max is 378
            'special': special,  # Max is 406
            'health_status': health_status
        }


class Player:
    def __init__(self, env):
        self.env = env
    
    def _pokedex_bit_count(self, pokedex_address):
        bit_count = 0
        for i in range(POKEDEX_ADDR_LENGTH):
            binary_value = bin(self.env.ram_interface.read_memory(pokedex_address + i))
            bit_count += binary_value.count('1')

        return bit_count

    def get_player_lineup_dict(self):
        return [Pokemon(self.env).get_pokemon_data_dict(i) for i in range(POKEMON_PARTY_SIZE)]

    def get_player_lineup_pokemon(self):
        return [Pokemon(self.env).get_pokemon(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_levels(self):
        return [Pokemon(self.env).get_pokemon_level(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_health(self):
        return [Pokemon(self.env).get_pokemon_health(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_xp(self):
        return [Pokemon(self.env).get_pokemon_xp(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_moves(self):
        return [Pokemon(self.env).get_pokemon_moves(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_pp(self):
        return [Pokemon(self.env).get_pokemon_pp_avail(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_stats(self):
        return [Pokemon(self.env).get_pokemon_stats(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_types(self):
        return [Pokemon(self.env).get_pokemon_type(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def get_player_lineup_status(self):
        return [Pokemon(self.env).get_pokemon_status(i * PARTY_OFFSET) for i in range(POKEMON_PARTY_SIZE)]
    
    def is_following_npc(self):
        if self.env.ram_interface.read_memory(FOLLOWING_NPC_FLAG) != 0x00:
            return self.env.GameState.FOLLOWING_NPC
        
        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def get_badges(self):
        return self.env.ram_interface.read_memory(OBTAINED_BADGES)
    
    def get_pokedex_seen(self):
        return self._pokedex_bit_count(POKEDEX_SEEN)
    
    def get_pokedex_owned(self):
        return self._pokedex_bit_count(POKEDEX_OWNED)
    
    def get_player_money(self):
        # Trigger warning, money is a base16 literal as base 10 numbers, max money 999,999
        money_bytes = [self.env.ram_interface.read_memory(addr) for addr in PLAYER_MONEY]
        money_hex = ''.join([f'{byte:02x}' for byte in money_bytes])
        money_int = int(money_hex, 10)
        return money_int

    
