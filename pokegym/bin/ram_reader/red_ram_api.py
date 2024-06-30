from enum import IntEnum
import numpy as np
from pyboy.utils import WindowEvent
# from pyboy import PyBoy
from pokegym.pyboy_singleton import get_pyboy_instance

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
        return self.pyboy.memory[address]

    def write_memory(self, address, value):
        self.pyboy.memory[address] = value
        

class Game:
    def __init__(self, pyboy):
        self.ram_interface = PyBoyRAMInterface(pyboy)

        self.world = World(self)
        self.battle = Battle(self)
        self.items = Items(self)
        self.map = Map(self)
        self.menus = Menus(self)
        self.player = Player(self)
        self.pokemon = Pokemon(self)

        self.game_state = self.GameState.GAME_STATE_UNKNOWN

        self.process_game_states()
        gb_path = "pokemon_red.gb"
        symbols_path = "pokemon_red.sym"
        pyboy = get_pyboy_instance(gb_path, symbols_path)

    class GameState(IntEnum):
        FILTERED_INPUT = 0
        IN_BATTLE = 1
        BATTLE_ANIMATION = 2
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

    def process_game_states(self):
        ORDERED_GAME_STATES = [
            self.menus.get_pre_battle_menu_state,
            self.battle.get_battle_state,
            self.player.is_following_npc,
            self.menus.get_menu_state,
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
            RedRamMenuValues.NAME_POKEMON_YES: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CONFIRM_STATS: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CHANGE_BOX: {WindowEvent.PRESS_BUTTON_A},
        }

        filtered_keys = FILTERED_INPUTS.get(self.game_state, None)
        if filtered_keys is None or input not in filtered_keys:
            return True

        return False

    def does_party_have_hm(self, hm_id):
        party_size = self.player._get_lineup_size()
        for i in range(party_size):
            pokemon_data = self.pokemon.get_pokemon_data_dict(i)
            for j in range(4):
                try:
                    if hm_id == pokemon_data[f"move_{j+1}"]:
                        return True
                except:
                    continue
        return False
    
    def print_menu_info(self):
        # Call the API to update game states
        self.process_game_states()
        
        # Print the current game state
        print(f"Current game state: {self.game_state.name}")

        # Check and print the current menu state if in a menu
        if self.game_state in [
            self.GameState.START_MENU, 
            self.GameState.GAME_MENU, 
            self.GameState.BATTLE_TEXT, 
            self.GameState.ON_PC, 
            self.GameState.MART, 
            self.GameState.GYM, 
            self.GameState.POKE_CENTER
        ]:
            current_menu_state = self.menus.get_menu_state()
            print(f"Current menu state: {current_menu_state}")

    def print_api_info(self):
        self.print_item_info()
        self.print_menu_info()
        self.print_player_info()
        self.print_pokemon_info()

    def print_item_info(self):
        bag_item_count = self.items.get_bag_item_count()
        item_ids = self.items.get_bag_item_ids()
        item_ids_no_lookup = self.items.get_bag_item_ids_no_lookup()
        print(f'Bag Item Count: {bag_item_count}')
        print(f'Bag Item IDs: {item_ids}')
        print(f'Bag Item IDs (No Lookup): {item_ids_no_lookup}')

    def print_player_info(self):
        lineup_pokemon = self.player.get_player_lineup_pokemon()
        lineup_levels = self.player.get_player_lineup_levels()
        print(f'Player Pokemon: {lineup_pokemon}')
        print(f'Pokemon Levels: {lineup_levels}')

    def print_pokemon_info(self):
        for i in range(self.player._get_lineup_size()):
            pokemon_data = self.pokemon.get_pokemon_data_dict(i)
            print(f'Pokemon {i+1}: {pokemon_data}')


class World:
    def __init__(self, env):
        self.env = env
    
    def get_game_milestones(self):
        return np.array([self.env.ram_interface.read_memory(item) for item in GAME_MILESTONES], dtype=np.uint8)
    
    def get_playing_audio_track(self):
        return self.env.ram_interface.read_memory(AUDIO_CURRENT_TRACK_NO_DELAY)
    
    def get_overlay_audio_track(self):
        return self.env.ram_interface.read_memory(AUDIO_OVERLAY_SOUND)
    
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
            if item_val == 0xFF:
                items[i] = ""  # Represent empty slots as "Empty"
            elif item_val in ITEM_LOOKUP:
                items[i] = ITEM_LOOKUP[item_val]
            elif item_val == 4:
                items[i] = "Pokeball"  # Add Pokeball to the lookup
            else:
                items[i] = ""  # Handle unknown items
        return items
    
    def _get_items_in_range_no_lookup(self, size, index, offset):
        items = [None] * size
        for i in range(size):
            item_val = self.env.ram_interface.read_memory(index + i * offset)
            if item_val == 0xFF:
                items[i] = ""  # Represent empty slots as "Empty"
            elif item_val == 4:
                items[i] = "Pokeball"  # Add Pokeball to the lookup
            else:
                items[i] = hex(item_val)  # Return the hexadecimal representation of the item value
        return items
        
    def get_bag_item_count(self):
        return self.env.ram_interface.read_memory(BAG_TOTAL_ITEMS)

    def get_bag_item_ids(self):
        return np.array(self._get_items_in_range(BAG_SIZE, BAG_ITEMS_INDEX, ITEMS_OFFSET))
    
    def get_bag_item_ids_no_lookup(self):
        return np.array(self._get_items_in_range_no_lookup(BAG_SIZE, BAG_ITEMS_INDEX, ITEMS_OFFSET))

    def get_bag_item_quantities(self):
        item_quan = [self.env.ram_interface.read_memory(BAG_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(self.get_bag_item_count())]
        padded_quan = np.pad(item_quan, (0, BAG_SIZE - len(item_quan)), constant_values=0)
        return np.array(padded_quan, dtype=np.uint8)
    
    def get_pc_item_count(self):
        return self.env.ram_interface.read_memory(PC_TOTAL_ITEMS)
        
    def get_pc_item_ids(self):
        return np.array(self._get_items_in_range(STORAGE_SIZE, PC_ITEMS_INDEX, ITEMS_OFFSET))
    
    def get_pc_item_quantities(self):
        item_quan = [self.env.ram_interface.read_memory(PC_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(self.get_pc_item_count())]
        try:
            padded_quan = np.pad(item_quan, (0, STORAGE_SIZE - len(item_quan)), constant_values=0)
        except:
            padded_quan = 0
        return np.array(padded_quan, dtype=np.uint8)
    
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
    
    def get_collision_pointer(self):
        return np.uint16((self.env.ram_interface.read_memory(TILE_COLLISION_PTR_1) << 8) + self.env.ram_interface.read_memory(TILE_COLLISION_PTR_2))
    
    def get_tileset_index(self):
        return self.env.ram_interface.read_memory(TILESET_INDEX)

    def get_collision_tiles(self):
        collision_ptr = self.get_collision_pointer()
        collection_tiles = set()
        while True:
            collision = self.env.ram_interface.read_memory(collision_ptr)
            if collision == 0xFF:
                break

            collection_tiles.add(collision)
            collision_ptr += 1

        return collection_tiles
    
    def get_screen_tilemaps(self):
        bsm = self.env.ram_interface.pyboy.botsupport_manager()
        ((scx, scy), (wx, wy)) = bsm.screen().tilemap_position()
        tilemap = np.array(bsm.tilemap_background()[:, :])
        screen_tiles = (np.roll(np.roll(tilemap, -scy // 8, axis=0), -scx // 8, axis=1)[:18, :20] - 0x100)

        top_left_tiles = screen_tiles[:screen_tiles.shape[0]: 2,::2]
        bottom_left_tiles = screen_tiles[1: 1 + screen_tiles.shape[0]: 2,::2]

        return top_left_tiles, bottom_left_tiles
    

    def get_npc_location_dict(self, skip_moving_npc=False):
        # Moderate testing show's NPC's are never on screen during map transitions
        sprites = {}
        for i, sprite_addr in enumerate(SPRITE_STARTING_ADDRESSES):
            on_screen = self.env.ram_interface.read_memory(sprite_addr + 0x0002)

            if on_screen == 0xFF:
                continue

            # Moving sprites can cause complexity, use at discretion
            if skip_moving_npc and self.env.ram_interface.read_memory(sprite_addr + 0x0106) != 0xFF:
                continue
            
            picture_id = self.env.ram_interface.read_memory(sprite_addr)
            x_pos = self.env.ram_interface.read_memory(sprite_addr + 0x0105) - 4  # topmost 2x2 tile has value 4), thus the offset
            y_pos = self.env.ram_interface.read_memory(sprite_addr + 0x0104) - 4  # topmost 2x2 tile has value 4), thus the offset
            # facing = self.env.ram_interface.read_memory(sprite_addr + 0x0009)

            sprites[(x_pos, y_pos, self.get_current_map())] = picture_id
            
        return sprites
    
    def get_warp_tile_count(self):
        return self.env.ram_interface.read_memory(WARP_TILE_COUNT)
    
    def get_warp_tile_positions(self):
        warp_tile_count = self.get_warp_tile_count()
        warp_tile_positions = set()
        for i in range(warp_tile_count):
            warp_tile_positions.add((self.env.ram_interface.read_memory(WARP_TILE_X_ENTRY + i * WARP_TILE_ENTRY_OFFSET),
                                     self.env.ram_interface.read_memory(WARP_TILE_Y_ENTRY + i * WARP_TILE_ENTRY_OFFSET)))
        
        return warp_tile_positions


class Menus:
    def __init__(self, env):
        self.env = env

    def _get_sub_menu_item_number(self):
        return self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_1) + self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_2) + 1

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
        id_working_reg = self.env.ram_interface.read_memory(PRE_DEF_ID)
        if (state == RedRamMenuValues.MENU_YES or state == RedRamMenuValues.MENU_NO) and id_working_reg == 0x2D:
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
        elif cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_N and text_dst_ptr == 0xB9 and id_working_reg == 0x2D:  # Shares submenu w/ mart 3-10 items
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
            
            # check HM menu overlays
            sub_state = self._get_hm_menu_state(cursor_location)
            if sub_state != RedRamSubMenuValues.UNKNOWN_MENU:
                return sub_state

            return state
        else:
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[0], 0x00)
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[1], 0x00)
            for i in range(POKEMART_AVAIL_SIZE):
                self.env.ram_interface.write_memory(POKEMART_ITEMS + i, 0x00)

        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def _get_hm_menu_state(self, cursor_location):
        cc50 = self.env.ram_interface.read_memory(0xCC50)        
        cc52 = self.env.ram_interface.read_memory(0xCC52)        
    
        # working reg's are used and set to 41 & 14 when in pokemart healing menu
        if (cc50 == 0x41 and cc52 == 0x14) and (cursor_location == RedRamMenuKeys.POKECENTER_HEAL or cursor_location == RedRamMenuKeys.POKECENTER_CANCEL):
            return RedRamSubMenuValues.UNKNOWN_MENU  # it's known but the next stage will set it to pokecenter

        # working reg's are used and set to 58 & 20 when in HM menu
        if not (cc50 == 0x58 and cc52 == 0x20 and self.env.ram_interface.read_memory(ITEM_COUNT_SCREEN_PEAK) == 0x7C):
            return RedRamSubMenuValues.UNKNOWN_MENU
        
        # Awful hack, strength shift the menu by 1 due to it's length so do another overwrite
        if cursor_location == RedRamMenuKeys.PC_SOMEONE_DEPOSIT_WITHDRAW:
            return RedRamMenuValues.MENU_SELECT_STATS
        elif cursor_location == RedRamMenuKeys.PC_SOMEONE_STATUS:
            return RedRamMenuValues.MENU_SELECT_SWITCH
        elif cursor_location == RedRamMenuKeys.PC_SOMEONE_CANCEL:
            return RedRamMenuValues.MENU_SELECT_CANCEL
                
        cursor_menu_position = self.env.ram_interface.read_memory(TEXT_MENU_LAST_MENU_ITEM)
        max_menu_elem = self.env.ram_interface.read_memory(TEXT_MENU_MAX_MENU_ITEM)
        menu_offset = max_menu_elem - cursor_menu_position - 3  # There are 3 menu's (stats, switch, cancel) 0-indexed

        # There are no HM's after the first 3 menu's
        if menu_offset < 0:
            return RedRamSubMenuValues.UNKNOWN_MENU

        pokemon_selected = self.env.ram_interface.read_memory(0xCC2B)
        move_1, move_2, move_3, move_4 = Pokemon(self.env).get_pokemon_moves(pokemon_selected * PARTY_OFFSET)

        for move in [move_4, move_3, move_2, move_1]:
            if move in HM_MENU_LOOKUP:
                menu_offset -= 1

            if menu_offset < 0:
                return HM_MENU_LOOKUP[move]
    
        return RedRamSubMenuValues.UNKNOWN_MENU


    
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
            if self.env.ram_interface.read_memory(ITEM_COUNT_SCREEN_PEAK) == 0x7E:  # 0x7E is the middle pokeball icon on screen, unique to the 3 sub menu pop out
                return RedRamMenuValues.ITEM_QUANTITY
            
            item_number = self._get_sub_menu_item_number()
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
    
    def _get_lineup_size(self):
        return self.env.ram_interface.read_memory(POKEMON_PARTY_COUNT)

    def get_player_lineup_dict(self):
        return [Pokemon(self.env).get_pokemon_data_dict(i) for i in range(self._get_lineup_size())]

    def get_player_lineup_pokemon(self):
        return [Pokemon(self.env).get_pokemon(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_levels(self):
        return [Pokemon(self.env).get_pokemon_level(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_health(self):
        return [Pokemon(self.env).get_pokemon_health(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_xp(self):
        return [Pokemon(self.env).get_pokemon_xp(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_moves(self):
        return [Pokemon(self.env).get_pokemon_moves(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_pp(self):
        return [Pokemon(self.env).get_pokemon_pp_avail(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_stats(self):
        return [Pokemon(self.env).get_pokemon_stats(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_types(self):
        return [Pokemon(self.env).get_pokemon_type(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
    def get_player_lineup_status(self):
        return [Pokemon(self.env).get_pokemon_status(i * PARTY_OFFSET) for i in range(self._get_lineup_size())]
    
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

    def is_player_dead(self):
        return self.env.ram_interface.read_memory(PLAYER_DEAD) == 0xFF
    
