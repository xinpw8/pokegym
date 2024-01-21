import os
from os.path import exists
import sys
from pyboy import PyBoy, WindowEvent

from .red_ram_api import *


def _pokemon_dict_pretty_str(lineup):
        pokemons = ""
        for pokemon in lineup:
                for key, val in pokemon.items():
                        if key == 'pokemon':
                                pokemons += f'{key}: {POKEMON_LOOKUP.get(val, "None")}, '
                        else:
                                pokemons += f'{key}: {val}, '
                pokemons += '\n'
        return pokemons

def get_player_str(game):
        pokemons = "Pokemon:\n" + _pokemon_dict_pretty_str(game.player.get_player_lineup_dict())

        money = game.player.get_player_money()
        pokedex_seen = game.player.get_pokedex_seen()
        pokedex_owned = game.player.get_pokedex_owned()
        badges = game.player.get_badges()

        return f'{pokemons}\nmoney: {money}, pokedex_seen: {pokedex_seen}, pokedex_owned: {pokedex_owned}, badges: {badges}'


def get_items_str(game):
        bag_ids = game.items.get_bag_item_ids()
        bag_quan = ' '.join(map(str, game.items.get_bag_item_quantities().flatten()))
        pc_item_ids = ' '.join(map(str, game.items.get_pc_item_ids().flatten()))
        pc_item_quan = ' '.join(map(str, game.items.get_pc_item_quantities().flatten()))
        pc_pokemon_count = game.items.get_pc_pokemon_count()
        pc_pokemon_data = ' '.join(map(str, game.items.get_pc_pokemon_stored().flatten()))
        item_quantity = game.items.get_item_quantity()

        return f'\n\nbag_ids: {bag_ids} \nbag_quan: {bag_quan} \npc_item_ids: {pc_item_ids} \npc_item_quan: {pc_item_quan} \npc_pokemon_count: {pc_pokemon_count} \npc_pokemon_data: {pc_pokemon_data} \nitem_selection_quantity: {item_quantity}'

def get_world_str(game):
        milestones = game.world.get_game_milestones()
        audio = game.world.get_playing_audio_track()
        pokemart = game.world.get_pokemart_options()

        return f'\n\nmilestones: {milestones}, audio: {audio}, pokemart: {pokemart}'

def get_battle_str(game):
        in_battle = game.battle.in_battle
        battle_done = game.battle.battle_done
        battle_type = game.battle.get_battle_type()
        enemys_left = game.battle.get_battles_pokemon_left()
        win_battle = game.battle.win_battle()
        player_head_index = game.battle.get_player_head_index()
        player_modifiers = game.battle.get_player_head_modifiers_dict()
        enemy_fighting = _pokemon_dict_pretty_str([game.battle.get_enemy_fighting_pokemon_dict()])
        player_move, enemy_move = game.battle.get_battle_turn_moves()
        type_hint = game.battle.get_battle_type_hint()

        return f'\n\nin_battle: {in_battle}, battle_done: {battle_done}, battle_type: {battle_type}, enemys_left: {enemys_left}, win_battle: {win_battle}\nhead_index: {player_head_index}\nplayer_mods: {player_modifiers} \nenemy_stats: {enemy_fighting} \ntype_hint: {type_hint}\player_move: {player_move}, enemy_move: {enemy_move}'

def get_map_str(game):
        location = game.map.get_current_location()
        tiles = game.map.get_centered_7x7_tiles()
        npc = game.map.get_npc_location_dict()

        return f'\n\nlocation: {location}\ntiles:\n{tiles}\nnpc: {npc}'

def get_debug_str(game):
        game_state = f'{game.game_state.name}\n'
        game_state += f'Menu Allowed: {game.allow_menu_selection(WindowEvent.PRESS_BUTTON_A)}\n\n'

        game_state += get_player_str(game)
        game_state += get_items_str(game)
        game_state += get_world_str(game)
        game_state += get_battle_str(game)
        # game_state += get_map_str(game)

        return game_state

