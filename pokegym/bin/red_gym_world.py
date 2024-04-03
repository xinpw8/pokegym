import numpy as np
from red_env_constants import *
from ram_reader.red_memory_items import *

class RedGymWorld:
    def __init__(self, env):
        self.env = env
        if env.debug:
            print('**** RedGymWorld ****')

        self.pokecenter_history = np.uint16(0)
        self.game_history = np.zeros((OBSERVATION_MEMORY_SIZE,), dtype=np.uint8)

    def get_pokecenter_reward(self):
        audio_id = self.env.game.world.get_playing_audio_track()
        if audio_id != 0xBD:
            return 0  # we aren't in a mart or pokecenter

        # Gen I has ~14 pokecenters, so we can use a 16-bit bitmask
        pokecenter_id = self.env.game.world.get_pokecenter_id()
        bitmask = 1 << pokecenter_id
        if self.pokecenter_history & bitmask:
            return 0

        self.pokecenter_history |= bitmask
        return 700

    def obs_game_state(self):
        self.game_history = np.roll(self.game_history, 1)
        self.game_history[0] = self.env.game.get_game_state()
        return self.game_history

    def obs_pokecenters_visited(self):
        # Convert the 16-bit number to bytes
        bytes_array = np.array([self.pokecenter_history], dtype=np.uint16).tobytes()
        # Convert bytes to an array of uint8
        uint8_array = np.frombuffer(bytes_array, dtype=np.uint8)
        # Unpack the uint8 array into bits
        pokecenter_array = np.unpackbits(uint8_array)
        return pokecenter_array
    
    def obs_playing_audio(self):
        current_audio = self.env.game.world.get_playing_audio_track()
        overlay_audio = self.env.game.world.get_overlay_audio_track()
        return np.array([current_audio, overlay_audio], dtype=np.uint8)
    
    def obs_pokemart_items(self):
        pokemart_items = self.env.game.world.get_pokemart_options()
        padded_items = np.pad(pokemart_items, (0, POKEMART_AVAIL_SIZE - len(pokemart_items)), constant_values=0)
        return np.array(padded_items, dtype=np.uint8)
    
    def obs_item_selection_quantity(self):
        return self.env.support.normalize_np_array(self.env.game.items.get_item_quantity())
    
    def obs_pc_pokemon(self):
        return self.env.game.items.get_pc_pokemon_stored().flatten()
