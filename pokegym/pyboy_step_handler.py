## pyboy_step_handler.py NEW NEW BET ADDED
from typing import Union
import warnings
import sys
import os
import time
from io import BytesIO
from pyboy.utils import WindowEvent
from pokegym.constants import *
from pokegym import ram_map_leanke
from pokegym import pyboy_binding 

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = DeprecationWarning)
    import numpy as np
    from PIL import Image, ImageFont, ImageDraw
    from pokegym.pyboy_singleton import get_pyboy_instance 

pyboy = get_pyboy_instance()

ACTION_FREQ = 24
VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

def addr_to_opcodes_list(addr: int)->list:
    return np.array([addr],dtype = np.uint16).view(np.uint8).tolist()

def cheat_bicycle(pb: pyboy)->None:
    if pb.memory[0xD700] == 0x00:
        pb.memory[0xD700] = 0x01

def cheat_battle_fastest_animations_styles_and_text_configs(pb: pyboy)->None:
    pb.memory[0xD355] = (pb.memory[0xD355] & 0x38) | 0xC0
    pb.memory[0xD358] = 0x00

def rom_hack_fast_bicycle(pb: pyboy)->None:
    (bank,addr)=(0x00,0x0D27)
    opcodes=addr_to_opcodes_list(addr)
    asm_jp_AdvancePlayerSprite=[0xC3]+opcodes
    asm_call_AdvancePlayerSprite=[0xCD]+opcodes
    (bank,addr)=(0x00,0x06A0)
    addr+=5
    for opc in [asm_call_AdvancePlayerSprite]*2+[asm_jp_AdvancePlayerSprite]:
        pb.memory[bank,addr:addr+len(opc)]=opc
        addr+=len(opc)
    (_,limit_addr)=(0x00,0x06B4)
    assert addr<limit_addr

def rom_hack_fast_text(pb: pyboy)->None:
    (bank,addr)=(0x00,0x1b33)
    pb.memory[bank,addr]=0xC9
    
def rom_hack_fast_battles(pb: pyboy)->None:
    for bank,addr in [
        [0x00,0x23B1],
        [0x03,0x7A1D],
        [0x1C,0x496D],
        [0x1E,0x40F1],
        [0x1E,0x417C],
        [0x1E,0x4D5E],
        [0x1E,0x5E6A],
        [0x1E,0x5E6A],
    ]:
        pb.memory[bank,addr]=0xC9

sys.dont_write_bytecode = True

__all__ = ["PyBoyStepHandlerPokeRed"]

def generate_gif_from_numpy(np_imgs: list, outfile_or_buff: Union[str, BytesIO, None] = None,
    return_buff: bool = True, frame_duration: int = 200, loop: bool = False,
) -> Union[bool, BytesIO]:
    if np_imgs is None or len(np_imgs) < 1:
        return False
    frames = []
    for img in np_imgs:
        try:
            frames.append(Image.fromarray(img))
        except (AttributeError, ValueError, OSError):
            pass
    buff = BytesIO() if outfile_or_buff is None else outfile_or_buff
    if len(frames) > 0:
        frames[0].save(buff, format = "GIF", optimize = True, append_images = frames,
            save_all = True, duration = max(8, int(frame_duration)), loop = 1 if loop else 0)
    if isinstance(buff, BytesIO):
        buff.seek(0)
    return buff if outfile_or_buff is None or (return_buff and isinstance(outfile_or_buff, BytesIO)) else len(frames) > 0

class PyBoyStepHandlerPokeRed:
    def __init__(self, pb: pyboy, go_between=None, verbose: bool = False, log_screen: bool = False):
        self.pyboy = pb
        self.go_between = go_between  # Save reference to GoBetween instance if provided
        if not isinstance(pb, pyboy):
            self._initialize_pyboy(pb, True)
        assert isinstance(self.pyboy, pyboy)
        self._configure_pyboy()
        self._apply_rom_hacks()
        self._apply_hooks()
        self.action_freq_dict = {0: 24, 1: 24, 2: 12, 3: 12, 4: 30}
        self.button_duration_dict = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}
        self.button_limit_dict = {0: 5, 1: 7, 2: 7, 3: 5, 4: 5}
        self.last_step_ticks = 0
        self.last_action = "n"
        self.state = 0
        self.disable_hooks = False
        self.last_sprite_update = 0
        self.return_step = 0
        self.extra_ticks = 0
        self.rendering_debug = False
        self.delayed_ticks = 0
        self.cheats_funcs_ptr = [
            cheat_bicycle,
            cheat_battle_fastest_animations_styles_and_text_configs,
        ]
        self.verbose = verbose
        self.log_screen = log_screen
        self.gif_frames = []
        self.upscale = 1
        self.debug_font = None
        self.update_font()
        self.action_freq = ACTION_FREQ

        # update init vars to True
        self.auto_teach_cut = True
        self.auto_use_cut = True
        self.auto_teach_cut = True
        self.auto_use_surf = True
        self.auto_pokeflute = True
        self.skip_rocket_hideout_bool = True
        self.skip_silph_co_bool = True

    def _stop_pyboy(self, save:bool = False) -> None:
        if isinstance(self.pyboy, pyboy) and hasattr(self.pyboy, "stop"):
            self.pyboy.stop(save)
    def close(self) -> None:
        self._stop_pyboy()

    def _initialize_pyboy(self, gamerom: Union[str, dict], headless: bool = True) -> None:
        """Initialize a pyboy instance."""
        self._stop_pyboy()
        assert isinstance(gamerom, (dict, str))
        pyboy_kwargs = (
            gamerom
            if isinstance(gamerom, dict)
            else {
                "gamerom": gamerom,
                "window": "null" if headless else "SDL2",
                "log_level": "ERROR",
                "symbols": os.path.join(os.path.dirname(__file__), "pokered.sym"),
            }
        )
        self.pyboy = pyboy(**pyboy_kwargs)
    
    def _configure_pyboy(self) -> None:
        self.pyboy.set_emulation_speed(0)

    def set_seed(self, seed: Union[int, None] = None) -> None:
        if seed is not None:
            self.pyboy.memory[0xFF04] = seed % 0x100

    def save_state(self, file_like_object) -> int:
        return self.pyboy.save_state(file_like_object)

    def load_state(self, file_like_object) -> int:
        if isinstance(file_like_object,str) and len(file_like_object)<0x1000:
            ret = False 
            with open(file_like_object,mode="rb") as f:
                ret = self.pyboy.load_state(f)
        else:
            ret = self.pyboy.load_state(file_like_object)
        self._apply_rom_hacks()
        self._apply_cheats()
        self.reset_gif_frames()
        return ret

    def update_font(self, upscale:int = 1) -> None:
        self.upscale = upscale
        self.debug_font = None
        allowed_fonts = [
            "OCRAEXT.TTF",
            "CascadiaMono.ttf",
            "consolab.ttf",
            "Lucida-Console.ttf",
            "couri.ttf",
        ]
        for font_name in allowed_fonts:
            try:
                self.debug_font = ImageFont.truetype(font_name, 16 * self.upscale)
                break
            except OSError:
                pass

    def _apply_rom_hacks(self) -> None:
        rom_hack_fast_bicycle(self.pyboy)
        rom_hack_fast_text(self.pyboy)
        rom_hack_fast_battles(self.pyboy)

    def _apply_hooks(self) -> None:
        hooks_data = [
            ["ScrollTextUpOneLine.WaitFrame", self._hook_callback_return_step, "ScrollTextUpOneLine.WaitFrame"],
            ["WaitForTextScrollButtonPress", self._hook_callback_return_step, "WaitForTextScrollButtonPress"],
            ["PlaceMenuCursor", self._hook_callback_menu_place_cursor, "PlaceMenuCursor"],
            ["EraseMenuCursor", self._hook_callback_menu_erase_cursor, "EraseMenuCursor"],
            ["TextBoxBorder", self._hook_callback_textbox, "TextBoxBorder"],
            ["UpdateSprites", self._hook_callback_update_sprite, "UpdateSprites"],
            ["OverworldLoopLessDelay.notSimulating", self._hook_callback_overworld_text_end, "OverworldLoopLessDelay.notSimulating"],
            ["CollisionCheckOnLand", self._hook_callback_collision, "CollisionCheckOnLand"],
            ["CheckWarpsNoCollision", self._hook_callback_nocollision, "CheckWarpsNoCollision"],
            ["GBFadeOutToBlack", self._hook_callback_exit_map, "GBFadeOutToBlack"],
            ["HandleLedges.foundMatch", self._hook_callback_ledge_jump, "HandleLedges.foundMatch"],
            ["_InitBattleCommon", self._hook_callback_start_battle, "_InitBattleCommon"],
        ]
        for hd in hooks_data:
            try:
                self.pyboy.hook_register(*([None]+hd[:3] if isinstance(hd[0],str) else hd[:4]))
            except ValueError:
                pass
    
    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.memory[addr]

    def read_bit(self, addr: str | int, bit: int) -> bool:
        return bool(int(self.read_m(addr)) & (1 << bit))
    
    def _print(self, *args):
        if self.verbose:
            print(*args)
    def _print_hook(self, context, *args):
        if self.verbose:
            print(f"\t{context:23.23} at step {self.pyboy.frame_count:d}",**args)

    def _is_in_battle(self) -> bool:
        return self.pyboy.memory[0xD057] > 0

    def _hook_callback_print(self, context: str = "") -> bool:
        print(f"\t+++\t{context:23.23} at step {self.pyboy.frame_count:d}")
        return True

    def _hook_callback_return_step(self, context: str = "") -> bool:
        self.disable_hooks = True
        self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_extra_ticks(self, context: list) -> bool:
        self.disable_hooks = True
        self.extra_ticks = context[0]
        self.return_step = 1
        self._print_hook(context[0],context[1])
        return True

    def _hook_callback_menu_place_cursor(self, context: str = "") -> bool:
        self.disable_hooks = True
        self.return_step = 1
        self.extra_ticks = 2
        self._print_hook(context)
        return True

    def _hook_callback_menu_erase_cursor(self, context: str = "") -> bool:
        self.disable_hooks = False
        self.return_step = 0
        self._print_hook(context)
        return True

    def _hook_callback_textbox(self, context: str = "") -> bool:
        if self.disable_hooks or self._is_in_battle():
            return False
        self.state = 2
        self.disable_hooks = True
        self.extra_ticks = 20
        self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_update_sprite(self, context: str = "") -> bool:
        self.last_sprite_update = self.pyboy.frame_count
        return True

    def _hook_callback_overworld_text_end(self, context: str = "") -> bool:
        if self.disable_hooks:
            return False
        if self.state == 2:
            self.state = 0
            self.disable_hooks = True
            self.extra_ticks = 0
            self.return_step = 1
            self._print_hook(context)
        return True

    def _hook_callback_collision(self, context: str = "") -> bool:
        if self.disable_hooks:
            return False
        self.extra_ticks = 2
        if self.state != 1:
            self.state = 1
            self.return_step = 0
            self._print_hook(context)
        else:
            self.state = 0
            self.disable_hooks = True
            self.return_step = 1
        return True

    def _hook_callback_nocollision(self, context: str = "") -> bool:
        if self.disable_hooks:
            return False
        self.state = 0
        if (self.pyboy.frame_count - self.last_sprite_update) < 2:
            self.extra_ticks = 1
            self.disable_hooks = True
            self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_exit_map(self, context: str = "") -> bool:
        self.state = 0
        self.disable_hooks = True
        self.return_step = 2
        self.extra_ticks = 3
        self.delayed_ticks = 72
        self._print_hook(context)
        return True

    def _hook_callback_ledge_jump(self, context: str = "") -> bool:
        self.state = 0
        self.disable_hooks = True
        self.return_step = 2
        one_turn_ledge = True
        if one_turn_ledge:
            self.extra_ticks = 48 - self.button_limit_dict[self.state]
            self.delayed_ticks = 0
        else:
            self.extra_ticks = 24 - self.button_limit_dict[self.state]
            self.delayed_ticks = 16
        self._print_hook(context)
        return True

    def _hook_callback_start_battle(self, context: str = "") -> bool:
        self.state = 4
        return True

    def _hook_callback_to_overworld(self, context: str = "") -> bool:
        self.state = 0
        return True

    def _apply_cheats(self) -> None:
        for func in self.cheats_funcs_ptr:
            func(self.pyboy)

    def read_bit(self, addr: str | int, bit: int) -> bool:
        return bool(int(self.read_m(addr)) & (1 << bit))
    
    def step(self, act) -> bool:
        print(f'pyboy_step_handler.py: step(): act={act}')
        self.last_action = act
        step_frame_count = self.pyboy.frame_count
        if self.delayed_ticks > 0:
            self.pyboy.tick(self.delayed_ticks, True)
        self.disable_hooks = False
        self.return_step = 0
        self.extra_ticks = 0
        self.delayed_ticks = 0
        expected_button_duration = self.button_duration_dict.get(self.state, 5)
        expected_button_limit = self.button_limit_dict.get(self.state, 7)
        if act not in {"n", -1}:
            self.pyboy.send_input(act)
            self.pyboy.send_input(VALID_RELEASE_ACTIONS[VALID_ACTIONS.index(act)], delay=expected_button_duration)
        expected_action_freq = self.action_freq_dict.get(self.state, 24)
        for i in range(expected_action_freq - 1):
            self._apply_cheats()
            self.pyboy.tick(1, self.rendering_debug)
            if self.return_step > 1:
                ret = self.pyboy.tick(self.extra_ticks, True)
                self._apply_cheats()
                return ret
            if i > expected_button_limit and self.return_step == 1:
                break
        for _ in range(int(self.extra_ticks)):
            self._apply_cheats()
            ret = self.pyboy.tick(1, self.rendering_debug)
        self._apply_cheats()
        ret = self.pyboy.tick(1, True)
        self._apply_cheats()
        self.last_step_ticks = self.pyboy.frame_count - step_frame_count

        has_flash_bool = self.check_if_party_has_hm(0xC8)
        print(f'has_flash_bool={has_flash_bool}')
        
        if self.read_bit(0xD803, 0):
            print(f'aaaaaaaaaa self.read_bit(0xD803, 0)={self.read_bit(0xD803, 0)}')
            if self.auto_teach_cut and not self.check_if_party_has_hm(0x0F):
                print(f'taught hm cut ttttttttttttttt')
                pyboy_binding.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            if self.auto_use_cut:
                print(f'calling self.cut_if_next() in step_handler.py')
                pyboy_binding.cut_if_next()

        if self.read_bit(0xD78E, 0):
            print(f' ffffffffff self.read_bit(0xD78E, 0)={self.read_bit(0xD78E, 0)}')
            if pyboy_binding.auto_teach_surf and not pyboy_binding.check_if_party_has_hm(0x39):
                pyboy_binding.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if self.auto_use_surf:
                pyboy_binding.surf_if_attempt(VALID_ACTIONS[act])

        # if self.read_bit(0xD857, 0):
        #     if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
        #         self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            # if self.auto_solve_strength_puzzles:
            #     self.solve_missable_strength_puzzle()
            #     self.solve_switch_strength_puzzle()

        if self.read_bit(0xD76C, 0) and self.auto_pokeflute:
            print(f' ffffffffff self.read_bit(0xD76C, 0)={self.read_bit(0xD76C, 0)}')
            pyboy_binding.use_pokeflute()

        if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0 and self.skip_rocket_hideout_bool:
            print(f' ggggggggggggg ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]={ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]}')
            pyboy_binding.skip_rocket_hideout()

        if self.skip_silph_co_bool and int(self.read_bit(0xD76C, 0)) != 0:  # has poke flute
            print(f' hhhhhhhhhh self.read_bit(0xD76C, 0)={self.read_bit(0xD76C, 0)}')
            pyboy_binding.skip_silph_co()
        
        return ret

    def screen_ndarray(self) -> np.ndarray:
        return self.pyboy.screen.ndarray[:, :, :3]

    def screen_pil(self) -> Image:
        return self.pyboy.screen.image

    def apply_debug_to_pil_image(self, img_pil: Image) -> Image:
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle((0, 0, 160 * self.upscale - 1, 16 * self.upscale - 1), fill=(255, 255, 255, 255))
        draw.text((1, 1), f" {self.last_action[0]:1s} {self.state:01d} {self.last_step_ticks:03d} {self.pyboy.frame_count:07d}",
            font=self.debug_font,fill=(0, 0, 0, 255))
        return img_pil

    def screen_debug(self) -> np.ndarray:
        return np.array(self.apply_debug_to_pil_image(self.screen_pil()), dtype=np.uint8, order="C")[:,:,:3]

    def reset_gif_frames(self) -> None:
        self.gif_frames.clear()

    def add_gif_frame(self) -> None:
        self.gif_frames.append(self.screen_debug())

    def save_gif(self, outfile_or_buff: Union[str, BytesIO, None] = None, return_buff: bool = True,
        delete_old: bool = True, speedup: int = 4, loop: bool = False
    ) -> Union[bool, BytesIO]:
        if speedup < 1:
            used_speedup = 1 if len(self.gif_frames) < 200 else 4
        else:
            used_speedup = int(speedup)
        for _ in range((4 * used_speedup) - 1):
            self.add_gif_frame()
        ret=generate_gif_from_numpy(self.gif_frames, outfile_or_buff, return_buff, 1000 * 24 / 60. / used_speedup, loop)
        if delete_old:
            self.reset_gif_frames()
        return ret

    def save_run_gif(self, delete_old: bool = True) -> None:
        if self.log_screen:
            self.save_gif(f"{os.path.realpath(sys.path[0])}{os.sep}run_t{int(time.time()):d}.gif", delete_old=delete_old, speedup=1)

    def run_action_on_emulator(self, action=-1) -> bool:
        print(f'pyboy_step_handler.py: run_action_on_emulator: action={action}')
        ret = self.step(action)
        # Add other actions below
        # self.cut_if_next()
        pyboy_binding.cut_if_next(self)
        if self.log_screen:
            self.add_gif_frame()
        return ret
    
    def check_if_party_has_hm(self, hm: int) -> bool:
        party_size = self.read_m("wPartyCount")
        print(f'497 party_size={party_size}')
        for i in range(party_size):
            # PRET 1-indexes
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            print(f'i={i} addr={addr}')
            if hm in self.pyboy.memory[addr : addr + 4]:
                return True
        return False

    # def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
    #     party_size = self.read_m("wPartyCount")
    #     for i in range(party_size):
    #         _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
    #         poke = self.pyboy.memory[species_addr]
    #         if poke in pokemon_species_ids:
    #             for slot in range(4):
    #                 if self.read_m(f"wPartyMon{i+1}Moves") not in {0xF, 0x13, 0x39, 0x46, 0x94}:
    #                     _, move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
    #                     _, pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
    #                     self.pyboy.memory[move_addr + slot] = tmhm
    #                     self.pyboy.memory[pp_addr + slot] = pp
    #                     break
    
    # def cut_if_next(self):
    #     print(f'pyboy_step_handler.py: cut_if_next() executing')
    #     in_erika_gym = self.read_m("wCurMapTileset") == Tilesets.GYM.value
    #     in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
    #     if in_erika_gym or in_overworld:
    #         _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
    #         tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
    #         tileMap = np.array(tileMap, dtype=np.uint8)
    #         tileMap = np.reshape(tileMap, (18, 20))
    #         y, x = 8, 8
    #         up, down, left, right = (
    #             tileMap[y - 2 : y, x : x + 2],  # up
    #             tileMap[y + 2 : y + 4, x : x + 2],  # down
    #             tileMap[y : y + 2, x - 2 : x],  # left
    #             tileMap[y : y + 2, x + 2 : x + 4],  # right
    #         )

    #         if (in_overworld and 0x3D in up) or (in_erika_gym and 0x50 in up):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         else:
    #             return

    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
    #         self.pyboy.tick(self.action_freq, render=True)
            
    #         for _ in range(24):
    #             if self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
    #                 break
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #         self.pyboy.tick(self.action_freq, render=True)

    #         for _ in range(7):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #             party_mon = self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
    #             _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
    #             if 0xF in self.pyboy.memory[addr : addr + 4]:
    #                 break

    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #         self.pyboy.tick(4 * self.action_freq, render=True)

    #         _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
    #         field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

    #         for _ in range(10):
    #             current_item = self.read_m("wCurrentMenuItem")
    #             if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
    #                 break
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)

    #         for _ in range(5):
    #             self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #             self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #             self.pyboy.tick(4 * self.action_freq, render=True)
    
    # def surf_if_attempt(self, action: WindowEvent):
    #     if not (
    #         self.read_m("wWalkBikeSurfState") != 2
    #         and self.check_if_party_has_hm(0x39)
    #         and action
    #         in [
    #             WindowEvent.PRESS_ARROW_DOWN,
    #             WindowEvent.PRESS_ARROW_LEFT,
    #             WindowEvent.PRESS_ARROW_RIGHT,
    #             WindowEvent.PRESS_ARROW_UP,
    #         ]
    #     ):
    #         return

    #     in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
    #     in_plateau = self.read_m("wCurMapTileset") == Tilesets.PLATEAU.value
    #     if in_overworld or in_plateau:
    #         _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
    #         tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
    #         tileMap = np.array(tileMap, dtype=np.uint8)
    #         tileMap = np.reshape(tileMap, (18, 20))
    #         y, x = 8, 8
    #         up, down, left, right = (
    #             tileMap[y - 2 : y, x : x + 2],  # up
    #             tileMap[y + 2 : y + 4, x : x + 2],  # down
    #             tileMap[y : y + 2, x - 2 : x],  # left
    #             tileMap[y : y + 2, x + 2 : x + 4],  # right
    #         )

    #         direction = self.read_m("wSpritePlayerStateData1FacingDirection")

    #         if not (
    #             (direction == 0x4 and action == WindowEvent.PRESS_ARROW_UP and 0x14 in up)
    #             or (direction == 0x0 and action == WindowEvent.PRESS_ARROW_DOWN and 0x14 in down)
    #             or (direction == 0x8 and action == WindowEvent.PRESS_ARROW_LEFT and 0x14 in left)
    #             or (direction == 0xC and action == WindowEvent.PRESS_ARROW_RIGHT and 0x14 in right)
    #         ):
    #             return

    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
    #         self.pyboy.tick(self.action_freq, render=True)
            
    #         for _ in range(24):
    #             if self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
    #                 break
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #         self.pyboy.tick(self.action_freq, render=True)

    #         for _ in range(7):
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #             party_mon = self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
    #             _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
    #             if 0x39 in self.pyboy.memory[addr : addr + 4]:
    #                 break

    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #         self.pyboy.tick(4 * self.action_freq, render=True)

    #         _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
    #         field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

    #         for _ in range(10):
    #             current_item = self.read_m("wCurrentMenuItem")
    #             if current_item < 4 and field_moves[current_item] in (
    #                 FieldMoves.SURF.value,
    #                 FieldMoves.SURF_2.value,
    #             ):
    #                 break
    #             self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    #             self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
    #             self.pyboy.tick(self.action_freq, render=True)

    #         for _ in range(5):
    #             self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #             self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
    #             self.pyboy.tick(4 * self.action_freq, render=True)
                
    # def use_pokeflute(self):
    #     in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
    #     if in_overworld:
    #         _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
    #         bag_items = self.pyboy.memory[wBagItems : wBagItems + 40]
    #         if ItemsThatGuy.POKE_FLUTE.value not in bag_items[::2]:
    #             return
    #         pokeflute_index = bag_items[::2].index(ItemsThatGuy.POKE_FLUTE.value)

    #         coords = self.get_game_coords()
    #         if coords == (9, 62, 23):
    #             self.pyboy.button("RIGHT", 8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif coords == (10, 63, 23):
    #             self.pyboy.button("UP", 8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif coords == (10, 61, 23):
    #             self.pyboy.button("DOWN", 8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif coords == (27, 10, 27):
    #             self.pyboy.button("LEFT", 8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         elif coords == (27, 10, 25):
    #             self.pyboy.button("RIGHT", 8)
    #             self.pyboy.tick(self.action_freq, render=True)
    #         else:
    #             return

    #         _, wMissableObjectFlags = self.pyboy.symbol_lookup("wMissableObjectFlags")
    #         _, wMissableObjectList = self.pyboy.symbol_lookup("wMissableObjectList")
    #         missable_objects_list = self.pyboy.memory[
    #             wMissableObjectList : wMissableObjectList + 34
    #         ]
    #         missable_objects_list = missable_objects_list[: missable_objects_list.index(0xFF)]
    #         missable_objects_sprite_ids = missable_objects_list[::2]
    #         missable_objects_flags = missable_objects_list[1::2]
    #         for sprite_id in missable_objects_sprite_ids:
    #             picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
    #             flags_bit = missable_objects_flags[missable_objects_sprite_ids.index(sprite_id)]
    #             flags_byte = flags_bit // 8
    #             flag_bit = flags_bit % 8
    #             flag_byte_value = self.read_bit(wMissableObjectFlags + flags_byte, flag_bit)
    #             if picture_id == 0x43 and not flag_byte_value:
    #                 self.pyboy.button("START", 8)
    #                 self.pyboy.tick(self.action_freq, render=True)

    #                 for _ in range(24):
    #                     if self.read_m("wCurrentMenuItem") == 2:
    #                         break
    #                     self.pyboy.button("DOWN", 8)
    #                     self.pyboy.tick(self.action_freq, render=True)
    #                 self.pyboy.button("A", 8)
    #                 self.pyboy.tick(self.action_freq, render=True)

    #                 for _ in range(20):
    #                     self.pyboy.button("UP", 8)
    #                     self.pyboy.tick(self.action_freq, render=True)

    #                 for _ in range(21):
    #                     if (
    #                         self.read_m("wCurrentMenuItem") + self.read_m("wListScrollOffset")
    #                         == pokeflute_index
    #                     ):
    #                         break
    #                     self.pyboy.button("DOWN", 8)
    #                     self.pyboy.tick(self.action_freq, render=True)

    #                 for _ in range(5):
    #                     self.pyboy.button("A", 8)
    #                     self.pyboy.tick(4 * self.action_freq, render=True)

    #                 break

    # def skip_rocket_hideout(self):
    #     r, c, map_n = self.get_game_coords()
    #     current_value = self.pyboy.memory[0xD81B]
    #     self.pyboy.memory[0xD81B] = current_value | (1 << 7)
    #     try:
    #         if self.skip_rocket_hideout_bool:    
    #             if c == 5 and r in list(range(11, 18)) and map_n == 135:
    #                 for _ in range(10):
    #                     self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    #                     self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
    #                     self.pyboy.tick(7 * self.action_freq, render=True)
    #             if c == 5 and r == 17 and map_n == 135:
    #                 self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    #                 self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
    #                 self.pyboy.tick(self.action_freq, render=True)
    #     except Exception as e:
    #             # logging.info(f'env_id: {self.env_id} had exception in skip_rocket_hideout in run_action_on_emulator. error={e}')
    #             pass

    # def skip_silph_co(self):
    #     r, c, map_n = self.get_game_coords()
    #     current_value = self.pyboy.memory[0xD81B]
    #     self.pyboy.memory[0xD81B] = current_value | (1 << 7)
    #     self.pyboy.memory[0xD838] = current_value | (1 << 5)
    #     try:
    #         if self.skip_silph_co_bool:
    #             if c == 0x12 and r == 0x16 and map_n == 10:
    #                 for _ in range(10):
    #                     self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    #                     self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
    #                     self.pyboy.tick(7 * self.action_freq, render=True)
    #             if c == 0x12 and r == 0x16 and map_n == 10:
    #                 self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    #                 self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
    #                 self.pyboy.tick(self.action_freq, render=True)
    #     except Exception as e:
    #             # logging.info(f'env_id: {self.env_id} had exception in skip_silph_co in run_action_on_emulator. error={e}')
    #             pass

    # def get_game_coords(self):
    #     return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    # def read_m(self, addr: str | int) -> int:
    #     if isinstance(addr, str):
    #         return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
    #     return self.pyboy.memory[addr]

    # def read_bit(self, addr: str | int, bit: int) -> bool:
    #     return bool(int(self.read_m(addr)) & (1 << bit))


# # pyboy_step_handler.py ## OLD BELOW

# from typing import Union
# import warnings
# import sys
# import os
# import time
# from io import BytesIO
# from pyboy.utils import WindowEvent
# from pokegym.pyboy_binding import GoBetween

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category = DeprecationWarning)
#     import numpy as np
#     from PIL import Image, ImageFont, ImageDraw
#     from pyboy import pyboy

# VALID_ACTIONS = [
#     WindowEvent.PRESS_ARROW_DOWN,
#     WindowEvent.PRESS_ARROW_LEFT,
#     WindowEvent.PRESS_ARROW_RIGHT,
#     WindowEvent.PRESS_ARROW_UP,
#     WindowEvent.PRESS_BUTTON_A,
#     WindowEvent.PRESS_BUTTON_B,
#     WindowEvent.PRESS_BUTTON_START,
# ]

# VALID_RELEASE_ACTIONS = [
#     WindowEvent.RELEASE_ARROW_DOWN,
#     WindowEvent.RELEASE_ARROW_LEFT,
#     WindowEvent.RELEASE_ARROW_RIGHT,
#     WindowEvent.RELEASE_ARROW_UP,
#     WindowEvent.RELEASE_BUTTON_A,
#     WindowEvent.RELEASE_BUTTON_B,
#     WindowEvent.RELEASE_BUTTON_START,
# ]

# VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

# def addr_to_opcodes_list(addr: int)->list:
#     return np.array([addr],dtype = np.uint16).view(np.uint8).tolist()

# def cheat_bicycle(pb: pyboy)->None:
#     if pb.memory[0xD700] == 0x00:
#         pb.memory[0xD700] = 0x01

# def cheat_battle_fastest_animations_styles_and_text_configs(pb: pyboy)->None:
#     pb.memory[0xD355] = (pb.memory[0xD355] & 0x38) | 0xC0
#     pb.memory[0xD358] = 0x00

# def rom_hack_fast_bicycle(pb: pyboy)->None:
#     (bank,addr)=(0x00,0x0D27)
#     opcodes=addr_to_opcodes_list(addr)
#     asm_jp_AdvancePlayerSprite=[0xC3]+opcodes
#     asm_call_AdvancePlayerSprite=[0xCD]+opcodes
#     (bank,addr)=(0x00,0x06A0)
#     addr+=5
#     for opc in [asm_call_AdvancePlayerSprite]*2+[asm_jp_AdvancePlayerSprite]:
#         pb.memory[bank,addr:addr+len(opc)]=opc
#         addr+=len(opc)
#     (_,limit_addr)=(0x00,0x06B4)
#     assert addr<limit_addr

# def rom_hack_fast_text(pb: pyboy)->None:
#     (bank,addr)=(0x00,0x1b33)
#     pb.memory[bank,addr]=0xC9
    
# def rom_hack_fast_battles(pb: pyboy)->None:
#     for bank,addr in [
#         [0x00,0x23B1],
#         [0x03,0x7A1D],
#         [0x1C,0x496D],
#         [0x1E,0x40F1],
#         [0x1E,0x417C],
#         [0x1E,0x4D5E],
#         [0x1E,0x5E6A],
#         [0x1E,0x5E6A],
#     ]:
#         pb.memory[bank,addr]=0xC9

# sys.dont_write_bytecode = True

# __all__ = ["PyBoyStepHandlerPokeRed"]

# def generate_gif_from_numpy(np_imgs: list, outfile_or_buff: Union[str, BytesIO, None] = None,
#     return_buff: bool = True, frame_duration: int = 200, loop: bool = False,
# ) -> Union[bool, BytesIO]:
#     if np_imgs is None or len(np_imgs) < 1:
#         return False
#     frames = []
#     for img in np_imgs:
#         try:
#             frames.append(Image.fromarray(img))
#         except (AttributeError, ValueError, OSError):
#             pass
#     buff = BytesIO() if outfile_or_buff is None else outfile_or_buff
#     if len(frames) > 0:
#         frames[0].save(buff, format = "GIF", optimize = True, append_images = frames,
#             save_all = True, duration = max(8, int(frame_duration)), loop = 1 if loop else 0)
#     if isinstance(buff, BytesIO):
#         buff.seek(0)
#     return buff if outfile_or_buff is None or (return_buff and isinstance(outfile_or_buff, BytesIO)) else len(frames) > 0

# class PyBoyStepHandlerPokeRed:
#     def __init__(self, pb: pyboy, go_between: GoBetween, verbose: bool = False, log_screen: bool = False):
#         self.pyboy = pb
#         self.go_between = go_between  # Save reference to GoBetween instance
#         if not isinstance(pb, pyboy):
#             self._initialize_pyboy(pb, True)
#         assert isinstance(self.pyboy, pyboy)
#         self._configure_pyboy()
#         self._apply_rom_hacks()
#         self._apply_hooks()
#         self.action_freq_dict = {0: 24, 1: 24, 2: 12, 3: 12, 4: 30}
#         self.button_duration_dict = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}
#         self.button_limit_dict = {0: 5, 1: 7, 2: 7, 3: 5, 4: 5}
#         self.last_step_ticks = 0
#         self.last_action = "n"
#         self.state = 0
#         self.disable_hooks = False
#         self.last_sprite_update = 0
#         self.return_step = 0
#         self.extra_ticks = 0
#         self.rendering_debug = False
#         self.delayed_ticks = 0
#         self.cheats_funcs_ptr = [
#             cheat_bicycle,
#             cheat_battle_fastest_animations_styles_and_text_configs,
#         ]
#         self.verbose = verbose
#         self.log_screen = log_screen
#         self.gif_frames = []
#         self.upscale = 1
#         self.debug_font = None
#         self.update_font()

#     def _stop_pyboy(self, save:bool = False) -> None:
#         if isinstance(self.pyboy, pyboy) and hasattr(self.pyboy, "stop"):
#             self.pyboy.stop(save)
#     def close(self) -> None:
#         self._stop_pyboy()

#     def _initialize_pyboy(self, gamerom: Union[str, dict], headless: bool = True) -> None:
#         """Initialize a pyboy instance."""
#         self._stop_pyboy()
#         assert isinstance(gamerom, (dict, str))
#         pyboy_kwargs = (
#             gamerom
#             if isinstance(gamerom, dict)
#             else {
#                 "gamerom": gamerom,
#                 "window": "null" if headless else "SDL2",
#                 "log_level": "ERROR",
#                 "symbols": os.path.join(os.path.dirname(__file__), "pokered.sym"),
#             }
#         )
#         self.pyboy = pyboy(**pyboy_kwargs)
    
#     def _configure_pyboy(self) -> None:
#         self.pyboy.set_emulation_speed(0)

#     def set_seed(self, seed: Union[int, None] = None) -> None:
#         if seed is not None:
#             self.pyboy.memory[0xFF04] = seed % 0x100

#     def save_state(self, file_like_object) -> int:
#         return self.pyboy.save_state(file_like_object)

#     def load_state(self, file_like_object) -> int:
#         if isinstance(file_like_object,str) and len(file_like_object)<0x1000:
#             ret = False 
#             with open(file_like_object,mode="rb") as f:
#                 ret = self.pyboy.load_state(f)
#         else:
#             ret = self.pyboy.load_state(file_like_object)
#         self._apply_rom_hacks()
#         self._apply_cheats()
#         self.reset_gif_frames()
#         return ret

#     def update_font(self, upscale:int = 1) -> None:
#         self.upscale = upscale
#         self.debug_font = None
#         allowed_fonts = [
#             "OCRAEXT.TTF",
#             "CascadiaMono.ttf",
#             "consolab.ttf",
#             "Lucida-Console.ttf",
#             "couri.ttf",
#         ]
#         for font_name in allowed_fonts:
#             try:
#                 self.debug_font = ImageFont.truetype(font_name, 16 * self.upscale)
#                 break
#             except OSError:
#                 pass

#     def _apply_rom_hacks(self) -> None:
#         rom_hack_fast_bicycle(self.pyboy)
#         rom_hack_fast_text(self.pyboy)
#         rom_hack_fast_battles(self.pyboy)

#     def _apply_hooks(self) -> None:
#         hooks_data = [
#             ["ScrollTextUpOneLine.WaitFrame", self._hook_callback_return_step, "ScrollTextUpOneLine.WaitFrame"],
#             ["WaitForTextScrollButtonPress", self._hook_callback_return_step, "WaitForTextScrollButtonPress"],
#             ["PlaceMenuCursor", self._hook_callback_menu_place_cursor, "PlaceMenuCursor"],
#             ["EraseMenuCursor", self._hook_callback_menu_erase_cursor, "EraseMenuCursor"],
#             ["TextBoxBorder", self._hook_callback_textbox, "TextBoxBorder"],
#             ["UpdateSprites", self._hook_callback_update_sprite, "UpdateSprites"],
#             ["OverworldLoopLessDelay.notSimulating", self._hook_callback_overworld_text_end, "OverworldLoopLessDelay.notSimulating"],
#             ["CollisionCheckOnLand", self._hook_callback_collision, "CollisionCheckOnLand"],
#             ["CheckWarpsNoCollision", self._hook_callback_nocollision, "CheckWarpsNoCollision"],
#             ["GBFadeOutToBlack", self._hook_callback_exit_map, "GBFadeOutToBlack"],
#             ["HandleLedges.foundMatch", self._hook_callback_ledge_jump, "HandleLedges.foundMatch"],
#             ["_InitBattleCommon", self._hook_callback_start_battle, "_InitBattleCommon"],
#         ]
#         for hd in hooks_data:
#             try:
#                 self.pyboy.hook_register(*([None]+hd[:3] if isinstance(hd[0],str) else hd[:4]))
#             except ValueError:
#                 pass

#     def _print(self, *args):
#         if self.verbose:
#             print(*args)
#     def _print_hook(self, context, *args):
#         if self.verbose:
#             print(f"\t{context:23.23} at step {self.pyboy.frame_count:d}",**args)

#     def _is_in_battle(self) -> bool:
#         return self.pyboy.memory[0xD057] > 0

#     def _hook_callback_print(self, context: str = "") -> bool:
#         print(f"\t+++\t{context:23.23} at step {self.pyboy.frame_count:d}")
#         return True

#     def _hook_callback_return_step(self, context: str = "") -> bool:
#         self.disable_hooks = True
#         self.return_step = 1
#         self._print_hook(context)
#         return True

#     def _hook_callback_extra_ticks(self, context: list) -> bool:
#         self.disable_hooks = True
#         self.extra_ticks = context[0]
#         self.return_step = 1
#         self._print_hook(context[0],context[1])
#         return True

#     def _hook_callback_menu_place_cursor(self, context: str = "") -> bool:
#         self.disable_hooks = True
#         self.return_step = 1
#         self.extra_ticks = 2
#         self._print_hook(context)
#         return True

#     def _hook_callback_menu_erase_cursor(self, context: str = "") -> bool:
#         self.disable_hooks = False
#         self.return_step = 0
#         self._print_hook(context)
#         return True

#     def _hook_callback_textbox(self, context: str = "") -> bool:
#         if self.disable_hooks or self._is_in_battle():
#             return False
#         self.state = 2
#         self.disable_hooks = True
#         self.extra_ticks = 20
#         self.return_step = 1
#         self._print_hook(context)
#         return True

#     def _hook_callback_update_sprite(self, context: str = "") -> bool:
#         self.last_sprite_update = self.pyboy.frame_count
#         return True

#     def _hook_callback_overworld_text_end(self, context: str = "") -> bool:
#         if self.disable_hooks:
#             return False
#         if self.state == 2:
#             self.state = 0
#             self.disable_hooks = True
#             self.extra_ticks = 0
#             self.return_step = 1
#             self._print_hook(context)
#         return True

#     def _hook_callback_collision(self, context: str = "") -> bool:
#         if self.disable_hooks:
#             return False
#         self.extra_ticks = 2
#         if self.state != 1:
#             self.state = 1
#             self.return_step = 0
#             self._print_hook(context)
#         else:
#             self.state = 0
#             self.disable_hooks = True
#             self.return_step = 1
#         return True

#     def _hook_callback_nocollision(self, context: str = "") -> bool:
#         if self.disable_hooks:
#             return False
#         self.state = 0
#         if (self.pyboy.frame_count - self.last_sprite_update) < 2:
#             self.extra_ticks = 1
#             self.disable_hooks = True
#             self.return_step = 1
#         self._print_hook(context)
#         return True

#     def _hook_callback_exit_map(self, context: str = "") -> bool:
#         self.state = 0
#         self.disable_hooks = True
#         self.return_step = 2
#         self.extra_ticks = 3
#         self.delayed_ticks = 72
#         self._print_hook(context)
#         return True

#     def _hook_callback_ledge_jump(self, context: str = "") -> bool:
#         self.state = 0
#         self.disable_hooks = True
#         self.return_step = 2
#         one_turn_ledge = True
#         if one_turn_ledge:
#             self.extra_ticks = 48 - self.button_limit_dict[self.state]
#             self.delayed_ticks = 0
#         else:
#             self.extra_ticks = 24 - self.button_limit_dict[self.state]
#             self.delayed_ticks = 16
#         self._print_hook(context)
#         return True

#     def _hook_callback_start_battle(self, context: str = "") -> bool:
#         self.state = 4
#         return True

#     def _hook_callback_to_overworld(self, context: str = "") -> bool:
#         self.state = 0
#         return True

#     def _apply_cheats(self) -> None:
#         for func in self.cheats_funcs_ptr:
#             func(self.pyboy)

#     def step(self, act) -> bool:
#         self.last_action = act
#         step_frame_count = self.pyboy.frame_count
#         if self.delayed_ticks > 0:
#             self.pyboy.tick(self.delayed_ticks, True)
#         self.disable_hooks = False
#         self.return_step = 0
#         self.extra_ticks = 0
#         self.delayed_ticks = 0
#         expected_button_duration = self.button_duration_dict.get(self.state, 5)
#         expected_button_limit = self.button_limit_dict.get(self.state, 7)
#         if act not in {"n", -1}:
#             self.pyboy.send_input(act)
#             self.pyboy.send_input(VALID_RELEASE_ACTIONS[VALID_ACTIONS.index(act)], delay=expected_button_duration)
#         expected_action_freq = self.action_freq_dict.get(self.state, 24)
#         for i in range(expected_action_freq - 1):
#             self._apply_cheats()
#             self.pyboy.tick(1, self.rendering_debug)
#             if self.return_step > 1:
#                 ret = self.pyboy.tick(self.extra_ticks, True)
#                 self._apply_cheats()
#                 return ret
#             if i > expected_button_limit and self.return_step == 1:
#                 break
#         for _ in range(int(self.extra_ticks)):
#             self._apply_cheats()
#             ret = self.pyboy.tick(1, self.rendering_debug)
#         self._apply_cheats()
#         ret = self.pyboy.tick(1, True)
#         self._apply_cheats()
#         self.last_step_ticks = self.pyboy.frame_count - step_frame_count
#         return ret


#     def screen_ndarray(self) -> np.ndarray:
#         return self.pyboy.screen.ndarray[:, :, :3]

#     def screen_pil(self) -> Image:
#         return self.pyboy.screen.image

#     def apply_debug_to_pil_image(self, img_pil: Image) -> Image:
#         draw = ImageDraw.Draw(img_pil)
#         draw.rectangle((0, 0, 160 * self.upscale - 1, 16 * self.upscale - 1), fill=(255, 255, 255, 255))
#         draw.text((1, 1), f" {self.last_action[0]:1s} {self.state:01d} {self.last_step_ticks:03d} {self.pyboy.frame_count:07d}",
#             font=self.debug_font,fill=(0, 0, 0, 255))
#         return img_pil

#     def screen_debug(self) -> np.ndarray:
#         return np.array(self.apply_debug_to_pil_image(self.screen_pil()), dtype=np.uint8, order="C")[:,:,:3]

#     def reset_gif_frames(self) -> None:
#         self.gif_frames.clear()

#     def add_gif_frame(self) -> None:
#         self.gif_frames.append(self.screen_debug())

#     def save_gif(self, outfile_or_buff: Union[str, BytesIO, None] = None, return_buff: bool = True,
#         delete_old: bool = True, speedup: int = 4, loop: bool = False
#     ) -> Union[bool, BytesIO]:
#         if speedup < 1:
#             used_speedup = 1 if len(self.gif_frames) < 200 else 4
#         else:
#             used_speedup = int(speedup)
#         for _ in range((4 * used_speedup) - 1):
#             self.add_gif_frame()
#         ret=generate_gif_from_numpy(self.gif_frames, outfile_or_buff, return_buff, 1000 * 24 / 60. / used_speedup, loop)
#         if delete_old:
#             self.reset_gif_frames()
#         return ret

#     def save_run_gif(self, delete_old: bool = True) -> None:
#         if self.log_screen:
#             self.save_gif(f"{os.path.realpath(sys.path[0])}{os.sep}run_t{int(time.time()):d}.gif", delete_old=delete_old, speedup=1)

#     def run_action_on_emulator(self, action=-1) -> bool:
#         ret = self.step(action)
#         if self.log_screen:
#             self.add_gif_frame()
#         return ret