from enum import IntEnum

# Text Interactions
TEXT_ON_SCREEN = 0x8800
TEXT_FONT_ON_LOADED = 0xCFC4
TEXT_DST_POINTER = 0xCC3A
TEXT_BOX_ID = 0xD125

# No idea what this is but it took 3 days but it seems to uniquely identify pokemon menu in battle,
# which allows other values to ID battle text, a nearly impossible task (days and days to figure this out,
# fix me if you can find a better value like 0xCF34 but it doesn't seem to work in pyboy??)
UNKNOWN_D730 = 0xD730  

# General purpose working registers, use extreme caution when using this
PRE_DEF_ID = 0xCC4E

# Core menu navigation identification
TEXT_MENU_CURSOR_LOCATION = (0xCC30, 0xCC31)
TEXT_MENU_CURSOR_COUNTER_1 = 0xCC26 # Stale after interaction
TEXT_MENU_CURSOR_COUNTER_2 = 0xCC36

# Alt menu navigation identification
TEXT_MENU_TILE_BEHIND_CURSOR = 0xCC27 # Stale after interaction
TEXT_MENU_MAX_MENU_ITEM = 0xCC28 # Stale after interaction
TEXT_MENU_MENU_WATCHED_KEYS = 0xCC29 # Stale after interaction
TEXT_MENU_LAST_MENU_ITEM = 0xCC2A # Stale after interaction
LAST_CURSOR_POSITION_PARTY_BILLS_PC = 0xCC2B # Stale after interaction
LAST_CURSOR_POSITION_ITEM_SCREEN = 0xCC2C # Stale after interaction
LAST_CURSOR_POSITION_START_MENU = 0xCC2D # Stale after interaction


# Menu Navigation
# NOTE: This assumes you have a pokedex, which is the first menu item otherwise, otherwise it's off by 1. 
class RedRamMenuKeys:
    # MENU CURSOR POSITIONS
    MENU_CLEAR = (0x00, 0x00)  # Custom define for when no menu is active to initialize stale data
    START_MENU_POKEDEX = (0xd3, 0xc3)
    START_MENU_POKEMON = (0xfB, 0xc3)
    START_MENU_ITEM = (0x23, 0xc4)
    START_MENU_SELF = (0x4b, 0xc4)
    START_MENU_SAVE = (0x73, 0xc4)
    START_MENU_OPTION = (0x9b, 0xc4)
    START_MENU_QUIT = (0xc3, 0xc4)
    POKECENTER_HEAL = (0x4C, 0xC4)
    POKECENTER_CANCEL = (0x74, 0xC4)
    POKEMART_BUY = (0xB5, 0xC3)
    POKEMART_SELL = (0xDD, 0xC3)
    POKEMART_QUIT = (0x05, 0xC4)
    PC_SOMEONE = (0xC9, 0xC3)
    PC_SELF = (0xF1, 0xC3)
    PC_OAK = (0x19, 0xC4)
    PC_LOGOFF = (0x41, 0xC4)
    PC_SOMEONE_DEPOSIT_WITHDRAW = (0x9A, 0xC4)
    PC_SOMEONE_STATUS = (0xC2, 0xC4)
    PC_SOMEONE_CANCEL = (0xEA, 0xC4)
    BATTLE_MENU_FIGHT = (0xC1, 0xC4)
    BATTLE_MOVE_1 = (0xA9, 0xC4)
    BATTLE_MOVE_2 = (0xBD, 0xC4)
    BATTLE_MOVE_3 = (0xD1, 0xC4)
    BATTLE_MOVE_4 = (0xE5, 0xC4)
    BATTLE_MENU_PKMN = (0xC7, 0xC4)
    BATTLE_ROSTER_PKMN_1 = (0xB4, 0xC3)
    BATTLE_ROSTER_PKMN_2 = (0xDC, 0xC3)
    BATTLE_ROSTER_PKMN_3 = (0x04, 0xC4)
    BATTLE_ROSTER_PKMN_4 = (0x2C, 0xC4)
    BATTLE_ROSTER_PKMN_5 = (0x54, 0xC4)
    BATTLE_ROSTER_PKMN_6 = (0x7C, 0xC4)
    BATTLE_MON_SUB_SELECT_1 = (0x9C, 0xC4)  # In battle this is switch, in start menu this is stats
    BATTLE_MON_SUB_SELECT_2 = (0xC4, 0xC4)  # In battle this is stats, in start menu this is switch
    BATTLE_ROSTER_CANCEL = (0xEC, 0xC4)
    BATTLE_MENU_ITEM = (0xE9, 0xC4)
    BATTLE_MENU_ITEM_X_USE = (0x8A, 0xC4)
    BATTLE_MENU_ITEM_X_TOSS = (0xB2, 0xC4)
    BATTLE_MART_PC_ITEM_1 = (0xF5, 0xC3)
    BATTLE_MART_PC_ITEM_2 = (0x1D, 0xC4)
    BATTLE_MART_PC_ITEM_N = (0x45, 0xC4)
    BATTLE_MART_PC_ITEM_CANCEL = (0x69, 0x01)
    BATTLE_MENU_RUN = (0xEF, 0xC4)
    MENU_YES = (0x4F, 0xC4)
    MENU_NO = (0x77, 0xC4)
    # OVERWRITE_MOVE_1 = (0x45, 0xC4)  shared with BATTLE_MART_PC_ITEM_N, has to be overwritten
    OVERWRITE_MOVE_2 = (0x59, 0xC4)
    OVERWRITE_MOVE_3 = (0x6D, 0xC4)
    OVERWRITE_MOVE_4 = (0x81, 0xC4)
    # TODO: Get Nickname state and add it to filter list (0xC4C0 text nickname key hack?)

    MENU_NO_HACK = (0x69, 0xC4)

    POKEMON_MENU_N0 = (0xFA, 0xC3)
    POKEMON_MENU_N1 = (0x22, 0xC4)
    POKEMON_MENU_N2 = (0x4A, 0xC4)
    POKEMON_MENU_N3 = (0x72, 0xC4)
    POKEMON_MENU_B0 = (0xFC, 0xC3)
    POKEMON_MENU_B1 = (0x24, 0xC4)
    POKEMON_MENU_B2 = (0x4C, 0xC4)
    POKEMON_MENU_B3 = (0x74, 0xC4)


class RedRamMenuValues(IntEnum):
    UNKNOWN_MENU = 20
    START_MENU_POKEDEX = 21
    START_MENU_POKEMON = 22
    START_MENU_ITEM = 23
    START_MENU_SELF = 24
    START_MENU_SAVE = 25
    START_MENU_OPTION = 26
    START_MENU_QUIT = 27

    POKECENTER_HEAL = 28
    POKECENTER_CANCEL = 29

    POKEMART_BUY = 30
    POKEMART_SELL = 31
    POKEMART_QUIT = 32

    PC_SOMEONE = 33
    PC_SELF = 34
    PC_OAK = 35
    PC_LOGOFF = 36

    PC_SOMEONE_CONFIRM = 37
    PC_SOMEONE_STATUS = 38
    PC_SOMEONE_CANCEL = 39

    BATTLE_MENU_FIGHT = 40
    BATTLE_MOVE_1 = 41
    BATTLE_MOVE_2 = 42
    BATTLE_MOVE_3 = 43
    BATTLE_MOVE_4 = 44
    BATTLE_MENU_PKMN = 45
    SELECT_POKEMON_1 = 46
    SELECT_POKEMON_2 = 47
    SELECT_POKEMON_3 = 48
    SELECT_POKEMON_4 = 49
    SELECT_POKEMON_5 = 50
    SELECT_POKEMON_6 = 51
    MENU_SELECT_STATS = 52
    MENU_SELECT_SWITCH = 53
    MENU_SELECT_CANCEL = 54
    BATTLE_SELECT_STATS = 55
    BATTLE_SELECT_SWITCH = 56
    BATTLE_MENU_ITEM = 57
    BATTLE_MENU_ITEM_X_USE = 58
    BATTLE_MENU_ITEM_X_TOSS = 59
    BATTLE_MART_PC_ITEM = 60  # number intentionally left blank
    BATTLE_MART_PC_ITEM_CANCEL = 61
    BATTLE_MENU_RUN = 62

    MENU_YES = 63
    MENU_NO = 64

    PC_SOMEONE_WITHDRAW = 65
    PC_SOMEONE_DEPOSIT = 66
    PC_SOMEONE_RELEASE = 67
    PC_SOMEONE_CHANGE_BOX = 68
    PC_SOMEONE_EXIT = 69

    PC_SELF_WITHDRAW_ITEM = 70
    PC_SELF_DEPOSIT_ITEM = 71
    PC_SELF_TOSS_ITEM = 72
    PC_SELF_EXIT = 73

    ITEM_1 = 74
    ITEM_2 = 75
    ITEM_3 = 76
    ITEM_4 = 77
    ITEM_5 = 78
    ITEM_6 = 79
    ITEM_7 = 80
    ITEM_8 = 81
    ITEM_9 = 82
    ITEM_10 = 83
    ITEM_11 = 84
    ITEM_12 = 85
    ITEM_13 = 86
    ITEM_14 = 87
    ITEM_15 = 88
    ITEM_16 = 89
    ITEM_17 = 90
    ITEM_18 = 91
    ITEM_19 = 92
    ITEM_20 = 93
    ITEM_RANGE_ERROR = 94
    ITEM_QUANTITY = 95

    START_MENU_LOADING = 96

    OVERWRITE_MOVE_YES = 97
    OVERWRITE_MOVE_NO = 98
    ABANDON_MOVE_YES = 99
    ABANDON_MOVE_NO = 100
    OVERWRITE_MOVE_1 = 101
    OVERWRITE_MOVE_2 = 102
    OVERWRITE_MOVE_3 = 103
    OVERWRITE_MOVE_4 = 104

    NAME_POKEMON_YES = 105
    NAME_POKEMON_NO = 106
    SWITCH_POKEMON_YES = 107
    SWITCH_POKEMON_NO = 108

    POKEMON_MENU_CUT = 109
    POKEMON_MENU_FLY = 110
    POKEMON_MENU_SURF = 111
    POKEMON_MENU_STRENGTH = 112
    POKEMON_MENU_FLASH = 113
    POKEMON_MENU_DIG = 114


BATTLE_MENU_STATES = {
    RedRamMenuValues.BATTLE_MENU_FIGHT,
    RedRamMenuValues.BATTLE_MOVE_1,
    RedRamMenuValues.BATTLE_MOVE_2,
    RedRamMenuValues.BATTLE_MOVE_3,
    RedRamMenuValues.BATTLE_MOVE_4,
    RedRamMenuValues.BATTLE_MENU_PKMN,
    RedRamMenuValues.SELECT_POKEMON_1,
    RedRamMenuValues.SELECT_POKEMON_2,
    RedRamMenuValues.SELECT_POKEMON_3,
    RedRamMenuValues.SELECT_POKEMON_4,
    RedRamMenuValues.SELECT_POKEMON_5,
    RedRamMenuValues.SELECT_POKEMON_6,
    RedRamMenuValues.MENU_SELECT_STATS,
    RedRamMenuValues.MENU_SELECT_SWITCH,
    RedRamMenuValues.MENU_SELECT_CANCEL,
    RedRamMenuValues.BATTLE_SELECT_STATS,
    RedRamMenuValues.BATTLE_SELECT_SWITCH,
    RedRamMenuValues.BATTLE_MENU_ITEM,
    RedRamMenuValues.BATTLE_MENU_ITEM_X_USE,
    RedRamMenuValues.BATTLE_MENU_ITEM_X_TOSS,
    RedRamMenuValues.BATTLE_MART_PC_ITEM,
    RedRamMenuValues.BATTLE_MART_PC_ITEM_CANCEL,
    RedRamMenuValues.BATTLE_MENU_RUN,
}


TEXT_MENU_CURSOR_LOCATIONS = {
    RedRamMenuKeys.START_MENU_POKEDEX: RedRamMenuValues.START_MENU_POKEDEX,
    RedRamMenuKeys.START_MENU_POKEMON: RedRamMenuValues.START_MENU_POKEMON,
    RedRamMenuKeys.START_MENU_ITEM: RedRamMenuValues.START_MENU_ITEM,
    RedRamMenuKeys.START_MENU_SELF: RedRamMenuValues.START_MENU_SELF,
    RedRamMenuKeys.START_MENU_SAVE: RedRamMenuValues.START_MENU_SAVE,
    RedRamMenuKeys.START_MENU_OPTION: RedRamMenuValues.START_MENU_OPTION,
    RedRamMenuKeys.START_MENU_QUIT: RedRamMenuValues.START_MENU_QUIT,

    RedRamMenuKeys.POKECENTER_HEAL: RedRamMenuValues.POKECENTER_HEAL,
    RedRamMenuKeys.POKECENTER_CANCEL: RedRamMenuValues.POKECENTER_CANCEL,

    RedRamMenuKeys.POKEMART_BUY: RedRamMenuValues.POKEMART_BUY,
    RedRamMenuKeys.POKEMART_SELL: RedRamMenuValues.POKEMART_SELL,
    RedRamMenuKeys.POKEMART_QUIT: RedRamMenuValues.POKEMART_QUIT,

    RedRamMenuKeys.PC_SOMEONE: RedRamMenuValues.PC_SOMEONE,
    RedRamMenuKeys.PC_SELF: RedRamMenuValues.PC_SELF,
    RedRamMenuKeys.PC_OAK: RedRamMenuValues.PC_OAK,
    RedRamMenuKeys.PC_LOGOFF: RedRamMenuValues.PC_LOGOFF,

    RedRamMenuKeys.PC_SOMEONE_DEPOSIT_WITHDRAW: RedRamMenuValues.PC_SOMEONE_CONFIRM,
    RedRamMenuKeys.PC_SOMEONE_STATUS: RedRamMenuValues.PC_SOMEONE_STATUS,
    RedRamMenuKeys.PC_SOMEONE_CANCEL: RedRamMenuValues.PC_SOMEONE_CANCEL,

    RedRamMenuKeys.BATTLE_MENU_FIGHT: RedRamMenuValues.BATTLE_MENU_FIGHT,
    RedRamMenuKeys.BATTLE_MOVE_1: RedRamMenuValues.BATTLE_MOVE_1,
    RedRamMenuKeys.BATTLE_MOVE_2: RedRamMenuValues.BATTLE_MOVE_2,
    RedRamMenuKeys.BATTLE_MOVE_3: RedRamMenuValues.BATTLE_MOVE_3,
    RedRamMenuKeys.BATTLE_MOVE_4: RedRamMenuValues.BATTLE_MOVE_4,

    RedRamMenuKeys.BATTLE_MENU_PKMN: RedRamMenuValues.BATTLE_MENU_PKMN,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_1: RedRamMenuValues.SELECT_POKEMON_1,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_2: RedRamMenuValues.SELECT_POKEMON_2,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_3: RedRamMenuValues.SELECT_POKEMON_3,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_4: RedRamMenuValues.SELECT_POKEMON_4,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_5: RedRamMenuValues.SELECT_POKEMON_5,
    RedRamMenuKeys.BATTLE_ROSTER_PKMN_6: RedRamMenuValues.SELECT_POKEMON_6,
    RedRamMenuKeys.BATTLE_MON_SUB_SELECT_1: RedRamMenuValues.MENU_SELECT_STATS,
    RedRamMenuKeys.BATTLE_MON_SUB_SELECT_2: RedRamMenuValues.MENU_SELECT_SWITCH,
    RedRamMenuKeys.BATTLE_ROSTER_CANCEL: RedRamMenuValues.MENU_SELECT_CANCEL,

    RedRamMenuKeys.BATTLE_MENU_ITEM: RedRamMenuValues.BATTLE_MENU_ITEM,
    RedRamMenuKeys.BATTLE_MENU_ITEM_X_USE: RedRamMenuValues.BATTLE_MENU_ITEM_X_USE,
    RedRamMenuKeys.BATTLE_MENU_ITEM_X_TOSS: RedRamMenuValues.BATTLE_MENU_ITEM_X_TOSS,

    RedRamMenuKeys.BATTLE_MART_PC_ITEM_1: RedRamMenuValues.BATTLE_MART_PC_ITEM,
    RedRamMenuKeys.BATTLE_MART_PC_ITEM_2: RedRamMenuValues.BATTLE_MART_PC_ITEM,
    RedRamMenuKeys.BATTLE_MART_PC_ITEM_N: RedRamMenuValues.BATTLE_MART_PC_ITEM,

    RedRamMenuKeys.BATTLE_MART_PC_ITEM_CANCEL: RedRamMenuValues.BATTLE_MART_PC_ITEM_CANCEL,
    RedRamMenuKeys.BATTLE_MENU_RUN: RedRamMenuValues.BATTLE_MENU_RUN,

    RedRamMenuKeys.MENU_YES: RedRamMenuValues.MENU_YES,
    RedRamMenuKeys.MENU_NO: RedRamMenuValues.MENU_NO,
    RedRamMenuKeys.MENU_NO_HACK: RedRamMenuValues.MENU_NO,

    RedRamMenuKeys.OVERWRITE_MOVE_2: RedRamMenuValues.OVERWRITE_MOVE_2,
    RedRamMenuKeys.OVERWRITE_MOVE_3: RedRamMenuValues.OVERWRITE_MOVE_3,
    RedRamMenuKeys.OVERWRITE_MOVE_4: RedRamMenuValues.OVERWRITE_MOVE_4,

    # POKEMON_MENU_N# - can't map directly b/c HM's can be in any slot
}

# The count when buying/selling items
ITEM_SELECTION_QUANTITY = 0xCF96
ITEM_COUNT_SCREEN_PEAK = 0xC48F

TEXT_MENU_ITEM_LOCATIONS = {
    1 : RedRamMenuValues.ITEM_1,
    2 : RedRamMenuValues.ITEM_2,
    3 : RedRamMenuValues.ITEM_3,
    4 : RedRamMenuValues.ITEM_4,
    5 : RedRamMenuValues.ITEM_5,
    6 : RedRamMenuValues.ITEM_6,
    7 : RedRamMenuValues.ITEM_7,
    8 : RedRamMenuValues.ITEM_8,
    9 : RedRamMenuValues.ITEM_9,
    10 : RedRamMenuValues.ITEM_10,
    11 : RedRamMenuValues.ITEM_11,
    12 : RedRamMenuValues.ITEM_12,
    13 : RedRamMenuValues.ITEM_13,
    14 : RedRamMenuValues.ITEM_14,
    15 : RedRamMenuValues.ITEM_15,
    16 : RedRamMenuValues.ITEM_16,
    17 : RedRamMenuValues.ITEM_17,
    18 : RedRamMenuValues.ITEM_18,
    19 : RedRamMenuValues.ITEM_19,
    20 : RedRamMenuValues.ITEM_20
}

# Just diff'd text memory until found unique diff in text char's, as they have diff text box's which are still constant
PC_SUB_MENU_SCREEN_PEEK = 0xC41A
PC_SUB_MENU_DEPO_WITH_SCREEN_PEEK = 0xC4A0
HM_N0_MENU_SCREEN_PEEK = 0xC3FD
HM_N1_MENU_SCREEN_PEEK = 0xC423
HM_N2_MENU_SCREEN_PEEK = 0xC44B
HM_N3_MENU_SCREEN_PEEK = 0xC475


# PC Sub Menu's, they have the same values as the main menu's so they need to be differentiated
class RedRamSubMenuKeys:
    SUB_MENU_1 = (0xC9, 0xC3)
    SUB_MENU_2 = (0xF1, 0xC3)
    SUB_MENU_3 = (0x19, 0xC4)
    SUB_MENU_4 = (0x41, 0xC4)
    SUB_MENU_5 = (0x69, 0xC4)
    SUB_MENU_6 = (0x9A, 0xC4)
    SUB_MENU_7 = (0xC2, 0xC4)
    SUB_MENU_8 = (0xEA, 0xC4)


class RedRamSubMenuValues(IntEnum):
    UNKNOWN_MENU = 115
    PC_SOMEONE_WITHDRAW = 116
    PC_SOMEONE_DEPOSIT = 117
    PC_SOMEONE_RELEASE = 118
    PC_SOMEONE_CHANGE_BOX = 119
    PC_SOMEONE_EXIT = 120

    PC_SOMEONE_CONFIRM = 121
    PC_SOMEONE_CONFIRM_STATS = 122
    PC_SOMEONE_CONFIRM_CANCEL = 123
    PC_SOMEONE_CONFIRM_WITHDRAW = 124
    PC_SOMEONE_CONFIRM_DEPOSIT = 125

    PC_SELF_WITHDRAW_ITEM = 126
    PC_SELF_DEPOSIT_ITEM = 127
    PC_SELF_TOSS_ITEM = 128
    PC_SELF_EXIT = 129

MENU_TOTAL_SIZE = int(RedRamSubMenuValues.PC_SELF_EXIT) + 1


PC_POKE_MENU_CURSOR_LOCATIONS = {
    RedRamSubMenuKeys.SUB_MENU_1: RedRamSubMenuValues.PC_SOMEONE_WITHDRAW,
    RedRamSubMenuKeys.SUB_MENU_2: RedRamSubMenuValues.PC_SOMEONE_DEPOSIT,
    RedRamSubMenuKeys.SUB_MENU_3: RedRamSubMenuValues.PC_SOMEONE_RELEASE,
    RedRamSubMenuKeys.SUB_MENU_4: RedRamSubMenuValues.PC_SOMEONE_CHANGE_BOX,
    RedRamSubMenuKeys.SUB_MENU_5: RedRamSubMenuValues.PC_SOMEONE_EXIT,
    RedRamSubMenuKeys.SUB_MENU_6: RedRamSubMenuValues.PC_SOMEONE_CONFIRM,
    RedRamSubMenuKeys.SUB_MENU_7: RedRamSubMenuValues.PC_SOMEONE_CONFIRM_STATS,
    RedRamSubMenuKeys.SUB_MENU_8: RedRamSubMenuValues.PC_SOMEONE_CONFIRM_CANCEL,
}

PC_ITEM_MENU_CURSOR_LOCATIONS = {
    RedRamSubMenuKeys.SUB_MENU_1: RedRamSubMenuValues.PC_SELF_WITHDRAW_ITEM,
    RedRamSubMenuKeys.SUB_MENU_2: RedRamSubMenuValues.PC_SELF_DEPOSIT_ITEM,
    RedRamSubMenuKeys.SUB_MENU_3: RedRamSubMenuValues.PC_SELF_TOSS_ITEM,
    RedRamSubMenuKeys.SUB_MENU_4: RedRamSubMenuValues.PC_SELF_EXIT,
}


HM_MENU_LOOKUP = {
    0x0F: RedRamMenuValues.POKEMON_MENU_CUT,
    0x13: RedRamMenuValues.POKEMON_MENU_FLY,
    0x39: RedRamMenuValues.POKEMON_MENU_SURF,
    0x46: RedRamMenuValues.POKEMON_MENU_STRENGTH,
    0x94: RedRamMenuValues.POKEMON_MENU_FLASH,
    0x5B: RedRamMenuValues.POKEMON_MENU_DIG,
}
