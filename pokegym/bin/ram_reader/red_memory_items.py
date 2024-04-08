# Pokemart items
from enum import IntEnum


POKEMART_AVAIL_SIZE = 10
POKEMART_TOTAL_ITEMS = 0xCF7B # number of items the pokemart selling, stale until interaction
POKEMART_ITEMS = 0xCF7C
# TODO: Celadon City Dept. Store has bigger list of items, need to add support for that


# Bank-related addresses
BOX_SPECIES = 0xDA81
BOX_MONS = 0xDA96

# Players Money Count
PLAYER_MONEY = (0xD347, 0xD348, 0xD349) # Val already in base10,  ie. 0x99 0x25 0x49 == 992549 coins

# Bag Items
BAG_SIZE = 20
ITEMS_OFFSET = 2

# The count of all the items held in players bag
BAG_TOTAL_ITEMS = 0xD31D

# Items in players bag
BAG_ITEMS_INDEX = 0xD31E

# Quantities of each item in players bag
BAG_ITEM_QUANTITY_INDEX = 0xD31F


# Storage/PC Items
STORAGE_SIZE = 50

# The count of all the items held in players pc
PC_TOTAL_ITEMS = 0xD53A

# Items in storage/pc (50 items max)
PC_ITEMS_INDEX = 0xD53B

# Items quantity in storage/pc
PC_ITEM_QUANTITY_INDEX = 0xD53C


# Storage/PC Pokemon
BOX_SIZE = 20 # There are 12 boxes in the game, each box can hold 20 pokemon. The API only supports accessing the first box.
BOX_OFFSET = 0x21

# Pokemon count in storage/pc
BOX_POKEMON_COUNT = 0xDA80
BOX_POKEMON_NUM = 0xDA81

# Pokémon 1
BOX_POKEMON_1 = 0xDA96 # Terminates list w/ 0xFF
BOX_POKEMON_1_HP = (0xDA97, 0xDA98)
BOX_POKEMON_1_LEVEL = 0xDA99

# Pokémon 2
BOX_POKEMON_2 = 0xDAB7
BOX_POKEMON_2_HP = (0xDAB8, 0xDAB9)
BOX_POKEMON_2_LEVEL = 0xDABA

# Pokémon 3
BOX_POKEMON_3 = 0xDAD8
BOX_POKEMON_3_HP = (0xDAD9, 0xDADA)
BOX_POKEMON_3_LEVEL = 0xDADB

# Pokémon 4
BOX_POKEMON_4 = 0xDAF9
BOX_POKEMON_4_HP = (0xDAFA, 0xDAFB)
BOX_POKEMON_4_LEVEL = 0xDAFC

# Pokémon 5
BOX_POKEMON_5 = 0xDB1A
BOX_POKEMON_5_HP = (0xDB1B, 0xDB1C)
BOX_POKEMON_5_LEVEL = 0xDB1D

# Pokémon 6
BOX_POKEMON_6 = 0xDB3B
BOX_POKEMON_6_HP = (0xDB3C, 0xDB3D)
BOX_POKEMON_6_LEVEL = 0xDB3E

# Pokémon 7
BOX_POKEMON_7 = 0xDB5C
BOX_POKEMON_7_HP = (0xDB5D, 0xDB5E)
BOX_POKEMON_7_LEVEL = 0xDB5F

# Pokémon 8
BOX_POKEMON_8 = 0xDB7D
BOX_POKEMON_8_HP = (0xDB7E, 0xDB7F)
BOX_POKEMON_8_LEVEL = 0xDB80

# Pokémon 9
BOX_POKEMON_9 = 0xDB9E
BOX_POKEMON_9_HP = (0xDB9F, 0xDBA0)
BOX_POKEMON_9_LEVEL = 0xDBA1

# Pokémon 10
BOX_POKEMON_10 = 0xDBBF
BOX_POKEMON_10_HP = (0xDBC0, 0xDBC1)
BOX_POKEMON_10_LEVEL = 0xDBC2

# Pokémon 11
BOX_POKEMON_11 = 0xDBE0
BOX_POKEMON_11_HP = (0xDBE1, 0xDBE2)
BOX_POKEMON_11_LEVEL = 0xDBE3

# Pokémon 12
BOX_POKEMON_12 = 0xDC01
BOX_POKEMON_12_HP = (0xDC02, 0xDC03)
BOX_POKEMON_12_LEVEL = 0xDC04

# Pokémon 13
BOX_POKEMON_13 = 0xDC22
BOX_POKEMON_13_HP = (0xDC23, 0xDC24)
BOX_POKEMON_13_LEVEL = 0xDC25

# Pokémon 14
BOX_POKEMON_14 = 0xDC43
BOX_POKEMON_14_HP = (0xDC44, 0xDC45)
BOX_POKEMON_14_LEVEL = 0xDC46

# Pokémon 15
BOX_POKEMON_15 = 0xDC64
BOX_POKEMON_15_HP = (0xDC65, 0xDC66)
BOX_POKEMON_15_LEVEL = 0xDC67

# Pokémon 16
BOX_POKEMON_16 = 0xDC85
BOX_POKEMON_16_HP = (0xDC86, 0xDC87)
BOX_POKEMON_16_LEVEL = 0xDC88

# Pokémon 17
BOX_POKEMON_17 = 0xDCA6
BOX_POKEMON_17_HP = (0xDCA7, 0xDCA8)
BOX_POKEMON_17_LEVEL = 0xDCA9

# Pokémon 18
BOX_POKEMON_18 = 0xDCC7
BOX_POKEMON_18_HP = (0xDCC8, 0xDCC9)
BOX_POKEMON_18_LEVEL = 0xDCCA

# Pokémon 19
BOX_POKEMON_19 = 0xDCE8
BOX_POKEMON_19_HP = (0xDCE9, 0xDCEA)
BOX_POKEMON_19_LEVEL = 0xDCEB

# Pokémon 20
BOX_POKEMON_20 = 0xDD09
BOX_POKEMON_20_HP = (0xDD0A, 0xDD0B)
BOX_POKEMON_20_LEVEL = 0xDD0C


class ITEMS_HEX(IntEnum):
    Calcium = 0x27
    Carbos = 0x26
    Dire_Hit = 0x3A
    Elixir = 0x52
    Escape_Rope = 0x1D
    Ether = 0x50
    Fresh_Water = 0x3C
    Full_Heal = 0x34
    Full_Restore = 0x10
    Guard_Spec = 0x37
    HP_Up = 0x23
    Hyper_Potion = 0x12
    Ice_Heal = 0x0D
    Iron = 0x25
    Lemonade = 0x3E
    Max_Elixir = 0x53
    Max_Ether = 0x51
    Max_Potion = 0x11
    Max_Repel = 0x39
    Max_Revive = 0x36
    Nugget = 0x31
    Parlyz_Heal = 0x0F
    Poke_Doll = 0x33
    Potion = 0x14
    PP_Up = 0x4F
    Protein = 0x24
    Rare_Candy = 0x28
    Repel = 0x1E
    Revive = 0x35
    Soda_Pop = 0x3D
    Super_Potion = 0x13
    Super_Repel = 0x38
    X_Accuracy = 0x2E
    X_Attack = 0x41
    X_Defend = 0x42
    X_Special = 0x44
    X_Speed = 0x43
    Bicycle = 0x06
    Card_Key = 0x30
    Coin = 0x3B
    Coin_Case = 0x45
    Dome_Fossil = 0x29
    Exp_All = 0x4B
    Gold_Teeth = 0x40
    Good_Rod = 0x4D
    Helix_Fossil = 0x2A
    Old_Amber = 0x1F
    Old_Rod = 0x4C
    Town_Map = 0x05
    Moon_Stone = 0x0A
    Fire_Stone = 0x20
    ThunderStone = 0x21
    Water_Stone = 0x22
    Leaf_Stone = 0x2F
    Pokeball = 0x04
    Great_Ball = 0x03
    Ultra_Ball = 0x02


ITEM_LOOKUP = {
    0xC4: "HM01 Cut",
    0xC5: "HM02 Fly",
    0xC6: "HM03 Surf",
    0xC7: "HM04 Strength",
    0xC8: "HM05 Flash",
    0xC9: "TM01 Mega Punch",
    0xCA: "TM02 Razor Wind",
    0xCB: "TM03 Swords Dance",
    0xCC: "TM04 Whirlwind",
    0xCD: "TM05 Mega Kick",
    0xCE: "TM06 Toxic",
    0xCF: "TM07 Horn Drill",
    0xD0: "TM08 Body Slam",
    0xD1: "TM09 Take Down",
    0xD2: "TM10 Double Edge",
    0xD3: "TM11 Bubblebeam",
    0xD4: "TM12 Water Gun",
    0xD5: "TM13 Ice Beam",
    0xD6: "TM14 Blizzard",
    0xD7: "TM15 Hyper Beam",
    0xD8: "TM16 Pay Day",
    0xD9: "TM17 Submission",
    0xDA: "TM18 Counter",
    0xDB: "TM19 Seismic Toss",
    0xDC: "TM20 Rage",
    0xDD: "TM21 Mega Drain",
    0xDE: "TM22 Solar Beam",
    0xDF: "TM23 Dragon Rage",
    0xE0: "TM24 Thunderbolt",
    0xE1: "TM25 Thunder",
    0xE2: "TM26 Earthquake",
    0xE3: "TM27 Fissure",
    0xE4: "TM28 Dig",
    0xE5: "TM29 Psychic",
    0xE6: "TM30 Teleport",
    0xE7: "TM31 Mimic",
    0xE8: "TM32 Double Team",
    0xE9: "TM33 Reflect",
    0xEA: "TM34 Bide",
    0xEB: "TM35 Metronome",
    0xEC: "TM36 Self Destruct",
    0xED: "TM37 Egg Bomb",
    0xEE: "TM38 Fire Blast",
    0xEF: "TM39 Swift",
    0xF0: "TM40 Skull Bash",
    0xF1: "TM41 Softboiled",
    0xF2: "TM42 Dream Eater",
    0xF3: "TM43 Sky Attack",
    0xF4: "TM44 Rest",
    0xF5: "TM45 Thunderwave",
    0xF6: "TM46 Psywave",
    0xF7: "TM47 Explosion",
    0xF8: "TM48 Rock Slide",
    0xF9: "TM49 Tri Attack",
    0xFA: "TM50 Substitute",
    
    0x0B: "Antidote",
    0x0E: "Awakening",
    0x0C: "Burn Heal",
    0x27: "Calcium",
    0x26: "Carbos",
    0x3A: "Dire Hit",
    0x52: "Elixir",
    0x1D: "Escape Rope",
    0x50: "Ether",
    0x3C: "Fresh Water",
    0x34: "Full Heal",
    0x10: "Full Restore",
    0x37: "Guard Spec",
    0x23: "HP Up",
    0x12: "Hyper Potion",
    0x0D: "Ice Heal",
    0x25: "Iron",
    0x3E: "Lemonade",
    0x53: "Max Elixir",
    0x51: "Max Ether",
    0x11: "Max Potion",
    0x39: "Max Repel",
    0x36: "Max Revive",
    0x31: "Nugget",
    0x0F: "Parlyz Heal",
    0x33: "Poke Doll",
    0x14: "Potion",
    0x4F: "PP Up",
    0x24: "Protein",
    0x28: "Rare Candy",
    0x1E: "Repel",
    0x35: "Revive",
    0x3D: "Soda Pop",
    0x13: "Super Potion",
    0x38: "Super Repel",
    0x2E: "X Accuracy",
    0x41: "X Attack",
    0x42: "X Defend",
    0x44: "X Special",
    0x43: "X Speed",
    0x06: "Bicycle",
    0x2D: "Bike Voucher",
    0x30: "Card Key",
    0x3B: "Coin",
    0x45: "Coin Case",
    0x29: "Dome Fossil",
    0x4B: "Exp. All",
    0x40: "Gold Teeth",
    0x4D: "Good Rod",
    0x2A: "Helix Fossil",
    0x47: "Item Finder",
    0x4A: "Lift Key",
    0x46: "Oak's Parcel",
    0x1F: "Old Amber",
    0x4C: "Old Rod",
    0x49: "Poke Flute",
    0x09: "PokeDex",
    0x3F: "S.S. Ticket",
    0x2B: "Secret Key",
    0x48: "Silph Scope",
    0x4E: "Super Rod",
    0x05: "Town Map",
    0x0A: "Moon Stone",
    0x20: "Fire Stone",
    0x21: "ThunderStone",
    0x22: "Water Stone",
    0x2F: "Leaf Stone",
    0x04: "Pokeball",
    0x03: "Great Ball",
    0x02: "Ultra Ball",
    0x01: "Master Ball",
}


ITEM_COSTS = {

    0xC9: 3000,
    0xCA: 2000,
    0xCD: 3000,
    0xCF: 2000,
    0xD1: 3000,
    0xD9: 3000,
    0xE8: 1000,
    0xE9: 1000,
    0xED: 2000,
    
    0x0B: 100,
    0x0E: 200,
    0x0C: 250,
    0x27: 9800,
    0x26: 9800,
    0x3A: 650,
    0x1D: 550,
    0x20: 2100,
    0x3C: 200,
    0x34: 600,
    0x10: 3000,
    0x37: 700,
    0x23: 9800,
    0x12: 1200,
    0x0D: 250,
    0x25: 9800,
    0x3E: 350,
    0x11: 2500,
    0x39: 700,
    0x36: 1500,
    0x31: 10000,
    0x0F: 200,
    0x33: 1000,
    0x14: 300,
    0x24: 9800,
    0x28: 4800,
    0x1E: 350,
    0x35: 1500,
    0x3D: 300,
    0x13: 700,
    0x38: 500,
    0x2E: 9500,
    0x41: 500,
    0x42: 550,
    0x44: 3500,
    0x43: 3500,
    0x06: 1000,
    0x2D: 1000,
    0x30: 200,
    0x3B: 10,
    0x45: 1000,
    0x29: 10000,
    0x4B: 9800,
    0x40: 9800,
    0x4D: 9800,
    0x47: 1000,
    0x4A: 1000,
    0x4C: 1000,
    0x4E: 10000,
    0x0A: 2100,
    0x20: 2100,
    0x21: 2100,
    0x22: 2100,
    0x2F: 2100,
    0x04: 200,
    0x03: 600,
    0x02: 1200,
}

