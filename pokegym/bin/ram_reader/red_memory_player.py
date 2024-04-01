# Player Party Overview
PARTY_OFFSET = 0x2C
POKEMON_PARTY_COUNT = 0xD163
POKEMON_1_ID = 0xD164 # ID of mon or 0x00 when none
POKEMON_2_ID = 0xD165 # 0xFF marks end of list, but prev EoL isn't cleared when party size shrinks, must
POKEMON_3_ID = 0xD166 # use LSB as 0xFF marker
POKEMON_4_ID = 0xD167
POKEMON_5_ID = 0xD168
POKEMON_6_ID = 0xD169

# Player Constants
POKEMON_TOTAL_ATTRIBUTES = 20
MAX_MONEY = 999999.0

# Pokemon 1 Details
POKEMON_1 = 0xD16B
POKEMON_1_CURRENT_HP = (0xD16C, 0xD16D) # HP as hex concat number ie. 0x01 0x045 -> 0x145 & 0x145 == 325hp
POKEMON_1_STATUS = 0xD16F
POKEMON_1_TYPES = (0xD170,0xD171)
POKEMON_1_MOVES = (0xD173, 0xD174, 0xD175, 0xD176)
POKEMON_1_EXPERIENCE = (0xD179, 0xD17A, 0xD17B) # Current XP @ l as 3 hex concat numbers ie. 0x00 0x01 0x080 == 348
POKEMON_1_PP_MOVES = (0xD188, 0xD189, 0xD18A, 0xD18B)
POKEMON_1_LEVEL_ACTUAL = 0xD18C
POKEMON_1_MAX_HP = (0xD18D, 0xD18E)
POKEMON_1_ATTACK = (0xD18F, 0xD190)
POKEMON_1_DEFENSE = (0xD191, 0xD192)
POKEMON_1_SPEED = (0xD193, 0xD194)
POKEMON_1_SPECIAL = (0xD195, 0xD196)

# Pokemon 2 Details
POKEMON_2 = 0xD197
POKEMON_2_CURRENT_HP = (0xD198, 0xD198)
POKEMON_2_STATUS = 0xD19B
POKEMON_2_TYPES = (0xD19C, 0xD19D)
POKEMON_2_MOVES = (0xD19F, 0xD1A0, 0xD1A1, 0xD1A2)
POKEMON_2_EXPERIENCE = (0xD1A5, 0xD1A6, 0xD1A7)
POKEMON_2_PP_MOVES = (0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7)
POKEMON_2_LEVEL_ACTUAL = 0xD1B8
POKEMON_2_MAX_HP = (0xD1B9, 0xD1BA)
POKEMON_2_ATTACK = (0xD1BB, 0xD1BC)
POKEMON_2_DEFENSE = (0xD1BD, 0xD1BE)
POKEMON_2_SPEED = (0xD1BF, 0xD1C0)
POKEMON_2_SPECIAL = (0xD1C1, 0xD1C2)

# Pokemon 3 Details
POKEMON_3 = 0xD1C3
POKEMON_3_CURRENT_HP = (0xD1C4, 0xD1C5)
POKEMON_3_STATUS = 0xD1C7
POKEMON_3_TYPES = (0xD1C8, 0xD1C9)
POKEMON_3_MOVES = (0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE)
POKEMON_3_EXPERIENCE = (0xD1D1, 0xD1D2, 0xD1D3)
POKEMON_3_PP_MOVES = (0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3)
POKEMON_3_LEVEL_ACTUAL = 0xD1E4
POKEMON_3_MAX_HP = (0xD1E5, 0xD1E6)
POKEMON_3_ATTACK = (0xD1E7, 0xD1E8)
POKEMON_3_DEFENSE = (0xD1E9, 0xD1EA)
POKEMON_3_SPEED = (0xD1EB, 0xD1EC)
POKEMON_3_SPECIAL = (0xD1ED, 0xD1EE)

# Pokemon 4 Details
POKEMON_4 = 0xD1EF
POKEMON_4_CURRENT_HP = (0xD1F0, 0xD1F1)
POKEMON_4_STATUS = 0xD1F3
POKEMON_4_TYPES = (0xD1F4, 0xD1F5)
POKEMON_4_MOVES = (0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA)
POKEMON_4_EXPERIENCE = (0xD1FD, 0xD1FE, 0xD1FF)
POKEMON_4_PP_MOVES = (0xD20C, 0xD20D, 0xD20E, 0xD20F)
POKEMON_4_LEVEL_ACTUAL = 0xD210
POKEMON_4_MAX_HP = (0xD211, 0xD212)
POKEMON_4_ATTACK = (0xD213, 0xD214)
POKEMON_4_DEFENSE = (0xD215, 0xD216)
POKEMON_4_SPEED = (0xD217, 0xD218)
POKEMON_4_SPECIAL = (0xD219, 0xD21A)

# Pokemon 5 Details
POKEMON_5 = 0xD21B
POKEMON_5_CURRENT_HP = (0xD21C, 0xD21D)
POKEMON_5_STATUS = 0xD21F
POKEMON_5_TYPES = (0xD220, 0xD221)
POKEMON_5_MOVES = (0xD223, 0xD224, 0xD225, 0xD226)
POKEMON_5_EXPERIENCE = (0xD229, 0xD22A, 0xD22B)
POKEMON_5_PP_MOVES = (0xD238, 0xD239, 0xD23A, 0xD23B)
POKEMON_5_LEVEL_ACTUAL = 0xD23C
POKEMON_5_MAX_HP = (0xD23D, 0xD23E)
POKEMON_5_ATTACK = (0xD23F, 0xD240)
POKEMON_5_DEFENSE = (0xD241, 0xD242)
POKEMON_5_SPEED = (0xD243, 0xD244)
POKEMON_5_SPECIAL = (0xD245, 0xD246)

# Pokemon 6 Details
POKEMON_6 = 0xD247
POKEMON_6_CURRENT_HP = (0xD248, 0xD249)
POKEMON_6_STATUS = 0xD24B
POKEMON_6_TYPES = (0xD24C, 0xD24D)
POKEMON_6_MOVES = (0xD24F, 0xD250, 0xD251, 0xD252)
POKEMON_6_EXPERIENCE = (0xD255, 0xD256, 0xD257)
POKEMON_6_PP_MOVES = (0xD264, 0xD265, 0xD266, 0xD267)
POKEMON_6_LEVEL_ACTUAL = 0xD268
POKEMON_6_MAX_HP = (0xD269, 0xD26A)
POKEMON_6_ATTACK = (0xD26B, 0xD26C)
POKEMON_6_DEFENSE = (0xD26D, 0xD26E)
POKEMON_6_SPEED = (0xD26F, 0xD270)
POKEMON_6_SPECIAL = (0xD271, 0xD272)

POKEMON_PARTY = [
    POKEMON_1,
    POKEMON_2,
    POKEMON_3,
    POKEMON_4,
    POKEMON_5,
    POKEMON_6
]
POKEMON_PARTY_SIZE = len(POKEMON_PARTY)

POKEMON_LOOKUP = {
    148: "Abra",
    171: "Aerodactyl",
    149: "Alakazam",
    45: "Arbok",
    20: "Arcanine",
    74: "Articuno",
    114: "Beedrill",
    188: "Bellsprout",
    28: "Blastoise",
    153: "Bulbasaur",
    125: "Butterfree",
    123: "Caterpie",
    40: "Chansey",
    180: "Charizard",
    176: "Charmander",
    178: "Charmeleon",
    142: "Clefable",
    4: "Clefairy",
    139: "Cloyster",
    17: "Cubone",
    120: "Dewgong",
    59: "Diglett",
    76: "Ditto",
    116: "Dodrio",
    70: "Doduo",
    89: "Dragonair",
    66: "Dragonite",
    88: "Dratini",
    48: "Drowzee",
    118: "Dugtrio",
    102: "Eevee",
    108: "Ekans",
    53: "Electabuzz",
    141: "Electrode",
    12: "Exeggcute",
    10: "Exeggutor",
    64: "Farfetch’d",
    35: "Fearow",
    103: "Flareon",
    25: "Gastly",
    14: "Gengar",
    169: "Geodude",
    186: "Gloom",
    130: "Golbat",
    157: "Goldeen",
    128: "Golduck",
    49: "Golem",
    39: "Graveler",
    13: "Grimer",
    33: "Growlithe",
    22: "Gyarados",
    147: "Haunter",
    44: "Hitmonchan",
    43: "Hitmonlee",
    92: "Horsea",
    129: "Hypno",
    9: "Ivysaur",
    100: "Jigglypuff",
    104: "Jolteon",
    72: "Jynx",
    90: "Kabuto",
    91: "Kabutops",
    38: "Kadabra",
    113: "Kakuna",
    2: "Kangaskhan",
    138: "Kingler",
    55: "Koffing",
    78: "Krabby",
    19: "Lapras",
    11: "Lickitung",
    126: "Machamp",
    41: "Machoke",
    106: "Machop",
    133: "Magikarp",
    51: "Magmar",
    173: "Magnemite",
    54: "Magneton",
    57: "Mankey",
    145: "Marowak",
    77: "Meowth",
    124: "Metapod",
    21: "Mew",
    131: "Mewtwo",
    73: "Moltres",
    42: "Mr. Mime",
    136: "Muk",
    7: "Nidoking",
    16: "Nidoqueen",
    15: "Nidoran F",
    3: "Nidoran M",
    167: "Nidorino",
    168: "Nidorina",
    83: "Ninetales",
    185: "Oddish",
    98: "Omanyte",
    99: "Omastar",
    34: "Onix",
    109: "Paras",
    46: "Parasect",
    144: "Persian",
    151: "Pidgeot",
    150: "Pidgeotto",
    36: "Pidgey",
    84: "Pikachu",
    29: "Pinsir",
    71: "Poliwag",
    110: "Poliwhirl",
    111: "Poliwrath",
    163: "Ponyta",
    170: "Porygon",
    117: "Primeape",
    47: "Psyduck",
    85: "Raichu",
    164: "Rapidash",
    166: "Raticate",
    165: "Rattata",
    1: "Rhydon",
    18: "Rhyhorn",
    96: "Sandshrew",
    97: "Sandslash",
    26: "Scyther",
    93: "Seadra",
    158: "Seaking",
    58: "Seel",
    23: "Shellder",
    8: "Slowbro",
    37: "Slowpoke",
    132: "Snorlax",
    5: "Spearow",
    177: "Squirtle",
    152: "Starmie",
    27: "Staryu",
    30: "Tangela",
    60: "Tauros",
    24: "Tentacool",
    155: "Tentacruel",
    105: "Vaporeon",
    119: "Venomoth",
    65: "Venonat",
    154: "Venusaur",
    190: "Victreebell",
    187: "Vileplume",
    6: "Voltorb",
    82: "Vulpix",
    179: "Wartortle",
    112: "Weedle",
    189: "Weepinbell",
    143: "Weezing",
    101: "Wigglytuff",
    75: "Zapdos",
    107: "Zubat"
}

# Dead, out of battle
PLAYER_DEAD = 0xD12D