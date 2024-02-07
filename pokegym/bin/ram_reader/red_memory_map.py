# Player addresses
PLAYER_LOCATION_X = 0xD362
PLAYER_LOCATION_Y = 0xD361
PLAYER_MAP = 0xD35E
PLAYER_ANIM_FRAME_COUNTER = 0xC108
PLAYER_FACING_DIR = 0xC109
PLAYER_COLLISION = 0xC10C # Running into NPC, doesn't count map boundary collisions
PLAYER_IN_GRASS = 0xC207 # 0x80 in poke grass, else 00

# Player's surroundings tiles (7x7)[not accurate when in chat/menu screen, as boxes overlay map tiles]
TILE_COL_0_ROW_0 = 0xC3B4
TILE_COL_1_ROW_0 = 0xC3B6
TILE_COL_2_ROW_0 = 0xC3B8
TILE_COL_3_ROW_0 = 0xC3BA
TILE_COL_4_ROW_0 = 0xC3BC
TILE_COL_5_ROW_0 = 0xC3BE
TILE_COL_6_ROW_0 = 0xC3C0
TILE_COL_7_ROW_0 = 0xC3C2
TILE_COL_8_ROW_0 = 0xC3C4

TILE_COL_0_ROW_1 = 0xC3DC
TILE_COL_1_ROW_1 = 0xC3DE
TILE_COL_2_ROW_1 = 0xC3E0
TILE_COL_3_ROW_1 = 0xC3E2
TILE_COL_4_ROW_1 = 0xC3E4
TILE_COL_5_ROW_1 = 0xC3E6
TILE_COL_6_ROW_1 = 0xC3E8
TILE_COL_7_ROW_1 = 0xC3EA
TILE_COL_8_ROW_1 = 0xC3EC

TILE_COL_0_ROW_2 = 0xC404
TILE_COL_1_ROW_2 = 0xC406
TILE_COL_2_ROW_2 = 0xC408
TILE_COL_3_ROW_2 = 0xC40A
TILE_COL_4_ROW_2 = 0xC40C
TILE_COL_5_ROW_2 = 0xC40E
TILE_COL_6_ROW_2 = 0xC410
TILE_COL_7_ROW_2 = 0xC412
TILE_COL_8_ROW_2 = 0xC414

TILE_COL_0_ROW_3 = 0xC42C
TILE_COL_1_ROW_3 = 0xC42E
TILE_COL_2_ROW_3 = 0xC430
TILE_COL_3_ROW_3 = 0xC432
TILE_COL_4_ROW_3 = 0xC434
TILE_COL_5_ROW_3 = 0xC436
TILE_COL_6_ROW_3 = 0xC438
TILE_COL_7_ROW_3 = 0xC43A
TILE_COL_8_ROW_3 = 0xC43C

TILE_COL_0_ROW_4 = 0xC454
TILE_COL_1_ROW_4 = 0xC456
TILE_COL_2_ROW_4 = 0xC458
TILE_COL_3_ROW_4 = 0xC45A
TILE_COL_4_ROW_4 = 0xC45C
TILE_COL_5_ROW_4 = 0xC45E
TILE_COL_6_ROW_4 = 0xC460
TILE_COL_7_ROW_4 = 0xC462
TILE_COL_8_ROW_4 = 0xC464

TILE_COL_0_ROW_5 = 0xC47C
TILE_COL_1_ROW_5 = 0xC47E
TILE_COL_2_ROW_5 = 0xC480
TILE_COL_3_ROW_5 = 0xC482
TILE_COL_4_ROW_5 = 0xC484
TILE_COL_5_ROW_5 = 0xC486
TILE_COL_6_ROW_5 = 0xC488
TILE_COL_7_ROW_5 = 0xC48A
TILE_COL_8_ROW_5 = 0xC48C

TILE_COL_0_ROW_6 = 0xC4A4
TILE_COL_1_ROW_6 = 0xC4A6
TILE_COL_2_ROW_6 = 0xC4A8
TILE_COL_3_ROW_6 = 0xC4AA
TILE_COL_4_ROW_6 = 0xC4AC
TILE_COL_5_ROW_6 = 0xC4AE
TILE_COL_6_ROW_6 = 0xC4B0
TILE_COL_7_ROW_6 = 0xC4B2
TILE_COL_8_ROW_6 = 0xC4B4

TILE_COL_0_ROW_7 = 0xC4CC
TILE_COL_1_ROW_7 = 0xC4CE
TILE_COL_2_ROW_7 = 0xC4D0
TILE_COL_3_ROW_7 = 0xC4D2
TILE_COL_4_ROW_7 = 0xC4D4
TILE_COL_5_ROW_7 = 0xC4D6
TILE_COL_6_ROW_7 = 0xC4D8
TILE_COL_7_ROW_7 = 0xC4DA
TILE_COL_8_ROW_7 = 0xC4DC

TILE_COL_0_ROW_8 = 0xC4F4
TILE_COL_1_ROW_8 = 0xC4F6
TILE_COL_2_ROW_8 = 0xC4F8
TILE_COL_3_ROW_8 = 0xC4FA
TILE_COL_4_ROW_8 = 0xC4FC
TILE_COL_5_ROW_8 = 0xC4FE
TILE_COL_6_ROW_8 = 0xC500
TILE_COL_7_ROW_8 = 0xC502
TILE_COL_8_ROW_8 = 0xC504

TILE_CURRENT_AND_FRONT_BUMP_PLAYER = 0xCFC6 # Tile ID of player until sprite bump obstacle, then it's obstacle tile ID

# NPCs
# Sprite 1
SPRITE_01_PICTURE_ID = 0xC110
SPRITE_01_Y_SCREEN_POSITION = 0xC114
SPRITE_01_X_SCREEN_POSITION = 0xC116
SPRITE_01_ANIMATION_FRAME_COUNTER = 0xC118
SPRITE_01_FACING_DIRECTION = 0xC119

# Sprite 2
SPRITE_02_PICTURE_ID = 0xC120
SPRITE_02_Y_SCREEN_POSITION = 0xC124
SPRITE_02_X_SCREEN_POSITION = 0xC126
SPRITE_02_ANIMATION_FRAME_COUNTER = 0xC128
SPRITE_02_FACING_DIRECTION = 0xC129

# Sprite 3
SPRITE_03_PICTURE_ID = 0xC130
SPRITE_03_Y_SCREEN_POSITION = 0xC134
SPRITE_03_X_SCREEN_POSITION = 0xC136
SPRITE_03_ANIMATION_FRAME_COUNTER = 0xC138
SPRITE_03_FACING_DIRECTION = 0xC139

# Sprite 4
SPRITE_04_PICTURE_ID = 0xC140
SPRITE_04_Y_SCREEN_POSITION = 0xC144
SPRITE_04_X_SCREEN_POSITION = 0xC146
SPRITE_04_ANIMATION_FRAME_COUNTER = 0xC148
SPRITE_04_FACING_DIRECTION = 0xC149

# Sprite 5
SPRITE_05_PICTURE_ID = 0xC150
SPRITE_05_Y_SCREEN_POSITION = 0xC154
SPRITE_05_X_SCREEN_POSITION = 0xC156
SPRITE_05_ANIMATION_FRAME_COUNTER = 0xC158
SPRITE_05_FACING_DIRECTION = 0xC159

# Sprite 6
SPRITE_06_PICTURE_ID = 0xC160
SPRITE_06_Y_SCREEN_POSITION = 0xC164
SPRITE_06_X_SCREEN_POSITION = 0xC166
SPRITE_06_ANIMATION_FRAME_COUNTER = 0xC168
SPRITE_06_FACING_DIRECTION = 0xC169

# Sprite 7
SPRITE_07_PICTURE_ID = 0xC170
SPRITE_07_Y_SCREEN_POSITION = 0xC174
SPRITE_07_X_SCREEN_POSITION = 0xC176
SPRITE_07_ANIMATION_FRAME_COUNTER = 0xC178
SPRITE_07_FACING_DIRECTION = 0xC179

# Sprite 8
SPRITE_08_PICTURE_ID = 0xC180
SPRITE_08_Y_SCREEN_POSITION = 0xC184
SPRITE_08_X_SCREEN_POSITION = 0xC186
SPRITE_08_ANIMATION_FRAME_COUNTER = 0xC188
SPRITE_08_FACING_DIRECTION = 0xC189

# Sprite 9
SPRITE_09_PICTURE_ID = 0xC190
SPRITE_09_Y_SCREEN_POSITION = 0xC194
SPRITE_09_X_SCREEN_POSITION = 0xC196
SPRITE_09_ANIMATION_FRAME_COUNTER = 0xC198
SPRITE_09_FACING_DIRECTION = 0xC199

# Sprite 10
SPRITE_0A_PICTURE_ID = 0xC1A0
SPRITE_0A_Y_SCREEN_POSITION = 0xC1A4
SPRITE_0A_X_SCREEN_POSITION = 0xC1A6
SPRITE_0A_ANIMATION_FRAME_COUNTER = 0xC1A8
SPRITE_0A_FACING_DIRECTION = 0xC1A9

# Sprite 11
SPRITE_0B_PICTURE_ID = 0xC1B0
SPRITE_0B_Y_SCREEN_POSITION = 0xC1B4
SPRITE_0B_X_SCREEN_POSITION = 0xC1B6
SPRITE_0B_ANIMATION_FRAME_COUNTER = 0xC1B8
SPRITE_0B_FACING_DIRECTION = 0xC1B9

# Sprite 12
SPRITE_0C_PICTURE_ID = 0xC1C0
SPRITE_0C_Y_SCREEN_POSITION = 0xC1C4
SPRITE_0C_X_SCREEN_POSITION = 0xC1C6
SPRITE_0C_ANIMATION_FRAME_COUNTER = 0xC1C8
SPRITE_0C_FACING_DIRECTION = 0xC1C9

# Sprite 13
SPRITE_0D_PICTURE_ID = 0xC1D0
SPRITE_0D_Y_SCREEN_POSITION = 0xC1D4
SPRITE_0D_X_SCREEN_POSITION = 0xC1D6
SPRITE_0D_ANIMATION_FRAME_COUNTER = 0xC1D8
SPRITE_0D_FACING_DIRECTION = 0xC1D9

# Sprite 14
SPRITE_0E_PICTURE_ID = 0xC1E0
SPRITE_0E_Y_SCREEN_POSITION = 0xC1E4
SPRITE_0E_X_SCREEN_POSITION = 0xC1E6
SPRITE_0E_ANIMATION_FRAME_COUNTER = 0xC1E8
SPRITE_0E_FACING_DIRECTION = 0xC1E9

# Sprite 15
SPRITE_0F_PICTURE_ID = 0xC1F0
SPRITE_0F_Y_SCREEN_POSITION = 0xC1F4
SPRITE_0F_X_SCREEN_POSITION = 0xC1F6
SPRITE_0F_ANIMATION_FRAME_COUNTER = 0xC1F8
SPRITE_0F_FACING_DIRECTION = 0xC1F9

SPRITE_STARTING_ADDRESSES = [
    SPRITE_01_PICTURE_ID, SPRITE_02_PICTURE_ID, SPRITE_03_PICTURE_ID, SPRITE_04_PICTURE_ID, SPRITE_05_PICTURE_ID, SPRITE_06_PICTURE_ID,  SPRITE_07_PICTURE_ID,  SPRITE_08_PICTURE_ID,  SPRITE_09_PICTURE_ID,  SPRITE_0A_PICTURE_ID,  SPRITE_0B_PICTURE_ID,  SPRITE_0C_PICTURE_ID,  SPRITE_0D_PICTURE_ID,  SPRITE_0E_PICTURE_ID,  SPRITE_0F_PICTURE_ID
]

ordered_locations = [
    # Route 1, Viridian City, back to Pallet Town (not implemented yet)
    # Route 2
    # Viridian Forest, Pewter City, Pewter Gym
    # Route 3
{'seq': 1, 'id': '0', 'name': 'Pallet Town', 'coordinates': [64, 318], 'tileSize': [20, 18]},
{'seq': 2, 'id': '12', 'name': 'Route 1', 'coordinates': [64, 282], 'tileSize': [20, 36]},
{'seq': 3, 'id': '1', 'name': 'Viridian City', 'coordinates': [54, 246], 'tileSize': [40, 36]},
{'seq': 4, 'id': '41', 'name': 'Pokemon Center (Viridian City)', 'coordinates': [95, 274], 'tileSize': [14, 8]},
{'seq': 5, 'id': '13', 'name': 'Route 2', 'coordinates': [64, 174], 'tileSize': [20, 72]},
{'seq': 6, 'id': '51', 'name': 'Viridian Forest', 'coordinates': [88, 188], 'tileSize': [34, 48]},
{'seq': 7, 'id': '2', 'name': 'Pewter City', 'coordinates': [54, 138], 'tileSize': [40, 36]},
{'seq': 8, 'id': '58', 'name': 'Pokecenter', 'coordinates': [39, 170], 'tileSize': [14, 8]},
{'seq': 9, 'id': '53', 'name': 'Museum F2', 'coordinates': [57, 122], 'tileSize': [14, 8]},
{'seq': 10, 'id': '14', 'name': 'Route 3', 'coordinates': [94, 146], 'tileSize': [70, 18]},
{'seq': 11, 'id': '59', 'name': 'Mt Moon Route 3', 'coordinates': [147, 92], 'tileSize': [40, 36]},
{'seq': 12, 'id': '64', 'name': 'Pokemon Center', 'coordinates': [265, 104], 'tileSize': [14, 8]},
{'seq': 13, 'id': '3', 'name': 'Cerulean City', 'coordinates': [234, 120], 'tileSize': [40, 36]},
{'seq': 14, 'id': '61', 'name': 'Mt Moon B2F', 'coordinates': [147, 26], 'tileSize': [40, 36]},
{'seq': 15, 'id': '62', 'name': 'House Breakin v1', 'coordinates': [280, 104], 'tileSize': [8, 8]},
{'seq': 16, 'id': '4', 'name': 'Lavender Town', 'coordinates': [334, 200], 'tileSize': [20, 18]},
{'seq': 17, 'id': '5', 'name': 'Vermilion City', 'coordinates': [234, 264], 'tileSize': [40, 36]},
{'seq': 18, 'id': '54', 'name': 'Pewter Gym', 'coordinates': [43, 145], 'tileSize': [10, 15]},
{'seq': 19, 'id': '57', 'name': 'Trainer House', 'coordinates': [55, 175], 'tileSize': [8, 8]},
{'seq': 20, 'id': '24', 'name': 'Route 13', 'coordinates': [294, 326], 'tileSize': [60, 18]},
{'seq': 21, 'id': '25', 'name': 'Route 14', 'coordinates': [274, 326], 'tileSize': [20, 54]},
{'seq': 22, 'id': '88', 'name': 'Bills Lab', 'coordinates': [307, 76], 'tileSize': [8, 8]},
{'seq': 23, 'id': '6', 'name': 'Celadon City', 'coordinates': [164, 192], 'tileSize': [50, 36]},
{'seq': 24, 'id': '9', 'name': 'Indigo Plateau', 'coordinates': [14, 84], 'tileSize': [20, 18]},
{'seq': 25, 'id': '10', 'name': 'Saffrdon City', 'coordinates': [234, 192], 'tileSize': [40, 36]},
{'seq': 26, 'id': '60', 'name': 'Mt Moon B1F', 'coordinates': [147, 63], 'tileSize': [28, 28]},
{'seq': 27, 'id': '21', 'name': 'Route 10', 'coordinates': [334, 128], 'tileSize': [20, 72]},
{'seq': 28, 'id': '92', 'name': 'Vermilion Gym', 'coordinates': [214, 282], 'tileSize': [10, 18]},
{'seq': 29, 'id': '6', 'name': 'Celadon City', 'coordinates': [164, 192], 'tileSize': [50, 36]},
{'seq': 30, 'id': '5', 'name': 'Vermilion City', 'coordinates': [234, 264], 'tileSize': [40, 36]},
{'seq': 32, 'id': '8', 'name': 'Cinnabar island', 'coordinates': [64, 426], 'tileSize': [20, 18]},
{'seq': 33, 'id': '7', 'name': 'Fuchsia City', 'coordinates': [174, 354], 'tileSize': [40, 36]},
{'seq': 34, 'id': '70', 'name': 'Underground Entrance', 'coordinates': [265, 157], 'tileSize': [8, 8]},
{'seq': 35, 'id': '67', 'name': 'Pokemart', 'coordinates': [225, 147], 'tileSize': [8, 8]},
{'seq': 36, 'id': '70', 'name': 'Underground Entrance', 'coordinates': [265, 157], 'tileSize': [8, 8]},
{'seq': 37, 'id': '4', 'name': 'Lavender Town', 'coordinates': [334, 200], 'tileSize': [20, 18]},
{'seq': 38, 'id': '19', 'name': 'Route 8', 'coordinates': [274, 200], 'tileSize': [60, 18]},
{'seq': 39, 'id': '10', 'name': 'Saffrdon City', 'coordinates': [234, 192], 'tileSize': [40, 36]},
{'seq': 40, 'id': '16', 'name': 'Route 5', 'coordinates': [244, 156], 'tileSize': [20, 36]},
{'seq': 41, 'id': '17', 'name': 'Route 6', 'coordinates': [244, 228], 'tileSize': [20, 36]},
{'seq': 42, 'id': '18', 'name': 'Route 7', 'coordinates': [214, 200], 'tileSize': [20, 18]},
{'seq': 43, 'id': '19', 'name': 'Route 8', 'coordinates': [274, 200], 'tileSize': [60, 18]},
{'seq': 44, 'id': '20', 'name': 'Route 9', 'coordinates': [274, 128], 'tileSize': [60, 18]},
{'seq': 45, 'id': '21', 'name': 'Route 10', 'coordinates': [334, 128], 'tileSize': [20, 72]},
{'seq': 46, 'id': '22', 'name': 'Route 11', 'coordinates': [274, 272], 'tileSize': [60, 18]},
]


TILE_COLLISION_PTR_1 = 0xD531
TILE_COLLISION_PTR_2 = 0xD530
WALKABLE_TILES = {
    (0x17, 0x35) : {0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D, 0x2E, 0x30, 0x31, 0x33, 0x39, 0x3C, 0x3E, 0x52, 0x54, 0x58, 0x5B, 0xFF,},
    (0x17, 0x7F) : {0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E, 0xFF,},
    (0x17, 0x65) : {0x1E, 0x20, 0x2E, 0x30, 0x34, 0x37, 0x39, 0x3A, 0x40, 0x51, 0x52, 0x5A, 0x5C, 0x5E, 0x5F, 0xFF,},
    (0x17, 0x49) : {0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1C, 0x1A, 0xFF,},
    (0x17, 0x75) : {0x01, 0x12, 0x14, 0x28, 0x32, 0x37, 0x44, 0x54, 0x5C, 0xFF,},
    (0x17, 0x59) : {0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C, 0x4D, 0x03, 0xFF,},
    (0x17, 0x53) : {0x11, 0x1A, 0x1C, 0x3C, 0x5E, 0xFF},
}
