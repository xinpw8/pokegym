from pokegym import data

HP_ADDR = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
WCUTTILE = 0xCD4D # 61 if Cut used; 0 default. resets to default on map_n change or battle.


GYM_LEADER = 5
GYM_TRAINER = 2
GYM_TASK = 2
TRAINER = 1
HM = 5
TM = 2
TASK = 2
POKEMON = 3
ITEM = 5
BILL_CAPT = 5
RIVAL = 3
QUEST = 5
EVENT = 1
BAD = -1


def silph_co(game):
  Beat_Silph_Co_2F_Trainer_0 = TRAINER * int(read_bit(game, 0xD825, 2))
  Beat_Silph_Co_2F_Trainer_1 = TRAINER * int(read_bit(game, 0xD825, 3))
  Beat_Silph_Co_2F_Trainer_2 = TRAINER * int(read_bit(game, 0xD825, 4))
  Beat_Silph_Co_2F_Trainer_3 = TRAINER * int(read_bit(game, 0xD825, 5))
  Silph_Co_2_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD826, 5))
  Silph_Co_2_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD826, 6))
  Beat_Silph_Co_3F_Trainer_0 = TRAINER * int(read_bit(game, 0xD827, 2))
  Beat_Silph_Co_3F_Trainer_1 = TRAINER * int(read_bit(game, 0xD827, 3))
  Silph_Co_3_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD828, 0))
  Silph_Co_3_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD828, 1))
  Beat_Silph_Co_4F_Trainer_0 = TRAINER * int(read_bit(game, 0xD829, 2))
  Beat_Silph_Co_4F_Trainer_1 = TRAINER * int(read_bit(game, 0xD829, 3))
  Beat_Silph_Co_4F_Trainer_2 = TRAINER * int(read_bit(game, 0xD829, 4))
  Silph_Co_4_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD82A, 0))
  Silph_Co_4_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD82A, 1))
  Beat_Silph_Co_5F_Trainer_0 = TRAINER * int(read_bit(game, 0xD82B, 2))
  Beat_Silph_Co_5F_Trainer_1 = TRAINER * int(read_bit(game, 0xD82B, 3))
  Beat_Silph_Co_5F_Trainer_2 = TRAINER * int(read_bit(game, 0xD82B, 4))
  Beat_Silph_Co_5F_Trainer_3 = TRAINER * int(read_bit(game, 0xD82B, 5))
  Silph_Co_5_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD82C, 0))
  Silph_Co_5_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD82C, 1))
  Silph_Co_5_Unlocked_Door3 = QUEST * int(read_bit(game, 0xD82C, 2))
  Beat_Silph_Co_6F_Trainer_0 = TRAINER * int(read_bit(game, 0xD82D, 6))
  Beat_Silph_Co_6F_Trainer_1 = TRAINER * int(read_bit(game, 0xD82D, 7))
  Beat_Silph_Co_6F_Trainer_2 = TRAINER * int(read_bit(game, 0xD82E, 0))
  Silph_Co_6_Unlocked_Door = QUEST * int(read_bit(game, 0xD82E, 7))
  Beat_Silph_Co_7F_Trainer_0 = TRAINER * int(read_bit(game, 0xD82F, 5))
  Beat_Silph_Co_7F_Trainer_1 = TRAINER * int(read_bit(game, 0xD82F, 6))
  Beat_Silph_Co_7F_Trainer_2 = TRAINER * int(read_bit(game, 0xD82F, 7))
  Beat_Silph_Co_7F_Trainer_3 = TRAINER * int(read_bit(game, 0xD830, 0))
  Silph_Co_7_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD830, 4))
  Silph_Co_7_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD830, 5))
  Silph_Co_7_Unlocked_Door3 = QUEST * int(read_bit(game, 0xD830, 6))
  Beat_Silph_Co_8F_Trainer_0 = TRAINER * int(read_bit(game, 0xD831, 2))
  Beat_Silph_Co_8F_Trainer_1 = TRAINER * int(read_bit(game, 0xD831, 3))
  Beat_Silph_Co_8F_Trainer_2 = TRAINER * int(read_bit(game, 0xD831, 4))
  Silph_Co_8_Unlocked_Door = QUEST * int(read_bit(game, 0xD832, 0))
  Beat_Silph_Co_9F_Trainer_0 = TRAINER * int(read_bit(game, 0xD833, 2))
  Beat_Silph_Co_9F_Trainer_1 = TRAINER * int(read_bit(game, 0xD833, 3))
  Beat_Silph_Co_9F_Trainer_2 = TRAINER * int(read_bit(game, 0xD833, 4))
  Silph_Co_9_Unlocked_Door1 = QUEST * int(read_bit(game, 0xD834, 0))
  Silph_Co_9_Unlocked_Door2 = QUEST * int(read_bit(game, 0xD834, 1))
  Silph_Co_9_Unlocked_Door3 = QUEST * int(read_bit(game, 0xD834, 2))
  Silph_Co_9_Unlocked_Door4 = QUEST * int(read_bit(game, 0xD834, 3))
  Beat_Silph_Co_10F_Trainer_0 = TRAINER * int(read_bit(game, 0xD835, 1))
  Beat_Silph_Co_10F_Trainer_1 = TRAINER * int(read_bit(game, 0xD835, 2))
  Silph_Co_10_Unlocked_Door = QUEST * int(read_bit(game, 0xD836, 0))
  Beat_Silph_Co_11F_Trainer_0 = TRAINER * int(read_bit(game, 0xD837, 4))
  Beat_Silph_Co_11F_Trainer_1 = TRAINER * int(read_bit(game, 0xD837, 5))
  Silph_Co_11_Unlocked_Door = QUEST * int(read_bit(game, 0xD838, 0))
  Got_Master_Ball = ITEM * int(read_bit(game, 0xD838, 5))
  Beat_Silph_Co_Giovanni = GYM_LEADER * int(read_bit(game, 0xD838, 7))
  Silph_Co_Receptionist_At_Desk = TASK * int(read_bit(game, 0xD7B9, 7))
  return sum([Beat_Silph_Co_2F_Trainer_0, Beat_Silph_Co_2F_Trainer_1, Beat_Silph_Co_2F_Trainer_2, Beat_Silph_Co_2F_Trainer_3, Silph_Co_2_Unlocked_Door1, 
              Silph_Co_2_Unlocked_Door2, Beat_Silph_Co_3F_Trainer_0, Beat_Silph_Co_3F_Trainer_1, Silph_Co_3_Unlocked_Door1, Silph_Co_3_Unlocked_Door2, 
              Beat_Silph_Co_4F_Trainer_0, Beat_Silph_Co_4F_Trainer_1, Beat_Silph_Co_4F_Trainer_2, Silph_Co_4_Unlocked_Door1, Silph_Co_4_Unlocked_Door2, 
              Beat_Silph_Co_5F_Trainer_0, Beat_Silph_Co_5F_Trainer_1, Beat_Silph_Co_5F_Trainer_2, Beat_Silph_Co_5F_Trainer_3, Silph_Co_5_Unlocked_Door1, 
              Silph_Co_5_Unlocked_Door2, Silph_Co_5_Unlocked_Door3, Beat_Silph_Co_6F_Trainer_0, Beat_Silph_Co_6F_Trainer_1, Beat_Silph_Co_6F_Trainer_2, 
              Silph_Co_6_Unlocked_Door, Beat_Silph_Co_7F_Trainer_0, Beat_Silph_Co_7F_Trainer_1, Beat_Silph_Co_7F_Trainer_2, Beat_Silph_Co_7F_Trainer_3, 
              Silph_Co_7_Unlocked_Door1, Silph_Co_7_Unlocked_Door2, Silph_Co_7_Unlocked_Door3, Beat_Silph_Co_8F_Trainer_0, Beat_Silph_Co_8F_Trainer_1, 
              Beat_Silph_Co_8F_Trainer_2, Silph_Co_8_Unlocked_Door, Beat_Silph_Co_9F_Trainer_0, Beat_Silph_Co_9F_Trainer_1, Beat_Silph_Co_9F_Trainer_2, 
              Silph_Co_9_Unlocked_Door1, Silph_Co_9_Unlocked_Door2, Silph_Co_9_Unlocked_Door3, Silph_Co_9_Unlocked_Door4, Beat_Silph_Co_10F_Trainer_0, 
              Beat_Silph_Co_10F_Trainer_1, Silph_Co_10_Unlocked_Door, Beat_Silph_Co_11F_Trainer_0, Beat_Silph_Co_11F_Trainer_1, Silph_Co_11_Unlocked_Door, 
              Got_Master_Ball, Beat_Silph_Co_Giovanni, Silph_Co_Receptionist_At_Desk])

def rock_tunnel(game):
  Beat_Rock_Tunnel_1_Trainer_0 = TRAINER * int(read_bit(game, 0xD7D2, 1))
  Beat_Rock_Tunnel_1_Trainer_1 = TRAINER * int(read_bit(game, 0xD7D2, 2))
  Beat_Rock_Tunnel_1_Trainer_2 = TRAINER * int(read_bit(game, 0xD7D2, 3))
  Beat_Rock_Tunnel_1_Trainer_3 = TRAINER * int(read_bit(game, 0xD7D2, 4))
  Beat_Rock_Tunnel_1_Trainer_4 = TRAINER * int(read_bit(game, 0xD7D2, 5))
  Beat_Rock_Tunnel_1_Trainer_5 = TRAINER * int(read_bit(game, 0xD7D2, 6))
  Beat_Rock_Tunnel_1_Trainer_6 = TRAINER * int(read_bit(game, 0xD7D2, 7))
  Beat_Rock_Tunnel_2_Trainer_0 = TRAINER * int(read_bit(game, 0xD87D, 1))
  Beat_Rock_Tunnel_2_Trainer_1 = TRAINER * int(read_bit(game, 0xD87D, 2))
  Beat_Rock_Tunnel_2_Trainer_2 = TRAINER * int(read_bit(game, 0xD87D, 3))
  Beat_Rock_Tunnel_2_Trainer_3 = TRAINER * int(read_bit(game, 0xD87D, 4))
  Beat_Rock_Tunnel_2_Trainer_4 = TRAINER * int(read_bit(game, 0xD87D, 5))
  Beat_Rock_Tunnel_2_Trainer_5 = TRAINER * int(read_bit(game, 0xD87D, 6))
  Beat_Rock_Tunnel_2_Trainer_6 = TRAINER * int(read_bit(game, 0xD87D, 7))
  Beat_Rock_Tunnel_2_Trainer_7 = TRAINER * int(read_bit(game, 0xD87E, 0))
  return sum([Beat_Rock_Tunnel_1_Trainer_0, Beat_Rock_Tunnel_1_Trainer_1, Beat_Rock_Tunnel_1_Trainer_2, Beat_Rock_Tunnel_1_Trainer_3, 
              Beat_Rock_Tunnel_1_Trainer_4, Beat_Rock_Tunnel_1_Trainer_5, Beat_Rock_Tunnel_1_Trainer_6, Beat_Rock_Tunnel_2_Trainer_0, 
              Beat_Rock_Tunnel_2_Trainer_1, Beat_Rock_Tunnel_2_Trainer_2, Beat_Rock_Tunnel_2_Trainer_3, Beat_Rock_Tunnel_2_Trainer_4, 
              Beat_Rock_Tunnel_2_Trainer_5, Beat_Rock_Tunnel_2_Trainer_6, Beat_Rock_Tunnel_2_Trainer_7])

def ssanne(game):
  Beat_Ss_Anne_5_Trainer_0 = TRAINER * int(read_bit(game, 0xD7FF, 4))
  Beat_Ss_Anne_5_Trainer_1 = TRAINER * int(read_bit(game, 0xD7FF, 5))
  Rubbed_Captains_Back = BILL_CAPT * int(read_bit(game, 0xD803, 1))
  Ss_Anne_Left = BILL_CAPT * int(read_bit(game, 0xD803, 2))
  Walked_Past_Guard_After_Ss_Anne_Left = BILL_CAPT * int(read_bit(game, 0xD803, 3))
  Started_Walking_Out_Of_Dock = BILL_CAPT * int(read_bit(game, 0xD803, 4))
  Walked_Out_Of_Dock = BILL_CAPT * int(read_bit(game, 0xD803, 5))
  Beat_Ss_Anne_8_Trainer_0 = TRAINER * int(read_bit(game, 0xD805, 1))
  Beat_Ss_Anne_8_Trainer_1 = TRAINER * int(read_bit(game, 0xD805, 2))
  Beat_Ss_Anne_8_Trainer_2 = TRAINER * int(read_bit(game, 0xD805, 3))
  Beat_Ss_Anne_8_Trainer_3 = TRAINER * int(read_bit(game, 0xD805, 4))
  Beat_Ss_Anne_9_Trainer_0 = TRAINER * int(read_bit(game, 0xD807, 1))
  Beat_Ss_Anne_9_Trainer_1 = TRAINER * int(read_bit(game, 0xD807, 2))
  Beat_Ss_Anne_9_Trainer_2 = TRAINER * int(read_bit(game, 0xD807, 3))
  Beat_Ss_Anne_9_Trainer_3 = TRAINER * int(read_bit(game, 0xD807, 4))
  Beat_Ss_Anne_10_Trainer_0 = TRAINER * int(read_bit(game, 0xD809, 1))
  Beat_Ss_Anne_10_Trainer_1 = TRAINER * int(read_bit(game, 0xD809, 2))
  Beat_Ss_Anne_10_Trainer_2 = TRAINER * int(read_bit(game, 0xD809, 3))
  Beat_Ss_Anne_10_Trainer_3 = TRAINER * int(read_bit(game, 0xD809, 4))
  Beat_Ss_Anne_10_Trainer_4 = TRAINER * int(read_bit(game, 0xD809, 5))
  Beat_Ss_Anne_10_Trainer_5 = TRAINER * int(read_bit(game, 0xD809, 6))
  return sum([Beat_Ss_Anne_5_Trainer_0, Beat_Ss_Anne_5_Trainer_1, Rubbed_Captains_Back, Ss_Anne_Left, 
              Walked_Past_Guard_After_Ss_Anne_Left, Started_Walking_Out_Of_Dock, Walked_Out_Of_Dock, Beat_Ss_Anne_8_Trainer_0, 
              Beat_Ss_Anne_8_Trainer_1, Beat_Ss_Anne_8_Trainer_2, Beat_Ss_Anne_8_Trainer_3, Beat_Ss_Anne_9_Trainer_0, 
              Beat_Ss_Anne_9_Trainer_1, Beat_Ss_Anne_9_Trainer_2, Beat_Ss_Anne_9_Trainer_3, Beat_Ss_Anne_10_Trainer_0, 
              Beat_Ss_Anne_10_Trainer_1, Beat_Ss_Anne_10_Trainer_2, Beat_Ss_Anne_10_Trainer_3, Beat_Ss_Anne_10_Trainer_4, Beat_Ss_Anne_10_Trainer_5])

def mtmoon(game):
  Beat_Mt_Moon_1_Trainer_1 = TRAINER * int(read_bit(game, 0xD7F5, 1))
  Beat_Mt_Moon_1_Trainer_2 = TRAINER * int(read_bit(game, 0xD7F5, 2))
  Beat_Mt_Moon_1_Trainer_3 = TRAINER * int(read_bit(game, 0xD7F5, 3))
  Beat_Mt_Moon_1_Trainer_4 = TRAINER * int(read_bit(game, 0xD7F5, 4))
  Beat_Mt_Moon_1_Trainer_5 = TRAINER * int(read_bit(game, 0xD7F5, 5))
  Beat_Mt_Moon_1_Trainer_6 = TRAINER * int(read_bit(game, 0xD7F5, 6))
  Beat_Mt_Moon_1_Trainer_7 = TRAINER * int(read_bit(game, 0xD7F5, 7))
  Beat_Mt_Moon_Super_Nerd = TRAINER * int(read_bit(game, 0xD7F6, 1))
  Beat_Mt_Moon_3_Trainer_0 = TRAINER * int(read_bit(game, 0xD7F6, 2))
  Beat_Mt_Moon_3_Trainer_1 = TRAINER * int(read_bit(game, 0xD7F6, 3))
  Beat_Mt_Moon_3_Trainer_2 = TRAINER * int(read_bit(game, 0xD7F6, 4))
  Beat_Mt_Moon_3_Trainer_3 = TRAINER * int(read_bit(game, 0xD7F6, 5))
  Got_Dome_Fossil = TASK * int(read_bit(game, 0xD7F6, 6))
  Got_Helix_Fossil = TASK * int(read_bit(game, 0xD7F6, 7))
  return sum([Beat_Mt_Moon_1_Trainer_1, Beat_Mt_Moon_1_Trainer_2, Beat_Mt_Moon_1_Trainer_3, Beat_Mt_Moon_1_Trainer_4, 
              Beat_Mt_Moon_1_Trainer_5, Beat_Mt_Moon_1_Trainer_6, Beat_Mt_Moon_1_Trainer_7, Beat_Mt_Moon_Super_Nerd, 
              Beat_Mt_Moon_3_Trainer_0, Beat_Mt_Moon_3_Trainer_1, Beat_Mt_Moon_3_Trainer_2, Beat_Mt_Moon_3_Trainer_3, 
              Got_Dome_Fossil, Got_Helix_Fossil])

def routes(game):
  route3_0 = TRAINER * int(read_bit(game, 0xD7C3, 2))
  route3_1 = TRAINER * int(read_bit(game, 0xD7C3, 3))
  route3_2 = TRAINER * int(read_bit(game, 0xD7C3, 4))
  route3_3 = TRAINER * int(read_bit(game, 0xD7C3, 5))
  route3_4 = TRAINER * int(read_bit(game, 0xD7C3, 6))
  route3_5 = TRAINER * int(read_bit(game, 0xD7C3, 7))
  route3_6 = TRAINER * int(read_bit(game, 0xD7C4, 0))
  route3_7 = TRAINER * int(read_bit(game, 0xD7C4, 1))

  route4_0 = TRAINER * int(read_bit(game, 0xD7C5, 2))

  route24_rocket = TRAINER * int(read_bit(game, 0xD7EF, 1))
  route24_0 = TRAINER * int(read_bit(game, 0xD7EF, 2))
  route24_1 = TRAINER * int(read_bit(game, 0xD7EF, 3))
  route24_2 = TRAINER * int(read_bit(game, 0xD7EF, 4))
  route24_3 = TRAINER * int(read_bit(game, 0xD7EF, 5))
  route24_4 = TRAINER * int(read_bit(game, 0xD7EF, 6))
  route24_5 = TRAINER * int(read_bit(game, 0xD7EF, 7))

  route25_0 = TRAINER * int(read_bit(game, 0xD7F1, 1))
  route25_1 = TRAINER * int(read_bit(game, 0xD7F1, 2))
  route25_2 = TRAINER * int(read_bit(game, 0xD7F1, 3))
  route25_3 = TRAINER * int(read_bit(game, 0xD7F1, 4))
  route25_4 = TRAINER * int(read_bit(game, 0xD7F1, 5))
  route25_5 = TRAINER * int(read_bit(game, 0xD7F1, 6))
  route25_6 = TRAINER * int(read_bit(game, 0xD7F1, 7))
  route25_7 = TRAINER * int(read_bit(game, 0xD7F2, 0))
  route25_8 = TRAINER * int(read_bit(game, 0xD7F2, 1))

  route9_0 = TRAINER * int(read_bit(game, 0xD7CF, 1))
  route9_1 = TRAINER * int(read_bit(game, 0xD7CF, 2))
  route9_2 = TRAINER * int(read_bit(game, 0xD7CF, 3))
  route9_3 = TRAINER * int(read_bit(game, 0xD7CF, 4))
  route9_4 = TRAINER * int(read_bit(game, 0xD7CF, 5))
  route9_5 = TRAINER * int(read_bit(game, 0xD7CF, 6))
  route9_6 = TRAINER * int(read_bit(game, 0xD7CF, 7))
  route9_7 = TRAINER * int(read_bit(game, 0xD7D0, 0))
  route9_8 = TRAINER * int(read_bit(game, 0xD7D0, 1))

  route6_0 = TRAINER * int(read_bit(game, 0xD7C9, 1))
  route6_1 = TRAINER * int(read_bit(game, 0xD7C9, 2))
  route6_2 = TRAINER * int(read_bit(game, 0xD7C9, 3))
  route6_3 = TRAINER * int(read_bit(game, 0xD7C9, 4))
  route6_4 = TRAINER * int(read_bit(game, 0xD7C9, 5))
  route6_5 = TRAINER * int(read_bit(game, 0xD7C9, 6))

  route11_0 = TRAINER * int(read_bit(game, 0xD7D5, 1))
  route11_1 = TRAINER * int(read_bit(game, 0xD7D5, 2))
  route11_2 = TRAINER * int(read_bit(game, 0xD7D5, 3))
  route11_3 = TRAINER * int(read_bit(game, 0xD7D5, 4))
  route11_4 = TRAINER * int(read_bit(game, 0xD7D5, 5))
  route11_5 = TRAINER * int(read_bit(game, 0xD7D5, 6))
  route11_6 = TRAINER * int(read_bit(game, 0xD7D5, 7))
  route11_7 = TRAINER * int(read_bit(game, 0xD7D6, 0))
  route11_8 = TRAINER * int(read_bit(game, 0xD7D6, 1))
  route11_9 = TRAINER * int(read_bit(game, 0xD7D6, 2))

  route8_0 = TRAINER * int(read_bit(game, 0xD7CD, 1))
  route8_1 = TRAINER * int(read_bit(game, 0xD7CD, 2))
  route8_2 = TRAINER * int(read_bit(game, 0xD7CD, 3))
  route8_3 = TRAINER * int(read_bit(game, 0xD7CD, 4))
  route8_4 = TRAINER * int(read_bit(game, 0xD7CD, 5))
  route8_5 = TRAINER * int(read_bit(game, 0xD7CD, 6))
  route8_6 = TRAINER * int(read_bit(game, 0xD7CD, 7))
  route8_7 = TRAINER * int(read_bit(game, 0xD7CE, 0))
  route8_8 = TRAINER * int(read_bit(game, 0xD7CE, 1))

  route10_0 = TRAINER * int(read_bit(game, 0xD7D1, 1))
  route10_1 = TRAINER * int(read_bit(game, 0xD7D1, 2))
  route10_2 = TRAINER * int(read_bit(game, 0xD7D1, 3))
  route10_3 = TRAINER * int(read_bit(game, 0xD7D1, 4))
  route10_4 = TRAINER * int(read_bit(game, 0xD7D1, 5))
  route10_5 = TRAINER * int(read_bit(game, 0xD7D1, 6))

  route12_0 = TRAINER * int(read_bit(game, 0xD7D7, 2))
  route12_1 = TRAINER * int(read_bit(game, 0xD7D7, 3))
  route12_2 = TRAINER * int(read_bit(game, 0xD7D7, 4))
  route12_3 = TRAINER * int(read_bit(game, 0xD7D7, 5))
  route12_4 = TRAINER * int(read_bit(game, 0xD7D7, 6))
  route12_5 = TRAINER * int(read_bit(game, 0xD7D7, 7))
  route12_6 = TRAINER * int(read_bit(game, 0xD7D8, 0))

  route16_0 = TRAINER * int(read_bit(game, 0xD7DF, 1))
  route16_1 = TRAINER * int(read_bit(game, 0xD7DF, 2))
  route16_2 = TRAINER * int(read_bit(game, 0xD7DF, 3))
  route16_3 = TRAINER * int(read_bit(game, 0xD7DF, 4))
  route16_4 = TRAINER * int(read_bit(game, 0xD7DF, 5))
  route16_5 = TRAINER * int(read_bit(game, 0xD7DF, 6))

  route17_0 = TRAINER * int(read_bit(game, 0xD7E1, 1))
  route17_1 = TRAINER * int(read_bit(game, 0xD7E1, 2))
  route17_2 = TRAINER * int(read_bit(game, 0xD7E1, 3))
  route17_3 = TRAINER * int(read_bit(game, 0xD7E1, 4))
  route17_4 = TRAINER * int(read_bit(game, 0xD7E1, 5))
  route17_5 = TRAINER * int(read_bit(game, 0xD7E1, 6))
  route17_6 = TRAINER * int(read_bit(game, 0xD7E1, 7))
  route17_7 = TRAINER * int(read_bit(game, 0xD7E2, 0))
  route17_8 = TRAINER * int(read_bit(game, 0xD7E2, 1))
  route17_9 = TRAINER * int(read_bit(game, 0xD7E2, 2))

  route13_0 = TRAINER * int(read_bit(game, 0xD7D9, 1))
  route13_1 = TRAINER * int(read_bit(game, 0xD7D9, 2))
  route13_2 = TRAINER * int(read_bit(game, 0xD7D9, 3))
  route13_3 = TRAINER * int(read_bit(game, 0xD7D9, 4))
  route13_4 = TRAINER * int(read_bit(game, 0xD7D9, 5))
  route13_5 = TRAINER * int(read_bit(game, 0xD7D9, 6))
  route13_6 = TRAINER * int(read_bit(game, 0xD7D9, 7))
  route13_7 = TRAINER * int(read_bit(game, 0xD7DA, 0))
  route13_8 = TRAINER * int(read_bit(game, 0xD7DA, 1))
  route13_9 = TRAINER * int(read_bit(game, 0xD7DA, 2))

  route14_0 = TRAINER * int(read_bit(game, 0xD7DB, 1))
  route14_1 = TRAINER * int(read_bit(game, 0xD7DB, 2))
  route14_2 = TRAINER * int(read_bit(game, 0xD7DB, 3))
  route14_3 = TRAINER * int(read_bit(game, 0xD7DB, 4))
  route14_4 = TRAINER * int(read_bit(game, 0xD7DB, 5))
  route14_5 = TRAINER * int(read_bit(game, 0xD7DB, 6))
  route14_6 = TRAINER * int(read_bit(game, 0xD7DB, 7))
  route14_7 = TRAINER * int(read_bit(game, 0xD7DC, 0))
  route14_8 = TRAINER * int(read_bit(game, 0xD7DC, 1))
  route14_9 = TRAINER * int(read_bit(game, 0xD7DC, 2))

  route15_0 = TRAINER * int(read_bit(game, 0xD7DD, 1))
  route15_1 = TRAINER * int(read_bit(game, 0xD7DD, 2))
  route15_2 = TRAINER * int(read_bit(game, 0xD7DD, 3))
  route15_3 = TRAINER * int(read_bit(game, 0xD7DD, 4))
  route15_4 = TRAINER * int(read_bit(game, 0xD7DD, 5))
  route15_5 = TRAINER * int(read_bit(game, 0xD7DD, 6))
  route15_6 = TRAINER * int(read_bit(game, 0xD7DD, 7))
  route15_7 = TRAINER * int(read_bit(game, 0xD7DE, 0))
  route15_8 = TRAINER * int(read_bit(game, 0xD7DE, 1))
  route15_9 = TRAINER * int(read_bit(game, 0xD7DE, 2))

  route18_0 = TRAINER * int(read_bit(game, 0xD7E3, 1))
  route18_1 = TRAINER * int(read_bit(game, 0xD7E3, 2))
  route18_2 = TRAINER * int(read_bit(game, 0xD7E3, 3))

  route19_0 = TRAINER * int(read_bit(game, 0xD7E5, 1))
  route19_1 = TRAINER * int(read_bit(game, 0xD7E5, 2))
  route19_2 = TRAINER * int(read_bit(game, 0xD7E5, 3))
  route19_3 = TRAINER * int(read_bit(game, 0xD7E5, 4))
  route19_4 = TRAINER * int(read_bit(game, 0xD7E5, 5))
  route19_5 = TRAINER * int(read_bit(game, 0xD7E5, 6))
  route19_6 = TRAINER * int(read_bit(game, 0xD7E5, 7))
  route19_7 = TRAINER * int(read_bit(game, 0xD7E6, 0))
  route19_8 = TRAINER * int(read_bit(game, 0xD7E6, 1))
  route19_9 = TRAINER * int(read_bit(game, 0xD7E6, 2))

  route20_0 = TRAINER * int(read_bit(game, 0xD7E7, 1))
  route20_1 = TRAINER * int(read_bit(game, 0xD7E7, 2))
  route20_2 = TRAINER * int(read_bit(game, 0xD7E7, 3))
  route20_3 = TRAINER * int(read_bit(game, 0xD7E7, 4))
  route20_4 = TRAINER * int(read_bit(game, 0xD7E7, 5))
  route20_5 = TRAINER * int(read_bit(game, 0xD7E7, 6))
  route20_6 = TRAINER * int(read_bit(game, 0xD7E7, 7))
  route20_7 = TRAINER * int(read_bit(game, 0xD7E8, 0))
  route20_8 = TRAINER * int(read_bit(game, 0xD7E8, 1))
  route20_9 = TRAINER * int(read_bit(game, 0xD7E8, 2))

  route21_0 = TRAINER * int(read_bit(game, 0xD7E9, 1))
  route21_1 = TRAINER * int(read_bit(game, 0xD7E9, 2))
  route21_2 = TRAINER * int(read_bit(game, 0xD7E9, 3))
  route21_3 = TRAINER * int(read_bit(game, 0xD7E9, 4))
  route21_4 = TRAINER * int(read_bit(game, 0xD7E9, 5))
  route21_5 = TRAINER * int(read_bit(game, 0xD7E9, 6))
  route21_6 = TRAINER * int(read_bit(game, 0xD7E9, 7))
  route21_7 = TRAINER * int(read_bit(game, 0xD7EA, 0))
  route21_8 = TRAINER * int(read_bit(game, 0xD7EA, 1))

  return sum([ route3_0, route3_1, route3_2, route3_3, route3_4, route3_5, route3_6, route3_7, 
              route4_0, route24_rocket, route24_0, route24_1, route24_2, route24_3, route24_4, 
              route24_5, route25_0, route25_1, route25_2, route25_3, route25_4, route25_5, route25_6, 
              route25_7, route25_8, route9_0, route9_1, route9_2, route9_3, route9_4, route9_5, 
              route9_6, route9_7, route9_8, route6_0, route6_1, route6_2, route6_3, route6_4, 
              route6_5, route11_0, route11_1, route11_2, route11_3, route11_4, route11_5, route11_6, 
              route11_7, route11_8, route11_9, route8_0, route8_1, route8_2, route8_3, route8_4, route8_5, 
              route8_6, route8_7, route8_8, route10_0, route10_1, route10_2, route10_3, route10_4, route10_5, 
              route12_0, route12_1, route12_2, route12_3, route12_4, route12_5, route12_6, route16_0, 
              route16_1, route16_2, route16_3, route16_4, route16_5, route17_0, route17_1, route17_2, 
              route17_3, route17_4, route17_5, route17_6, route17_7, route17_8, route17_9, route13_0, 
              route13_1, route13_2, route13_3, route13_4, route13_5, route13_6, route13_7, route13_8, 
              route13_9, route14_0, route14_1, route14_2, route14_3, route14_4, route14_5, route14_6, 
              route14_7, route14_8, route14_9, route15_0, route15_1, route15_2, route15_3, route15_4, 
              route15_5, route15_6, route15_7, route15_8, route15_9,  route18_0, route18_1, route18_2, 
              route19_0, route19_1, route19_2, route19_3, route19_4, route19_5, route19_6,  route19_7, 
              route19_8, route19_9, route20_0, route20_1, route20_2, route20_3, route20_4, route20_5, 
              route20_6,  route20_7, route20_8, route20_9, route21_0, route21_1, route21_2, route21_3, 
              route21_4, route21_5, route21_6,  route21_7, route21_8])

def misc(game):
    bought_magikarp = TASK * int(read_bit(game, 0xD7C6, 7))
    hall_of_fame_dex_rating = TASK * int(read_bit(game, 0xD747, 3))
    daisy_walking = TASK * int(read_bit(game, 0xD74A, 2))
    # bought_museum_ticket = TASK * int(read_bit(game, 0xD754, 0))
    got_old_amber = TASK * int(read_bit(game, 0xD754, 1))
    got_bike_voucher = TASK * int(read_bit(game, 0xD771, 1))
    got_10_coins = TASK * int(read_bit(game, 0xD77E, 2))
    got_20_coins_1 = TASK * int(read_bit(game, 0xD77E, 3))
    got_20_coins_2 = TASK * int(read_bit(game, 0xD77E, 4))
    got_coin_case = TASK * int(read_bit(game, 0xD783, 0))
    got_potion_sample = TASK * int(read_bit(game, 0xD7BF, 0))
    got_itemfinder = TASK * int(read_bit(game, 0xD7D6, 7))
    got_exp_all = TASK * int(read_bit(game, 0xD7DD, 0))
    rescued_mr_fuji = TASK * int(read_bit(game, 0xD7E0, 7))
    beat_mewtwo = TASK * int(read_bit(game, 0xD85F, 1))
    rescued_mr_fuji_2 = TASK * int(read_bit(game, 0xD769, 7))

    return sum([bought_magikarp, hall_of_fame_dex_rating, daisy_walking,
                got_old_amber, got_bike_voucher, got_10_coins, got_20_coins_1, got_20_coins_2,
                got_coin_case, got_potion_sample, got_itemfinder, got_exp_all, rescued_mr_fuji,
                beat_mewtwo, rescued_mr_fuji_2]) #  bought_museum_ticket,

def snorlax(game):
  route12_snorlax_fight = POKEMON * int(read_bit(game, 0xD7D8, 6))
  route12_snorlax_beat = POKEMON * int(read_bit(game, 0xD7D8, 7))
  route16_snorlax_fight = POKEMON * int(read_bit(game, 0xD7E0, 0))
  route16_snorlax_beat = POKEMON * int(read_bit(game, 0xD7E0, 1))

  return sum([route12_snorlax_fight, route12_snorlax_beat, route16_snorlax_fight, route16_snorlax_beat])

def hmtm(game):
  hm01 = HM * int(read_bit(game, 0xD803, 0))
  hm02 = HM * int(read_bit(game, 0xD7E0, 6))
  hm03 = HM * int(read_bit(game, 0xD857, 0))
  hm04 = HM * int(read_bit(game, 0xD78E, 0))
  hm05 = HM * int(read_bit(game, 0xD7C2, 0))

  tm34 = TM * int(read_bit(game, 0xD755, 6))
  tm11 = TM * int(read_bit(game, 0xD75E, 6))
  tm41 = TM * int(read_bit(game, 0xD777, 0))
  tm13 = TM * int(read_bit(game, 0xD778, 4))
  tm48 = TM * int(read_bit(game, 0xD778, 5))
  tm49 = TM * int(read_bit(game, 0xD778, 6))
  tm18 = TM * int(read_bit(game, 0xD778, 7))
  tm21 = TM * int(read_bit(game, 0xD77C, 0))
  tm06 = TM * int(read_bit(game, 0xD792, 0))
  tm24 = TM * int(read_bit(game, 0xD773, 6))
  tm29 = TM * int(read_bit(game, 0xD7BD, 0))
  tm31 = TM * int(read_bit(game, 0xD7AF, 0))
  tm35 = TM * int(read_bit(game, 0xD7A1, 7))
  tm36 = TM * int(read_bit(game, 0xD826, 7))
  tm38 = TM * int(read_bit(game, 0xD79A, 0))
  tm27 = TM * int(read_bit(game, 0xD751, 0))
  tm42 = TM * int(read_bit(game, 0xD74C, 1))
  tm46 = TM * int(read_bit(game, 0xD7B3, 0))
  tm39 = TM * int(read_bit(game, 0xD7D7, 0))


  return sum([hm01, hm02, hm03, hm04, hm05, tm34, tm11, tm41, tm13, tm48, tm49, tm18, tm21, tm06, tm24, tm29, tm31, tm35, tm36, tm38, tm27, tm42, tm46, tm39])

def bill(game):
  met_bill = BILL_CAPT * int(read_bit(game, 0xD7F1, 0))
  used_cell_separator_on_bill = BILL_CAPT * int(read_bit(game, 0xD7F2, 3))
  got_ss_ticket = BILL_CAPT * int(read_bit(game, 0xD7F2, 4))
  met_bill_2 = BILL_CAPT * int(read_bit(game, 0xD7F2, 5))
  bill_said_use_cell_separator = BILL_CAPT * int(read_bit(game, 0xD7F2, 6))
  left_bills_house_after_helping = BILL_CAPT * int(read_bit(game, 0xD7F2, 7))

  return sum([met_bill, used_cell_separator_on_bill, got_ss_ticket, met_bill_2, bill_said_use_cell_separator, left_bills_house_after_helping])

def oak(game):
  oak_appeared_in_pallet = TASK * int(read_bit(game, 0xD74B, 7))
  followed_oak_into_lab = TASK * int(read_bit(game, 0xD747, 0))
  oak_asked_to_choose_mon = TASK * int(read_bit(game, 0xD74B, 1))
  got_starter = TASK * int(read_bit(game, 0xD74B, 2))
  followed_oak_into_lab_2 = TASK * int(read_bit(game, 0xD74B, 0))
  got_pokedex = QUEST * int(read_bit(game, 0xD74B, 5))
  got_oaks_parcel = QUEST * int(read_bit(game, 0xD74E, 1))
  pallet_after_getting_pokeballs = QUEST * int(read_bit(game, 0xD747, 6))
  oak_got_parcel = QUEST * int(read_bit(game, 0xD74E, 0))
  got_pokeballs_from_oak = TASK * int(read_bit(game, 0xD74B, 4))
  pallet_after_getting_pokeballs_2 = TASK * int(read_bit(game, 0xD74B, 6))

  return sum([oak_appeared_in_pallet, followed_oak_into_lab, oak_asked_to_choose_mon, got_starter, followed_oak_into_lab_2, got_pokedex, 
              got_oaks_parcel, pallet_after_getting_pokeballs, oak_got_parcel, got_pokeballs_from_oak, pallet_after_getting_pokeballs_2])

def towns(game):
  got_town_map = TASK * int(read_bit(game, 0xD74A, 0))
  entered_blues_house = TASK * int(read_bit(game, 0xD74A, 1))
  beat_viridian_forest_trainer_0 = TRAINER * int(read_bit(game, 0xD7F3, 2))
  beat_viridian_forest_trainer_1 = TRAINER * int(read_bit(game, 0xD7F3, 3))
  beat_viridian_forest_trainer_2 = TRAINER * int(read_bit(game, 0xD7F3, 4))
  got_nugget = TASK * int(read_bit(game, 0xD7EF, 0))
  nugget_reward_available = TASK * int(read_bit(game, 0xD7F0, 1))
  beat_cerulean_rocket_thief = TRAINER * int(read_bit(game, 0xD75B, 7))
  got_bicycle = QUEST * int(read_bit(game, 0xD75F, 0))
  seel_fan_boast = TASK * int(read_bit(game, 0xD771, 6))
  pikachu_fan_boast = TASK * int(read_bit(game, 0xD771, 7))
  got_poke_flute = QUEST * int(read_bit(game, 0xD76C, 0))

  return sum([got_town_map, entered_blues_house, beat_viridian_forest_trainer_0,
    beat_viridian_forest_trainer_1, beat_viridian_forest_trainer_2, got_nugget,
    nugget_reward_available, beat_cerulean_rocket_thief, got_bicycle,
    seel_fan_boast, pikachu_fan_boast, got_poke_flute])

def lab(game):
  gave_fossil_to_lab = TASK * int(read_bit(game, 0xD7A3, 0))
  lab_still_reviving_fossil = TASK * int(read_bit(game, 0xD7A3, 1))
  lab_handing_over_fossil_mon = TASK * int(read_bit(game, 0xD7A3, 2))

  return sum([gave_fossil_to_lab, lab_still_reviving_fossil, lab_handing_over_fossil_mon])

def mansion(game):
  beat_mansion_2_trainer_0 = TRAINER * int(read_bit(game, 0xD847, 1))
  beat_mansion_3_trainer_0 = TRAINER * int(read_bit(game, 0xD849, 1))
  beat_mansion_3_trainer_1 = TRAINER * int(read_bit(game, 0xD849, 2))
  beat_mansion_4_trainer_0 = TRAINER * int(read_bit(game, 0xD84B, 1))
  beat_mansion_4_trainer_1 = TRAINER * int(read_bit(game, 0xD84B, 2))
  mansion_switch_on = QUEST * int(read_bit(game, 0xD796, 0))
  beat_mansion_1_trainer_0 = TRAINER * int(read_bit(game, 0xD798, 1))

  return sum([beat_mansion_2_trainer_0, beat_mansion_3_trainer_0, beat_mansion_3_trainer_1,
    beat_mansion_4_trainer_0, beat_mansion_4_trainer_1, mansion_switch_on, beat_mansion_1_trainer_0])

def safari(game):
  gave_gold_teeth = QUEST * int(read_bit(game, 0xD78E, 1))
  safari_game_over = EVENT * int(read_bit(game, 0xD790, 6))
  in_safari_zone = EVENT * int(read_bit(game, 0xD790, 7))

  return sum([gave_gold_teeth, safari_game_over, in_safari_zone])

def dojo(game):
  defeated_fighting_dojo = BAD * int(read_bit(game, 0xD7B1, 0))
  beat_karate_master = GYM_LEADER * int(read_bit(game, 0xD7B1, 1))
  beat_dojo_trainer_0 = TRAINER * int(read_bit(game, 0xD7B1, 2))
  beat_dojo_trainer_1 = TRAINER * int(read_bit(game, 0xD7B1, 3))
  beat_dojo_trainer_2 = TRAINER * int(read_bit(game, 0xD7B1, 4))
  beat_dojo_trainer_3 = TRAINER * int(read_bit(game, 0xD7B1, 5))
  got_hitmonlee = POKEMON * int(read_bit(game, 0xD7B1, 6))
  got_hitmonchan = POKEMON * int(read_bit(game, 0xD7B1, 7))

  return sum([defeated_fighting_dojo, beat_karate_master, beat_dojo_trainer_0,
    beat_dojo_trainer_1, beat_dojo_trainer_2, beat_dojo_trainer_3,
    got_hitmonlee, got_hitmonchan])

def hideout(game):
  beat_rocket_hideout_1_trainer_0 = GYM_TRAINER * int(read_bit(game, 0xD815, 1))
  beat_rocket_hideout_1_trainer_1 = GYM_TRAINER * int(read_bit(game, 0xD815, 2))
  beat_rocket_hideout_1_trainer_2 = GYM_TRAINER * int(read_bit(game, 0xD815, 3))
  beat_rocket_hideout_1_trainer_3 = GYM_TRAINER * int(read_bit(game, 0xD815, 4))
  beat_rocket_hideout_1_trainer_4 = GYM_TRAINER * int(read_bit(game, 0xD815, 5))
  beat_rocket_hideout_2_trainer_0 = GYM_TRAINER * int(read_bit(game, 0xD817, 1))
  beat_rocket_hideout_3_trainer_0 = GYM_TRAINER * int(read_bit(game, 0xD819, 1))
  beat_rocket_hideout_3_trainer_1 = GYM_TRAINER * int(read_bit(game, 0xD819, 2))
  beat_rocket_hideout_4_trainer_0 = GYM_TRAINER * int(read_bit(game, 0xD81B, 2))
  beat_rocket_hideout_4_trainer_1 = GYM_TRAINER * int(read_bit(game, 0xD81B, 3))
  beat_rocket_hideout_4_trainer_2 = GYM_TRAINER * int(read_bit(game, 0xD81B, 4))
  rocket_hideout_4_door_unlocked = QUEST * int(read_bit(game, 0xD81B, 5))
  rocket_dropped_lift_key = QUEST * int(read_bit(game, 0xD81B, 6))
  beat_rocket_hideout_giovanni = GYM_LEADER * int(read_bit(game, 0xD81B, 7))
  found_rocket_hideout = QUEST * int(read_bit(game, 0xD77E, 1))

  return sum([beat_rocket_hideout_1_trainer_0, beat_rocket_hideout_1_trainer_1, beat_rocket_hideout_1_trainer_2, beat_rocket_hideout_1_trainer_3,
    beat_rocket_hideout_1_trainer_4, beat_rocket_hideout_2_trainer_0, beat_rocket_hideout_3_trainer_0, beat_rocket_hideout_3_trainer_1,
    beat_rocket_hideout_4_trainer_0, beat_rocket_hideout_4_trainer_1, beat_rocket_hideout_4_trainer_2, rocket_hideout_4_door_unlocked,
    rocket_dropped_lift_key, beat_rocket_hideout_giovanni, found_rocket_hideout])

def poke_tower(game):
  beat_pokemontower_3_trainer_0 = TRAINER * int(read_bit(game, 0xD765, 1))
  beat_pokemontower_3_trainer_1 = TRAINER * int(read_bit(game, 0xD765, 2))
  beat_pokemontower_3_trainer_2 = TRAINER * int(read_bit(game, 0xD765, 3))
  beat_pokemontower_4_trainer_0 = TRAINER * int(read_bit(game, 0xD766, 1))
  beat_pokemontower_4_trainer_1 = TRAINER * int(read_bit(game, 0xD766, 2))
  beat_pokemontower_4_trainer_2 = TRAINER * int(read_bit(game, 0xD766, 3))
  beat_pokemontower_5_trainer_0 = TRAINER * int(read_bit(game, 0xD767, 2))
  beat_pokemontower_5_trainer_1 = TRAINER * int(read_bit(game, 0xD767, 3))
  beat_pokemontower_5_trainer_2 = TRAINER * int(read_bit(game, 0xD767, 4))
  beat_pokemontower_5_trainer_3 = TRAINER * int(read_bit(game, 0xD767, 5))
#   in_purified_zone = EVENT * int(read_bit(game, 0xD767, 7)) # purified zone
  beat_pokemontower_6_trainer_0 = TRAINER * int(read_bit(game, 0xD768, 1))
  beat_pokemontower_6_trainer_1 = TRAINER * int(read_bit(game, 0xD768, 2))
  beat_pokemontower_6_trainer_2 = TRAINER * int(read_bit(game, 0xD768, 3))
  beat_ghost_marowak = QUEST * int(read_bit(game, 0xD768, 7))
  beat_pokemontower_7_trainer_0 = TRAINER * int(read_bit(game, 0xD769, 1))
  beat_pokemontower_7_trainer_1 = TRAINER * int(read_bit(game, 0xD769, 2))
  beat_pokemontower_7_trainer_2 = TRAINER * int(read_bit(game, 0xD769, 3))

  return sum([beat_pokemontower_3_trainer_0, beat_pokemontower_3_trainer_1, beat_pokemontower_3_trainer_2, beat_pokemontower_4_trainer_0,
    beat_pokemontower_4_trainer_1, beat_pokemontower_4_trainer_2, beat_pokemontower_5_trainer_0, beat_pokemontower_5_trainer_1,
    beat_pokemontower_5_trainer_2, beat_pokemontower_5_trainer_3, beat_pokemontower_6_trainer_0,
    beat_pokemontower_6_trainer_1, beat_pokemontower_6_trainer_2, beat_ghost_marowak, beat_pokemontower_7_trainer_0,
    beat_pokemontower_7_trainer_1, beat_pokemontower_7_trainer_2]) # in_purified_zone,

def gym1(game):
   #gym 1 Pewter	
    one = GYM_LEADER * int(read_bit(game, 0xD755, 7))
    g1_1 = GYM_TRAINER * int(read_bit(game, 0xD755, 2)) #	"0xD755-2": "Beat Pewter Gym Trainer 0",
    return sum([one, g1_1, ])

def gym2(game):
   #gym 2 Cerulean	
    two = GYM_LEADER * int(read_bit(game, 0xD75E, 7))
    g2_1 = GYM_TRAINER * int(read_bit(game, 0xD75E, 2)) #	"0xD75E-2": "Beat Cerulean Gym Trainer 0",
    g2_2 = GYM_TRAINER * int(read_bit(game, 0xD75E, 3)) #	"0xD75E-3": "Beat Cerulean Gym Trainer 1",
    return sum([two, g2_1, g2_2, ])

def gym3(game):
   #gym 3 Vermilion	
    lock_one = GYM_TASK * int(read_bit(game, 0xD773, 1)) # "0xD773-1": "1S Lock Opened",
    lock_two = GYM_TASK * int(read_bit(game, 0xD773, 0))# "0xD773-0": "2Nd Lock Opened",
    three = GYM_LEADER * int(read_bit(game, 0xD773, 7))
    g3_1 = GYM_TRAINER * int(read_bit(game, 0xD773, 2)) #	"0xD773-2": "Beat Vermilion Gym Trainer 0",
    g3_2 = GYM_TRAINER * int(read_bit(game, 0xD773, 3)) #	"0xD773-3": "Beat Vermilion Gym Trainer 1",
    g3_3 = GYM_TRAINER * int(read_bit(game, 0xD773, 4)) #	"0xD773-4": "Beat Vermilion Gym Trainer 2",
    return sum([three, g3_1, g3_2, g3_3, lock_one, lock_two])

def gym4(game):
   #gym 4 Celadon	
    four = GYM_LEADER * int(read_bit(game, 0xD792, 1))
    g4_1 = GYM_TRAINER * int(read_bit(game, 0xD77C, 2)) #	"0xD77C-2": "Beat Celadon Gym Trainer 0",
    g4_2 = GYM_TRAINER * int(read_bit(game, 0xD77C, 3)) #	"0xD77C-3": "Beat Celadon Gym Trainer 1",
    g4_3 = GYM_TRAINER * int(read_bit(game, 0xD77C, 4)) #	"0xD77C-4": "Beat Celadon Gym Trainer 2",
    g4_4 = GYM_TRAINER * int(read_bit(game, 0xD77C, 5)) #	"0xD77C-5": "Beat Celadon Gym Trainer 3",
    g4_5 = GYM_TRAINER * int(read_bit(game, 0xD77C, 6)) #	"0xD77C-6": "Beat Celadon Gym Trainer 4",
    g4_6 = GYM_TRAINER * int(read_bit(game, 0xD77C, 7)) #	"0xD77C-7": "Beat Celadon Gym Trainer 5",
    g4_7 = GYM_TRAINER * int(read_bit(game, 0xD77D, 0)) #	"0xD77D-0": "Beat Celadon Gym Trainer 6",
    return sum([four, g4_1, g4_2, g4_3, g4_4, g4_5, g4_6, g4_7, ])

def gym5(game):
   #gym 5 Fuchsia	
    five = GYM_LEADER * int(read_bit(game, 0xD7B3, 1))
    g5_1 = GYM_TRAINER * int(read_bit(game, 0xD792, 2)) #	"0xD792-2": "Beat Fuchsia Gym Trainer 0",
    g5_2 = GYM_TRAINER * int(read_bit(game, 0xD792, 3)) #	"0xD792-3": "Beat Fuchsia Gym Trainer 1",
    g5_3 = GYM_TRAINER * int(read_bit(game, 0xD792, 4)) #	"0xD792-4": "Beat Fuchsia Gym Trainer 2",
    g5_4 = GYM_TRAINER * int(read_bit(game, 0xD792, 5)) #	"0xD792-5": "Beat Fuchsia Gym Trainer 3",
    g5_5 = GYM_TRAINER * int(read_bit(game, 0xD792, 6)) #	"0xD792-6": "Beat Fuchsia Gym Trainer 4",
    g5_6 = GYM_TRAINER * int(read_bit(game, 0xD792, 7)) #	"0xD792-7": "Beat Fuchsia Gym Trainer 5",
    return sum([five, g5_1, g5_2, g5_3, g5_4, g5_5, g5_6, ])

def gym6(game):
   #gym 6 Saffron	
    six = GYM_LEADER * int(read_bit(game, 0xD7B3, 1))
    g6_1 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 2)) #	"0xD7B3-2": "Beat Saffron Gym Trainer 0",
    g6_2 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 3)) #	"0xD7B3-3": "Beat Saffron Gym Trainer 1",
    g6_3 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 4)) #	"0xD7B3-4": "Beat Saffron Gym Trainer 2",
    g6_4 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 5)) #	"0xD7B3-5": "Beat Saffron Gym Trainer 3",
    g6_5 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 6)) #	"0xD7B3-6": "Beat Saffron Gym Trainer 4",
    g6_6 = GYM_TRAINER * int(read_bit(game, 0xD7B3, 7)) #	"0xD7B3-7": "Beat Saffron Gym Trainer 5",
    g6_7 = GYM_TRAINER * int(read_bit(game, 0xD7B4, 0)) #	"0xD7B4-0": "Beat Saffron Gym Trainer 6",
    return sum([six, g6_1, g6_2, g6_3, g6_4, g6_5, g6_6, g6_7, ])

def gym7(game):
   #gym 7 Cinnabar	
    seven = GYM_LEADER * int(read_bit(game, 0xD79A, 1))
    g7_1 = GYM_TRAINER * int(read_bit(game, 0xD79A, 2)) #	"0xD79A-2": "Beat Cinnabar Gym Trainer 0",
    g7_2 = GYM_TRAINER * int(read_bit(game, 0xD79A, 3)) #	"0xD79A-3": "Beat Cinnabar Gym Trainer 1",
    g7_3 = GYM_TRAINER * int(read_bit(game, 0xD79A, 4)) #	"0xD79A-4": "Beat Cinnabar Gym Trainer 2",
    g7_4 = GYM_TRAINER * int(read_bit(game, 0xD79A, 5)) #	"0xD79A-5": "Beat Cinnabar Gym Trainer 3",
    g7_5 = GYM_TRAINER * int(read_bit(game, 0xD79A, 6)) #	"0xD79A-6": "Beat Cinnabar Gym Trainer 4",
    g7_6 = GYM_TRAINER * int(read_bit(game, 0xD79A, 7)) #	"0xD79A-7": "Beat Cinnabar Gym Trainer 5",
    g7_7 = GYM_TRAINER * int(read_bit(game, 0xD79B, 0)) #	"0xD79B-0": "Beat Cinnabar Gym Trainer 6",

    return sum([seven, g7_1, g7_2, g7_3, g7_4, g7_5, g7_6, g7_7, ])

def gym8(game):
   #gym 8 Viridian	
  # "0xD74C-0": "Viridian Gym Open",
  gym_door = GYM_TASK * int(read_bit(game, 0xD74C, 0))
  eight = GYM_LEADER * int(read_bit(game, 0xD751, 1))
  g8_1 = GYM_TRAINER * int(read_bit(game, 0xD751, 2)) #	"0xD751-2": "Beat Viridian Gym Trainer 0",
  g8_2 = GYM_TRAINER * int(read_bit(game, 0xD751, 3)) #	"0xD751-3": "Beat Viridian Gym Trainer 1",
  g8_3 = GYM_TRAINER * int(read_bit(game, 0xD751, 4)) #	"0xD751-4": "Beat Viridian Gym Trainer 2",
  g8_4 = GYM_TRAINER * int(read_bit(game, 0xD751, 5)) #	"0xD751-5": "Beat Viridian Gym Trainer 3",
  g8_5 = GYM_TRAINER * int(read_bit(game, 0xD751, 6)) #	"0xD751-6": "Beat Viridian Gym Trainer 4",
  g8_6 = GYM_TRAINER * int(read_bit(game, 0xD751, 7)) #	"0xD751-7": "Beat Viridian Gym Trainer 5",
  g8_7 = GYM_TRAINER * int(read_bit(game, 0xD752, 0)) #	"0xD752-0": "Beat Viridian Gym Trainer 6",
  g8_8 = GYM_TRAINER * int(read_bit(game, 0xD752, 1)) #	"0xD752-1": "Beat Viridian Gym Trainer 7",
  return sum([eight, g8_1, g8_2, g8_3, g8_4, g8_5, g8_6, g8_7, g8_8, gym_door])

def rival(game):
  one = RIVAL * int(read_bit(game, 0xD74B, 3))
  two = RIVAL * int(read_bit(game, 0xD7EB, 0))
  three = RIVAL * int(read_bit(game, 0xD7EB, 1))
  four = RIVAL * int(read_bit(game, 0xD7EB, 5))
  five = RIVAL * int(read_bit(game, 0xD7EB, 6))
  six = RIVAL * int(read_bit(game, 0xD75A, 0))
  seven = RIVAL * int(read_bit(game, 0xD764, 6))
  eight = RIVAL * int(read_bit(game, 0xD764, 7))
  nine = RIVAL * int(read_bit(game, 0xD7EB, 7))
  Beat_Silph_Co_Rival = RIVAL * int(read_bit(game, 0xD82F, 0))

  return sum([one, two, three, four, five, six, seven, eight, nine, Beat_Silph_Co_Rival])

######################################################################################################

def bcd(num):
    return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

def bit_count(bits):
    return bin(bits).count("1")

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit - 1] == "1"

def mem_val(game, addr):
    mem = game.get_memory_value(addr)
    return mem

def read_uint16(game, start_addr):
    """Read 2 bytes"""
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256 * val_256 + val_1

######################################################################################################

def get_hm_count(game):
    hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
    items = get_items_in_bag(game)
    total_hm_cnt = 0
    for hm_id in hm_ids:
        if hm_id in items:
            total_hm_cnt += 1
    return total_hm_cnt * 1

def get_items_in_bag(game, one_indexed=0):
    first_item = 0xD31E
    item_ids = []
    for i in range(0, 40, 2):
        item_id = game.get_memory_value(first_item + i)
        if item_id == 0 or item_id == 0xff:
            break
        item_ids.append(item_id + one_indexed)
    return item_ids

def position(game):
    r_pos = game.get_memory_value(Y_POS_ADDR)
    c_pos = game.get_memory_value(X_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    if r_pos >= 443:
        r_pos = 444
    if r_pos <= 0:
        r_pos = 0
    if c_pos >= 443:
        c_pos = 444
    if c_pos <= 0:
        c_pos = 0
    if map_n > 247:
        map_n = 247
    if map_n < -1:
        map_n = -1
    return r_pos, c_pos, map_n

def party(game):
    # party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [x for x in [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR] if x > 0]
    return party_size, party_levels # [x for x in party_levels if x > 0]

def hp(game):
    """Percentage of total party HP"""
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]
    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1
    return sum(party_hp) / sum_max_hp

def used_cut(game):
    return game.get_memory_value(WCUTTILE)

def write_mem(game, addr, value):
    mem = game.set_memory_value(addr, value)
    return mem

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)