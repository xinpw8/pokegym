Available actions:
0: <class 'pokegym.pyboy_binding.Down'>
1: <class 'pokegym.pyboy_binding.Left'>
2: <class 'pokegym.pyboy_binding.Right'>
3: <class 'pokegym.pyboy_binding.Up'>
4: <class 'pokegym.pyboy_binding.A'>
5: <class 'pokegym.pyboy_binding.B'>
6: <class 'pokegym.pyboy_binding.Start'>
7: <class 'pokegym.pyboy_binding.Select'>

event_reward=63

bill_capt_rew: 63
seen_pokemon_reward: 40
caught_pokemon_reward: 184.0
moves_obtained_reward: 12.0
bill_reward: 16.0
hm_reward: 5
level_reward: 0
death_reward: 33
badges_reward: 0
healing_reward: 20
exploration_reward: 0
cut_rew: 0.02
that_guy: 0.0
cut_coords: 0.0
cut_tiles: 0.0
tree_distance_reward: 0.0
dojo_reward: 0.0
hideout_reward: 0
lemonade_in_bag_reward: 0
silph_scope_in_bag_reward: 0
lift_key_in_bag_reward: 0
pokedoll_in_bag_reward: 0
bicycle_in_bag_reward: 0
special_location_rewards: 0
can_reward: 44


Total Reward: 0.0
event: 63
level: 33
opponent_level: 0.138
badges: 20
bill_saved_reward: 5
exploration: 0.02
seen_pokemon_reward: 184.0
caught_pokemon_reward: 12.0
moves_obtained_reward: 16.0

total_sum=333.158

new Reward: 0


event_reward=63

bill_capt_rew: 63
seen_pokemon_reward: 40
caught_pokemon_reward: 184.0
moves_obtained_reward: 12.0
bill_reward: 16.0
hm_reward: 5
level_reward: 0
death_reward: 33
badges_reward: 0
healing_reward: 20
exploration_reward: 0
cut_rew: 0.04
that_guy: 4.0
cut_coords: 0.1
cut_tiles: 0.0
tree_distance_reward: 0.0
dojo_reward: 0.0
hideout_reward: 0
lemonade_in_bag_reward: 0
silph_scope_in_bag_reward: 0
lift_key_in_bag_reward: 0
pokedoll_in_bag_reward: 0
bicycle_in_bag_reward: 0
special_location_rewards: 0
can_reward: 44
Total Reward: 0.0
delta: 1700.3600000000001
event: 63
level: 33
opponent_level: 0.138
badges: 20
bill_saved_reward: 5
exploration: 0.04
seen_pokemon_reward: 184.0
caught_pokemon_reward: 12.0
moves_obtained_reward: 16.0
used_cut_reward: 8

total_sum=2041.538

new Reward: 1700.3600000000001


event_reward=63

bill_capt_rew: 63
seen_pokemon_reward: 40
caught_pokemon_reward: 184.0
moves_obtained_reward: 12.0
bill_reward: 16.0
hm_reward: 5
level_reward: 0
death_reward: 33
badges_reward: 0
healing_reward: 20
exploration_reward: 0
cut_rew: 0.06
that_guy: 4.0
cut_coords: 0.1
cut_tiles: 0.0
tree_distance_reward: 0.0
dojo_reward: 0.0
hideout_reward: 0
lemonade_in_bag_reward: 0
silph_scope_in_bag_reward: 0
lift_key_in_bag_reward: 0
pokedoll_in_bag_reward: 0
bicycle_in_bag_reward: 0
special_location_rewards: 0
can_reward: 44
Total Reward: 0.0
delta: 0.07999999999992724
event: 63
level: 33
opponent_level: 0.138
badges: 20
bill_saved_reward: 5
exploration: 0.06
seen_pokemon_reward: 184.0
caught_pokemon_reward: 12.0
moves_obtained_reward: 16.0
used_cut_reward: 8

total_sum=341.2779999999999

new Reward: 0.07999999999992724



        special_location_reward = (dojo_events_reward + silph_co_events_reward + 
               hideout_events_reward + poke_tower_events_reward + 
               gym3_events_reward + gym4_events_reward + 
               gym5_events_reward + gym6_events_reward + 
               gym7_events_reward)

        self.compute_and_print_rewards(event_reward, bill_capt_rew, seen_pokemon_reward, caught_pokemon_reward, moves_obtained_reward, bill_reward, hm_reward, level_reward, death_reward, badges_reward, healing_reward, self.exploration_reward, cut_rew, that_guy, cut_coords, cut_tiles, tree_distance_reward, dojo_reward, hideout_reward, self.has_lemonade_in_bag_reward, self.has_silph_scope_in_bag_reward, self.has_lift_key_in_bag_reward, self.has_pokedoll_in_bag_reward, self.has_bicycle_in_bag_reward, special_location_reward, self.can_reward)


        reward, self.last_reward = self.subtract_previous_reward_v1(self.last_reward, reward)
        
            def subtract_previous_reward_v1(self, last_reward, reward):
        if self.last_reward is None:
            updated_reward = 0
            updated_last_reward = 0
        else:
            updated_reward = reward - self.last_reward
            updated_last_reward = reward
        return updated_reward, updated_last_reward