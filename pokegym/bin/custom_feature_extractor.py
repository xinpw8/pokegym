import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from red_env_constants import *


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)


        # Screen Class
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layer for coordinates
        self.coordinates_fc = nn.Sequential(
            nn.Linear(3 * 8, features_dim),  # Flattened size of coordinates, repeated 3 times
            nn.ReLU()
        )


        # Game Class
        self.game_state_lstm = nn.LSTM(input_size = (130 + 7) * OBSERVATION_MEMORY_SIZE, hidden_size=features_dim, batch_first=True)

        # Move Class
        self.player_moves_embedding = nn.Embedding(num_embeddings=256, embedding_dim=8)
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

        self.move_fc = nn.Sequential(
            nn.Linear(4032, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 32),
            nn.ReLU(),
        )

        # Pokemon Class
        self.player_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))
        self.pokemon_fc = nn.Sequential(
            nn.Linear(1938, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # Player Class
        self.player_fc = nn.Sequential(
            nn.Linear(96, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 32),
            nn.ReLU(),
        )

        # Player Fighter Class
        self.player_fighter_fc = nn.Sequential(
            nn.Linear(305, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # Battle Turn Class
        self.battle_turn_fc = nn.Sequential(
            nn.Linear(613, features_dim),
            nn.ReLU(),
        )

        # Enemy Battle Class
        self.enemy_battle_fc = nn.Sequential(
            nn.Linear(324, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 32),
            nn.ReLU(),
        )

        self.progress_fc = nn.Sequential(
            nn.Linear(24, features_dim),
            nn.ReLU(),
        )

        # Items Class
        self.items_fc = nn.Sequential(
            nn.Linear(5141, features_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=4),
        )


        # World
        self.mart_fc = nn.Sequential(
            nn.Linear(2561, features_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=4),
        )

        self.pc_fc = nn.Sequential(
            nn.Linear(10240, 8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=4),
        )

        # Fully connected layers for output
        self.fc_layers = nn.Sequential(
            nn.Linear(1027, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )


    def forward(self, observations):
        # Explicitly use batch_size for reshaping, after n_steps there will be a big batch
        batch_size = observations["visited"].size(0)
        device = observations["screen"].device


        # Screen Class
        screen_input = torch.cat([
            observations["screen"],
            observations["visited"],
        ], dim=1)

        # Convert the screen to spatial representation
        screen_features = self.cnn(screen_input).to(device)


        # Game Class
        coordinates_input = observations["coordinates"].view(batch_size, -1)
        coordinates_features = self.coordinates_fc(coordinates_input).to(device)

        action_input = observations["action"].int().view(batch_size, -1).to(device).float()
        game_state_input = observations["game_state"].int().view(batch_size, -1).to(device).float()
        game_input = torch.cat([
            action_input,
            game_state_input,
        ], dim=1)

        game_state_lstm_features, _ = self.game_state_lstm(game_input)


        # Move Class
        player_moves_input = observations["player_moves"].view(batch_size, -1)
        player_pp = observations["player_pp"].view(batch_size, -1)

        moves_input = torch.cat([
            player_moves_input,
            player_pp

        ], dim=1)

        moves_features = self.move_fc(moves_input).to(device)


        # Pokemon Class
        player_pokemon_input = observations["player_pokemon"].view(batch_size, -1)
        player_levels_input = observations["player_levels"].view(batch_size, -1)
        player_types_input = observations["player_types"].view(batch_size, -1)
        player_hp_input = observations["player_hp"].view(batch_size, -1)
        player_xp_input = observations["player_xp"].view(batch_size, -1)
        player_stats_input = observations["player_stats"].view(batch_size, -1)
        player_status_input = observations["player_status"].view(batch_size, -1)

        pokemon_input = torch.cat([
            player_pokemon_input,
            player_levels_input,
            player_types_input,
            player_hp_input,
            player_xp_input,
            player_stats_input,
            player_status_input,
        ], dim=1)
        
        pokemon_features = self.pokemon_fc(pokemon_input).to(device)


        # Player Class
        player_input = torch.cat([
            pokemon_features,
            moves_features,
        ], dim=1)
        player_features = self.player_fc(player_input).to(device)

        # Player Fighter Class
        player_head_index = observations["player_head_index"].view(batch_size, -1)
        player_head_pokemon = observations["player_head_pokemon"].view(batch_size, -1)
        player_modifiers_input = observations["player_modifiers"].view(batch_size, -1)
        type_hint_input = observations["type_hint"].view(batch_size, -1)

        player_fighter_input = torch.cat([
            player_head_index,
            player_head_pokemon,
            player_modifiers_input,
            type_hint_input,
            player_features,  # TODO: Can we focus on just the head pokemon?, and breakout player to global fc
        ], dim=1)
        player_fighter_features = self.player_fighter_fc(player_fighter_input).to(device)


        # Enemy Battle Class
        enemy_head_input = observations["enemy_head"].view(batch_size, -1)
        enemy_level_input = observations["enemy_level"].view(batch_size, -1)
        enemy_hp_input = observations["enemy_hp"].view(batch_size, -1)
        enemy_types_input = observations["enemy_types"].view(batch_size, -1)
        enemy_modifiers_input = observations["enemy_modifiers"].view(batch_size, -1)
        enemy_status_input = observations["enemy_status"].view(batch_size, -1)

        enemy_battle_input = torch.cat([
            enemy_head_input,
            enemy_level_input,
            enemy_hp_input,
            enemy_types_input,
            enemy_modifiers_input,
            enemy_status_input,
        ], dim=1)
        enemy_battle_features = self.enemy_battle_fc(enemy_battle_input).to(device)


        # Battle Turn Class
        battle_type_input = observations["battle_type"].view(batch_size, -1)
        enemies_left_input = observations["enemies_left"].view(batch_size, -1)
        move_selection_input = observations["move_selection"].view(batch_size, -1)  # TODO: Players move w/ history to LTSM

        battle_turn_input = torch.cat([
            battle_type_input,
            enemies_left_input,
            move_selection_input,
            player_fighter_features,
            enemy_battle_features,
        ], dim=1)
        battle_turn_features = self.battle_turn_fc(battle_turn_input).to(device)


        # Progress Class
        badges_input = observations["badges"].view(batch_size, -1)
        pokecenters_input = observations["pokecenters"].view(batch_size, -1)
        progress_features = self.progress_fc(torch.cat([
            badges_input,
            pokecenters_input,
        ], dim=1)).to(device)


        # Items Class
        money_input = observations["money"].view(batch_size, -1)
        bag_ids_input = observations["bag_ids"].view(batch_size, -1)
        bag_quantities_input = observations["bag_quantities"].view(batch_size, -1)
        item_features = self.items_fc(torch.cat([
            money_input,
            bag_ids_input,
            bag_quantities_input,
        ], dim=1)).to(device)


        # World
        audio_input = observations["audio"].view(batch_size, -1)
        age_input = observations["age"].view(batch_size, -1)


        # Final FC layer
        combined_input = torch.cat([
            screen_features,
            coordinates_features,
            game_state_lstm_features,
            battle_turn_features,
            badges_input,
            pokecenters_input,
            item_features,
            audio_input,
            age_input,
        ], dim=1)


        return self.fc_layers(combined_input).to(device)