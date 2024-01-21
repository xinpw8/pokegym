import os
from os.path import exists
from pathlib import Path
import uuid

import numpy as np
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from red_env_constants import *



class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # Define CNN architecture for spatial inputs
	    # Note: Possible to do 2x convo(out 16, then 32) learns a little faster & explores a little better before 50M at the cost of size, both equal around 50M, 2x overfits after 50M
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layer for coordinates
        self.coordinates_fc = nn.Sequential(
            nn.Linear(3 * 8, features_dim),  # Flattened size of coordinates, repeated 3 times
            nn.ReLU()
        )

        # Game Class
        self.game_state_lstm = nn.LSTM(input_size=1584, hidden_size=features_dim, batch_first=True)

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

        # Fully connected layers for output
        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )


    def forward(self, observations):
        # Explicitly use batch_size for reshaping, after n_steps there will be a big batch
        batch_size = observations["visited"].size(0)
        device = observations["screen"].device  # Assuming 'screen' is part of your observations

        combined_input = torch.cat([observations["screen"].unsqueeze(1),
                                    observations["visited"].unsqueeze(1),
                                    observations["walkable"].unsqueeze(1)], dim=1)

        # Apply CNN to spatial inputs
        screen_features = self.cnn(combined_input).to(device)

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
        #player_moves_input = self.player_moves_embedding(observations["player_moves"].to(torch.int)).view(batch_size, -1)
        player_moves_input = observations["player_moves"].view(batch_size, -1)
        player_pp = observations["player_pp"].view(batch_size, -1)

        moves_input = torch.cat([
            player_moves_input,
            player_pp

        ], dim=1)

        moves_features = self.move_fc(moves_input).to(device)
        #moves_features = self.move_max_pool(moves_input.unsqueeze(1)).squeeze(1)  # TODO: Try with and without pooling


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
        #pokemon_features = self.player_max_pool(pokemon_input.unsqueeze(1)).squeeze(1)  # TODO: Try with and without pooling


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


        # Final FC layer
        combined_input = torch.cat([
            screen_features,
            coordinates_features,
            game_state_lstm_features,
            battle_turn_features,
            badges_input,
            pokecenters_input,
        ], dim=1)


        return self.fc_layers(combined_input).to(device)


def make_env(thread_id, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param id: (int) index of the subprocess
    """

    def _init():
        return RedGymEnv(thread_id, env_conf)

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    use_wandb_logging = True
    ep_length = 2048 * 1
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'../saved_runs/session_{sess_id}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../pokemon_ai_squirt_poke_balls.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': False, 'reward_scale': 1, 'extra_buttons': False,
        'explore_weight': 3  # 2.5
    }

    num_cpu = 1  # Also sets the number of episodes per training iteration

    if 0 < num_cpu < 50:
        env_config['debug'] = True
        env_config['headless'] = False
        use_wandb_logging = False

    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config, GLOBAL_SEED) for i in range(num_cpu)])
    #env = DummyVecEnv([make_env(i, env_config, GLOBAL_SEED) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length * 1, save_path=os.path.abspath(sess_path),
                                             name_prefix='poke')

    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=sess_path,
        )
        callbacks.append(WandbCallback())

    # put a checkpoint here you want to start from
    file_name = ''
    # file_name = '../' + "saved_runs/session_c62778b9/poke_125452288_steps.zip"

    model = None
    checkpoint_exists = exists(file_name)
    if len(file_name) != 0 and not checkpoint_exists:
        print('\nERROR: Checkpoint not found!')
    elif checkpoint_exists:
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # policy_kwargs={"features_extractor_class": CustomFeatureExtractor, "features_extractor_kwargs": {"features_dim": 64}},
        model = PPO("MultiInputPolicy", env, policy_kwargs={"features_extractor_class": CustomFeatureExtractor, "features_extractor_kwargs": {"features_dim": 64}},
                    verbose=1, n_steps=2048 // 4, batch_size=512, n_epochs=3, gamma=0.998, ent_coef=0.01,
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
