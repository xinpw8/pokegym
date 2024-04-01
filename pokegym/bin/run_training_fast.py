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
import torch.nn as nn

from custom_feature_extractor import CustomFeatureExtractor
from red_env_constants import *



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
    ep_length = 2048 * 2
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
    #file_name = '../' + "saved_runs/session_6ff6aae5/poke_355532800_steps.zip"

    model = None
    checkpoint_exists = exists(file_name)
    if len(file_name) != 0 and not checkpoint_exists:
        print('\nERROR: Checkpoint not found!')
        exit(1)
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
                    verbose=1, n_steps=2048 // 4, batch_size=512, n_epochs=3, gamma=0.997, ent_coef=0.01,
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * 128 * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
