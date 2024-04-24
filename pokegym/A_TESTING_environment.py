from pdb import set_trace as T
import numpy as np
import cv2

import os
import random
import time
import uuid
import logging

from collections import defaultdict
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.utils
import pufferlib.emulation
import pufferlib.vectorization
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool

from collections import deque
import sys
from pathlib import Path
working_dir = Path.cwd()
sys.path.append(f'{working_dir}/clean_pufferl.py')
import json
import pokemon_red_eval
from datetime import datetime
import heapq
import math
from datetime import timedelta
import io
from multiprocessing import Queue
from typing import Any, Callable
from eval import make_pokemon_red_overlay


@pufferlib.dataclass
class Performance:
    total_uptime = 0
    total_updates = 0
    total_agent_steps = 0
    epoch_time = 0
    epoch_sps = 0
    evaluation_time = 0
    evaluation_sps = 0
    evaluation_memory = 0
    evaluation_pytorch_memory = 0
    env_time = 0
    env_sps = 0
    inference_time = 0
    inference_sps = 0
    train_time = 0
    train_sps = 0
    train_memory = 0
    train_pytorch_memory = 0
    misc_time = 0

@pufferlib.dataclass
class Losses:
    policy_loss = 0
    value_loss = 0
    entropy = 0
    old_approx_kl = 0
    approx_kl = 0
    clipfrac = 0
    explained_variance = 0

@pufferlib.dataclass
class Charts:
    global_step = 0
    SPS = 0
    learning_rate = 0

def create(
        self: object = None,
        config: pufferlib.namespace = None,
        exp_name: str = None,
        track: bool = False,
        # Agent
        agent: nn.Module = None,
        agent_creator: callable = None,
        agent_kwargs: dict = None,
        # Environment
        env_creator: callable = None,
        env_creator_kwargs: dict = None,
        vectorization: ... = pufferlib.vectorization.Serial,
        # Policy Pool options
        policy_selector: callable = pufferlib.policy_pool.random_selector,
    ):


    env_send_queues = env_creator_kwargs["async_config"]["send_queues"]
    env_recv_queues = env_creator_kwargs["async_config"]["recv_queues"]

    # # At the end of the clean_pufferl.create function
    # print(f"Send Queues: {data.env_send_queues}, Receive Queues: {data.env_recv_queues}")
    
    
    # Easy logic for dir struct experiments/{exp_name}/sessions
    # Get the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    num_dirs = len([d for d in working_dir.iterdir() if d.is_dir()])
    sequence_number = num_dirs - 1
    filename = f"{date_time}_{sequence_number}_"
        
    new_exp_name = str(uuid.uuid4())[:8]
    new_exp_name = filename + new_exp_name
    print(f'line33 CLEANPUFFERL: {new_exp_name}')
    experiments_base_dir = working_dir / 'experiments'
    experiment_dir = experiments_base_dir / new_exp_name
    required_resources_path = experiment_dir / "required_resources"
    required_resources_path.mkdir(parents=True, exist_ok=True)
    files = ["running_experiment.txt", "clean_pufferl_log.log", "percent_complete.txt",] #, "test_exp.txt", "stats.txt"]
    for file_name in files:
        file_path = required_resources_path / file_name
        file_path.touch(exist_ok=True)
    running_experiment_file_path = required_resources_path / "running_experiment.txt"
    log_file_path = required_resources_path / "clean_pufferl_log.log"
    percent_complete_path = required_resources_path / "percent_complete.txt"
    # test_exp_file_path = required_resources_path / "test_exp.txt"
    print(f'\n{experiments_base_dir}\n{experiment_dir}\n{required_resources_path}\n{running_experiment_file_path}\n{percent_complete_path}\n')
    exp_name = f"{new_exp_name}"
    with open(running_experiment_file_path, 'w') as file:
        file.write(f"{exp_name}")
        
        
    # Set up logging
    logger = logging.getLogger(f'')
    logger.setLevel(logging.INFO)  # Set the base level to debug
    
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.INFO)  # Set the file handler's level

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)
    logger.propagate = False

    # Example of a log message
    logger.info(f"Logging set up for {new_exp_name}") 
    
    
    if config is None:
        config = pufferlib.args.CleanPuffeRL() 
    # Check if exp_name is set, otherwise generate a new one
    if exp_name is None:
        exp_name = new_exp_name
        # exp_name = str(uuid.uuid4())[:8]
    
    wandb = None
    if track:
        import wandb

    start_time = time.time()
    seed_everything(config.seed, config.torch_deterministic)
    total_updates = config.total_timesteps // config.batch_size

    device = config.device

    # # Write parsed config to file; environment.py reads for initialization
    # with open(test_exp_file_path, 'w') as file:
    #     file.write(f"{config}")    

    # # Create environments, agent, and optimizer
    # init_profiler = pufferlib.utils.Profiler(memory=True)
    # with init_profiler:
    #     pool = vectorization(
    #         env_creator,
    #         env_kwargs=env_creator_kwargs,
    #         num_envs=config.num_envs,
    #         envs_per_worker=config.envs_per_worker,
    #         envs_per_batch=config.envs_per_batch,
    #         env_pool=config.env_pool,
    #         mask_agents=True,
    #     )
    #     print(f'pool=cprl  {pool}')
    
    # Create environments, agent, and optimizer
    init_profiler = pufferlib.utils.Profiler(memory=True)
    with init_profiler:
        pool = vectorization(
            env_creator,
            env_kwargs=env_creator_kwargs,
            num_envs=config.num_envs,
            envs_per_worker=config.envs_per_worker,
            envs_per_batch=config.envs_per_batch,
            env_pool=config.env_pool,
            mask_agents=True,
        )
        print(f'pool created with configuration: {pool}')

    # Ensure the pool has been created before using it
    if not pool:
        raise Exception("Failed to create pool. Check vectorization and env_creator configurations.")

    # Reset the pool with the provided seed
    try:
        pool.async_reset(config.seed)
        print("Pool has been reset with the seed.")
    except AttributeError as e:
        print(f"Error during pool reset: {e}")
        raise

    obs_shape = pool.single_observation_space.shape
    atn_shape = pool.single_action_space.shape
    num_agents = pool.agents_per_env
    total_agents = num_agents * config.num_envs
    
    agent = pufferlib.emulation.make_object(
        agent, agent_creator, [pool.driver_env], agent_kwargs
        )
    
    resume_state = {}
    
    # Assuming `path` is correctly set to the experiment's directory
    path = os.path.join(config.data_dir, exp_name)
    if os.path.exists(path):
        print(f'path={path}')
        # Assume trainer_state.pt is the main file you want to load
        trainer_path = os.path.join(path, 'trainer_state.pt')
        
        if os.path.exists(trainer_path):
            resume_state = torch.load(trainer_path)
            # Assuming resume_state contains a key to indicate the model version or step
            # For example, resume_state might contain: {"update": 1, "model_name": "model_state", ...}
            model_version = str(resume_state["update"]).zfill(6)  # This pads the number with zeros, e.g., 1 becomes 000001
            model_filename = f"model_{model_version}_state.pth"  # Construct model filename based on the version
            model_path = os.path.join(path, model_filename)
            
            if os.path.exists(model_path):
                agent.load_state_dict(torch.load(model_path, map_location=device))
                print(f'Resumed from update {resume_state["update"]} '
                    f'with policy {model_filename}')
            else:
                print(f'Model file not found: {model_path}')
        else:
            print(f'Trainer state file not found: {trainer_path}')
    else:
        print(f'Experiment directory not found: {path}')
        
    # Some data to preserve run parameters when loading a saved model
    global_step = resume_state.get("global_step", 0)
    agent_step = resume_state.get("agent_step", 0)
    update = resume_state.get("update", 0)
    lr_update = resume_state.get("lr_update", 0) # BET ADDED 20
   
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    uncompiled_agent = agent # Needed to save the model
    opt_state = resume_state.get("optimizer_state_dict", None)
    
    if config.compile:
        agent = torch.compile(agent, mode=config.compile_mode)

    if config.verbose:
        n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"Model Size: {n_params//1000} K parameters")

    opt_state = resume_state.get("optimizer_state_dict", None)
    if config.load_optimizer_state is True and opt_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

    # Create policy pool
    pool_agents = num_agents * pool.envs_per_batch
    policy_pool = pufferlib.policy_pool.PolicyPool(
        agent, 
        pool_agents, 
        atn_shape, 
        device, 
        path,
        config.pool_kernel, 
        policy_selector,
    )


    # Allocate Storage
    storage_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
    pool.async_reset(config.seed)
    next_lstm_state = []
    # pool.async_reset(config.seed)
    # next_lstm_state = None
    
    # BET ADDED 15 (through line 172)
    if hasattr(agent, "lstm"):
        shape = (agent.lstm.num_layers, total_agents, agent.lstm.hidden_size)
        next_lstm_state = (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )
    obs=torch.zeros(config.batch_size + 1, *obs_shape, pin_memory=True) # added , pin_memory=True)
    actions=torch.zeros(config.batch_size + 1, *atn_shape, dtype=int)
    logprobs=torch.zeros(config.batch_size + 1)
    rewards=torch.zeros(config.batch_size + 1)
    dones=torch.zeros(config.batch_size + 1)
    truncateds=torch.zeros(config.batch_size + 1)
    values=torch.zeros(config.batch_size + 1)

    obs_ary = np.asarray(obs)
    actions_ary = np.asarray(actions)
    logprobs_ary = np.asarray(logprobs)
    rewards_ary = np.asarray(rewards)
    dones_ary = np.asarray(dones)
    truncateds_ary = np.asarray(truncateds)
    values_ary = np.asarray(values)
    
    storage_profiler.stop()

    init_performance = pufferlib.namespace(
        init_time = time.time() - start_time,
        init_env_time = init_profiler.elapsed,
        init_env_memory = init_profiler.memory,
        tensor_memory = storage_profiler.memory,
        tensor_pytorch_memory = storage_profiler.pytorch_memory,
    )
    # Load map data
    with open('pokegym/pokegym/map_data.json', 'r') as file:
        map_data = json.load(file)
    total_envs = map_data
 
    return pufferlib.namespace(self,        
        # Agent, Optimizer, and Environment
        config=config,
        pool = pool,
        agent = agent,
        uncompiled_agent = uncompiled_agent,
        optimizer = optimizer,
        policy_pool = policy_pool,

        # Logging
        exp_name = exp_name,
        track = track, # BET ADDED 17
        wandb = wandb,
        learning_rate=config.learning_rate,
        lr_update = lr_update,
        losses = Losses(),
        init_performance = init_performance,
        performance = Performance(),

        # Storage
        sort_keys = [],
        next_lstm_state = next_lstm_state,
        obs = obs,
        actions = actions,
        logprobs = logprobs,
        rewards = rewards,
        dones = dones,
        values = values,
        # BET ADDED 22
        reward_buffer = deque(maxlen=1_000),
        taught_cut = False,
        total_envs = total_envs,
        map_counts = defaultdict(int), # np.zeros(246),
        env_reports = defaultdict(int), 
        infos = {},
        stats = {},
        obs_ary = obs_ary,
        actions_ary = actions_ary,
        logprobs_ary = logprobs_ary,
        rewards_ary = rewards_ary,
        dones_ary = dones_ary,
        truncateds_ary = truncateds_ary,
        values_ary = values_ary,
        # Logging
        logger = logger,
        percent_complete_path = percent_complete_path,
        
        # Swarming
        env_send_queues = env_send_queues,
        env_recv_queues = env_recv_queues,

        # Misc
        total_updates = total_updates,
        update = update,
        global_step = global_step,
        device = device,
        start_time = start_time,
    )

@pufferlib.utils.profile
def evaluate(data):
    config = data.config
    if data.wandb is not None and data.performance.total_uptime > 0:
        # Prepare the dictionary you plan to log
        log_dict = {
            'SPS': data.SPS,
            'global_step': data.global_step,
            'learning_rate': data.optimizer.param_groups[0]["lr"],
            **{f'losses/{k}': v for k, v in data.losses.items()},
            **{f'performance/{k}': v for k, v in data.performance.items()},
            **{f'stats/{k}': v for k, v in data.stats.items()},
            **{f"max_stats/{k}": v for k, v in data.max_stats.items()},
            **{f'skillrank/{policy}': elo for policy, elo in data.policy_pool.ranker.ratings.items()},
        }
        
        # Log the data
        data.wandb.log(log_dict)
        
        
        # Default values to empty lists if keys are absent
        badge_counts = data.infos["learner"].get("stats/badges", [])
        bill_counts = data.infos["learner"].get("stats/bill_saved", [])
        rocket_hideout_completions = data.infos["learner"].get("stats/beat_rocket_hideout_giovanni", [])
        pokemon_tower_completions = data.infos["learner"].get("stats/rescued_mr_fuji", [])
        silph_co_completions = data.infos["learner"].get("stats/beat_silph_co_giovanni", [])

        # Calculate the percentage of environments with current badge level or higher
        num_envs_with_badge_1 = sum(1 for badge in badge_counts if badge >= 1)
        print(f'{num_envs_with_badge_1}')
        num_envs_with_badge_2 = sum(1 for badge in badge_counts if badge >= 2)
        num_envs_with_badge_3 = sum(1 for badge in badge_counts if badge >= 3)
        num_envs_with_badge_4 = sum(1 for badge in badge_counts if badge >= 4)
        num_envs_with_badge_5 = sum(1 for badge in badge_counts if badge >= 5)
        num_envs_with_badge_6 = sum(1 for badge in badge_counts if badge >= 6)
        num_envs_with_badge_7 = sum(1 for badge in badge_counts if badge >= 7)
        num_envs_with_badge_8 = sum(1 for badge in badge_counts if badge >= 8)
        bill_saved = sum(1 for bill in bill_counts if bill)
        rocket_hideout_completions = sum(1 for progress in rocket_hideout_completions if progress)
        pokemon_tower_completions = sum(1 for progress in pokemon_tower_completions if progress)
        silph_co_completions = sum(1 for progress in silph_co_completions if progress)
        
        # Convert counts to percentages
        total_envs = data.config.num_envs
        completion_dict = {
            "badge_1": f"{num_envs_with_badge_1 / total_envs:4f}",
            "badge_2": f"{num_envs_with_badge_2 / total_envs:4f}",
            "badge_3": f"{num_envs_with_badge_3 / total_envs:4f}",
            "badge_4": f"{num_envs_with_badge_4 / total_envs:4f}",
            "badge_5": f"{num_envs_with_badge_5 / total_envs:4f}",
            "badge_6": f"{num_envs_with_badge_6 / total_envs:4f}",
            "badge_7": f"{num_envs_with_badge_7 / total_envs:4f}",
            "badge_8": f"{num_envs_with_badge_8 / total_envs:4f}",
            "bill_saved": f"{bill_saved / total_envs:4f}",
            "beat_rocket_hideout_giovanni": f"{rocket_hideout_completions / total_envs:4f}",
            "rescued_mr_fuji": f"{pokemon_tower_completions / total_envs:4f}",
            "beat_silph_co_giovanni": f"{silph_co_completions / total_envs:4f}",
        }

        # Write the dictionary to a file in JSON format
        # with open(data.percent_complete_path, 'w') as file:
        #     json.dump(completion_dict, file, indent=4)

        # print("Completion data written to file:", data.percent_complete_path)
        
        
        
        
        
        
        
        # now for a tricky bit:
        # if we have swarm_frequency, we will take the top swarm_pct envs and evenly distribute
        # their states to the bottom 90%.
        # we do this here so the environment can remain "pure"

        # print(f'CPRL: LINE409 SWARMING TESTING ABOVE INITIAL IF')
        # logging.critical(f'CPRL: LINE409 SWARMING TESTING ABOVE INITIAL IF')
        # Assuming that data.infos['learner'] includes both 'state' and 'pkl' keys for each environment
        sf = skp = learner = statsbadges = statepyboy = statepkl = False
        if hasattr(data.config, "swarm_frequency"):
            sf = True
        if hasattr(data.config, "swarm_keep_pct"):
            skp = True
        if "learner" in data.infos:
            learner = True
        if "stats/badges" in data.infos["learner"]:
            statsbadges = True
            # logging.critical(f'badges={data.infos["learner"]["stats/badges"]}')
            # print(f'badges={data.infos["learner"]["stats/badges"]}')
        if  "state/pyboy" in data.infos["learner"]:
            statepyboy = True
        if "state/pkl" in data.infos["learner"]:
            statepkl = True

        flags = [
            ("Swarm Frequency", sf),
            ("Swarm Keep Percentage", skp),
            ("Learner", learner),
            ("Stats/Badges", statsbadges),
            ("State/PyBoy", statepyboy),
            ("State/Pkl", statepkl)
        ]

        for n, st in flags:
            print(f'cprl LINE 436: {n}={st}')
            logging.critical(f'cprl LINE 436: {n}={st}')
            
        # print(f'cprl LINE 441: keys in data.infos["learner"]={data.infos["learner"].keys()}')
        # logging.critical(f'cprl LINE 441: keys in data.infos["learner"]={data.infos["learner"].keys()}')
    
            
        if (
            hasattr(data.config, "swarm_frequency") and
            hasattr(data.config, "swarm_keep_pct") and
            "learner" in data.infos and
            "stats/badges" in data.infos["learner"]
        ):
            badge_threshold = (data.config.swarm_frequency % 3) + 1  # Badge level check
            badge_counts = data.infos["learner"]["stats/badges"]
            logging.info(f'cprl: badge_counts={badge_counts}')
            # Calculate the percentage of environments with current badge level or higher
            num_envs_with_current_badge = sum(1 for badge in badge_counts if badge >= badge_threshold)
            percentage_with_current_badge = num_envs_with_current_badge / data.config.num_envs
            print(f'Badge Level: {badge_threshold}, Percentage with Badge >= {badge_threshold}: {percentage_with_current_badge}, Required Percentage: {data.config.swarm_keep_pct}')
            # data.logger.info((f'Update: {data.update}, Badge Level: {badge_threshold}, Percentage with Badge >= {badge_threshold}: {percentage_with_current_badge}, Required Percentage: {data.config.swarm_keep_pct}'))        
            
            if  ("state/pyboy" in data.infos["learner"] and
                "state/pkl" in data.infos["learner"]
            ): 
                # # TESTING
                # badge_threshold = 0
                # badge_counts = data.infos["learner"]["stats/map"] 

                # If conditional is met, load up the given badge to all envs
                if percentage_with_current_badge > data.config.swarm_keep_pct:
                    # Collect the top swarm_keep_pct % of envs based on badges - these will all have the given badge
                    largest = [
                        x[0]
                        for x in heapq.nlargest(
                            math.ceil(data.config.num_envs * data.config.swarm_keep_pct),
                            enumerate(badge_counts),
                            key=lambda x: x[1],
                        )
                    ]
                    # Increment badge number counter and set current badge completion to 1
                    data.config.swarm_frequency += 1
                    # percentage_with_current_badge = 1
                    # Make a list of the pyboy state, pkl state, and % with badge for environment use

                    pyboy = data.infos["learner"]["state/pyboy"]
                    pkl = data.infos["learner"]["state/pkl"]
                    
                    import dill
                    try:
                        with io.BytesIO(pkl[0]) as memory_file:
                            memory_file.seek(0)
                            m1 = dill.load(memory_file)
                            logging.critical(f'1) cprl: SUCCESSFULLY deserialized test pkl before sending in queue: {m1}')
                    except dill.PickleError as e:
                        logging.error(f"1] cprl: Failed to deserialize test pkl before sending in queue: {e}")
                        pass
                    
                    # Collect and serialize the environment's state
                    combined_state = [pyboy, pkl, percentage_with_current_badge]
                    logging.info(f'cprl: percentage_with_current_badge={percentage_with_current_badge}')
                    # # Remove previous "state" dict if present
                    # if "state" in data.infos["learner"]:
                    #     del data.infos["learner"]["state"]
                    # Make the new "state" dict
                    learner_info = data.infos["learner"]
                    # Create or update the 'state' key safely
                    # try:
                    learner_info['state'] = combined_state 
                    # except Exception as e:
                    #     logging.critical(f'cprl: could not update the namespace data.infos["learner"].update("state": combined_state)')
                    # Remove the old saved states from the namespace to prevent oom
                    # del data.infos["learner"]["state/pkl"]
                    # del data.infos["learner"]["state/pyboy"]
                    try:
                        pkl2 = learner_info['state'][1][0]
                        with io.BytesIO(pkl2) as memory_file2:
                            memory_file2.seek(0)
                            m2 = dill.load(memory_file2)
                            logging.critical(f'2) cprl: SUCCESSFULLY deserialized test pkl before sending in queue AFTER making combined_state list: {m2}')
                    except dill.PickleError as e:
                        logging.error(f"2] cprl: Failed to deserialize test pkl before sending in queue AFTER making combined_state list:  {e}")
                        pass
                    
                    
                    if "state" in data.infos["learner"]:
                        print("Migrating states:")
                        logging.critical("Migrating states:")
                        waiting_for = []

                        # # TESTING
                        # largest = [0]
                        for i in range(data.config.num_envs):
                            if i not in largest:
                                new_state = random.choice(largest)
                                print(f'\t {i+1} -> {new_state+1}, badge scores: {data.infos["learner"]["stats/badges"][i]} -> {data.infos["learner"]["stats/badges"][new_state]}')
                                # print(f'\t {i+1} -> {new_state+1}, map_n={data.infos["learner"]["stats/map"][i]} -> {data.infos["learner"]["stats/map"][new_state]}')

                                # Queue handling for state and pkl
                                # try:
                                data.env_recv_queues[i + 1].put((data.infos["learner"]["state"][new_state]))
                                waiting_for.append(i + 1)
                                print(f'put states for {i} successfully')
                                logging.critical(f'put states for {i} successfully')
                                # except:
                                #     print(f'DID NOT put states for {i} successfully')
                                #     logging.critical(f'DID NOT put states for {i} successfully')

                        for i in waiting_for:
                            # try:
                            data.env_send_queues[i].get()
                            print(f"Received confirmation from env {i}")
                            logging.critical(f"Received confirmation from env {i}")
                            # except Exception as e:
                            #     print(f"Error processing in env {i}: {e}")
                            #     logging.critical(f"Error processing in env {i}: {e}")
                            #     data.env_send_queues[i].put(None)
                    else:
                        logging.critical(f'"state" not found in data.infos["learner"]!!! CANNOT PUT OR GET STATES!!!')
    
    
    data.policy_pool.update_policies()
    performance = defaultdict(list)
    env_profiler = pufferlib.utils.Profiler()
    inference_profiler = pufferlib.utils.Profiler()
    eval_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
    misc_profiler = pufferlib.utils.Profiler() # BET ADDED 2

    ptr = step = padded_steps_collected = agent_steps_collected = 0
    infos = defaultdict(lambda: defaultdict(list))
    while True:
        step += 1
        if ptr == config.batch_size + 1:
            break

        with env_profiler:
            o, r, d, t, i, env_id, mask = data.pool.recv()

        with misc_profiler:
            i = data.policy_pool.update_scores(i, "return")
            # TODO: Update this for policy pool
            for ii, ee  in zip(i['learner'], env_id):
                ii['env_id'] = ee
                
        with inference_profiler, torch.no_grad():
            o = torch.as_tensor(o).to(device=data.device, non_blocking=True)
            r = (torch.as_tensor(r, dtype=torch.float32).to(device=data.device, non_blocking=True).view(-1))
            d = (torch.as_tensor(d, dtype=torch.float32).to(device=data.device, non_blocking=True).view(-1))

            agent_steps_collected += sum(mask)
            padded_steps_collected += len(mask)

            # Multiple policies will not work with new envpool
            next_lstm_state = data.next_lstm_state
            if next_lstm_state is not None:
                next_lstm_state = (
                    next_lstm_state[0][:, env_id],
                    next_lstm_state[1][:, env_id],
                )

            actions, logprob, value, next_lstm_state = data.policy_pool.forwards(
                    o.to(data.device), next_lstm_state)

            if next_lstm_state is not None:
                h, c = next_lstm_state
                data.next_lstm_state[0][:, env_id] = h
                data.next_lstm_state[1][:, env_id] = c

            value = value.flatten()
        
        with misc_profiler:
            actions = actions.cpu().numpy()

            learner_mask = torch.Tensor(mask * data.policy_pool.mask) # BET ADDED 10

            # Ensure indices do not exceed batch size
            indices = torch.where(learner_mask)[0][:config.batch_size - ptr + 1].numpy()
            end = ptr + len(indices)

            # Batch indexing
            data.obs_ary[ptr:end] = o.cpu().numpy()[indices]
            data.values_ary[ptr:end] = value.cpu().numpy()[indices]
            data.actions_ary[ptr:end] = actions[indices]
            data.logprobs_ary[ptr:end] = logprob.cpu().numpy()[indices]
            data.rewards_ary[ptr:end] = r.cpu().numpy()[indices]
            data.dones_ary[ptr:end] = d.cpu().numpy()[indices]
            data.sort_keys.extend([(env_id[i], step) for i in indices])

            # Update pointer
            ptr += len(indices)
            

            # for policy_name, policy_i in i.items():
            #     for agent_i in policy_i:
            #         for name, dat in unroll_nested_dict(agent_i):
            #             infos[policy_name][name].append(dat)


            for policy_name, policy_i in i.items():
                for agent_i in policy_i:
                    for name, dat in unroll_nested_dict(agent_i):
                        if policy_name not in data.infos:
                            data.infos[policy_name] = {}
                        if name not in data.infos[policy_name]:
                            data.infos[policy_name][name] = [
                                np.zeros_like(dat)
                            ] * config.num_envs
                        data.infos[policy_name][name][agent_i["env_id"]] = dat
                        # infos[policy_name][name].append(dat)
        with env_profiler:
            data.pool.send(actions)

    # data.reward_buffer.append(r.cpu().sum().numpy())
    # Probably should normalize the rewards before trying to take the variance...
    # reward_var = np.var(data.reward_buffer)
    # if data.wandb is not None:
    #     data.wandb.log(
    #         {
                # "reward/reward_var": reward_var,
                # "reward/reward_buffer_len": len(data.reward_buffer),
    #         },
    #         step=data.global_step
            
    #     )
    # if (
    #     data.taught_cut
    #     and len(data.reward_buffer) == data.reward_buffer.maxlen
        # and reward_var < 2.5e-3
    # ):
        # data.reward_buffer.clear()
        # reset lr update if the reward starts stalling
        # data.lr_update = 1.0    

    eval_profiler.stop()

    data.global_step += padded_steps_collected
    # thatguy steps (cuz why not?)
    if "learner" in data.infos and "stats/step" in data.infos["learner"]:
        try:
            new_step = np.mean(data.infos["learner"]["stats/step"])
            if new_step > data.global_step:
                data.global_step = new_step
                data.log = True
        except KeyError:
            print(f'KeyError clean_pufferl data.infos["learner"]["stats/step"]')
            pass

    data.reward = float(torch.mean(data.rewards))
    data.SPS = int(padded_steps_collected / eval_profiler.elapsed)

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_agent_steps = data.global_step
    perf.env_time = env_profiler.elapsed
    perf.env_sps = int(agent_steps_collected / env_profiler.elapsed)
    perf.inference_time = inference_profiler.elapsed
    perf.inference_sps = int(padded_steps_collected / inference_profiler.elapsed)
    perf.eval_time = eval_profiler.elapsed
    perf.eval_sps = int(padded_steps_collected / eval_profiler.elapsed)
    perf.eval_memory = eval_profiler.end_mem
    perf.eval_pytorch_memory = eval_profiler.end_torch_mem
    perf.misc_time = misc_profiler.elapsed # BET ADDED 25

    
    data.stats = {}
    data.max_stats = {} # BET ADDED 26
    infos = infos['learner']
    
    # if data.wandb is not None:
    #     if 'pokemon_exploration_map' in infos and data.update % 5 == 0:
    #         # # Create a mapping from map ID to name
    #         # map_id_to_name = {int(region["id"]): region["name"] for region in data.total_envs["regions"]}
    #         # if data.update % 10 == 0:
    #         if 'pokemon_exploration_map' in infos:
    #             for idx, pmap in zip(infos['env_id'], infos['pokemon_exploration_map']):
    #                 if not hasattr(data, 'map_updater'):
    #                     data.map_updater = pokemon_red_eval.map_updater()
    #                     data.map_buffer = np.zeros((data.config.num_envs, *pmap.shape))
    #                 data.map_buffer[idx] = pmap
    #             pokemon_map = np.sum(data.map_buffer, axis=0)
    #             rendered = data.map_updater(pokemon_map)
    #             data.stats['Media/exploration_map'] = data.wandb.Image(rendered)
    #         # Process 'stats/map' and increment map_counts
    #         if 'stats/map' in infos:
    #             for item in infos['stats/map']:
    #                 if isinstance(item, int):
    #                     data.map_counts[item] += 1
    #         # # Increment env_reports for each environment ID
    #         # for env_id in infos['env_id']:
    #         #     data.env_reports[env_id] += 1
    #         # Calculate mean for numeric data in infos and store in data.stats
    #         for k, v in infos.items():
    #             try:
    #                 if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
    #                     data.stats[k] = np.mean(v)
    #             except Exception as e:
    #                 print(f"Error processing {k}: {e}")
                    
                    
        # # Prepare data for the bar chart
        # labels = [f"{map_id} - {map_id_to_name.get(map_id, 'Unknown')}" for map_id in data.map_counts.keys()]
        # values = list(data.map_counts.values())
        # # Create a table for logging to wandb
        # table = data.wandb.Table(data=list(zip(labels, values)), columns=["Map ID and Name", "Count"])
        # # Log the bar chart to wandb
        # data.wandb.log({
        #     "map_distribution_bar_chart": data.wandb.plot.bar(table, "Map ID and Name", "Count",
        #                                                     title="Map Distribution")
        # })
        # data.map_counts.clear()  # Reset map_counts for the next reporting period
        # data.env_reports.clear()  # Reset env_reports as well


    # data.stats = {}

    # # logging = f'logging'
    # # log_path = os.path.join(data.config.data_dir, data.exp_name, logging)
    # # if not os.path.exists(log_path):
    # #     os.makedirs(log_path)

    data.stats = {}
    data.max_stats = {}
    for k, v in data.infos["learner"].items():
        if "pokemon_exploration_map" in k:
            if data.update % 10 == 0: # config.overlay_interval == 0:
                overlay = make_pokemon_red_overlay(np.stack(v, axis=0))
                if data.wandb is not None:
                    data.stats["Media/aggregate_exploration_map"] = data.wandb.Image(overlay)
        elif "cut_exploration_map" in k: # and config.save_overlay is True:
            if data.update % config.overlay_interval == 0:
                overlay = make_pokemon_red_overlay(np.stack(v, axis=0))
                if data.wandb is not None:
                    data.stats["Media/aggregate_cut_exploration_map"] = data.wandb.Image(
                        overlay
                    )
        elif "state" in k:
            pass
        else:
            try:  # TODO: Better checks on log data types
                # data.stats[f"Histogram/{k}"] = data.wandb.Histogram(v, num_bins=16)
                data.stats[k] = np.mean(v)
                data.max_stats[k] = np.max(v)
            except:  # noqa
                continue

    if config.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    # for k, v in infos['learner'].items():
    #     if 'Task_eval_fn' in k:
    #         # Temporary hack for NMMO competition
    #         continue
    #     if 'pokemon_exploration_map' in k:
    #         import cv2
    #         from pokemon_red_eval import make_pokemon_red_overlay
    #         bg = cv2.imread('kanto_map_dsv.png')
    #         overlay = make_pokemon_red_overlay(bg, sum(v))
    #         if data.wandb is not None:
    #             data.stats['Media/exploration_map'] = data.wandb.Image(overlay)
    #         # @Leanke: Add your infos['learner']['x'] etc
    #     # if 'logging' in k:
    #     #     pre.logger(v, log_path)
    #     try: # TODO: Better checks on log data types
    #         data.stats[k] = np.mean(v)
    #     except:
    #         continue

    # if config.verbose:
    #     print_dashboard(data.stats, data.init_performance, data.performance)

    # return data.stats, infos

    return data.stats, data.infos


@pufferlib.utils.profile
def train(data):
    if done_training(data):
        raise RuntimeError(
            f"Max training updates {data.total_updates} already reached")

    config = data.config
    # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
    train_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True)
    train_profiler.start()

    if config.anneal_lr:
        frac = 1.0 - (data.lr_update - 1.0) / data.total_updates
        lrnow = frac * config.learning_rate
        data.optimizer.param_groups[0]["lr"] = lrnow

    num_minibatches = config.batch_size // config.bptt_horizon // config.batch_rows
    idxs = sorted(range(len(data.sort_keys)), key=data.sort_keys.__getitem__)
    data.sort_keys = []

    b_idxs = (
        torch.Tensor(idxs).long()[:-1]
        .reshape(config.batch_rows, num_minibatches, config.bptt_horizon)
        .transpose(0, 1)
    )

    # bootstrap value if not done
    with torch.no_grad():
        advantages = torch.zeros(config.batch_size, device=data.device)
        lastgaelam = 0
        for t in reversed(range(config.batch_size)):
            i, i_nxt = idxs[t], idxs[t + 1]
            nextnonterminal = 1.0 - data.dones[i_nxt]
            nextvalues = data.values[i_nxt]
            delta = (
                data.rewards[i_nxt]
                + config.gamma * nextvalues * nextnonterminal
                - data.values[i]
            )
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )

    data.b_obs = b_obs = data.obs[b_idxs].to(data.device, non_blocking=True) # torch.Tensor(data.obs_ary[b_idxs])
    b_actions = torch.Tensor(data.actions_ary[b_idxs]).to(data.device, non_blocking=True)
    b_logprobs = torch.Tensor(data.logprobs_ary[b_idxs]).to(data.device, non_blocking=True)
    b_dones = torch.Tensor(data.dones_ary[b_idxs]).to(data.device, non_blocking=True)
    b_values = torch.Tensor(data.values_ary[b_idxs]).to(data.device, non_blocking=True)

    b_advantages = advantages.reshape(
        config.batch_rows, num_minibatches, config.bptt_horizon
    ).transpose(0, 1)
    b_returns = b_advantages + b_values

    # Optimizing the policy and value network
    train_time = time.time()
    pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []

    # COMMENTED OUT BET
    # mb_obs_buffer = torch.zeros_like(b_obs[0], pin_memory=(data.device == "cuda"))
    
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(num_minibatches):
            mb_obs = b_obs[mb]
            # COMMENTED OUT BET
            # mb_obs_buffer.copy_(b_obs[mb], non_blocking=True)
            # mb_obs = mb_obs_buffer.to(data.device, non_blocking=True)
            
            mb_actions = b_actions[mb].contiguous()
            mb_values = b_values[mb].reshape(-1)
            mb_advantages = b_advantages[mb].reshape(-1)
            mb_returns = b_returns[mb].reshape(-1)

            if hasattr(data.agent, 'lstm'):
                _, newlogprob, entropy, newvalue, lstm_state = data.agent.get_action_and_value(
                    mb_obs, state=lstm_state, action=mb_actions)
                lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
            else:
                _, newlogprob, entropy, newvalue = data.agent.get_action_and_value(
                    mb_obs.reshape(-1, *data.pool.single_observation_space.shape),
                    action=mb_actions,
                )

            logratio = newlogprob - b_logprobs[mb].reshape(-1)
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                old_kls.append(old_approx_kl.item())
                approx_kl = ((ratio - 1) - logratio).mean()
                kls.append(approx_kl.item())
                clipfracs += [
                    ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                ]

            mb_advantages = mb_advantages.reshape(-1)
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_losses.append(pg_loss.item())

            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -config.vf_clip_coef,
                    config.vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
            v_losses.append(v_loss.item())

            entropy_loss = entropy.mean()
            entropy_losses.append(entropy_loss.item())

            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
            data.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(data.agent.parameters(), config.max_grad_norm)
            data.optimizer.step()

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    train_profiler.stop()
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    losses = data.losses
    losses.policy_loss = np.mean(pg_losses)
    losses.value_loss = np.mean(v_losses)
    losses.entropy = np.mean(entropy_losses)
    losses.old_approx_kl = np.mean(old_kls)
    losses.approx_kl = np.mean(kls)
    losses.clipfrac = np.mean(clipfracs)
    losses.explained_variance = explained_var

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_updates = data.update + 1
    perf.train_time = time.time() - train_time
    perf.train_sps = int(config.batch_size / perf.train_time)
    perf.train_memory = train_profiler.end_mem
    perf.train_pytorch_memory = train_profiler.end_torch_mem
    perf.epoch_time = perf.eval_time + perf.train_time
    perf.epoch_sps = int(config.batch_size / perf.epoch_time)

    if config.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    data.update += 1
    data.lr_update += 1
    
    if data.update % config.checkpoint_interval == 0 or done_training(data):
       save_checkpoint(data)

def close(data):
    data.pool.close()

    ## BET ADDED 35
    if data.wandb is not None:
        artifact_name = f"{data.exp_name}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()

def rollout(env_creator, env_kwargs, agent_creator, agent_kwargs,
        model_path=None, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True
 
    while True:
        if terminal or truncated:
            if verbose:
                print('---  Reset  ---')

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        chars = env.render()
        print("\033c", end="")
        print(chars)

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        time.sleep(0.5)
        step += 1

def done_training(data):
    return data.update >= data.total_updates

def save_checkpoint(data):
    path = os.path.join(data.config.data_dir, data.exp_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # model_name = f'model_{data.update:06d}.pt'
    model_name = f"model_{data.update:06d}_state.pth"
    model_path = os.path.join(path, model_name)

    # Already saved
    if os.path.exists(model_path):
        return model_path

    # To handleboth uncompiled and compiled data.agent, when getting state_dict()
    torch.save(getattr(data.agent, "_orig_mod", data.agent).state_dict(), model_path)
    # torch.save(data.uncompiled_agent, model_path)

    state = {
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.update,
        "model_name": model_name,
    }

    if data.wandb:
        state['exp_name'] = data.exp_name

    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)

    # Also save a copy
    torch.save(state, os.path.join(path, f"trainer_state_{data.update:06d}.pt"))

    return model_path

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

def print_dashboard(stats, init_performance, performance):
    output = []
    data = {**stats, **init_performance, **performance}
    
    grouped_data = defaultdict(dict)
    
    for k, v in data.items():
        if k == 'total_uptime':
            v = timedelta(seconds=v)
        if 'memory' in k:
            v = pufferlib.utils.format_bytes(v)
        elif 'time' in k:
            try:
                v = f"{v:.2f} s"
            except:
                pass
        
        first_word, *rest_words = k.split('_')
        rest_words = ' '.join(rest_words).title()
        
        grouped_data[first_word][rest_words] = v
    
    for main_key, sub_dict in grouped_data.items():
        output.append(f"{main_key.title()}")
        for sub_key, sub_value in sub_dict.items():
            output.append(f"    {sub_key}: {sub_value}")
    
    print("\033c", end="")
    print('\n'.join(output))
    time.sleep(1/20)
