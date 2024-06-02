# currently not in use

def save_all_states_v3(self, is_failed=False):
    # Assume self.save_state_dir is properly initialized in Base class init
    # self.save_state_dir = Path(__file__).parent / "save_states/"
    # self.save_state_dir.mkdir(exist_ok=True)       
    # Constructing state directory path
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    self.state_dir = self.save_state_dir / Path(f'event_reward_{self.event_reward}') / f'env_id_{self.env_id}'        
    self.state_dir.mkdir(parents=True, exist_ok=True)
    self.state_file_path = self.state_dir / f"env_id_{self.env_id}_{datetime_str}_state.pkl"
    self.pyboy_state_file_path = self.state_dir / f"env_id_{self.env_id}_{datetime_str}_state.state" 
    # Explicitly exclude non-serializable variables
    excluded_vars = ['game', 'screen', 'api']
    state = {key: value for key, value in vars(self).items() if key not in excluded_vars}
    try:
        with open(self.state_file_path, 'wb') as f:
            dill.dump(state, f)
        print(f"Saved state to {self.state_file_path}")
    except Exception as e:
        print(f"Failed to save state. Error: {e}")  
    try:
        with open(self.pyboy_state_file_path, 'wb') as f:
            self.game.save_state(f)
        print(f'saved pyboy state file {self.pyboy_state_file_path}')
    except:
        print(f'{self.pyboy_state_file_path} failed to save pyboy state..')

def select_best_states(self):
    # Function to scan all folders containing .pkl .state files and order them
    # from highest to lowest using various metrics:
    # badge_1 completion "badge_1": float(self.badges >= 1)
    # mt moon completion 
    # badge_2 completion "badge_2": float(self.badges >= 2)
    # bill completion int(ram_map.read_bit(self.game, 0xD7F2, 7))
    # rubbed_captains_back int(ram_map.read_bit(self.game, 0xD803, 1))
    # taught cut self.cut
    # used cut on a good tree self.used_cut>0
    pass
    
# Version of loading fn that sorts env saved state folders by event_reward (in this case)
def load_state_from_directory_v2(self):
    # Assuming self.exp_path is correctly set to the base path
    # For example: '/puffertank/0.7/pufferlib/experiments/ydbo1df4'
    # Ensure the base experiments path exists
    if not self.exp_path.exists():
        print(f"Experiments path does not exist: {self.exp_path}")
        return
    # Recursively find all .pkl files within the experiments path
    pkl_files = list(self.exp_path.rglob('*_state.pkl'))
    if not pkl_files:
        print("No .pkl state files found across any sessions.")
        return
    # Extract the event_reward value and associate it with each found .pkl file
    pkl_files_with_reward = [(file, float(re.search(r'event_reward_([0-9\.]+)', file.parent.as_posix()).group(1))) for file in pkl_files]
    # Find the .pkl file with the highest event_reward value
    highest_reward_file = max(pkl_files_with_reward, key=lambda x: x[1])[0]
    # The corresponding .state file should have the same name except for the extension
    selected_pyboy_file = highest_reward_file.with_suffix('.state')
    print(f"Selected .pkl for loading: {highest_reward_file}")
    print(f"Selected .state for loading: {selected_pyboy_file}")
    # Load state from the .pkl file
    try:
        with open(highest_reward_file, 'rb') as f:
            state = pickle.load(f)
            for key, value in state.items():
                setattr(self, key, value)
        print("Environment state loaded successfully.")
    except Exception as e:
        print(f"Failed to load environment state. Error: {e}")
    # Load PyBoy state if the .state file exists
    if selected_pyboy_file.exists():
        try:
            with open(selected_pyboy_file, 'rb') as f:
                self.game.load_state(f)  # Ensure this is the correct method to load your PyBoy state
            print("PyBoy state loaded successfully.")
        except Exception as e:
            print(f"Failed to load PyBoy state. Error: {e}")
    else:
        print("Matching .state file not found.")
    # Reset or initialize as necessary post-loading
    self.reset_count = 0
    self.step_count = 0
    self.reset_count += 1
    
def load_state_from_tuple(self):
    # Retrieve the tuple of file paths for the current environment
    with Base.lock:
        files_to_load = self.shared_data[self.env_id].get("files_to_load", None)

    if not files_to_load:
        print("No files specified for loading.")
        return

    if files_to_load != None:
        # Unpack the tuple into its components
        pickled_file_path, pyboy_state_file_path = files_to_load

        print(f"Selected .pkl for loading: {pickled_file_path}")
        print(f"Selected .state for loading: {pyboy_state_file_path}")
        
        # Load state from the .pkl file
        try:
            with open(pickled_file_path, 'rb') as f:
                state = pickle.load(f)
                for key, value in state.items():
                    setattr(self, key, value)
            print(f"Environment state at {pickled_file_path} loaded successfully.")
        except Exception as e:
            print(f"Failed to load environment state. Error: {e}")
        
        # Load PyBoy state if the .state file exists
        try:
            with open(pyboy_state_file_path, 'rb') as f:
                self.game.load_state(f)  # Ensure this is the correct method to load your PyBoy state
            print(f"PyBoy state at {pyboy_state_file_path} loaded successfully.")
        except Exception as e:
            print(f"Failed to load PyBoy state. Error: {e}")

        # Reset or initialize as necessary post-loading
        self.reset_count = 0
        self.step_count = 0
        self.reset_count += 1

def update_milestone(self, key, condition):
    # Only update the milestone if the condition is met and it's either not recorded yet or we're recording a better time
    if condition:
        # Check if we've already recorded this milestone
        if key not in self.shared_data[self.env_id] or self.current_time < self.shared_data[self.env_id][key][1]:
            # Record the milestone as achieved with the current time
            with Base.lock:  # Assuming you're using a multiprocessing lock for thread safety
                self.shared_data[self.env_id][key] = (1.0, self.current_time)

def assess_milestone_completion_percentage(self):
    self.current_time = datetime.now() 
    # Fetch current milestones data from shared_data or initialize if not present
    current_milestones = self.shared_data.get(self.env_id, {})
    
    # Define a local function to update milestones only if needed

                    
    # Update each milestone if its condition is met
    self.update_milestone("badge_1", self.badges >= 1)
    self.update_milestone("mt_moon_completion", 'mt_moon_key' in self.completed_milestones)
    self.update_milestone("badge_2", self.badges >= 2)
    self.update_milestone("bill_completion", ram_map.read_bit(self.game, 0xD7F2, 7))
    self.update_milestone("rubbed_captains_back", ram_map.read_bit(self.game, 0xD803, 1))
    self.update_milestone("taught_cut", self.cut)
    self.update_milestone("used_cut_on_good_tree", self.used_cut > 0)
    self.shared_data[self.env_id]["files_to_load"] = None
    
    # Lock and update shared_data for this environment
    with Base.lock:
        self.shared_data[self.env_id] = current_milestones
    
# One env computes each milestone completion % over all envs
def compute_overall_progress(self, shared_data):
    milestone_keys = ["badge_1", "mt_moon_completion", "badge_2", 
                    "bill_completion", "rubbed_captains_back", 
                    "taught_cut", "used_cut_on_good_tree"]
    overall_progress = {key: 0 for key in milestone_keys}  # Initialize dict  
    for key in milestone_keys:
        # Calculate average completion percentage for each milestone
        overall_progress[key] = sum(data.get(key, (0, 0))[0] for data in shared_data.values()) / len(shared_data)
    return overall_progress

def assess_progress_for_action(self, overall_progress, milestone_threshold_dict):
    actions_required = {}  # Dict to hold milestones that meet the threshold        
    for key, threshold in milestone_threshold_dict.items():
        if overall_progress.get(key, 0) >= threshold:
            # Calculate percentage of envs to act upon based on completion
            actions_required[key] = 1 - overall_progress[key]        
    return actions_required

# Call like assess_states_for_loading(shared_data, pkl_files, 
# (assess_progress_for_action(compute_overall_progress(shared_data)), milestone_threshold_dict))
def sort_and_assess_envs_for_loading(self, shared_data, actions_required, num_envs):
    worst_envs, best_env_files = [], []
    
    # Assuming the shared_data structure now includes a 'files_to_load' entry for best environments
    # that contains tuples of (pkl_file_path, state_file_path)
    for milestone, action_percentage in actions_required.items():
        # Filter and sort environments based on completion time
        envs_completion_times = [(env_id, data[milestone][1], data.get("files_to_load", None)) 
                                for env_id, data in shared_data.items() 
                                if milestone in data]
        sorted_envs = sorted(envs_completion_times, key=lambda x: x[1])
        
        # Calculate the split index based on action percentage
        split_index = int(len(sorted_envs) * (1 - action_percentage))
        
        # Best environments - we're interested in their state files for loading
        for _, _, files_to_load in sorted_envs[:split_index]:
            if files_to_load is not None:
                best_env_files.append(files_to_load)
        
        # Worst environments - these need new states loaded into them
        worst_envs.extend([env_id for env_id, _, _ in sorted_envs[split_index:]])
    
    # Deduplicate while preserving order for worst environments
    worst_envs = list(dict.fromkeys(worst_envs))
    
    # Randomly assign state files from the best environments to the worst environments
    for env_id in worst_envs:
        if best_env_files:  # Check if there are any best environment files available
            selected_files = random.choice(best_env_files)  # Randomly select a tuple of files
            with Base.lock:
                self.shared_data[env_id]["files_to_load"] = selected_files
        else:
            print("No best environment files available for loading.")