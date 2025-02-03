import os
import h5py
import torch
import pickle
import json
import random
import logging
import numpy as np
from gym.spaces import Box, Discrete, Tuple
import ipdb

# from envs import get_dim
try:
    from replay_buffer import ReplayBuffer
except:
    from SimpleSAC.replay_buffer import ReplayBuffer
# import ipdb


class MixedReplayBuffer(ReplayBuffer):
    def __init__(self, reward_scale, reward_bias, clip_action, state_dim, action_dim, max_episode_steps, task="halfcheetah", data_source="medium_replay",  device="cuda", scale_rewards=True, scale_state=False, buffer_ratio=10., residual_ratio=1., store_init_observations=False):
        super().__init__(state_dim, action_dim, device=device)

        self.scale_rewards = scale_rewards
        self.scale_state = scale_state
        self.buffer_ratio = buffer_ratio
        self.residual_ratio = residual_ratio
        self.max_episode_steps = max_episode_steps
        
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.clip_action = clip_action

        # load offline dataset into the replay buffer
        if task == "dkittywalkrandom":
            folder_paths = ["../d4rl_mujoco_dataset/DKitty_replay_buffer_169398/target_1m", "../d4rl_mujoco_dataset/DKitty_replay_buffer_169398/target_2m"]
            all_json_data = []

            # Load data from DKitty Json files in both folders
            for folder_path in folder_paths:
                for filename in os.listdir(folder_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(folder_path, filename)
                        with open(file_path, 'r') as json_file:
                            print(file_path)
                            json_data = json.load(json_file)

                        all_json_data.extend(json_data)

            # flat_all_json_data = [item for sublist in all_json_data for item in sublist]

            # Calculate the total number of data points
            total_num = len(all_json_data)

            # Generate random indices and sort them
            idx = np.sort(np.random.choice(range(total_num), int(total_num * residual_ratio), replace=False))

            # Load data based on the sorted indices
            # ipdb.set_trace()
            s = np.array([all_json_data[i]["state"] for i in idx]).astype(np.float32)
            a = np.array([all_json_data[i]["action"] for i in idx]).astype(np.float32)
            r = np.array([all_json_data[i]["reward"] for i in idx]).astype(np.float32).reshape(-1, 1)
            s_ = np.array([all_json_data[i]["next_state"] for i in idx]).astype(np.float32)
            done = np.array([all_json_data[i]["done"] for i in idx]).astype(np.bool).reshape(-1, 1)

        elif task == "humanoid":
            with open("../d4rl_mujoco_dataset/dataset_1934657_r6077.33_s0.00.pickle", "rb") as f:
                # dataset = pickle.load(f)
                try:
                    # load complete file
                    dataset = pickle.load(f)
                except pickle.UnpicklingError:
                    # if loading fails, go back to beginning
                    f.seek(0)
                    # load data before truncated
                    dataset = pickle.load(f)
                total_num = len(dataset)
                # idx = list(range(int(total_num*0.5)))
                idx = np.sort(np.random.choice(range(total_num), int(total_num * self.residual_ratio), replace=False))

                s = np.array([dataset[i][0] for i in idx]).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of observations
                a = np.array([dataset[i][1] for i in idx]).astype(np.float32) # An (N, dim_action)-dimensional numpy array of actions
                r = np.expand_dims(np.array([dataset[i][2] for i in idx]).astype(np.float32), axis=1) # An (N,)-dimensional numpy array of rewards
                s_ = np.array([dataset[i][3] for i in idx]).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of next observations
                done = np.expand_dims(np.array([dataset[i][4] for i in idx]), axis=1) # An (N,)-dimensional numpy array of terminal flags
        else:
            path = os.path.join("../d4rl_mujoco_dataset", "{}_{}-v2.hdf5".format(task, data_source))
            with h5py.File(path, "r") as dataset:
                total_num = dataset['observations'].shape[0]
                use_timeouts = ('timeouts' in dataset)
                # idx = random.sample(range(total_num), int(total_num * self.residual_ratio))
                idx = np.sort(np.random.choice(range(total_num), int(total_num * self.residual_ratio), replace=False))
                # ipdb.set_trace()
                s = np.vstack(np.array(dataset['observations'])).astype(np.float32)[idx, :] # An (N, dim_observation)-dimensional numpy array of observations
                a = np.vstack(np.array(dataset['actions'])).astype(np.float32)[idx, :] # An (N, dim_action)-dimensional numpy array of actions
                r = np.vstack(np.array(dataset['rewards'])).astype(np.float32)[idx, :] # An (N,)-dimensional numpy array of rewards
                s_ = np.vstack(np.array(dataset['next_observations'])).astype(np.float32)[idx, :] # An (N, dim_observation)-dimensional numpy array of next observations
                done = np.vstack(np.array(dataset['terminals']))[idx, :] # An (N,)-dimensional numpy array of terminal flags
                if use_timeouts:
                    is_final_timestep = np.array(dataset['timeouts'])[idx]
                else:
                    is_final_timestep = np.zeros(done.shape)
                    is_final_timestep[np.arange(1, total_num + 1, self.max_episode_steps)] = True
                    is_final_timestep = is_final_timestep[idx, :]

        # whether to bias the reward
        r = r * self.reward_scale + self.reward_bias
        # whether to clip actions
        a = np.clip(a, -self.clip_action, self.clip_action)
        
        fixed_dataset_size = r.shape[0]
        self.fixed_dataset_size = fixed_dataset_size
        self.ptr = fixed_dataset_size
        self.size = fixed_dataset_size
        self.max_size = int((self.buffer_ratio + 1) * fixed_dataset_size)

        self.state = np.vstack((s, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.action = np.vstack((a, np.zeros((self.max_size - self.fixed_dataset_size, action_dim))))
        self.next_state = np.vstack((s_, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.reward = np.vstack((r, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.done = np.vstack((done, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.device = torch.device(device)
        
        if store_init_observations:
            final_indices, = np.where(is_final_timestep)
            if final_indices.size:
                if final_indices[-1] >= self.size - 1:
                    final_indices = final_indices[:-1]
                # ipdb.set_trace()
                self.init_observation_buffer = self.state[final_indices + 1]
                self.init_action_buffer = self.action[final_indices + 1]
            else:
                self.init_observation_buffer = np.array([[]])
                self.init_action_buffer = np.array([[]])
        # if store_init_observations:
        #     done_indices, _ = np.where(self.done != 0)
        #     if done_indices.size:
        #         if done_indices[-1] >= self.size - 1:
        #             done_indices = done_indices[:-1]
        #         # ipdb.set_trace()
        #         self.init_observation_buffer = self.state[done_indices + 1]
        #     else:
        #         self.init_observation_buffer = np.array([[]])
        
        # ipdb.set_trace()
        # State normalization
        self.normalize_states()

    def normalize_states(self, eps=1e-3):
        # STATE: standard normalization
        self.state_mean = self.state.mean(0, keepdims=True)
        self.state_std = self.state.std(0, keepdims=True) + eps
        if self.scale_state:
            self.state = (self.state - self.state_mean) / self.state_std
            self.next_state = (self.next_state - self.state_mean) / self.state_std
            
    # def append_init_obs(self, init_obs):
    #     self.init_observation_buffer = np.vstack((self.init_observation_buffer, init_obs)) if self.init_observation_buffer.size else init_obs
        
    def sample_init_obs(self, batch_size):
        init_obs_buffer_size = self.init_observation_buffer.shape[0]
        if batch_size < init_obs_buffer_size:
            ind = np.random.randint(0, init_obs_buffer_size, size=batch_size)
            return torch.FloatTensor(self.init_observation_buffer[ind]).to(self.device), torch.FloatTensor(self.init_action_buffer[ind]).to(self.device)
        else:
            return torch.FloatTensor(self.init_observation_buffer[:]).to(self.device), torch.FloatTensor(self.init_action_buffer[:]).to(self.device)

    def append(self, s, a, r, s_, done):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_
        self.reward[self.ptr] = r * self.reward_scale
        self.done[self.ptr] = done

        # fix the offline dataset and shuffle the simulated part
        self.ptr = (self.ptr + 1 - self.fixed_dataset_size) % (self.max_size - self.fixed_dataset_size) + self.fixed_dataset_size
        self.size = min(self.size + 1, self.max_size)
    
    def clear_source_buffer(self):
        self.ptr = self.fixed_dataset_size

    def append_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.append(o, a, r, no, d)

    def sample(self, batch_size, scope=None, type=None):
        if scope == None:
            # ind = np.random.choice(self.size, size=batch_size, replace=False)
            ind = np.random.randint(0, self.size, size=batch_size)
        elif scope == "real":
            ind = np.random.randint(0, self.fixed_dataset_size, size=batch_size)
        elif scope == "sim":
            # ind = np.random.choice(range(self.fixed_dataset_size, self.size), size=batch_size, replace=False)
            ind = np.random.randint(self.fixed_dataset_size, self.size, size=batch_size)
        else: 
            raise RuntimeError("Misspecified range for replay buffer sampling")

        if type == None:
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
                'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
                'dones': torch.FloatTensor(self.done[ind]).to(self.device)
                }
        elif type == "sas":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device)
                }
        elif type == "sa":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device)
                }
        else: 
            raise RuntimeError("Misspecified return data types for replay buffer sampling")

    def get_mean_std(self):
        return torch.FloatTensor(self.state_mean).to(self.device), torch.FloatTensor(self.state_std).to(self.device)


class NewReplayBuffer(MixedReplayBuffer, ReplayBuffer):
    def __init__(self, file_path, env_name, reward_scale, reward_bias, clip_action, state_dim, action_dim,
                 device="cuda", scale_rewards=True, scale_state=False,
                 buffer_ratio=10., residual_ratio=1., store_init_observations=False):
        ReplayBuffer.__init__(self, state_dim=state_dim, action_dim=action_dim, device=device)
        self._file_path = file_path
        self._env_name = env_name
        self.scale_rewards = scale_rewards
        self.scale_state = scale_state
        self.buffer_ratio = buffer_ratio
        self.residual_ratio = residual_ratio
        # self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.clip_action = clip_action

        # Load isaac gym data from offline file.
        s, a, s_, a_, r, done = self._load_data()
        # whether to bias the reward
        r = r * self.reward_scale + self.reward_bias
        # whether to clip actions
        a = np.clip(a, -self.clip_action, self.clip_action)

        fixed_dataset_size = r.shape[0]
        self.fixed_dataset_size = fixed_dataset_size
        self.ptr = fixed_dataset_size
        self.size = fixed_dataset_size
        self.max_size = int((self.buffer_ratio + 1) * fixed_dataset_size)

        self.state = np.vstack((s, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.action = np.vstack((a, np.zeros((self.max_size - self.fixed_dataset_size, action_dim))))
        self.next_state = np.vstack((s_, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.reward = np.vstack((r, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.done = np.vstack((done, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.device = torch.device(device)

        # if store_init_observations:
        #     final_indices, = np.where(is_final_timestep)
        #     if final_indices.size:
        #         if final_indices[-1] >= self.size - 1:
        #             final_indices = final_indices[:-1]
        #         # ipdb.set_trace()
        #         self.init_observation_buffer = self.state[final_indices + 1]
        #         self.init_action_buffer = self.action[final_indices + 1]
        #     else:
        #         self.init_observation_buffer = np.array([[]])
        #         self.init_action_buffer = np.array([[]])
        self.normalize_states()

    def _load_data(self):
        dataset_file = np.loadtxt(self._file_path + '.txt')
        dataset_size = dataset_file.shape[0]
        logging.info('=' * 20 + f'The original data size is: {dataset_size}.' + '=' * 20)

        # dataset_file[:, 4] = dataset_file[:, 4] / 2

        # Delete data that have action > 4
        self._max_push_effort = 3
        safe_index = np.where(dataset_file[:, 4] <= self._max_push_effort)[0]
        dataset_file = dataset_file[safe_index]
        dataset_size = dataset_file.shape[0]
        logging.info('=' * 20 + f'The safety data size is: {dataset_size}.' + '=' * 20)

        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(dataset_file[:, -1]),
                np.arange(dataset_size) < dataset_size - 1))
        logging.info('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))

        demo_s1 = dataset_file[:, :4][nonterminal_steps]
        demo_s2 = dataset_file[:, :4][nonterminal_steps + 1]
        demo_a1 = dataset_file[:, 4][nonterminal_steps]
        demo_a2 = dataset_file[:, 4][nonterminal_steps + 1]

        # Reward for Wheellegged standing.
        if self._env_name == 'WheelLegged':
            demo_r = 30.0 - demo_a1 * demo_a1 - (demo_s2 * demo_s2).sum(axis=1)
            # demo_r = 30.0 - demo_a1 * demo_a1 - (demo_s2[:, 1:] * demo_s2[:, 1:]).sum(axis=1) \
            #          - 50 * demo_s2[:, 0] * demo_s2[:, 0]
        # Reward for Wheellegged straight.
        elif self._env_name == 'WheelLegged-straight':
            demo_r = 15.0 - ((demo_s1[:, 1] - 0.2) ** 2 + demo_a1 ** 2)
            # demo_r = 15.0 - (100 * (demo_s1[:, 1] - 0.2) ** 2 + demo_a1 ** 2)
        else:
            raise ValueError(f'The env_name: {self._env_name} is wrong!')

        demo_d = dataset_file[:, 5][nonterminal_steps + 1]
        demo_a1 = np.reshape(demo_a1, (-1, 1)) / self._max_push_effort
        demo_a2 = np.reshape(demo_a2, (-1, 1)) / self._max_push_effort

        return demo_s1, demo_a1, demo_s2, demo_a2, demo_r.reshape((-1, 1)), demo_d.reshape((-1, 1))



