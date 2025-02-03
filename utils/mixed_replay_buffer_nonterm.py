import os
import h5py
import torch
import copy
import numpy as np
from gym.spaces import Box, Discrete, Tuple

from utils.envs import get_dim
from utils.replay_buffer import ReplayBuffer


class MixedReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, task="halfcheetah", data_source="medium_replay",  device="cuda", scale_rewards=True, scale_state=True, ratio=2):
        super().__init__(state_dim, action_dim, device=device)

        self.scale_rewards = scale_rewards
        self.scale_state = scale_state
        self.ratio = ratio

        # load expert dataset into the replay buffer
        path = os.path.join("../d4rl_mujoco_dataset", "{}_{}-v2.hdf5".format(task, data_source))
        with h5py.File(path, "r") as dataset:
            s = np.vstack(np.array(dataset['observations'])).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of observations
            a = np.vstack(np.array(dataset['actions'])).astype(np.float32) # An (N, dim_action)-dimensional numpy array of actions
            r = np.vstack(np.array(dataset['rewards'])).astype(np.float32) # An (N,)-dimensional numpy array of rewards
            s_ = np.vstack(np.array(dataset['next_observations'])).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of next observations
            done = np.vstack(np.array(dataset['terminals'])) # An (N,)-dimensional numpy array of terminal flags
        print(r.shape)
        fixed_dataset_size = r.shape[0]
        self.fixed_dataset_size = fixed_dataset_size
        
        self.size = fixed_dataset_size
        self.max_size = self.ratio * fixed_dataset_size
        self.state = np.vstack((s, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.action = np.vstack((a, np.zeros((self.max_size - self.fixed_dataset_size, action_dim))))
        self.next_state = np.vstack((s_, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.reward = np.vstack((r, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.not_done = np.vstack((1. - done, np.ones((self.max_size - self.fixed_dataset_size, 1))))
        print(self.not_done)
        self.device = torch.device(device)
        self.state_mean, self.state_std, self.ptr = self.convert_D4RL(self.not_done)
        print(self.ptr)

        # reset fixed dataset size: leave out the terminal steps
        self.fixed_dataset_size = self.ptr

    def convert_D4RL(self, not_done):
        # print("Done Shape: ", done.shape)
        # print("Done Type: ", type(done))
        # print("dataset size: ", self.fixed_dataset_size)
        aaa = np.squeeze(not_done.T)
        # print(aaa)
        bbb = np.arange(self.max_size) < self.max_size - 1
        # print(bbb)
        nonterminal_steps, = np.where(np.logical_and(aaa, bbb))
        print(nonterminal_steps)
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), self.max_size)) 

        self.state = self.state[nonterminal_steps]

        self.next_action = copy.deepcopy(self.action)
        
        self.action = self.action[nonterminal_steps]
        self.next_state = self.next_state[nonterminal_steps]
        self.next_action = self.next_action[nonterminal_steps + 1]
        self.reward = self.reward[nonterminal_steps].reshape(-1, 1)
        self.not_done = self.not_done[nonterminal_steps + 1].reshape(-1, 1)

        # # REWARD: min_max normalization
        # if self.scale_rewards:
        #     r_max = np.max(self.reward)
        #     r_min = np.min(self.reward)
        #     self.reward = (self.reward - r_min) / (r_max - r_min)

        # STATE: standard normalization
        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)
        if self.scale_state:
            self.state = (self.state - s_mean) / (s_std + 1e-3)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-3)

        return s_mean, s_std, len(nonterminal_steps) - self.fixed_dataset_size

    # def normalize_states(self, eps=1e-3):
    #     mean = self.state.mean(0, keepdims=True)
    #     std = self.state.std(0, keepdims=True) + eps
    #     self.state = (self.state - mean) / std
    #     self.next_state = (self.next_state - mean) / std
    #     return mean, std

    def append(self, s, a, r, s_, done):
        # # the current number of samples in buffer
        # current_num = self.buffer.shape[0]
        # if current_num < self.fixed_dataset_size:
        #     raise IndexError("The number of samples in the buffer is less than that in the fixed dataset")
        # if current_num > self.max_size:
        #     index = np.random.choice(current_num - self.fixed_dataset_size, self.size - self.fixed_dataset_size, replace=False).tolist()
        #     fixed_bias = np.repeat(self.fixed_dataset_size, self.size - self.fixed_dataset_size)
        #     self.buffer = self.buffer[index + fixed_bias]
        # else:
        #     pass
        
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_
        self.reward[self.ptr] = r
        self.not_done[self.ptr] = 1. - done

        # fix the offline dataset and shuffle the simulated part
        self.ptr = (self.ptr + 1 - self.fixed_dataset_size) % (self.max_size - self.fixed_dataset_size) + self.fixed_dataset_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, range=None, type=None):
        if range == None:
            ind = np.random.randint(0, self.size, size=batch_size)
        elif range == "real":
            ind = np.random.randint(0, self.fixed_dataset_size, size=batch_size)
        elif range == "sim":
            ind = np.random.randint(self.fixed_dataset_size, self.size, size=batch_size)
        else: 
            raise RuntimeError("Misspecified range for replay buffer sampling")

        if type == None:
            return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            # torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        elif type == "sas":
            return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device)
        )
        elif type == "sa":
            return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device)
        )
        else: 
            raise RuntimeError("Misspecified return data types for replay buffer sampling")
