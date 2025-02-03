import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device(device)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset, scale_rewards=True, scale_state=True):
        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(dataset['terminals']),
                np.arange(dataset_size) < dataset_size - 1))

        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size)) 

        self.state = dataset['observations'][nonterminal_steps]
        self.action = dataset['actions'][nonterminal_steps]
        self.next_state = dataset['next_observations'][nonterminal_steps + 1]
        self.next_action = dataset['actions'][nonterminal_steps + 1]
        self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)
        self.size = self.state.shape[0]

        if scale_rewards:
            r_max = np.max(self.reward)
            r_min = np.min(self.reward)
            self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean()
        s_std = self.state.std()

        return s_mean, s_std

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std