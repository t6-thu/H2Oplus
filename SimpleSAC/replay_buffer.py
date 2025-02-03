'''
  Replay buffer for storing samples.
  @python version : 3.6.8
'''
import torch
import logging
try:
    import d4rl
except:
    logging.warning('Missing the d4rl pkg!')
import numpy as np
from gym.spaces import Box, Discrete, Tuple

# from envs import get_dim


class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass

class ReplayBuffer(Buffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device(device)

    def append(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def append_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.append(o, a, r, no, d)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
            'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
            'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
            'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
            'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }
        
    def sample_np(self):
        ind = np.random.randint(0, self.size, size=self.size)

        return {
            'observations': self.state[ind], 
            'actions': self.action[ind], 
            'rewards': self.reward[ind], 
            'next_observations': self.next_state[ind], 
            'dones': self.done[ind]
            }

def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
# class ReplayBuffer(Buffer):

#     def __init__(self, env, max_size=int(1e6), batch_size=256, device="cuda"):

#         self.size = max_size
#         self.batch_size = batch_size

#         self.env = env
#         self._ob_space = env.observation_space
#         self._action_space = env.action_space
#         self.s_dim = get_dim(self._ob_space)
#         self.a_dim = get_dim(self._action_space)
#         # self.s_dim = s_dim
#         # self.a_dim = a_dim

#         self.buffer = np.array([], dtype=np.float32)
#         self.device = torch.device(device)
#         # torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # self.device = torch.device(device)

#     def append(self, s, a, r, s_, done):
#         # the current number of samples in buffer
#         sample_num = self.buffer.shape[0]
#         if sample_num > self.size:
#             index = np.random.choice(sample_num, self.size, replace=False).tolist()
#             self.buffer = self.buffer[index]
#         else:
#             pass
        
#         # transpose s, a, r, s_
#         s = s.astype(np.float32)
#         a = a.astype(np.float32)
#         r = np.array(r).astype(np.float32)
#         s_ = s_.astype(np.float32)
#         not_done = np.array(1. - done)

#         recorder = np.hstack((s, a, s_, r, not_done))

#         # append the record into the buffer
#         if sample_num == 0:
#             self.buffer = recorder.copy()
#         else:
#             self.buffer = np.vstack((self.buffer, recorder))

#     def sample(self, batchsize=None):
#         size = batchsize if batchsize is not None else self.batch_size

#         sample_num = self.buffer.shape[0]
#         # batch size > current buffer size, repetition is okay
#         if sample_num < size:
#             sample_index = np.random.choice(sample_num, size, replace=True).tolist()
#         else:
#             sample_index = np.random.choice(sample_num, size, replace=False).tolist()

#         sample = self.buffer[sample_index]

#         s = sample[ : , : self.s_dim]
#         a = sample[ : , self.s_dim : self.s_dim + self.a_dim]
#         s_ = sample[ : , self.s_dim + self.a_dim : 2 * self.s_dim + self.a_dim]
#         r = sample[ : , 2 * self.s_dim + self.a_dim]
#         not_done = sample[ : , 2 * self.s_dim + self.a_dim + 1]

#         s = torch.FloatTensor(np.array(s).astype(np.float32)).to(self.device)
#         a = torch.FloatTensor(np.array(a).astype(np.float32)).to(self.device)
#         s_ = torch.FloatTensor(np.array(s_).astype(np.float32)).to(self.device)
#         r = torch.FloatTensor(np.array(r).astype(np.float32)).to(self.device)
#         not_done = torch.FloatTensor(np.array(not_done)).to(self.device)

#         # sample_dict = {
#         #     "state": s,
#         #     "action": a,
#         #     "state_": s_,
#         #     "reward": r,
#         #     "not_done": not_done
#         # }

#         return s, a, s_, r, not_done

#     def sample_sa_(self):

#         sample_num = self.buffer.shape[0]
#         if sample_num < self.batch_size:
#             sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
#         else:
#             sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

#         sample = self.buffer[sample_index]

#         s = sample[ : , : self.s_dim]
#         a = sample[ : , self.s_dim : self.s_dim + self.a_dim]

#         s = torch.FloatTensor(np.array(s).astype(np.float32)).to(self.device)
#         a = torch.FloatTensor(np.array(a).astype(np.float32)).to(self.device)

#         # sample_dict = {
#         #     "state": s,
#         #     "action": a
#         # }

#         return s, a
