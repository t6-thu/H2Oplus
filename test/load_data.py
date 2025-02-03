import d3rlpy
import d4rl
import gym
import datetime
import h5py
import numpy as np

filename = "../d4rl_mujoco_dataset/halfcheetah_expert-v2.hdf5"

with h5py.File(filename, "r") as dataset:
    # List all groups
    print("Keys: %s" % dataset.keys())


    # print(np.array(dataset['observations']).shape) # An (N, dim_observation)-dimensional numpy array of observations
    # print(np.array(dataset['actions']).shape) # An (N, dim_action)-dimensional numpy array of actions
    # print(np.array(dataset['rewards']).shape) # An (N,)-dimensional numpy array of rewards
    # print(np.array(dataset['next_observations']).shape) # An (N, dim_observation)-dimensional numpy array of next observations
    # print(np.array(dataset['terminals']).shape)
    # print(np.hstack((s, a, s_, r, done))
    print(dataset['rewards'].shape[0])
    s = np.vstack(np.array(dataset['observations'])).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of observations
    a = np.vstack(np.array(dataset['actions'])).astype(np.float32) # An (N, dim_action)-dimensional numpy array of actions
    r = np.vstack(np.array(dataset['rewards'])).astype(np.float32) # An (N,)-dimensional numpy array of rewards
    s_ = np.vstack(np.array(dataset['next_observations'])).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of next observations
    done = np.vstack(np.array(dataset['terminals'])) # An (N,)-dimensional numpy array of terminal flags
    recorder = np.hstack((s, a, s_, r, done))
    print(r.shape[0])
    # print(recorder[5000])
    # print(len(s))
    # print(recorder.shape[0])

# dataset = d3rlpy.dataset.MDPDataset.load('data/real_dataset_21-12-1717:26:45.h5')
# print(dataset.episodes[0]

# # Create the environment
# env = gym.make('halfcheetah-expert-v2')

# # Automatically download and return the dataset
# dataset = env.get_dataset()

