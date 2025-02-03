'''
  Record Real data
  @python version : 3.6.8
'''

import d3rlpy
import gym
import datetime

nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')

env = gym.make("HalfCheetah-v2")
sac = d3rlpy.algos.SAC()
sac.load_model("data_collector.pt")
buffer = d3rlpy.online.buffers.ReplayBuffer(1000000, env=env)

# collect data w/o training
sac.collect(env, buffer, n_steps=1000000)
# convert replay buffer to MDPDataset
dataset = buffer.to_mdp_dataset()

dataset.dump('data/real_dataset_{}.h5'.format(nowTime))
