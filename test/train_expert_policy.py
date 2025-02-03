'''
  train expert policy
  @python version : 3.9.5
'''
import d3rlpy
import gym
import datetime

nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')

# dataset, env = d3rlpy.datasets.get_dataset("halfcheetah-medium-v0")
env = gym.make("HalfCheetah-v2")
sac = d3rlpy.algos.SAC()

# sac.fit(dataset=dataset, 
#         n_steps=500000, 
#         eval_episodes=dataset.episodes, 
#         scorers={
#             "environment": d3rlpy.metrics.evaluate_on_environment(env), 
#             "average_value": d3rlpy.metrics.average_value_estimation_scorer, 
#         }
#     )

sac.fit_online(env, n_steps=1000000, eval_env=gym.make("HalfCheetah-v2"))

sac.save_model("data_collector_{}.pt".format(nowTime))

