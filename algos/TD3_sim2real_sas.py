'''
delta_r on both real and sim data
'''

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time
import gym
import os
import d4rl
from utils.mixed_replay_buffer_nonterm import MixedReplayBuffer
# from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic
from Network.Weight_net import ConcatDiscriminator
import matplotlib.pyplot as plt

DEBUG_AppendInterval = []
DEBUG_SampleInterval = []

class Sim2real_TD3:
    def __init__(self,
                sim_env, 
                real_env, 
                real_env_name,
                data_source, 
                num_hidden=int(256),
                gamma=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                explore_freq=10,
                start_steps=25e3, 
                alpha=1.0, 
                device='cpu'):

        super(Sim2real_TD3, self).__init__()
        # TODO prepare sim and real environments
        self.eval_env = real_env
        self.env = sim_env
        
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        # TODO Mixed Replay Buffer
        self.replay_buffer = MixedReplayBuffer(state_dim=num_state, action_dim=num_action, task=real_env_name.split("-")[0].lower(), data_source=data_source, device=device)

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # TODO prepare the discriminator for (s, a) and (s, a, s') respectively
        self.d_sas = ConcatDiscriminator(int(2* num_state + num_action), int(num_hidden), int(1), device).float().to(device)
        self.dsas_optim = torch.optim.Adam(self.d_sas.parameters(), lr=3e-4)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.explore_freq = explore_freq
        self.start_steps = start_steps
        self.noise_clip = noise_clip
        self.evaluate_freq = 2000
        self.batch_size = 256
        self.device = device
        self.max_action = 1.
        self.alpha = alpha

        self.total_it = 0

        # Q and Critic file location
        # self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+6):
        episode_timesteps = 0
        state, done = self.env.reset(), False

        for t in range(int(total_time_step)):
            episode_timesteps += 1
            self.total_it += 1

            # TODO collect data
            # if t <= self.start_steps:
            #     action = self.env.action_space.sample()
            # else:
            # print("State: ", state)
            action = self.select_action(state)
            # while True:
            #     action = self.select_action(state)
            #     if not np.isnan(action[0]):
            #         break

            # if t < self.start_steps:
            #     print("Normal Action:", action)
            # else:
            #     print("Abnormal Action:", action)
            next_state, reward, done, _ = self.env.step(action)
            done_bool = float(done) if episode_timesteps < 1000 else 0

            # add data in replay buffer
            t1 = time.perf_counter()
            self.replay_buffer.append(state, action, s_=next_state, r=reward, done=done_bool)
            t2 = time.perf_counter()
            DEBUG_AppendInterval.append((t2 - t1)*1000)
            state = next_state

            # train agent after collection sufficient data
            if t > self.batch_size:
                dsas_loss = self.train_discriminator()
                if t > self.start_steps:
                    # sample mixed data for training
                    t3 = time.perf_counter()
                    s_train, a_train, next_s_train, r_train, not_done_train = self.replay_buffer.sample(self.batch_size)
                    t4 = time.perf_counter()
                    DEBUG_SampleInterval.append((t4 - t3)*1000)


                    real_sas_prob = self.d_sas(s_train, a_train, next_s_train)
                    with torch.no_grad():
                        delta_r = torch.log(real_sas_prob + 1e-5) - torch.log(1. - real_sas_prob + 1e-5)
                        assert torch.isnan(delta_r).sum()==0, print(delta_r)
                        assert torch.isinf(delta_r).sum() == 0, print(delta_r)
                        # print(delta_r)
                        # FIXME reward calibration
                        r_train += self.alpha * delta_r
                else:
                    t3 = time.perf_counter()
                    s_train, a_train, next_s_train, r_train, not_done_train = self.replay_buffer.sample(self.batch_size, range="real")
                    t4 = time.perf_counter()
                    DEBUG_SampleInterval.append((t4 - t3)*1000)
            else:
                t3 = time.perf_counter()
                s_train, a_train, next_s_train, r_train, not_done_train = self.replay_buffer.sample(self.batch_size, range="real")
                t4 = time.perf_counter()
                DEBUG_SampleInterval.append((t4 - t3)*1000)


            # update Critic
            critic_loss = self.train_Q_pi(s_train, a_train, next_s_train, r_train, not_done_train)

            # delayed policy update
            if self.total_it % self.policy_freq == 0:
                Q_mean = self.train_actor(s_train)
                if self.total_it % self.evaluate_freq == 0:
                    ep_rewards = self.rollout_evaluate()
                    eval_dsas_loss = self.discriminator_evaluate()
                    wandb.log({"critic_loss": critic_loss,
                            "-actor_loss": Q_mean,
                            "steps": t,
                            "eval_episode_rewards": ep_rewards, 
                            "d_sas_loss": dsas_loss,
                            "eval_dsas": eval_dsas_loss,
                            "append_interval/ms": sum(DEBUG_AppendInterval) / len(DEBUG_AppendInterval), 
                            "sample_interval/ms": sum(DEBUG_SampleInterval) /
                            len(DEBUG_SampleInterval)
                            })
                    DEBUG_AppendInterval.clear()
                    DEBUG_SampleInterval.clear()

            if done:
                state, done = self.env.reset(), False
                episode_timesteps = 0

    def train_discriminator(self):
        # # current_step = 0
        # for t in range(int(time_steps)):
        #     # current_step += 1

        real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.batch_size, range="real", type="sas")
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.batch_size, range="sim", type="sas")
        # assert torch.isnan(real_action).sum() == 0, print(real_action)
        # assert torch.isnan(sim_action).sum() == 0, print(sim_action)
        
        real_sas_prob = self.d_sas(real_obs, real_action, real_next_obs)
        sim_sas_prob = self.d_sas(sim_obs, sim_action, sim_next_obs)

        # assert torch.isnan(real_sas_prob).sum() == 0, print(real_sas_prob)
        # assert torch.isnan(sim_sas_prob).sum() == 0, print(sim_sas_prob)

        dsas_loss = (-torch.log(real_sas_prob + 1e-5)- torch.log(1. - sim_sas_prob + 1e-5)).mean()
        # assert torch.isnan(dsas_loss).sum() == 0, print(dsas_loss)
        

        # Optimize discriminator(s,a) and discriminator(s,a,s')

        self.dsas_optim.zero_grad()
        dsas_loss.backward()
        self.dsas_optim.step()


        return dsas_loss.cpu().detach().numpy().item()



    def train_Q_pi(self, state, action, next_state, reward, not_done):
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # assert torch.isnan(next_action).sum()==0, print(next_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        Q1, Q2 = self.critic_net(state, action)
        # assert torch.isnan(Q1).sum() == 0, print(Q1)
        # assert torch.isnan(Q2).sum() == 0, print(Q2)
        # assert torch.isnan(target_Q).sum() == 0, print(target_Q)
        # assert torch.isinf(target_Q).sum() == 0, print(target_Q)
        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)
        # assert torch.isnan(critic_loss).sum() == 0, print(critic_loss)
        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_actor(self, state):
        # Actor loss
        action_pi = self.actor_net(state)
        Q = self.critic_net.Q1(state, action_pi)
        # assert torch.isnan(Q).sum()==0, print(Q)
        actor_loss = -Q.mean()
        # assert torch.isnan(action_pi).sum() == 0, print(action_pi)
        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # assert torch.isnan(actor_loss).sum() == 0, print(actor_loss)
        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return Q.mean().cpu().detach().numpy().item()

    def select_action(self, state):
        action = self.actor_net(state)
        # assert torch.isnan(action).sum() == 0, print(action)
        # print("Pure Action: ", action)
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # print("Noise: ", noise)
        action = (action + noise).clamp(-self.max_action, self.max_action)
        # print("Clamped Action: ", action)
        return action.cpu().detach().numpy()

    def rollout_evaluate(self):
        ep_rews = 0.
        ep_lens = 0
        state = self.eval_env.reset()
        while True:
            ep_lens += 1
            action = self.select_action(state)
            state, reward, done, _ = self.eval_env.step(action)
            ep_rews += reward
            if done:
                break

        return ep_rews
    
    def discriminator_evaluate(self):
        s_real, a_real, next_s_real, r_real, not_done_real = self.replay_buffer.sample(int(self.batch_size/2), range="real")
        s_sim, a_sim, next_s_sim, r_sim, not_done_sim = self.replay_buffer.sample(int(self.batch_size/2), range="sim")

        real_sas_prob = self.d_sas(s_real, a_real, next_s_real)
        sim_sas_prob = self.d_sas(s_sim, a_sim, next_s_sim)
        dsas_loss = (- torch.log(real_sas_prob + 1e-5) - torch.log(1. - sim_sas_prob + 1e-5)).mean()

        return dsas_loss.cpu().detach().numpy().item()