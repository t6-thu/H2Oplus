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
from sample.Sample_from_dataset import ReplayBuffer
from sample.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic
import matplotlib.pyplot as plt

DEBUG_AppendInterval = []
DEBUG_SampleInterval = []

class TD3:
    def __init__(self,
                env_name,
                num_hidden=256,
                gamma=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                explore_freq=10,
                start_steps=25e3,
                device='cpu'):

        super(TD3, self).__init__()
        # prepare the environment
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.explore_freq = explore_freq
        self.start_steps = start_steps
        self.noise_clip = noise_clip
        self.evaluate_freq = 3000
        self.batch_size = 256
        self.device = device
        self.max_action = 1.

        self.total_it = 0

        # Q and Critic file location
        # self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+6):
        episode_timesteps = 0
        state, done = self.env.reset(), False

        for t in range(int(total_time_step)):
            episode_timesteps += 1

            # collect data
            if t <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            done_bool = float(done) if episode_timesteps < 1000 else 0

            # add data in replay buffer
            t1 = time.perf_counter()
            self.replay_buffer.add(state, action, next_state, reward, done_bool)
            t2 = time.perf_counter()
            DEBUG_AppendInterval.append((t2 - t1)*1000)
            state = next_state

            # train agent after collection sufficient data
            if t >= self.start_steps:

                self.total_it += 1

                # sample data
                t3 = time.perf_counter()
                s_train, a_train, next_s_train, _, r_train, not_done_train = self.replay_buffer.sample(self.batch_size)
                t4 = time.perf_counter()
                DEBUG_SampleInterval.append((t4 - t3)*1000)

                # update Critic
                critic_loss = self.train_Q_pi(s_train, a_train, next_s_train, r_train, not_done_train)

                # delayed policy update
                if self.total_it % self.policy_freq == 0:
                    Q_mean = self.train_actor(s_train)

                    if self.total_it % self.evaluate_freq == 0:
                        ep_rews = self.rollout_evaluate()
                        wandb.log({"critic_loss": critic_loss,
                                "-actor_loss": Q_mean,
                                "steps": t,
                                "eval_episode_rewards": ep_rews, 
                                "append_interval/ms": sum(DEBUG_AppendInterval) / len(DEBUG_AppendInterval), 
                                "sample_interval/ms": sum(DEBUG_SampleInterval) /
                                len(DEBUG_SampleInterval)
                                })
                        DEBUG_AppendInterval.clear()
                        DEBUG_SampleInterval.clear()

            if done:
                state, done = self.env.reset(), False
                episode_timesteps = 0

    def train_Q_pi(self, state, action, next_state, reward, not_done):
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        Q1, Q2 = self.critic_net(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_actor(self, state):
        # Actor loss
        action_pi = self.actor_net(state)
        Q = self.critic_net.Q1(state, action_pi)
        actor_loss = -Q.mean()

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return Q.mean().cpu().detach().numpy().item()

    def select_action(self, state):
        action = self.actor_net(state)
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action = (action + noise).clamp(-self.max_action, self.max_action)
        return action.cpu().detach().numpy()

    def rollout_evaluate(self):
        ep_rews = 0.
        ep_lens = 0
        state = self.env.reset()
        while True:
            ep_lens += 1
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            ep_rews += reward
            if done:
                break
        return ep_rews