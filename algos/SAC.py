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
from sample.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor, Double_Critic, Alpha
import matplotlib.pyplot as plt

ALPHA_MAX = 500.
ALPHA_MIN = 0.2
EPS = 1e-8


class SAC:
    max_grad_norm = 0.5
    soft_update = 0.005

    def __init__(self, env_name,
                num_hidden,
                lr_actor=3e-4,
                lr_critic=3e-4,
                gamma=0.99,
                start_steps=10000,
                update_after=1000,
                update_every=50,
                num_test_episodes=10,
                alpha=0.2,
                auto_alpha=True,
                batch_size=256,
                device='cpu'):

        super(SAC, self).__init__()
        self.env = gym.make(env_name)
        # num_state = self.env.observation_space.shape[0]
        # num_action = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()
        self.replay_buffer = MixedReplayBuffer(env=self.env, device=device)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha

        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)

        self.update_after = update_after

        self.q_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.lr_critic)

        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.lr_critic)

        self.file_loc = prepare_env(env_name)

        self.total_it = 0

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # time_steps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
        }

    def learn(self, total_time_step=1e+6):
        episode_timesteps = 0
        state, done = self.env.reset(), False

        # train !
        for t in range(int(total_time_step)):
            episode_timesteps += 1

            # collect data
            if t <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.actor_net.get_action(state).cpu().detach().numpy()
            next_state, reward, done, _ = self.env.step(action)
            done_bool = float(done) if episode_timesteps < 1000 else 0

            # add data in replay buffer
            self.replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state

            # train agent after collection sufficient data
            if t >= self.start_steps:

                self.total_it += 1

                # sample data
                s_train, a_train, next_s_train, _, r_train, not_done_train = self.replay_buffer.sample(self.batch_size)

                # update Critic
                critic_loss = self.train_Q_pi(s_train, a_train, next_s_train, r_train, not_done_train)

                # delayed policy update
                if self.total_it % self.update_after == 0:
                    Q_mean = self.train_actor(s_train)

                    wandb.log({"Q_loss": critic_loss,
                            "Q_mean/-Actor_loss": Q_mean,
                            "steps": t
                            })

            if done:
                state, done = self.env.reset(), False
                episode_timesteps = 0

            if t % self.evaluate_freq == 0:
                self.rollout_evaluate()

        # while i_so_far < total_time_step:
        #     i_so_far += 1
        #
        #     # sample data
        #     state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)
        #     # train Q
        #     q_pi_loss, q_pi_mean = self.train_Q_pi(state, action, next_state, reward, not_done)
        #
        #     # Actor_standard, calculate the log(\miu)
        #     action_pi = self.actor_net.get_action(state)
        #
        #     A = self.q_net(state, action_pi)
        #
        #     # policy update
        #     actor_loss = torch.mean(-1. * A - self.alpha * log_miu)
        #     self.actor_optim.zero_grad()
        #     actor_loss.backward()
        #     self.actor_optim.step()
        #
        #     # evaluate
        #     if i_so_far % 500 == 0:
        #         self.rollout_evaluate(pi='pi')
        #         self.rollout_evaluate(pi='miu')
        #
        #     wandb.log({"actor_loss": actor_loss.item(),
        #                "alpha": self.alpha.item(),
        #                # "w_loss": w_loss.item(),
        #                # "w_mean": w.mean().item(),
        #                "q_pi_loss": q_pi_loss,
        #                "q_pi_mean": q_pi_mean,
        #                })

    def select_action(self, state):
        action = self.actor_net.get_action(state)
        return action.cpu().detach().numpy()

    def update_alpha(self, state, action, log_miu):
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        return self.alpha_net(state, action).detach()

    def train_Q_pi(self, s, a, next_s, r, not_done):
        next_s_pi = next_s
        next_action_pi = self.actor_net.get_action(next_s_pi)

        target_q = r + not_done * self.target_q_net(next_s, next_action_pi).detach() * self.gamma
        loss_q = nn.MSELoss()(target_q, self.q_net(s, a))

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()
        q_mean = self.q_net(s, a).mean().item()

        # target Q update
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.soft_update * param + (1 - self.soft_update) * target_param)

        return loss_q, q_mean

    def rollout_evaluate(self, pi='pi'):
        ep_rews = 0.
        state = self.env.reset()
        ep_t = 0  # episode length
        while True:
            ep_t += 1
            action = self.actor_net.get_action(state).cpu().detach().numpy()
            state, reward, done, _ = self.env.step(action)
            ep_rews += reward
            if done:
                break
        wandb.log({"pi_episode_reward": ep_rews})