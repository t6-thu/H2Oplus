import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import wandb
import time
import gym
import os
import d4rl
from utils.mixed_replay_buffer import MixedReplayBuffer
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
                only_sim=0, 
                penalize_sim=1, 
                sample_ratio=1.0, 
                learning_steps=1e6, 
                sim_warmup=0.0, 
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
        self.d_sa = ConcatDiscriminator(int(num_state + num_action), int(num_hidden), 2, device).float().to(device)
        self.dsa_optim = torch.optim.Adam(self.d_sa.parameters(), lr=3e-4)
        self.d_sas = ConcatDiscriminator(int(2* num_state + num_action), int(num_hidden), 2, device).float().to(device) 
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
        self.sim_only = only_sim
        self.penalize_sim = penalize_sim
        self.sample_ratio = sample_ratio
        self.learning_steps=learning_steps
        self.sim_warmup = sim_warmup
        print(self.penalize_sim)

        self.total_it = 0
        # self.state_mean, self.state_std = self.replay_buffer.get_mean_std()

    def learn(self):
        episode_timesteps = 0
        state, done = self.env.reset(), False

        for t in range(int(self.learning_steps)):
            episode_timesteps += 1
            if self.sim_only and self.sim_warmup > 0:
                self.sim_only = True if t < self.sim_warmup * self.learning_steps else False

            # collect data
            if t <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)
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
                dsa_loss, dsas_loss = self.train_discriminator()

                if t > self.start_steps:
                    self.total_it += 1
                    # For Ablation: Mixed training data stream V.S. Sim Only
                    if self.sim_only:
                        # sample mixed data for training
                        t3 = time.perf_counter()
                        s_train, a_train, next_s_train, r_train, not_done_train = self.replay_buffer.sample(self.batch_size, scope="sim")
                        t4 = time.perf_counter()
                        DEBUG_SampleInterval.append((t4 - t3)*1000)
                        
                        # For Ablation: whether penalize the sim reward
                        if self.penalize_sim:
                            sa_logits = self.d_sa(s_train, a_train)
                            sa_prob = F.softmax(sa_logits, dim=1)
                            # print(sa_prob.shape)
                            adv_logits = self.d_sas(s_train, a_train, next_s_train)
                            sas_prob = F.softmax(adv_logits + sa_logits, dim=1)
                            # print(sas_prob.shape)
                            with torch.no_grad():
                                delta_r = torch.log(sas_prob[:, :1] + 1e-5) - torch.log(sas_prob[:, 1:] + 1e-5) - torch.log(sa_prob[:, :1] + 1e-5) + torch.log(sa_prob[:, 1:] + 1e-5)

                                # another computing pattern
                                # delta_r = torch.log(torch.div(sas_prob[:, :1], sa_prob[:, :1]) + 1e-5) - torch.log(torch.div(sas_prob[:, 1:], sa_prob[:, 1:]) + 1e-5)
                                # assert torch.isnan(delta_r).sum()==0, print(delta_r)
                                # assert torch.isinf(delta_r).sum()==0, print(delta_r)

                                # constant delta_r
                                delta_r = - 0.5 * torch.ones(self.batch_size, 1).to(self.device)
                                # print(delta_r)
                                # FIXME reward calibration
                                r_train += self.alpha * delta_r

                    else:
                        # sample mixed data for training
                        t3 = time.perf_counter()
                        s_real, a_real, next_s_real, r_real, not_done_real = self.replay_buffer.sample(int(self.batch_size / (1 + self.sample_ratio)), scope="real")
                        s_sim, a_sim, next_s_sim, r_sim, not_done_sim = self.replay_buffer.sample(int(self.batch_size * self.sample_ratio / (1 + self.sample_ratio)), scope="sim")
                        t4 = time.perf_counter()
                        DEBUG_SampleInterval.append((t4 - t3)*1000)
                        # print(self.penalize_sim)

                        if self.penalize_sim:
                            sa_logits = self.d_sa(s_sim, a_sim)
                            sa_prob = F.softmax(sa_logits, dim=1)
                            # print(sa_prob.shape)
                            adv_logits = self.d_sas(s_sim, a_sim, next_s_sim)
                            sas_prob = F.softmax(adv_logits + sa_logits, dim=1)
                            # print(sas_prob.shape)
                            with torch.no_grad():
                                delta_r = torch.log(sas_prob[:, :1] + 1e-5) - torch.log(sas_prob[:, 1:] + 1e-5) - torch.log(sa_prob[:, :1] + 1e-5) + torch.log(sa_prob[:, 1:] + 1e-5)

                                # delta_r = torch.log(torch.div(sas_prob[:, :1], sa_prob[:, :1]) + 1e-5) - torch.log(torch.div(sas_prob[:, 1:], sa_prob[:, 1:]) + 1e-5)
                                # assert torch.isnan(delta_r).sum()==0, print(delta_r)
                                # assert torch.isinf(delta_r).sum() == 0, print(delta_r)

                                # delta_r = - 0.5 * torch.ones(int(self.batch_size * self.sample_ratio / (1 + self.sample_ratio)), 1).to(self.device)
                                # print(delta_r)
                                # FIXME reward calibration
                                r_sim += self.alpha * delta_r

                        s_train = torch.cat((s_real, s_sim), dim=0)
                        a_train = torch.cat((a_real, a_sim), dim=0)
                        next_s_train = torch.cat((next_s_real, next_s_sim), dim=0)
                        r_train = torch.cat((r_real, r_sim), dim=0)
                        not_done_train = torch.cat((not_done_real, not_done_sim), dim=0)

                    # update Critic
                    critic_loss = self.train_Q_pi(s_train, a_train, next_s_train, r_train, not_done_train)

                    # delayed policy update
                    if self.total_it % self.policy_freq == 0:
                        # actor loss
                        Q_mean = self.train_actor(s_train)
                        if self.total_it % self.evaluate_freq == 0:
                            ep_rewards = self.rollout_evaluate()
                            eval_dsa_loss, eval_dsas_loss = self.discriminator_evaluate()
                            if self.penalize_sim:
                                delta_max = delta_r.max().cpu().detach().numpy().item()
                                delta_min = delta_r.min().cpu().detach().numpy().item()
                                delta_mean = delta_r.mean().cpu().detach().numpy().item()
                                sas_real_prob = sas_prob[:, :1].mean().cpu().detach().numpy().item()
                                sa_real_prob = sa_prob[:, :1].mean().cpu().detach().numpy().item()
                            else:
                                delta_max, delta_min, delta_mean, sas_real_prob, sa_real_prob = 0, 0, 0, 1, 1
                            wandb.log({
                                    "critic_loss": critic_loss,
                                    "-actor_loss": Q_mean, 
                                    "steps": t,
                                    "eval_episode_rewards": ep_rewards, 
                                    "d_sa_loss": dsa_loss,
                                    "d_sas_loss": dsas_loss,
                                    "eval_dsa": eval_dsa_loss,
                                    "eval_dsas": eval_dsas_loss, 
                                    "max(delta_r)": delta_max,
                                    "min(delta_r)": delta_min, 
                                    "mean(delta_r)": delta_mean,
                                    "sas_real_prob": sas_real_prob, 
                                    "sa_real_prob": sa_real_prob, 
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
        real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.batch_size, scope="real", type="sas")
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.batch_size, scope="sim", type="sas")
        # assert torch.isnan(real_action).sum() == 0, print(real_action)
        # assert torch.isnan(sim_action).sum() == 0, print(sim_action)

        real_sa_logits = self.d_sa(real_obs, real_action)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        # assert torch.isnan(real_sa_prob).sum() == 0, print(real_sa_prob)
        sim_sa_logits = self.d_sa(sim_obs, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        # assert torch.isnan(sim_sa_prob).sum() == 0, print(sim_sa_prob)
        
        real_adv_logits = self.d_sas(real_obs, real_action, real_next_obs)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(sim_obs, sim_action, sim_next_obs)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)

        # assert torch.isnan(real_sas_prob).sum() == 0, print(real_sas_prob)
        # assert torch.isnan(sim_sas_prob).sum() == 0, print(sim_sas_prob)

        dsa_loss = (-torch.log(real_sa_prob[:, :1] + 1e-5) - torch.log(sim_sa_prob[:, 1:] + 1e-5)).mean()
        # assert torch.isnan(dsa_loss).sum() == 0, print(dsa_loss)
        dsas_loss = (-torch.log(real_sas_prob[:, :1] + 1e-5) - torch.log(sim_sas_prob[:, 1:] + 1e-5)).mean()
        # assert torch.isnan(dsas_loss).sum() == 0, print(dsas_loss)

        # Optimize discriminator(s,a) and discriminator(s,a,s')
        self.dsa_optim.zero_grad()
        dsa_loss.backward(retain_graph=True)

        self.dsas_optim.zero_grad()
        dsas_loss.backward()

        self.dsa_optim.step()
        self.dsas_optim.step()


        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()



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
        # Actor loss: + bc loss for warm up
        action_pi = self.actor_net(state)
        Q = self.critic_net.Q1(state, action_pi)
        actor_loss = - Q.mean()

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
            # state = (state - self.state_mean) / self.state_std
            action = self.select_action(state)
            state, reward, done, _ = self.eval_env.step(action)
            ep_rews += reward
            if done:
                break

        return ep_rews
    
    def discriminator_evaluate(self):
        s_real, a_real, next_s_real, r_real, not_done_real = self.replay_buffer.sample(int(self.batch_size/2), scope="real")
        s_sim, a_sim, next_s_sim, r_sim, not_done_sim = self.replay_buffer.sample(int(self.batch_size/2), scope="sim")

        real_sa_logits = self.d_sa(s_real, a_real)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(s_sim, a_sim)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        dsa_loss = ( - torch.log(real_sa_prob[:, :1] + 1e-5) - torch.log(sim_sa_prob[:, 1:] + 1e-5)).mean()

        real_adv_logits = self.d_sas(s_real, a_real, next_s_real)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(s_sim, a_sim, next_s_sim)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)
        dsas_loss = ( - torch.log(real_sas_prob[:, :1] + 1e-5) - torch.log(sim_sas_prob[:, 1:] + 1e-5)).mean()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()