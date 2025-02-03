import imp
import pdb
from collections import OrderedDict
from copy import deepcopy
from distutils.command.config import config
# from turtle import pd
from certifi import where

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections import ConfigDict
from torch import ne, nn as nn

from model import Scalar, soft_target_update
from utils import prefix_metrics
# import ipdb


class H2OPLUS(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.batch_size = 256
        config.batch_sim_ratio = 0.5
        config.device = 'cuda'
        config.discount = 0.99
        config.quantile = 0.5
        config.exploit_coeff = 0.5
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.automatic_exploit_coeff_tuning = False
        config.backup_policy_entropy = True
        config.backup_dynamics_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.vf_lr = 3e-4
        config.d_sa_lr = 3e-4
        config.d_sas_lr = 3e-4
        config.dynamics_ratio_lr = 3e-4
        config.noise_std_discriminator = 0.1
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.temperature = 1
        # config.use_cql = True
        # config.use_variant = False
        # config.u_ablation = False
        config.in_sample = True
        config.use_quantile_regression = True
        config.sparse_alpha = 1.0
        config.use_td_target_ratio = True
        config.use_real_blended_target = True
        config.use_sim_blended_target = True
        # config.use_sim_q_coeff = True
        # config.use_kl_baseline = False
        # config.fix_baseline_steps = 10
        # kl divergence: E_pM log(pM/pM^)
        # config.sim_q_coeff_min = 1e-45
        # config.sim_q_coeff_max = 10
        config.sampling_n_next_states = 10
        config.f = "algae"
        # config.s_prime_std_ratio = 1.
        # config.cql_n_actions = 10
        # config.cql_importance_sample = True
        # config.cql_lagrange = False
        # config.cql_target_action_gap = 1.0
        # config.cql_temp = 1.0
        # config.cql_min_q_weight = 0.01
        # config.cql_max_target_backup = False
        # config.cql_clip_diff_min = -1000
        # config.cql_clip_diff_max = 1000
        # pM/pM^
        config.clip_dynamics_ratio_min = 1e-5
        config.clip_dynamics_ratio_max = 1
        # config.sa_prob_clip = 0.0
        # gumbel loss
        # config.use_gumbel_regression = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, d_sa=None, d_sas=None, dynamics_model=None, dynamics_ratio_estimator=None):
        self.config = H2OPLUS.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        self.next_observation_sampler = dynamics_model
        if self.next_observation_sampler:
            self.dynamics_ratio_estimator = dynamics_ratio_estimator
            self._f = self.config.f
            self.dr_activation = lambda x: torch.relu(x)
            
            # f-divergence functions
            # NOTE: g(x) = f(ReLU((f')^{-1}(x)))
            # NOTE: r(x) = ReLU((f')^{-1}(x)
            if self._f == 'algae':
                self._f_fn = lambda x : 0.5 * x ** 2
                self._f_star = lambda x : 0.5 * x ** 2
                self._f_star_prime = lambda x: torch.relu(x)
                # self._f_prime_fn = lambda x: torch.where(x < 1e-10, 1e-10, x)
                self._f_prime_fn = lambda x: torch.where(x.double() < 1e-10, 1e-10, x.double())
            elif self._f == 'js':
                pass
            elif self._f == 'chisquare':
                self._f_fn = lambda x: 0.5 * (x - 1) ** 2
                self._f_star_prime = lambda x: torch.relu(x + 1)
                self._f_star = lambda x: 0.5 * x ** 2 + x 
                self._f_prime_inv_fn = lambda x: x + 1
                self._g_fn = lambda x: 0.5 * (nn.relu(x + 1) - 1) ** 2
                self._r_fn = lambda x: nn.relu(self._f_prime_inv_fn(x))
                self._log_r_fn = lambda x: torch.where(x < 0, torch.log(1e-10), torch.log(torch.maximum(x, 0) + 1))
            elif self._f == 'kl':
                self._f_fn = lambda x: x * torch.log(x + 1e-10)
                self._f_star = lambda x: torch.exp(x - 1)
                self._f_star_prime = lambda x: torch.exp(x - 1)
                self._f_prime_inv_fn = lambda x: torch.exp(x - 1)
                # self._f_prime_fn = lambda x: torch.where(1 + torch.log(x + 1e-10) < 1e-10, 1e-10, 1 + torch.log(x + 1e-10))
                # self._f_prime_fn = lambda x: torch.where(1 + torch.log(torch.maximum(x, 1e-10)) < 1e-10, 1e-10, 1 + torch.log(torch.maximum(x, 1e-10)))
                # self._f_prime_fn = lambda x: torch.where(x.double() <= torch.exp(torch.tensor(-1.0)), 1e-10, 1 + torch.log(x.double() + 1e-10))
                self._f_prime_fn = lambda x: 1 + torch.log(x.double() + 1e-10)
                # self._f_prime_fn = lambda x: torch.where(x <= 0.4, 1e-10, 1 + torch.log(x))
                # self._f_prime_fn = lambda x: torch.where(x < 0, torch.log(1e-10), torch.log(torch.maximum(x, 0) + 1))
                self._g_fn = lambda x: torch.exp(x - 1) * (x - 1)
                self._r_fn = lambda x: self._f_prime_inv_fn(x)
                self._log_r_fn = lambda x: x - 1
            elif self._f == 'elu':
                self._f_fn = lambda x: torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2)
                self._f_prime_inv_fn = lambda x: torch.where(x < 0, torch.exp(torch.minimum(x, 0)), x + 1)
                self._g_fn = lambda x: torch.where(x < 0, torch.exp(torch.minimum(x, 0)) * (torch.minimum(x, 0) - 1) + 1, 0.5 * x ** 2)
                self._r_fn = lambda x: self._f_prime_inv_fn(x)
                self._log_r_fn = lambda x: torch.where(x < 0, x, torch.log(torch.maximum(x, 0) + 1))
            else:
                raise NotImplementedError()
        else:
            self.d_sa = d_sa
            self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.quantile = self.config.quantile
        self.exploit_coeff = self.config.exploit_coeff
        self.mean, self.std = self.replay_buffer.get_mean_std()
        self.kl_baseline = 1
        self._last_q_grad = torch.ones((int(self.config.batch_size * (1 - self.config.batch_sim_ratio)), 1)).to(device=self.config.device)

        '''
        Optimizers
        '''
        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), self.config.vf_lr
        )

        if self.next_observation_sampler:
            self.dynamics_ratio_estimator_optimizer = optimizer_class(self.dynamics_ratio_estimator.parameters(), self.config.dynamics_ratio_lr)
        else:
            self.d_sa_optimizer = optimizer_class(self.d_sa.parameters(), self.config.d_sa_lr)
            self.d_sas_optimizer = optimizer_class(self.d_sas.parameters(), self.config.d_sas_lr)

        # whether to use automatic entropy tuning (True in default)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch_size, pretrain_steps):
        self._total_steps += 1
        
        real_batch = self.replay_buffer.sample(int(batch_size * (1 - self.config.batch_sim_ratio)), scope="real")
        sim_batch = self.replay_buffer.sample(int(batch_size * self.config.batch_sim_ratio), scope="sim")

        # real transitions from d^{\pi_\beta}_\mathcal{M}
        real_observations = real_batch['observations']
        real_actions = real_batch['actions']
        real_rewards = real_batch['rewards'].squeeze()
        real_next_observations = real_batch['next_observations']
        real_dones = real_batch['dones'].squeeze() 
        
        if self._total_steps <= pretrain_steps:
            real_q = torch.min(
                self.qf1(real_observations, real_actions),
                self.qf2(real_observations, real_actions),
            )
            real_v = self.vf(real_observations)
            exp_weight = torch.exp((real_q - real_v.squeeze()) * self.config.temperature)
            exp_weight = torch.clamp(exp_weight, max=100)
            real_log_prob = self.policy.log_prob(real_observations, real_actions)
            policy_loss = - (exp_weight.detach() * real_log_prob).mean()
            
            """ Q function loss """
            # Q function in real data and sim data
            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)
            
            # * Exploitation Bellman Backup
            # estimate BQ(s,a)
            real_next_vf_pred = self.vf(real_next_observations).squeeze()
            real_exploit_td_target = real_rewards + (1. - real_dones) * self.config.discount * real_next_vf_pred

            real_qf1_loss = F.mse_loss(real_q1_pred, real_exploit_td_target.detach())
            real_qf2_loss = F.mse_loss(real_q2_pred, real_exploit_td_target.detach())
            
            qf_loss = real_qf1_loss + real_qf2_loss
            
            metrics = dict(
                mean_real_rewards=real_rewards.mean(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_real_exploit_target_q=real_exploit_td_target.mean().item(),
                
                total_steps=self.total_steps,
            )
        else:
            # sim transitions from d^\pi_\mathcal{\widehat{M}}
            sim_observations = sim_batch['observations']
            sim_actions = sim_batch['actions']
            sim_rewards = sim_batch['rewards'].squeeze()
            sim_next_observations = sim_batch['next_observations']
            sim_dones = sim_batch['dones'].squeeze()
            
            # mixed transitions from d_f = f * d^{\pi_\beta}_\mathcal{M} + (1-f) * d^\pi_\mathcal{\widehat{M}}
            # ipdb.set_trace()
            df_observations = torch.cat([real_observations, sim_observations], dim=0)
            df_actions =  torch.cat([real_actions, sim_actions], dim=0)
            # df_rewards =  torch.cat([real_rewards, sim_rewards], dim=0)
            # df_next_observations =  torch.cat([real_next_observations, sim_next_observations], dim=0)
            # df_dones =  torch.cat([real_dones, sim_dones], dim=0)
            if self.next_observation_sampler:
                dynamics_ratio_loss = self.train_dynamics_ratio()
            else:
                dsa_loss, dsas_loss = self.train_discriminator()

            # TODO new_action and log pi
            df_new_actions, df_log_pi = self.policy(df_observations)

            # True by default
            if self.config.use_automatic_entropy_tuning:
                alpha_loss = - (self.log_alpha() * (df_log_pi + self.config.target_entropy).detach()).mean()
                alpha = self.log_alpha().exp() * self.config.alpha_multiplier
            else:
                alpha_loss = df_observations.new_tensor(0.0)
                alpha = df_observations.new_tensor(self.config.alpha_multiplier)

            """ Policy loss """
            q_new_actions = torch.min(
                self.qf1(df_observations, df_new_actions),
                self.qf2(df_observations, df_new_actions),
            )
            policy_loss = (alpha * df_log_pi - q_new_actions).mean()
            
            # if self._total_steps % 1000 == 0:
            #     ipdb.set_trace()
            # if self.next_observation_sampler:
            #     # ipdb.set_trace()
            #     log_sim_real_dynamics_ratio = - torch.log(self.dr_activation(self.dynamics_ratio_estimator(sim_observations, sim_actions, sim_next_observations)))
            # else:
            #     log_sim_real_dynamics_ratio = self.log_sim_real_dynamics_ratio(sim_observations, sim_actions, sim_next_observations)
            # q_new_actions = self.qf1(df_observations, df_new_actions)
            # policy_loss = - q_new_actions.mean()
            # if self.config.backup_dynamics_entropy:
            #     policy_loss += self.config.batch_sim_ratio * log_sim_real_dynamics_ratio.mean()
            
            # df_q = torch.min(
            #     self.qf1(df_observations, df_actions),
            #     self.qf2(df_observations, df_actions),
            # )
            # df_v = self.vf(df_observations)
            # exp_weight = torch.exp((df_q - df_v.squeeze()) * self.config.temperature)
            # exp_weight = torch.clamp(exp_weight, max=100)
            # df_log_prob = self.policy.log_prob(df_observations, df_actions)
            # policy_loss = - (exp_weight.detach() * df_log_prob).mean()
            # # ipdb.set_trace()

            """ Q function loss """
            # Q function in real data and sim data
            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)
            sim_q1_pred = self.qf1(sim_observations, sim_actions)
            sim_q2_pred = self.qf2(sim_observations, sim_actions)
            
            # * Exploration Bellman Backup
            # with torch.no_grad():
            real_new_next_actions, real_next_log_pi = self.policy(real_next_observations)
            real_target_q_values = torch.min(
                self.target_qf1(real_next_observations, real_new_next_actions),
                self.target_qf2(real_next_observations, real_new_next_actions),
            )
            sim_new_next_actions, sim_next_log_pi = self.policy(sim_next_observations)
            sim_target_q_values = torch.min(
                self.target_qf1(sim_next_observations, sim_new_next_actions),
                self.target_qf2(sim_next_observations, sim_new_next_actions),
            )
            
            # estimate real and sim BQ(s,a)
            if self.config.backup_policy_entropy:
                real_target_q_values -= alpha * real_next_log_pi
                sim_target_q_values -= alpha * sim_next_log_pi
            real_explore_td_target = real_rewards + (1. - real_dones) * self.config.discount * real_target_q_values
            sim_explore_td_target = sim_rewards + (1. - sim_dones) * self.config.discount * sim_target_q_values
            
            # * Exploitation Bellman Backup
            # estimate BQ(s,a)
            real_next_vf_pred = self.vf(real_next_observations).squeeze()
            real_exploit_td_target = real_rewards + (1. - real_dones) * self.config.discount * real_next_vf_pred
            # sim_vf_pred = self.vf(sim_observations).squeeze()
            sim_next_vf_pred = self.vf(sim_next_observations).squeeze()
            sim_exploit_td_target = sim_rewards + (1. - sim_dones) * self.config.discount * sim_next_vf_pred
            
            if self.config.automatic_exploit_coeff_tuning:
                q_grad = real_target_q_values - torch.min(real_q1_pred, real_q2_pred)
                lambda_q_d = q_grad / self._last_q_grad
                self.exploit_coeff = torch.clamp(lambda_q_d, 0., 1.).detach()
                self._last_q_grad = q_grad
                self.exploit_coeff = torch.mean(self.exploit_coeff[0])
            
            # * Mixed Bellman Backup: Exploration and Exploitation
            if self.config.use_real_blended_target:
                real_td_target = self.exploit_coeff * real_exploit_td_target + (1 - self.exploit_coeff) * real_explore_td_target
            else:
                real_td_target = real_exploit_td_target
                
            if self.config.use_sim_blended_target:
                sim_td_target = self.exploit_coeff * sim_exploit_td_target + (1 - self.exploit_coeff) * sim_explore_td_target
            else:
                sim_td_target = sim_explore_td_target
            real_qf1_loss = F.mse_loss(real_q1_pred, real_td_target.detach())
            real_qf2_loss = F.mse_loss(real_q2_pred, real_td_target.detach())
            # importance sampling on td error due to the dyanmics shift
            if self.config.use_td_target_ratio:
                if self.next_observation_sampler:
                    sqrt_real_sim_ratio = torch.clamp(self.dr_activation(self.dynamics_ratio_estimator(sim_observations, sim_actions, sim_next_observations)).detach(), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max).sqrt()
                else:
                    sqrt_real_sim_ratio = torch.clamp(self.real_sim_dynamics_ratio(sim_observations, sim_actions, sim_next_observations), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max).sqrt()
            else:
                sqrt_real_sim_ratio = torch.ones((sim_observations.shape[0],)).to(self.config.device)
            sim_qf1_loss = F.mse_loss(sqrt_real_sim_ratio.squeeze() * sim_q1_pred, sqrt_real_sim_ratio.squeeze() * sim_td_target.detach())
            sim_qf2_loss = F.mse_loss(sqrt_real_sim_ratio.squeeze() * sim_q2_pred, sqrt_real_sim_ratio.squeeze() * sim_td_target.detach())
            
            qf_loss = real_qf1_loss + sim_qf1_loss + real_qf2_loss + sim_qf2_loss
            
            metrics = dict(
                # log_sim_real_dynamics_ratio=log_sim_real_dynamics_ratio.mean(),
                exploit_coeff=self.exploit_coeff,
                sqrt_IS_ratio=sqrt_real_sim_ratio.mean(),
                mean_real_rewards=real_rewards.mean(),
                mean_sim_rewards=sim_rewards.mean(),
                max_sim_rewards=sim_rewards.max(),
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_sim_qf1=sim_q1_pred.mean().item(),
                average_sim_qf2=sim_q2_pred.mean().item(),
                
                average_real_explore_target_q=real_target_q_values.mean().item(),
                average_sim_explore_target_q=sim_target_q_values.mean().item(),
                average_real_exploit_target_q=real_exploit_td_target.mean().item(),
                average_sim_exploit_target_q=sim_exploit_td_target.mean().item(),
                
                average_real_explore_td_target=real_explore_td_target.mean().item(),
                average_real_exploit_td_target=real_exploit_td_target.mean().item(),
                average_sim_explore_td_target=sim_explore_td_target.mean().item(),
                average_sim_exploit_td_target=sim_exploit_td_target.mean().item(),
                average_blended_real_td_target=real_td_target.mean().item(),
                average_blended_sim_td_target=sim_td_target.mean().item(),
                
                total_steps=self.total_steps,
            )
            if self.next_observation_sampler:
                metrics.update(dict(
                    dynamics_ratio_loss=dynamics_ratio_loss,
                ))
            else:
                metrics.update(dict(
                    dsa_train_loss=dsa_loss,
                    dsas_train_loss=dsas_loss,
                ))
                
            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            
        """ V function loss """
        # Q_target(s,a)
        real_vf_pred = self.vf(real_observations).squeeze()
        real_qf_target_pred = torch.min(
            self.target_qf1(real_observations, real_actions),
            self.target_qf2(real_observations, real_actions),
        )
        # Q_target(s,a) - V(s)
        real_vf_error = real_qf_target_pred - real_vf_pred
        if self.config.use_quantile_regression:
            # quantile regression
            vf_sign = (real_vf_error < 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (real_vf_error ** 2)).mean()
        else:
            # sparse regression
            sparse_term = real_vf_error / (2 * self.config.sparse_alpha) + 1.0
            vf_sign = (sparse_term > 0).float()
            vf_loss = (vf_sign * (sparse_term ** 2) + real_vf_pred / self.config.sparse_alpha).mean()
            
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        # nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=0.1)
        # nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=0.1)
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        metrics.update(dict(
            vf_loss=vf_loss.item(),
            df_vf_pred=real_vf_pred.mean().item(),
            df_qf_target_pred=real_qf_target_pred.mean().item(),
            df_vf_error=real_vf_error.mean().item(),
        ))

        return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.in_sample:
            modules.append(self.vf)
        # if self.config.cql_lagrange:
        #     modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    # def gumbel_loss(pred, label, eta=1.0, clip=1000):
    #     z = (label - pred) / eta
    #     z = torch.clamp(z, -clip, clip)
    #     max_z = torch.max(z)
    #     max_z = torch.where(max_z < 1.0, torch.tensor(-1.0), max_z)
    #     max_z = max_z.detach()
    #     loss = torch.exp(-max_z) * (torch.exp(z) - z - 1)
    #     return loss.mean()

    def train_discriminator(self):
        real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()

        # input noise: prevents overfitting
        if self.config.noise_std_discriminator > 0:
            real_obs += torch.randn(real_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            real_action += torch.randn(real_action.shape, device=self.config.device) * self.config.noise_std_discriminator
            real_next_obs += torch.randn(real_next_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_obs += torch.randn(sim_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_action += torch.randn(sim_action.shape, device=self.config.device) * self.config.noise_std_discriminator
            sim_next_obs += torch.randn(sim_next_obs.shape, device=self.config.device) * self.config.noise_std_discriminator
        
        real_sa_logits = self.d_sa(real_obs, real_action)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(sim_obs, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        
        # pdb.set_trace()
        real_adv_logits = self.d_sas(real_obs, real_action, real_next_obs)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(sim_obs, sim_action, sim_next_obs)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)

        dsa_loss = (- torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()
        dsas_loss = (- torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()

        # Optimize discriminator(s,a) and discriminator(s,a,s')
        self.d_sa_optimizer.zero_grad()
        dsa_loss.backward(retain_graph=True)

        self.d_sas_optimizer.zero_grad()
        dsas_loss.backward()

        self.d_sa_optimizer.step()
        self.d_sas_optimizer.step()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()


    def discriminator_evaluate(self):
        s_real, a_real, next_s_real = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        s_sim, a_sim, next_s_sim = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()
        
        real_sa_logits = self.d_sa(s_real, a_real)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(s_sim, a_sim)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        dsa_loss = ( - torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()

        real_adv_logits = self.d_sas(s_real, a_real, next_s_real)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(s_sim, a_sim, next_s_sim)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)
        dsas_loss = ( - torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()


    def log_sim_real_dynamics_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        # sa_prob = np.clip(sa_prob, a_min=self.config.sa_prob_clip, a_max=1 - self.config.sa_prob_clip)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM^/pM
            log_ratio = - torch.log(sas_prob[:, 0]) \
                    + torch.log(sas_prob[:, 1]) \
                    + torch.log(sa_prob[:, 0]) \
                    - torch.log(sa_prob[:, 1])
            # log_ratio = torch.clamp(torch.log(sas_prob[:, 0]) - torch.log(sas_prob[:, 1]) - torch.log(sa_prob[:, 0]) + torch.log(sa_prob[:, 1]), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max)
        
        return log_ratio

    def log_real_sim_dynamics_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        # pdb.set_trace()
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            log_ratio = torch.log(sas_prob[:, 0]) \
                    - torch.log(sas_prob[:, 1]) \
                    - torch.log(sa_prob[:, 0]) \
                    + torch.log(sa_prob[:, 1])
            # log_ratio = torch.clamp(torch.log(sas_prob[:, 0]) - torch.log(sas_prob[:, 1]) - torch.log(sa_prob[:, 0]) + torch.log(sa_prob[:, 1]), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max)
        
        return log_ratio

    def real_sim_dynamics_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            ratio = (sas_prob[:, 0] * sa_prob[:, 1]) / (sas_prob[:, 1] * sa_prob[:, 0])
            # log_ratio = torch.clamp(torch.log(sas_prob[:, 0]) - torch.log(sas_prob[:, 1]) - torch.log(sa_prob[:, 0]) + torch.log(sa_prob[:, 1]), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max)
        
        return ratio
    
    def train_dynamics_ratio(self):
        # real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()
        
        # expand_obs = torch.repeat_interleave(sim_obs, self.config.sampling_n_next_states, dim=0)
        # expand_action = torch.repeat_interleave(sim_action, self.config.sampling_n_next_states, dim=0)
        # expand_sim_next_obs = torch.repeat_interleave(sim_next_obs, self.config.sampling_n_next_states, dim=0)
        real_model_next_obs = self.next_observation_sampler.get_next_state(sim_obs, sim_action, 1).reshape((-1, sim_obs.shape[1]))
        
        if self._f == 'js':
            sim_weight = self.dr_activation(
                self.dynamics_ratio_estimator(sim_obs, sim_action, sim_next_obs)
            )
            # sim_f_star ++ -> 0 if offline_weight -> 1
            sim_f_star = - torch.log(2.0 / (sim_weight + 1) + 1e-10)

            real_weight = self.dr_activation(self.dynamics_ratio_estimator(sim_obs, sim_action, real_model_next_obs.detach()))
            # real_f_prime -- -> -infty if online_weight -> 0,  
            real_f_prime = torch.log(2 * real_weight / (real_weight + 1) + 1e-10)

            dynamics_ratio_loss = (sim_f_star - real_f_prime).mean()
        else:
            dynamics_ratio_loss = (self._f_star(self._f_prime_fn(self.dr_activation(self.dynamics_ratio_estimator(sim_obs, sim_action, sim_next_obs)))) - self._f_prime_fn(self.dr_activation(self.dynamics_ratio_estimator(sim_obs, sim_action, real_model_next_obs)))).mean()
        
        self.dynamics_ratio_estimator_optimizer.zero_grad()
        dynamics_ratio_loss.backward()
        self.dynamics_ratio_estimator_optimizer.step()

        return dynamics_ratio_loss.cpu().detach().numpy().item()