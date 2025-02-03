import imp
import pdb
from collections import OrderedDict
from copy import deepcopy
from distutils.command.config import config
# from turtle import pd
# from certifi import where

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections import ConfigDict
from torch import ne, nn as nn

from model import Scalar, soft_target_update
from utils import prefix_metrics


class Sim2realSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.batch_size = 256
        config.device = 'cuda'
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.d_sa_lr = 3e-4
        config.d_sas_lr = 3e-4
        config.d_early_stop_steps = 1000000
        config.noise_std_discriminator = 0.1
        config.start_steps = 0
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = True
        config.use_variant = False
        config.u_ablation = False
        config.use_td_target_ratio = True
        config.use_sim_q_coeff = True
        config.use_kl_baseline = False
        config.fix_baseline_steps = 10
        # kl divergence: E_pM log(pM/pM^)
        config.sim_q_coeff_min = 1e-45
        config.sim_q_coeff_max = 10
        config.sampling_n_next_states = 10
        config.s_prime_std_ratio = 1.
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 0.01
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -1000
        config.cql_clip_diff_max = 1000
        # pM/pM^
        config.clip_dynamics_ratio_min = 1e-5
        config.clip_dynamics_ratio_max = 1
        config.sa_prob_clip = 0.0
        # gumbel loss
        config.use_gumbel_regression = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, d_sa, d_sas, replay_buffer, dynamics_model=None):
        self.config = Sim2realSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.mean, self.std = self.replay_buffer.get_mean_std()
        self.next_observation_sampler = dynamics_model
        self.kl_baseline = 1

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

        # whether to use the lagrange version of CQL: False by default
        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, real_batch_size, sim_batch_size, bc=False):
        self._total_steps += 1

        if self._total_steps < self.config.start_steps:
            batch_size = real_batch_size + sim_batch_size
            batch = self.replay_buffer.sample(batch_size, scope="real")
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            new_actions, log_pi = self.policy(observations)

            if self._total_steps > batch_size:
                dsa_loss, dsas_loss = self.train_discriminator()


            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
                alpha = self.log_alpha().exp() * self.config.alpha_multiplier
            else:
                alpha_loss = observations.new_tensor(0.0)
                alpha = observations.new_tensor(self.config.alpha_multiplier)

            """ Policy loss """
            if bc:
                log_probs = self.policy.log_prob(observations, actions)
                policy_loss = (alpha * log_pi - log_probs).mean()
            else:
                q_new_actions = torch.min(
                    self.qf1(observations, new_actions),
                    self.qf2(observations, new_actions),
                )
                policy_loss = (alpha * log_pi - q_new_actions).mean()

            """ Q function loss """
            q1_pred = self.qf1(observations, actions)
            q2_pred = self.qf2(observations, actions)

            if self.config.cql_max_target_backup:
                new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
                # TODO
                target_q_values, max_target_indices = torch.max(
                    torch.min(
                        self.target_qf1(next_observations, new_next_actions),
                        self.target_qf2(next_observations, new_next_actions),
                    ),
                    dim=-1
                )
                # target_q_values, max_target_indices = torch.max(
                #         self.target_qf1(next_observations, new_next_actions), dim=-1
                # )
                next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
            else:
                new_next_actions, next_log_pi = self.policy(next_observations)
                # TODO
                target_q_values = torch.min(
                    self.target_qf1(next_observations, new_next_actions),
                    self.target_qf2(next_observations, new_next_actions),
                )
                # target_q_values = self.target_qf1(next_observations, new_next_actions)

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = torch.squeeze(rewards, -1) + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values
            qf1_loss = F.mse_loss(q1_pred, td_target.detach())
            # pdb.set_trace()
            qf2_loss = F.mse_loss(q2_pred, td_target.detach())


            ### CQL
            if not self.config.use_cql:
                qf_loss = qf1_loss + qf2_loss
            else:
                batch_size = actions.shape[0]
                action_dim = actions.shape[-1]
                
                cql_random_actions = actions.new_empty((batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
                
                cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
                cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
                
                cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
                cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

                cql_q1_rand = self.qf1(observations, cql_random_actions)
                cql_q2_rand = self.qf2(observations, cql_random_actions)
                cql_q1_current_actions = self.qf1(observations, cql_current_actions)
                cql_q2_current_actions = self.qf2(observations, cql_current_actions)
                cql_q1_next_actions = self.qf1(observations, cql_next_actions)
                cql_q2_next_actions = self.qf2(observations, cql_next_actions)

                cql_cat_q1 = torch.cat(
                    [cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
                )
                cql_std_q1 = torch.std(cql_cat_q1, dim=1)
                cql_std_q2 = torch.std(cql_cat_q2, dim=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5 ** action_dim)
                    cql_cat_q1 = torch.cat(
                        [cql_q1_rand - random_density,
                        cql_q1_next_actions - cql_next_log_pis.detach(),
                        cql_q1_current_actions - cql_current_log_pis.detach()],
                        dim=1
                    )
                    cql_cat_q2 = torch.cat(
                        [cql_q2_rand - random_density,
                        cql_q2_next_actions - cql_next_log_pis.detach(),
                        cql_q2_current_actions - cql_current_log_pis.detach()],
                        dim=1
                    )

                # Q values on the actions sampled from the policy
                cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
                cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp

                """Q values on the actions taken by the policy - Q values on data"""
                cql_qf1_diff = torch.clamp(
                    cql_qf1_ood - q1_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = torch.clamp(
                    cql_qf2_ood - q2_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                    cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                    self.alpha_prime_optimizer.zero_grad()
                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                    alpha_prime_loss.backward(retain_graph=True)
                    self.alpha_prime_optimizer.step()
                else:
                    cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                    alpha_prime_loss = observations.new_tensor(0.0)
                    alpha_prime = observations.new_tensor(0.0)


                qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.total_steps % self.config.target_update_period == 0:
                self.update_target_network(
                    self.config.soft_target_update_rate
                )


            metrics = dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_pred.mean().item(),
                average_qf2=q2_pred.mean().item(),
                average_target_q=target_q_values.mean().item(),
                total_steps=self.total_steps,
            )

            if self.config.use_cql:
                metrics.update(prefix_metrics(dict(
                    cql_std_q1=cql_std_q1.mean().item(),
                    cql_std_q2=cql_std_q2.mean().item(),
                    cql_q1_rand=cql_q1_rand.mean().item(),
                    cql_q2_rand=cql_q2_rand.mean().item(),
                    cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                    cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                    cql_qf1_diff=cql_qf1_diff.mean().item(),
                    cql_qf2_diff=cql_qf2_diff.mean().item(),
                    cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                    cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                    cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                    cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                    alpha_prime_loss=alpha_prime_loss.item(),
                    alpha_prime=alpha_prime.item(),
                ), 'cql'))

            return metrics




















        #TODO Sim2real CQL
        else:
            real_batch = self.replay_buffer.sample(real_batch_size, scope="real")
            sim_batch = self.replay_buffer.sample(sim_batch_size, scope="sim")

            # real transitions from d^{\pi_\beta}_\mathcal{M}
            real_observations = real_batch['observations']
            real_actions = real_batch['actions']
            real_rewards = real_batch['rewards']
            real_next_observations = real_batch['next_observations']
            real_dones = real_batch['dones'] 

            # sim transitions from d^\pi_\mathcal{\widehat{M}}
            sim_observations = sim_batch['observations']
            sim_actions = sim_batch['actions']
            sim_rewards = sim_batch['rewards']
            sim_next_observations = sim_batch['next_observations']
            sim_dones = sim_batch['dones']
            
            # mixed transitions from d_f = f * d^{\pi_\beta}_\mathcal{M} + (1-f) * d^\pi_\mathcal{\widehat{M}}
            df_observations = torch.cat([real_observations, sim_observations], dim=0)
            df_actions =  torch.cat([real_actions, sim_actions], dim=0)
            df_rewards =  torch.cat([real_rewards, sim_rewards], dim=0)
            # df_next_observations =  torch.cat([real_next_observations, sim_next_observations], dim=0)
            # df_dones =  torch.cat([real_dones, sim_dones], dim=0)

            dsa_loss, dsas_loss = self.train_discriminator()

            # TODO new_action and log pi
            df_new_actions, df_log_pi = self.policy(df_observations)

            # True by default
            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha() * (df_log_pi + self.config.target_entropy).detach()).mean()
                alpha = self.log_alpha().exp() * self.config.alpha_multiplier
            else:
                alpha_loss = df_observations.new_tensor(0.0)
                alpha = df_observations.new_tensor(self.config.alpha_multiplier)

            """ Policy loss """
            # Improve policy under state marginal distribution d_f
            if bc:
                log_probs = self.policy.log_prob(df_observations, df_actions)
                policy_loss = (alpha * df_log_pi - log_probs).mean()
            else:
                # TODO
                q_new_actions = torch.min(
                    self.qf1(df_observations, df_new_actions),
                    self.qf2(df_observations, df_new_actions),
                )
                # q_new_actions = self.qf1(df_observations, df_new_actions)
                policy_loss = (alpha * df_log_pi - q_new_actions).mean() 

            """ Q function loss """
            # Q function in real data and sim data
            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)
            sim_q1_pred = self.qf1(sim_observations, sim_actions)
            sim_q2_pred = self.qf2(sim_observations, sim_actions)

            # False by default (enabling self.config.cql_max_target_backup)
            # TODO check if this is correct
            real_new_next_actions, real_next_log_pi = self.policy(real_next_observations)
            # TODO
            real_target_q_values = torch.min(
                self.target_qf1(real_next_observations, real_new_next_actions),
                self.target_qf2(real_next_observations, real_new_next_actions),
            )
            # real_target_q_values = self.target_qf1(real_next_observations, real_new_next_actions)
            sim_new_next_actions, sim_next_log_pi = self.policy(sim_next_observations)
            # TODO
            sim_target_q_values = torch.min(
                self.target_qf1(sim_next_observations, sim_new_next_actions),
                self.target_qf2(sim_next_observations, sim_new_next_actions),
            )
            # sim_target_q_values = self.target_qf1(sim_next_observations, sim_new_next_actions)

            # False by default
            if self.config.backup_entropy:
                real_target_q_values = real_target_q_values - alpha * real_next_log_pi
                sim_target_q_values = sim_target_q_values - alpha * sim_next_log_pi
            # pdb.set_trace()
            real_td_target = torch.squeeze(real_rewards, -1) + (1. - torch.squeeze(real_dones, -1)) * self.config.discount * real_target_q_values
            sim_td_target = torch.squeeze(sim_rewards, -1) + (1. - torch.squeeze(sim_dones, -1)) * self.config.discount * sim_target_q_values

            real_qf1_loss = F.mse_loss(real_q1_pred, real_td_target.detach())
            real_qf2_loss = F.mse_loss(real_q2_pred, real_td_target.detach())

            # importance sampling on td error due to the dyanmics shift
            # TODO more elegant?
            if self.config.use_td_target_ratio:
                sqrt_IS_ratio = torch.clamp(self.real_sim_dynacmis_ratio(sim_observations, sim_actions, sim_next_observations), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max).sqrt()
            else:
                sqrt_IS_ratio = torch.ones((sim_observations.shape[0],)).to(self.config.device)
            # pdb.set_trace()
            sim_qf1_loss = F.mse_loss(sqrt_IS_ratio.squeeze() * sim_q1_pred, sqrt_IS_ratio.squeeze() * sim_td_target.detach())
            sim_qf2_loss = F.mse_loss(sqrt_IS_ratio.squeeze() * sim_q2_pred, sqrt_IS_ratio.squeeze() * sim_td_target.detach())
            # sim_qf1_loss = F.mse_loss(sim_q1_pred, sim_td_target.detach())
            # sim_qf2_loss = F.mse_loss(sim_q2_pred, sim_td_target.detach())

            qf1_loss = real_qf1_loss + sim_qf1_loss
            qf2_loss = real_qf2_loss + sim_qf2_loss
            # qf1_loss = sim_qf1_loss
            # qf2_loss = sim_qf2_loss
            # pdb.set_trace()

            ### Conservative Penalty loss: sim data
            if not self.config.use_cql:
                qf_loss = qf1_loss + qf2_loss
            else:
                # shape [128]
                cql_q1 = self.qf1(sim_observations, sim_actions)
                cql_q2 = self.qf2(sim_observations, sim_actions)
                # TODO Q + log(u)
                if self.config.use_sim_q_coeff and self.next_observation_sampler == None:
                    u_sa = self.kl_sim_divergence(sim_observations, sim_actions, sim_next_observations)
                elif self.config.use_sim_q_coeff:
                    u_sa = self.kl_real_divergence(sim_observations, sim_actions, sim_next_observations)
                else:
                    u_sa = torch.ones(sim_rewards.shape[0], device=self.config.device)
                
                omega = u_sa / u_sa.sum()
                # if self.config.u_ablation and self._total_steps % 1000 == 0:
                #     x_velocity  = sim_observations[:,8].cpu().detach().numpy().reshape((-1,1))
                #     u_log = u_sa.cpu().detach().numpy().reshape((-1,1))
                #     omega_log = omega.cpu().detach().numpy().reshape((-1,1))
                #     Q_log = sim_q1_pred.cpu().detach().numpy().reshape((-1,1))
                #     loggings = np.concatenate((x_velocity, u_log, omega_log, Q_log), axis=1)
                #     df = pd.DataFrame(loggings,columns=["velocity", "u_sa", "omega_sa", "Q value"])
                #     df.to_csv("ablation_log/step_{}-v3.csv".format(int(self._total_steps/1000)))
                    
                    # zero_one_idx = torch.where((x_velocity[:,8]>0)&(x_velocity[:,8]<1))
                    # one_two_idx = torch.where((x_velocity[:,8]>1)&(x_velocity[:,8]<2))
                    # two_three_idx = torch.where((x_velocity[:,8]>2)&(x_velocity[:,8]<3))
                    # three_four_idx = torch.where((x_velocity[:,8]>3)&(x_velocity[:,8]<4))

                # pdb.set_trace()
                if not self.config.use_variant:
                    cql_q1 += torch.log(omega)
                    cql_q2 += torch.log(omega)
                # pdb.set_trace()
                std_omega  = omega.std()
                
                # # shape [128]
                # cql_q1 = self.qf1(sim_observations, sim_actions)
                # cql_q2 = self.qf2(sim_observations, sim_actions)
                # # TODO Q + log(u)
                # if self.config.use_sim_q_coeff:
                #     # pdb.set_trace()
                #     u_sa = self.kl_divergence(sim_observations, sim_actions, sim_next_observations)
                #     omega = u_sa /  u_sa.sum()
                #     if self.config.use_kl_baseline and self._total_steps % self.config.fix_baseline_steps == 0:
                #         self.kl_baseline = self.kl_divergence(real_observations, real_actions, real_next_observations)
                #     # pdb.set_trace()
                #         cql_q1 += torch.log(torch.clip(omega - self.kl_baseline, self.config.sim_q_coeff_min, self.config.sim_q_coeff_max))
                #         cql_q2 += torch.log(torch.clip(omega - self.kl_baseline, self.config.sim_q_coeff_min, self.config.sim_q_coeff_max))
                #     else:
                #         # TODO
                #         cql_q1 += torch.log(omega)
                #         cql_q2 += torch.log(omega)
                # else:
                #     u_sa = torch.ones(sim_rewards.shape)
                #     omega = u_sa /  u_sa.sum()
                # # pdb.set_trace()
                # std_omega  = omega.std()

                # sim_batch_size = sim_actions.shape[0]
                # action_dim = sim_actions.shape[-1]
                
                # cql_random_actions = sim_actions.new_empty((sim_batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
                # cql_current_actions, cql_current_log_pis = self.policy(sim_observations, repeat=self.config.cql_n_actions)
                # cql_next_actions, cql_next_log_pis = self.policy(sim_next_observations, repeat=self.config.cql_n_actions)
                
                # cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
                # cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

                # # pdb.set_trace()
                # cql_q1_rand = self.qf1(sim_observations, cql_random_actions)
                # cql_q2_rand = self.qf2(sim_observations, cql_random_actions)
                # cql_q1_current_actions = self.qf1(sim_observations, cql_current_actions)
                # cql_q2_current_actions = self.qf2(sim_observations, cql_current_actions)
                # cql_q1_next_actions = self.qf1(sim_observations, cql_next_actions)
                # cql_q2_next_actions = self.qf2(sim_observations, cql_next_actions)

                # # Q + log(u)
                # if self.config.use_sim_q_coeff and self._total_steps > self.config.start_steps:
                #     # pdb.set_trace()
                #     cql_q1_rand += torch.log(self.kl_divergence(sim_observations, cql_random_actions)).reshape(cql_q1_rand.shape)
                #     cql_q2_rand += torch.log(self.kl_divergence(sim_observations, cql_random_actions)).reshape(cql_q1_rand.shape)
                #     cql_q1_current_actions += torch.log(self.kl_divergence(sim_observations, cql_current_actions)).reshape(cql_q1_rand.shape)
                #     cql_q2_current_actions += torch.log(self.kl_divergence(sim_observations, cql_current_actions)).reshape(cql_q1_rand.shape)
                #     cql_q1_next_actions += torch.log(self.kl_divergence(sim_observations, cql_next_actions)).reshape(cql_q1_rand.shape)
                #     cql_q2_next_actions += torch.log(self.kl_divergence(sim_observations, cql_next_actions)).reshape(cql_q1_rand.shape)

                # # cql_cat_q1 = torch.cat(
                # #     [cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
                # # )
                # # cql_cat_q2 = torch.cat(
                # #     [cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
                # # )

                # if self.config.cql_importance_sample:
                #     random_density = np.log(0.5 ** action_dim)
                #     cql_cat_q1 = torch.cat(
                #         [cql_q1_rand - random_density,
                #         cql_q1_next_actions - cql_next_log_pis.detach(),
                #         cql_q1_current_actions - cql_current_log_pis.detach()],
                #         dim=1
                #     )                               
                #     # 128 * 30
                #     cql_cat_q2 = torch.cat(
                #         [cql_q2_rand - random_density,
                #         cql_q2_next_actions - cql_next_log_pis.detach(),
                #         cql_q2_current_actions - cql_current_log_pis.detach()],
                #         dim=1
                #     )
                #     # 128 * 30
                # cql_std_q1 = torch.std(cql_q1, dim=1)
                # cql_std_q2 = torch.std(cql_q2, dim=1)

                # Q values on the actions sampled from the policy
                if self.config.use_variant:
                    cql_qf1_gap = (omega * cql_q1).sum()
                    cql_qf2_gap = (omega * cql_q2).sum()
                    # pdb.set_trace()
                else:
                    cql_qf1_gap = torch.logsumexp(cql_q1 / self.config.cql_temp, dim=0) * self.config.cql_temp
                    cql_qf2_gap = torch.logsumexp(cql_q2 / self.config.cql_temp, dim=0) * self.config.cql_temp


                """Q values on the stat-actions with larger dynamics gap - Q values on data"""
                cql_qf1_diff = torch.clamp(
                    cql_qf1_gap - real_q1_pred.mean(),
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                )
                cql_qf2_diff = torch.clamp(
                    cql_qf2_gap - real_q2_pred.mean(),
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                )

                # False by default
                if self.config.cql_lagrange:
                    alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                    cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                    self.alpha_prime_optimizer.zero_grad()
                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                    alpha_prime_loss.backward(retain_graph=True)
                    
                    self.alpha_prime_optimizer.step()
                else:
                    cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                    alpha_prime_loss = df_observations.new_tensor(0.0)
                    alpha_prime = df_observations.new_tensor(0.0)

                qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.total_steps % self.config.target_update_period == 0:
                self.update_target_network(
                    self.config.soft_target_update_rate
                )

            metrics = dict(
                dsa_train_loss=dsa_loss,
                dsas_train_loss=dsas_loss,
                mean_real_rewards=real_rewards.mean(),
                mean_sim_rewards=sim_rewards.mean(),
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_sim_qf1=sim_q1_pred.mean().item(),
                average_sim_qf2=sim_q2_pred.mean().item(),
                average_real_target_q=real_target_q_values.mean().item(),
                average_sim_target_q=sim_target_q_values.mean().item(),
                total_steps=self.total_steps,
            )
            # if self.config.u_ablation:
            #     metrics.update(dict(
            #         u_sa_high=u_sa_high.mean().item(),
            #         u_sa_low=u_sa_low.mean().item(),
            #         sim_q_high_penalty=sim_q_high.mean().item(),
            #         sim_q_low_penalty=sim_q_low.mean().item()
            #     ))
            
            # if self._total_steps < self.config.d_early_stop_steps:
            #     metrics.update(dict(
            #         dsa_train_loss=dsa_loss,
            #         dsas_train_loss=dsas_loss
            #     ))

            # pdb.set_trace()
            if self.config.use_cql:
                metrics.update(prefix_metrics(dict(
                    u_sa=u_sa.mean().item(),
                    std_omega=std_omega.mean().item(),
                    sqrt_IS_ratio=sqrt_IS_ratio.mean().item(),
                    cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                    cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                    cql_qf1_diff=cql_qf1_diff.mean().item(),
                    cql_qf2_diff=cql_qf2_diff.mean().item(),
                    cql_qf1_gap=cql_qf1_gap.mean().item(),
                    cql_qf2_gap=cql_qf2_gap.mean().item(),
                ), 'cql'))

            return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    def gumbel_loss(pred, label, eta=1.0, clip=1000):
        z = (label - pred) / eta
        z = torch.clamp(z, -clip, clip)
        max_z = torch.max(z)
        max_z = torch.where(max_z < 1.0, torch.tensor(-1.0), max_z)
        max_z = max_z.detach()
        loss = torch.exp(-max_z) * (torch.exp(z) - z - 1)
        return loss.mean()

    def train_discriminator(self):
        real_obs, real_action, real_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="real", type="sas").values()
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(self.config.batch_size, scope="sim", type="sas").values()
        # assert torch.isnan(real_action).sum() == 0, print(real_action)
        # assert torch.isnan(sim_action).sum() == 0, print(sim_action)

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
        # assert torch.isnan(real_sa_prob).sum() == 0, print(real_sa_prob)
        sim_sa_logits = self.d_sa(sim_obs, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        # assert torch.isnan(sim_sa_prob).sum() == 0, print(sim_sa_prob)
        
        # pdb.set_trace()
        real_adv_logits = self.d_sas(real_obs, real_action, real_next_obs)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(sim_obs, sim_action, sim_next_obs)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)

        # assert torch.isnan(real_sas_prob).sum() == 0, print(real_sas_prob)
        # assert torch.isnan(sim_sas_prob).sum() == 0, print(sim_sas_prob)

        dsa_loss = (- torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()
        # assert torch.isnan(dsa_loss).sum() == 0, print(dsa_loss)
        dsas_loss = (- torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()
        # assert torch.isnan(dsas_loss).sum() == 0, print(dsas_loss)

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
    
    # def gaussian_noise(inputs, stddev=1, device='cuda'):
    #     input = inputs.cpu()
    #     # input_array = input.data.numpy()
    #     out = torch.randn(input.shape) * stddev

    #     # noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(input_array))

    #     # out = np.add(input_array, noise)

    #     # output_tensor = torch.from_numpy(out)
    #     # out_tensor = torch.tensor(output_tensor)
    #     # out = out_tensor.to(device)
    #     # out = out.float()
    #     return out

    # return u(s,a): the KL divergence of sim and real dynamics

    def kl_sim_divergence(self, observations, actions, next_observations):
        # # expand observations and actions into N times length
        # # pdb.set_trace()
        # observations = torch.repeat_interleave(observations, actions.shape[1], dim=0)
        # actions = torch.reshape(actions, (-1, actions.shape[-1]))

        # expectation on next observation over learned dynamics from the real offline data
        if self.next_observation_sampler == None:
            # TODO
            observations = torch.repeat_interleave(observations, self.config.sampling_n_next_states, dim=0)
            actions = torch.repeat_interleave(actions, self.config.sampling_n_next_states, dim=0)
            next_observations = torch.repeat_interleave(next_observations, self.config.sampling_n_next_states, dim=0)
            next_observations += torch.randn(next_observations.shape, device=self.config.device) * self.std * self.config.s_prime_std_ratio
            log_ratio = self.log_sim_real_dynacmis_ratio(observations, actions, next_observations).reshape((-1, self.config.sampling_n_next_states))

            # sum_log_ratio = 0
            # for i in range(self.config.sampling_n_next_states):
            #     next_observations += torch.randn(next_observations.shape, device=self.config.device) * std
            #     # pdb.set_trace()
            #     sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, next_observations)
        else:
            # TODO
            observations = torch.repeat_interleave(observations, self.config.sampling_n_next_states, dim=0)
            actions = torch.repeat_interleave(actions, self.config.sampling_n_next_states, dim=0)
            # next_observations = torch.repeat_interleave(next_observations, self.config.sampling_n_next_states, dim=0)
            # next_observations += torch.randn(next_observations.shape, device=self.config.device) * self.std * self.config.s_prime_std_ratio
            next_observations = self.next_observation_sampler.get_next_state(observations, actions, self.config.sampling_n_next_states)
            log_ratio = self.log_sim_real_dynacmis_ratio(observations, actions, next_observations).reshape((-1, self.config.sampling_n_next_states))
            # for i in range(1, self.config.sampling_n_next_states):
            #     sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, next_observations[i])

        # TODO
        # pdb.set_trace()
        return torch.clamp(log_ratio.mean(dim=1), self.config.sim_q_coeff_min, self.config.sim_q_coeff_max)
        # return torch.clamp(sum_log_ratio.squeeze(1) / self.config.sampling_n_next_states, self.config.sim_q_coeff_min, self.config.sim_q_coeff_max)

    def kl_real_divergence(self, observations, actions, next_observations):
        # # expand observations and actions into N times length
        # # pdb.set_trace()
        # observations = torch.repeat_interleave(observations, actions.shape[1], dim=0)
        # actions = torch.reshape(actions, (-1, actions.shape[-1]))

        # expectation on next observation over learned dynamics from the real offline data
        if self.next_observation_sampler == None:
            # TODO
            observations = torch.repeat_interleave(observations, self.config.sampling_n_next_states, dim=0)
            actions = torch.repeat_interleave(actions, self.config.sampling_n_next_states, dim=0)
            next_observations = torch.repeat_interleave(next_observations, self.config.sampling_n_next_states, dim=0)
            next_observations += torch.randn(next_observations.shape, device=self.config.device) * self.std * self.config.s_prime_std_ratio
            log_ratio = self.log_real_sim_dynacmis_ratio(observations, actions, next_observations).reshape((-1, self.config.sampling_n_next_states))

            # sum_log_ratio = 0
            # for i in range(self.config.sampling_n_next_states):
            #     next_observations += torch.randn(next_observations.shape, device=self.config.device) * std
            #     # pdb.set_trace()
            #     sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, next_observations)
        else:
            # TODO
            # pdb.set_trace()
            observations = torch.repeat_interleave(observations, self.config.sampling_n_next_states, dim=0)
            actions = torch.repeat_interleave(actions, self.config.sampling_n_next_states, dim=0)
            # next_observations = torch.repeat_interleave(next_observations, self.config.sampling_n_next_states, dim=0)
            # next_observations += torch.randn(next_observations.shape, device=self.config.device) * self.std * self.config.s_prime_std_ratio
            next_observations = self.next_observation_sampler.get_next_state(observations, actions, 1).reshape((-1, observations.shape[1]))
            # pdb.set_trace()
            log_ratio = self.log_real_sim_dynacmis_ratio(observations, actions, next_observations).reshape((-1, self.config.sampling_n_next_states))
            # for i in range(1, self.config.sampling_n_next_states):
            #     sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, next_observations[i])

        # TODO
        # pdb.set_trace()
        return torch.clamp(log_ratio.mean(dim=1), self.config.sim_q_coeff_min, self.config.sim_q_coeff_max)
        # return torch.clamp(sum_log_ratio.squeeze(1) / self.config.sampling_n_next_states, self.config.sim_q_coeff_min, self.config.sim_q_coeff_max)


    def log_sim_real_dynacmis_ratio(self, observations, actions, next_observations):
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

    def log_real_sim_dynacmis_ratio(self, observations, actions, next_observations):
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

    def real_sim_dynacmis_ratio(self, observations, actions, next_observations):
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            ratio = (sas_prob[:, 0] * sa_prob[:, 1]) / (sas_prob[:, 1] * sa_prob[:, 0])
            # log_ratio = torch.clamp(torch.log(sas_prob[:, 0]) - torch.log(sas_prob[:, 1]) - torch.log(sa_prob[:, 0]) + torch.log(sa_prob[:, 1]), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max)
        
        return ratio