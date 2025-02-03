import pdb
from collections import OrderedDict
from copy import deepcopy
from distutils.command.config import config

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections import ConfigDict
from torch import ne, nn as nn

from model import Scalar, soft_target_update
from utils import prefix_metrics


class DarcSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 1.0
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = True
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.d_sa_lr = 3e-4
        config.d_sas_lr = 3e-4
        config.noise_std_discriminator = 0.1
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.batch_size = 256
        config.device = 'cuda'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, d_sa, d_sas, replay_buffer):
        self.config = DarcSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.mean, self.std = self.replay_buffer.get_mean_std()

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

    def train(self, batch):
        self._total_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        new_actions, log_pi = self.policy(observations)

        dsa_loss, dsas_loss = self.train_discriminator()
        # pdb.set_trace()
        # DARC
        rewards = torch.squeeze(rewards, -1) + self.log_real_sim_dynacmis_ratio(observations, actions, next_observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        q_new_actions = torch.min(
            self.qf1(observations, new_actions),
            self.qf2(observations, new_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        new_next_actions, next_log_pi = self.policy(next_observations)
        target_q_values = torch.min(
            self.target_qf1(next_observations, new_next_actions),
            self.target_qf2(next_observations, new_next_actions),
        )

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        q_target = self.config.reward_scale * rewards + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

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

        return dict(
            dsa_train_loss=dsa_loss,
            dsas_train_loss=dsas_loss,
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

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        # if self.config.cql_lagrange:
        #     modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    def train_discriminator(self):
        # pdb.set_trace()
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

    def kl_divergence(self, observations, actions, next_observations):
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
            n_next_observations = self.next_observation_sampler.get_next_state(observations, actions, self.config.sampling_n_next_states)
            sum_log_ratio = self.log_sim_real_dynacmis_ratio(observations, actions, n_next_observations[0])
            for i in range(1, self.config.sampling_n_next_states):
                sum_log_ratio += self.log_sim_real_dynacmis_ratio(observations, actions, n_next_observations[i])

        # TODO
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
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            log_ratio = torch.log(sas_prob[:, 0]) \
                    - torch.log(sas_prob[:, 1]) \
                    - torch.log(sa_prob[:, 0]) \
                    + torch.log(sa_prob[:, 1])
            # log_ratio = torch.clamp(torch.log(sas_prob[:, 0]) - torch.log(sas_prob[:, 1]) - torch.log(sa_prob[:, 0]) + torch.log(sa_prob[:, 1]), self.config.clip_dynamics_ratio_min, self.config.clip_dynamics_ratio_max)
        
        return torch.clamp(log_ratio, -10, 10)

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